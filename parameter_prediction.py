import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import subprocess
import sys
import glob

# =============================================================================
# CUDA MULTIPROCESSING FIX: Set spawn method before any CUDA operations
# This prevents "Cannot re-initialize CUDA in forked subprocess" errors
# =============================================================================
print("🔧 CUDA MULTIPROCESSING DIAGNOSTIC:")
print(f"   Current multiprocessing start method: {mp.get_start_method()}")
print(f"   Available start methods: {mp.get_all_start_methods()}")

if mp.get_start_method() != 'spawn':
    print("   ⚠️  Current method is not 'spawn' - this can cause CUDA issues in DataLoader workers")
    print("   🔧 Setting multiprocessing start method to 'spawn' for CUDA compatibility")
    try:
        mp.set_start_method('spawn', force=True)
        print(f"   ✓ Successfully set start method to: {mp.get_start_method()}")
    except RuntimeError as e:
        print(f"   ⚠️  Could not set start method (already set): {e}")
        print("   💡 Consider setting this at the very beginning of your script")
else:
    print("   ✓ Start method is already 'spawn' - good for CUDA compatibility")
print()

import wandb
from datasets import *
from models import *
from simulator import (Gaussian2DSimulator, MCEGSimulator, RealisticDIS, SimplifiedDIS)
from utils import *

# Import precomputed dataset classes
try:
    from precomputed_datasets import PrecomputedDataset, DistributedPrecomputedDataset, create_precomputed_dataloader, filter_valid_precomputed_files
    PRECOMPUTED_AVAILABLE = True
except ImportError:
    print("Warning: precomputed_datasets.py not found. Precomputed data support disabled.")
    PRECOMPUTED_AVAILABLE = False
    
    # Define a fallback version of the utility function
    def filter_valid_precomputed_files(file_list):
        """Fallback implementation of filter_valid_precomputed_files when precomputed_datasets is not available."""
        valid_files = []
        for file_path in file_list:
            filename = os.path.basename(file_path)
            if filename.endswith('.npz') and '.tmp' not in filename:
                valid_files.append(file_path)
        return valid_files

def generate_precomputed_data_if_needed(problem, num_samples, num_events, n_repeat=2, output_dir="precomputed_data"):
    """
    Check if precomputed data exists for the given parameters, and raise an error if not found.
    
    This function enforces exact matching: only files with the exact parameters 
    (problem, num_samples, num_events, n_repeat) will be accepted. No fallback 
    to files with different parameters is performed.
    
    Args:
        problem: Problem type ('gaussian', 'simplified_dis', 'realistic_dis', 'mceg')
        num_samples: Number of theta parameter samples
        num_events: Number of events per simulation
        n_repeat: Number of repeated simulations per theta
        output_dir: Directory where precomputed data should be stored
    
    Returns:
        str: Path to the data directory
        
    Raises:
        FileNotFoundError: If the exact matching file does not exist
        RuntimeError: If precomputed data support is not available
    """
    print("🔍 PRECOMPUTED DATA DIAGNOSTIC:")
    print(f"   Looking for problem: '{problem}' with ns={num_samples}, ne={num_events}, nr={n_repeat}")
    print(f"   Data directory: '{output_dir}'")
    
    if not PRECOMPUTED_AVAILABLE:
        print("   ✗ Precomputed data support not available. Please check precomputed_datasets.py")
        raise RuntimeError("Precomputed data support not available. Please check precomputed_datasets.py")
    
    # Check if data already exists
    expected_filename = f"{problem}_ns{num_samples}_ne{num_events}_nr{n_repeat}.npz"
    exact_file_path = os.path.join(output_dir, expected_filename)
    print(f"   Required exact file: '{exact_file_path}'")
    
    if os.path.exists(exact_file_path):
        print(f"   ✓ Found exact matching precomputed data: {exact_file_path}")
        return output_dir
    else:
        # print(f"   ✗ Exact matching file not found: {expected_filename}")
        # print(f"   📁 Searched in directory: {output_dir}")
        
        # # Show what files ARE available to help user debug
        # available_pattern = os.path.join(output_dir, f"{problem}_*.npz")
        # available_files = glob.glob(available_pattern)
        # if available_files:
        #     print(f"   📋 Available files for problem '{problem}':")
        #     for file_path in available_files:
        #         filename = os.path.basename(file_path)
        #         print(f"      - {filename}")
        #     print(f"   💡 None of these match the exact parameters: ns={num_samples}, ne={num_events}, nr={n_repeat}")
        # else:
        #     print(f"   📋 No files found for problem '{problem}' in {output_dir}")
        
        # # Raise informative error
        # error_msg = (
        #     f"Exact precomputed data file not found: '{expected_filename}'\n"
        #     f"  Required parameters: problem='{problem}', num_samples={num_samples}, "
        #     f"num_events={num_events}, n_repeat={n_repeat}\n"
        #     f"  Searched in: {output_dir}\n"
        #     f"  Expected file: {exact_file_path}\n"
        # )
        
        # if available_files:
        #     error_msg += f"  Available files: {[os.path.basename(f) for f in available_files]}\n"
        #     error_msg += "  Hint: None of the available files match the exact required parameters."
        # else:
        #     error_msg += f"  No precomputed data files found for problem '{problem}'."
            
        # raise FileNotFoundError(error_msg)
        #     # Generate new data
        print(f"Precomputed data not found for {problem} with ns={num_samples}, ne={num_events}, nr={n_repeat}")
        print("Generating precomputed data automatically...")
        
        try:
            # Import and run the data generation function
            from generate_precomputed_data import generate_data_for_problem
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Generating data on device: {device}")
            
            filepath = generate_data_for_problem(
                problem, num_samples, num_events, n_repeat, device, output_dir
            )
            print(f"Successfully generated precomputed data: {filepath}")
            return output_dir
            
        except Exception as e:
            print(f"Error generating precomputed data: {e}")
            print("Falling back to on-the-fly data generation")
            raise RuntimeError(f"Failed to generate precomputed data: {e}")

def check_precomputed_data_exists(problem, output_dir="precomputed_data"):
    """
    Check if any valid precomputed data exists for the given problem.
    
    Args:
        problem: Problem type
        output_dir: Directory to check for data
    
    Returns:
        bool: True if valid data exists, False otherwise
    """
    pattern = os.path.join(output_dir, f"{problem}_*.npz")
    all_files = glob.glob(pattern)
    valid_files = filter_valid_precomputed_files(all_files)
    return len(valid_files) > 0

# ---------------------------
# Joint Training Function
# ---------------------------
def train_joint(model, param_prediction_model, train_dataloader, val_dataloader,
                epochs, lr, rank, wandb_enabled, output_dir, save_every=10, args=None):
    device = next(model.parameters()).device
    param_prediction_model = param_prediction_model.to(device)
    opt = optim.Adam(list(model.parameters()) + list(param_prediction_model.parameters()),
                     lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    param_prediction_model.train()
    torch.backends.cudnn.benchmark = True
    if args.problem == 'mceg':
        theta_bounds = torch.tensor([
                [-1.0, 10.0],
                [0.0, 10.0],
                [-10.0, 10.0],
                [-10.0, 10.0],
            ]).to(device)   
    elif args.problem == 'simplified_dis':
        theta_bounds = torch.tensor([
            [0.0, 5.0],
            [0.0, 5.0],
            [0.0, 5.0],
            [0.0, 5.0],
        ]).to(device)
    
    # Print parameter bounds for transparency
    if rank == 0:  # Only print once in distributed training
        print(f"\n📊 PARAMETER TRAINING BOUNDS:")
        print(f"   Problem type: {args.problem}")
        print(f"   Bounds (min, max): {theta_bounds.cpu().tolist()}")
        print(f"   Ranges: {(theta_bounds[:, 1] - theta_bounds[:, 0]).cpu().tolist()}")
        print()
    
    theta_min, theta_max = theta_bounds[:, 0], theta_bounds[:, 1]
    theta_range = theta_max - theta_min
    def normalize_theta(theta):
        """
        Map theta from [min, max] to [-1, 1].
        """
        return 2 * (theta - theta_min) / theta_range - 1

    def denormalize_theta(norm_theta):
        """
        Map normalized value in [-1,1] back to original scale.
        """
        return (norm_theta + 1) * 0.5 * theta_range + theta_min
        
    # one-time precompute (before for epoch in range(...))
    theta_range = (theta_max - theta_min).to(device)       # shape (D,)
    range_half_sq = ((theta_range / 2.0) ** 2).view(1, -1)  # (1, D) in device
    mean_weight = range_half_sq.mean().item()
    range_half_sq = range_half_sq.to(device)

    for epoch in range(epochs):
        total_train_loss = 0.0
        num_train_batches = 0

        for theta, x_sets in train_dataloader:
            # ---- prepare batch ----
            x_sets = x_sets.to(torch.float32)                       # cpu->float32
            B, n_repeat, num_points, feat_dim = x_sets.shape
            x_sets = x_sets.reshape(B * n_repeat, num_points, feat_dim)
            x_sets = x_sets.to(device)
            theta = theta.to(device)
            theta = theta.repeat_interleave(n_repeat, dim=0)
            
            # Optional diagnostic: Check if training thetas are within bounds
            if hasattr(args, 'check_training_bounds') and args.check_training_bounds and epoch == 0 and num_train_batches == 0:
                # Only check on first batch of first epoch to avoid spam
                print(f"🔍 Training data bounds check (first batch):")
                theta_min_cpu = theta_min.cpu()
                theta_max_cpu = theta_max.cpu()
                theta_cpu = theta.cpu()
                
                for i in range(theta_bounds.shape[0]):
                    param_vals = theta_cpu[:, i]
                    min_val, max_val = param_vals.min().item(), param_vals.max().item()
                    bound_min, bound_max = theta_min_cpu[i].item(), theta_max_cpu[i].item()
                    
                    if min_val < bound_min or max_val > bound_max:
                        print(f"   ⚠️  Parameter {i}: range=[{min_val:.3f}, {max_val:.3f}], bounds=[{bound_min:.1f}, {bound_max:.1f}]")
                    else:
                        print(f"   ✅ Parameter {i}: range=[{min_val:.3f}, {max_val:.3f}], bounds=[{bound_min:.1f}, {bound_max:.1f}]")
                print()
            
            opt.zero_grad(set_to_none=True)

            # ---- forward/backward with AMP ----
            # with torch.cuda.amp.autocast():   # default dtype (fp16 on CUDA)
            # compute normalized target
            theta_norm = normalize_theta(theta)   # (B, D)

            emb = model(x_sets)           # (B*n_repeat, latent)

            # get raw model output (original units) and normalize it for loss
            raw_pred = param_prediction_model(emb)        # (B, D)  <-- raw in original units
            predicted_theta = normalize_theta(raw_pred)   # (B, D) normalized

            # weighted normalized MSE equivalent to unnormalized MSE
            diff = predicted_theta - theta_norm          # (B, D)
            weighted = (diff**2 * range_half_sq).mean()  # scalar

            # scale magnitude for optimizer stability (optional but recommended)
            loss = weighted / mean_weight

            # logging metric in original units (for humans)
            preds_unnorm = denormalize_theta(predicted_theta)  # or simply raw_pred if raw_pred is in original units
            unnormalized_loss = F.mse_loss(preds_unnorm, theta)

            # basic NaN/Inf check
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss detected: {loss}")

            # scale, backward, unscale, clip, step
            scaler.scale(loss).backward()

            # Unscale before clipping
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(param_prediction_model.parameters(), max_norm=1.0)

            scaler.step(opt)
            scaler.update()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / max(1, num_train_batches)

        # ---- Validation (FP32, no autocast) ----
        model.eval()
        param_prediction_model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for theta, x_sets in val_dataloader:
                x_sets = x_sets.to(torch.float32)
                B, n_repeat, num_points, feat_dim = x_sets.shape
                x_sets = x_sets.reshape(B * n_repeat, num_points, feat_dim)
                x_sets = x_sets.to(device)
                theta = theta.to(device)
                theta = theta.repeat_interleave(n_repeat, dim=0)

                # compute normalized target
                theta_norm = normalize_theta(theta)   # (B, D)

                # No autocast here — evaluate in FP32 for stability
                emb = model(x_sets)

                # get raw model output (original units) and normalize it for loss
                raw_pred = param_prediction_model(emb)        # (B, D)  <-- raw in original units
                predicted_theta = normalize_theta(raw_pred)   # (B, D) normalized

                # weighted normalized MSE equivalent to unnormalized MSE
                diff = predicted_theta - theta_norm          # (B, D)
                weighted = (diff**2 * range_half_sq).mean()  # scalar

                # scale magnitude for optimizer stability (optional but recommended)
                val_loss = weighted / mean_weight

                # logging metric in original units (for humans)
                preds_unnorm = denormalize_theta(predicted_theta)  # or simply raw_pred if raw_pred is in original units
                unnormalized_val_loss = F.mse_loss(preds_unnorm, theta)
                if not torch.isfinite(val_loss):
                    raise RuntimeError(f"Non-finite val loss detected: {val_loss}")

                total_val_loss += val_loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / max(1, num_val_batches)

        # Switch back to training mode
        model.train()
        param_prediction_model.train()

        # Logging
        print(f"Epoch {epoch:03d} train_loss={avg_train_loss:.6g} val_loss={avg_val_loss:.6g}, unnormalized_train_loss={unnormalized_loss.item():.6g}, unnormalized_val_loss={unnormalized_val_loss.item():.6g}")

        # Logging and checkpointing (only on rank 0)
        if rank == 0:
            if wandb_enabled:
                wandb.log({
                    "epoch": epoch + 1, 
                    "train_mse_loss": avg_train_loss,
                    "val_mse_loss": avg_val_loss,
                    "train_unnormalized_mse_loss": unnormalized_loss.item(),
                    "val_unnormalized_mse_loss": unnormalized_val_loss.item()
                })
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"))
                torch.save(param_prediction_model.state_dict(), os.path.join(output_dir, f"params_model_epoch_{epoch+1}.pth"))
    if rank == 0:
        torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
        torch.save(param_prediction_model.state_dict(), os.path.join(output_dir, "final_params_model.pth"))
        
        # Try to save Laplace model if available
        try:
            from laplace import Laplace
            lap_transformer = Laplace(param_prediction_model, 'regression', subset_of_weights='last_layer', hessian_structure='kron')
            lap_transformer.fit(train_dataloader)  # Use training dataloader for Laplace fitting
            
            # Try to import save_laplace or create a simple version
            try:
                from utils import save_laplace
            except ImportError:
                def save_laplace(lap_model, output_dir, filename, likelihood, subset_of_weights, hessian_structure):
                    """Simple save function for Laplace model"""
                    print(f"Saving Laplace model to {os.path.join(output_dir, filename)}")
                    # Basic save if detailed save not available
                    torch.save(lap_model.state_dict(), os.path.join(output_dir, filename))
            
            save_laplace(lap_transformer, output_dir,
                        filename="laplace_transformer.pt",
                        likelihood="regression",
                        subset_of_weights="last_layer",
                        hessian_structure="kron")
        except ImportError as e:
            print(f"Laplace not available, skipping Laplace model save: {e}")
        except Exception as e:
            print(f"Error saving Laplace model: {e}")

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)

def cleanup():
    dist.destroy_process_group()

def create_train_val_datasets(args, rank, world_size, device):
    """
    Create training and validation datasets with clear separation.
    Training dataset uses args.num_samples, validation uses args.val_samples (default 1000).
    """
    val_samples = getattr(args, 'val_samples', 1000)
    
    if hasattr(args, 'use_precomputed') and args.use_precomputed and PRECOMPUTED_AVAILABLE:
        print(f"Using precomputed data for {args.problem}")
        # Create training dataset
        try:
            train_data_dir = generate_precomputed_data_if_needed(
                args.problem, 
                args.num_samples, 
                args.num_events, 
                n_repeat=args.num_repeat,
                output_dir=args.precomputed_data_dir
            )
            
            # Create validation dataset (separate from training)
            val_data_dir = generate_precomputed_data_if_needed(
                args.problem,
                val_samples, 
                args.num_events, 
                n_repeat=1,
                output_dir=args.precomputed_data_dir
            )
            
            # Create distributed datasets with exact parameter matching
            if world_size > 1:
                train_dataset = DistributedPrecomputedDataset(
                    train_data_dir, args.problem, rank, world_size, shuffle=True,
                    exact_ns=args.num_samples, exact_ne=args.num_events, exact_nr=1
                )
                val_dataset = DistributedPrecomputedDataset(
                    val_data_dir, args.problem, rank, world_size, shuffle=False,
                    exact_ns=val_samples, exact_ne=args.num_events, exact_nr=1
                )
            else:
                train_dataset = PrecomputedDataset(train_data_dir, args.problem, shuffle=True,
                                                 exact_ns=args.num_samples, exact_ne=args.num_events, exact_nr=1)
                val_dataset = PrecomputedDataset(val_data_dir, args.problem, shuffle=False,
                                                exact_ns=val_samples, exact_ne=args.num_events, exact_nr=1)
            
            # Get input dimension from dataset metadata
            metadata = train_dataset.get_metadata()
            input_dim = metadata['feature_dim']
            
            print(f"Loaded precomputed datasets - Train: {metadata}, Val: {val_samples} samples")
            
        except Exception as e:
            print(f"Failed to use precomputed data: {e}")
            print("Falling back to on-the-fly data generation")
            args.use_precomputed = False
    
    # Fall back to original on-the-fly data generation
    if not (hasattr(args, 'use_precomputed') and args.use_precomputed and PRECOMPUTED_AVAILABLE):
        print(f"Using on-the-fly data generation for {args.problem}")
        
        if args.problem == "simplified_dis":
            simulator = SimplifiedDIS(device=device)
            train_dataset = DISDataset(simulator, args.num_samples, args.num_events, rank, world_size)
            val_dataset = DISDataset(simulator, val_samples, args.num_events, rank, world_size)
            input_dim = 6
        elif args.problem == "realistic_dis":
            simulator = RealisticDIS(device=device, smear=True, smear_std=0.05)
            train_dataset = RealisticDISDataset(
                simulator, args.num_samples, args.num_events, rank, world_size, feature_engineering=log_feature_engineering
            )
            val_dataset = RealisticDISDataset(
                simulator, val_samples, args.num_events, rank, world_size, feature_engineering=log_feature_engineering
            )
            input_dim = 6
        elif args.problem == "mceg":
            simulator = MCEGSimulator(device=device)
            train_dataset = MCEGDISDataset(
                simulator, args.num_samples, args.num_events, rank, world_size, feature_engineering=improved_feature_engineering
            )
            val_dataset = MCEGDISDataset(
                simulator, val_samples, args.num_events, rank, world_size, feature_engineering=improved_feature_engineering
            )
            input_dim = 2
        elif args.problem == "gaussian":
            simulator = Gaussian2DSimulator(device=device)
            train_dataset = Gaussian2DDataset(simulator, args.num_samples, args.num_events, rank, world_size)
            val_dataset = Gaussian2DDataset(simulator, val_samples, args.num_events, rank, world_size)
            input_dim = 2
    
    return train_dataset, val_dataset, input_dim


def main_worker(rank, world_size, args):
    # Only setup distributed training if world_size > 1
    if world_size > 1:
        setup(rank, world_size)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Use cuda:rank for multi-GPU, cuda:0 for single-GPU, cpu if no CUDA
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    if args.experiment_name is None:
        args.experiment_name = f"{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}_parameter_predidction"

    output_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Distributed dataset creation to prevent race conditions
    if world_size > 1 and hasattr(args, 'use_precomputed') and args.use_precomputed:
        # Only rank 0 generates precomputed data if needed
        if rank == 0:
            # This call may trigger generation if data doesn't exist
            train_dataset, val_dataset, input_dim = create_train_val_datasets(args, rank, world_size, device)
        
        # All ranks wait at barrier until data is ready
        dist.barrier()
        
        # After barrier, all ranks (including rank 0) create/load their datasets
        train_dataset, val_dataset, input_dim = create_train_val_datasets(args, rank, world_size, device)
    else:
        # Single rank case or non-precomputed data - no barrier needed
        train_dataset, val_dataset, input_dim = create_train_val_datasets(args, rank, world_size, device)

    print("🔧 DATALOADER CONFIGURATION DIAGNOSTIC:")
    print(f"   Creating DataLoaders with num_workers={args.dataloader_workers}")
    print(f"   Current multiprocessing start method: {mp.get_start_method()}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        print(f"   Current device: {device}")
        print(f"   CUDA initialized: {torch.cuda.is_initialized()}")
    
    if args.dataloader_workers > 0:
        print(f"   💡 Using {args.dataloader_workers} worker processes")
        print(f"      If you get 'Cannot re-initialize CUDA in forked subprocess' error,")
        print(f"      try --dataloader_workers=0 to disable multiprocessing")
    else:
        print(f"   🔧 Using 0 workers (main process only) - avoids multiprocessing issues")
    
    # Configure persistent workers only if using workers
    use_persistent = args.dataloader_workers > 0
    if not use_persistent:
        print(f"   🔧 Disabling persistent_workers (not compatible with num_workers=0)")
    print()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=4 if args.dataloader_workers > 0 else None,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=4 if args.dataloader_workers > 0 else None,
    )

    # Get theta and feature dims from a batch
    theta, x_sets = next(iter(train_dataloader))
    B, n_repeat, num_points, feat_dim = x_sets.shape
    theta_dim = theta.shape[1]
    latent_dim = args.latent_dim

    # Model setup
    if args.problem == 'mceg':
        model = ChunkedPointNetPMA(input_dim=input_dim, latent_dim=latent_dim, dropout=0.3, chunk_latent=128, num_seeds=8, num_heads=4).to(device)
    else:    
        model = PointNetPMA(input_dim=input_dim, latent_dim=latent_dim).to(device)
    if torch.__version__ >= "2.0":
        os.environ["TRITON_CACHE_DIR"] = "/pscratch/sd/k/katiekee/triton_cache"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/pscratch/sd/k/katiekee/inductor_cache"
        try:
            model = torch.compile(model, mode="default", dynamic=True)
        except Exception as e:
            print(f"[Rank {rank}] torch.compile failed: {e}")

    # Only use distributed wrappers in multi-GPU mode
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])
    
    if args.problem == 'mceg':
        theta_bounds = torch.tensor([
                [-1.0, 10.0],
                [0.0, 10.0],
                [-10.0, 10.0],
                [-10.0, 10.0],
            ])
    else:
        theta_bounds = None
    
    # Print parameter bounds for transparency
    if rank == 0:  # Only print once in distributed training
        print(f"\n📊 PARAMETER BOUNDS (main_worker):")
        print(f"   Problem type: {args.problem}")
        if theta_bounds is not None:
            print(f"   Bounds (min, max): {theta_bounds.tolist()}")
            print(f"   Ranges: {(theta_bounds[:, 1] - theta_bounds[:, 0]).tolist()}")
        else:
            print(f"   Bounds: Not specified for problem type '{args.problem}'")
        print()
    
    if theta_bounds is not None:
        theta_min, theta_max = theta_bounds[:, 0], theta_bounds[:, 1]
        theta_range = theta_max - theta_min
    else:
        theta_min = theta_max = theta_range = None
    param_prediction_model = MLPHead(latent_dim, theta_dim, dropout=0.3)
    param_prediction_model = param_prediction_model.to(device)
    
    # Only use DDP in multi-GPU mode
    if world_size > 1:
        param_prediction_model = DDP(param_prediction_model, device_ids=[rank])

    if rank != 0:
        os.environ["WANDB_MODE"] = "disabled"

    print("WANDB WAS ENABLED: ", args.wandb)
    if rank == 0 and args.wandb:
        wandb.init(project="quantom_end_to_end", name=args.experiment_name, config=vars(args))

    train_joint(model, param_prediction_model, train_dataloader, val_dataloader, args.num_epochs, args.lr, rank, args.wandb, output_dir, args=args)
    
    # Only cleanup in distributed mode
    if world_size > 1:
        cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--num_events", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_repeat", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--problem", type=str, default="simplified_dis")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default=None, help="Unique name for this ablation run")
    parser.add_argument("--use_precomputed", action="store_true", 
                       help="Use precomputed data instead of generating on-the-fly. Automatically generates data if not found.")
    parser.add_argument("--precomputed_data_dir", type=str, default="precomputed_data",
                       help="Directory containing precomputed data files")
    parser.add_argument("--single_gpu", action="store_true",
                       help="Force single-GPU mode even if multiple GPUs are available")
    parser.add_argument("--dataloader_workers", type=int, default=1,
                       help="Number of DataLoader worker processes (use 0 to avoid multiprocessing issues)")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("RANK", 0))

    # Determine training mode
    if args.single_gpu or world_size <= 1:
        print(f"🚀 EXECUTION MODE: Single-GPU")
        print(f"   single_gpu flag: {args.single_gpu}")
        print(f"   detected GPUs: {world_size}")
        if torch.cuda.is_available():
            print(f"   Using GPU: cuda:0")
        else:
            print("   No CUDA GPUs available, using CPU")
        print()
        # Run directly without mp.spawn
        try:
            main_worker(0, 1, args)
        except RuntimeError as e:
            if "Cannot re-initialize CUDA in forked subprocess" in str(e):
                print("🚨 CUDA MULTIPROCESSING ERROR DETECTED!")
                print("   This error occurs when CUDA is initialized before multiprocessing fork.")
                print("   💡 SOLUTIONS:")
                print("   1. Set multiprocessing start method to 'spawn' at script start:")
                print("      import torch.multiprocessing as mp")
                print("      mp.set_start_method('spawn', force=True)")
                print("   2. Or reduce DataLoader workers: add --dataloader_workers=0")
                print("   3. Or use CPU-only mode if testing")
                print()
            raise
    else:
        print(f"🚀 EXECUTION MODE: Distributed Multi-GPU")
        print(f"   Using {world_size} GPUs with mp.spawn")
        print(f"   Current multiprocessing method: {mp.get_start_method()}")
        if mp.get_start_method() != 'spawn':
            print(f"   ⚠️  WARNING: Current method is not 'spawn' - may cause CUDA issues")
        print()
        try:
            mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
        except RuntimeError as e:
            if "Cannot re-initialize CUDA in forked subprocess" in str(e):
                print("🚨 CUDA MULTIPROCESSING ERROR IN DISTRIBUTED MODE!")
                print("   This is likely due to CUDA being initialized before mp.spawn.")
                print("   💡 SOLUTIONS:")
                print("   1. Ensure multiprocessing start method is set to 'spawn' before any imports")
                print("   2. Move CUDA operations inside main_worker function")
                print("   3. Use --single_gpu flag to avoid distributed mode")
                print()
            raise
    if args.wandb and rank == 0:
        wandb.finish()