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
    Check if precomputed data exists for the given parameters, and generate it if not.
    
    Args:
        problem: Problem type ('gaussian', 'simplified_dis', 'realistic_dis', 'mceg')
        num_samples: Number of theta parameter samples
        num_events: Number of events per simulation
        n_repeat: Number of repeated simulations per theta
        output_dir: Directory where precomputed data should be stored
    
    Returns:
        str: Path to the data directory
    """
    if not PRECOMPUTED_AVAILABLE:
        raise RuntimeError("Precomputed data support not available. Please check precomputed_datasets.py")
    
    # Check if data already exists
    pattern = os.path.join(output_dir, f"{problem}_ns{num_samples}_ne{num_events}_nr{n_repeat}.npz")
    if os.path.exists(pattern):
        print(f"Precomputed data already exists: {pattern}")
        return output_dir
    
    # Check for any existing data files for this problem (with different parameters)
    existing_pattern = os.path.join(output_dir, f"{problem}_*.npz")
    all_existing_files = glob.glob(existing_pattern)
    existing_files = filter_valid_precomputed_files(all_existing_files)
    
    if existing_files:
        print(f"Found {len(existing_files)} valid data files for {problem}: {existing_files}")
        if len(all_existing_files) > len(existing_files):
            temp_files = [f for f in all_existing_files if f not in existing_files]
            print(f"Ignored {len(temp_files)} temporary/incomplete files: {temp_files}")
        print(f"Using existing valid data instead of generating new data")
        return output_dir
    elif all_existing_files:
        temp_files = [f for f in all_existing_files if f not in existing_files]
        print(f"Found {len(temp_files)} temporary/incomplete files for {problem}: {temp_files}")
        print(f"No valid precomputed data found - all found files are temporary/incomplete")
        print(f"Proceeding to generate new data...")
    
    # Generate new data
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
def train_joint(model, param_prediction_model, train_dataloader, val_dataloader, epochs, lr, rank, wandb_enabled, output_dir, save_every=10, args=None):
    device = next(model.parameters()).device
    param_prediction_model = param_prediction_model.to(device)  # FIX: Move CNF to GPU before DDP
    opt = optim.Adam(list(model.parameters()) + list(param_prediction_model.parameters()), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    param_prediction_model.train()
    torch.backends.cudnn.benchmark = True

    for epoch in range(epochs):
        # Training phase
        total_train_loss = 0.0
        num_train_batches = 0
        
        for theta, x_sets in train_dataloader:
            x_sets = x_sets.to(torch.float32)
            B, n_repeat, num_points, feat_dim = x_sets.shape  # B = batch_size*n_repeat, N = num_points, F = feat_dim
            x_sets = x_sets.reshape(B*n_repeat, num_points, feat_dim)  # [B*N, F]
            x_sets = x_sets.to(device)
            theta = theta.repeat_interleave(n_repeat, dim=0).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                emb = model(x_sets)  # [B*n_repeat, latent_dim]
                predicted_theta = param_prediction_model(emb)  # [B*n_repeat, theta_dim]
                loss = F.mse_loss(predicted_theta, theta)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        
        # Validation phase
        model.eval()
        param_prediction_model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for theta, x_sets in val_dataloader:
                x_sets = x_sets.to(torch.float32)
                B, n_repeat, num_points, feat_dim = x_sets.shape
                x_sets = x_sets.reshape(B*n_repeat, num_points, feat_dim)
                x_sets = x_sets.to(device)
                theta = theta.repeat_interleave(n_repeat, dim=0).to(device)
                
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    emb = model(x_sets)
                    predicted_theta = param_prediction_model(emb)
                    val_loss = F.mse_loss(predicted_theta, theta)
                
                total_val_loss += val_loss.item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
        
        # Switch back to training mode
        model.train()
        param_prediction_model.train()

        # Logging and checkpointing (only on rank 0)
        if rank == 0:
            if wandb_enabled:
                wandb.log({
                    "epoch": epoch + 1, 
                    "train_mse_loss": avg_train_loss,
                    "val_mse_loss": avg_val_loss
                })
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"))
                torch.save(param_prediction_model.state_dict(), os.path.join(output_dir, f"params_model_epoch_{epoch+1}.pth"))
    if rank == 0:
        torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
        torch.save(param_prediction_model.state_dict(), os.path.join(output_dir, "final_params_model.pth"))
        from laplace import Laplace
        lap_transformer = Laplace(param_prediction_model, 'regression', subset_of_weights='last_layer', hessian_structure='kron')
        lap_transformer.fit(train_dataloader)  # Use training dataloader for Laplace fitting
        # after fitting:
        save_laplace(lap_transformer, output_dir,
                    filename="laplace_transformer.pt",
                    likelihood="regression",
                    subset_of_weights="last_layer",
                    hessian_structure="kron")

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
                n_repeat=1,
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
            
            # Create distributed datasets
            if world_size > 1:
                train_dataset = DistributedPrecomputedDataset(
                    train_data_dir, args.problem, rank, world_size, shuffle=True
                )
                val_dataset = DistributedPrecomputedDataset(
                    val_data_dir, args.problem, rank, world_size, shuffle=False
                )
            else:
                train_dataset = PrecomputedDataset(train_data_dir, args.problem, shuffle=True)
                val_dataset = PrecomputedDataset(val_data_dir, f"{args.problem}_val", shuffle=False)
            
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
                simulator, args.num_samples, args.num_events, rank, world_size, feature_engineering=log_feature_engineering
            )
            val_dataset = MCEGDISDataset(
                simulator, val_samples, args.num_events, rank, world_size, feature_engineering=log_feature_engineering
            )
            input_dim = 2
        elif args.problem == "gaussian":
            simulator = Gaussian2DSimulator(device=device)
            train_dataset = Gaussian2DDataset(simulator, args.num_samples, args.num_events, rank, world_size)
            val_dataset = Gaussian2DDataset(simulator, val_samples, args.num_events, rank, world_size)
            input_dim = 2
    
    return train_dataset, val_dataset, input_dim


def main_worker(rank, world_size, args):
    setup(rank, world_size)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(f"cuda:{rank}")

    if args.experiment_name is None:
        args.experiment_name = f"{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}_parameter_predidction"

    output_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Create training and validation datasets
    train_dataset, val_dataset, input_dim = create_train_val_datasets(args, rank, world_size, device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Get theta and feature dims from a batch
    theta, x_sets = next(iter(train_dataloader))
    B, n_repeat, num_points, feat_dim = x_sets.shape
    theta_dim = theta.shape[1]
    latent_dim = args.latent_dim

    # Model setup
    model = PointNetPMA(input_dim=input_dim, latent_dim=latent_dim, predict_theta=False).to(device)
    if torch.__version__ >= "2.0":
        os.environ["TRITON_CACHE_DIR"] = "/pscratch/sd/k/katiekee/triton_cache"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/pscratch/sd/k/katiekee/inductor_cache"
        try:
            model = torch.compile(model, mode="default", dynamic=True)
        except Exception as e:
            print(f"[Rank {rank}] torch.compile failed: {e}")

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
    param_prediction_model = TransformerHead(latent_dim, theta_dim, ranges=theta_bounds)
    param_prediction_model = param_prediction_model.to(device)
    param_prediction_model = DDP(param_prediction_model, device_ids=[rank])

    if rank != 0:
        os.environ["WANDB_MODE"] = "disabled"

    print("WANDB WAS ENABLED: ", args.wandb)
    if rank == 0 and args.wandb:
        wandb.init(project="quantom_end_to_end", name=args.experiment_name, config=vars(args))

    train_joint(model, param_prediction_model, train_dataloader, val_dataloader, args.num_epochs, args.lr, rank, args.wandb, output_dir, args=args)
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--num_events", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--problem", type=str, default="simplified_dis")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default=None, help="Unique name for this ablation run")
    parser.add_argument("--use_precomputed", action="store_true", 
                       help="Use precomputed data instead of generating on-the-fly. Automatically generates data if not found.")
    parser.add_argument("--precomputed_data_dir", type=str, default="precomputed_data",
                       help="Directory containing precomputed data files")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("RANK", 0))

    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    if args.wandb and rank == 0:
        wandb.finish()