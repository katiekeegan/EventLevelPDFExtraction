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
from torch.utils.data import DataLoader, IterableDataset

# Try to import from PDF_learning_UQ, but provide fallbacks if not available
try:
    from PDF_learning_UQ import MLPHead, TransformerHead
except ImportError:
    # Create minimal implementations
    class MLPHead(nn.Module):
        def __init__(self, latent_dim, theta_dim, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, theta_dim)
            )
        
        def forward(self, x):
            return self.net(x)
    
    class TransformerHead(nn.Module):
        def __init__(self, latent_dim, theta_dim, num_heads=8):
            super().__init__()
            self.latent_dim = latent_dim
            self.theta_dim = theta_dim
            
            # Simple transformer-like structure
            self.attention = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(latent_dim)
            self.ff = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 2),
                nn.ReLU(),
                nn.Linear(latent_dim * 2, latent_dim)
            )
            self.norm2 = nn.LayerNorm(latent_dim)
            self.output_proj = nn.Linear(latent_dim, theta_dim)
        
        def forward(self, x):
            # x shape: [batch_size, latent_dim]
            # Add sequence dimension for attention
            x = x.unsqueeze(1)  # [batch_size, 1, latent_dim]
            
            # Self-attention
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            
            # Feed-forward
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
            
            # Remove sequence dimension and project to output
            x = x.squeeze(1)  # [batch_size, latent_dim]
            return self.output_proj(x)

try:
    from utils import pdf_theta_loss
except ImportError:
    def pdf_theta_loss(pred, target, problem):
        return F.mse_loss(pred, target)

import wandb
from datasets import *
from models import PointNetPMA
try:
    from simulator import (Gaussian2DSimulator, MCEGSimulator, RealisticDIS,
                           SimplifiedDIS)
    from utils import *
except ImportError:
    # We'll handle this in the dataset creation function
    pass

# Import precomputed dataset functionality
try:
    from precomputed_datasets import create_precomputed_dataloader
    PRECOMPUTED_AVAILABLE = True
except ImportError:
    PRECOMPUTED_AVAILABLE = False
    print("Warning: Precomputed datasets not available, using on-the-fly generation only.")


class TinyThetaHead(nn.Module):
    """
    Extremely small head: latent_dim -> theta_dim
    One tiny hidden layer to keep capacity minimal.
    """
    def __init__(self, latent_dim: int, theta_dim: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, theta_dim, bias=True)
        )

    def forward(self, z):
        return self.net(z)

def train(
    model,
    dataloader,
    epochs,
    lr,
    rank,
    wandb_enabled,
    output_dir,
    save_every=10,
    args=None,
    theta_head: nn.Module = None,
    beta_l1: float = 0.1,      # weight for SmoothL1(theta_pred, theta)
    beta_cos: float = 0.1,     # weight for cosine alignment
    alpha1: float = 0.1,      # existing L2 reg weight
    alpha2: float = 0.1,        # existing decorrelation weight
    simulator=None
):
    device = next(model.parameters()).device
    assert theta_head is not None, "Please create a TinyThetaHead and pass it as theta_head."
    theta_head = theta_head.to(device)

    # IMPORTANT: optimize both networks
    opt = optim.Adam(
        list(model.parameters()) + list(theta_head.parameters()),
        lr=lr, weight_decay=1e-4
    )

    scaler = torch.cuda.amp.GradScaler()
    l1_loss = nn.SmoothL1Loss(reduction="mean")
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

    model.train()
    theta_head.train()
    torch.backends.cudnn.benchmark = True  # Enable autotuner

    for epoch in range(epochs):
        total_loss = 0.0

        for theta, x_sets in dataloader:
            x_sets = x_sets.to(torch.float32)
            B, n_repeat, num_points, feat_dim = x_sets.shape

            # Efficient reshape and repeat
            x_sets = x_sets.reshape(B * n_repeat, num_points, feat_dim).to(device)
            theta = theta.repeat_interleave(n_repeat, dim=0).to(device)  # [B*n_repeat, theta_dim]

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):  # good for A100s
                # Get latent embedding from your encoder
                latent = model(x_sets)  # shape [B*n_repeat, D]

                # # Existing contrastive objective (assumes your function is defined elsewhere)
                # contrastive = triplet_theta_contrastive_loss(latent, theta, margin=0.2, sim_threshold=0.1, dissim_threshold=0.5)

                # # Regularization: L2 norm of embeddings
                l2_reg = latent.norm(p=2, dim=1).mean()

                # # Covariance decorrelation (fast)
                z = latent - latent.mean(dim=0, keepdim=True)
                cov = z.T @ z / (z.size(0) - 1)
                off_diag = cov * (1 - torch.eye(cov.size(0), device=device))
                decorrelation = (off_diag**2).sum()

                # ===== Auxiliary theta head =====
                theta_pred = theta_head(latent)
                                   # [B*n_repeat, theta_dim]
                # loss_theta_l1 = l1_loss(theta_pred, theta)        # Smooth L1
                # # Cosine similarity -> convert to a loss in [0, 2]
                # loss_theta_cos = (1.0 - cos_sim(theta_pred, theta)).mean()

                # loss = pdf_theta_loss(
                    # theta_pred, theta, args.problem
                # )

                l1_loss = F.l1_loss(theta_pred, theta)

                loss = l1_loss #+ alpha1 * decorrelation + alpha2 * l2_reg

                # # Total loss
                # loss = (
                #     contrastive
                #     + alpha1 * l2_reg
                #     + alpha2 * decorrelation
                #     + beta_l1 * loss_theta_l1
                #     + beta_cos * loss_theta_cos
                # )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()

        # Logging and checkpointing (only on rank 0)
        if rank == 0:
            if wandb_enabled:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "loss_total": loss.item(),
                        # "contrastive": contrastive.item(),
                        "l2_reg": l2_reg.item(),
                        "decorrelation": decorrelation.item(),
                        # "aux_theta_l1": loss_theta_l1.item(),
                        # "aux_theta_cos": loss_theta_cos.item(),
                    }
                )
            print(
                f"Epoch {epoch + 1} | "
                f"Total: {loss.item():.4f} | "
                # f"Contr: {contrastive.item():.4f} | "
                f"L2: {l2_reg.item():.4f} | "
                f"Decorr: {decorrelation.item():.4f} | "
                # f"ThetaL1: {loss_theta_l1.item():.4f} | "
                # f"ThetaCos: {loss_theta_cos.item():.4f}"
            )

            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                torch.save( model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"), )
                torch.save( theta_head.state_dict(), os.path.join(output_dir, f"theta_heads_epoch_{epoch+1}.pth"), )
    if rank == 0:
        torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
        torch.save(theta_head.state_dict(), os.path.join(output_dir, f"theta_head_{args.problem}.pth"))

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)


def cleanup():
    dist.destroy_process_group()


def create_dataset_and_dataloader(args, rank, world_size, device):
    """
    Create dataset and dataloader based on args.use_precomputed flag.
    
    Returns:
        dataloader: DataLoader instance
        input_dim: Input dimension for the model
        simulator: Simulator instance (may be None for precomputed)
    """
    
    if args.use_precomputed and PRECOMPUTED_AVAILABLE:
        print(f"[Rank {rank}] Using precomputed data from {args.precomputed_data_dir}")
        
        # Create precomputed dataloader
        dataloader = create_precomputed_dataloader(
            data_dir=args.precomputed_data_dir,
            problem=args.problem,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,
            rank=rank,
            world_size=world_size,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        
        # Determine input dimension from a sample batch
        sample_theta, sample_events = next(iter(dataloader))
        input_dim = sample_events.shape[-1]  # feature dimension
        
        print(f"[Rank {rank}] Precomputed data: theta shape {sample_theta.shape}, "
              f"events shape {sample_events.shape}, input_dim {input_dim}")
        
        return dataloader, input_dim, None
        
    else:
        print(f"[Rank {rank}] Using on-the-fly data generation")
        
        try:
            # Import simulators with fallback handling
            from simulator import (Gaussian2DSimulator, MCEGSimulator, RealisticDIS,
                                   SimplifiedDIS)
            from utils import log_feature_engineering
        except ImportError as e:
            raise ImportError(f"On-the-fly data generation requires working simulators. "
                              f"Import error: {e}. Please use --use_precomputed instead.")
        
        # Create on-the-fly datasets (original implementation)
        if args.problem == "simplified_dis":
            simulator = SimplifiedDIS(device=device)
            dataset = DISDataset(
                simulator, args.num_samples, args.num_events, rank, world_size
            )
            input_dim = 4
        elif args.problem == "realistic_dis":
            simulator = RealisticDIS(device=device, smear=True, smear_std=0.05)
            print("Simulator constructed!")
            dataset = RealisticDISDataset(
                simulator,
                args.num_samples,
                args.num_events,
                rank,
                world_size,
                feature_engineering=log_feature_engineering,
            )
            print("Dataset created!")
            input_dim = 6
        elif args.problem == "mceg":
            simulator = MCEGSimulator(device=device)
            print("Simulator constructed!")
            dataset = MCEGDISDataset(
                simulator,
                args.num_samples,
                args.num_events,
                rank,
                world_size,
                feature_engineering=log_feature_engineering,
            )
            print("Dataset created!")
            input_dim = 4
        elif args.problem == "gaussian":
            simulator = Gaussian2DSimulator(device=device)
            dataset = Gaussian2DDataset(
                simulator, args.num_samples, args.num_events, rank, world_size
            )
            input_dim = 2
        else:
            raise ValueError(f"Unknown problem: {args.problem}")

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        
        return dataloader, input_dim, simulator


def main_worker(rank, world_size, args):
    setup(rank, world_size)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(f"cuda:{rank}")

    # Generate experiment name if not provided
    if args.experiment_name is None:
        data_suffix = "precomputed" if args.use_precomputed else "onthefly"
        args.experiment_name = f"{args.problem}_{data_suffix}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}"

    # Define output directory and create it
    output_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Create dataset and dataloader
    dataloader, input_dim, simulator = create_dataset_and_dataloader(args, rank, world_size, device)
    # Handle model creation - need to determine actual input_dim for non-gaussian problems
    if not (args.problem == "gaussian"):
        if args.problem == "mceg":
            model_input_dim = 2
        else:
            if not args.use_precomputed:
                # For on-the-fly generation, determine input_dim by creating a dummy sample
                try:
                    from utils import log_feature_engineering
                    dummy_theta = torch.zeros(input_dim, device=device)
                    dummy_x = simulator.sample(dummy_theta, args.num_events)
                    model_input_dim = log_feature_engineering(dummy_x).shape[-1]
                except ImportError:
                    # Fallback: just use the input_dim as-is
                    model_input_dim = input_dim
            else:
                # For precomputed data, input_dim is already determined from the loaded data
                model_input_dim = input_dim
                
        model = PointNetPMA(
            input_dim=model_input_dim, latent_dim=args.latent_dim, predict_theta=True
        ).to(device)

        if torch.__version__ >= "2.0":
            os.environ["TRITON_CACHE_DIR"] = "/pscratch/sd/k/katiekee/triton_cache"
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = (
                "/pscratch/sd/k/katiekee/inductor_cache"
            )
            try:
                model = torch.compile(model, mode="default", dynamic=True)
            except Exception as e:
                print(f"[Rank {rank}] torch.compile failed: {e}")

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])
        
        # After you have a batch:
        theta, x_sets = next(iter(dataloader))
        B, n_repeat, num_points, feat_dim = x_sets.shape

        param = next(model.parameters())
        param_device = param.device
        param_dtype = param.dtype

        # move + cast to exactly the model's dtype/device
        theta   = theta.to(device=param_device, dtype=param_dtype)
        x_sets  = x_sets.to(device=param_device, dtype=param_dtype)

        x_test = x_sets.reshape(B * n_repeat, num_points, feat_dim)
        with torch.no_grad():
            z = model(x_test)  # [B*n_repeat, D]
        latent_dim = z.shape[1]
        theta_dim = theta.shape[1]
        if args.head == "MLP":
            theta_head = MLPHead(latent_dim, theta_dim).to(device)
        elif args.head == "Transformer":
            theta_head = TransformerHead(latent_dim, theta_dim).to(device)
        theta_head = DDP(theta_head, device_ids=[rank])             # no SyncBN needed
    else:
        # For gaussian problem, create a simpler model setup if needed
        model = PointNetPMA(
            input_dim=input_dim, latent_dim=args.latent_dim, predict_theta=True
        ).to(device)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])
        
        # Determine theta head parameters
        theta, x_sets = next(iter(dataloader))
        B, n_repeat, num_points, feat_dim = x_sets.shape
        
        param = next(model.parameters())
        param_device = param.device
        param_dtype = param.dtype
        
        theta = theta.to(device=param_device, dtype=param_dtype)
        x_sets = x_sets.to(device=param_device, dtype=param_dtype)
        
        x_test = x_sets.reshape(B * n_repeat, num_points, feat_dim)
        with torch.no_grad():
            z = model(x_test)
        latent_dim = z.shape[1]
        theta_dim = theta.shape[1]
        
        if args.head == "MLP":
            theta_head = MLPHead(latent_dim, theta_dim).to(device)
        elif args.head == "Transformer":
            theta_head = TransformerHead(latent_dim, theta_dim).to(device)
        theta_head = DDP(theta_head, device_ids=[rank])

        # Wandb safety setup
    if rank != 0:
        os.environ["WANDB_MODE"] = "disabled"

    print("WANDB WAS ENABLED: ", args.wandb)
    if rank == 0 and args.wandb:
        wandb.init(project="quantom_end_to_end", name=args.experiment_name, config=vars(args))

    train(model, dataloader, args.num_epochs, args.lr, rank, args.wandb, output_dir, args=args, theta_head=theta_head, simulator=simulator)
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_events", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--problem", type=str, default="simplified_dis")
    parser.add_argument('--head', type=str, default='MLP')
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Unique name for this ablation run",
    )
    
    # New arguments for precomputed data
    parser.add_argument(
        "--use_precomputed", 
        action="store_true",
        help="Use precomputed datasets instead of on-the-fly generation"
    )
    parser.add_argument(
        "--precomputed_data_dir",
        type=str,
        default="precomputed_data",
        help="Directory containing precomputed .npz files"
    )
    
    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    # DDP setup: get rank from environment or torch.distributed
    rank = int(os.environ.get("RANK", 0))

    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    # Optionally finish wandb run
    if args.wandb and rank == 0:
        wandb.finish()