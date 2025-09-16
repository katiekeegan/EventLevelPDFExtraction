#!/usr/bin/env python3
"""
Test version of parameter_prediction.py with train-test split functionality
Uses simple simulators to avoid external dependencies
"""
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
import subprocess
import sys
import glob

import wandb
from simple_simulator import SimpleGaussian2DSimulator, SimpleSimplifiedDIS, log_feature_engineering, advanced_feature_engineering


class SimplePointNetPMA(nn.Module):
    """Simple PointNet-like model for testing"""
    def __init__(self, input_dim, latent_dim, predict_theta=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.predict_theta = predict_theta
        
    def forward(self, x):
        # x shape: [B, N, input_dim]
        # Apply encoder to each point
        encoded = self.encoder(x)  # [B, N, latent_dim]
        # Pool over points dimension
        pooled = encoded.mean(dim=1)  # [B, latent_dim]
        return pooled


class SimpleTransformerHead(nn.Module):
    """Simple transformer head for parameter prediction"""
    def __init__(self, latent_dim, theta_dim, ranges=None):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, theta_dim)
        )
        self.ranges = ranges
        
    def forward(self, x):
        out = self.layers(x)
        if self.ranges is not None:
            # Apply bounds if specified
            out = torch.sigmoid(out)
            ranges_tensor = self.ranges.to(x.device)
            out = out * (ranges_tensor[:, 1] - ranges_tensor[:, 0]) + ranges_tensor[:, 0]
        return out


class SimpleDISDataset(IterableDataset):
    """Simple DIS dataset for testing"""
    def __init__(self, simulator, num_samples, num_events, rank, world_size, theta_dim=4, n_repeat=2):
        self.simulator = simulator
        self.total_samples = num_samples
        self.samples_per_rank = num_samples // world_size
        self.num_events = num_events
        self.rank = rank
        self.world_size = world_size
        self.theta_bounds = torch.tensor([[0.0, 5.0]] * theta_dim)
        self.n_repeat = n_repeat
        self.theta_dim = theta_dim
        self.feature_engineering = advanced_feature_engineering

    def __len__(self):
        return self.samples_per_rank

    def __iter__(self):
        device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self.simulator.device = device

        # Set seed for reproducibility
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        seed = self.rank * 10000 + worker_id
        torch.manual_seed(seed)
        np.random.seed(seed)

        theta_bounds = self.theta_bounds.to(device)

        for _ in range(self.samples_per_rank):
            theta = torch.rand(self.theta_dim, device=device)
            theta = theta * (theta_bounds[:, 1] - theta_bounds[:, 0]) + theta_bounds[:, 0]

            xs = []
            for _ in range(self.n_repeat):
                x = self.simulator.sample(theta, self.num_events)
                xs.append(self.feature_engineering(x).cpu())

            yield theta.cpu().contiguous(), torch.stack(xs).cpu().contiguous()


class SimpleGaussian2DDataset(IterableDataset):
    """Simple Gaussian 2D dataset for testing"""
    def __init__(self, simulator, num_samples, num_events, rank, world_size, theta_dim=5, n_repeat=2):
        self.simulator = simulator
        self.total_samples = num_samples
        self.samples_per_rank = num_samples // world_size
        self.num_events = num_events
        self.rank = rank
        self.world_size = world_size
        self.theta_dim = theta_dim
        self.n_repeat = n_repeat

        # [mu_x, mu_y, sigma_x, sigma_y, rho]
        self.theta_bounds = torch.tensor([
            [-2.0, 2.0],   # mu_x
            [-2.0, 2.0],   # mu_y
            [0.5, 2.0],    # sigma_x
            [0.5, 2.0],    # sigma_y
            [-0.8, 0.8],   # rho
        ])

    def __len__(self):
        return self.samples_per_rank

    def __iter__(self):
        device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self.simulator.device = device

        # Set seed for reproducibility
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        seed = self.rank * 10000 + worker_id
        torch.manual_seed(seed)
        np.random.seed(seed)

        theta_bounds = self.theta_bounds.to(device)

        for _ in range(self.samples_per_rank):
            theta = torch.rand(self.theta_dim, device=device)
            theta = theta * (theta_bounds[:, 1] - theta_bounds[:, 0]) + theta_bounds[:, 0]

            xs = []
            for _ in range(self.n_repeat):
                x = self.simulator.sample(theta, self.num_events)
                xs.append(x.cpu())

            yield theta.cpu().contiguous(), torch.stack(xs).cpu().contiguous()


def save_laplace(lap_model, output_dir, filename, likelihood, subset_of_weights, hessian_structure):
    """Simple save function for Laplace model"""
    print(f"Saving Laplace model to {os.path.join(output_dir, filename)}")
    # For testing, just create a dummy file
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write("Laplace model saved successfully")


def train_joint(model, param_prediction_model, train_dataloader, val_dataloader, epochs, lr, rank, wandb_enabled, output_dir, save_every=10, args=None):
    device = next(model.parameters()).device
    param_prediction_model = param_prediction_model.to(device)
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
            B, n_repeat, num_points, feat_dim = x_sets.shape
            x_sets = x_sets.reshape(B*n_repeat, num_points, feat_dim)
            x_sets = x_sets.to(device)
            theta = theta.repeat_interleave(n_repeat, dim=0).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                emb = model(x_sets)
                predicted_theta = param_prediction_model(emb)
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
        
        # Simple Laplace approximation (mock for testing)
        save_laplace(None, output_dir,
                    filename="laplace_transformer.pt",
                    likelihood="regression",
                    subset_of_weights="last_layer",
                    hessian_structure="kron")


def create_train_val_datasets(args, rank, world_size, device):
    """Create training and validation datasets with clear separation."""
    val_samples = getattr(args, 'val_samples', 1000)
    
    print(f"Using on-the-fly data generation for {args.problem}")
    
    if args.problem == "simplified_dis":
        simulator = SimpleSimplifiedDIS(device=device)
        train_dataset = SimpleDISDataset(simulator, args.num_samples, args.num_events, rank, world_size)
        val_dataset = SimpleDISDataset(simulator, val_samples, args.num_events, rank, world_size)
        input_dim = 6
    elif args.problem == "gaussian":
        simulator = SimpleGaussian2DSimulator(device=device)
        train_dataset = SimpleGaussian2DDataset(simulator, args.num_samples, args.num_events, rank, world_size)
        val_dataset = SimpleGaussian2DDataset(simulator, val_samples, args.num_events, rank, world_size)
        input_dim = 2
    else:
        raise ValueError(f"Problem {args.problem} not supported in test version")
    
    return train_dataset, val_dataset, input_dim


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    torch.set_num_threads(1)


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, world_size, args):
    setup(rank, world_size)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if args.experiment_name is None:
        args.experiment_name = f"{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}_vs_{args.val_samples}_parameter_prediction"

    output_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Create training and validation datasets
    train_dataset, val_dataset, input_dim = create_train_val_datasets(args, rank, world_size, device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,  # Set to 0 for simplicity in testing
        pin_memory=False,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0,  # Set to 0 for simplicity in testing
        pin_memory=False,
    )

    # Get theta and feature dims from a batch
    theta, x_sets = next(iter(train_dataloader))
    B, n_repeat, num_points, feat_dim = x_sets.shape
    theta_dim = theta.shape[1]
    latent_dim = args.latent_dim

    # Model setup
    model = SimplePointNetPMA(input_dim=input_dim, latent_dim=latent_dim, predict_theta=False).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    if args.problem == 'mceg':
        theta_bounds = torch.tensor([
                [-1.0, 10.0],
                [0.0, 10.0],
                [-10.0, 10.0],
                [-10.0, 10.0],
            ])
    else:
        theta_bounds = None
    param_prediction_model = SimpleTransformerHead(latent_dim, theta_dim, ranges=theta_bounds)
    param_prediction_model = param_prediction_model.to(device)
    
    if world_size > 1:
        param_prediction_model = DDP(param_prediction_model, device_ids=[rank] if torch.cuda.is_available() else None)

    if rank != 0:
        os.environ["WANDB_MODE"] = "disabled"

    print("WANDB WAS ENABLED: ", args.wandb)
    if rank == 0 and args.wandb:
        wandb.init(project="quantom_end_to_end", name=args.experiment_name, config=vars(args))

    print(f"Starting training with {len(train_dataset)//world_size} training samples and {len(val_dataset)//world_size} validation samples per rank")
    
    train_joint(model, param_prediction_model, train_dataloader, val_dataloader, args.num_epochs, args.lr, rank, args.wandb, output_dir, args=args)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--num_events", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--problem", type=str, default="simplified_dis", choices=["simplified_dis", "gaussian"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default=None, help="Unique name for this ablation run")
    args = parser.parse_args()

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, args)
    
    if args.wandb and rank == 0:
        wandb.finish()