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

import wandb
from datasets import *
from models import *
from simulator import (Gaussian2DSimulator, MCEGSimulator, RealisticDIS, SimplifiedDIS)
from utils import *

# ---------------------------
# Conditional Normalizing Flow (CNF)
# ---------------------------
class AffineCoupling(nn.Module):
    """Simple affine coupling layer for RealNVP-style flows."""
    def __init__(self, dim, hidden_dim, context_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        self.F = nn.Sequential(
            nn.Linear(dim // 2 + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2 * 2)
        )

    def forward(self, x, context, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        h = torch.cat([x1, context], dim=1)
        out = self.F(h)
        shift, scale = out.chunk(2, dim=1)
        scale = torch.tanh(scale)  # Stabilize scale

        if not reverse:
            y2 = x2 * torch.exp(scale) + shift
            log_det = scale.sum(dim=1)
        else:
            y2 = (x2 - shift) * torch.exp(-scale)
            log_det = -scale.sum(dim=1)
        y = torch.cat([x1, y2], dim=1)
        return y, log_det

class SimpleCNF(nn.Module):
    """Stack of affine coupling layers with context."""
    def __init__(self, theta_dim, context_dim, hidden_dim=128, num_layers=6):
        super().__init__()
        self.theta_dim = theta_dim
        self.context_dim = context_dim
        self.layers = nn.ModuleList([
            AffineCoupling(theta_dim, hidden_dim, context_dim)
            for _ in range(num_layers)
        ])
        # Learnable base distribution parameters
        self.base_mean = nn.Parameter(torch.zeros(theta_dim))
        self.base_logstd = nn.Parameter(torch.zeros(theta_dim))

    def forward(self, theta, context):
        z = theta
        log_det_sum = torch.zeros(theta.size(0), device=theta.device)
        for layer in self.layers:
            z, log_det = layer(z, context, reverse=False)
            log_det_sum += log_det
        return z, log_det_sum

    def inverse(self, z, context):
        x = z
        log_det_sum = torch.zeros(z.size(0), device=z.device)
        for layer in reversed(self.layers):
            x, log_det = layer(x, context, reverse=True)
            log_det_sum += log_det
        return x, log_det_sum

    def log_prob(self, theta, context):
        z, log_det = self.forward(theta, context)
        log_pz = -0.5 * (((z - self.base_mean) / self.base_logstd.exp()) ** 2).sum(dim=1)
        log_pz += -self.base_logstd.sum() - 0.5 * self.theta_dim * np.log(2 * np.pi)
        return log_pz + log_det

# ---------------------------
# Joint Training Function
# ---------------------------
def train_joint(model, param_prediction_model, dataloader, epochs, lr, rank, wandb_enabled, output_dir, save_every=10, args=None):
    device = next(model.parameters()).device
    param_prediction_model = param_prediction_model.to(device)  # FIX: Move CNF to GPU before DDP
    opt = optim.Adam(list(model.parameters()) + list(param_prediction_model.parameters()), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    param_prediction_model.train()
    torch.backends.cudnn.benchmark = True

    for epoch in range(epochs):
        total_loss = 0.0
        for theta, x_sets in dataloader:
            x_sets = x_sets.to(torch.float32)
            B, n_repeat, num_points, feat_dim = x_sets.shape  # B = batch_size*n_repeat, N = num_points, F = feat_dim
            x_sets = x_sets.reshape(B*n_repeat, num_points, feat_dim)  # [B*N, F]
            x_sets = x_sets.to(device)
            theta = theta.repeat_interleave(n_repeat, dim=0).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                emb = model(x_sets)  # [B*n_repeat, latent_dim]
                # log_prob = cnf.module.log_prob(theta, emb)
                # loss = -log_prob.mean()
                predicted_theta = param_prediction_model(emb)  # [B*n_repeat, theta_dim]
                loss = F.mse_loss(predicted_theta, theta)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()

        # Logging and checkpointing (only on rank 0)
        if rank == 0:
            if wandb_enabled:
                wandb.log({"epoch": epoch + 1, "mse_loss": loss.item()})
            print(f"Epoch {epoch + 1} | MSE Loss: {loss.item():.4f}")

            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"))
                torch.save(param_prediction_model.state_dict(), os.path.join(output_dir, f"params_model_epoch_{epoch+1}.pth"))
    if rank == 0:
        torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
        torch.save(param_prediction_model.state_dict(), os.path.join(output_dir, "final_params_model.pth"))
        from laplace import Laplace
        lap_transformer = Laplace(param_prediction_model, 'regression', subset_of_weights='last_layer', hessian_structure='kron')
        lap_transformer.fit(train_loader)
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

def main_worker(rank, world_size, args):
    setup(rank, world_size)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(f"cuda:{rank}")

    if args.experiment_name is None:
        args.experiment_name = f"{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}_parameter_predidction"

    output_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Dataset and simulator setup
    if args.problem == "simplified_dis":
        simulator = SimplifiedDIS(device=device)
        dataset = DISDataset(simulator, args.num_samples, args.num_events, rank, world_size)
        input_dim = 6
    elif args.problem == "realistic_dis":
        simulator = RealisticDIS(device=device, smear=True, smear_std=0.05)
        dataset = RealisticDISDataset(
            simulator, args.num_samples, args.num_events, rank, world_size, feature_engineering=log_feature_engineering
        )
        input_dim = 6
    elif args.problem == "mceg":
        simulator = MCEGSimulator(device=device)
        dataset = MCEGDISDataset(
            simulator, args.num_samples, args.num_events, rank, world_size, feature_engineering=log_feature_engineering
        )
        input_dim = 2
    elif args.problem == "gaussian":
        simulator = Gaussian2DSimulator(device=device)
        dataset = Gaussian2DDataset(simulator, args.num_samples, args.num_events, rank, world_size)
        input_dim = 2

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Get theta and feature dims from a batch
    theta, x_sets = next(iter(dataloader))
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
    else
        theta_bounds = None
    param_prediction_model = TransformerHead(latent_dim, theta_dim, ranges=theta_bounds))
    param_prediction_model = param_prediction_model.to(device)
    param_prediction_model = DDP(param_prediction_model, device_ids=[rank])

    if rank != 0:
        os.environ["WANDB_MODE"] = "disabled"

    print("WANDB WAS ENABLED: ", args.wandb)
    if rank == 0 and args.wandb:
        wandb.init(project="quantom_end_to_end", name=args.experiment_name, config=vars(args))

    train_joint(model, param_prediction_model, dataloader, args.num_epochs, args.lr, rank, args.wandb, output_dir, args=args)
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_events", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--problem", type=str, default="simplified_dis")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default=None, help="Unique name for this ablation run")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("RANK", 0))

    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    if args.wandb and rank == 0:
        wandb.finish()