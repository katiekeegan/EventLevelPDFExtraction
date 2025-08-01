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

import wandb
from datasets import *
from models import PointNetPMA
from simulator import (Gaussian2DSimulator, MCEGSimulator, RealisticDIS,
                       SimplifiedDIS)
from utils import *


def train(
    model, dataloader, epochs, lr, rank, wandb_enabled, output_dir, save_every=10
):
    device = next(model.parameters()).device
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    alpha1, alpha2 = 0.01, 0.01  # regularization weights

    model.train()
    torch.backends.cudnn.benchmark = True  # Enable autotuner

    for epoch in range(epochs):
        total_loss = 0.0

        for theta, x_sets in dataloader:
            x_sets = x_sets.to(torch.float32)  # or .float()
            B, n_repeat, num_points, feat_dim = x_sets.shape

            # Efficient reshape and repeat
            x_sets = x_sets.reshape(B * n_repeat, num_points, feat_dim).to(device)
            theta = theta.repeat_interleave(n_repeat, dim=0).to(device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                dtype=torch.float16
            ):  # try float16 for better perf on A100s
                latent = model(x_sets)  # [B * n_repeat, D]
                contrastive = triplet_theta_contrastive_loss(latent, theta)

                # Regularization: L2 norm of embeddings
                l2_reg = latent.norm(p=2, dim=1).mean()

                # Covariance decorrelation (fast)
                z = latent - latent.mean(dim=0, keepdim=True)
                cov = z.T @ z / (z.size(0) - 1)
                off_diag = cov * (1 - torch.eye(cov.size(0), device=device))
                decorrelation = (off_diag**2).sum()

                loss = contrastive + alpha1 * l2_reg + alpha2 * decorrelation

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
                        "loss": loss.item(),
                        "contrastive": contrastive.item(),
                        "l2_reg": l2_reg.item(),
                        "decorrelation": decorrelation.item(),
                        "param_mse": param_accuracy,
                    }
                )
            print(
                f"Epoch {epoch + 1}, "
                f"Loss: {loss.item():.4f}, "
                f"Contrastive: {contrastive.item():.4f}, "
                f"L2 Reg: {l2_reg.item():.4f}, "
                f"Decorrelation: {decorrelation.item():.4f}"
            )
            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"),
                )

    if rank == 0:
        torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))


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

    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}"

    # Define output directory and create it
    output_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

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

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    if not (args.problem == "gaussian"):
        dummy_theta = torch.zeros(input_dim, device=device)
        if args.problem == "mceg":
            # Just generating a test parameter to get the right input dimension
            dummy_theta = torch.tensor([-7.10000000e-01, 3.48000000e+00, 1.34000000e+00,2.33000000e+01], device=device)
        dummy_x = simulator.sample(dummy_theta, args.num_events)
        input_dim = log_feature_engineering(dummy_x).shape[-1]
        print("Sampled successfully from MCEG simulator!")
        model = PointNetPMA(
            input_dim=input_dim, latent_dim=args.latent_dim, predict_theta=True
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

        # Wandb safety setup
    if rank != 0:
        os.environ["WANDB_MODE"] = "disabled"

    if rank == 0 and args.wandb:
        wandb.init(project="quantom_cl", name=args.experiment_name, config=vars(args))

    train(model, dataloader, args.num_epochs, args.lr, rank, args.wandb, output_dir)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_events", type=int, default=100000)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--problem", type=str, default="simplified_dis")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Unique name for this ablation run",
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    # DDP setup: get rank from environment or torch.distributed
    rank = int(os.environ.get("RANK", 0))

    # Only initialize wandb in main process
    if args.wandb and rank == 0:
        wandb.init(
            project="PDFParameterInference",
            config={
                # add your config here, e.g.
                "learning_rate": 1e-3,
                "epochs": 10,
            }
        )
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    # Optionally finish wandb run
    if args.wandb and rank == 0:
        wandb.finish()
