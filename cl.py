import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from datasets import *
from utils import *

from models import PointNetPMA

from simulator import RealisticDIS, Gaussian2DSimulator, MCEGSimulator

class SimplifiedDIS:
    def __init__(self, device=None, smear=False, smear_std=0.05):
        self.device = device
        self.smear = smear
        self.smear_std = smear_std
        self.Nu = 1
        self.Nd = 2
        self.au, self.bu, self.ad, self.bd = None, None, None, None

    def init(self, params):
        self.au, self.bu, self.ad, self.bd = [p.to(self.device) for p in params]

    def up(self, x):
        return self.Nu * (x ** self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        return self.Nd * (x ** self.ad) * ((1 - x) ** self.bd)

    def sample(self, params, nevents=1):
        self.init(params)
        eps = 1e-6
        rand = lambda: torch.clamp(torch.rand(nevents, device=self.device), min=eps, max=1 - eps)
        smear_noise = lambda s: s + torch.randn_like(s) * (self.smear_std * s) if self.smear else s

        xs_p, xs_n = rand(), rand()
        sigma_p = smear_noise(4 * self.up(xs_p) + self.down(xs_p))
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0, posinf=1e8, neginf=0.0)
        sigma_n = smear_noise(4 * self.down(xs_n) + self.up(xs_n))
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0, posinf=1e8, neginf=0.0)
        return torch.stack([sigma_p, sigma_n], dim=-1)

def pairwise_cosine_similarity(x):
    x = F.normalize(x, p=2, dim=1)
    return x @ x.T  # [B, B]


def triplet_theta_contrastive_loss(z, theta, margin=0.5, sim_threshold=0.1, dissim_threshold=0.3):
    # Normalize embeddings
    z = F.normalize(z, p=2, dim=1)  # [B, D]

    # Pairwise distances in parameter space (faster than cdist)
    theta_diff = theta[:, None, :] - theta[None, :, :]  # [B, B, D_theta]
    theta_d = theta_diff.norm(dim=-1)  # [B, B]

    # Positive and negative masks (excluding diagonal)
    eye = torch.eye(theta_d.size(0), device=theta.device).bool()
    sim = (theta_d < sim_threshold) & (~eye)
    dissim = (theta_d > dissim_threshold) & (~eye)

    dist = 1 - z @ z.T  # [B, B], cosine distances

    # Safe fill values depending on AMP dtype
    fill_neg = torch.tensor(-1e3, dtype=z.dtype, device=z.device)
    fill_pos = torch.tensor(1e3, dtype=z.dtype, device=z.device)

    hardest_pos = dist.masked_fill(~sim, fill_neg)
    easiest_neg = dist.masked_fill(~dissim, fill_pos)

    hardest_pos_val, _ = hardest_pos.max(dim=1)
    easiest_neg_val, _ = easiest_neg.min(dim=1)

    triplet_loss = F.relu(hardest_pos_val - easiest_neg_val + margin)
    return triplet_loss.mean()

def train(model, dataloader, epochs, lr, rank, wandb_enabled, output_dir, save_every=10):
    device = next(model.parameters()).device
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    alpha1, alpha2 = 0.01, 0.01  # regularization weights

    model.train()
    torch.backends.cudnn.benchmark = True  # Enable autotuner

    for epoch in range(epochs):
        total_loss = 0.0

        for theta, x_sets in dataloader:
            B, n_repeat, num_points, feat_dim = x_sets.shape

            # Efficient reshape and repeat
            x_sets = x_sets.reshape(B * n_repeat, num_points, feat_dim).to(device)
            theta = theta.repeat_interleave(n_repeat, dim=0).to(device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):  # try float16 for better perf on A100s
                latent = model(x_sets)  # [B * n_repeat, D]
                contrastive = triplet_theta_contrastive_loss(latent, theta)

                # Regularization: L2 norm of embeddings
                l2_reg = latent.norm(p=2, dim=1).mean()

                # Covariance decorrelation (fast)
                z = latent - latent.mean(dim=0, keepdim=True)
                cov = z.T @ z / (z.size(0) - 1)
                off_diag = cov * (1 - torch.eye(cov.size(0), device=device))
                decorrelation = (off_diag ** 2).sum()

                loss = contrastive + alpha1 * l2_reg + alpha2 * decorrelation

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()

        # Logging and checkpointing (only on rank 0)
        if rank == 0:
            if wandb_enabled:
                wandb.log({
                    "epoch": epoch + 1,
                    "loss": loss.item(),
                    "contrastive": contrastive.item(),
                    "l2_reg": l2_reg.item(),
                    "decorrelation": decorrelation.item(),
                    "param_mse": param_accuracy,
                })
            print(
                f"Epoch {epoch + 1}, "
                f"Loss: {loss.item():.4f}, "
                f"Contrastive: {contrastive.item():.4f}, "
                f"L2 Reg: {l2_reg.item():.4f}, "
                f"Decorrelation: {decorrelation.item():.4f}"
            )
            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"))

    if rank == 0:
        torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))




def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
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

    if args.problem == 'simplified_dis':
        simulator = SimplifiedDIS(device=device)
        dataset = DISDataset(simulator, args.num_samples, args.num_events, rank, world_size)
        input_dim = 4
    elif args.problem == 'realistic_dis':
        simulator = RealisticDIS(device=device, smear=True, smear_std=0.05)
        print("Simulator constructed!")
        dataset = RealisticDISDataset(simulator, args.num_samples, args.num_events, rank, world_size, feature_engineering=log_feature_engineering)
        print("Dataset created!")
        input_dim = 6
    elif args.problem == 'mceg':
        simulator = MCEGSimulator(device=device)
        print("Simulator constructed!")
        dataset = RealisticDISDataset(simulator, args.num_samples, args.num_events, rank, world_size, feature_engineering=log_feature_engineering)
        print("Dataset created!")
        input_dim = 4
    elif args.problem == 'gaussian':
        simulator = Gaussian2DSimulator(device=device)
        dataset = Gaussian2DDataset(
        simulator, args.num_samples, args.num_events, rank, world_size
        )
        input_dim = 2

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    if not (args.problem == 'gaussian'):
        dummy_theta = torch.zeros(input_dim, device=device)
        if args.problem == 'mceg':
            dummy_theta = torch.tensor([1.0, 1.0, -10.0, -10.0], device=device)
        dummy_x = simulator.sample(dummy_theta, args.num_events)
        input_dim = log_feature_engineering(dummy_x).shape[-1]

        model = PointNetPMA(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            predict_theta=True
        ).to(device)

        if torch.__version__ >= "2.0":
            os.environ["TRITON_CACHE_DIR"] = "/pscratch/sd/k/katiekee/triton_cache"
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/pscratch/sd/k/katiekee/inductor_cache"
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
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_events', type=int, default=100000)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--problem', type=str, default='simplified_dis')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--experiment_name', type=str, default=None, help='Unique name for this ablation run')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
