import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from models import *
from datasets import *
from utils import *
from cl import triplet_theta_contrastive_loss, train, SimplifiedDIS
import argparse

import socket

import numpy as np

def get_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)

def find_free_port():
    """Finds a free port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)

def train_stage_2_worker(rank, world_size, args, xs, thetas, pointnet_model, dataset, problem='simplified_dis'):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    pointnet_model.eval().to(device)

    latent_dim = args.latent_dim
    inference_net = InferenceNet(embedding_dim=latent_dim, output_dim=thetas.size(-1)).to(device)
    inference_net = DDP(inference_net, device_ids=[rank])

    if args.no_precompute_latents:
        def latent_collate_fn(batch):
            xs_batch, thetas_batch = zip(*batch)
            xs_batch = torch.stack(xs_batch)  # keep on CPU
            thetas_batch = torch.stack(thetas_batch)  # also CPU

            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                xs_batch = xs_batch.to(device)
                thetas_batch = thetas_batch.to(device)
                latents = pointnet_model(xs_batch)[0]

            return latents, thetas_batch

        # IterableDataset can't use sampler — instead set shuffle=True manually if needed
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=latent_collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    else:
        latent_path = os.path.abspath('latent_features.h5')
        if rank == 0:
            if os.path.exists(latent_path):
                os.remove(latent_path)
            precompute_features_and_latents_to_disk(pointnet_model, xs, thetas, latent_path, chunk_size=2)

        dist.barrier()
        dataset = H5Dataset(latent_path)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            sampler=sampler,
            collate_fn=EventDataset.collate_fn,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False
        )

    optimizer = get_optimizer(inference_net, lr=1e-4)
    scaler = amp.GradScaler()

    for epoch in range(args.num_epochs_stage_2):
        if not args.no_precompute_latents:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for latent_embeddings, true_params in dataloader:
            torch.cuda.empty_cache()
            latent_embeddings = latent_embeddings.to(device, non_blocking=True)
            true_params = true_params.to(device, non_blocking=True)

            with amp.autocast(dtype=torch.float16):
                recon_theta = inference_net(latent_embeddings)
                if problem == 'simplified_dis':
                    param_mins = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
                    param_maxs = torch.tensor([5.0, 5.0, 5.0, 5.0], device=device)
                elif problem == 'realistic_dis':
                    param_mins = torch.tensor([-2.0, -1.0, 0.0, 0.0, -5.0, -5.0], device=device)
                    param_maxs = torch.tensor([2.0, 1.0, 5.0, 10.0, 5.0, 5.0], device=device)

                normalized_pred = (recon_theta - param_mins) / (param_maxs - param_mins)
                normalized_true = (true_params - param_mins) / (param_maxs - param_mins)
                loss = F.mse_loss(normalized_pred, normalized_true)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        if rank == 0:
            print(f"[Stage 2 - Rank {rank}] Epoch {epoch}, Loss: {epoch_loss / len(dataloader)}")
            if epoch % 10 == 0:
                torch.save(inference_net.module.state_dict(), f"trained_inference_net_epoch_{epoch}.pth")

    if rank == 0:
        torch.save(inference_net.module.state_dict(), "final_inference_net.pth")

    cleanup()

def setup(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")  # only used if not already set
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # ----- Setup experiment -----
    if args.experiment_name is None:
        args.experiment_name = f"{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}"
    output_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # ----- Simulator & Dataset -----
    if args.problem == 'simplified_dis':
        simulator = SimplifiedDIS(device=device)
        dataset = DISDataset(simulator, args.num_samples, args.num_events, rank, world_size)
        input_dim = 4
    elif args.problem == 'realistic_dis':
        simulator = RealisticDIS(device=device, smear=True, smear_std=0.05)
        dataset = RealisticDISDataset(simulator, args.num_samples, args.num_events, rank, world_size, feature_engineering=log_feature_engineering)
        input_dim = 6
    elif args.problem == 'gaussian':
        simulator = Gaussian2DSimulator(device=device)
        dataset = Gaussian2DDataset(simulator, args.num_samples, args.num_events, rank, world_size)
        input_dim = 2

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # ----- Stage 1: Train PointNet with contrastive loss -----
    dummy_theta = torch.rand(input_dim, device=device)
    dummy_x = simulator.sample(dummy_theta, args.num_events)
    input_dim = log_feature_engineering(dummy_x).shape[-1]

    model = PointNetPMA(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        predict_theta=True
    ).to(device)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    if rank != 0:
        os.environ["WANDB_MODE"] = "disabled"
    if rank == 0 and args.wandb:
        wandb.init(project="quantom_cl", name=args.experiment_name + "_stage1", config=vars(args))

    train(model, dataloader, args.num_epochs_stage_1, args.lr, rank, args.wandb, output_dir)

    torch.cuda.empty_cache()
    dist.barrier()  # Make sure all ranks finish Stage 1

    # ----- Stage 2 setup and launch -----
    thetas, xs = generate_data(args.num_samples, args.num_events, problem=args.problem, device=device)

    if rank == 0:
        torch.save(model.module.state_dict(), 'most_recent_model.pth')
        print("Saved PointNet model, now launching Stage 2 training...")

    # All ranks launch Stage 2 (DDP-aware)
    train_stage_2(args, model.module, xs, thetas, dataset)

    cleanup()

def train_stage_2(args, pointnet_model, xs, thetas, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pointnet_model.eval()
    xs = xs.cpu()
    thetas = thetas.cpu()

    if args.gpus > 1:
        world_size = args.gpus * args.nodes
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = find_free_port()

        torch.multiprocessing.spawn(
            train_stage_2_worker,
            args=(world_size, args, xs, thetas, pointnet_model, dataset, args.problem),
            nprocs=args.gpus
        )
    else:
        # Single GPU version (rank=0, world_size=1)
        train_stage_2_worker(0, 1, args, xs, thetas, pointnet_model, dataset, args.problem)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, choices=['simplified_dis', 'realistic_dis', 'gaussian'], default='simplified_dis')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_events', type=int, default=100000)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--num_epochs-stage-1', type=int, default=20)
    parser.add_argument('--num_epochs-stage-2', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes for DDP')
    parser.add_argument('--no_precompute_latents', action='store_true', help='Skip latent precomputation and reuse the same dataset as in Stage 1')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    world_size = args.gpus
    torch.multiprocessing.spawn(main_worker, args=(world_size, args), nprocs=world_size)
