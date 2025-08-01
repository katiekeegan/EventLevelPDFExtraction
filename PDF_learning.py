import os
import warnings

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributions import *
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from datasets import *
from models import *
from simulator import (Gaussian2DSimulator, MCEGSimulator, RealisticDIS,
                       SimplifiedDIS)
from utils import *

# Ensure reproducibility
torch.manual_seed(42)
# Set default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Suppress warnings from h5py about file locking
warnings.filterwarnings("ignore", category=UserWarning, module="h5py")
# Suppress warnings from numpy about deprecated features
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
# Suppress warnings from torch about deprecated features
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
# Suppress warnings from torch about numerical issues
warnings.filterwarnings("ignore", category=UserWarning, module="torch._tensor")
# Suppress warnings from torch about deprecated features
# Suppress numerical warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def train(
    rank,
    world_size,
    args,
    xs,
    thetas,
    pointnet_model,
    problem="simplified_dis",
    output_dir=None,
):
    # Setup distributed training
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Initialize models with DDP
    def latent_fn(event):
        # Automatically use the same device as the model
        model_device = next(pointnet_model.parameters()).device
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            event = event.to(model_device)
            latent, _ = pointnet_model(event)
            latent = latent.squeeze(0).cpu()
            return latent

    # Create dataset with distributed sampler
    # dataset = EventDataset(xs_tensor_engineered, thetas, latent_fn)
    # Compute latents in chunks (saves memory)
    latent_path = os.path.abspath("latent_features.h5")

    # Only rank 0 creates the file
    if rank == 0:
        if os.path.exists(latent_path):
            os.remove(latent_path)
        precompute_features_and_latents_to_disk(
            pointnet_model, xs, thetas, latent_path, chunk_size=2
        )

    # All ranks wait until the file is ready
    dist.barrier()
    inference_net = InferenceNet(
        embedding_dim=args.latent_dim, output_dim=thetas.size(-1), nll_mode=args.nll_loss
    ).to(device)
    inference_net = DDP(inference_net, device_ids=[rank])
    dataset = H5Dataset(latent_path)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        sampler=sampler,
        collate_fn=EventDataset.collate_fn,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    # Optimizer and scheduler
    optimizer = get_optimizer(inference_net, lr=1e-4)
    # scheduler = get_scheduler(optimizer, epochs=args.epochs)

    # Mixed precision and gradient scaling
    scaler = amp.GradScaler()
    if rank == 0:
        recon_losses = []
        normalized_losses = []
        total_losses = []

    # Training loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # Important for shuffling
        epoch_loss = 0.0

        for batch_idx, (latent_embeddings, true_params) in enumerate(dataloader):
            torch.cuda.empty_cache()
            latent_embeddings = latent_embeddings.to(device, non_blocking=True)
            true_params = true_params.to(device, non_blocking=True)
            with amp.autocast(dtype=torch.float16):
                # Forward pass
                if args.nll_loss:
                    # NLL mode: get means and log-variances
                    means, log_vars = inference_net(latent_embeddings)
                    
                    # Normalize for consistent comparison
                    if problem == "simplified_dis":
                        param_mins = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
                        param_maxs = torch.tensor([5.0, 5.0, 5.0, 5.0], device=device)
                    elif problem == "realistic_dis":
                        param_mins = torch.tensor(
                            [-2.0, -1.0, 0.0, 0.0, -5.0, -5.0], device=device
                        )
                        param_maxs = torch.tensor(
                            [2.0, 1.0, 5.0, 10.0, 5.0, 5.0], device=device
                        )
                    
                    # Normalize means and true parameters
                    normalized_means = (means - param_mins) / (param_maxs - param_mins)
                    normalized_true = (true_params - param_mins) / (param_maxs - param_mins)
                    
                    # Use Gaussian NLL loss
                    loss = gaussian_nll_loss(normalized_means, log_vars, normalized_true)
                else:
                    # Original MSE mode
                    recon_theta = inference_net(latent_embeddings)
                    if problem == "simplified_dis":
                        param_mins = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
                        param_maxs = torch.tensor([5.0, 5.0, 5.0, 5.0], device=device)
                    elif problem == "realistic_dis":
                        param_mins = torch.tensor(
                            [-2.0, -1.0, 0.0, 0.0, -5.0, -5.0], device=device
                        )
                        param_maxs = torch.tensor(
                            [2.0, 1.0, 5.0, 10.0, 5.0, 5.0], device=device
                        )
                    normalized_pred = (recon_theta - param_mins) / (param_maxs - param_mins)
                    normalized_true = (true_params - param_mins) / (param_maxs - param_mins)
                    loss = F.mse_loss(normalized_pred, normalized_true)

            # Backward pass with gradient scaling
            optimizer.zero_grad(set_to_none=True)
            # print(loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()

            epoch_loss += loss.item()

        # Only print from rank 0
        if rank == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss/len(dataloader)}")
            total_losses.append(loss.item())
            if epoch % 100 == 0:
                torch.save(
                    inference_net.module.state_dict(),
                    os.path.join(
                        output_dir, f"trained_inference_net_epoch_{epoch}.pth"
                    ),
                )

    # Final save from rank 0
    if rank == 0:
        torch.save(
            inference_net.module.state_dict(),
            os.path.join(output_dir, "final_inference_net.pth"),
        )
        np.save("loss_total.npy", np.array(total_losses))

    cleanup()


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--nodes", type=int, default=1)  # For multi-node support
    parser.add_argument(
        "--problem",
        type=str,
        default="simplified_dis",
        choices=["simplified_dis", "realistic_dis"],
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_events", type=int, default=100000)
    parser.add_argument("--nll-loss", action="store_true", 
                       help="Use Gaussian negative log-likelihood loss with mean and variance prediction")
    args = parser.parse_args()

    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}_inference"

    # Define output directory and create it
    output_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Common setup for both single and multi-GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Sample and prepare data (do this once, not in both branches)
    num_samples = 1000
    num_events = 100000
    thetas, xs = generate_data(
        args.num_samples, args.num_events, problem=args.problem, device=device
    )
    if args.problem == "simplified_dis":
        input_dim = 6
    elif args.problem == "realistic_dis":
        input_dim = 12
    pointnet_model = PointNetPMA(
        input_dim=input_dim, latent_dim=args.latent_dim, predict_theta=True
    )
    # pointnet_model.load_state_dict(torch.load('pointnet_embedding_latent_dim_1024.pth', map_location='cpu'))
    state_dict = torch.load(
        "experiments/simplified_dis_latent1024_ns_1000_ne_100000/final_model.pth",
        map_location="cpu",
    )
    # Remove 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # Strip '_orig_mod.' prefix
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Load into your model
    pointnet_model.load_state_dict(state_dict)
    pointnet_model.eval()

    def latent_fn(event):
        # Automatically use the same device as the model
        model_device = next(pointnet_model.parameters()).device
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            event = event.to(model_device)
            latent, _ = pointnet_model(event)
            latent = latent.squeeze(0).cpu()
            return latent

    if args.gpus > 1:
        # Multi-GPU setup
        world_size = args.gpus * args.nodes
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        torch.multiprocessing.spawn(
            train,
            args=(
                world_size,
                args,
                xs,
                thetas,
                pointnet_model,
                args.problem,
                output_dir,
            ),
            nprocs=args.gpus,
        )
    else:
        # Single GPU setup
        pointnet_model = pointnet_model.to(device)
        # Initialize dataset with streaming
        # dataset = EventDataset(xs_tensor_engineered, thetas, latent_fn)
        # Compute latents in chunks (saves memory)
        latent_path = "latent_features.h5"
        if not os.path.exists(latent_path):
            xs_tensor_engineered = log_feature_engineering(xs)

            # Initialize PointNetEmbedding model (do this once)
            input_dim = xs_tensor_engineered.shape[-1]
            print(f"[precompute] Input dimension: {input_dim}")
            print(f"xs_tensor_engineered shape: {xs_tensor_engineered.shape}")
            precompute_latents_to_disk(
                pointnet_model, xs_tensor_engineered, latent_path, chunk_size=8
            )
            del xs_tensor_engineered
            del xs

        dataset = H5Dataset(latent_path)

        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=EventDataset.collate_fn,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )

        inference_net = InferenceNet(
            embedding_dim=args.latent_dim, output_dim=thetas.size(-1), nll_mode=args.nll_loss
        ).to(device)
        optimizer = get_optimizer(inference_net, lr=1e-4)
        # scheduler = get_scheduler(optimizer, epochs=args.epochs)
        scaler = amp.GradScaler()

        # Training loop
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for batch_idx, (latent_embeddings, true_params) in enumerate(dataloader):
                torch.cuda.empty_cache()
                latent_embeddings = latent_embeddings.to(device, non_blocking=True)
                true_params = true_params.to(device, non_blocking=True)

                with amp.autocast(dtype=torch.float16):
                    # Forward pass
                    if args.nll_loss:
                        # NLL mode: get means and log-variances
                        means, log_vars = inference_net(latent_embeddings)
                        
                        # Get parameter bounds for normalization
                        if args.problem == "simplified_dis":
                            param_mins = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
                            param_maxs = torch.tensor([5.0, 5.0, 5.0, 5.0], device=device)
                        elif args.problem == "realistic_dis":
                            param_mins = torch.tensor(
                                [-2.0, -1.0, 0.0, 0.0, -5.0, -5.0], device=device
                            )
                            param_maxs = torch.tensor(
                                [2.0, 1.0, 5.0, 10.0, 5.0, 5.0], device=device
                            )
                        
                        # Normalize means and true parameters  
                        normalized_means = (means - param_mins) / (param_maxs - param_mins)
                        normalized_true = (true_params - param_mins) / (param_maxs - param_mins)
                        
                        # Use Gaussian NLL loss
                        loss = gaussian_nll_loss(normalized_means, log_vars, normalized_true)
                    else:
                        # Original MSE mode
                        recon_theta = inference_net(latent_embeddings)
                        if args.problem == "simplified_dis":
                            param_mins = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
                            param_maxs = torch.tensor([5.0, 5.0, 5.0, 5.0], device=device)
                        elif args.problem == "realistic_dis":
                            param_mins = torch.tensor(
                                [-2.0, -1.0, 0.0, 0.0, -5.0, -5.0], device=device
                            )
                            param_maxs = torch.tensor(
                                [2.0, 1.0, 5.0, 10.0, 5.0, 5.0], device=device
                            )
                        normalized_pred = (recon_theta - param_mins) / (
                            param_maxs - param_mins
                        )
                        normalized_true = (true_params - param_mins) / (
                            param_maxs - param_mins
                        )
                        loss = F.mse_loss(normalized_pred, normalized_true)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()

                epoch_loss += loss.item()

            print(f"Epoch: {epoch}, Loss: {epoch_loss/len(dataloader)}")
            if epoch % 10 == 0:
                torch.save(
                    inference_net.state_dict(),
                    os.path.join(output_dir, "trained_inference_net.pth"),
                )

        torch.save(
            inference_net.state_dict(),
            os.path.join(output_dir, "final_inference_net.pth"),
        )


if __name__ == "__main__":
    main()
