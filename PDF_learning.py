import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import *
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.parallel import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.cuda.amp as amp
import warnings
from models import *
import os
from simulator import RealisticDIS
from utils import *
from datasets import *

# Suppress numerical warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# SimplifiedDIS class with improved numerical stability
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

def get_optimizer(model, lr):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

def get_scheduler(optimizer, epochs):
    return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                       total_steps=epochs, 
                                       pct_start=0.3)

def setup(rank, world_size):
    import socket
    from contextlib import closing
    
    # Find a free port
    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]
    
    port = find_free_port()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args, xs, thetas, pointnet_model, problem='simplified_dis', output_dir=None):
    # Setup distributed training
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Initialize models with DDP
    def latent_fn(event):
        # Automatically use the same device as the model
        model_device = next(pointnet_model.parameters()).device
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            event = event.to(model_device)
            latent, _ = pointnet_model(event)
            latent = latent.squeeze(0).cpu()
            return latent
    
    # Create dataset with distributed sampler
    # dataset = EventDataset(xs_tensor_engineered, thetas, latent_fn)
    # Compute latents in chunks (saves memory)
    latent_path = os.path.abspath('latent_features.h5')

    # Only rank 0 creates the file
    if rank == 0:
        if os.path.exists(latent_path):
            os.remove(latent_path)
        precompute_features_and_latents_to_disk(pointnet_model, xs, thetas, latent_path, chunk_size=2)

    # All ranks wait until the file is ready
    dist.barrier()
    inference_net = InferenceNet(embedding_dim=args.latent_dim, output_dim=thetas.size(-1)).to(device)
    inference_net = DDP(inference_net, device_ids=[rank])
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
            print(f'Epoch: {epoch}, Loss: {epoch_loss/len(dataloader)}')
            total_losses.append(loss.item())
            if epoch % 100 == 0:
                torch.save(inference_net.module.state_dict(), os.path.join(output_dir, f"trained_inference_net_epoch_{epoch}.pth"))
    
    # Final save from rank 0
    if rank == 0:
        torch.save(inference_net.module.state_dict(), os.path.join(output_dir, "final_inference_net.pth"))
        np.save("loss_total.npy", np.array(total_losses))
    
    cleanup()

def setup(rank, world_size):
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--nodes', type=int, default=1)  # For multi-node support
    parser.add_argument('--problem', type=str, default='simplified_dis', choices=['simplified_dis', 'realistic_dis'])
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_events', type=int, default=100000)
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
    thetas, xs = generate_data(args.num_samples, args.num_events, problem=args.problem, device=device)
    input_dim = 6
    pointnet_model = PointNetPMA(input_dim=input_dim, latent_dim=args.latent_dim, predict_theta=True)
    # pointnet_model.load_state_dict(torch.load('pointnet_embedding_latent_dim_1024.pth', map_location='cpu'))
    state_dict = torch.load('experiments/simplified_dis_latent1024_ns_1000_ne_100000/model_epoch_200.pth', map_location='cpu')
    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    pointnet_model.load_state_dict(state_dict)
    pointnet_model.eval()

    def latent_fn(event):
        # Automatically use the same device as the model
        model_device = next(pointnet_model.parameters()).device
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            event = event.to(model_device)
            latent, _ = pointnet_model(event)
            latent = latent.squeeze(0).cpu()
            return latent
    
    if args.gpus > 1:
        # Multi-GPU setup
        world_size = args.gpus * args.nodes
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        torch.multiprocessing.spawn(
            train,
            args=(world_size, args, xs, thetas, pointnet_model, args.problem, output_dir),
            nprocs=args.gpus
        )
    else:
        # Single GPU setup
        pointnet_model = pointnet_model.to(device)
        # Initialize dataset with streaming
        # dataset = EventDataset(xs_tensor_engineered, thetas, latent_fn)
        # Compute latents in chunks (saves memory)
        latent_path = 'latent_features.h5'
        if not os.path.exists(latent_path):
            xs_tensor_engineered = log_feature_engineering(xs)

            # Initialize PointNetEmbedding model (do this once)
            input_dim = xs_tensor_engineered.shape[-1]
            print(f'[precompute] Input dimension: {input_dim}')
            print(f'xs_tensor_engineered shape: {xs_tensor_engineered.shape}')
            precompute_latents_to_disk(pointnet_model, 
                                    xs_tensor_engineered,
                                    latent_path,
                                    chunk_size=8)
            del xs_tensor_engineered
            del xs

        dataset = H5Dataset(latent_path, thetas)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, 
                                collate_fn=EventDataset.collate_fn, num_workers=0, pin_memory=True, persistent_workers=False)
        
        inference_net = InferenceNet(embedding_dim=args.latent_dim, output_dim=thetas.size(-1)).to(device)
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
                    pred_params = inference_net(latent_embeddings)
                    xs = torch.rand(1000, device=device)
                    
                    recon_theta = inference_net(latent_embeddings)
                    recon_loss = F.mse_loss(recon_theta, true_params)
                    param_mins = torch.tensor([0.1, -1.0, 0.1, -1.0], device=device)
                    param_maxs = torch.tensor([5.0, -0.5, 5.0, -0.5], device=device)
                    normalized_pred = (recon_theta - param_mins) / (param_maxs - param_mins)
                    normalized_true = (true_params - param_mins) / (param_maxs - param_mins)
                    loss = F.mse_loss(normalized_pred, normalized_true)
                
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()
                
                epoch_loss += loss.item()
            
            print(f'Epoch: {epoch}, Loss: {epoch_loss/len(dataloader)}')
            if epoch % 10 == 0:
                torch.save(inference_net.state_dict(), os.path.join(output_dir, "trained_inference_net.pth"))
        
        torch.save(inference_net.state_dict(), os.path.join(output_dir,"final_inference_net.pth"))

if __name__ == "__main__":
    main()