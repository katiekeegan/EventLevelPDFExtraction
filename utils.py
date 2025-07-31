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
import h5py
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset
# Ensure reproducibility
torch.manual_seed(42)
# Set default device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature Engineering with improved stability
def log_feature_engineering(xs_tensor):
    # Basic features with clamping for numerical stability
    xs_clamped = torch.clamp(xs_tensor, min=1e-8, max=1e8)
    del xs_tensor
    log_features = torch.log1p(xs_clamped)
    symlog_features = torch.sign(xs_clamped) * torch.log1p(xs_clamped.abs())

    # Pairwise features with vectorized operations
    n_features = xs_clamped.shape[-1]
    # del xs_tensor
    combinations = torch.combinations(torch.arange(n_features), r=2)
    i, j = combinations[:, 0], combinations[:, 1]
    del combinations
    # Safe division with clamping
    ratio = xs_clamped[..., i] / (xs_clamped[..., j] + 1e-8)
    ratio_features = torch.log1p(ratio.abs())
    del ratio
    
    diff_features = torch.log1p(xs_clamped[..., i]) - torch.log1p(xs_clamped[..., j])
    del xs_clamped
    data = torch.cat([log_features, symlog_features, ratio_features, diff_features], dim=-1)
    return data

def precompute_features_and_latents_to_disk(pointnet_model, xs_tensor, thetas, output_path, chunk_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pointnet_model = pointnet_model.to(device)
    pointnet_model.eval()

    latent_dim = 1024
    num_samples = xs_tensor.shape[0]
    print(f"[precompute] Saving HDF5 to: {output_path}")
    if os.path.dirname(output_path) != '':
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        latent_dset = f.create_dataset('latents', shape=(num_samples, latent_dim), dtype=np.float32)
        theta_dset = f.create_dataset('thetas', data=thetas.cpu().numpy(), dtype=np.float32)

        for i in tqdm(range(num_samples)):
            print(f"xs_tensor shape: {xs_tensor.shape}")
            xs_sample = xs_tensor[i].unsqueeze(0).cpu()  # shape: (1, ...)
            xs_engineered = log_feature_engineering(xs_sample)  # CPU-safe
            xs_engineered = xs_engineered.to(device)

            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                latent = pointnet_model(xs_engineered)
                latent = latent.squeeze().cpu().numpy()  # shape: (latent_dim,)

            latent_dset[i] = latent

            del xs_sample, xs_engineered, latent
            torch.cuda.empty_cache()

def precompute_latents_to_disk(pointnet_model, xs_tensor, thetas, output_path, chunk_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pointnet_model = pointnet_model.to(device)
    pointnet_model.eval()
    
    with h5py.File(output_path, 'w') as f:
        latent_shape = (len(xs_tensor), 1024)
        latent_dset = f.create_dataset('latents', shape=latent_shape, dtype=np.float32,
                                       chunks=(chunk_size, latent_shape[1]))
        
        theta_dset = f.create_dataset('thetas', data=thetas.cpu().numpy(), dtype=np.float32)

        for i in tqdm(range(0, len(xs_tensor), chunk_size)):
            chunk = xs_tensor[i:i+chunk_size].to(device)
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                latent = pointnet_model(chunk)
                latent = latent.squeeze(1).cpu().numpy()
            latent_dset[i:i+len(latent)] = latent
            del chunk, latent
            torch.cuda.empty_cache()
    
    return output_path

def precompute_latents_chunked(pointnet_model, xs_tensor, chunk_size=32, device='cuda'):
    """
    Compute latent features in memory-friendly chunks and normalize them.
    
    Returns:
        latents_normalized: Tensor of normalized latent features
        z_mean: Tensor of latent feature means (shape: [latent_dim])
        z_std: Tensor of latent feature stds (shape: [latent_dim])
    """
    pointnet_model = pointnet_model.to(device)
    pointnet_model.eval()
    
    latents = []
    with torch.no_grad():
        for i in range(0, len(xs_tensor), chunk_size):
            chunk = xs_tensor[i:i+chunk_size].to(device)
            latent, _ = pointnet_model(chunk).squeeze(1).cpu()  # shape: [B, latent_dim]
            latents.append(latent)
            del chunk, latent
            torch.cuda.empty_cache()
    
    latents = torch.cat(latents, dim=0)  # shape: [N, latent_dim]
    
    # Compute normalization stats
    z_mean = latents.mean(dim=0, keepdim=True)  # shape: [1, latent_dim]
    z_std = latents.std(dim=0, keepdim=True) + 1e-6  # Avoid divide-by-zero
    
    # Normalize
    latents_normalized = (latents - z_mean) / z_std

    return latents_normalized, z_mean.squeeze(), z_std.squeeze()

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

def get_optimizer(model, lr):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

def get_scheduler(optimizer, epochs):
    return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                       total_steps=epochs, 
                                       pct_start=0.3)


def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size):
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)