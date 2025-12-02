import os

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import *
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (DataLoader, Dataset, IterableDataset,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from simulator import *

# Ensure reproducibility
torch.manual_seed(42)
# Set default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    data = torch.cat(
        [log_features, symlog_features, ratio_features, diff_features], dim=-1
    )
    return data


def precompute_features_and_latents_to_disk(
    pointnet_model, xs_tensor, thetas, output_path, chunk_size=4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pointnet_model = pointnet_model.to(device)
    pointnet_model.eval()

    latent_dim = 1024
    num_samples = xs_tensor.shape[0]
    print(f"[precompute] Saving HDF5 to: {output_path}")
    if os.path.dirname(output_path) != "":
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as f:
        latent_dset = f.create_dataset(
            "latents", shape=(num_samples, latent_dim), dtype=np.float32
        )
        theta_dset = f.create_dataset(
            "thetas", data=thetas.cpu().numpy(), dtype=np.float32
        )

        for i in tqdm(range(num_samples)):
            print(f"xs_tensor shape: {xs_tensor.shape}")
            xs_sample = xs_tensor[i].unsqueeze(0).cpu()  # shape: (1, ...)
            xs_engineered = log_feature_engineering(xs_sample)  # CPU-safe
            xs_engineered = xs_engineered.to(device)

            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                latent = pointnet_model(xs_engineered)
                latent = latent.squeeze().cpu().numpy()  # shape: (latent_dim,)

            latent_dset[i] = latent

            del xs_sample, xs_engineered, latent
            torch.cuda.empty_cache()


def precompute_latents_to_disk(
    pointnet_model, xs_tensor, thetas, output_path, latent_dim, chunk_size=4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pointnet_model = pointnet_model.to(device)
    pointnet_model.eval()

    with h5py.File(output_path, "w") as f:
        latent_shape = (len(xs_tensor), latent_dim)
        latent_dset = f.create_dataset(
            "latents",
            shape=latent_shape,
            dtype=np.float32,
            chunks=(chunk_size, latent_shape[1]),
        )

        theta_dset = f.create_dataset(
            "thetas", data=thetas.cpu().numpy(), dtype=np.float32
        )

        for i in tqdm(range(0, len(xs_tensor), chunk_size)):
            chunk = xs_tensor[i : i + chunk_size].to(device)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                latent = pointnet_model(chunk)
                latent = latent.squeeze(1).cpu().numpy()
            latent_dset[i : i + len(latent)] = latent
            del chunk, latent
            torch.cuda.empty_cache()

    return output_path


def precompute_latents_chunked(pointnet_model, xs_tensor, chunk_size=32, device="cuda"):
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
            chunk = xs_tensor[i : i + chunk_size].to(device)
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


def triplet_theta_contrastive_loss(
    z, theta, margin=0.5, sim_threshold=0.1, dissim_threshold=0.3
):
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
    return optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=epochs, pct_start=0.3
    )


def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


@torch.no_grad()
def _make_default_grids(
    device, x_range=(1e-3, 0.9), n_x=500, Q2_slices=(1.0, 1.5, 2.0, 10.0, 50.0)
):
    # log-space in x is common for PDFs
    x_vals = torch.logspace(
        torch.log10(torch.tensor(x_range[0])),
        torch.log10(torch.tensor(x_range[1])),
        steps=n_x,
        device=device,
        dtype=torch.float32,
    )
    Q2_vals = torch.tensor(Q2_slices, device=device, dtype=torch.float32)  # (S,)
    return x_vals, Q2_vals


def _compute_pdf_grid(simulator, params_batch, x_vals, Q2_vals):
    """
    params_batch: (B, P) tensor of parameters
    x_vals: (X,)  ; Q2_vals: (S,)
    returns: q_pred of shape (B, S, X)
    """
    B = params_batch.shape[0]
    S = Q2_vals.shape[0]
    X = x_vals.shape[0]
    q_pred = []

    for b in range(B):
        simulator.init(params_batch[b])
        # stack all S slices for efficiency
        q_slices = []
        for s in range(S):
            x_vec = x_vals
            Q2_vec = torch.full_like(x_vals, Q2_vals[s])
            # q(x, Q2) -> (X,)
            q_slices.append(simulator.q(x_vec, Q2_vec).reshape(1, X))
        q_pred.append(torch.cat(q_slices, dim=0).unsqueeze(0))  # (1, S, X)

    return torch.cat(q_pred, dim=0)  # (B, S, X)


def pdf_theta_loss(theta_pred, theta_true, problem):
    """
    Generalized PDF loss with signature (theta_pred, simulator) only.

    What the simulator should (ideally) provide:
      - Ground truth params:
           simulator.theta_true  (B, P) tensor, or
           simulator.get_true_params() -> (B, P)
      - Optional evaluation grids:
           simulator.x_grid  -> (X,)
           simulator.Q2_grid -> (S,)
      - For RealisticDIS-style:
           EITHER simulator.q_from_theta(theta_batch, x_grid, Q2_grid) -> (B, S, X)
           OR     simulator.init(theta); simulator.q(x_vec, Q2_vec) -> (X,)
      - For SimplifiedDIS-style (no 'q'):
           List of component function names on the simulator:
             simulator.pdf_fn_names = ['up','down', ...]
           Each fn is callable: getattr(simulator, name)(x_vec) -> (X,)
           (If pdf_fn_names is absent, it will try to use ['up','down'] if present.)

    Returns:
        scalar loss tensor (log-relative entrywise PDF discrepancy).
    """
    device = theta_pred.device
    eps = 1e-12

    if problem == "simplified_dis":
        simulator = SimplifiedDIS(device=device)
        x_vals = torch.linspace(0, 1, 500).to(device)
        pred_vals_all = []
        true_vals_all = []
        for fn_name in ["up", "down"]:
            fn_vals_all = []
            fn_true_vals_all = []
            for i in range(theta_pred.shape[0]):
                simulator.init(theta_pred[i])
                fn = getattr(simulator, fn_name)
                vals_pred = fn(x_vals).unsqueeze(0)
                # Replace NaN with 1000, +inf with 1000, -inf with -1000
                vals_pred = torch.nan_to_num(
                    vals_pred, nan=10000.0, posinf=10000.0, neginf=-10000.0
                )
                fn_vals_all.append(vals_pred)

                simulator.init(theta_true[i])
                fn_true = getattr(simulator, fn_name)
                vals_true = fn_true(x_vals).unsqueeze(0)
                vals_true = torch.nan_to_num(
                    vals_true, nan=10000.0, posinf=10000.0, neginf=-10000.0
                )
                fn_true_vals_all.append(vals_true)

                # print(f"Predicted Parameters: {theta_pred[i]}")
                # print(f"Mean {fn_name} for pred: {fn_vals_all[-1].mean()}, true: {fn_true_vals_all[-1].mean()}")
            fn_stack = torch.cat(fn_vals_all, dim=0)  # [batch_size, 500]
            fn_true_stack = torch.cat(fn_true_vals_all, dim=0)  # [batch_size, 500]
            pred_vals_all.append(fn_stack)
            true_vals_all.append(fn_true_stack)
        pred_vals = torch.stack(pred_vals_all, dim=1)  # [batch_size, 2, 500]
        true_vals = torch.stack(true_vals_all, dim=1)
        loss = (
            torch.log(torch.clamp(pred_vals, min=eps) / torch.clamp(true_vals, min=eps))
            .abs()
            .mean()
        )

    return loss


EPS = 1e-8


def improved_feature_engineering(xs_tensor):
    """
    Drop-in replacement for your original function.
    Input:
      xs_tensor: (..., F) raw features (can be negative or positive)
    Output:
      data: (..., F_out) engineered features (torch.float32)
    Notes:
      - Preserves sign information for symlog-like features.
      - Uses log1p(abs(x)) for stability and log-difference for ratio-like features.
      - Does NOT fit or apply a scaler; if you want standardization, use improved_feature_engineering below.
    """
    x = xs_tensor.to(torch.float32)
    eps = EPS

    # basic transforms
    log1p_abs = torch.log1p(x.abs().clamp(min=eps))  # log1p of magnitude
    symlog_feat = torch.sign(x) * log1p_abs  # symmetric log-like

    # positive-only log1p (keeps zeros for negatives)
    log1p_pos = torch.log1p(torch.clamp(x, min=0.0) + eps)

    # pairwise indices
    F = x.shape[-1]
    if F >= 2:
        idxs = torch.combinations(torch.arange(F), r=2, with_replacement=False)
        i, j = idxs[:, 0], idxs[:, 1]
        # log-difference proxy for ratio: log1p(|x_i|) - log1p(|x_j|)
        pair_logdiff = log1p_abs[..., i] - log1p_abs[..., j]
        # signed raw difference as additional signal
        pair_diff = x[..., i] - x[..., j]
        pair_feats = torch.cat([pair_logdiff, pair_diff], dim=-1)
    else:
        pair_feats = x.new_empty(x.shape[:-1] + (0,))

    # assemble: raw_clipped, symlog, log1p_pos, pairwise
    raw_clipped = x.clamp(min=-1e6, max=1e6)
    data = torch.cat([raw_clipped, symlog_feat, log1p_pos, pair_feats], dim=-1)
    return data
