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
np.random.seed(42)
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

def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

EPS = 1e-8

def improved_feature_engineering(xs_tensor):
    """
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
