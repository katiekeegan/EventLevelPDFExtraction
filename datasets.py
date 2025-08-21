import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import *
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
from torch.nn.parallel import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.cuda.amp as amp
import warnings
import os
import h5py
import numpy as np
from tqdm import tqdm
from simulator import SimplifiedDIS, RealisticDIS, up, down, advanced_feature_engineering, MCEGSimulator, Gaussian2DSimulator
from utils import log_feature_engineering

from scipy.stats import beta

def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _coerce_sample(arr, num_events, x_dim=None):
    """
    Take arbitrary-shaped sample `arr` and coerce to (num_events, F).
    - Detect the 'event' axis (prefer exact match with num_events, else closest).
    - Move event axis to front.
    - Flatten remaining axes as features.
    - Truncate/pad events to num_events.
    - If x_dim is given, truncate/pad features to exactly x_dim.
    """
    a = _as_numpy(arr)
    a = np.squeeze(a)  # drop size-1 dims
    if a.size == 0:
        # degenerate case: just return zeros
        F = x_dim if x_dim is not None else 1
        return np.zeros((num_events, F), dtype=np.float32)

    # If 1D -> interpret as (N, 1)
    if a.ndim == 1:
        a = a.reshape(-1, 1)

    # Pick event axis: exact match to num_events if possible, else nearest size
    if any(s == num_events for s in a.shape):
        event_axis = next(i for i, s in enumerate(a.shape) if s == num_events)
    else:
        # Choose axis whose size is closest to num_events
        sizes = list(a.shape)
        diffs = [abs(s - num_events) for s in sizes]
        event_axis = int(np.argmin(diffs))

    # Move event axis to axis 0
    if event_axis != 0:
        a = np.moveaxis(a, event_axis, 0)

    # Now a is (N_events_like, ...features...)
    N = a.shape[0]
    feat = int(np.prod(a.shape[1:])) if a.ndim > 1 else 1
    a = a.reshape(N, feat)

    # Truncate/pad events
    if N >= num_events:
        a = a[:num_events, :]
    else:
        pad_events = np.zeros((num_events - N, feat), dtype=a.dtype)
        a = np.concatenate([a, pad_events], axis=0)

    # Optionally force feature size
    if x_dim is not None:
        if a.shape[1] >= x_dim:
            a = a[:, :x_dim]
        else:
            pad_feat = np.zeros((num_events, x_dim - a.shape[1]), dtype=a.dtype)
            a = np.concatenate([a, pad_feat], axis=1)

    return a.astype(np.float32)


def sample_skewed_uniform(low, high, size, alpha=0.5, beta_param=1.0):
    # Sample from Beta(α, β), which is defined on [0,1]
    raw = beta.rvs(alpha, beta_param, size=size)
    return low + (high - low) * raw

def generate_gaussian2d_dataset(n_samples, n_events, device=None):
    """
    Generate a dataset of n_samples parameter vectors and n_events events each.
    Parameter vector: [mu_x, mu_y, sigma_x, sigma_y, rho]
    """
    device = device or torch.device("cpu")
    # Example: fixed ranges for parameters, can be changed as needed
    mus = torch.empty((n_samples, 2)).uniform_(-2, 2)
    sigmas = torch.empty((n_samples, 2)).uniform_(0.5, 2.0)
    rhos = torch.empty((n_samples, 1)).uniform_(-0.8, 0.8)
    thetas = torch.cat([mus, sigmas, rhos], dim=1).to(device)

    simulator = Gaussian2DSimulator(device=device)
    xs = []
    for i in range(n_samples):
        x = simulator.sample(thetas[i], nevents=n_events)
        xs.append(x)
    xs = torch.stack(xs, dim=0)  # (n_samples, n_events, 2)
    return thetas, xs

class Gaussian2DDataset(IterableDataset):
    def __init__(
        self,
        simulator,
        num_samples,
        num_events,
        rank,
        world_size,
        theta_dim=5,        # mu_x, mu_y, sigma_x, sigma_y, rho
        theta_bounds=None,
        n_repeat=2,
        feature_engineering=None,
    ):
        self.simulator = simulator
        self.num_samples = num_samples // world_size
        self.num_events = num_events
        self.rank = rank
        self.world_size = world_size
        self.theta_dim = theta_dim
        self.n_repeat = n_repeat

        if theta_bounds is None:
            # [mu_x, mu_y, sigma_x, sigma_y, rho]
            self.theta_bounds = torch.tensor([
                [-2.0, 2.0],   # mu_x
                [-2.0, 2.0],   # mu_y
                [0.5, 2.0],    # sigma_x
                [0.5, 2.0],    # sigma_y
                [-0.8, 0.8],   # rho
            ])
        else:
            self.theta_bounds = torch.tensor(theta_bounds)

        self.feature_engineering = feature_engineering

    def __iter__(self):
        device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self.simulator.device = device
        theta_bounds = self.theta_bounds.to(device)

        for _ in range(self.num_samples):
            theta = torch.rand(self.theta_dim, device=device)
            theta = theta * (theta_bounds[:, 1] - theta_bounds[:, 0]) + theta_bounds[:, 0]

            xs = []
            for _ in range(self.n_repeat):
                x = self.simulator.sample(theta, self.num_events)  # shape: [num_events, 2]
                if self.feature_engineering is not None:
                    x = self.feature_engineering(x)
                xs.append(x.cpu())
            yield theta.cpu(), torch.stack(xs)  # shape: [n_repeat, num_events, 2]

class H5Dataset(Dataset):
    def __init__(self, latent_path):
        self.latent_file = h5py.File(latent_path, 'r')
        self.latents = self.latent_file['latents']
        self.thetas = self.latent_file['thetas']
        self.indices = np.arange(len(self.latents), dtype=np.int64)  # correct indexing

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # PyTorch sometimes gives batched index arrays → ensure scalar access
        if isinstance(idx, (list, tuple, np.ndarray)):
            idx = idx[0]  # Only support 1-sample fetches (because we're using collate_fn)
        real_idx = self.indices[int(idx)]
        latent = torch.from_numpy(self.latents[real_idx])
        theta = torch.from_numpy(self.thetas[real_idx])
        return latent, theta

    def __del__(self):
        try:
            self.latent_file.close()
        except Exception:
            pass

class EventDataset(Dataset):
    def __init__(self, event_data, param_data, latent_fn):
        self.event_data = event_data
        self.param_data = param_data
        self.latent_fn = latent_fn

    def __len__(self):
        return len(self.param_data)

    def __getitem__(self, idx):
        latent = self.latent_fn(self.event_data[idx].unsqueeze(0))  # compute on-the-fly in main process
        return latent, self.param_data[idx]

    @staticmethod
    def collate_fn(batch):
        latents, params = zip(*batch)
        return torch.stack(latents), torch.stack(params)

# Data Generation with improved stability
# def generate_data(num_samples, num_events, problem, theta_dim=4, x_dim=2, device=torch.device("cpu")):
#     if problem == 'simplified_dis':
#         simulator = SimplifiedDIS(device)
#         theta_dim = 4 # [au, bu, ad, bd]
#         ranges = [(0.0, 5), (0.0, 5), (0.0, 5), (0.0, 5)]  # Example ranges
#     elif problem == 'realistic_dis':
#         simulator = RealisticDIS(device)
#         theta_dim = 6
#         ranges = [
#                 (-2.0, 2.0),   # logA0
#                 (-1.0, 1.0),   # delta
#                 (0.0, 5.0),    # a
#                 (0.0, 10.0),   # b
#                 (-5.0, 5.0),   # c
#                 (-5.0, 5.0),   # d
#         ]
#     elif problem == 'mceg':
#         simulator = MCEGSimulator(device)
#         theta_dim = 4    
#         ranges = [
#                 (-1.0, 10.0),
#                 (0.0, 10.0),
#                 (-10.0, 10.0),
#                 (-10.0, 10.0),
#             ]
#     # thetas = np.column_stack([np.random.uniform(low, high, size=num_samples) for low, high in ranges])
#     thetas = np.column_stack([
#     sample_skewed_uniform(low, high, num_samples, alpha=1.0, beta_param=2.0)
#     for low, high in ranges
#     ])
#     if problem == 'mceg':
#         xs = np.array([simulator.sample(theta, num_events+1000)[:, :num_events, ...].cpu().numpy() for theta in thetas]) + 1e-8
#         # xs = x # Ensure we only take the first num_events events
#     else:
#         xs = np.array([simulator.sample(theta, num_events).cpu().numpy() for theta in thetas]) + 1e-8
#     thetas_tensor = torch.tensor(thetas, dtype=torch.float32, device=device)
#     xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
#     return thetas_tensor, xs_tensor

def generate_data(num_samples, num_events, problem, theta_dim=4, x_dim=None, device=torch.device("cpu")):
    # 1) Simulator + numeric ranges (use tuples of floats)
    if problem == 'simplified_dis':
        simulator = SimplifiedDIS(device)
        theta_dim = 4  # [au, bu, ad, bd]
        ranges = [(0.0, 5.0), (0.0, 5.0), (0.0, 5.0), (0.0, 5.0)]
    elif problem == 'realistic_dis':
        simulator = RealisticDIS(device)
        theta_dim = 6
        ranges = [
            (-2.0, 2.0),   # logA0
            (-1.0, 1.0),   # delta
            (0.0, 5.0),    # a
            (0.0, 10.0),   # b
            (-5.0, 5.0),   # c
            (-5.0, 5.0),   # d
        ]
    elif problem == 'mceg':
        simulator = MCEGSimulator(device)
        theta_dim = 4
        ranges = [(-1.0, 10.0), (0.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    else:
        raise ValueError(f"Unknown problem: {problem}")

    # 2) Sample thetas (floats)
    thetas = np.column_stack([
        sample_skewed_uniform(float(low), float(high), num_samples, alpha=1.0, beta_param=2.0)
        for (low, high) in ranges
    ]).astype(np.float32)

    # 3) Collect samples with full flexibility
    processed = []
    max_feat = 0

    for i in range(num_samples):
        theta_np = thetas[i]
        theta_t = torch.tensor(theta_np, dtype=torch.float32, device=device)

        # Try sampling; if some sims underproduce events, that's fine—_coerce_sample will pad.
        # If a specific simulator needs oversampling, do it here (but don't assume event axis).
        sample_t = simulator.sample(theta_t, int(num_events))
        s = _coerce_sample(sample_t, num_events, x_dim=x_dim)
        processed.append(s)
        if s.shape[1] > max_feat:
            max_feat = s.shape[1]

    # 4) Pad features to the max seen (if x_dim not fixed)
    if x_dim is None:
        x_dim_eff = max_feat
    else:
        x_dim_eff = x_dim

    xs_np = np.zeros((num_samples, num_events, x_dim_eff), dtype=np.float32)
    for i, s in enumerate(processed):
        f = min(s.shape[1], x_dim_eff)
        xs_np[i, :, :f] = s[:, :f]

    # 5) Convert to tensors (+ tiny epsilon if you like)
    thetas_tensor = torch.tensor(thetas, dtype=torch.float32, device=device)
    xs_tensor = torch.tensor(xs_np + 1e-8, dtype=torch.float32, device=device)
    return thetas_tensor, xs_tensor

class DISDataset(IterableDataset):
    def __init__(self, simulator, num_samples, num_events, rank, world_size, theta_dim=4, n_repeat=2):
        self.simulator = simulator
        self.total_samples = num_samples
        self.samples_per_rank = num_samples // world_size
        self.num_events = num_events
        self.rank = rank
        self.world_size = world_size
        self.theta_bounds = torch.tensor([[0.0, 5]] * theta_dim)
        self.n_repeat = n_repeat
        self.theta_dim = theta_dim
        self.feature_engineering = advanced_feature_engineering

    def __len__(self):
        return self.samples_per_rank

    def __iter__(self):
        # Pin device and RNG seed
        device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(device)
        self.simulator.device = device

        # Optional: make RNG deterministic per worker
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        seed = self.rank * 10000 + worker_id
        torch.manual_seed(seed)
        np.random.seed(seed)

        theta_bounds = self.theta_bounds.to(device)
        n_feat = None

        for _ in range(self.samples_per_rank):
            tries = 0
            while tries < 20:
                theta = torch.rand(self.theta_dim, device=device)
                theta = theta * (theta_bounds[:, 1] - theta_bounds[:, 0]) + theta_bounds[:, 0]

                xs = []
                bad_sample = False
                for _ in range(self.n_repeat):
                    x = self.simulator.sample(theta, self.num_events + 1000)
                    if bool(getattr(self.simulator, "clip_alert", False)):
                        bad_sample = self.simulator.clip_alert
                        break   
                    x = x[:self.num_events, ...]
                    xs.append(self.feature_engineering(x).cpu())

                if not bad_sample:
                    yield theta.cpu().contiguous(), torch.stack(xs).cpu().contiguous()
                    break  # accepted
                else:
                    tries += 1

class RealisticDISDataset(IterableDataset):
    def __init__(
        self,
        simulator,
        num_samples,
        num_events,
        rank,
        world_size,
        theta_dim=6,
        theta_bounds=None,
        n_repeat=2,
        feature_engineering=None,
    ):
        self.simulator = simulator
        self.total_samples = num_samples
        self.samples_per_rank = num_samples // world_size
        self.num_events = num_events
        self.rank = rank
        self.world_size = world_size
        self.theta_dim = theta_dim
        self.n_repeat = n_repeat

        if theta_bounds is None:
            self.theta_bounds = torch.tensor([
                [-2.0, 2.0],
                [-1.0, 1.0],
                [0.0, 5.0],
                [0.0, 10.0],
                [-5.0, 5.0],
                [-5.0, 5.0],
            ])
        else:
            self.theta_bounds = torch.tensor(theta_bounds)

        self.feature_engineering = feature_engineering

    def __iter__(self):
        device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        self.simulator.device = device

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

            yield theta.cpu(), torch.stack(xs).cpu()

class MCEGDISDataset(IterableDataset):
    def __init__(
        self,
        simulator,
        num_samples,
        num_events,
        rank,
        world_size,
        theta_dim=4,
        n_repeat=2,
        feature_engineering=None,
    ):
        self.simulator = simulator
        self.total_samples = num_samples
        self.samples_per_rank = num_samples // world_size
        self.num_events = num_events
        self.rank = rank
        self.world_size = world_size
        self.theta_dim = theta_dim
        self.n_repeat = n_repeat

        self.theta_bounds = torch.tensor([
                [-1.0, 10.0],
                [0.0, 10.0],
                [-10.0, 10.0],
                [-10.0, 10.0],
            ])

    def feature_engineering(self, x):
        return torch.log(x + 1e-8)  # Avoid log(0) by adding a small constant

    def __iter__(self):
        device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        self.simulator.device = device

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
                x = self.simulator.sample(theta, self.num_events + 1000)
                x = x[:self.num_events, ...]
                fe_x = self.feature_engineering(x)
                # Ensure tensor shape and type
                if not isinstance(fe_x, torch.Tensor):
                    fe_x = torch.tensor(fe_x)
                fe_x = fe_x.cpu().contiguous().clone()
                xs.append(fe_x)
            stacked_xs = torch.stack(xs).cpu().contiguous().clone()

            print(f"theta shape: {theta.shape}, xs shape: {stacked_xs.shape}")
            yield theta.cpu().contiguous().clone(), stacked_xs
