import os

import numpy as np
import torch
from sbi.inference import MCABC, SNPE, simulate_for_sbi
from sbi.utils.torchutils import BoxUniform
from scipy.stats import wasserstein_distance

# Try to import MCEG simulator
try:
    from simulator import MCEGSimulator

    HAS_MCEG = True
except Exception:
    HAS_MCEG = False

########################
# SIMULATOR DEFINITIONS (MCEG4DIS)
########################


@torch.no_grad()
def simulator(theta, n_events=10000):
    """Simulate events from MCEG for a single parameter vector theta.
    Returns a torch.tensor of shape [n_events, 2] with columns (x, Q2).
    """
    if not HAS_MCEG:
        raise RuntimeError(
            "MCEGSimulator not available (missing optional dependencies)"
        )
    sim = MCEGSimulator(device=torch.device("cpu"))
    # The MCEGSimulator.sample expects params and n_events
    evts = sim.sample(theta, n_events)
    # evts should be a tensor-like [n_events, 2] (x, Q2) or similar
    return evts.float()


def simulator_batch(theta_batch, n_events=10000):
    return [simulator(theta, n_events=n_events) for theta in theta_batch]


def histogram_summary(
    evts, nx=30, nQ2=20, x_min=1e-4, x_max=1e-1, Q2_min=10.0, Q2_max=1000.0
):
    """Compute a flattened, normalized 2D histogram summary over log(x) and log(Q2).

    Parameters
    ----------
    evts : torch.Tensor or np.ndarray
        Shape [N, 2] with columns (x, Q2) or possibly more columns; only first two used.
    nx, nQ2 : int
        Number of bins in x and Q2 (in log space)
    Returns
    -------
    np.ndarray
        Flattened histogram of shape (nx * nQ2,) normalized to sum to 1.
    """
    if hasattr(evts, "detach"):
        evts = evts.detach().cpu().numpy()
    if evts.size == 0:
        return np.zeros(nx * nQ2, dtype=float)
    x = evts[:, 0]
    Q2 = evts[:, 1]
    # Clip to sensible ranges to avoid infinities
    x = np.clip(x, x_min, x_max)
    Q2 = np.clip(Q2, Q2_min, Q2_max)
    log_x = np.log(x)
    log_Q2 = np.log(Q2)
    x_edges = np.linspace(np.log(x_min), np.log(x_max), nx + 1)
    Q2_edges = np.linspace(np.log(Q2_min), np.log(Q2_max), nQ2 + 1)
    H, xe, qe = np.histogram2d(log_x, log_Q2, bins=[x_edges, Q2_edges])
    # Normalize to density (sum to 1)
    total = H.sum()
    if total <= 0:
        return np.zeros(nx * nQ2, dtype=float)
    H = H / total
    return H.ravel()


def simulator_batch_summary(theta_batch, n_events=10000, nx=30, nQ2=20):
    """Return [B, nx*nQ2] summary for a batch of theta vectors."""
    return torch.stack(
        [
            torch.from_numpy(
                histogram_summary(simulator(theta, n_events=n_events), nx=nx, nQ2=nQ2)
            )
            for theta in theta_batch
        ]
    ).float()


########################
# BENCHMARKS
########################


def snpe_benchmark(simulator_summary_fn, param_prior, num_simulations=10000):
    """Run SNPE using summary simulator function (maps theta_batch->summary).
    Returns a posterior object with .sample()."""

    def sim_fn(theta_batch):
        return simulator_summary_fn(theta_batch)

    theta, x = simulate_for_sbi(sim_fn, param_prior, num_simulations=num_simulations)
    inference = SNPE(param_prior)
    density_estimator = inference.append_simulations(theta, x).train()
    return inference.build_posterior(density_estimator)


def wasserstein_abc_benchmark(simulator_summary_fn, param_prior, num_simulations=10000):
    inference = MCABC(
        prior=param_prior,
        simulator=simulator_summary_fn,
        distance=wasserstein_distance_wrapper,
    )
    # pick an example true theta (arbitrary)
    true_theta = torch.tensor([1.0, 1.0, 1.0, 1.0])
    x_o = histogram_summary(simulator(true_theta))
    return inference(x_o, num_simulations=num_simulations, quantile=0.01)


########################
# DISTANCE FUNCTIONS
########################


def sliced_wasserstein(x1, x2, num_projections=50):
    """Compute sliced Wasserstein between 1D samples or summary vectors (as 1D arrays).
    x1, x2: 1D arrays
    """
    if x1.ndim == 1:
        x1 = x1.reshape(1, -1)
    if x2.ndim == 1:
        x2 = x2.reshape(1, -1)
    d = x1.shape[1]
    distances = []
    for _ in range(num_projections):
        proj = np.random.randn(d)
        proj = proj / np.linalg.norm(proj)
        x1_proj = (x1 @ proj).ravel()
        x2_proj = (x2 @ proj).ravel()
        if len(x1_proj) == 0 or len(x2_proj) == 0:
            distances.append(np.nan)
        else:
            distances.append(wasserstein_distance(x1_proj, x2_proj))
    return np.nanmean(distances)


def wasserstein_distance_vectorized(x1, x_batch):
    distances = []
    for x2 in x_batch:
        distances.append(sliced_wasserstein(x1, x2))
    distances = np.array(distances)
    return torch.tensor(distances)


def wasserstein_distance_wrapper(x1, x2):
    return sliced_wasserstein(x1, x2)


if __name__ == "__main__":
    if not HAS_MCEG:
        print("MCEG dependencies not available. Cannot run MCEG4DIS benchmark.")
        raise SystemExit(1)

    # Prior: 4-dimensional box like original benchmark
    prior_dist = BoxUniform(low=torch.zeros(4), high=5 * torch.ones(4))

    # True observation
    true_theta = torch.tensor([1.0, 1.2, 1.1, 0.5])
    x_o = simulator(true_theta, n_events=10000)
    x_o_summary = histogram_summary(x_o)

    # Sanity check: simulate some prior samples and compute distances
    sample_thetas = prior_dist.sample((100,))
    summaries = simulator_batch_summary(sample_thetas, n_events=10000)
    dists = [np.linalg.norm(x_o_summary - s.numpy()) for s in summaries]
    print("Example distances (L2):", dists[:10])
    print("Mean distance:", np.mean(dists))

    ### 1. MCABC (L2)
    print("Running MCABC (L2 Distance)...")
    from functools import partial

    def l2_distance_vectorized_np(x1, x_batch):
        # x1: 1D numpy array, x_batch: [B, D] numpy or torch
        if hasattr(x_batch, "numpy"):
            x_batch = x_batch.numpy()
        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.numpy()
        if isinstance(x_batch, list):
            x_batch = np.array(x_batch)
        if isinstance(x1, torch.Tensor):
            x1 = x1.numpy()
        dists = np.linalg.norm(x_batch - x1.reshape(1, -1), axis=1)
        dists = torch.tensor(dists)
        return dists

    # Wrap simulator_summary for MCABC api: it expects simulator(theta_batch) -> a batch of summaries
    def mcabc_simulator_wrapper(theta_batch):
        # theta_batch may be a torch.Tensor with shape [B, D] or an iterable of tensors
        summaries = []
        if isinstance(theta_batch, torch.Tensor):
            for theta in theta_batch:
                summaries.append(
                    torch.from_numpy(
                        histogram_summary(simulator(theta, n_events=10000))
                    )
                )
            return torch.stack(summaries)
        else:
            for theta in theta_batch:
                summaries.append(
                    torch.from_numpy(
                        histogram_summary(simulator(theta, n_events=10000))
                    )
                )
            return torch.stack(summaries)

    mcabc = MCABC(
        prior=prior_dist,
        simulator=mcabc_simulator_wrapper,
        distance=l2_distance_vectorized_np,
    )
    samples_mmd = mcabc(x_o_summary, num_simulations=10000, quantile=0.01)
    np.savetxt("samples_mmd_mceg.txt", samples_mmd.cpu().numpy())
    print("MMD Posterior median:", samples_mmd.median(0))

    ### 2. SNPE
    print("Running SNPE (2D log-histogram summaries)...")
    posterior_snpe = snpe_benchmark(
        simulator_batch_summary, prior_dist, num_simulations=10000
    )
    samples_snpe = posterior_snpe.sample((100,), x=x_o_summary)
    print("SNPE Posterior mean:", samples_snpe.mean(0))
    np.savetxt("samples_snpe_mceg.txt", samples_snpe.cpu().numpy())

    ### 3. Wasserstein ABC
    print("Running MCABC (Wasserstein Distance)...")
    mcabc_wass = MCABC(
        prior=prior_dist,
        simulator=mcabc_simulator_wrapper,
        distance=wasserstein_distance_vectorized,
    )
    samples_wass = mcabc_wass(x_o_summary, num_simulations=10000, quantile=0.01)
    print("Wasserstein Posterior mean:", samples_wass.mean(0))
    np.savetxt("samples_wasserstein_mceg.txt", samples_wass.cpu().numpy())
