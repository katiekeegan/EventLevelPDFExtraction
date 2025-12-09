import numpy as np
import torch
from sbi.inference import MCABC, SNPE, simulate_for_sbi
from sbi.utils.torchutils import BoxUniform
from scipy.stats import wasserstein_distance

np.random.seed(42)
torch.manual_seed(42)

########################
# SIMULATOR DEFINITIONS
########################


@torch.no_grad()
def simulator(theta):
    N = 10000  # Number of events in cross-sections

    def up(x, theta):
        return 1 * (x ** theta[0]) * ((1 - x) ** theta[1])

    def down(x, theta):
        return 2 * (x ** theta[2]) * ((1 - x) ** theta[3])

    xs_p = torch.rand(N)
    sigma_p = 4 * up(xs_p, theta) + down(xs_p, theta)
    sigma_p = torch.nan_to_num(sigma_p, nan=0.0)

    xs_n = torch.rand(N)
    sigma_n = 4 * down(xs_n, theta) + up(xs_n, theta)
    sigma_n = torch.nan_to_num(sigma_n, nan=0.0)

    x = torch.cat([sigma_p.unsqueeze(0), sigma_n.unsqueeze(0)], dim=0).t()
    return x.float()


def simulator_batch(theta_batch):
    return [simulator(theta) for theta in theta_batch]


def simulator_batch_summary(theta_batch):
    """
    Returns [B, summary_dim] where each row is a feature vector (summary)
    """
    return torch.stack([histogram_summary(simulator(theta)) for theta in theta_batch])


def _hist_from_edges(x1d, edges):
    idx = torch.bucketize(x1d, edges, right=False) - 1
    B = edges.numel() - 1
    idx = idx.clamp(min=0, max=B - 1)
    counts = torch.bincount(idx, minlength=B)
    widths = edges[1:] - edges[:-1]
    return counts, widths


def histogram_summary(x, nbins_per_side=16, min_abs=1e-6, max_abs=10.0, density=True):
    """
    Symmetric log-spaced bins for data that may be negative, zero, or positive.
    Bins are geometric in |x| and symmetric about 0.
    """
    x = x.detach()
    D = x.shape[1]
    device, dtype = x.device, x.dtype

    # Use torch.logspace instead of geomspace for compatibility
    pos_edges = torch.logspace(
        torch.log10(torch.tensor(min_abs, device=device, dtype=dtype)),
        torch.log10(torch.tensor(max_abs, device=device, dtype=dtype)),
        steps=nbins_per_side + 1,
        device=device,
        dtype=dtype,
    )
    neg_edges = -torch.logspace(
        torch.log10(torch.tensor(max_abs, device=device, dtype=dtype)),
        torch.log10(torch.tensor(min_abs, device=device, dtype=dtype)),
        steps=nbins_per_side + 1,
        device=device,
        dtype=dtype,
    )
    edges = torch.cat([neg_edges, pos_edges])

    summaries = []
    for d in range(D):
        counts, widths = _hist_from_edges(x[:, d], edges)
        h = counts.to(dtype)
        if density:
            total = counts.sum().clamp_min(1)
            h = h / total / widths
            Z = (h * widths).sum().clamp_min(1e-12)
            h = h / Z
        else:
            h = h / counts.sum().clamp_min(1)
        summaries.append(h)
    return torch.cat(summaries, dim=0)


########################
# BENCHMARKS
########################


def snpe_benchmark(simulator, param_prior, nbins=32):
    def sim_fn(theta_batch):
        return simulator_batch_summary(theta_batch)

    theta, x = simulate_for_sbi(sim_fn, param_prior, num_simulations=10000)
    inference = SNPE(param_prior)
    density_estimator = inference.append_simulations(theta, x).train()
    return inference.build_posterior(density_estimator)


def wasserstein_abc_benchmark(simulator, param_prior):
    inference = MCABC(
        prior=param_prior,
        simulator=simulator_batch,
        distance=wasserstein_distance_wrapper,
    )
    true_theta = torch.tensor([1.0, 2.0, 0.5, 1.0])
    x_o = simulator(true_theta)
    return inference(x_o, num_simulations=10000, quantile=0.01)


########################
# DISTANCE FUNCTIONS
########################


def pairwise_squared_dist(x, y):
    """
    Compute the pairwise squared Euclidean distance between two sets of samples.
    x: [N1, D] tensor of samples
    y: [N2, D] tensor of samples
    returns: [N1, N2] tensor of squared distances
    """
    x2 = x.pow(2).sum(dim=1, keepdim=True)
    y2 = y.pow(2).sum(dim=1, keepdim=True).T
    xy = x @ y.T
    return x2 - 2 * xy + y2


def l2_distance_vectorized(x1, x_batch):
    """
    Compute vectorized L2 distances between a single summary vector and a batch of them.

    Args:
        x1: [D] tensor (summary vector for observed data)
        x_batch: [B, D] tensor (batch of summary vectors)

    Returns:
        [B] tensor of distances
    """
    if x1.ndim == 1:
        x1 = x1.unsqueeze(0)  # [1, D]
    return torch.norm(x_batch - x1, dim=1)


def sliced_wasserstein(x1, x2, num_projections=50):
    """
    Compute the sliced Wasserstein distance between two sets of samples.
    x1: [N1, D] tensor of samples
    x2: [N2, D] tensor of samples
    num_projections: number of random projections to use
    returns: scalar distance
    """
    if x1.ndim == 1:
        x1 = x1.unsqueeze(0)  # [1, D]
    if x2.ndim == 1:
        x2 = x2.unsqueeze(0)  # [1, D]

    d = x1.shape[1]
    distances = []
    for _ in range(num_projections):
        proj = torch.randn(d, device=x1.device)  # random projection vector
        proj = proj / torch.norm(proj)  # normalize projection vector

        x1_proj = (x1 @ proj).view(-1).cpu().numpy()  # always 1D
        x2_proj = (x2 @ proj).view(-1).cpu().numpy()

        if len(x1_proj) == 0 or len(x2_proj) == 0:
            distances.append(np.nan)  # handle empty projections
        else:
            w = wasserstein_distance(x1_proj, x2_proj)
            distances.append(w)

    return np.nanmean(distances)  # return mean distance across projections


def wasserstein_distance_vectorized(x1, x_batch):
    """
    x1: [N1, D] sample from the observation
    x_batch: [B, N2, D] batch of simulated samples
    returns: torch.tensor of shape [B] with scalar distances
    """
    distances = []
    for x2 in x_batch:
        dist = sliced_wasserstein(x1, x2)  # returns scalar
        distances.append(dist)
    return torch.tensor(distances)


def wasserstein_distance_wrapper(x1, x2):
    return sliced_wasserstein(x1, x2)


if __name__ == "__main__":

    prior_dist = BoxUniform(
        low=torch.zeros(4), high=5 * torch.ones(4)
    )  # 4-dimensional uniform prior

    # True observation cross-section
    true_theta = torch.tensor(
        [1.0, 1.2, 1.1, 0.5]
    )  # example true parameters - arbitrary
    x_o = simulator(true_theta)  # [N, D] tensor of observed data
    x_o_summary = histogram_summary(x_o)  # [D * nbins] summary vector
    samples = simulator_batch_summary(
        prior_dist.sample((100,))
    )  # [B, D * nbins] batch of simulated summaries
    dists = [
        l2_distance_vectorized(x_o_summary, s) for s in samples
    ]  # compute distances to each sample

    # Sanity Check
    print("Checking to make sure distances are computed correctly...")
    print("Example distances:", dists[:10])
    print("Mean distance:", torch.tensor(dists).mean())
    print("Example distances:", dists[:10])
    print("Mean distance:", torch.tensor(dists).mean())

    ### 1. MCABC (L2)

    print("Running MCABC (L2 Distance)...")
    mcabc = MCABC(
        prior=prior_dist,
        simulator=simulator_batch_summary,
        distance=l2_distance_vectorized,  # L2 already passed here
    )
    samples_mmd = mcabc(x_o_summary, num_simulations=10000, quantile=0.01)
    np.savetxt("samples_mmd.txt", samples_mmd.cpu().numpy())
    print("MMD Posterior median:", samples_mmd.median(0))

    ### 2. SNPE
    print("Running SNPE (histograms)...")
    posterior_snpe = snpe_benchmark(simulator, prior_dist)
    x_o_hist = histogram_summary(x_o)
    samples_snpe = posterior_snpe.sample((100,), x=x_o_hist)
    print("SNPE Posterior mean:", samples_snpe.mean(0))
    np.savetxt("samples_snpe.txt", samples_snpe.cpu().numpy())

    ### 3. Wasserstein ABC
    print("Running MCABC (Wasserstein Distance)...")
    samples_wass = MCABC(
        prior=prior_dist,
        simulator=simulator_batch_summary,
        distance=wasserstein_distance_vectorized,
    )(x_o_summary, num_simulations=10000, quantile=0.01)
    print("Wasserstein Posterior mean:", samples_wass.mean(0))
    np.savetxt("samples_wasserstein.txt", samples_wass.cpu().numpy())
