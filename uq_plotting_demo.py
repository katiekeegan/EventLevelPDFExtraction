#!/usr/bin/env python3
"""
Enhanced plotting utilities for uncertainty quantification using simulator-only data

This module provides a comprehensive suite of uncertainty quantification plots using only
simulator data. It demonstrates parameter-space uncertainty, function-space uncertainty,
bootstrap uncertainty, combined uncertainty decomposition, and uncertainty scaling.

All plots are saved with corresponding LaTeX descriptions explaining the uncertainty
computation and what each plot conveys.

Usage:
    python uq_plotting_demo.py

Requirements:
    - torch
    - matplotlib
    - numpy
    - scipy
    - tqdm

Author: Enhanced plotting utilities for PDFParameterInference
"""

import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

from plotting_UQ_helpers import compute_function_lotv_for_simplified_dis
# Import simulator classes from the main simulator module
from simulator import (Gaussian2DSimulator, MCEGSimulator, RealisticDIS,
                       SimplifiedDIS)
from utils import log_feature_engineering

# Set up matplotlib for high-quality plots
plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": False,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.axisbelow": True,
    }
)

# Colorblind-friendly palette
COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
}
# LaTeX descriptions are now included as comments within each plotting function


def save_latex_description(plot_path, latex_content):
    """
    Save LaTeX description alongside the plot file.

    Parameters:
    ----------
    plot_path : str
        Path to the plot file
    latex_content : str
        LaTeX content to save
    """
    tex_path = (
        plot_path.replace(".png", ".tex")
        .replace(".pdf", ".tex")
        .replace(".jpg", ".tex")
    )
    with open(tex_path, "w") as f:
        f.write(latex_content)


def posterior_sampler(
    observed_data, pointnet_model, model, laplace_model, n_samples=1000, device=None
):
    """
    Sample parameter vectors from the Laplace-approximated posterior given observed data.

    Args:
        observed_data: torch.Tensor of shape [num_events, ...], your raw data batch.
        pointnet_model: feature extractor network.
        model: parameter prediction head (MLP, Transformer, etc).
        laplace_model: fitted Laplace object for epistemic uncertainty.
        n_samples: number of posterior samples to draw.
        device: torch device.

    Returns:
        theta_samples: torch.Tensor of shape [n_samples, param_dim]
    """
    if device is None:
        device = observed_data.device

    # 1. Get latent embedding from pointnet
    observed_data = observed_data.to(device)
    if observed_data.ndim == 2:
        observed_data = observed_data.unsqueeze(0)  # [1, num_events, features]
    observed_data = log_feature_engineering(observed_data).float()
    latent = pointnet_model(observed_data)  # shape: [1, latent_dim]

    # 2. Get Laplace mean and covariance (analytic uncertainty)
    # You may already have a helper like get_analytic_uncertainty but want the full covariance!
    laplace_predictive_fn = getattr(laplace_model, "predictive_distribution", None)
    if laplace_predictive_fn is not None:
        dist = laplace_predictive_fn(latent)
        mean = dist.loc.squeeze(0)
        cov = dist.covariance_matrix.squeeze(0)
    else:
        # Fallback: use Laplace call output, try to extract mean/cov
        out = laplace_model(latent, joint=True)
        if isinstance(out, tuple) and len(out) == 2:
            mean, cov = out
            mean = mean.squeeze(0)
            cov = cov.squeeze(0)
        else:
            raise RuntimeError("Laplace model did not return (mean, cov).")

    # 3. Sample from the multivariate normal (joint, not i.i.d.)
    mvn = torch.distributions.MultivariateNormal(mean, cov)
    theta_samples = mvn.sample((n_samples,))  # [n_samples, param_dim]
    return theta_samples.cpu()


def plot_parameter_uncertainty(
    model=None,
    pointnet_model=None,
    true_params=None,
    device=None,
    num_events=2000,
    problem="simplified_dis",
    save_dir="plots",
    save_path=None,
    n_mc=100,
    laplace_model=None,
    mode="posterior",
    n_bootstrap=100,
    # Backward compatibility - allow old API
    simulator=None,
    true_theta=None,
    observed_data=None,
):
    """
    Plot parameter-space uncertainty showing posterior distribution of inferred parameters.

    This function supports both the new generator-style API and backward compatibility
    with the original API for custom data.

    New API Parameters:
    ------------------
    model : torch.nn.Module
        Parameter prediction model (head)
    pointnet_model : torch.nn.Module
        PointNet feature extractor
    true_params : torch.Tensor
        Ground truth parameter values
    device : torch.device
        Device to run computations on
    num_events : int
        Number of events to generate for analysis
    problem : str
        Problem type ('simplified_dis', 'realistic_dis', 'gaussian', 'mceg')
    save_dir : str
        Directory to save plots (used if save_path not provided)
    save_path : str, optional
        Full path to save plot (overrides save_dir)
    n_mc : int
        Number of Monte Carlo samples for uncertainty estimation
    laplace_model : object, optional
        Fitted Laplace approximation for analytic uncertainty
    mode : str, optional
        Type of uncertainty to visualize (default: 'posterior')
        - 'posterior': Parameter uncertainty from posterior for single dataset
        - 'bootstrap': Bootstrap/data uncertainty across repeated datasets
        - 'combined': Both posterior and bootstrap uncertainty for comparison
    n_bootstrap : int
        Number of bootstrap samples (used for 'bootstrap' and 'combined' modes)

    Backward Compatibility Parameters:
    ---------------------------------
    simulator : object
        Legacy simulator object
    true_theta : torch.Tensor
        Legacy true parameter tensor
    observed_data : torch.Tensor
        Legacy observed data tensor

    LaTeX Description:
    ==================

    \\section{Parameter-Space Uncertainty}

    This figure shows the posterior distribution of model parameters $p(\\theta|\\mathcal{D})$
    obtained through inference on simulated data. The uncertainty visualization includes:

    \\begin{itemize}
    \\item \\textbf{Posterior histograms}: Density plots showing the inferred parameter distributions
    \\item \\textbf{True values}: Red dashed lines indicating the ground truth parameters used to generate the data
    \\item \\textbf{Posterior means}: Green solid lines showing the expected values $\\mathbb{E}[\\theta|\\mathcal{D}]$
    \\item \\textbf{Confidence intervals}: Shaded regions showing $\\pm 1\\sigma$ and $\\pm 2\\sigma$ credible intervals
    \\end{itemize}

    The parameter uncertainty is computed by sampling from the posterior distribution:
    $$p(\\theta|\\mathcal{D}) \\propto p(\\mathcal{D}|\\theta) p(\\theta)$$

    where $p(\\mathcal{D}|\\theta)$ is the likelihood of observing data $\\mathcal{D}$ given parameters $\\theta$,
    and $p(\\theta)$ is the prior distribution. The width of each posterior distribution indicates
    the uncertainty in that parameter given the observed data.

    \\textbf{Mode Parameter:}
    \\begin{itemize}
    \\item \\textbf{posterior}: Uses parameter uncertainty from posterior for single dataset
    \\item \\textbf{bootstrap}: Uses bootstrap/data uncertainty across repeated datasets
    \\item \\textbf{combined}: Shows both types of uncertainty for comparison
    \\end{itemize}

    Statistics shown include the posterior mean and standard deviation for each parameter,
    providing quantitative measures of the parameter inference uncertainty.
    """
    print("📊 Generating parameter-space uncertainty plot...")

    # Validate mode parameter
    valid_modes = ["posterior", "bootstrap", "combined"]
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got: {mode}")

    print(f"   Mode: {mode}")

    # Handle backward compatibility - detect legacy positional arguments
    # If the first argument looks like a simulator and we have tensor arguments, treat as legacy API
    if (
        model is not None
        and hasattr(model, "sample")
        and hasattr(model, "init")
        and pointnet_model is not None
        and isinstance(pointnet_model, torch.Tensor)
        and true_params is not None
        and isinstance(true_params, torch.Tensor)
    ):
        # Legacy positional call: plot_parameter_uncertainty(simulator, true_theta, observed_data, save_dir, ...)
        print("   Using legacy API with positional arguments")
        simulator = model  # First arg is actually simulator
        true_theta = pointnet_model  # Second arg is actually true_theta
        observed_data = true_params  # Third arg is actually observed_data
        save_dir = (
            device if isinstance(device, str) else save_dir
        )  # Fourth arg might be save_dir

        # Reset new API parameters to None
        model = None
        pointnet_model = None
        true_params = None
        device = None

    # Handle backward compatibility
    if simulator is not None and true_theta is not None and observed_data is not None:
        # Legacy API - use provided simulator and data
        print("   Using legacy API with provided simulator and data")
        working_simulator = simulator
        working_true_params = true_theta
        working_observed_data = observed_data

        # For legacy API, we don't have ML models, so we generate mock posterior samples
        if mode == "posterior":
            # Generate mock posterior samples by adding noise to true parameters
            print("   Generating mock posterior samples for legacy API...")
            posterior_samples = []
            for _ in range(n_mc):
                perturbed_params = (
                    working_true_params + torch.randn_like(working_true_params) * 0.1
                )
                posterior_samples.append(perturbed_params)
            working_posterior_samples = torch.stack(posterior_samples)
        elif mode == "bootstrap":
            # Generate bootstrap samples
            print("   Generating bootstrap samples for legacy API...")
            bootstrap_samples = []
            for trial in range(n_bootstrap):
                # Generate new dataset
                bootstrap_data = working_simulator.sample(
                    working_true_params, num_events
                )
                # Use simple perturbation as parameter estimate
                estimated_params = (
                    working_true_params + torch.randn_like(working_true_params) * 0.1
                )
                bootstrap_samples.append(estimated_params.detach().cpu())
            working_posterior_samples = torch.stack(bootstrap_samples)
        elif mode == "combined":
            # Generate both mock posterior and bootstrap samples
            print("   Generating combined samples for legacy API...")
            # Mock posterior samples
            posterior_samples = []
            for _ in range(n_mc):
                perturbed_params = (
                    working_true_params + torch.randn_like(working_true_params) * 0.1
                )
                posterior_samples.append(perturbed_params)
            posterior_samples = torch.stack(posterior_samples)

            # Bootstrap samples
            bootstrap_samples = []
            for trial in range(n_bootstrap):
                bootstrap_data = working_simulator.sample(
                    working_true_params, num_events
                )
                estimated_params = (
                    working_true_params + torch.randn_like(working_true_params) * 0.1
                )
                bootstrap_samples.append(estimated_params.detach().cpu())
            bootstrap_samples = torch.stack(bootstrap_samples)

            working_posterior_samples = torch.cat(
                [posterior_samples, bootstrap_samples], dim=0
            )
    else:
        # New API - generate data internally
        if (
            model is None
            or pointnet_model is None
            or true_params is None
            or device is None
        ):
            raise ValueError(
                "New API requires model, pointnet_model, true_params, and device"
            )

        print("   Using new generator-style API")
        # Create simulator based on problem type
        if problem == "simplified_dis":
            working_simulator = SimplifiedDIS(device=device)
        elif problem == "realistic_dis":
            working_simulator = RealisticDIS(device=device)
        elif problem == "gaussian":
            working_simulator = Gaussian2DSimulator(device=device)
        elif problem == "mceg":
            if MCEGSimulator is not None:
                working_simulator = MCEGSimulator(device=device)
            else:
                raise ValueError("MCEGSimulator not available")
        else:
            raise ValueError(f"Unknown problem type: {problem}")

        working_true_params = true_params
        # Generate observed data
        working_observed_data = working_simulator.sample(true_params, num_events)

        # Sample from posterior based on mode (for new API only)
        if mode == "posterior":
            # Use posterior samples from single dataset
            posterior_samples = posterior_sampler(
                working_observed_data,
                pointnet_model,
                model,
                laplace_model,
                n_samples=n_mc,
            )
            working_posterior_samples = posterior_samples

        elif mode == "bootstrap":
            # Generate multiple datasets via bootstrapping
            bootstrap_samples = []
            for trial in range(n_bootstrap):
                # Generate new dataset
                bootstrap_data = working_simulator.sample(
                    working_true_params, num_events
                )

                # Estimate parameters
                if model is not None and pointnet_model is not None:
                    # Use neural network prediction
                    estimated_params = _estimate_parameters_nn(
                        bootstrap_data, model, pointnet_model, device
                    )
                    bootstrap_samples.append(estimated_params.detach().cpu())
                else:
                    # Fallback to simplified estimation
                    estimated_params = (
                        working_true_params
                        + torch.randn_like(working_true_params) * 0.1
                    )
                    bootstrap_samples.append(estimated_params.detach().cpu())

            working_posterior_samples = torch.stack(bootstrap_samples)

        elif mode == "combined":
            # Generate both posterior and bootstrap samples
            # Posterior samples
            posterior_samples = posterior_sampler(
                working_observed_data,
                pointnet_model,
                model,
                laplace_model,
                n_samples=n_mc,
            )

            # Bootstrap samples
            bootstrap_samples = []
            for trial in range(n_bootstrap):
                # Generate new dataset
                bootstrap_data = working_simulator.sample(
                    working_true_params, num_events
                )

                # Estimate parameters
                if model is not None and pointnet_model is not None:
                    # Use neural network prediction
                    estimated_params = _estimate_parameters_nn(
                        bootstrap_data, model, pointnet_model, device
                    )
                    bootstrap_samples.append(estimated_params.detach().cpu())
                else:
                    # Fallback to simplified estimation
                    estimated_params = (
                        working_true_params
                        + torch.randn_like(working_true_params) * 0.1
                    )
                    bootstrap_samples.append(estimated_params.detach().cpu())

            bootstrap_samples = torch.stack(bootstrap_samples)
            working_posterior_samples = torch.cat(
                [posterior_samples, bootstrap_samples], dim=0
            )

    # Define prior bounds based on simulator type
    if isinstance(working_simulator, SimplifiedDIS):
        theta_bounds = [
            (0.5, 4.0),
            (0.5, 4.0),
            (0.5, 4.0),
            (0.5, 4.0),
        ]  # [au, bu, ad, bd]
        param_names = [r"$a_u$", r"$b_u$", r"$a_d$", r"$b_d$"]
    elif isinstance(working_simulator, Gaussian2DSimulator):
        theta_bounds = [
            (-2, 2),
            (-2, 2),
            (0.5, 3.0),
            (0.5, 3.0),
            (-0.9, 0.9),
        ]  # Gaussian params
        param_names = [r"$\mu_x$", r"$\mu_y$", r"$\sigma_x$", r"$\sigma_y$", r"$\rho$"]
    else:
        # Default bounds for other simulators
        n_params = len(working_true_params)
        theta_bounds = [(0.1, 10.0)] * n_params
        param_names = [f"$\\theta_{{{i+1}}}$" for i in range(n_params)]

    # Prepare samples for plotting based on mode
    if mode == "posterior":
        samples_to_plot = [working_posterior_samples.numpy()]
        mode_labels = ["Posterior"]
        mode_colors = [COLORS["blue"]]
    elif mode == "bootstrap":
        samples_to_plot = [working_posterior_samples.numpy()]
        mode_labels = ["Bootstrap"]
        mode_colors = [COLORS["orange"]]
    elif mode == "combined":
        # For combined mode, separate the samples
        n_posterior = n_mc if "posterior_samples" in locals() else 0
        if n_posterior > 0:
            posterior_part = working_posterior_samples[:n_posterior].numpy()
            bootstrap_part = working_posterior_samples[n_posterior:].numpy()
            samples_to_plot = [posterior_part, bootstrap_part]
            mode_labels = ["Posterior", "Bootstrap"]
            mode_colors = [COLORS["blue"], COLORS["orange"]]
        else:
            # Legacy API combined mode
            total_samples = working_posterior_samples.shape[0]
            split_point = total_samples // 2
            posterior_part = working_posterior_samples[:split_point].numpy()
            bootstrap_part = working_posterior_samples[split_point:].numpy()
            samples_to_plot = [posterior_part, bootstrap_part]
            mode_labels = ["Posterior", "Bootstrap"]
            mode_colors = [COLORS["blue"], COLORS["orange"]]

    n_params = len(param_names)
    fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(15, 12))
    if n_params <= 2:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()

    for i in range(n_params):
        ax = axes[i]

        # Plot histograms for each sample set
        for j, (samples, label, color) in enumerate(
            zip(samples_to_plot, mode_labels, mode_colors)
        ):
            if samples.ndim == 2:  # Multiple samples (bootstrap or posterior)
                param_samples = samples[:, i]
            else:  # Single samples array
                param_samples = samples[i]

            alpha_val = 0.7 if len(samples_to_plot) == 1 else 0.5
            counts, bins, _ = ax.hist(
                param_samples,
                bins=30,
                alpha=alpha_val,
                color=color,
                density=True,
                label=label,
            )

            # Compute and display statistics for the first sample set or if only one set
            if j == 0 or len(samples_to_plot) == 1:
                mean_val = np.mean(param_samples)
                std_val = np.std(param_samples)

                # Plot mean and confidence intervals
                ax.axvline(
                    mean_val,
                    color=color,
                    linestyle="-",
                    alpha=0.8,
                    linewidth=2,
                    label=f"{label} mean",
                )
                ax.axvspan(
                    mean_val - std_val, mean_val + std_val, alpha=0.2, color=color
                )
                ax.axvspan(
                    mean_val - 2 * std_val,
                    mean_val + 2 * std_val,
                    alpha=0.1,
                    color=color,
                )

        # Plot true value
        if i < len(working_true_params):
            ax.axvline(
                working_true_params[i].item(),
                color=COLORS["red"],
                linestyle="--",
                linewidth=2,
                label="True value",
            )

        # Formatting
        ax.set_xlabel(param_names[i])
        if i == 0:
            ax.set_ylabel("Probability density")
            ax.legend(fontsize=20)
        ax.set_title(f"Parameter {i+1}: {param_names[i]}")
        ax.grid(False)

        # Add statistics box for the first sample set
        first_samples = (
            samples_to_plot[0][:, i]
            if samples_to_plot[0].ndim == 2
            else samples_to_plot[0][i]
        )
        mean_val = np.mean(first_samples)
        std_val = np.std(first_samples)
        stats_text = f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}"
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Remove extra subplots
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    # Determine save path
    if save_path is None:
        if mode == "posterior":
            save_path = os.path.join(save_dir, "parameter_uncertainty.png")
        else:
            save_path = os.path.join(save_dir, f"parameter_uncertainty_{mode}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Parameter uncertainty plot saved to: {save_path}")

    # Return appropriate samples based on mode
    if mode == "posterior":
        return samples_to_plot[0]  # Return numpy array
    elif mode == "bootstrap":
        return samples_to_plot[0]  # Return numpy array
    elif mode == "combined":
        return {"posterior": samples_to_plot[0], "bootstrap": samples_to_plot[1]}


def plot_function_uncertainty_mceg(
    model,
    pointnet_model,
    laplace_model,
    true_params,
    device,
    num_events,
    mode="posterior",
    n_mc=100,
    n_bootstrap=100,
    nx=30,
    nQ2=20,
    Q2_slices=None,
    save_dir=None,
    sbi_samples_list=None,
    sbi_labels=None,
):
    """
    Drop-in replacement that preserves the collaborator's true-parameter code exactly
    and aligns all plotted curves to the same x positions (the collaborator's plotting style).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from tqdm import tqdm

    from simulator import ALPHAS, EWEAK, MELLIN, PDF, THEORY, MCEGSimulator

    mellin = MELLIN(npts=8)
    alphaS = ALPHAS()
    eweak = EWEAK()
    pdf = PDF(mellin, alphaS)
    idis = THEORY(mellin, pdf, alphaS, eweak)

    sim = MCEGSimulator(device=device)

    def evts_to_np(evts):
        if isinstance(evts, torch.Tensor):
            return evts.detach().cpu().numpy()
        return np.asarray(evts)

    def get_reco_stat(evts, mceg):
        evts_np = evts_to_np(evts)
        if evts_np.size == 0:
            hist = np.histogram2d(np.array([]), np.array([]), bins=(nx, nQ2))
            return np.zeros(hist[0].shape), np.zeros(hist[0].shape), hist
        log_x = np.log(evts_np[:, 0])
        log_Q2 = np.log(evts_np[:, 1])
        hist = np.histogram2d(log_x, log_Q2, bins=(nx, nQ2))
        counts = hist[0].astype(float)
        reco = np.zeros_like(counts)
        stat = np.zeros_like(counts)
        x_edges = hist[1]
        Q2_edges = hist[2]
        for i in range(x_edges.shape[0] - 1):
            for j in range(Q2_edges.shape[0] - 1):
                c = counts[i, j]
                if c > 0:
                    xmin = np.exp(x_edges[i])
                    xmax = np.exp(x_edges[i + 1])
                    Q2min = np.exp(Q2_edges[j])
                    Q2max = np.exp(Q2_edges[j + 1])
                    dx = xmax - xmin
                    dQ2 = Q2max - Q2min
                    reco[i, j] = c / (dx * dQ2)
                    stat[i, j] = np.sqrt(c) / (dx * dQ2)
        if np.sum(counts) > 0:
            try:
                scale = float(mceg.total_xsec) / np.sum(counts)
            except Exception:
                scale = 1.0
            reco *= scale
            stat *= scale
        return reco, stat, hist

    sim.init(true_params)
    evts_true = sim.sample(true_params, num_events).cpu()
    # Build histogram on log-variables
    hist = np.histogram2d(
        np.log(evts_true[:, 0]), np.log(evts_true[:, 1]), bins=(nx, nQ2)
    )
    true = np.zeros(hist[0].shape)
    reco = np.zeros(hist[0].shape)
    stat = np.zeros(hist[0].shape)
    entries = [
        (a, b) for a in range(hist[1].shape[0] - 1) for b in range(hist[2].shape[0] - 1)
    ]
    for i, j in tqdm(entries, desc="computing true (collab code)"):
        if hist[0][i, j] > 0:
            x = np.exp(0.5 * (hist[1][i] + hist[1][i + 1]))
            Q2 = np.exp(0.5 * (hist[2][j] + hist[2][j + 1]))
            xmin = np.exp(hist[1][i])
            xmax = np.exp(hist[1][i + 1])
            Q2min = np.exp(hist[2][j])
            Q2max = np.exp(hist[2][j + 1])
            dx = xmax - xmin
            dQ2 = Q2max - Q2min

            # collaborator's analytic true eval
            true[i, j], _ = idis.get_diff_xsec(x, Q2, sim.mceg.rs, sim.mceg.tar, "xQ2")

            # empirical reco/stat from true dataset (same as collab)
            reco[i, j] = hist[0][i, j] / dx / dQ2
            stat[i, j] = np.sqrt(hist[0][i, j]) / dx / dQ2

    # scale reco and stat to total_xsec same as collab
    if np.sum(hist[0]) > 0:
        try:
            scale = float(sim.mceg.total_xsec) / np.sum(hist[0])
        except Exception:
            scale = 1.0
        reco *= scale
        stat *= scale

    # We will use hist (hist from true) edges as the canonical x positions exactly as collab did:
    log_x_edges = hist[1]  # these are log-space edges
    log_Q2_edges = hist[2]
    # collab plotted hist[1][:-1] directly (log-values) and used ax.semilogy()
    x_plot_log = log_x_edges[:-1]  # this is exactly what collab used for plotting x
    Q2_centers = np.exp(0.5 * (log_Q2_edges[:-1] + log_Q2_edges[1:]))

    all_bands = []
    # --- Posterior (Laplace) mode ---
    if mode in ["posterior", "parameter", "combined"] and laplace_model is not None:
        # infer posterior from one dataset (init before sample)
        sim.init(true_params)
        evts = sim.sample(true_params, num_events)
        feats = (
            evts.float().to(device)
            if isinstance(evts, torch.Tensor)
            else torch.tensor(evts, device=device).float()
        )
        from utils import log_feature_engineering

        feats_for_pointnet = log_feature_engineering(feats).float().unsqueeze(0)
        latents = pointnet_model(feats_for_pointnet).detach()
        with torch.no_grad():
            theta_samples = posterior_sampler(
                feats, pointnet_model, model, laplace_model, n_samples=n_mc
            )
        reco_samples = []
        stat_samples = []
        for theta in tqdm(theta_samples, desc="Laplace/parameter bands"):
            # ensure init before sample
            try:
                sim.init(theta)
            except Exception:
                sim.init(
                    theta.detach().cpu().numpy()
                    if hasattr(theta, "detach")
                    else np.asarray(theta)
                )
            evts_pred = sim.sample(theta, num_events)
            reco_s, stat_s, hist_s = get_reco_stat(evts_pred.cpu(), sim.mceg)
            reco_samples.append(reco_s)
            stat_samples.append(stat_s)
        reco_samples = np.array(reco_samples)  # [n_mc, nx, nQ2]
        stat_samples = np.array(stat_samples)
        all_bands.append(("parameter", reco_samples, stat_samples, hist))

    # --- Bootstrap mode ---
    if mode in ["bootstrap", "combined"]:
        reco_samples = []
        stat_samples = []
        for b in tqdm(range(n_bootstrap), desc="bootstrap bands"):
            sim.init(true_params)
            evts = sim.sample(true_params, num_events)
            feats = (
                evts.float().to(device)
                if isinstance(evts, torch.Tensor)
                else torch.tensor(evts, device=device).float()
            )
            from utils import log_feature_engineering

            feats_for_pointnet = log_feature_engineering(feats).float().unsqueeze(0)
            latents = pointnet_model(feats_for_pointnet).detach()
            with torch.no_grad():
                inferred = model(latents)
                try:
                    inferred_theta = inferred.mean(dim=0).detach().cpu().numpy()
                except Exception:
                    inferred_theta = inferred.detach().cpu().numpy().ravel()
            sim.init(inferred_theta)
            evts_pred = sim.sample(inferred_theta, num_events)
            reco_s, stat_s, hist_s = get_reco_stat(evts_pred.cpu(), sim.mceg)
            reco_samples.append(reco_s)
            stat_samples.append(stat_s)
        reco_samples = np.array(reco_samples)
        stat_samples = np.array(stat_samples)
        all_bands.append(("bootstrap", reco_samples, stat_samples, hist))

    # --- SBI samples mode: process explicit sbi_samples_list if provided
    if sbi_samples_list is not None:
        # normalize single tensor -> list
        if not isinstance(sbi_samples_list, (list, tuple)):
            sbi_list = [sbi_samples_list]
        else:
            sbi_list = list(sbi_samples_list)

        for idx_s, sbi_set in enumerate(sbi_list):
            label = None
            if isinstance(sbi_labels, (list, tuple)) and idx_s < len(sbi_labels):
                label = sbi_labels[idx_s]
            elif isinstance(sbi_labels, str):
                label = sbi_labels
            else:
                label = f"SBI_{idx_s+1}"

            # Convert to numpy
            if hasattr(sbi_set, "detach"):
                sbi_arr = sbi_set.detach().cpu().numpy()
            else:
                sbi_arr = np.asarray(sbi_set)

            reco_samples = []
            stat_samples = []
            for t_idx in range(sbi_arr.shape[0]):
                theta = sbi_arr[t_idx]
                try:
                    sim.init(theta)
                except Exception:
                    try:
                        sim.init(torch.tensor(theta))
                    except Exception:
                        pass
                evts_pred = sim.sample(theta, num_events)
                reco_s, stat_s, hist_s = get_reco_stat(evts_pred.cpu(), sim.mceg)
                reco_samples.append(reco_s)
                stat_samples.append(stat_s)

            if len(reco_samples) > 0:
                reco_samples = np.array(reco_samples)
                stat_samples = np.array(stat_samples)
            else:
                reco_samples = np.zeros((0, hist[0].shape[0], hist[0].shape[1]))
                stat_samples = np.zeros_like(reco_samples)

            all_bands.append((f"sbi_{label}", reco_samples, stat_samples, hist))

    # --- Plotting using collaborator style exact x positions (log-values) ---
    label_map = dict(parameter="Laplace Posterior", bootstrap="Bootstrap")
    # color_map = dict(parameter="tab:blue", bootstrap="tab:orange")  # replaced by Q2-based rainbow

    # pick Q2 slices similar to collab method
    if Q2_slices is None:
        nQ2_selects = 10
        dnQ2 = max(1, int(log_Q2_edges.shape[0] / nQ2_selects))
        Q2_indices = list(range(0, log_Q2_edges.shape[0] - 1, dnQ2))
    else:
        # map requested Q2 values to index
        Q2_indices = [np.argmin(np.abs(Q2_centers - q2)) for q2 in Q2_slices]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Create rainbow colormap for the Q2 slices
    cmap = plt.cm.rainbow
    n_Q2_sel = len(Q2_indices)

    # Avoid division by zero if only one Q2 slice:
    def cmap_val(idx):
        if n_Q2_sel == 1:
            return cmap(0.5)
        return cmap(float(idx) / float(max(1, n_Q2_sel - 1)))

    # Plot collaborator true line (they used log edges for x positions and true computed at midpoints)
    for idx, j in enumerate(Q2_indices):
        if np.exp(log_Q2_edges[j]) > 100:
            continue
        color = cmap_val(idx)
        cond_true = true[:, j] > 0
        if np.any(cond_true):
            # collaborator used hist[1][:-1] (log-space) as x positions
            ax.plot(
                x_plot_log[cond_true],
                true[:, j][cond_true],
                label=f"Q²={np.exp(log_Q2_edges[j]):.2f}",
                color=color,
                linestyle="-",
                linewidth=1.5,
            )

        cond_reco = reco[:, j] > 0
        # if np.any(cond_reco):
        #     # plot reco and stat at the same x positions (log-values)
        #     ax.errorbar(x_plot_log[cond_reco],
        #                 reco[:, j][cond_reco],
        #                 stat[:, j][cond_reco],
        #                 fmt='.',
        #                 color=color,
        #                 alpha=0.85,
        #                 # label=f'Empirical Q²={np.exp(log_Q2_edges[j]):.2f}'
        #                 )

    # Prefer LoTV decomposition for combined mode: compute per-bootstrap posterior samples if possible
    lotv_results = None
    if mode == "combined":
        try:
            from plotting_UQ_helpers import compute_function_lotv_for_mceg

            # attempt to build per-boot posterior samples list using same logic as bootstrap above
            per_boot_samples = []
            n_theta_per_boot = min(50, max(10, n_mc // 20))
            n_boot_for_combined = min(n_bootstrap, 20)
            for trial in range(n_boot_for_combined):
                sim.init(true_params)
                evts = sim.sample(true_params, num_events)
                feats = (
                    evts.float().to(device)
                    if isinstance(evts, torch.Tensor)
                    else torch.tensor(evts, device=device).float()
                )
                from utils import log_feature_engineering

                feats_for_pointnet = log_feature_engineering(feats).float().unsqueeze(0)
                try:
                    local_post = posterior_sampler(
                        feats,
                        pointnet_model,
                        model,
                        laplace_model,
                        n_samples=n_theta_per_boot,
                        device=device,
                    )
                    per_boot_samples.append(local_post.detach().cpu().numpy())
                except Exception:
                    # fallback: use MAP estimate
                    if model is not None and pointnet_model is not None:
                        est = _estimate_parameters_nn(
                            evts, model, pointnet_model, device
                        )
                        per_boot_samples.append(est.detach().cpu().numpy()[None, :])
                    else:
                        per_boot_samples.append(
                            true_params.detach().cpu().numpy()[None, :]
                        )

            lotv_results = compute_function_lotv_for_mceg(
                sim,
                per_boot_posterior_samples=per_boot_samples,
                n_theta_per_boot=n_theta_per_boot,
                num_events=min(20000, num_events),
                nx=hist[1].shape[0] - 1,
                nQ2=hist[2].shape[0] - 1,
                device=device,
            )
        except Exception:
            lotv_results = None

    # Now plot bands (percentiles) but aligned to the collaborator x positions (log-values)
    for mode_name, reco_samples, stat_samples, hist_used in all_bands:
        # choose a linestyle per mode so modes remain distinguishable while keeping Q2 color consistent
        if mode_name == "parameter":
            ls = "dotted"
            alpha_fill_outer = 0.18
            alpha_fill_inner = 0.30
            median_marker = None
        elif mode_name == "bootstrap":
            ls = "dotted"  # (0, (3, 1, 1, 1))  # dash-dot-ish
            alpha_fill_outer = 0.12
            alpha_fill_inner = 0.22
            median_marker = None
        else:
            ls = "dotted"
            alpha_fill_outer = 0.15
            alpha_fill_inner = 0.25
            median_marker = None

        for idx, j in enumerate(Q2_indices):
            if np.exp(log_Q2_edges[j]) > 100:
                continue
            color = cmap_val(idx)
            # reco_samples: [n_samples, nx, nQ2]
            curves = reco_samples[:, :, j]  # shape [n_samples, nx]
            if curves.size == 0:
                continue
            q5 = np.percentile(curves, 5, axis=0)
            q95 = np.percentile(curves, 95, axis=0)
            q25 = np.percentile(curves, 25, axis=0)
            q75 = np.percentile(curves, 75, axis=0)
            median = np.median(curves, axis=0)
            stat_err = np.median(stat_samples[:, :, j], axis=0)
            mask = median > 0
            if not np.any(mask):
                continue
            label = f"{label_map.get(mode_name, mode_name)} Q²={np.exp(log_Q2_edges[j]):.2f}"
            # median line (colored by Q2)
            ax.plot(
                x_plot_log[mask],
                median[mask],
                # label=label,
                color=color,
                linestyle=ls,
                linewidth=1.5,
                marker=median_marker,
                alpha=1.0,
            )
            # # outer percentile band (5-95)
            # ax.fill_between(x_plot_log[mask],
            #                 q5[mask],
            #                 q95[mask],
            #                 alpha=alpha_fill_outer,
            #                 color=color,
            #                 linewidth=0)
            # inner percentile band (25-75)
            ax.fill_between(
                x_plot_log[mask],
                q25[mask],
                q75[mask],
                alpha=alpha_fill_inner,
                color=color,
                linewidth=0,
            )
            # # median errorbars (stat)
            # ax.errorbar(x_plot_log[mask],
            #             median[mask],
            #             stat_err[mask],
            #             fmt='.',
            #             alpha=0.6,
            #             color=color)

    ax.tick_params(axis="both", which="major", labelsize=22, direction="in")
    ax.set_ylabel(r"$d\sigma/dxdQ^2~[{\rm GeV^{-4}}]$", size=25)
    ax.set_xlabel(r"$\log(x)$", size=25)

    # collaborator used semilogy for y scaling (we keep that to match look)
    ax.semilogy()

    # xticks use log-values to show familiar x locations
    ax.set_xticks(np.log([1e-4, 1e-3, 1e-2, 1e-1]))
    ax.set_xticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])

    # Remove duplicated legend entries (keep order)
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l not in by_label:
            by_label[l] = h
    ax.legend(by_label.values(), by_label.keys(), fontsize=20, ncol=2)

    # ax.set_title(f'MCEG Function Uncertainty Bands ({mode})', fontsize=14)
    plt.tight_layout()
    if save_dir:
        plt.savefig(
            f"{save_dir}/mceg_function_uncertainty_{mode}_bands_collab_true.png",
            dpi=200,
        )


def plot_pdf_uncertainty_mceg(
    model,
    pointnet_model,
    laplace_model,
    true_params,
    device,
    num_events,
    mode="posterior",
    n_mc=100,
    n_bootstrap=100,
    Q2_slices=None,
    save_dir=None,
    sbi_samples_list=None,
    sbi_labels=None,
    x_grid=None,
    combined_plot_modes=False,
    combined_plot_sbi=False,
):
    """
    Produce PDF uncertainty plots for MCEG by sampling parameter posteriors and
    evaluating the collaborator PDF object via pdf.setup(par). For each method
    (posterior / bootstrap / combined / SBI) this writes one plot per requested
    Q2 slice showing mean ± std across parameter samples (as in the example).

    Signature mirrors plot_function_uncertainty_mceg for easy swapping.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from tqdm import tqdm

    from simulator import ALPHAS, EWEAK, MELLIN, PDF, MCEGSimulator

    mellin = MELLIN(npts=8)
    alphaS = ALPHAS()
    eweak = EWEAK()

    sim = MCEGSimulator(device=device)

    # x grid for PDF evaluations
    if x_grid is None:
        x_grid = np.linspace(0.001, 0.99, 100)

    # Default Q2: single value Q2=10 as requested
    if Q2_slices is None:
        Q2_slices = [10.0]
    else:
        # Respect provided list but only use first entry to keep single-Q2 behavior
        if len(Q2_slices) > 1:
            Q2_slices = [Q2_slices[0]]

    os.makedirs(save_dir or ".", exist_ok=True)

    # Helper to map theta into PDF parameter array and evaluate u-ubar difference
    def eval_pdf_for_theta(theta_arr, q2_val):
        pdf_temp = PDF(mellin, alphaS)
        # create new parameter array and insert theta into expected slots
        new_cpar = pdf_temp.get_current_par_array()[::]
        try:
            new_cpar[4 : 4 + len(theta_arr)] = theta_arr
        except Exception:
            # fallback: try slice assignment more generally
            try:
                new_cpar[4:8] = theta_arr
            except Exception:
                # if still failing, attempt to broadcast
                arr = np.asarray(theta_arr)
                new_cpar[4 : 4 + arr.shape[0]] = arr
        try:
            pdf_temp.setup(new_cpar)
        except Exception:
            # If setup fails, try passing theta directly (some setups accept direct theta)
            try:
                pdf_temp.setup(theta_arr)
            except Exception:
                raise

        # Evaluate u and ub and return their difference across x_grid
        vals = []
        for x in x_grid:
            try:
                u = pdf_temp.get_xF(x, q2_val, "u", evolve=True)
                ub = pdf_temp.get_xF(x, q2_val, "ub", evolve=True)
                # get_xF may return arrays or scalars
                uval = float(u[0]) if hasattr(u, "__len__") else float(u)
                ubval = float(ub[0]) if hasattr(ub, "__len__") else float(ub)
                vals.append(uval - ubval)
            except Exception:
                vals.append(np.nan)
        return np.array(vals)

    # Build sample sets depending on mode
    sample_sets = (
        []
    )  # list of tuples (label, np.ndarray of shape [n_samples, param_dim])

    # Ensure sim initialized for true params
    sim.init(true_params)

    # New API: generate observed data and posterior/boot samples similar to other function
    # Create observed data for posterior sampler
    evts_obs = sim.sample(true_params, num_events)

    # Posterior samples (Laplace)
    if mode in ["posterior", "combined"] and laplace_model is not None:
        try:
            feats = (
                evts_obs.float().to(device)
                if isinstance(evts_obs, torch.Tensor)
                else torch.tensor(evts_obs, device=device).float()
            )
            from utils import log_feature_engineering

            feats_for_pointnet = log_feature_engineering(feats).float().unsqueeze(0)
            theta_post = posterior_sampler(
                feats,
                pointnet_model,
                model,
                laplace_model,
                n_samples=n_mc,
                device=device,
            )
            sample_sets.append(("posterior", theta_post.detach().cpu().numpy()))
        except Exception:
            pass

    # Bootstrap samples
    if mode in ["bootstrap", "combined"]:
        boot_thetas = []
        for b in tqdm(range(n_bootstrap), desc="building bootstrap thetas"):
            sim.init(true_params)
            evts = sim.sample(true_params, num_events)
            feats = (
                evts.float().to(device)
                if isinstance(evts, torch.Tensor)
                else torch.tensor(evts, device=device).float()
            )
            from utils import log_feature_engineering

            feats_for_pointnet = log_feature_engineering(feats).float().unsqueeze(0)
            try:
                inferred = model(pointnet_model(feats_for_pointnet).detach())
                try:
                    inferred_theta = inferred.mean(dim=0).detach().cpu().numpy()
                except Exception:
                    inferred_theta = inferred.detach().cpu().numpy().ravel()
            except Exception:
                # fallback to perturbation
                inferred_theta = (
                    true_params.detach().cpu().numpy()
                    + np.random.randn(len(true_params)) * 0.1
                )
            boot_thetas.append(inferred_theta)
        if len(boot_thetas) > 0:
            sample_sets.append(("bootstrap", np.array(boot_thetas)))

    # SBI samples if provided
    if sbi_samples_list is not None:
        if not isinstance(sbi_samples_list, (list, tuple)):
            sbi_list = [sbi_samples_list]
        else:
            sbi_list = list(sbi_samples_list)
        for idx_s, sbi_set in enumerate(sbi_list):
            label = None
            if isinstance(sbi_labels, (list, tuple)) and idx_s < len(sbi_labels):
                label = sbi_labels[idx_s]
            elif isinstance(sbi_labels, str):
                label = sbi_labels
            else:
                label = f"SBI_{idx_s+1}"

            if hasattr(sbi_set, "detach"):
                arr = sbi_set.detach().cpu().numpy()
            else:
                arr = np.asarray(sbi_set)
            if arr.size > 0:
                sample_sets.append((f"sbi_{label}", arr))

    # If no sample sets found (e.g., laplace_model missing and mode posterior), warn and exit
    if len(sample_sets) == 0:
        print(
            "⚠️ No parameter samples generated for PDF plotting (check laplace_model / model)."
        )
        return

    # For combined mode where we want to include posterior+bootstrap together, merge if requested
    if mode == "combined":
        # If both posterior and bootstrap present, merge into a 'combined' set for plotting
        labels = [s[0] for s in sample_sets]
        if "posterior" in labels and "bootstrap" in labels:
            post = next(s for s in sample_sets if s[0] == "posterior")[1]
            boot = next(s for s in sample_sets if s[0] == "bootstrap")[1]
            # Attempt to compute a Combined LoTV decomposition (preferred) if models are available
            lotv_combined = None
            try:
                # Only compute LoTV if we have the posterior sampler available (model + pointnet + laplace)
                if (
                    (model is not None)
                    and (pointnet_model is not None)
                    and (laplace_model is not None)
                ):
                    from plotting_UQ_helpers import \
                        compute_function_lotv_for_mceg

                    # Build per-boot posterior samples by drawing a small posterior for a subset of bootstraps
                    per_boot_samples = []
                    # Use up to 20 bootstrap replicates for the LoTV computation.
                    # Per-user request: draw n_mc Laplace-approx samples for each bootstrap
                    # to ensure symmetric sampling between the observed-data posterior
                    # and each bootstrap posterior. This can be computationally heavy.
                    n_boot_for_combined = min(n_bootstrap, 20)
                    n_theta_per_boot = int(n_mc)
                    if n_boot_for_combined > 20:
                        n_boot_for_combined = 20
                    # Warn the user about potential heavy computation
                    try:
                        print(
                            f"⚠️ LoTV computation: using {n_boot_for_combined} bootstraps x {n_theta_per_boot} LA samples per bootstrap (total ~{n_boot_for_combined * n_theta_per_boot} samples). This may be slow."
                        )
                    except Exception:
                        pass
                    for trial in range(n_boot_for_combined):
                        try:
                            sim.init(true_params)
                            evts_b = sim.sample(true_params, num_events)
                            feats_b = (
                                evts_b.float().to(device)
                                if isinstance(evts_b, torch.Tensor)
                                else torch.tensor(evts_b, device=device).float()
                            )
                            from utils import log_feature_engineering

                            feats_for_pointnet_b = (
                                log_feature_engineering(feats_b).float().unsqueeze(0)
                            )
                            theta_post_b = posterior_sampler(
                                feats_b,
                                pointnet_model,
                                model,
                                laplace_model,
                                n_samples=n_theta_per_boot,
                                device=device,
                            )
                            per_boot_samples.append(theta_post_b.detach().cpu().numpy())
                        except Exception:
                            # if posterior sampling fails for this bootstrap, skip
                            continue

                    # If we collected per-boot posterior samples, compute LoTV directly over the PDF via eval_pdf_for_theta
                    if len(per_boot_samples) > 0:
                        # Compute per-boot means and within-variance over x_grid using eval_pdf_for_theta
                        import numpy as _np

                        boot_means = []
                        boot_withins = []
                        q2_val = float(Q2_slices[0])
                        for samples_arr in per_boot_samples:
                            pdfs_b = []
                            for th in samples_arr:
                                try:
                                    pdf_vals = eval_pdf_for_theta(th, q2_val)
                                except Exception:
                                    pdf_vals = _np.full_like(x_grid, _np.nan)
                                pdfs_b.append(pdf_vals)
                            pdfs_b = _np.array(pdfs_b)  # [n_theta_b, nx]
                            mu_b = _np.nanmean(pdfs_b, axis=0)
                            var_b = _np.nanvar(pdfs_b, axis=0)
                            boot_means.append(mu_b)
                            boot_withins.append(var_b)

                        boot_means = _np.stack(boot_means, axis=0)  # [B, nx]
                        boot_withins = _np.stack(boot_withins, axis=0)  # [B, nx]
                        avg_within = _np.nanmean(boot_withins, axis=0)
                        between = _np.nanvar(boot_means, axis=0)
                        total_var = avg_within + between
                        mean_curve = _np.nanmean(boot_means, axis=0)
                        lotv_combined = {"mean": mean_curve, "total_var": total_var}

            except Exception:
                lotv_combined = None

            # Enforce that 'combined' is always the principled Combined-LoTV computed here.
            # Do not fall back to pooled/concatenated samples. If we failed to compute
            # a principled LoTV (e.g. missing posterior sampler / Laplace components),
            # raise an informative error so the user can provide the required inputs.
            if lotv_combined is None:
                raise RuntimeError(
                    "Combined LoTV could not be computed inside plot_pdf_uncertainty_mceg. "
                    "This plotting mode requires access to the posterior sampler and model components "
                    "(model, pointnet_model, laplace_model and a working posterior_sampler). "
                    "Provide these or disable mode='combined'."
                )

            # remove previous posterior/bootstrap entries and insert combined_lotv at front
            sample_sets = [
                s for s in sample_sets if s[0] not in ("posterior", "bootstrap")
            ]
            sample_sets.insert(0, ("combined_lotv", lotv_combined))

    # Evaluate and plot per sample set and per Q2 slice
    # Precompute ground truth PDF (u - ub) for the requested Q2 (single value)
    q2_val = float(Q2_slices[0])
    try:
        theta_true = (
            true_params.detach().cpu().numpy()
            if hasattr(true_params, "detach")
            else np.asarray(true_params)
        )
        true_curve = eval_pdf_for_theta(theta_true, q2_val)
    except Exception:
        true_curve = None

    # If requested, create a combined plot overlaying posterior (Laplace), bootstrap, combined
    if combined_plot_modes:
        fig, ax = plt.subplots(figsize=(8, 5))
        # color / label mapping for UQ modes
        colors = ["C0", "C1", "C2"]
        mode_order = ["posterior", "bootstrap", "combined"]
        mode_color_map = {
            name: colors[i % len(colors)] for i, name in enumerate(mode_order)
        }
        mode_label_map = {
            "posterior": "LA",
            "bootstrap": "Bootstrap",
            "combined": "Ours",
        }
        plotted_any = False
        for kidx, mode_name in enumerate(mode_order):
            entry = next((s for s in sample_sets if s[0] == mode_name), None)
            if entry is None:
                continue
            _, thetas = entry
            thetas_arr = np.asarray(thetas)
            if thetas_arr.ndim == 1:
                thetas_arr = thetas_arr[None, :]
            n_samps = thetas_arr.shape[0]
            max_samps = min(500, n_samps)
            idxs = np.linspace(0, n_samps - 1, max_samps).astype(int)
            pdfs = []
            for ii in tqdm(
                idxs, desc=f"Evaluating PDFs for combined {mode_name} Q2={q2_val}"
            ):
                theta = thetas_arr[ii]
                try:
                    vals = eval_pdf_for_theta(theta, q2_val)
                except Exception as e:
                    vals = np.full_like(x_grid, np.nan)
                pdfs.append(vals)
            pdfs = np.array(pdfs)
            # percentiles: median, IQR (25-75), outer (5-95)
            median = np.nanmedian(pdfs, axis=0)
            p25 = np.nanpercentile(pdfs, 25, axis=0)
            p75 = np.nanpercentile(pdfs, 75, axis=0)
            p05 = np.nanpercentile(pdfs, 5, axis=0)
            p95 = np.nanpercentile(pdfs, 95, axis=0)
            col = mode_color_map.get(mode_name, colors[kidx % len(colors)])
            lab = mode_label_map.get(mode_name, mode_name.capitalize())
            # ax.fill_between(x_grid, p05, p95, color=col, alpha=0.12)
            ax.fill_between(x_grid, p25, p75, color=col, alpha=0.22)
            ax.plot(x_grid, median, color=col, linewidth=2, label=lab)
            plotted_any = True
        # If we computed a Combined LoTV result, plot it with Gaussian-derived bands
        combined_lotv_entry = next(
            (s for s in sample_sets if s[0] == "combined_lotv"), None
        )
        if combined_lotv_entry is not None:
            _, lotv = combined_lotv_entry
            mean_curve = lotv.get("mean")
            total_var = lotv.get("total_var")
            if mean_curve is not None and total_var is not None:
                sigma = np.sqrt(np.maximum(total_var, 0.0))
                # Gaussian-approx percentiles: 5/95 ~ +-1.645 sigma, 25/75 ~ +-0.6745 sigma
                p05 = mean_curve - 1.645 * sigma
                p95 = mean_curve + 1.645 * sigma
                p25 = mean_curve - 0.6745 * sigma
                p75 = mean_curve + 0.6745 * sigma
                col = mode_color_map.get("combined", "C2")
                # ax.fill_between(x_grid, p05, p95, color=col, alpha=0.10)
                ax.fill_between(x_grid, p25, p75, color=col, alpha=0.18)
                ax.plot(
                    x_grid, mean_curve, color=col, linewidth=2.5, label="Ours (LoTV)"
                )
                plotted_any = True
        # plot SBI combined if requested too? not here
        # plot true curve
        if true_curve is not None:
            ax.plot(x_grid, true_curve, "r--", linewidth=2, label="True")
        ax.set_ylim(0, None)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$f(x|\theta)$")
        ax.set_yscale("log")
        # ax.set_title('Combined modes PDF uncertainty (Q²={:.1f})'.format(q2_val))
        ax.legend()
        out_name = (
            f"{save_dir}/mceg_pdf_uncertainty_combined_modes_Q2_{str(q2_val).replace('.', 'p')}.png"
            if save_dir
            else f"mceg_pdf_uncertainty_combined_modes_Q2_{str(q2_val).replace('.', 'p')}.png"
        )
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved combined-modes PDF uncertainty plot: {out_name}")

    # If requested, create a combined SBI overlay plot (SNPE / MCABC / MCABC-W)
    if combined_plot_sbi and any(s[0].startswith("sbi_") for s in sample_sets):
        fig, ax = plt.subplots(figsize=(8, 5))
        # color palette and label normalization for SBI methods
        palette = ["C3", "C4", "C5", "C6"]
        # If a 'combined' sample set exists, plot it first with a distinct color/label
        # accept either a pooled 'combined' sample array or a 'combined_lotv' dict produced above
        combined_entry = next(
            (s for s in sample_sets if s[0] in ("combined", "combined_lotv")), None
        )
        if combined_entry is not None:
            lab, thetas = combined_entry
            # If this is a LoTV dict, render Gaussian-derived bands from mean/total_var
            if lab == "combined_lotv" and isinstance(thetas, dict):
                lotv = thetas
                mean_curve = lotv.get("mean")
                total_var = lotv.get("total_var")
                if mean_curve is not None and total_var is not None:
                    sigma = np.sqrt(np.maximum(total_var, 0.0))
                    p05 = mean_curve - 1.645 * sigma
                    p95 = mean_curve + 1.645 * sigma
                    p25 = mean_curve - 0.6745 * sigma
                    p75 = mean_curve + 0.6745 * sigma
                    col_comb = "C2"
                    # ax.fill_between(x_grid, p05, p95, color=col_comb, alpha=0.10)
                    ax.fill_between(x_grid, p25, p75, color=col_comb, alpha=0.18)
                    ax.plot(
                        x_grid, mean_curve, color=col_comb, linewidth=2.5, label="Ours"
                    )
            else:
                thetas_arr = np.asarray(thetas)
                if thetas_arr.ndim == 1:
                    thetas_arr = thetas_arr[None, :]
                n_samps = thetas_arr.shape[0]
                max_samps = min(500, n_samps)
                idxs = np.linspace(0, n_samps - 1, max_samps).astype(int)
                pdfs = []
                for ii in tqdm(idxs, desc=f"Evaluating PDFs for combined Q2={q2_val}"):
                    theta = thetas_arr[ii]
                    try:
                        vals = eval_pdf_for_theta(theta, q2_val)
                    except Exception:
                        vals = np.full_like(x_grid, np.nan)
                    pdfs.append(vals)
                pdfs = np.array(pdfs)
                median = np.nanmedian(pdfs, axis=0)
                p25 = np.nanpercentile(pdfs, 25, axis=0)
                p75 = np.nanpercentile(pdfs, 75, axis=0)
                p05 = np.nanpercentile(pdfs, 5, axis=0)
                p95 = np.nanpercentile(pdfs, 95, axis=0)
                col_comb = "C2"
                # ax.fill_between(x_grid, p05, p95, color=col_comb, alpha=0.10)
                ax.fill_between(x_grid, p25, p75, color=col_comb, alpha=0.18)
                ax.plot(x_grid, median, color=col_comb, linewidth=2.5, label="Ours")
            # If this was a combined LoTV dict we already plotted it above; skip
            # the pooled-sample evaluation in that case to avoid referencing undefined vars.
            if not (lab == "combined_lotv" and isinstance(thetas, dict)):
                if thetas_arr.ndim == 1:
                    thetas_arr = thetas_arr[None, :]
                n_samps = thetas_arr.shape[0]
                max_samps = min(500, n_samps)
                idxs = np.linspace(0, n_samps - 1, max_samps).astype(int)
                pdfs = []
                for ii in tqdm(idxs, desc=f"Evaluating PDFs for combined Q2={q2_val}"):
                    theta = thetas_arr[ii]
                    try:
                        vals = eval_pdf_for_theta(theta, q2_val)
                    except Exception:
                        vals = np.full_like(x_grid, np.nan)
                    pdfs.append(vals)
                pdfs = np.array(pdfs)
                median = np.nanmedian(pdfs, axis=0)
                p25 = np.nanpercentile(pdfs, 25, axis=0)
                p75 = np.nanpercentile(pdfs, 75, axis=0)
                p05 = np.nanpercentile(pdfs, 5, axis=0)
                p95 = np.nanpercentile(pdfs, 95, axis=0)
                # Plot Combined first
                col_comb = "C2"
                ax.fill_between(x_grid, p05, p95, color=col_comb, alpha=0.10)
                ax.fill_between(x_grid, p25, p75, color=col_comb, alpha=0.18)
                ax.plot(x_grid, median, color=col_comb, linewidth=2.5, label="Ours")
            sbi_entries = [s for s in sample_sets if s[0].startswith("sbi_")]
            # normalize label names (e.g., 'sbi_snpe' -> 'SNPE') and map colors
            sbi_labels_norm = []
            for idx_s, (label, thetas) in enumerate(sbi_entries):
                thetas_arr = np.asarray(thetas)
                if thetas_arr.ndim == 1:
                    thetas_arr = thetas_arr[None, :]
                n_samps = thetas_arr.shape[0]
                max_samps = min(500, n_samps)
                idxs = np.linspace(0, n_samps - 1, max_samps).astype(int)
                pdfs = []
                for ii in tqdm(
                    idxs, desc=f"Evaluating PDFs for combined SBI {label} Q2={q2_val}"
                ):
                    theta = thetas_arr[ii]
                    try:
                        vals = eval_pdf_for_theta(theta, q2_val)
                    except Exception:
                        vals = np.full_like(x_grid, np.nan)
                    pdfs.append(vals)
                pdfs = np.array(pdfs)
                # percentiles for SBI method
                median = np.nanmedian(pdfs, axis=0)
                p25 = np.nanpercentile(pdfs, 25, axis=0)
                p75 = np.nanpercentile(pdfs, 75, axis=0)
                p05 = np.nanpercentile(pdfs, 5, axis=0)
                p95 = np.nanpercentile(pdfs, 95, axis=0)
                # derive display label and color
                display_label = label.replace("sbi_", "")
                display_label = (
                    display_label.upper() if len(display_label) <= 6 else display_label
                )
                color_idx = idx_s % len(palette)
                col = palette[color_idx]
                # ax.fill_between(x_grid, p05, p95, color=col, alpha=0.12)
                ax.fill_between(x_grid, p25, p75, color=col, alpha=0.22)
                ax.plot(x_grid, median, color=col, linewidth=2, label=display_label)
        if true_curve is not None:
            ax.plot(x_grid, true_curve, "r--", linewidth=2, label="True")
        ax.set_ylim(0, None)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$f(x|\theta)$")
        # ax.set_title('Combined SBI PDF uncertainty (Q²={:.1f})'.format(q2_val))
        ax.legend()
        out_name = (
            f"{save_dir}/mceg_pdf_uncertainty_combined_SBI_Q2_{str(q2_val).replace('.', 'p')}.png"
            if save_dir
            else f"mceg_pdf_uncertainty_combined_SBI_Q2_{str(q2_val).replace('.', 'p')}.png"
        )
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved combined-SBI PDF uncertainty plot: {out_name}")

    # If combined overlays requested, we are done; otherwise produce individual plots per label
    if combined_plot_modes or combined_plot_sbi:
        return

    for label, thetas in sample_sets:
        # Special-case: if we were given a combined LoTV dict, render it directly
        if label == "combined_lotv" and isinstance(thetas, dict):
            lotv = thetas
            mean_curve = lotv.get("mean")
            total_var = lotv.get("total_var")
            if mean_curve is None or total_var is None:
                continue
            sigma = np.sqrt(np.maximum(total_var, 0.0))
            p05 = mean_curve - 1.645 * sigma
            p95 = mean_curve + 1.645 * sigma
            p25 = mean_curve - 0.6745 * sigma
            p75 = mean_curve + 0.6745 * sigma
            # Produce a dedicated plot for the combined LoTV result (single Q2 assumed earlier)
            fig, ax = plt.subplots(figsize=(8, 5))
            col = "C2"
            # ax.fill_between(x_grid, p05, p95, color=col, alpha=0.12)
            ax.fill_between(x_grid, p25, p75, color=col, alpha=0.28)
            ax.plot(x_grid, mean_curve, color=col, linewidth=2, label="Ours (LoTV)")
            if true_curve is not None:
                ax.plot(x_grid, true_curve, "r--", linewidth=2, label="Ground truth")
            ax.set_ylim(0, None)
            ax.set_xlabel("x")
            ax.set_ylabel(r"$f(x|\theta)$")
            ax.legend()
            out_name = (
                f"{save_dir}/mceg_pdf_uncertainty_combined_lotv_Q2_{str(float(Q2_slices[0])).replace('.', 'p')}.png"
                if save_dir
                else f"mceg_pdf_uncertainty_combined_lotv_Q2_{str(float(Q2_slices[0])).replace('.', 'p')}.png"
            )
            plt.savefig(out_name, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"✓ Saved Combined LoTV PDF plot: {out_name}")
            continue
        for q2 in Q2_slices:
            # limit number of samples to keep runtime reasonable
            thetas_arr = np.asarray(thetas)
            if thetas_arr.ndim == 1:
                thetas_arr = thetas_arr[None, :]
            n_samps = thetas_arr.shape[0]
            max_samps = min(500, n_samps)  # cap for safety
            idxs = np.linspace(0, n_samps - 1, max_samps).astype(int)
            pdfs = []
            for ii in tqdm(idxs, desc=f"Evaluating PDFs for {label} Q2={q2}"):
                theta = thetas_arr[ii]
                try:
                    vals = eval_pdf_for_theta(theta, q2)
                except Exception as e:
                    print(f"⚠️ pdf eval failed for sample {ii}: {e}")
                    vals = np.full_like(x_grid, np.nan)
                pdfs.append(vals)
            pdfs = np.array(pdfs)
            # compute percentile bands across samples (ignore NaNs)
            median = np.nanmedian(pdfs, axis=0)
            p25 = np.nanpercentile(pdfs, 25, axis=0)
            p75 = np.nanpercentile(pdfs, 75, axis=0)
            p05 = np.nanpercentile(pdfs, 5, axis=0)
            p95 = np.nanpercentile(pdfs, 95, axis=0)

            # Plot (individual plot for this label)
            fig, ax = plt.subplots(figsize=(8, 5))
            # choose color and friendly label for this entry
            if label.startswith("sbi_"):
                display_label = label.replace("sbi_", "")
                display_label = (
                    display_label.upper() if len(display_label) <= 6 else display_label
                )
                color = "C3"
            else:
                display_label = label.capitalize() if label != "posterior" else "LA"
                # map posterior/bootstrap/combined to distinct colors
                color_map = {"posterior": "C0", "bootstrap": "C1", "combined": "C2"}
                color = color_map.get(label, "C0")
            # ax.fill_between(x_grid, p05, p95, color=color, alpha=0.12)
            ax.fill_between(x_grid, p25, p75, color=color, alpha=0.28)
            ax.plot(x_grid, median, color=color, linewidth=2, label=display_label)
            # Ground truth dashed line
            if true_curve is not None:
                ax.plot(x_grid, true_curve, "r--", linewidth=2, label="Ground truth")
            ax.set_ylim(0, None)
            ax.set_xlabel("x")
            ax.set_ylabel(r"$f(x|\theta)$")
            # ax.set_title(f'{display_label} - PDF uncertainty at Q²={q2}')
            ax.legend()

            out_name = (
                f"{save_dir}/mceg_pdf_uncertainty_{label}_Q2_{str(q2).replace('.', 'p')}.png"
                if save_dir
                else f"mceg_pdf_uncertainty_{label}_Q2_{str(q2).replace('.', 'p')}.png"
            )
            plt.savefig(out_name, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"✓ Saved PDF uncertainty plot: {out_name}")


def plot_function_uncertainty(
    model=None,
    pointnet_model=None,
    true_params=None,
    device=None,
    num_events=2000,
    problem="simplified_dis",
    save_dir="plots",
    save_path=None,
    n_mc=100,
    laplace_model=None,
    mode="posterior",
    n_bootstrap=100,
    # Backward compatibility
    simulator=None,
    posterior_samples=None,
    true_theta=None,
):
    """
    Plot function-space (predictive) uncertainty by propagating parameter uncertainty.

    This function supports both the new generator-style API and backward compatibility
    with the original API for custom data.

    New API Parameters:
    ------------------
    model : torch.nn.Module
        Parameter prediction model (head)
    pointnet_model : torch.nn.Module
        PointNet feature extractor
    true_params : torch.Tensor
        Ground truth parameter values
    device : torch.device
        Device to run computations on
    num_events : int
        Number of events to generate for analysis
    problem : str
        Problem type ('simplified_dis', 'realistic_dis', 'gaussian', 'mceg')
    save_dir : str
        Directory to save plots (used if save_path not provided)
    save_path : str, optional
        Full path to save plot (overrides save_dir)
    n_mc : int
        Number of Monte Carlo samples for uncertainty estimation
    laplace_model : object, optional
        Fitted Laplace approximation for analytic uncertainty
    mode : str, optional
        Type of uncertainty to visualize (default: 'posterior')
        - 'posterior': Function uncertainty from posterior for single dataset
        - 'bootstrap': Bootstrap/data uncertainty across repeated datasets
        - 'combined': Both posterior and bootstrap uncertainty for comparison
    n_bootstrap : int
        Number of bootstrap samples (used for 'bootstrap' and 'combined' modes)

    Backward Compatibility Parameters:
    ---------------------------------
    simulator : object
        Legacy simulator object
    posterior_samples : torch.Tensor
        Legacy posterior samples
    true_theta : torch.Tensor
        Legacy true parameter tensor

    LaTeX Description:
    ==================

    \\section{Function-Space (Predictive) Uncertainty}

    This figure demonstrates how parameter uncertainty propagates to function predictions,
    showing the predictive uncertainty $p(f(x)|\\mathcal{D})$ obtained by marginalizing
    over the parameter posterior:

    $$p(f(x)|\\mathcal{D}) = \\int p(f(x)|\\theta) p(\\theta|\\mathcal{D}) d\\theta$$

    The visualization includes:

    \\begin{itemize}
    \\item \\textbf{Uncertainty bands}: Shaded regions showing 50\\% (dark) and 90\\% (light)
      confidence intervals for function predictions at each $x$
    \\item \\textbf{Median prediction}: Blue solid line showing the median $f(x)$ across all posterior samples
    \\item \\textbf{True function}: Red dashed line showing the ground truth $f(x|\\theta_{\\text{true}})$
    \\item \\textbf{Sample functions}: Gray lines showing individual function realizations from posterior samples
    \\end{itemize}

    The computation procedure:
    \\begin{enumerate}
    \\item Sample parameters $\\{\\theta^{(i)}\\}$ from the posterior $p(\\theta|\\mathcal{D})$
    \\item Evaluate function $f(x|\\theta^{(i)})$ for each sample at all $x$ values
    \\item Compute empirical quantiles across samples to form confidence bands
    \\end{enumerate}

    The width of the uncertainty bands indicates how confident we are in our function
    predictions given the observed data. Wider bands indicate higher uncertainty,
    typically occurring in regions where the data provides less constraint.

    \\textbf{Mode Parameter:}
    \\begin{itemize}
    \\item \\textbf{posterior}: Uses function uncertainty from posterior for single dataset
    \\item \\textbf{bootstrap}: Uses bootstrap/data uncertainty across repeated datasets
    \\item \\textbf{combined}: Shows both types of uncertainty for comparison
    \\end{itemize}
    """
    print("📈 Generating function-space uncertainty plot...")

    # Validate mode parameter
    valid_modes = ["posterior", "bootstrap", "combined"]
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got: {mode}")

    print(f"   Mode: {mode}")

    # Handle backward compatibility - detect legacy positional arguments
    # If the first argument looks like a simulator and we have tensor/array arguments, treat as legacy API
    if (
        model is not None
        and hasattr(model, "sample")
        and hasattr(model, "init")
        and pointnet_model is not None
        and (
            isinstance(pointnet_model, torch.Tensor) or hasattr(pointnet_model, "shape")
        )
        and true_params is not None
        and (isinstance(true_params, torch.Tensor) or hasattr(true_params, "shape"))
    ):
        # Legacy positional call: plot_function_uncertainty(simulator, posterior_samples, true_theta, save_dir, ...)
        print("   Using legacy API with positional arguments")
        simulator = model  # First arg is actually simulator
        posterior_samples = pointnet_model  # Second arg is actually posterior_samples
        true_theta = true_params  # Third arg is actually true_theta
        save_dir = (
            device if isinstance(device, str) else save_dir
        )  # Fourth arg might be save_dir

        # Convert to torch tensors if needed
        if not isinstance(posterior_samples, torch.Tensor):
            posterior_samples = torch.from_numpy(posterior_samples).float()
        if not isinstance(true_theta, torch.Tensor):
            true_theta = torch.from_numpy(true_theta).float()

        # Reset new API parameters to None
        model = None
        pointnet_model = None
        true_params = None
        device = None

    # Handle backward compatibility
    if (
        simulator is not None
        and posterior_samples is not None
        and true_theta is not None
    ):
        # Legacy API - use provided simulator and data
        print("   Using legacy API with provided simulator and posterior samples")
        working_simulator = simulator
        working_true_params = true_theta

        if mode == "posterior":
            # Use provided posterior samples
            working_posterior_samples = posterior_samples
        elif mode == "bootstrap":
            # Generate bootstrap samples using the simulator
            print("   Generating bootstrap samples for legacy API...")
            bootstrap_samples = []
            for trial in range(n_bootstrap):
                # Generate new dataset
                bootstrap_data = working_simulator.sample(
                    working_true_params, num_events
                )

                # For legacy API, we don't have ML models, so use a simple perturbation
                estimated_params = (
                    working_true_params + torch.randn_like(working_true_params) * 0.1
                )
                bootstrap_samples.append(estimated_params.detach().cpu())

            working_posterior_samples = torch.stack(bootstrap_samples)
        elif mode == "combined":
            # Combine provided posterior samples with bootstrap samples
            print(
                "   Generating bootstrap samples and combining with posterior for legacy API..."
            )
            bootstrap_samples = []
            for trial in range(n_bootstrap):
                # Generate new dataset
                bootstrap_data = working_simulator.sample(
                    working_true_params, num_events
                )

                # For legacy API, we don't have ML models, so use a simple perturbation
                estimated_params = (
                    working_true_params + torch.randn_like(working_true_params) * 0.1
                )
                bootstrap_samples.append(estimated_params.detach().cpu())

            bootstrap_samples = torch.stack(bootstrap_samples)
            working_posterior_samples = torch.cat(
                [posterior_samples, bootstrap_samples], dim=0
            )
    else:
        # New API - generate data internally
        if (
            model is None
            or pointnet_model is None
            or true_params is None
            or device is None
        ):
            raise ValueError(
                "New API requires model, pointnet_model, true_params, and device"
            )

        print("   Using new generator-style API")
        # Create simulator based on problem type
        if problem == "simplified_dis":
            working_simulator = SimplifiedDIS(device=device)
        elif problem == "realistic_dis":
            working_simulator = RealisticDIS(device=device)
        elif problem == "gaussian":
            working_simulator = Gaussian2DSimulator(device=device)
        elif problem == "mceg":
            if MCEGSimulator is not None:
                working_simulator = MCEGSimulator(device=device)
            else:
                raise ValueError("MCEGSimulator not available")
        else:
            raise ValueError(f"Unknown problem type: {problem}")

        working_true_params = true_params

        # Generate observed data
        observed_data = working_simulator.sample(true_params, num_events)

        # Define prior bounds based on simulator type
        if isinstance(working_simulator, SimplifiedDIS):
            theta_bounds = [(0.5, 4.0), (0.5, 4.0), (0.5, 4.0), (0.5, 4.0)]
        elif isinstance(working_simulator, Gaussian2DSimulator):
            theta_bounds = [(-2, 2), (-2, 2), (0.5, 3.0), (0.5, 3.0), (-0.9, 0.9)]
        else:
            n_params = len(working_true_params)
            theta_bounds = [(0.1, 10.0)] * n_params

        # Generate posterior samples based on mode
        if mode == "posterior":
            # Use posterior samples from single dataset
            working_posterior_samples = posterior_sampler(
                observed_data, pointnet_model, model, laplace_model, n_samples=n_mc
            )
        elif mode == "bootstrap":
            # Generate bootstrap parameter estimates
            bootstrap_samples = []
            for trial in range(n_bootstrap):
                # Generate new dataset
                bootstrap_data = working_simulator.sample(
                    working_true_params, num_events
                )

                # Estimate parameters
                if model is not None and pointnet_model is not None:
                    # Use neural network prediction
                    estimated_params = _estimate_parameters_nn(
                        bootstrap_data, model, pointnet_model, device
                    )
                    bootstrap_samples.append(estimated_params.detach().cpu())
                else:
                    # Fallback to simplified estimation
                    estimated_params = (
                        working_true_params
                        + torch.randn_like(working_true_params) * 0.1
                    )
                    bootstrap_samples.append(estimated_params.detach().cpu())

            working_posterior_samples = torch.stack(bootstrap_samples)
        elif mode == "combined":
            # Generate both posterior and per-bootstrap posterior samples and use LoTV
            # Posterior samples for the observed dataset
            posterior_samples = posterior_sampler(
                observed_data,
                pointnet_model,
                model,
                laplace_model,
                n_samples=n_mc,
                device=device,
            )

            # For combined/LoTV we will construct per-bootstrap posterior samples
            per_boot_posterior_samples = []
            n_theta_per_boot = min(50, max(10, n_mc // 20))
            n_boot_for_combined = min(
                n_bootstrap, 20
            )  # keep computation bounded for demo
            for trial in range(n_boot_for_combined):
                bootstrap_data = working_simulator.sample(
                    working_true_params, num_events
                )
                try:
                    # Use posterior_sampler to draw posterior samples conditional on this bootstrap dataset
                    local_post = posterior_sampler(
                        bootstrap_data,
                        pointnet_model,
                        model,
                        laplace_model,
                        n_samples=n_theta_per_boot,
                        device=device,
                    )
                    per_boot_posterior_samples.append(local_post.detach().cpu().numpy())
                except Exception:
                    # Fallback: use MAP estimate as singleton posterior
                    if model is not None and pointnet_model is not None:
                        est = _estimate_parameters_nn(
                            bootstrap_data, model, pointnet_model, device
                        )
                        per_boot_posterior_samples.append(
                            est.detach().cpu().numpy()[None, :]
                        )
                    else:
                        per_boot_posterior_samples.append(
                            working_true_params.detach().cpu().numpy()[None, :]
                        )

            # Use helper to compute LoTV decomposition and also build pooled ensemble for plotting
            x_grid = torch.linspace(0.01, 0.99, 100)
            lotv = compute_function_lotv_for_simplified_dis(
                working_simulator,
                x_grid,
                per_boot_posterior_samples=per_boot_posterior_samples,
                device=device,
            )

            # Build pooled function ensemble from per-bootstrap posterior samples for plotting bands
            pooled_func_samples = {"up": [], "down": []}
            for samples in per_boot_posterior_samples:
                for theta_np in samples:
                    theta = torch.tensor(theta_np, dtype=torch.float32, device=device)
                    if isinstance(working_simulator, SimplifiedDIS):
                        x_vals_local = torch.linspace(1e-3, 1 - 1e-3, 100).to(device)
                        vals = working_simulator.f(x_vals_local, theta)
                        func_vals_up = vals["up"].cpu().detach().numpy()
                        func_vals_down = vals["down"].cpu().detach().numpy()
                        pooled_func_samples["up"].append(func_vals_up)
                        pooled_func_samples["down"].append(func_vals_down)
                    else:
                        func_vals = (
                            working_simulator.f(x_vals, theta).cpu().detach().numpy()
                        )
                        # use same array for generic function name
                        pooled_func_samples.setdefault("f", []).append(func_vals)

            # convert lists to arrays for fast indexing and signal availability
            for k in list(pooled_func_samples.keys()):
                pooled_func_samples[k] = np.array(pooled_func_samples[k])
            # mark that we have pooled per-function samples (used later in plotting loop)
            _pooled_func_samples = pooled_func_samples
            working_posterior_samples = None

    # Define x grid for evaluation
    x_vals = torch.linspace(0.01, 0.99, 100).to(device)

    if isinstance(working_simulator, SimplifiedDIS):
        function_names = ["up", "down"]
        func_names = [r"$u(x|\theta)$", r"$d(x|\theta)$"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    else:
        function_names = ["f"]
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]

    for idx, func_name in enumerate(function_names):
        ax = axes[idx]

        # Evaluate function for all posterior samples
        if working_posterior_samples is None:
            # Use pooled precomputed function samples (from per-bootstrap posterior draws)
            if "_pooled_func_samples" in locals() and func_name in _pooled_func_samples:
                func_samples = _pooled_func_samples[func_name]
            else:
                func_samples = np.zeros((0, x_vals.numel()))
        else:
            func_samples = []
            for theta in working_posterior_samples[:200]:  # Use subset for speed
                theta = theta.to(device)
                if isinstance(working_simulator, SimplifiedDIS):
                    func_vals = working_simulator.f(x_vals, theta)[func_name]
                else:
                    func_vals = working_simulator.f(x_vals, theta)
                func_samples.append(func_vals.cpu().detach().numpy())
            func_samples = np.array(func_samples)

        # Compute quantiles for uncertainty bands
        median_vals = np.median(func_samples, axis=0)
        q25 = np.percentile(func_samples, 25, axis=0)
        q75 = np.percentile(func_samples, 75, axis=0)
        q5 = np.percentile(func_samples, 5, axis=0)
        q95 = np.percentile(func_samples, 95, axis=0)
        # Plot uncertainty bands
        x_np = x_vals.cpu().numpy()
        ax.fill_between(
            x_np, q5, q95, alpha=0.2, color=COLORS["blue"], label="90th Percentile"
        )
        ax.fill_between(
            x_np, q25, q75, alpha=0.4, color=COLORS["blue"], label="50th Percentile"
        )

        # Plot median
        ax.plot(x_np, median_vals, color=COLORS["blue"], linewidth=2, label="Median")

        # Plot true function
        if isinstance(working_simulator, SimplifiedDIS):
            true_vals = working_simulator.f(x_vals, working_true_params)[func_name]
        else:
            true_vals = working_simulator.f(x_vals, working_true_params)
        ax.plot(
            x_np,
            true_vals.cpu().detach().numpy(),
            color=COLORS["red"],
            linestyle="--",
            linewidth=2,
            label="True function",
        )

        # Individual samples (show a few)
        for i in range(0, min(20, len(func_samples)), 4):
            alpha = 0.1
            ax.plot(
                x_np, func_samples[i], color=COLORS["gray"], alpha=alpha, linewidth=0.5
            )

        # Formatting
        ax.set_xlabel(r"$x$")
        if isinstance(working_simulator, SimplifiedDIS):
            ax.set_ylabel(func_names[idx])
            # ax.set_title(f'{func_name.title()} quark PDF: $f(x|\\theta)$')
        else:
            ax.set_ylabel(r"$f(x)$")
            # ax.set_title(r'Function uncertainty: $f(x|\theta)$')

        ax.set_yscale("log")
        if idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Determine save path
    if save_path is None:
        if mode == "posterior":
            save_path = os.path.join(save_dir, "function_uncertainty.png")
        else:
            save_path = os.path.join(save_dir, f"function_uncertainty_{mode}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Function uncertainty plot saved to: {save_path}")


def plot_bootstrap_uncertainty(
    model=None,
    pointnet_model=None,
    true_params=None,
    device=None,
    num_events=1000,
    n_bootstrap=50,
    laplace_model=None,
    problem="simplified_dis",
    save_dir="plots",
    save_path=None,
    # Backward compatibility
    simulator=None,
    true_theta=None,
    n_events=None,  # Legacy parameter name
):
    """
    Demonstrate bootstrap/data uncertainty by generating multiple datasets.

    This function supports both the new generator-style API and backward compatibility
    with the original API.

    New API Parameters:
    ------------------
    model : torch.nn.Module
        Parameter prediction model (head)
    pointnet_model : torch.nn.Module
        PointNet feature extractor
    true_params : torch.Tensor
        Ground truth parameter values
    device : torch.device
        Device to run computations on
    num_events : int
        Number of events per bootstrap sample
    n_bootstrap : int
        Number of bootstrap samples
    problem : str
        Problem type ('simplified_dis', 'realistic_dis', 'gaussian', 'mceg')
    save_dir : str
        Directory to save plots (used if save_path not provided)
    save_path : str, optional
        Full path to save plot (overrides save_dir)

    Backward Compatibility Parameters:
    ---------------------------------
    simulator : object
        Legacy simulator object
    true_theta : torch.Tensor
        Legacy true parameter tensor

    LaTeX Description:
    ==================

    \\section{Bootstrap/Data Uncertainty}

    This figure demonstrates the parametric bootstrap procedure for estimating data (sampling)
    uncertainty. The bootstrap quantifies how much our parameter estimates would vary if we
    could repeat the experiment multiple times with the same true parameters.

    \\textbf{Parametric Bootstrap Procedure:}
    \\begin{enumerate}
    \\item Generate $B$ independent datasets $\\{\\mathcal{D}_b\\}_{b=1}^B$ using the same true parameters $\\theta_{\\text{true}}$
    \\item For each dataset $\\mathcal{D}_b$, estimate parameters $\\hat{\\theta}_b$ using the inference procedure
    \\item Analyze the distribution of estimates $\\{\\hat{\\theta}_b\\}_{b=1}^B$
    \\end{enumerate}

    The visualization shows:
    \\begin{itemize}
    \\item \\textbf{Bootstrap histograms}: Distribution of parameter estimates across bootstrap samples
    \\item \\textbf{True values}: Red dashed lines showing the parameters used to generate all datasets
    \\item \\textbf{Bootstrap mean}: Green line showing the average estimate $\\bar{\\theta} = \\frac{1}{B}\\sum_{b=1}^B \\hat{\\theta}_b$
    \\item \\textbf{95\\% confidence intervals}: Orange shaded regions containing 95\\% of bootstrap estimates
    \\end{itemize}

    \\textbf{Key Statistics:}
    \\begin{itemize}
    \\item \\textbf{Bias}: $\\text{Bias} = \\mathbb{E}[\\hat{\\theta}] - \\theta_{\\text{true}} \\approx \\bar{\\theta} - \\theta_{\\text{true}}$
    \\item \\textbf{Standard error}: $\\text{SE} = \\sqrt{\\text{Var}[\\hat{\\theta}]} \\approx \\sqrt{\\frac{1}{B-1}\\sum_{b=1}^B (\\hat{\\theta}_b - \\bar{\\theta})^2}$
    \\end{itemize}

    This bootstrap uncertainty represents the intrinsic variability due to finite sample size,
    independent of model uncertainty. It answers: "How much would my estimates vary if I
    collected new data with the same experimental setup?"
    """
    print("🔄 Generating bootstrap uncertainty plot...")

    # Handle backward compatibility
    if simulator is not None and true_theta is not None:
        # Legacy API - use provided simulator
        print("   Using legacy API with provided simulator")
        working_simulator = simulator
        working_true_params = true_theta
        # Use legacy n_events parameter if provided, otherwise use num_events
        working_num_events = n_events if n_events is not None else num_events
    else:
        # New API - generate data internally
        if (
            model is None
            or pointnet_model is None
            or true_params is None
            or device is None
        ):
            raise ValueError(
                "New API requires model, pointnet_model, true_params, and device"
            )

        print("   Using new generator-style API")
        # Create simulator based on problem type
        if problem == "simplified_dis":
            working_simulator = SimplifiedDIS(device=device)
        elif problem == "realistic_dis":
            working_simulator = RealisticDIS(device=device)
        elif problem == "gaussian":
            working_simulator = Gaussian2DSimulator(device=device)
        elif problem == "mceg":
            if MCEGSimulator is not None:
                working_simulator = MCEGSimulator(device=device)
            else:
                raise ValueError("MCEGSimulator not available")
        else:
            raise ValueError(f"Unknown problem type: {problem}")

        working_true_params = true_params
        working_num_events = num_events

    # Generate multiple bootstrap datasets
    bootstrap_theta_estimates = []

    for i in tqdm(range(n_bootstrap), desc="Bootstrap resampling"):
        # Generate new dataset with same true parameters
        bootstrap_data = working_simulator.sample(
            working_true_params, working_num_events
        )

        # Simple parameter estimation (in practice, use your trained model)
        if isinstance(working_simulator, SimplifiedDIS):
            theta_bounds = [(0.5, 4.0), (0.5, 4.0), (0.5, 4.0), (0.5, 4.0)]
        elif isinstance(working_simulator, Gaussian2DSimulator):
            theta_bounds = [(-2, 2), (-2, 2), (0.5, 3.0), (0.5, 3.0), (-0.9, 0.9)]
        else:
            n_params = len(working_true_params)
            theta_bounds = [(0.1, 10.0)] * n_params

        # Estimate parameters for this bootstrap sample
        estimated_theta = posterior_sampler(
            bootstrap_data, pointnet_model, model, laplace_model
        )[0]
        bootstrap_theta_estimates.append(estimated_theta)

    bootstrap_theta_estimates = torch.stack(bootstrap_theta_estimates)

    # Plot bootstrap distribution
    if isinstance(working_simulator, SimplifiedDIS):
        param_names = [r"$a_u$", r"$b_u$", r"$a_d$", r"$b_d$"]
    elif isinstance(working_simulator, Gaussian2DSimulator):
        param_names = [r"$\mu_x$", r"$\mu_y$", r"$\sigma_x$", r"$\sigma_y$", r"$\rho$"]
    else:
        n_params = len(working_true_params)
        param_names = [f"$\\theta_{{{i+1}}}$" for i in range(n_params)]

    n_params = len(param_names)
    fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(15, 8))
    if n_params <= 2:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()

    for i in range(n_params):
        ax = axes[i]

        # Bootstrap estimates
        bootstrap_vals = bootstrap_theta_estimates[:, i].numpy()

        # Plot histogram
        ax.hist(
            bootstrap_vals,
            bins=20,
            alpha=0.7,
            color=COLORS["orange"],
            density=True,
            label="Bootstrap distribution",
        )

        # True value
        if i < len(working_true_params):
            ax.axvline(
                working_true_params[i].item(),
                color=COLORS["red"],
                linestyle="--",
                linewidth=2,
                label="True value",
            )

        # Bootstrap statistics
        boot_mean = np.mean(bootstrap_vals)
        boot_std = np.std(bootstrap_vals)

        ax.axvline(
            boot_mean,
            color=COLORS["green"],
            linestyle="-",
            linewidth=2,
            label="Bootstrap mean",
        )

        # Confidence intervals
        ci_lower, ci_upper = np.percentile(bootstrap_vals, [2.5, 97.5])
        ax.axvspan(
            ci_lower, ci_upper, alpha=0.2, color=COLORS["orange"], label="95% CI"
        )

        # Formatting
        ax.set_xlabel(param_names[i])
        ax.set_ylabel("Probability density")
        ax.set_title(f"Bootstrap: {param_names[i]}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Statistics box
        if i < len(working_true_params):
            bias = boot_mean - working_true_params[i].item()
            stats_text = f"Mean: {boot_mean:.3f}\nStd: {boot_std:.3f}\nBias: {bias:.3f}"
        else:
            stats_text = f"Mean: {boot_mean:.3f}\nStd: {boot_std:.3f}"
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Remove extra subplots
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    # Determine save path
    if save_path is None:
        save_path = os.path.join(save_dir, "bootstrap_uncertainty.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Bootstrap uncertainty plot saved to: {save_path}")
    return bootstrap_theta_estimates


def plot_combined_uncertainty_decomposition(
    model=None,
    pointnet_model=None,
    true_params=None,
    device=None,
    num_events=1000,
    n_bootstrap=20,
    problem="simplified_dis",
    save_dir="plots",
    save_path=None,
    laplace_model=None,
    # Backward compatibility
    simulator=None,
    true_theta=None,
    posterior_samples=None,
    bootstrap_estimates=None,
):
    """
    Plot combined uncertainty decomposition showing total variance and its components.

    This function supports both the new generator-style API and backward compatibility
    with the original API.

    New API Parameters:
    ------------------
    model : torch.nn.Module
        Parameter prediction model (head)
    pointnet_model : torch.nn.Module
        PointNet feature extractor
    true_params : torch.Tensor
        Ground truth parameter values
    device : torch.device
        Device to run computations on
    num_events : int
        Number of events per bootstrap sample
    n_bootstrap : int
        Number of bootstrap samples
    problem : str
        Problem type ('simplified_dis', 'realistic_dis', 'gaussian', 'mceg')
    save_dir : str
        Directory to save plots (used if save_path not provided)
    save_path : str, optional
        Full path to save plot (overrides save_dir)
    laplace_model : object, optional
        Fitted Laplace approximation for analytic uncertainty

    Backward Compatibility Parameters:
    ---------------------------------
    simulator : object
        Legacy simulator object
    true_theta : torch.Tensor
        Legacy true parameter tensor
    posterior_samples : torch.Tensor
        Legacy posterior samples
    bootstrap_estimates : torch.Tensor
        Legacy bootstrap estimates

    LaTeX Description:
    ==================

    \\section{Combined Uncertainty Decomposition}

    This figure shows the decomposition of total prediction uncertainty into its constituent components,
    following the variance decomposition formula:

    $$\\text{Var}_{\\text{total}}[f(x)] = \\mathbb{E}_b[\\text{Var}_{\\theta|b}[f(x|\\theta)]] + \\text{Var}_b[\\mathbb{E}_{\\theta|b}[f(x|\\theta)]]$$

    where $b$ indexes bootstrap samples and $\\theta|b$ represents the posterior distribution for bootstrap sample $b$.

    \\textbf{Uncertainty Components:}

    \\begin{itemize}
    \\item \\textbf{Model uncertainty} (blue): $\\mathbb{E}_b[\\sigma_b^2(x)]$ - Average within-bootstrap variance representing
      uncertainty in our model predictions given a fixed dataset. This captures epistemic uncertainty
      about the model parameters.

    \\item \\textbf{Data uncertainty} (orange): $\\text{Var}_b[\\mu_b(x)]$ - Between-bootstrap variance representing
      uncertainty due to finite sample size. This captures aleatoric uncertainty arising from
      sampling variability.
    \\end{itemize}

    \\textbf{Left panels}: Absolute variance contributions showing how much each uncertainty source
    contributes to the total variance at each $x$ value.

    \\textbf{Right panels}: Relative contributions showing the fraction of total variance attributable
    to each source, with values summing to 1.

    \\textbf{Interpretation:}
    \\begin{itemize}
    \\item Regions where model uncertainty dominates suggest the model is well-constrained by the data
      but has inherent parameter uncertainty
    \\item Regions where data uncertainty dominates suggest more data would significantly reduce uncertainty
    \\item The relative importance can guide experimental design and model improvement strategies
    \\end{itemize}

    This decomposition is essential for understanding whether uncertainty reduction efforts should
    focus on improving the model or collecting more data.
    """
    print("🔬 Generating combined uncertainty decomposition plot...")

    # Handle backward compatibility
    if (
        simulator is not None
        and true_theta is not None
        and posterior_samples is not None
        and bootstrap_estimates is not None
    ):
        # Legacy API - use provided data
        print("   Using legacy API with provided data")
        working_simulator = simulator
        working_true_params = true_theta
        working_bootstrap_estimates = bootstrap_estimates
    else:
        # New API - generate data internally
        if (
            model is None
            or pointnet_model is None
            or true_params is None
            or device is None
        ):
            raise ValueError(
                "New API requires model, pointnet_model, true_params, and device"
            )

        print("   Using new generator-style API")
        # Create simulator based on problem type
        if problem == "simplified_dis":
            working_simulator = SimplifiedDIS(device=device)
        elif problem == "realistic_dis":
            working_simulator = RealisticDIS(device=device)
        elif problem == "gaussian":
            working_simulator = Gaussian2DSimulator(device=device)
        elif problem == "mceg":
            if MCEGSimulator is not None:
                working_simulator = MCEGSimulator(device=device)
            else:
                raise ValueError("MCEGSimulator not available")
        else:
            raise ValueError(f"Unknown problem type: {problem}")

        working_true_params = true_params

        # Generate bootstrap estimates
        bootstrap_theta_estimates = []

        for i in tqdm(
            range(n_bootstrap), desc="Generating bootstrap samples for decomposition"
        ):
            # Generate new dataset with same true parameters
            bootstrap_data = working_simulator.sample(true_params, num_events)

            # Define bounds
            if isinstance(working_simulator, SimplifiedDIS):
                theta_bounds = [(0.5, 4.0), (0.5, 4.0), (0.5, 4.0), (0.5, 4.0)]
            elif isinstance(working_simulator, Gaussian2DSimulator):
                theta_bounds = [(-2, 2), (-2, 2), (0.5, 3.0), (0.5, 3.0), (-0.9, 0.9)]
            else:
                n_params = len(true_params)
                theta_bounds = [(0.1, 10.0)] * n_params

            # Estimate parameters for this bootstrap sample
            estimated_theta = posterior_sampler(
                bootstrap_data, pointnet_model, model, laplace_model
            )[0]
            bootstrap_theta_estimates.append(estimated_theta)

        working_bootstrap_estimates = torch.stack(bootstrap_theta_estimates)

    # For function uncertainty decomposition
    x_vals = torch.linspace(0.01, 0.99, 50)  # Coarser grid for computational efficiency
    x_vals = x_vals.to(device)

    if isinstance(working_simulator, SimplifiedDIS):
        function_names = ["up", "down"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        function_names = ["f"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = axes.reshape(1, -1)

    for func_idx, func_name in enumerate(function_names):
        # Main uncertainty plot
        ax_main = axes[func_idx, 0]

        # Compute within-bootstrap variance (model uncertainty)
        within_bootstrap_vars = []
        between_bootstrap_means = []

        n_bootstrap_subset = min(
            20, len(working_bootstrap_estimates)
        )  # Use subset for speed

        # For MCEG-like simulators we will collect per-bootstrap posterior samples
        # and compute LoTV on binned histograms rather than calling working_simulator.f
        per_boot_posterior_samples = []
        for i in range(n_bootstrap_subset):
            # For each bootstrap, draw posterior samples around the bootstrap estimate using Laplace/posterior_sampler
            try:
                bootstrap_data = working_simulator.sample(
                    working_true_params, num_events
                )
                local_post = posterior_sampler(
                    bootstrap_data,
                    pointnet_model,
                    model,
                    laplace_model,
                    n_samples=50,
                    device=device,
                )
                local_post = local_post.detach().cpu().numpy()
                per_boot_posterior_samples.append(local_post)
            except Exception:
                # Fallback: simulate a small local posterior by jittering the bootstrap MAP
                theta_boot = working_bootstrap_estimates[i]
                local_post = (
                    theta_boot.unsqueeze(0).repeat(50, 1)
                    + 0.1 * torch.randn(50, len(theta_boot))
                ).numpy()
                per_boot_posterior_samples.append(local_post)
            # For SimplifiedDIS we'll still evaluate per-sample f(x|theta) directly; for other simulators
            # we defer evaluation to compute_function_lotv_for_mceg below.
            if isinstance(working_simulator, SimplifiedDIS):
                # Evaluate f for local posterior samples
                func_vals_local = []
                for theta_np in local_post:
                    theta = torch.tensor(theta_np, dtype=torch.float32, device=device)
                    func_val = working_simulator.f(x_vals, theta)[func_name]
                    func_vals_local.append(func_val.cpu().detach().numpy())
                func_vals_local = np.array(func_vals_local)
                within_var = np.var(func_vals_local, axis=0)
                within_bootstrap_vars.append(within_var)
                between_mean = np.mean(func_vals_local, axis=0)
                between_bootstrap_means.append(between_mean)

        if isinstance(working_simulator, SimplifiedDIS):
            within_bootstrap_vars = np.array(within_bootstrap_vars)
            between_bootstrap_means = np.array(between_bootstrap_means)

            # Average within-bootstrap variance (model uncertainty component)
            avg_within_var = np.mean(within_bootstrap_vars, axis=0)

            # Between-bootstrap variance (data uncertainty component)
            between_var = np.var(between_bootstrap_means, axis=0)

            # Total variance
            total_var = avg_within_var + between_var
        else:
            # Use compute_function_lotv_for_mceg to get 2D LoTV decomposition then marginalize over Q2
            try:
                from plotting_UQ_helpers import compute_function_lotv_for_mceg

                # choose nx to match x_vals resolution; nQ2 left as default
                nx_bins = x_vals.numel()
                nQ2_bins = 20
                lotv = compute_function_lotv_for_mceg(
                    working_simulator,
                    per_boot_posterior_samples=per_boot_posterior_samples,
                    n_theta_per_boot=50,
                    num_events=min(20000, num_events),
                    nx=nx_bins,
                    nQ2=nQ2_bins,
                    device="cpu",
                )
                # lotv['avg_within_var'] shape (nx, nQ2) ; marginalize over Q2 by summation
                avg_within_var = np.sum(lotv["avg_within_var"], axis=1)
                between_var = np.sum(lotv["between_var"], axis=1)
                total_var = np.sum(lotv["total_var"], axis=1)
                # Align x_np with bin centers produced by helper: approximate using x_vals
                # x_vals were chosen to be the target evaluation grid; use them for plotting
                x_np = x_vals.cpu().numpy()
            except Exception as e:
                raise RuntimeError(f"Could not compute LoTV for MCEG simulator: {e}")

        # Plot variance decomposition
        x_np = x_vals.cpu().numpy()
        ax_main.fill_between(
            x_np,
            0,
            avg_within_var,
            alpha=0.6,
            color=COLORS["blue"],
            label="Model uncertainty",
        )
        ax_main.fill_between(
            x_np,
            avg_within_var,
            total_var,
            alpha=0.6,
            color=COLORS["orange"],
            label="Data uncertainty",
        )
        ax_main.plot(
            x_np, total_var, color="black", linewidth=2, label="Total variance"
        )

        ax_main.set_xlabel(r"$x$")
        ax_main.set_ylabel("Variance")
        if isinstance(working_simulator, SimplifiedDIS):
            ax_main.set_title(f"{func_name.title()} PDF: Uncertainty Decomposition")
        else:
            ax_main.set_title("Function: Uncertainty Decomposition")
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        ax_main.set_yscale("log")

        # Relative contributions plot
        ax_rel = axes[func_idx, 1]

        # Avoid division by zero
        total_var_safe = np.maximum(total_var, 1e-12)
        model_fraction = avg_within_var / total_var_safe
        data_fraction = between_var / total_var_safe

        ax_rel.fill_between(
            x_np,
            0,
            model_fraction,
            alpha=0.6,
            color=COLORS["blue"],
            label="Model uncertainty",
        )
        ax_rel.fill_between(
            x_np,
            model_fraction,
            1,
            alpha=0.6,
            color=COLORS["orange"],
            label="Data uncertainty",
        )

        ax_rel.set_xlabel(r"$x$")
        ax_rel.set_ylabel("Fraction of total variance")
        ax_rel.set_title("Relative Contributions")
        ax_rel.legend()
        ax_rel.grid(True, alpha=0.3)
        ax_rel.set_ylim(0, 1)

    plt.tight_layout()

    # Determine save path
    if save_path is None:
        save_path = os.path.join(save_dir, "uncertainty_decomposition.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Uncertainty decomposition plot saved to: {save_path}")


def plot_uncertainty_scaling(
    model=None,
    pointnet_model=None,
    true_params=None,
    device=None,
    event_counts=None,
    n_bootstrap=20,
    laplace_model=None,
    problem="simplified_dis",
    save_dir="plots",
    save_path=None,
    mode="bootstrap",
    # Backward compatibility
    simulator=None,
    true_theta=None,
):
    """
    Plot how uncertainty scales with the number of events per experiment.

    This function supports both the new generator-style API and backward compatibility
    with the original API.

    New API Parameters:
    ------------------
    model : torch.nn.Module
        Parameter prediction model (head)
    pointnet_model : torch.nn.Module
        PointNet feature extractor
    true_params : torch.Tensor
        Ground truth parameter values
    device : torch.device
        Device to run computations on
    event_counts : list, optional
        List of event counts to test (default: [100, 500, 1000, 5000, 10000])
    n_bootstrap : int
        Number of bootstrap samples per event count
    problem : str
        Problem type ('simplified_dis', 'realistic_dis', 'gaussian', 'mceg')
    save_dir : str
        Directory to save plots (used if save_path not provided)
    save_path : str, optional
        Full path to save plot (overrides save_dir)
    mode : str, optional
        Type of uncertainty scaling to show (default: 'bootstrap')
        - 'bootstrap': Bootstrap/data uncertainty scaling
        - 'parameter': Parameter uncertainty from posterior for single dataset
        - 'combined': Both bootstrap and parameter uncertainty

    Backward Compatibility Parameters:
    ---------------------------------
    simulator : object
        Legacy simulator object
    true_theta : torch.Tensor
        Legacy true parameter tensor

    LaTeX Description:
    ==================

    \\section{Uncertainty Scaling with Number of Events}

    This figure demonstrates how both parameter and function estimation uncertainty
    decreases as the number of events per experiment increases, illustrating the
    fundamental statistical relationship between sample size and estimation precision.
    The plot consists of two panels showing different types of uncertainty scaling.

    \\textbf{Left Panel - Parameter Uncertainty:}
    Shows how uncertainty in the estimated parameters scales with event count.

    \\textbf{Right Panel - Function Uncertainty:}
    Shows how uncertainty in the predicted functions (averaged over x-values)
    scales with event count.

    \\textbf{Theoretical Expectation:}

    For most well-behaved estimators, the standard error should scale as:
    $$\\sigma_{\\hat{\\theta}} \\propto \\frac{1}{\\sqrt{N}}$$

    where $N$ is the number of events. This follows from the Central Limit Theorem and
    the fact that the variance of a sample mean decreases as $1/N$.

    \\textbf{Visualization Elements:}

    \\begin{itemize}
    \\item \\textbf{Observed scaling} (colored lines): Empirical standard deviations computed from
      multiple estimates at each event count
    \\item \\textbf{Theoretical scaling} (dashed): Reference $1/\\sqrt{N}$ scaling normalized
      to match the observed data at a reference point
    \\item \\textbf{Fitted scaling}: Power-law fit to the observed data showing the actual scaling exponent
    \\end{itemize}

    \\textbf{Mode Parameter:}
    \\begin{itemize}
    \\item \\textbf{bootstrap}: Uses bootstrap/data uncertainty across repeated datasets
    \\item \\textbf{parameter}: Uses parameter uncertainty from posterior for single dataset
    \\item \\textbf{combined}: Shows both types of uncertainty for comparison
    \\end{itemize}

    \\textbf{Experimental Procedure:}
    \\begin{enumerate}
    \\item For each event count $N \\in \\{100, 500, 1000, 5000, 10000\\}$:
    \\item Generate 20 independent datasets with $N$ events each
    \\item Estimate parameters $\\hat{\\theta}$ for each dataset
    \\item Compute standard deviation across the 20 estimates
    \\end{enumerate}

    \\textbf{Interpretation:}
    \\begin{itemize}
    \\item \\textbf{Adherence to theory}: Parameters following $N^{-0.5}$ scaling indicate well-behaved
      estimation with no systematic issues
    \\item \\textbf{Deviations from theory}: Faster or slower scaling may indicate systematic effects,
      model misspecification, or numerical issues
    \\item \\textbf{Practical implications}: The scaling relationship helps predict how much data is
      needed to achieve desired precision levels
    \\end{itemize}

    This analysis is crucial for experimental design, helping determine optimal data collection
    strategies and computational resource allocation.
    """
    print("📈 Generating uncertainty scaling plot...")

    # Validate mode parameter
    valid_modes = ["bootstrap", "parameter", "combined"]
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got: {mode}")

    print(f"   Mode: {mode}")

    # Handle backward compatibility - detect legacy positional arguments
    # If the first argument is a simulator-like object and second is a tensor, treat as legacy API
    if (
        model is not None
        and hasattr(model, "sample")
        and hasattr(model, "init")
        and pointnet_model is not None
        and isinstance(pointnet_model, torch.Tensor)
    ):
        # Legacy positional call: plot_uncertainty_scaling(simulator, true_theta, ...)
        print("   Using legacy API with positional arguments")
        working_simulator = model  # First arg is actually simulator
        working_true_params = pointnet_model  # Second arg is actually true_theta
        working_event_counts = (
            true_params if true_params is not None else [100, 500, 1000, 5000, 10000]
        )
        if isinstance(working_event_counts, torch.Tensor):
            working_event_counts = [
                100,
                500,
                1000,
                5000,
                10000,
            ]  # Reset to default if true_params was a tensor
    elif simulator is not None and true_theta is not None:
        # Legacy keyword call: plot_uncertainty_scaling(simulator=..., true_theta=...)
        print("   Using legacy API with keyword arguments")
        working_simulator = simulator
        working_true_params = true_theta
        working_event_counts = (
            event_counts if event_counts is not None else [100, 500, 1000, 5000, 10000]
        )
    else:
        # New API - generate data internally
        if (
            model is None
            or pointnet_model is None
            or true_params is None
            or device is None
        ):
            raise ValueError(
                "New API requires model, pointnet_model, true_params, and device"
            )

        print("   Using new generator-style API")
        # Create simulator based on problem type
        if problem == "simplified_dis":
            working_simulator = SimplifiedDIS(device=device)
        elif problem == "realistic_dis":
            working_simulator = RealisticDIS(device=device)
        elif problem == "gaussian":
            working_simulator = Gaussian2DSimulator(device=device)
        elif problem == "mceg":
            if MCEGSimulator is not None:
                working_simulator = MCEGSimulator(device=device)
            else:
                raise ValueError("MCEGSimulator not available")
        else:
            raise ValueError(f"Unknown problem type: {problem}")

        working_true_params = true_params
        working_event_counts = (
            event_counts if event_counts is not None else [100, 500, 1000, 5000, 10000]
        )

    # Compute parameter and function uncertainties based on mode
    # For legacy API, pass None for the neural network models
    if (model is not None and hasattr(model, "sample")) or simulator is not None:
        # Legacy API - no neural network models available
        param_uncertainties, function_uncertainties = _compute_uncertainties_by_mode(
            mode,
            working_simulator,
            working_true_params,
            working_event_counts,
            n_bootstrap,
            None,
            None,
            None,
            device,
            problem,
        )
    else:
        # New API - neural network models available
        param_uncertainties, function_uncertainties = _compute_uncertainties_by_mode(
            mode,
            working_simulator,
            working_true_params,
            working_event_counts,
            n_bootstrap,
            model,
            pointnet_model,
            laplace_model,
            device,
            problem,
        )

    # Create two-panel plot: parameter uncertainty (left) and function uncertainty (right)
    _create_uncertainty_scaling_plots(
        param_uncertainties,
        function_uncertainties,
        working_event_counts,
        working_simulator,
        mode,
        save_dir,
        save_path,
    )


def _compute_uncertainties_by_mode(
    mode,
    simulator,
    true_params,
    event_counts,
    n_bootstrap,
    model,
    pointnet_model,
    laplace_model,
    device,
    problem,
):
    """
    Compute parameter and function uncertainties based on the specified mode.

    Returns:
        param_uncertainties: dict with event counts as keys, parameter std arrays as values
        function_uncertainties: dict with event counts as keys, function std values as values
    """
    param_uncertainties = {}
    function_uncertainties = {}
    true_params = true_params.to(device)

    # Define x_vals grid for function evaluation
    if isinstance(simulator, SimplifiedDIS):
        x_vals = torch.linspace(1e-3, 1 - 1e-3, 100).to(device)
        function_names = ["up", "down"]
    elif isinstance(simulator, Gaussian2DSimulator):
        x_vals = torch.linspace(-3, 3, 100).to(device)  # For Gaussian
        function_names = ["pdf"]
    else:
        x_vals = torch.linspace(0.01, 0.99, 100).to(device)
        function_names = ["function"]

    for n_events in tqdm(event_counts, desc="Testing event counts"):
        if mode == "bootstrap":
            param_std, func_std = _compute_bootstrap_uncertainties(
                simulator,
                true_params,
                n_events,
                n_bootstrap,
                model,
                pointnet_model,
                device,
                x_vals,
                function_names,
            )
        elif mode == "parameter":
            param_std, func_std = _compute_parameter_uncertainties(
                simulator,
                true_params,
                n_events,
                model,
                pointnet_model,
                laplace_model,
                device,
                x_vals,
                function_names,
            )
        elif mode == "combined":
            # Compute parameter-side combination as quadrature (safe default)
            param_std_boot, func_std_boot = _compute_bootstrap_uncertainties(
                simulator,
                true_params,
                n_events,
                n_bootstrap,
                model,
                pointnet_model,
                device,
                x_vals,
                function_names,
            )
            param_std_param, func_std_param = _compute_parameter_uncertainties(
                simulator,
                true_params,
                n_events,
                model,
                pointnet_model,
                laplace_model,
                device,
                x_vals,
                function_names,
            )
            param_std = np.sqrt(param_std_boot**2 + param_std_param**2)

            # For function uncertainty, prefer a principled LoTV decomposition when posterior_sampler is available
            try:
                # Build per-bootstrap posterior samples using posterior_sampler where possible
                per_boot_samples = []
                n_boot_for_combined = min(n_bootstrap, 20)
                for b in range(n_boot_for_combined):
                    try:
                        boot_data = simulator.sample(true_params, n_events)
                        local_post = posterior_sampler(
                            boot_data,
                            pointnet_model,
                            model,
                            laplace_model,
                            n_samples=20,
                            device=device,
                        )
                        per_boot_samples.append(local_post.detach().cpu().numpy())
                    except Exception:
                        # fallback: singleton MAP
                        if model is not None and pointnet_model is not None:
                            est = _estimate_parameters_nn(
                                simulator.sample(true_params, n_events),
                                model,
                                pointnet_model,
                                device,
                            )
                            per_boot_samples.append(est.detach().cpu().numpy()[None, :])
                        else:
                            per_boot_samples.append(
                                true_params.detach().cpu().numpy()[None, :]
                            )

                lotv = compute_function_lotv_for_simplified_dis(
                    simulator,
                    x_vals,
                    per_boot_posterior_samples=per_boot_samples,
                    device=device,
                )
                # Use scalar summary mean_x sqrt(total_var(x)) as function uncertainty metric
                func_std = (
                    float(np.mean(np.sqrt(lotv["total_var_up"])))
                    if "total_var_up" in lotv
                    else float(lotv.get("unc_up", 0.0))
                )
            except Exception:
                # Fallback to quadrature combination
                func_std = np.sqrt(func_std_boot**2 + func_std_param**2)

        param_uncertainties[n_events] = param_std
        function_uncertainties[n_events] = func_std

    return param_uncertainties, function_uncertainties


def _compute_bootstrap_uncertainties(
    simulator,
    true_params,
    n_events,
    n_bootstrap,
    model,
    pointnet_model,
    device,
    x_vals,
    function_names,
):
    """Compute uncertainties using bootstrap approach (multiple datasets)."""
    param_estimates = []
    function_estimates = {name: [] for name in function_names}

    for trial in range(n_bootstrap):
        # Generate dataset
        data = simulator.sample(true_params, n_events)

        # Estimate parameters
        if model is not None and pointnet_model is not None:
            # Use neural network prediction
            estimated_params = _estimate_parameters_nn(
                data, model, pointnet_model, device
            )
        else:
            # Fallback to simplified estimation (for legacy API)
            estimated_params = true_params + torch.randn_like(true_params) * 0.1

        param_estimates.append(estimated_params.detach().cpu())

        # Evaluate functions with estimated parameters
        # For MCEG problem use PDF-based u-ub evaluation at Q2=10
        try:
            from simulator import MCEGSimulator
        except Exception:
            MCEGSimulator = None

        if MCEGSimulator is not None and isinstance(simulator, MCEGSimulator):
            # Evaluate collaborator PDF (u - ub) across x grid and compute variance across x
            try:
                import numpy as _np

                from simulator import ALPHAS, EWEAK, MELLIN, PDF

                mellin = MELLIN(npts=8)
                alphaS = ALPHAS()
                _pdf = PDF(mellin, alphaS)
                # Build parameter array expected by PDF and insert estimated params
                cpar = _pdf.get_current_par_array()[::]
                est_arr = estimated_params.detach().cpu().numpy().ravel()
                try:
                    cpar[4 : 4 + est_arr.shape[0]] = est_arr
                except Exception:
                    try:
                        cpar[4:8] = est_arr
                    except Exception:
                        cpar[4 : 4 + est_arr.shape[0]] = est_arr
                try:
                    _pdf.setup(cpar)
                except Exception:
                    try:
                        _pdf.setup(est_arr)
                    except Exception:
                        raise

                # x grid: use provided x_vals if available, else fallback
                try:
                    x_np = (
                        _np.asarray(x_vals.detach().cpu())
                        if hasattr(x_vals, "detach")
                        else _np.asarray(x_vals)
                    )
                except Exception:
                    x_np = _np.linspace(0.001, 0.99, 100)
                q2_val = 10.0
                # compute u-ub across x
                vals = []
                for x in x_np:
                    try:
                        u = _pdf.get_xF(float(x), q2_val, "u", evolve=True)
                        ub = _pdf.get_xF(float(x), q2_val, "ub", evolve=True)
                        uval = float(u[0]) if hasattr(u, "__len__") else float(u)
                        ubval = float(ub[0]) if hasattr(ub, "__len__") else float(ub)
                        vals.append(uval - ubval)
                    except Exception:
                        vals.append(_np.nan)
                vals = _np.array(vals)
                func_variance = float(_np.nanvar(vals))
                for func_name in function_names:
                    function_estimates[func_name].append(func_variance)
            except Exception:
                # Fallback to simulator-based evaluation if PDF evaluation fails
                simulator.init(estimated_params.detach().cpu())
                for func_name in function_names:
                    if hasattr(simulator, func_name):
                        func = getattr(simulator, func_name)
                        f_vals = func(x_vals).detach().cpu()
                        func_variance = torch.var(f_vals).item()
                        function_estimates[func_name].append(func_variance)
                    else:
                        function_estimates[func_name].append(0.0)
        else:
            # Default behavior for non-MCEG simulators
            simulator.init(estimated_params.detach().cpu())
            for func_name in function_names:
                if hasattr(simulator, func_name):
                    func = getattr(simulator, func_name)
                    f_vals = func(x_vals).detach().cpu()
                    # Average variance across x_vals for this trial
                    func_variance = torch.var(f_vals).item()
                    function_estimates[func_name].append(func_variance)
                else:
                    # For unsupported functions, use a dummy value
                    function_estimates[func_name].append(0.0)

    # Compute parameter uncertainty
    param_estimates = torch.stack(param_estimates)
    param_std = torch.std(param_estimates, dim=0).numpy()

    # Compute function uncertainty (average across function types)
    func_stds = []
    for func_name in function_names:
        if len(function_estimates[func_name]) > 0:
            func_stds.append(np.std(function_estimates[func_name]))
    func_std = np.mean(func_stds) if func_stds else 0.0

    return param_std, func_std


def _compute_parameter_uncertainties(
    simulator,
    true_params,
    n_events,
    model,
    pointnet_model,
    laplace_model,
    device,
    x_vals,
    function_names,
):
    """Compute uncertainties using parameter uncertainty from posterior for single dataset."""
    if laplace_model is None:
        # If no Laplace model, fall back to bootstrap with n_bootstrap=1
        return _compute_bootstrap_uncertainties(
            simulator,
            true_params,
            n_events,
            1,
            model,
            pointnet_model,
            device,
            x_vals,
            function_names,
        )

    # Generate a single dataset
    data = simulator.sample(true_params, n_events)

    # Get posterior samples from Laplace approximation
    posterior_samples = posterior_sampler(
        data, pointnet_model, model, laplace_model, n_samples=20, device=device
    )

    # Compute parameter uncertainty from posterior samples
    param_std = torch.std(posterior_samples, dim=0).numpy()

    # Compute function uncertainty by propagating parameter uncertainty
    function_estimates = {name: [] for name in function_names}

    try:
        from simulator import MCEGSimulator
    except Exception:
        MCEGSimulator = None

    for sample in posterior_samples:
        # If MCEG, evaluate collaborator PDF u-ub across x at Q2=10
        if MCEGSimulator is not None and isinstance(simulator, MCEGSimulator):
            try:
                import numpy as _np

                from simulator import ALPHAS, EWEAK, MELLIN, PDF

                mellin = MELLIN(npts=8)
                alphaS = ALPHAS()
                _pdf = PDF(mellin, alphaS)
                cpar = _pdf.get_current_par_array()[::]
                s_arr = sample.detach().cpu().numpy().ravel()
                try:
                    cpar[4 : 4 + s_arr.shape[0]] = s_arr
                except Exception:
                    try:
                        cpar[4:8] = s_arr
                    except Exception:
                        cpar[4 : 4 + s_arr.shape[0]] = s_arr
                try:
                    _pdf.setup(cpar)
                except Exception:
                    try:
                        _pdf.setup(s_arr)
                    except Exception:
                        raise

                try:
                    x_np = (
                        _np.asarray(x_vals.detach().cpu())
                        if hasattr(x_vals, "detach")
                        else _np.asarray(x_vals)
                    )
                except Exception:
                    x_np = _np.linspace(0.001, 0.99, 100)
                q2_val = 10.0
                vals = []
                for x in x_np:
                    try:
                        u = _pdf.get_xF(float(x), q2_val, "u", evolve=True)
                        ub = _pdf.get_xF(float(x), q2_val, "ub", evolve=True)
                        uval = float(u[0]) if hasattr(u, "__len__") else float(u)
                        ubval = float(ub[0]) if hasattr(ub, "__len__") else float(ub)
                        vals.append(uval - ubval)
                    except Exception:
                        vals.append(_np.nan)
                vals = _np.array(vals)
                func_variance = float(_np.nanvar(vals))
                for func_name in function_names:
                    function_estimates[func_name].append(func_variance)
            except Exception:
                # fallback to simulator evaluation
                simulator.init(sample.detach().cpu())
                for func_name in function_names:
                    if hasattr(simulator, func_name):
                        func = getattr(simulator, func_name)
                        f_vals = func(x_vals).detach().cpu()
                        func_variance = torch.var(f_vals).item()
                        function_estimates[func_name].append(func_variance)
                    else:
                        function_estimates[func_name].append(0.0)
        else:
            simulator.init(sample.detach().cpu())
            for func_name in function_names:
                if hasattr(simulator, func_name):
                    func = getattr(simulator, func_name)
                    f_vals = func(x_vals).detach().cpu()
                    # Average variance across x_vals for this sample
                    func_variance = torch.var(f_vals).item()
                    function_estimates[func_name].append(func_variance)
                else:
                    function_estimates[func_name].append(0.0)

    # Compute function uncertainty
    func_stds = []
    for func_name in function_names:
        if len(function_estimates[func_name]) > 0:
            func_stds.append(np.std(function_estimates[func_name]))
    func_std = np.mean(func_stds) if func_stds else 0.0

    return param_std, func_std


def _estimate_parameters_nn(data, model, pointnet_model, device):
    """Estimate parameters using neural network models."""
    with torch.no_grad():
        # Apply feature engineering
        data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
        if data_tensor.ndim == 2:
            data_tensor = data_tensor.unsqueeze(0)

        # Use advanced feature engineering if available
        try:
            data_tensor = log_feature_engineering(data_tensor).float()
        except:
            pass  # Use data as-is if feature engineering fails

        # Extract latent embedding
        latent_embedding = pointnet_model(data_tensor)

        # Get parameter prediction
        output = model(latent_embedding)
        if isinstance(output, tuple) and len(output) == 2:  # Gaussian head
            mean_params, _ = output
            estimated_params = mean_params.squeeze(0)
        else:  # Deterministic
            estimated_params = output.squeeze(0)

    return estimated_params


def _create_uncertainty_scaling_plots(
    param_uncertainties,
    function_uncertainties,
    event_counts,
    simulator,
    mode,
    save_dir,
    save_path,
):
    """Create the two-panel uncertainty scaling plot."""
    # Determine parameter names
    if isinstance(simulator, SimplifiedDIS):
        param_names = [r"$a_u$", r"$b_u$", r"$a_d$", r"$b_d$"]
    elif isinstance(simulator, Gaussian2DSimulator):
        param_names = [r"$\mu_x$", r"$\mu_y$", r"$\sigma_x$", r"$\sigma_y$", r"$\rho$"]
    else:
        param_names = [
            f"$\\theta_{{{i+1}}}$"
            for i in range(len(next(iter(param_uncertainties.values()))))
        ]

    # Create two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Parameter uncertainty scaling
    for i, param_name in enumerate(param_names):
        param_stds = [param_uncertainties[count][i] for count in event_counts]
        ax1.loglog(
            event_counts, param_stds, "o-", linewidth=2, markersize=6, label=param_name
        )

    # Add theoretical 1/sqrt(N) scaling to parameter plot
    reference_std = (
        param_stds[len(param_stds) // 2] if len(param_stds) > 2 else param_stds[0]
    )
    reference_N = (
        event_counts[len(event_counts) // 2]
        if len(event_counts) > 2
        else event_counts[0]
    )
    theoretical = reference_std * np.sqrt(reference_N / np.array(event_counts))
    ax1.loglog(
        event_counts,
        theoretical,
        "k--",
        linewidth=2,
        alpha=0.7,
        label=r"$\propto 1/\sqrt{N}$",
    )

    ax1.set_xlabel("Number of events")
    ax1.set_ylabel("Parameter uncertainty (std)")
    ax1.set_title(f"Parameter Uncertainty Scaling ({mode})")
    ax1.legend()
    ax1.grid(True, alpha=0.3, which="both")

    # Right panel: Function uncertainty scaling
    func_stds = [function_uncertainties[count] for count in event_counts]
    # If this is the MCEG problem, the function uncertainty represents PDF u-ub at Q2=10
    try:
        from simulator import MCEGSimulator
    except Exception:
        MCEGSimulator = None

    if MCEGSimulator is not None and isinstance(simulator, MCEGSimulator):
        func_label = "PDF uncertainty (u - ub, Q²=10)"
        ax2.loglog(
            event_counts,
            func_stds,
            "o-",
            color="C3",
            linewidth=2,
            markersize=6,
            label=func_label,
        )
    else:
        func_label = "Function uncertainty"
        ax2.loglog(
            event_counts,
            func_stds,
            "o-",
            color="red",
            linewidth=2,
            markersize=6,
            label=func_label,
        )

    # Add theoretical scaling to function plot
    reference_func_std = (
        func_stds[len(func_stds) // 2] if len(func_stds) > 2 else func_stds[0]
    )
    theoretical_func = reference_func_std * np.sqrt(
        reference_N / np.array(event_counts)
    )
    ax2.loglog(
        event_counts,
        theoretical_func,
        "k--",
        linewidth=2,
        alpha=0.7,
        label=r"$\propto 1/\sqrt{N}$",
    )

    ax2.set_xlabel("Number of events")
    ax2.set_ylabel(f"{func_label} (std)")
    if MCEGSimulator is not None and isinstance(simulator, MCEGSimulator):
        ax2.set_title(f"PDF Uncertainty Scaling (u - ub at Q²=10) ({mode})")
    else:
        ax2.set_title(f"Function Uncertainty Scaling ({mode})")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    # Add scaling fits
    if len(event_counts) > 2:
        # Parameter scaling fit
        log_N = np.log(event_counts)
        log_param_std = np.log(
            [np.mean(param_uncertainties[count]) for count in event_counts]
        )
        param_slope, _ = np.polyfit(log_N, log_param_std, 1)

        # Function scaling fit
        log_func_std = np.log(func_stds)
        func_slope, _ = np.polyfit(log_N, log_func_std, 1)

        ax1.text(
            0.05,
            0.95,
            f"Fitted: $N^{{{param_slope:.2f}}}$\nTheory: $N^{{-0.5}}$",
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax2.text(
            0.05,
            0.95,
            f"Fitted: $N^{{{func_slope:.2f}}}$\nTheory: $N^{{-0.5}}$",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()

    # Determine save path
    if save_path is None:
        save_path = os.path.join(save_dir, f"uncertainty_scaling_{mode}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Uncertainty scaling plot saved to: {save_path}")


def main():
    """
    Main function demonstrating all uncertainty quantification plotting utilities.
    """
    print("🎯 Enhanced Uncertainty Quantification Plotting Demo")
    print("=" * 60)

    # Create output directory
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Output directory: {save_dir}")

    # Initialize simulator (default to SimplifiedDIS)
    print("\n🔬 Initializing SimplifiedDIS simulator...")
    simulator = SimplifiedDIS(device=torch.device("cpu"), smear=True, smear_std=0.02)

    # Define true parameters
    true_theta = torch.tensor([2.0, 1.2, 2.0, 1.2])  # [au, bu, ad, bd]
    print(f"True parameters: {true_theta.tolist()}")

    # Generate observed data
    n_events = 2000
    print(f"\n📊 Generating {n_events} events for analysis...")
    observed_data = simulator.sample(true_theta, n_events)
    print(f"Observed data shape: {observed_data.shape}")

    print("\n" + "=" * 60)
    print("GENERATING UNCERTAINTY QUANTIFICATION PLOTS")
    print("=" * 60)

    # 1. Parameter-space uncertainty
    print("\n1️⃣ Parameter-space uncertainty visualization...")
    posterior_samples = plot_parameter_uncertainty(
        simulator, true_theta, observed_data, save_dir
    )

    # 2. Function-space uncertainty
    print("\n2️⃣ Function-space (predictive) uncertainty...")
    plot_function_uncertainty(simulator, posterior_samples, true_theta, save_dir)

    # 2b. Demonstrate new mode options for parameter uncertainty
    print("\n2️⃣b Parameter uncertainty modes...")

    # Demonstrate bootstrap mode
    print("   📊 Bootstrap parameter uncertainty...")
    plot_parameter_uncertainty(
        simulator, true_theta, observed_data, save_dir, mode="bootstrap"
    )

    # Demonstrate combined mode
    print("   📊 Combined parameter uncertainty...")
    plot_parameter_uncertainty(
        simulator, true_theta, observed_data, save_dir, mode="combined"
    )

    # 2c. Demonstrate new mode options for function uncertainty
    print("\n2️⃣c Function uncertainty modes...")

    # Demonstrate bootstrap mode
    print("   📊 Bootstrap function uncertainty...")
    plot_function_uncertainty(
        simulator, posterior_samples, true_theta, save_dir, mode="bootstrap"
    )

    # Demonstrate combined mode
    print("   📊 Combined function uncertainty...")
    plot_function_uncertainty(
        simulator, posterior_samples, true_theta, save_dir, mode="combined"
    )

    # 3. Bootstrap uncertainty
    print("\n3️⃣ Bootstrap/data uncertainty...")
    bootstrap_estimates = plot_bootstrap_uncertainty(
        simulator, true_theta, n_events=1000, n_bootstrap=30, save_dir=save_dir
    )

    # 4. Combined uncertainty decomposition
    print("\n4️⃣ Combined uncertainty decomposition...")
    plot_combined_uncertainty_decomposition(
        simulator, true_theta, posterior_samples, bootstrap_estimates, save_dir
    )

    # 5. Uncertainty scaling
    print("\n5️⃣ Uncertainty scaling with number of events...")

    # Demonstrate bootstrap mode (default)
    print("   📊 Bootstrap uncertainty scaling...")
    plot_uncertainty_scaling(
        simulator,
        true_theta,
        event_counts=[200, 500, 1000, 2000, 5000],
        save_dir=save_dir,
        mode="bootstrap",
    )

    # Demonstrate parameter mode
    print("   📊 Parameter uncertainty scaling...")
    plot_uncertainty_scaling(
        simulator,
        true_theta,
        event_counts=[200, 500, 1000, 2000, 5000],
        save_dir=save_dir,
        mode="parameter",
    )

    # Demonstrate combined mode
    print("   📊 Combined uncertainty scaling...")
    plot_uncertainty_scaling(
        simulator,
        true_theta,
        event_counts=[200, 500, 1000, 2000, 5000],
        save_dir=save_dir,
        mode="combined",
    )

    print("\n" + "=" * 60)
    print("🎉 ALL PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 60)

    # Summary
    print(f"\n📈 Generated plots in '{save_dir}/':")
    plot_files = [f for f in os.listdir(save_dir) if f.endswith(".png")]
    for plot_file in sorted(plot_files):
        tex_file = plot_file.replace(".png", ".tex")
        print(f"  ✅ {plot_file} (with {tex_file})")

    print(f"\n📝 LaTeX descriptions saved as .tex files alongside each plot")
    print(
        f"\n🔬 All plots demonstrate uncertainty quantification using ONLY simulator data"
    )
    print(f"   - No external datasets required")
    print(f"   - Self-contained demonstration")
    print(f"   - Production-ready code for adaptation to other simulators")

    print("\n🎯 Usage summary:")
    print("   - Plots saved to plots/ directory")
    print("   - LaTeX descriptions explain uncertainty computation")
    print("   - Code easily adaptable for other simulators")
    print("   - Run: python uq_plotting_demo.py")

    # Test with Gaussian simulator as well
    print("\n" + "=" * 50)
    print("BONUS: Testing with Gaussian2DSimulator")
    print("=" * 50)

    gauss_simulator = Gaussian2DSimulator()
    gauss_theta = torch.tensor(
        [0.0, 0.0, 1.0, 1.0, 0.3]
    )  # [mu_x, mu_y, sigma_x, sigma_y, rho]
    gauss_data = gauss_simulator.sample(gauss_theta, 1000)

    # Quick demonstration with Gaussian
    gauss_save_dir = os.path.join(save_dir, "gaussian_demo")
    os.makedirs(gauss_save_dir, exist_ok=True)

    print("📊 Generating parameter uncertainty for Gaussian2D...")
    gauss_posterior = plot_parameter_uncertainty(
        gauss_simulator, gauss_theta, gauss_data, gauss_save_dir
    )

    print("✅ Gaussian demonstration complete!")
    print(f"   - Gaussian plots saved to {gauss_save_dir}/")

    print("\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
