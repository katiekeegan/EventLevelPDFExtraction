# -*- coding: utf-8 -*-
"""
Enhanced Plotting Utilities for Uncertainty Quantification in PDF Parameter Inference

This module provides publication-ready plotting functions for visualizing uncertainty
in PDF parameter inference, supporting multiple problem types including simplified_dis,
and mceg4dis. All plots are designed to be beautiful, clear, and suitable
for publication.

Problem Type Support:
====================

- simplified_dis: 1D PDF inputs (x only) with up/down quark distributions
- mceg4dis: 2D PDF inputs (x, Q2) using Monte Carlo Event Generator for DIS
  * Full 2D support for visualizing PDF uncertainty over both x and Q2 dimensions
  * Enhanced plotting functions handle the 2D nature of mceg4dis inputs
  * Compatible with existing mceg simulator while providing improved visualization

New Functions:
==============

1. plot_parameter_error_histogram:
   Creates histograms showing parameter errors across multiple parameter choices.
   Features both absolute and relative error analysis with statistical annotations.

2. plot_function_error_histogram:
   Creates histograms of average entrywise function value errors with probability
    # end of plot_function_posterior_from_sbi_samples
# Parameter distributions with confidence intervals
plot_params_distribution_single(
    model, pointnet_model, true_params, device,
    laplace_model=laplace_model,  # For analytic uncertainty
    save_path="param_distributions.png"
)

# PDF uncertainty with multiple confidence levels
plot_PDF_distribution_single(
    model, pointnet_model, true_params, device,
    laplace_model=laplace_model,
    save_dir="./pdf_plots/"
)

# NEW: Automated parameter error benchmarking (standalone function)
generate_parameter_error_histogram(
    model, pointnet_model, device,

    'olive': '#bcbd22',
    'cyan': '#17becf',
    'dark_blue': '#0c2c84',
    'dark_orange': '#cc5500',
    'dark_green': '#006400'
}
"""

import glob
# utils_laplace.py
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from laplace.laplace import Laplace

from datasets import *
from models import *
from plotting_UQ_utils import *
from simulator import *

np.random.seed(42)
torch.manual_seed(42)

# Set up matplotlib for high-quality plots
plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
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
# Define colorblind-friendly color palette
COLORBLIND_COLORS = {
    # Okabe-Ito palette: optimized for colorblind accessibility
    # Works for red-green colorblindness, blue-yellow colorblindness, and grayscale
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "green": "#029E73",
    "red": "#CC78BC",
    "yellow": "#CA9161",
    "gray": "#949494",
    "black": "#ECE133",
    # Extended palette for more colors when needed
    "sky_blue": "#56B4E9",
    "blueish_green": "#009E73",
    "light_blue": "#86C4D0",
    "light_orange": "#F0B555",
    "light_gray": "#D0D0D0",
    # Additional aliases for compatibility
    "purple": "#CC78BC",
    "brown": "#CA9161",
    "pink": "#CC78BC",
    "olive": "#029E73",
    "cyan": "#0173B2",
    "dark_blue": "#0173B2",
    "dark_orange": "#DE8F05",
    "dark_green": "#029E73",
}
# Enhanced color schemes for specific plot types
UNCERTAINTY_COLORS = {
    "model": COLORBLIND_COLORS["blue"],
    "data": COLORBLIND_COLORS["orange"],
    "combined": COLORBLIND_COLORS["purple"],
    "true": COLORBLIND_COLORS["dark_green"],
    "predicted": COLORBLIND_COLORS["red"],
}

PDF_FUNCTION_COLORS = {
    "up": COLORBLIND_COLORS["blue"],
    "down": COLORBLIND_COLORS["orange"],
    "q": COLORBLIND_COLORS["green"],
}


# Import optional dependencies with fallbacks
try:
    import umap.umap_ as umap
except (ImportError, AttributeError):
    umap = None

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None
    PCA = None


# Import simulator and other modules only when needed
def get_simulator_module():
    try:
        from simulator import MCEGSimulator,  SimplifiedDIS

        return SimplifiedDIS, MCEGSimulator
    except ImportError:
        return None, None, None


def get_log_feature_engineering():
    try:
        from utils import log_feature_engineering

        return log_feature_engineering
    except ImportError:
        return lambda x: x  # fallback identity function


# ===========================
# Robust Precomputed Data Loading
# ========


def generate_precomputed_data_if_needed(
    problem,
    num_samples,
    num_events,
    n_repeat=2,
    output_dir="precomputed_data",
    ignore_nr=True,
):
    """
    Check if precomputed data exists for the given parameters, and generate it if not found.

    When ignore_nr is True, the function will accept any precomputed file that matches
    the problem, num_samples and num_events, ignoring the stored nr value. If multiple
    matching files exist it picks the one with the largest nr.

    Args:
        problem: Problem type ('simplified_dis', 'mceg')
        num_samples: Number of theta parameter samples
        num_events: Number of events per simulation
        n_repeat: Number of repeated simulations per theta (used for generation when needed)
        output_dir: Directory where precomputed data should be stored
        ignore_nr: If True, accept files with any nr that match (problem, ns, ne)

    Returns:
        str: Path to the data directory

    Raises:
        RuntimeError: If precomputed data support is not available or generation fails
    """
    import glob
    import os
    import re

    print("🔍 PRECOMPUTED DATA DIAGNOSTIC:")
    print(
        f"   Looking for problem: '{problem}' with ns={num_samples}, ne={num_events}, nr={n_repeat}"
    )
    print(f"   Data directory: '{output_dir}'")
    print(f"   ignore_nr = {ignore_nr}")

    # Check precomputed data availability
    try:
        from precomputed_datasets import PrecomputedDataset

        PRECOMPUTED_AVAILABLE = True
    except ImportError:
        PRECOMPUTED_AVAILABLE = False

    if not PRECOMPUTED_AVAILABLE:
        print(
            "   ✗ Precomputed data support not available. Please check precomputed_datasets.py"
        )
        raise RuntimeError(
            "Precomputed data support not available. Please check precomputed_datasets.py"
        )

    # Ensure output directory exists
    if not os.path.isdir(output_dir):
        print(f"   Note: output_dir '{output_dir}' does not exist yet.")
    # Exact filename we would accept if strict matching is desired
    expected_filename = f"{problem}_ns{num_samples}_ne{num_events}_nr{n_repeat}.npz"
    exact_file_path = os.path.join(output_dir, expected_filename)
    print(f"   Required exact file (if strict): '{exact_file_path}'")

    # If strict matching requested (ignore_nr==False) check exact path first
    if not ignore_nr:
        if os.path.exists(exact_file_path):
            print(f"   ✓ Found exact matching precomputed data: {exact_file_path}")
            return output_dir
        else:
            print(f"   ✗ Exact matching precomputed data not found.")
    else:
        # ignore_nr == True: look for any file matching problem_ns<num>_ne<num>_nr*.npz
        pattern = os.path.join(
            output_dir, f"{problem}_ns{num_samples}_ne{num_events}_nr*.npz"
        )
        candidates = glob.glob(pattern)
        if candidates:
            # Prefer the candidate with the largest nr (safer: uses the most repeated data)
            nr_re = re.compile(r"_nr(\d+)\.npz$")
            best = None
            best_nr = -1
            for c in candidates:
                m = nr_re.search(c)
                if m:
                    nr_val = int(m.group(1))
                else:
                    nr_val = 0
                if nr_val > best_nr:
                    best = c
                    best_nr = nr_val
            print(
                f"   ✓ Found precomputed data matching ns/ne (ignoring nr): {best} (nr={best_nr})"
            )
            return output_dir
        else:
            print(
                "   ✗ No precomputed files found that match problem/ns/ne (ignoring nr)."
            )

    # If we reach here: no acceptable precomputed data found — generate it
    print(
        f"📊 Precomputed data not found for {problem} with ns={num_samples}, ne={num_events} (nr ignored={ignore_nr})"
    )
    print("🚀 Generating precomputed data automatically...")

    try:
        # Import and run the data generation function
        import torch

        from generate_precomputed_data import generate_data_for_problem

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Generating data on device: {device}")

        filepath = generate_data_for_problem(
            problem, num_samples, num_events, n_repeat, device, output_dir
        )
        print(f"✅ Successfully generated precomputed data: {filepath}")
        return output_dir

    except Exception as e:
        print(f"❌ Error generating precomputed data: {e}")
        print("⚠️  Falling back to simulation-based plotting (less reproducible)")
        raise RuntimeError(f"Failed to generate precomputed data: {e}")

"""
MAJOR UPDATE: Function-Level Uncertainty Quantification

This module now focuses uncertainty quantification on the PREDICTED FUNCTIONS f(x) 
rather than just the model parameters θ. This provides more interpretable uncertainty
for PDF predictions in physics applications.

KEY CHANGES:
1. plot_combined_uncertainty_PDF_distribution(): Now computes pointwise uncertainty
   over PDF functions u(x), d(x), q(x) by evaluating f(x|θ) for multiple θ samples
   and aggregating statistics at each x-point.

2. Pointwise uncertainty combination: total_variance(x) = var_bootstrap(x) + var_laplace(x)
   - Data uncertainty: variance across bootstrap samples of function values at each x
   - Model uncertainty: variance from Laplace parameter posterior propagated to functions
   - Combined pointwise in function space, not parameter space

3. Output files emphasize function uncertainty:
   - function_uncertainty_pdf_{name}.png: PDF plots with function-level uncertainty bands  
   - function_uncertainty_breakdown_{name}.txt: Pointwise statistics for each x
   - function_uncertainty_methodology.txt: Detailed explanation of the approach

4. More interpretable uncertainty bands that show prediction uncertainty of the PDFs
   themselves, which is what practitioners care about for physics applications.

Why this change: Parameter uncertainty θ ± σ_θ is less interpretable than function
uncertainty f(x) ± σ_f(x). The new approach provides uncertainty bands directly on
the predicted PDF curves, making it easier to assess prediction quality at specific
x-values and understand the confidence in different regions of the PDF.
"""


def get_analytic_uncertainty(model, latent_embedding, laplace_model=None):
    """
    Return (mean_params, std_params) for the model outputs given a Laplace posterior.
    Works across older laplace-torch builds by trying several APIs.

    FIXED: Enhanced with diagnostic information to track Laplace success/failure
    """
    device = latent_embedding.device
    model.eval()

    if laplace_model is not None:
        # Add diagnostic information about Laplace model
        # print(f"🔧 [FIX] get_analytic_uncertainty: Laplace model provided")
        # print(f"🔧 [FIX] Laplace model type: {type(laplace_model)}")
        # print(f"🔧 [FIX] Latent embedding shape: {latent_embedding.shape}")

        with torch.no_grad():
            # --- Path 1: predictive_distribution(x) -> distribution with .loc and .scale
            pred_dist_fn = getattr(laplace_model, "predictive_distribution", None)
            if callable(pred_dist_fn):
                try:
                    dist = pred_dist_fn(latent_embedding)
                    mean_params = dist.loc
                    std_params = dist.scale
                    # print(f"✅ [FIXED] Path 1 SUCCESS - Laplace uncertainty working!")
                    return mean_params.cpu(), std_params.cpu()
                except Exception as e:
                    print(f"❌ [DEBUG] Path 1 FAILED: {type(e).__name__}: {e}")

            # --- Path 2: calling the object sometimes returns (mean, var)
            try:
                out = laplace_model(latent_embedding, joint=False)
                if isinstance(out, tuple) and len(out) == 2:
                    pred_mean, pred_var = out
                    if pred_var.dim() == 3:
                        pred_std = torch.sqrt(
                            torch.diagonal(pred_var, dim1=-2, dim2=-1)
                        )
                    else:
                        pred_std = torch.sqrt(pred_var.clamp_min(0))
                    # print(f"✅ [FIXED] Path 2 SUCCESS - Laplace uncertainty working!")
                    return pred_mean.cpu(), pred_std.cpu()
                else:
                    pass
                    # print(f"❌ [DEBUG] Path 2 - output not a 2-tuple: {type(out)}")
            except Exception as e:
                pass  # print(f"❌ [DEBUG] Path 2 FAILED: {type(e).__name__}: {e}")

            # --- Path 3: predict(..., pred_type='glm', link_approx='mc')
            predict_fn = getattr(laplace_model, "predict", None)
            if callable(predict_fn):
                try:
                    pred = predict_fn(
                        latent_embedding,
                        pred_type="glm",
                        link_approx="mc",
                        n_samples=200,
                    )
                    if isinstance(pred, tuple) and len(pred) == 2:
                        mean, var = pred
                        std = torch.sqrt(var.clamp_min(0))
                        # print(f"✅ [FIXED] Path 3 SUCCESS - Laplace uncertainty working!")
                        return mean.cpu(), std.cpu()
                    if hasattr(pred, "loc") and hasattr(pred, "scale"):
                        # print(f"✅ [FIXED] Path 3 SUCCESS - Laplace uncertainty working!")
                        return pred.loc.cpu(), pred.scale.cpu()
                    else:
                        print(
                            f"❌ [DEBUG] Path 3 - predict output lacks expected attributes"
                        )
                except Exception as e:
                    print(f"❌ [DEBUG] Path 3 FAILED: {type(e).__name__}: {e}")

        print(f"⚠️  [DEBUG] ALL LAPLACE PATHS FAILED - falling back to standard model")
    else:
        print(f"⚠️  [DEBUG] No Laplace model provided - using standard model only")

    # --- Fallbacks (no Laplace available) ---
    with torch.no_grad():
        output = model(latent_embedding.to(device))
    if isinstance(output, tuple) and len(output) == 2:  # Gaussian head
        means, logvars = output
        stds = torch.exp(0.5 * logvars)
        return means.cpu(), stds.cpu()
    elif isinstance(output, tuple) and len(output) == 3:  # Multimodal head
        means, logvars, weights = output
        b = means.shape[0]
        idx = torch.argmax(weights, dim=-1)
        sel_means = means[torch.arange(b), idx]
        sel_stds = torch.exp(0.5 * logvars[torch.arange(b), idx])
        return sel_means.cpu(), sel_stds.cpu()
    else:  # deterministic
        pred_mean = output
        pred_std = torch.zeros_like(pred_mean)
        return pred_mean.cpu(), pred_std.cpu()


def get_gaussian_samples(model, latent_embedding, n_samples=100, laplace_model=None):
    """
    Generate parameter samples from an approximate Gaussian posterior.

    This helper produces θ samples for workflows that compute quantiles,
    histograms, or Monte Carlo-based function bands. When a Laplace model is
    provided, it first obtains analytic mean/std via `get_analytic_uncertainty`
    and then draws samples from N(mean, std^2); otherwise it falls back to the
    model head's outputs.

    Args:
        model: Neural network model (head)
        latent_embedding: Input latent embedding tensor
        n_samples: Number of parameter samples to draw
        laplace_model: Fitted Laplace approximation object (optional)

    Returns:
        torch.Tensor: Generated samples [n_samples, param_dim]

    Note:
        Prefer `get_analytic_uncertainty` when you only need mean/std without
        sampling overhead. Keep using this function when your code expects
        explicit samples for visualization or quantile computation.
    """
    # For backward compatibility, convert analytic uncertainty to samples
    mean_params, std_params = get_analytic_uncertainty(
        model, latent_embedding, laplace_model
    )

    # Generate samples from analytic Gaussian distribution
    device = latent_embedding.device
    batch_size, param_dim = mean_params.shape

    # Generate n_samples for each batch element
    samples = []
    for i in range(n_samples):
        # Sample from Gaussian with analytic mean and std
        sample = torch.randn_like(mean_params) * std_params + mean_params
        samples.append(sample.squeeze(0) if batch_size == 1 else sample[0])

    return torch.stack(samples)


def plot_params_distribution_single(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,  # Kept for backward compatibility but ignored when using analytic uncertainty
    laplace_model=None,
    compare_with_sbi=False,
    sbi_posteriors=None,
    sbi_labels=None,
    save_path="Dist.png",
    problem="simplified_dis",
    lotv_param_samples=None,
    lotv_label="LoTV",
):
    """
    Create publication-ready parameter distribution plots with analytic Laplace uncertainty propagation.

    This function generates beautiful, clear distribution plots showing posterior uncertainty
    over model parameters, with optional comparison to simulation-based inference (SBI) methods.

    Parameters:
    -----------
    model : torch.nn.Module
        The parameter prediction model (head)
    pointnet_model : torch.nn.Module
        The PointNet feature extractor
    true_params : torch.Tensor
        True parameter values for comparison
    device : torch.device
        Device to run computations on
    n_mc : int
        Number of Monte Carlo samples (ignored when using analytic uncertainty)
    laplace_model : object, optional
        Fitted Laplace approximation object for analytic uncertainty
    compare_with_sbi : bool
        Whether to include SBI posterior comparisons
    sbi_posteriors : list of torch.Tensor, optional
        SBI posterior samples for comparison
    sbi_labels : list of str, optional
        Labels for SBI methods
    save_path : str
        Path to save the plot
    problem : str
        Problem type ('simplified_dis', 'mceg', 'mceg4dis')

    Returns:
    --------
    None
        Saves the publication-ready plot to save_path

    Notes:
    ------
    When laplace_model is provided, uses analytic uncertainty propagation via
    delta method instead of Monte Carlo sampling for improved speed and accuracy.
    The resulting plots feature:
    - Colorblind-friendly color palette
    - Clear mathematical notation in labels
    - Professional typography and layout
    - Proper uncertainty visualization with filled regions
    - Comparison with true parameter values
    """
    model.eval()
    pointnet_model.eval()

    # Get simulators with fallback
    SimplifiedDIS, MCEGSimulator = get_simulator_module()
    if problem in ["mceg", "mceg4dis"]:
        simulator = MCEGSimulator(torch.device("cpu"))
    else:
        simulator = SimplifiedDIS(torch.device("cpu"))

    log_feature_engineering = get_log_feature_engineering()

    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    if problem not in ["mceg", "mceg4dis"]:
        xs_tensor = log_feature_engineering(xs_tensor)
    else:
        from utils import log_feature_engineering

        xs_tensor = log_feature_engineering(xs_tensor).float()
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    if laplace_model is not None:
        # Use analytic uncertainty propagation (delta method)
        mean_params, std_params = get_analytic_uncertainty(
            model, latent_embedding, laplace_model
        )
        mean_params = mean_params.cpu().squeeze(0)
        std_params = std_params.cpu().squeeze(0)
        # Also generate samples from the analytic Gaussian posterior so we can show entrywise histograms
        samples = get_gaussian_samples(
            model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model
        ).cpu()
        use_analytic = True
        uncertainty_label = "Analytic (Laplace)"
    else:
        # Fallback to MC sampling for backward compatibility
        samples = get_gaussian_samples(
            model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model
        ).cpu()
        mean_params = torch.mean(samples, dim=0)
        std_params = torch.std(samples, dim=0)
        use_analytic = False
        uncertainty_label = "Monte Carlo"

    n_params = true_params.size(0)

    # Set parameter names with proper mathematical notation
    if problem == "simplified_dis":
        param_names = [r"$a_u$", r"$b_u$", r"$a_d$", r"$b_d$"]
    elif problem in ["mceg", "mceg4dis"]:
        param_names = [r"$\mu_1$", r"$\mu_2$", r"$\sigma_1$", r"$\sigma_2$"]
    else:
        param_names = [f"$\theta_{{{i+1}}}$" for i in range(n_params)]

    # Set up color palette
    base_colors = [
        COLORBLIND_COLORS["blue"],
        COLORBLIND_COLORS["orange"],
        COLORBLIND_COLORS["green"],
        COLORBLIND_COLORS["purple"],
        COLORBLIND_COLORS["brown"],
        COLORBLIND_COLORS["pink"],
    ]

    # Create subplots with publication-ready sizing
    if problem == "simplified_dis" and n_params == 4:
        rows, cols = 2, 2
        figsize = (14, 12)
    else:
        cols = min(n_params, 4)
        rows = (n_params + cols - 1) // cols
        figsize = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # normalize axes to a flat list
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Hide any extra axes
    for j in range(n_params, len(axes)):
        try:
            axes[j].set_visible(False)
        except Exception:
            pass

    # Prepare SBI data for proper color cycling
    # Always include our-approach samples (either MC fallback or analytic->samples) so we can show per-dim histograms
    all_samples = [samples]
    all_labels = [uncertainty_label]

    if compare_with_sbi and sbi_posteriors is not None:
        all_samples.extend(sbi_posteriors)
        if sbi_labels is not None:
            all_labels.extend(sbi_labels)
        else:
            all_labels.extend([f"SBI Method {i+1}" for i in range(len(sbi_posteriors))])

    for i in range(n_params):
        ax = axes[i]

        # Larger per-axes font defaults for publication
        title_fs = 20
        label_fs = 18
        tick_fs = 18
        legend_fs = 18

        # Choose the main sample set to plot for "our" approach.
        # If Combined_LOTV (lotv_param_samples) is provided, prefer that as the main histogram.
        try:
            if lotv_param_samples is not None:
                lp = np.asarray(lotv_param_samples)
                if lp.ndim == 3:
                    lp = lp.reshape(-1, lp.shape[-1])
                if lp.ndim == 2 and lp.shape[1] > i:
                    main_vals = lp[:, i]
                    main_label = lotv_label
                else:
                    main_vals = samples[:, i].numpy()
                    main_label = uncertainty_label
            else:
                main_vals = samples[:, i].numpy()
                main_label = uncertainty_label
        except Exception:
            main_vals = samples[:, i].numpy()
            main_label = uncertainty_label

        # Compute plotting bounds and plot histogram (our approach as histogram)
        try:
            xmin = float(np.nanmin(main_vals))
            xmax = float(np.nanmax(main_vals))
        except Exception:
            xmin, xmax = -1.0, 1.0
        padding = 0.12 * (xmax - xmin) if xmax > xmin else 0.1
        xmin -= padding
        xmax += padding

        n, bins, patches = ax.hist(
            main_vals,
            bins=36,
            alpha=0.8,
            density=True,
            color=base_colors[0],
            label=main_label,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_xlim(xmin, xmax)

        # If analytic Laplace is available, overlay its Gaussian posterior as a dashed reference
        if laplace_model is not None:
            try:
                mu = float(mean_params[i].item())
                sigma = float(std_params[i].item())
                x_range = max(
                    4 * sigma if sigma > 0 else 0.5,
                    0.1 * abs(mu) if abs(mu) > 0 else 0.5,
                )
                x_vals = np.linspace(mu - x_range, mu + x_range, 1000)
                from scipy.stats import norm

                gaussian_pdf = norm.pdf(
                    x_vals, loc=mu, scale=sigma if sigma > 0 else 1.0
                )
                ax.plot(
                    x_vals,
                    gaussian_pdf,
                    color="k",
                    linestyle="--",
                    linewidth=2.0,
                    label="Analytic Gaussian",
                    zorder=4,
                )
                # Add light filled area for ±1σ as reference (not primary)
                ax.fill_between(
                    x_vals, 0, gaussian_pdf, color="k", alpha=0.05, zorder=1
                )
            except Exception:
                pass

        # Add SBI comparison if requested
        if compare_with_sbi and sbi_posteriors is not None and sbi_labels is not None:
            for j, sbi_samples in enumerate(sbi_posteriors):
                color_idx = (j + 1) % len(base_colors)
                label = sbi_labels[j] if j < len(sbi_labels) else f"SBI {j+1}"
                ax.hist(
                    sbi_samples[:, i].detach().cpu().numpy(),
                    bins=30,
                    alpha=0.6,
                    density=True,
                    color=base_colors[color_idx],
                    label=label,
                    edgecolor="white",
                    linewidth=0.5,
                )

        # Add true value line with enhanced styling
        true_val = true_params[i].item()
        ax.axvline(
            true_val,
            color=COLORBLIND_COLORS["red"],
            linestyle="--",
            linewidth=2.5,
            label="True Value",
            alpha=0.95,
            zorder=4,
        )

        # Enhanced axis styling (larger fonts for publication)
        ax.set_title(
            f"Parameter {param_names[i]}", fontsize=title_fs, pad=14, fontweight="bold"
        )
        ax.set_xlabel(f"{param_names[i]}", fontsize=label_fs)
        ax.set_ylabel("Probability Density", fontsize=label_fs)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.6)
        ax.tick_params(which="both", direction="in", labelsize=tick_fs)

        # Add legend only to first subplot to avoid clutter
        if i == 0:
            ax.legend(
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=legend_fs,
                loc="upper right",
            )

        # Add statistics text box (per subplot)
        if use_analytic:
            stats_text = f"μ = {mu:.3f}\nσ = {sigma:.3f}"
        else:
            sample_mean = torch.mean(samples[:, i]).item()
            sample_std = torch.std(samples[:, i]).item()
            stats_text = f"μ = {sample_mean:.3f}\nσ = {sample_std:.3f}"

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=20,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )

    # Add overall title
    method_str = "Analytic Laplace" if use_analytic else "Monte Carlo"
    # fig.suptitle(f'Parameter Posterior Distributions ({method_str} Uncertainty)',
    #             fontsize=16, y=0.98, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    fig.suptitle(
        f'Parameter Posterior Distributions ({"Analytic Laplace" if use_analytic else "Monte Carlo"})',
        fontsize=22,
        y=0.98,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_PDF_distribution_single_same_plot(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,  # Kept for backward compatibility but ignored when using analytic uncertainty
    laplace_model=None,
    problem="simplified_dis",
    Q2_slices=None,
    plot_IQR=False,
    save_path="pdf_overlay.png",
):
    """
    Plot PDF distributions on the same plot using analytic Laplace uncertainty propagation.

    When laplace_model is provided, uses analytic uncertainty propagation to
    compute error bands instead of Monte Carlo sampling for improved speed and accuracy.
    """
    model.eval()
    pointnet_model.eval()
    simulator = SimplifiedDIS(torch.device("cpu"))

    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    if problem not in ["mceg", "mceg4dis"]:
        from utils import log_feature_engineering
        xs_tensor = log_feature_engineering(xs_tensor)
    else:
        # FIXED: Apply log_feature_engineering for mceg/mceg4dis to match training
        from utils import log_feature_engineering

        original_shape = xs_tensor.shape
        xs_tensor = log_feature_engineering(xs_tensor).float()
        print(
            f"✅ [SAFETY CHECK] mceg/mceg4dis feature engineering applied: {original_shape} -> {xs_tensor.shape}"
        )
        if xs_tensor.shape[-1] != 6:
            print(
                f"⚠️  [WARNING] Expected 6D features for mceg/mceg4dis, got {xs_tensor.shape[-1]}D"
            )
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    if laplace_model is not None:
        # Use analytic uncertainty propagation (delta method)
        mean_params, std_params = get_analytic_uncertainty(
            model, latent_embedding, laplace_model
        )
        mean_params = mean_params.cpu().squeeze(0)
        std_params = std_params.cpu().squeeze(0)
        use_analytic = True
    else:
        # Fallback to MC sampling for backward compatibility
        samples = get_gaussian_samples(
            model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model
        ).cpu()
        use_analytic = False


def plot_parameter_error_histogram(
    true_params_list,
    predicted_params_list,
    param_names=None,
    save_path="parameter_error_histogram.png",
    problem="simplified_dis",
):
    """
    Create publication-ready histograms of parameter errors across multiple parameter choices.

    Parameters:
    -----------
    true_params_list : list of numpy arrays or torch tensors
        List of true parameter sets, each array/tensor of shape (n_params,)
    predicted_params_list : list of numpy arrays or torch tensors
        List of predicted parameter sets, each array/tensor of shape (n_params,)
    param_names : list of str, optional
        Parameter names for axis labels. Auto-generated if None.
    save_path : str
        Path to save the histogram plot
    problem : str
        Problem type ('simplified_dis') for default param names

    Returns:
    --------
    None
        Saves the plot to save_path
    """
    # Convert to numpy if needed and compute errors
    if isinstance(true_params_list[0], torch.Tensor):
        true_params_list = [p.detach().cpu().numpy() for p in true_params_list]
    if isinstance(predicted_params_list[0], torch.Tensor):
        predicted_params_list = [
            p.detach().cpu().numpy() for p in predicted_params_list
        ]

    true_params = np.array(true_params_list)  # Shape: (n_samples, n_params)
    predicted_params = np.array(predicted_params_list)

    # Compute parameter errors
    param_errors = predicted_params - true_params  # Shape: (n_samples, n_params)
    relative_errors = param_errors / (true_params + 1e-8)  # Avoid division by zero

    n_params = param_errors.shape[1]

    # Set default parameter names
    if param_names is None:
        if problem == "simplified_dis":
            param_names = [r"$a_u$", r"$b_u$", r"$a_d$", r"$b_d$"]
        else:
            param_names = [r"$a$", r"$b$", r"$c$", r"$d$"]

    # Create subplots
    fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 10))
    if n_params == 1:
        axes = axes.reshape(2, 1)

    colors_list = [
        COLORBLIND_COLORS["blue"],
        COLORBLIND_COLORS["orange"],
        COLORBLIND_COLORS["green"],
        COLORBLIND_COLORS["red"],
    ]

    for i in range(n_params):
        color = colors_list[i % len(colors_list)]

        # Absolute errors (top row)
        ax_abs = axes[0, i]
        n_bins = min(50, max(10, len(param_errors) // 5))  # Adaptive binning
        counts, bins, patches = ax_abs.hist(
            param_errors[:, i],
            bins=n_bins,
            alpha=0.7,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

        # Statistics text (larger fonts for readability)
        mean_err = np.mean(param_errors[:, i])
        std_err = np.std(param_errors[:, i])
        ax_abs.text(
            0.02,
            0.98,
            f"μ = {mean_err:.3f}\nσ = {std_err:.3f}",
            transform=ax_abs.transAxes,
            verticalalignment="top",
            fontsize=20,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        # Axis labels and styling (no log scale, no true-value line or legend)
        if i == 0:
            ax_abs.set_xlabel("Absolute Error", fontsize=20)
            ax_abs.set_ylabel("Frequency", fontsize=20)
        ax_abs.tick_params(axis="x", labelsize=14)
        ax_abs.tick_params(axis="y", labelsize=14)
        ax_abs.grid(True, alpha=0.3)

        # Relative errors (bottom row)
        ax_rel = axes[1, i]
        counts, bins, patches = ax_rel.hist(
            relative_errors[:, i] * 100,  # Convert to percentage
            bins=n_bins,
            alpha=0.7,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

        # Statistics text (relative errors, larger fonts)
        mean_rel_err = np.mean(relative_errors[:, i]) * 100
        std_rel_err = np.std(relative_errors[:, i]) * 100
        ax_rel.text(
            0.02,
            0.98,
            f"μ = {mean_rel_err:.1f}%\nσ = {std_rel_err:.1f}%",
            transform=ax_rel.transAxes,
            verticalalignment="top",
            fontsize=20,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )
        if i == 0:
            ax_rel.set_xlabel("Relative Error (%)", fontsize=20)
            ax_rel.set_ylabel("Frequency", fontsize=20)
        ax_rel.tick_params(axis="x", labelsize=14)
        ax_rel.tick_params(axis="y", labelsize=14)
        ax_rel.grid(True, alpha=0.3)

    # plt.suptitle('Parameter Error Analysis', fontsize=18, y=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_function_error_histogram_simplified_dis(
    true_params_list,
    predicted_params_list,
    param_names=None,
    save_path="function_error_histogram_simplified_dis.png",
    device="cpu",
    nx=100,
    n_bins_hist=40,
    seed=None,
):
    """
    Plot separate histograms of function errors (simplified_dis) for up and down functions.

    For each (true_params, predicted_params) pair:
    1. Evaluate u(x) and d(x) at true parameters
    2. Evaluate u(x) and d(x) at predicted parameters
    3. Compute entrywise absolute errors: |f_true(x) - f_pred(x)|
    4. Average over x for each function separately
    5. Create two side-by-side histograms (one for up, one for down)

    Parameters:
    -----------
    true_params_list : list of numpy arrays or torch tensors
        List of true parameter sets, each array/tensor of shape (n_params,)
    predicted_params_list : list of numpy arrays or torch tensors
        List of predicted parameter sets, each array/tensor of shape (n_params,)
    param_names : list of str, optional
        Parameter names (unused, kept for consistency with other functions)
    save_path : str
        Path to save the histogram plot
    device : str or torch.device
        Device for simulation
    nx : int
        Number of x evaluation points
    n_bins_hist : int
        Number of bins for histogram
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary with keys 'up' and 'down', each containing arrays of errors for that function
    """
    # Ensure reproducibility when requested
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Convert inputs to numpy lists
    def to_numpy_list(lst):
        if len(lst) == 0:
            return []
        first = lst[0]
        if isinstance(first, torch.Tensor):
            return [p.detach().cpu().numpy() for p in lst]
        return [np.asarray(p) for p in lst]

    true_params_list = to_numpy_list(true_params_list)
    predicted_params_list = to_numpy_list(predicted_params_list)

    if len(true_params_list) != len(predicted_params_list):
        raise ValueError(
            "true_params_list and predicted_params_list must have same length."
        )

    n_samples = len(true_params_list)

    # Get simulator
    SimplifiedDIS, _ = get_simulator_module()
    if SimplifiedDIS is None:
        raise ImportError("SimplifiedDIS simulator not available.")

    sim = SimplifiedDIS(device=torch.device("cpu"))
    x_grid = torch.linspace(0.001, 0.99, nx)

    # Separate arrays for up and down errors
    errors_up = np.zeros(n_samples, dtype=float)
    errors_down = np.zeros(n_samples, dtype=float)

    for idx in range(n_samples):
        true_theta = torch.tensor(true_params_list[idx], dtype=torch.float32)
        pred_theta = torch.tensor(predicted_params_list[idx], dtype=torch.float32)

        # Evaluate both u and d functions
        for fn_idx, fn_name in enumerate(["up", "down"]):
            try:
                # True function
                sim.init(true_theta)
                fn_true = getattr(sim, fn_name)(x_grid).detach().cpu().numpy()
            except Exception:
                fn_true = np.full(nx, np.nan)

            try:
                # Predicted function
                sim.init(pred_theta)
                fn_pred = getattr(sim, fn_name)(x_grid).detach().cpu().numpy()
            except Exception:
                fn_pred = np.full(nx, np.nan)

            # Compute absolute errors
            diff = np.abs(fn_true - fn_pred)
            mask = np.isfinite(diff)
            avg_error = float(np.nanmean(diff[mask])) if np.any(mask) else 0.0
            
            # Store in appropriate array
            if fn_idx == 0:
                errors_up[idx] = avg_error
            else:
                errors_down[idx] = avg_error

    # --- Compute NPE errors if available ---
    npe_errors_up = np.array([])
    npe_errors_down = np.array([])
    try:
        import pickle
        import os
        npe_path = "npe_posterior.pkl"
        if os.path.exists(npe_path):
            print(f"Loading NPE posterior from {npe_path}...")
            with open(npe_path, "rb") as f:
                posterior_npe = pickle.load(f)
            
            # Sample parameters from NPE for each true parameter
            npe_errors_up_list = []
            npe_errors_down_list = []
            for idx in range(n_samples):
                true_theta = torch.tensor(true_params_list[idx], dtype=torch.float32)
                
                # Generate observation from true parameters
                xs = sim.sample(true_theta.detach().cpu(), 10000)
                xs_tensor = torch.tensor(xs, dtype=torch.float32, device=torch.device("cpu"))
                
                # Get summary for NPE
                try:
                    from sbibm_benchmark import histogram_summary
                    x_o_summary = histogram_summary(xs_tensor)
                except Exception:
                    continue
                
                # Sample from NPE posterior
                try:
                    npe_samples = posterior_npe.sample((1,), x=x_o_summary)
                    pred_theta_npe = npe_samples[0]
                except Exception:
                    continue
                
                # Evaluate both u and d functions
                for fn_idx, fn_name in enumerate(["up", "down"]):
                    try:
                        # True function
                        sim.init(true_theta)
                        fn_true = getattr(sim, fn_name)(x_grid).detach().cpu().numpy()
                    except Exception:
                        fn_true = np.full(nx, np.nan)
                    
                    try:
                        # NPE predicted function
                        sim.init(pred_theta_npe)
                        fn_pred = getattr(sim, fn_name)(x_grid).detach().cpu().numpy()
                    except Exception:
                        fn_pred = np.full(nx, np.nan)
                    
                    # Compute absolute errors
                    diff = np.abs(fn_true - fn_pred)
                    mask = np.isfinite(diff)
                    avg_error = float(np.nanmean(diff[mask])) if np.any(mask) else 0.0
                    
                    # Store in appropriate array
                    if fn_idx == 0:
                        npe_errors_up_list.append(avg_error)
                    else:
                        npe_errors_down_list.append(avg_error)
            
            npe_errors_up = np.array(npe_errors_up_list)
            npe_errors_down = np.array(npe_errors_down_list)
            print(f"✓ Computed NPE errors: {len(npe_errors_up)} samples")
        else:
            print(f"NPE posterior not found at {npe_path}, skipping NPE comparison")
    except Exception as e:
        print(f"Could not compute NPE errors: {e}")
    
    # Create side-by-side histograms (now with NPE overlay if available)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Function names and properties
    func_data = [
        (errors_up, "up", r"$u(x)$", COLORBLIND_COLORS["blue"]),
        (errors_down, "down", r"$d(x)$", COLORBLIND_COLORS["orange"]),
    ]
    npe_data = [npe_errors_up, npe_errors_down]

    for ax, (errors, fn_name, fn_label, color), npe_errors in zip(axes, func_data, npe_data):
        # Create log-spaced bins for better visualization
        all_errors_combined = errors
        if len(npe_errors) > 0:
            all_errors_combined = np.concatenate([errors, npe_errors])
        
        # Ensure all values are positive for log spacing
        min_err = np.min(all_errors_combined[all_errors_combined > 0]) if np.any(all_errors_combined > 0) else 1e-10
        max_err = np.max(all_errors_combined)
        log_bins = np.logspace(np.log10(min_err), np.log10(max_err), n_bins_hist + 1)
        
        ax.hist(
            errors,
            bins=log_bins,
            alpha=0.6,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label="Ours",
        )
        
        # Overlay NPE histogram if available
        if len(npe_errors) > 0:
            ax.hist(
                npe_errors,
                bins=log_bins,
                alpha=0.6,
                color=COLORBLIND_COLORS["purple"],
                edgecolor="white",
                linewidth=0.8,
                label="NPE",
            )
        
        # Set log scale for x-axis
        ax.set_xscale('log')

        mean_err = np.mean(errors)
        std_err = np.std(errors)
        median_err = np.median(errors)

        # Add vertical lines for mean and median (Ours)
        ax.axvline(
            mean_err,
            color=COLORBLIND_COLORS["red"],
            linestyle="--",
            linewidth=3,
            label=f"Ours (Mean) = {mean_err:.4g}",
            alpha=0.9,
        )
        ax.axvline(
            median_err,
            color=COLORBLIND_COLORS["dark_green"],
            linestyle=":",
            linewidth=3,
            label=f"Ours (Median) = {median_err:.4g}",
            alpha=0.9,
        )
        
        # Add NPE median line if available
        if len(npe_errors) > 0:
            npe_median = np.median(npe_errors)
            ax.axvline(
                npe_median,
                color=COLORBLIND_COLORS["purple"],
                linestyle="-.",
                linewidth=3,
                label=f"NPE (Median) = {npe_median:.4g}",
                alpha=0.9,
            )

        # Statistics text box (our method)
        stats_text = f"Ours:\nμ = {mean_err:.4g}\nσ = {std_err:.4g}"
        if len(npe_errors) > 0:
            npe_mean = np.mean(npe_errors)
            npe_std = np.std(npe_errors)
            stats_text += f"\n\nNPE:\nμ = {npe_mean:.4g}\nσ = {npe_std:.4g}"
        
        ax.text(
            0.98,
            0.97,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=22,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.95,
                edgecolor="gray",
            ),
        )

        # Axis labels and styling
        ax.set_xlabel(
            "Errors",
            fontsize=24,
        )
        ax.set_ylabel("Frequency", fontsize=24)
        # ax.set_title(
        #     f"Function Error Distribution: {fn_label}",
        #     fontsize=18,
        #     fontweight="bold",
        # )
        ax.tick_params(which="both", direction="in", labelsize=22)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, which="both")
        ax.legend(fontsize=20, loc="upper left", frameon=True, fancybox=True, shadow=True)
        
        # Format x-axis labels with 10^x notation
        from matplotlib.ticker import FuncFormatter
        def log_formatter(x, pos):
            if x <= 0:
                return ''
            exponent = np.log10(x)
            if abs(exponent - round(exponent)) < 0.01:  # Close to integer power
                return f'$10^{{{int(round(exponent))}}}$'
            else:
                return f'$10^{{{exponent:.1f}}}$'
        ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))

    # fig.suptitle(
    #     "Function Error Analysis: Simplified DIS (up and down separately)",
    #     fontsize=20,
    #     fontweight="bold",
    #     y=1.00,
    # )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"up": errors_up, "down": errors_down}


    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(
        final_errors,
        bins=n_bins_hist,
        alpha=0.8,
        color=COLORBLIND_COLORS["blue"],
        edgecolor="white",
        linewidth=0.8,
    )

    mean_final = np.mean(final_errors)
    std_final = np.std(final_errors)
    median_final = np.median(final_errors)

    # Add vertical lines for mean and median
    ax.axvline(
        mean_final,
        color=COLORBLIND_COLORS["red"],
        linestyle="--",
        linewidth=2.5,
        label=f"Mean = {mean_final:.4g}",
        alpha=0.9,
    )
    ax.axvline(
        median_final,
        color=COLORBLIND_COLORS["dark_green"],
        linestyle=":",
        linewidth=2,
        label=f"Median = {median_final:.4g}",
        alpha=0.9,
    )

    # Statistics text box
    ax.text(
        0.98,
        0.97,
        f"μ = {mean_final:.4g}\nσ = {std_final:.4g}\nn = {n_samples}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=18,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor="gray"),
    )

    # Axis labels and styling
    ax.set_xlabel(
        "Average Function Error: mean |u(x,true) - u(x,pred)| and |d(x,true) - d(x,pred)|",
        fontsize=16,
    )
    ax.set_ylabel("Frequency", fontsize=16)
    ax.set_title(
        "Histogram of Average Function Errors (Simplified DIS)",
        fontsize=18,
        fontweight="bold",
    )
    ax.tick_params(which="both", direction="in", labelsize=14)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.legend(fontsize=16, loc="upper left", frameon=True, fancybox=True, shadow=True)

    # Sparse x-ticks to avoid overlap
    xmin, xmax = ax.get_xlim()
    xticks = np.linspace(xmin, xmax, num=6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.2g}" for t in xticks])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return final_errors


def plot_function_error_histogram_mceg(
    true_params_list,
    predicted_params_list,
    param_names=None,
    save_path="function_error_histogram_mceg.png",
    problem="simplified_dis",
    # MCEG-specific args:
    device="cpu",
    num_events=10000,
    nx=30,
    nQ2=20,
    Q2_slices=None,
    # plotting options
    n_bins_hist=40,
    seed=None,
    # Optional LoTV inputs: if provided, compute errors from function-space LoTV decomposition
    per_boot_posterior_samples=None,
    lotv_n_theta_per_boot=20,
    lotv_num_events=20000,
    lotv_nx=30,
    lotv_nQ2=20,
):
    """
    Plot histogram of average function errors (MCEG) across parameter sets.

    Behavior:
      - If problem == 'mceg': for each (true_params, predicted_params) pair:
            * simulate events at true params -> build log(x), log(Q2) histogram (nx x nQ2)
            * simulate events at predicted params -> build same-style histogram
            * for selected Q2 indices, compute entrywise abs error across x bins:
                mean_error_j = mean_{x bins in that Q2} |f_true(x,Q2) - f_pred(x,Q2)|
              final_error_for_pair = mean_j mean_error_j  (average across chosen Q2)
            * collect final_error_for_pair for all parameter pairs and plot histogram & stats
      - If problem != 'mceg' falls back to parameter-error histogram behavior (absolute & relative).
    Parameters:
      - true_params_list, predicted_params_list : lists of arrays or tensors (n_samples x n_params)
      - param_names: list of strings (optional) for fallback parameter histograms
      - save_path: where to save the figure
      - problem: 'mceg' or other (default 'simplified_dis')
      - device, num_events, nx, nQ2, Q2_slices: control MCEG simulation/binning
    """
    # Ensure reproducibility when requested
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Convert inputs to numpy lists
    def to_numpy_list(lst):
        if len(lst) == 0:
            return []
        first = lst[0]
        if isinstance(first, torch.Tensor):
            return [p.detach().cpu().numpy() for p in lst]
        return [np.asarray(p) for p in lst]

    true_params_list = to_numpy_list(true_params_list)
    predicted_params_list = to_numpy_list(predicted_params_list)

    if len(true_params_list) != len(predicted_params_list):
        raise ValueError(
            "true_params_list and predicted_params_list must have same length."
        )

    n_samples = len(true_params_list)

    # If not MCEG, fall back to parameter-error histograms (unchanged behavior)
    if problem != "mceg":
        # Reuse the simpler parameter-error plotting already present in this file
        true_params = np.vstack(true_params_list)
        predicted_params = np.vstack(predicted_params_list)
        param_errors = predicted_params - true_params
        relative_errors = param_errors / (true_params + 1e-8)
        n_params = param_errors.shape[1]

        if param_names is None:
            if problem == "simplified_dis":
                param_names = [r"$a_u$", r"$b_u$", r"$a_d$", r"$b_d$"][:n_params]
            else:
                param_names = [f"$\\theta_{{{i}}}$" for i in range(n_params)]

        fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 10))
        if n_params == 1:
            axes = axes.reshape(2, 1)

        colors_list = [
            COLORBLIND_COLORS["blue"],
            COLORBLIND_COLORS["orange"],
            COLORBLIND_COLORS["green"],
            COLORBLIND_COLORS["red"],
        ]

        for i in range(n_params):
            color = colors_list[i % len(colors_list)]
            ax_abs = axes[0, i]
            n_bins = min(50, max(10, len(param_errors) // 5))
            ax_abs.hist(
                param_errors[:, i],
                bins=n_bins,
                alpha=0.7,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            ax_abs.axvline(0, color="red", linestyle="--", alpha=0.8, linewidth=2)
            mean_err = np.mean(param_errors[:, i])
            std_err = np.std(param_errors[:, i])
            ax_abs.text(
                0.02,
                0.98,
                f"μ = {mean_err:.3f}\nσ = {std_err:.3f}",
                transform=ax_abs.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            ax_abs.set_xlabel(f"Error in {param_names[i]}")
            ax_abs.set_ylabel("Frequency")
            ax_abs.set_title(f"Absolute Error: {param_names[i]}")
            ax_abs.grid(True, alpha=0.3)

            ax_rel = axes[1, i]
            ax_rel.hist(
                relative_errors[:, i] * 100,
                bins=n_bins,
                alpha=0.7,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            ax_rel.axvline(0, color="red", linestyle="--", alpha=0.8, linewidth=2)
            mean_rel_err = np.mean(relative_errors[:, i]) * 100
            std_rel_err = np.std(relative_errors[:, i]) * 100
            ax_rel.text(
                0.02,
                0.98,
                f"μ = {mean_rel_err:.1f}%\nσ = {std_rel_err:.1f}%",
                transform=ax_rel.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            ax_rel.set_xlabel(f"Relative Error in {param_names[i]} (%)")
            ax_rel.set_ylabel("Frequency")
            ax_rel.set_title(f"Relative Error: {param_names[i]}")
            ax_rel.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    # MCEG branch: compute per-pair function error using collaborator PDF u-ub at Q2 = 10
    try:
        from simulator import ALPHAS, EWEAK, MELLIN, PDF, MCEGSimulator
    except Exception:
        # Fallback: if collaborator PDF not available, raise with informative message
        raise RuntimeError(
            "Required MCEG/PDF simulator components not available for PDF-based error computation."
        )

    sim = MCEGSimulator(device=device)

    q2_val = 10.0
    x_grid = np.linspace(0.001, 0.99, 100)

    # helper: evaluate u-ub for a parameter vector using collaborator PDF (robust)
    def eval_pdf_u_minus_ub(theta_arr):
        pdf_temp = PDF(MELLIN(npts=8), ALPHAS())
        cpar = pdf_temp.get_current_par_array()[::]
        arr = np.asarray(theta_arr)
        try:
            cpar[4 : 4 + arr.shape[0]] = arr
        except Exception:
            try:
                cpar[4:8] = arr
            except Exception:
                cpar[4 : 4 + arr.shape[0]] = arr
        try:
            pdf_temp.setup(cpar)
        except Exception:
            try:
                pdf_temp.setup(arr)
            except Exception:
                raise

        vals = []
        for x in x_grid:
            try:
                u = pdf_temp.get_xF(float(x), q2_val, "u", evolve=True)
                ub = pdf_temp.get_xF(float(x), q2_val, "ub", evolve=True)
                uval = float(u[0]) if hasattr(u, "__len__") else float(u)
                ubval = float(ub[0]) if hasattr(ub, "__len__") else float(ub)
                vals.append(uval - ubval)
            except Exception:
                vals.append(np.nan)
        return np.asarray(vals)

    final_errors = np.zeros(n_samples, dtype=float)

    for idx in range(n_samples):
        true_theta = true_params_list[idx]
        pred_theta = predicted_params_list[idx]
        try:
            true_curve = eval_pdf_u_minus_ub(true_theta)
        except Exception:
            true_curve = np.full_like(x_grid, np.nan)
        try:
            pred_curve = eval_pdf_u_minus_ub(pred_theta)
        except Exception:
            pred_curve = np.full_like(x_grid, np.nan)

        diff = np.abs(true_curve - pred_curve)
        mask = np.isfinite(diff)
        final_errors[idx] = float(np.nanmean(diff[mask])) if np.any(mask) else 0.0

    # Plot histogram with sparse x-ticks to avoid overlap
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(final_errors, bins=n_bins_hist, alpha=0.8, edgecolor="white")
    mean_final = np.mean(final_errors)
    std_final = np.std(final_errors)
    median_final = np.median(final_errors)
    ax.axvline(
        mean_final,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean = {mean_final:.4g}",
    )
    ax.axvline(
        median_final,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"Median = {median_final:.4g}",
    )
    ax.set_xlabel("Average function error (mean |u-ub| over x at Q^2=10)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Histogram of Average Function Errors (MCEG, PDF-based at Q²=10)", fontsize=14
    )
    ax.text(
        0.98,
        0.98,
        f"μ = {mean_final:.4g}\nσ = {std_final:.4g}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # Sparse xticks: choose up to 5 nicely spaced ticks
    xmin, xmax = ax.get_xlim()
    xticks = np.linspace(xmin, xmax, num=5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.2g}" for t in xticks])

    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return final_errors

    # Now plot histogram of final_errors
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(final_errors, bins=n_bins_hist, alpha=0.8, edgecolor="white")
    mean_final = np.mean(final_errors)
    std_final = np.std(final_errors)
    median_final = np.median(final_errors)
    ax.axvline(
        mean_final,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean = {mean_final:.4g}",
    )
    ax.axvline(
        median_final,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"Median = {median_final:.4g}",
    )
    ax.set_xlabel("Average function error (averaged over x then Q²)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Histogram of Average Function Errors (MCEG)", fontsize=14)
    ax.text(
        0.98,
        0.98,
        f"μ = {mean_final:.4g}\nσ = {std_final:.4g}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Also return the final errors array so user can inspect them programmatically
    return final_errors


def plot_event_histogram_simplified_DIS(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,  # Kept for backward compatibility but ignored when using analytic uncertainty
    laplace_model=None,
    num_events=100000,
    save_path="event_histogram_simplified.png",
    problem="simplified_dis",
    plot_type="both",  # 'scatter', 'histogram', or 'both'
    bins=50,
    figsize=(15, 8),
):
    """
    Plot event histograms using analytic Laplace uncertainty propagation.
    Provides both scatter plots and true 2D histograms for reconstructed events.

    ⚠️  **REPRODUCIBILITY WARNING**: This function requires simulation to generate
    event data for histogram plotting. Results may vary between runs unless random
    seeds are fixed. For reproducible latent extraction, prefer functions that use
    precomputed data from extract_latents_from_data().

    Parameters:
    -----------
    model : torch.nn.Module
        The parameter prediction model (head)
    pointnet_model : torch.nn.Module
        The PointNet feature extractor
    true_params : torch.Tensor
        True parameter values
    device : torch.device
        Device to run computations on
    n_mc : int
        Number of Monte Carlo samples (kept for backward compatibility)
    laplace_model : object, optional
        Fitted Laplace approximation object for analytic uncertainty
    num_events : int
        Number of events to generate via simulation
    save_path : str
        Path to save the plot
    problem : str
        Problem type ('simplified_dis', 'mceg')
    plot_type : str
        Type of plot: 'scatter', 'histogram', or 'both'
    bins : int or array-like
        Number of bins for 2D histogram
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    None
        Saves the plot to save_path
    """
    model.eval()
    pointnet_model.eval()

    print(f"📊 Generating event histogram via simulation for {problem}")
    print(f"🎲 Simulating {num_events} events for histogram analysis")

    # Get the appropriate simulator
    SimplifiedDIS, MCEGSimulator = get_simulator_module()
    if problem in ["mceg", "mceg4dis"]:
        simulator = MCEGSimulator(torch.device("cpu"))
    else:
        simulator = SimplifiedDIS(torch.device("cpu"))

    log_feature_engineering = get_log_feature_engineering()

    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), num_events).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)

    if problem not in ["mceg", "mceg4dis"]:
        xs_tensor = log_feature_engineering(xs_tensor)
    else:
        from utils import log_feature_engineering

        xs_tensor = log_feature_engineering(xs_tensor).float()
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    if laplace_model is not None:
        # Use analytic uncertainty propagation (delta method)
        mean_params, std_params = get_analytic_uncertainty(
            model, latent_embedding, laplace_model
        )
        predicted_params = mean_params.cpu().squeeze(0)
        use_analytic = True
    else:
        # Fallback to MC sampling for backward compatibility
        samples = get_gaussian_samples(
            model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model
        ).cpu()
        predicted_params = torch.median(samples, dim=0).values
        use_analytic = False

    # Generate events using the predicted parameters
    generated_events = simulator.sample(predicted_params.detach().cpu(), num_events).to(
        device
    )

    # Convert to numpy for plotting
    true_events_np = xs.detach().cpu().numpy()
    generated_events_np = generated_events.detach().cpu().numpy()

    # Determine number of subplots based on plot_type
    if plot_type == "both":
        fig, axes = plt.subplots(2, 2, figsize=(figsize[0], figsize[1] * 1.2))
        ax_true_scatter, ax_gen_scatter = axes[0, 0], axes[0, 1]
        ax_true_hist, ax_gen_hist = axes[1, 0], axes[1, 1]
    elif plot_type in ["scatter", "histogram"]:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_true, ax_gen = axes[0], axes[1]
    else:
        raise ValueError("plot_type must be 'scatter', 'histogram', or 'both'")

    # method_label = "MAP (Analytic)" if use_analytic else "Median (MC)"

    # Color scheme
    true_color = COLORBLIND_COLORS["cyan"]
    gen_color = COLORBLIND_COLORS["orange"]

    # Helper function to set log scales and labels
    def setup_axes(ax, title, is_generated=False):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=25, pad=10)
        if problem == "simplified_dis":
            if is_generated:
                ax.set_xlabel(rf"$x_{{u}} \sim u(x|\hat{{\theta}})$", fontsize=25)
                ax.set_ylabel(rf"$x_{{d}} \sim d(x|\hat{{\theta}})$", fontsize=25)
            else:
                ax.set_xlabel(r"$x_{u} \sim u(x|\theta^{*})$", fontsize=25)
                ax.set_ylabel(r"$x_{d} \sim d(x|\theta^{*})$", fontsize=25)
        else:
            ax.set_xlabel("$x$", fontsize=25)
            ax.set_ylabel("$Q^2$", fontsize=25)
        ax.grid(True, alpha=0.3, which="both")
        ax.tick_params(which="both", direction="in")

    # Compute consistent axis limits for both scatter and histogram plots
    x_min = min(np.min(true_events_np[:, 0]), np.min(generated_events_np[:, 0]))
    x_max = max(np.max(true_events_np[:, 0]), np.max(generated_events_np[:, 0]))
    y_min = min(np.min(true_events_np[:, 1]), np.min(generated_events_np[:, 1]))
    y_max = max(np.max(true_events_np[:, 1]), np.max(generated_events_np[:, 1]))

    # Add small margin in log space
    x_margin = (np.log10(x_max) - np.log10(x_min)) * 0.05
    y_margin = (np.log10(y_max) - np.log10(y_min)) * 0.05
    
    x_lim = [10**(np.log10(x_min) - x_margin), 10**(np.log10(x_max) + x_margin)]
    y_lim = [10**(np.log10(y_min) - y_margin), 10**(np.log10(y_max) + y_margin)]

    # Plot scatter plots if requested
    if plot_type in ["scatter", "both"]:
        if plot_type == "both":
            ax_true_scat, ax_gen_scat = ax_true_scatter, ax_gen_scatter
        else:
            ax_true_scat, ax_gen_scat = ax_true, ax_gen

        # True events scatter
        ax_true_scat.scatter(
            true_events_np[:, 0],
            true_events_np[:, 1],
            color=true_color,
            alpha=0.3,
            s=1.5,
            edgecolors="none",
        )
        setup_axes(ax_true_scat, r"$\Xi[\theta^{*}]$ (True) - Scatter", False)
        ax_true_scat.set_xlim(x_lim)
        ax_true_scat.set_ylim(y_lim)

        # Generated events scatter
        ax_gen_scat.scatter(
            generated_events_np[:, 0],
            generated_events_np[:, 1],
            color=gen_color,
            alpha=0.3,
            s=1.5,
            edgecolors="none",
        )
        setup_axes(ax_gen_scat, rf"$\Xi[\hat{{\theta}}]$ (Generated) - Scatter", True)
        ax_gen_scat.set_xlim(x_lim)
        ax_gen_scat.set_ylim(y_lim)

    # Plot 2D histograms if requested
    if plot_type in ["histogram", "both"]:
        if plot_type == "both":
            ax_true_hist_ax, ax_gen_hist_ax = ax_true_hist, ax_gen_hist
        else:
            ax_true_hist_ax, ax_gen_hist_ax = ax_true, ax_gen

        # Create log-spaced bins using the same limits as scatter plots
        x_bins = np.logspace(
            np.log10(x_lim[0]), np.log10(x_lim[1]), bins
        )
        y_bins = np.logspace(
            np.log10(y_lim[0]), np.log10(y_lim[1]), bins
        )

        # True events histogram
        hist_true, _, _ = np.histogram2d(
            true_events_np[:, 0], true_events_np[:, 1], bins=[x_bins, y_bins]
        )
        hist_true = hist_true.T  # Transpose for correct orientation

        # Only show non-zero bins
        hist_true_masked = np.ma.masked_where(hist_true == 0, hist_true)

        im_true = ax_true_hist_ax.pcolormesh(
            x_bins, y_bins, hist_true_masked, cmap="Blues", norm=colors.LogNorm(vmin=1)
        )
        setup_axes(ax_true_hist_ax, r"$\Xi[\theta^{*}]$ (True) - 2D Histogram", False)
        ax_true_hist_ax.set_xlim(x_lim)
        ax_true_hist_ax.set_ylim(y_lim)

        # Add colorbar for true events
        cbar_true = plt.colorbar(im_true, ax=ax_true_hist_ax, fraction=0.046, pad=0.04)
        cbar_true.set_label("Event Count", fontsize=25)
        cbar_true.ax.tick_params(labelsize=25)

        # Generated events histogram
        hist_gen, _, _ = np.histogram2d(
            generated_events_np[:, 0], generated_events_np[:, 1], bins=[x_bins, y_bins]
        )
        hist_gen = hist_gen.T  # Transpose for correct orientation

        # Only show non-zero bins
        hist_gen_masked = np.ma.masked_where(hist_gen == 0, hist_gen)

        im_gen = ax_gen_hist_ax.pcolormesh(
            x_bins, y_bins, hist_gen_masked, cmap="Oranges", norm=colors.LogNorm(vmin=1)
        )
        setup_axes(
            ax_gen_hist_ax, rf"$\Xi[\hat{{\theta}}]$ (Generated) - 2D Histogram", True
        )
        ax_gen_hist_ax.set_xlim(x_lim)
        ax_gen_hist_ax.set_ylim(y_lim)

        # Add colorbar for generated events
        cbar_gen = plt.colorbar(im_gen, ax=ax_gen_hist_ax, fraction=0.046, pad=0.04)
        cbar_gen.set_label("Event Count", fontsize=25)
        cbar_gen.ax.tick_params(labelsize=25)

    # # Add overall title
    # if plot_type == 'both':
    #     fig.suptitle('Event Distribution Analysis: Scatter & Histogram Views', fontsize=16, y=0.95)
    # elif plot_type == 'scatter':
    #     fig.suptitle('Event Distribution Analysis: Scatter View', fontsize=16, y=0.95)
    # else:
    #     fig.suptitle('Event Distribution Analysis: Histogram View', fontsize=16, y=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_validation_dataset_batch(args, problem, device, num_samples=1000):
    """
    Load a batch from the validation dataset using precomputed data.

    **REFACTORED**: Now uses robust precomputed data loading with automatic generation
    if data is missing. Removes silent fallback to deprecated generate_data function.

    Args:
        args: Argument namespace containing validation parameters
        problem: Problem type string
        device: PyTorch device
        num_samples: Number of samples to load

    Returns:
        thetas: [n_samples, param_dim] - parameter vectors
        xs: [n_samples, n_events, feature_dim] - event data (already feature engineered)

    Raises:
        RuntimeError: If precomputed data cannot be loaded or generated

    Note:
        The returned xs data is already feature engineered as it was saved with
        engineering applied during precomputed data generation. Do NOT apply
        feature engineering again to avoid double processing.
    """
    # Default validation samples from parameter_prediction.py
    val_samples = getattr(args, "val_samples", 1000)

    # Construct validation data path
    val_data_dir = getattr(args, "precomputed_data_dir", "precomputed_data")

    # Ensure precomputed data exists, generate if needed
    try:
        print(
            f"📂 Loading validation batch for {problem} with {num_samples} samples..."
        )
        generate_precomputed_data_if_needed(
            problem=problem,
            num_samples=val_samples,
            num_events=args.num_events,
            n_repeat=1,  # Validation data typically uses n_repeat=1
            output_dir=val_data_dir,
        )

        # Load precomputed validation dataset
        from precomputed_datasets import PrecomputedDataset

        val_dataset = PrecomputedDataset(
            val_data_dir,
            problem,
            shuffle=False,
            exact_ns=val_samples,
            exact_ne=args.num_events,
            exact_nr=1,
        )

        # Sample num_samples from validation dataset
        actual_num_samples = min(num_samples, len(val_dataset))
        thetas_list = []
        xs_list = []

        for i in range(actual_num_samples):
            theta, events = val_dataset[i]
            thetas_list.append(theta)
            # events is [n_repeat, num_events, feature_dim], take first repeat
            xs_list.append(events[0])  # [num_events, feature_dim]

        thetas = torch.stack(thetas_list).to(device)  # [n_samples, param_dim]
        xs = torch.stack(xs_list).to(device)  # [n_samples, num_events, feature_dim]

        print(
            f"✅ Loaded validation batch: thetas.shape={thetas.shape}, xs.shape={xs.shape}"
        )
        print(
            f"ℹ️  Note: Data is already feature engineered - do not apply engineering again"
        )
        return thetas, xs

    except Exception as e:
        print(f"❌ Could not load precomputed validation dataset: {e}")
        raise RuntimeError(
            f"Failed to load validation data for {problem}. "
            f"Precomputed data loading failed: {e}. "
            f"Please check precomputed_datasets.py and generate_precomputed_data.py"
        )


def extract_latents_from_data(pointnet_model, args, problem, device, num_samples=1000):
    """
    Load data from validation dataset and extract latents using PointNet model.

    **REFACTORED**: Uses robust precomputed data loading and ensures feature engineering
    consistency. Precomputed data is already feature engineered, so this function
    now skips re-applying feature engineering to prevent double processing.

    Args:
        pointnet_model: Trained PointNet model for latent extraction
        args: Argument namespace containing data parameters
        problem: Problem type string
        device: PyTorch device
        num_samples: Number of samples to process

    Returns:
        latents: [n_samples, latent_dim] - extracted latent representations
        thetas: [n_samples, param_dim] - corresponding parameter vectors

    Note:
        This function may trigger automatic precomputed data generation if data
        is missing. Progress messages will be printed to inform about data operations.
    """
    # Load parameters and simulated events from validation dataset (already feature engineered)
    thetas, xs = load_validation_dataset_batch(args, problem, device, num_samples)

    # IMPORTANT: Precomputed data is already feature engineered during generation,
    # so we DO NOT apply feature engineering again to prevent double processing.
    # The data in xs is already in the correct format for PointNet input.
    print(
        f"ℹ️  Using precomputed data - feature engineering already applied during generation"
    )

    # xs shape: [n_samples, n_events, feature_dim] where feature_dim is already engineered
    feats = xs  # Use data as-is since it's already feature engineered

    # Extract latents using PointNet
    n_samples = thetas.shape[0]
    latents = []

    print(f"🧠 Extracting latents for {n_samples} samples...")
    with torch.no_grad():
        for i in range(n_samples):
            # feats[i] has shape [n_events, feature_dim]
            latent = pointnet_model(feats[i].unsqueeze(0))  # [1, latent_dim]
            latents.append(latent.cpu().numpy().squeeze(0))

    latents = np.array(latents)
    print(f"✅ Extracted latents: shape={latents.shape}")

    return latents, thetas.cpu().numpy()


def plot_latents_umap(
    latents,
    params,
    color_mode="single",
    param_idx=0,
    method="umap",
    save_path=None,
    show=True,
):
    """
    Plot latent vectors (n_samples x latent_dim) reduced to 2D via UMAP or t-SNE,
    colored by parameters (n_samples x param_dim).
    """
    # Reduce latents to 2D with robust fallbacks
    method = (method or "umap").lower()
    reducer = None
    if method == "umap":
        if umap is not None:
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            # Fallback to TSNE if UMAP isn't available
            if TSNE is not None:
                print("[plot_latents_umap] UMAP not available; falling back to t-SNE.")
                reducer = TSNE(n_components=2, random_state=42)
            else:
                raise ImportError(
                    "Neither UMAP nor TSNE is available. Please install 'umap-learn' or 'scikit-learn'."
                )
    elif method == "tsne":
        if TSNE is not None:
            reducer = TSNE(n_components=2, random_state=42)
        else:
            if umap is not None:
                print("[plot_latents_umap] TSNE not available; falling back to UMAP.")
                reducer = umap.UMAP(n_components=2, random_state=42)
            else:
                raise ImportError(
                    "Neither TSNE nor UMAP is available. Please install 'scikit-learn' or 'umap-learn'."
                )
    else:
        raise ValueError("method must be 'umap' or 'tsne'")
    emb = reducer.fit_transform(latents)

    # Determine coloring
    if color_mode == "single":
        color = params[:, param_idx]
        label = f"Parameter {param_idx}"
    elif color_mode == "mean":
        color = np.mean(params, axis=1)
        label = "Mean parameter"
    elif color_mode == "pca":
        if PCA is None:
            raise ImportError("PCA not available (scikit-learn missing)")
        pca = PCA(n_components=1)
        color = pca.fit_transform(params).flatten()
        label = "First principal component of parameters"
    else:
        raise ValueError("color_mode must be 'single', 'mean', or 'pca'")

    # Plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=color, cmap="viridis", s=30)
    plt.xlabel(f"{method.upper()} dim 1")
    plt.ylabel(f"{method.upper()} dim 2")
    plt.title(f"Latent space ({method.upper()}), colored by {label}")
    plt.colorbar(sc, label=label)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()


def plot_latents_all_params(
    latents, params, method="umap", save_path=None, show=True, param_names=None
):
    """
    Plot latent vectors (n_samples x latent_dim) reduced to 2D via UMAP or t-SNE,
    with one subplot per parameter dimension and a unique colormap for each.
    """
    # Reduce latents to 2D with robust fallbacks
    method = (method or "umap").lower()
    reducer = None
    if method == "umap":
        if umap is not None:
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            if TSNE is not None:
                print("[plot_latents_all_params] UMAP not available; falling back to t-SNE.")
                reducer = TSNE(n_components=2, random_state=42)
            else:
                raise ImportError(
                    "Neither UMAP nor TSNE is available. Please install 'umap-learn' or 'scikit-learn'."
                )
    elif method == "tsne":
        if TSNE is not None:
            reducer = TSNE(n_components=2, random_state=42)
        else:
            if umap is not None:
                print("[plot_latents_all_params] TSNE not available; falling back to UMAP.")
                reducer = umap.UMAP(n_components=2, random_state=42)
            else:
                raise ImportError(
                    "Neither TSNE nor UMAP is available. Please install 'scikit-learn' or 'umap-learn'."
                )
    else:
        raise ValueError("method must be 'umap' or 'tsne'")
    emb = reducer.fit_transform(latents)

    # distinct colormaps (extend or cycle automatically)
    cmaps = [
        "viridis",
        "plasma",
        "winter",
        "autumn",
        "inferno",
        "magma",
        "cividis",
        "cool",
        "hot",
        "spring",
        "summer",
        "Spectral",
        "turbo",
        "twilight",
        "hsv",
    ]

    n_params = params.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))
    if n_params == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        color = params[:, i]
        cmap = cmaps[i % len(cmaps)]  # cycle through colormaps
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=color, cmap=cmap, s=30)
        if i == 0:
            ax.set_xlabel(f"{method.upper()} dim 1", fontsize=20)
            ax.set_ylabel(f"{method.upper()} dim 2", fontsize=20)
        # ax.set_title(param_names[i] if param_names else f"Parameter {i}", fontsize=18)
        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()


def _bin_edges_log(
    evts_list, nx_bins=50, nQ2_bins=50, x_min=1e-4, x_max=1e-1, Q2_min=10.0, Q2_max=1e3
):
    """
    Build common log-space edges across all provided event clouds.
    Fallbacks let you clamp the plotting window for stability/comparability.
    """
    # Allow autoscale if user doesn't want fixed ranges
    xs = []
    Q2s = []
    for E in evts_list:
        if E is None or len(E) == 0:
            continue
        xs.append(E[:, 0])
        Q2s.append(E[:, 1])
    if xs:
        x_min = max(x_min, np.nanmax([np.nanmin(a[a > 0]) for a in xs]))
        x_max = min(x_max, np.nanmax([np.nanmax(a) for a in xs]))
    if Q2s:
        Q2_min = max(Q2_min, np.nanmax([np.nanmin(a[a > 0]) for a in Q2s]))
        Q2_max = min(Q2_max, np.nanmax([np.nanmax(a) for a in Q2s]))
    logx_edges = np.linspace(np.log(x_min), np.log(x_max), nx_bins + 1)
    logQ2_edges = np.linspace(np.log(Q2_min), np.log(Q2_max), nQ2_bins + 1)
    return logx_edges, logQ2_edges


def _hist2d_density_log(evts, logx_edges, logQ2_edges, total_xsec=None):
    """
    Histogram in (log x, log Q2), convert to differential rate by dividing by bin area (dx*dQ2).
    Optionally scale to match total cross section.
    """
    if evts is None or len(evts) == 0:
        H = np.zeros((len(logx_edges) - 1, len(logQ2_edges) - 1), dtype=float)
        return H, (logx_edges, logQ2_edges)

    H, xedges, q2edges = np.histogram2d(
        np.log(evts[:, 0]), np.log(evts[:, 1]), bins=(logx_edges, logQ2_edges)
    )
    # Convert counts to density via (dx*dQ2)
    # Precompute dx, dQ2 on linear scale for each bin
    dx = np.exp(xedges[1:]) - np.exp(xedges[:-1])  # (nx,)
    dQ2 = np.exp(q2edges[1:]) - np.exp(q2edges[:-1])  # (nQ2,)
    area = dx[:, None] * dQ2[None, :]
    density = np.divide(H, area, where=(area > 0))
    if total_xsec is not None and H.sum() > 0:
        density *= total_xsec / H.sum()
    return density, (xedges, q2edges)


def _theory_grid(idis, xedges, q2edges, rs, tar, mode="xQ2"):
    """
    Evaluate theory on bin centers defined by (xedges, q2edges).
    """
    nx = len(xedges) - 1
    nQ2 = len(q2edges) - 1
    out = np.zeros((nx, nQ2), dtype=float)
    # Bin centers in linear space
    x_centers = np.exp(0.5 * (xedges[:-1] + xedges[1:]))
    q2_centers = np.exp(0.5 * (q2edges[:-1] + q2edges[1:]))

    for i in tqdm(range(nx), desc="theory x bins", leave=False):
        x = x_centers[i]
        for j in range(nQ2):
            Q2 = q2_centers[j]
            out[i, j] = idis.get_diff_xsec(x, Q2, rs, tar, mode)
    return out


def _theory_grid_masked(idis, xedges, q2edges, rs, tar, mode, occupancy_counts):
    nx, nQ2 = len(xedges) - 1, len(q2edges) - 1
    out = np.zeros((nx, nQ2), dtype=float)
    x_centers = np.exp(0.5 * (xedges[:-1] + xedges[1:]))
    q2_centers = np.exp(0.5 * (q2edges[:-1] + q2edges[1:]))
    for i in range(nx):
        for j in range(nQ2):
            if occupancy_counts[i, j] > 0:  # <-- workbook’s guard
                val = idis.get_diff_xsec(x_centers[i], q2_centers[j], rs, tar, mode)
                out[i, j] = _to_scalar_xsec(val, mode_hint=mode)
    return out


def _to_scalar_xsec(v, mode_hint="xQ2"):
    import numpy as np

    try:
        import torch

        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(v, dict):
        if mode_hint in v:
            return float(np.asarray(v[mode_hint]).squeeze().reshape(()))
        return float(np.asarray(next(iter(v.values()))).squeeze().reshape(()))
    if isinstance(v, (list, tuple)):
        v = v[0]
    v = np.asarray(v).squeeze()
    if v.size == 0:
        return 0.0
    if v.ndim > 0:
        v = v.flat[0]
    return float(v)


# 0) Filter bad/zero/non-finite events before logging
def _valid_evts(ev):
    if ev is None:
        return None
    ev = np.asarray(ev)
    m = np.isfinite(ev).all(axis=1) & (ev[:, 0] > 0) & (ev[:, 1] > 0)
    return ev[m]


def safe_log_levels(A, n=60, lo_pct=1.0, hi_pct=99.0, default=(1e-6, 1.0)):
    A = np.asarray(A, dtype=float)

    # keep only positive finite values
    A = np.where(np.isfinite(A) & (A > 0), A, np.nan)

    # pick percentiles to avoid extreme outliers
    vmin = np.nanpercentile(A, lo_pct)
    vmax = np.nanpercentile(A, hi_pct)

    # fallback if bad or empty
    if (
        not np.isfinite(vmin)
        or not np.isfinite(vmax)
        or vmin <= 0
        or vmax <= 0
        or vmin >= vmax
    ):
        vmin, vmax = default

    # log-spaced levels
    levels = 10 ** np.linspace(np.log10(vmin), np.log10(vmax), n)

    return levels


from typing import Optional, Tuple


def plot_PDF_distribution_single_same_plot_mceg(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,  # kept for backward compatibility
    laplace_model=None,
    problem="simplified_dis",
    Q2_slices=None,  # list of Q² values to show; if None we auto-pick
    plot_IQR=False,  # used only for MC mode (overlay disabled by default)
    save_dir=None,
    nx=100,
    nQ2=100,
    n_events=1000000,
    max_Q2_for_plot=100.0,
    # LoTV options: if provided, overlay LoTV mean +/- std on Q2 slice plots
    per_boot_posterior_samples=None,
    lotv_n_theta_per_boot=20,
    lotv_num_events=20000,
    lotv_nx=30,
    lotv_nQ2=20,
):
    """
    Enhanced mceg4dis-compatible PDF plotting function with 2D histogram binning in log(x), log(Q2) space.

    This function implements the mceg4dis-compatible PDF plotting approach:
    - Uses numpy.histogram2d to bin events in log(x), log(Q2)
    - For each bin, computes x, Q2, dx, dQ2 using bin edges
    - For each bin: true[i,j] = idis.get_diff_xsec(x,Q2,...), reco[i,j] = hist[0][i,j]/dx/dQ2, stat[i,j] = sqrt(hist[0][i,j])/dx/dQ2
    - Normalizes reco/stat by total_xsec/np.sum(hist[0])
    - Plots slices in Q2, with log-scaled axes and error bars

    Supports both 'mceg' and 'mceg4dis' problem types with identical functionality.
    """
    # -------- setup ----------
    model.eval()
    pointnet_model.eval()
    # Ensure simulator classes are available in this scope
    try:
        from simulator import MCEGSimulator, SimplifiedDIS
    except Exception:
        SimplifiedDIS = MCEGSimulator = None
    if problem in ["mceg", "mceg4dis"]:
        if MCEGSimulator is None:
            raise RuntimeError(
                "MCEGSimulator not available; ensure simulator.py and mceg deps are installed"
            )
        simulator = MCEGSimulator(torch.device("cpu"))
    elif problem == "simplified_dis":
        simulator = SimplifiedDIS(torch.device("cpu"))

    # Make sure params live on correct device
    true_params = true_params.to(device)

    # -------- initialize theory components ----------

    mellin = MELLIN(npts=8)
    alphaS = ALPHAS()
    eweak = EWEAK()
    pdf = PDF(mellin, alphaS)

    # -------- sample events for the reconstructed histogram ----------
    with torch.no_grad():
        events = simulator.sample(
            true_params.detach().cpu(), n_events
        )  # expected shape (N, 2) = [x, Q2]
    events = np.asarray(events)
    x_ev = events[:, 0]
    Q2_ev = events[:, 1]

    xs_tensor = torch.tensor(events, dtype=torch.float32, device=device)

    # print(f"🔧 [FIX] mceg feature engineering - problem: {problem}")
    # print(f"🔧 [FIX] Raw events shape: {xs_tensor.shape}")

    if problem not in ["mceg", "mceg4dis"]:
        xs_tensor = log_feature_engineering(xs_tensor)
        # print(f"🔧 [FIX] After log_feature_engineering: {xs_tensor.shape}")
    else:
        # FIXED: Apply log_feature_engineering for mceg/mceg4dis to match training
        from utils import log_feature_engineering

        xs_tensor = log_feature_engineering(xs_tensor).float()
        # print(f"✅ [FIXED] mceg/mceg4dis: Applied log_feature_engineering - shape: {xs_tensor.shape}")

    # print(f"🔧 [FIX] Final tensor shape for PointNet: {xs_tensor.shape}")
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))
    # print(f"🔧 [FIX] Latent embedding shape: {latent_embedding.shape}")

    # Now prediction should work correctly with proper input dimensions
    theta_pred = model(latent_embedding).cpu().squeeze(0).detach()
    # print(f"✅ [FIXED] Predicted parameters: {theta_pred}")

    new_cpar = pdf.get_current_par_array()[::]
    # Assume parameters are only corresponding to 'uv1' parameters
    if not isinstance(theta_pred, torch.Tensor):
        new_cpar[4:8] = theta_pred
    else:
        new_cpar[4:8] = theta_pred.cpu().numpy()  # Update uv1 parameters
    pdf.setup(new_cpar)
    idis = THEORY(mellin, pdf, alphaS, eweak)
    new_cpar_true = pdf.get_current_par_array()[::]
    new_cpar_true[4:8] = (
        true_params.cpu().numpy()
        if isinstance(true_params, torch.Tensor)
        else true_params
    )
    pdf_true = PDF(mellin, alphaS)
    pdf_true.setup(new_cpar_true)
    idis_true = THEORY(mellin, pdf_true, alphaS, eweak)
    mceg = MCEG(idis, rs=140, tar="p", W2min=10, nx=nx, nQ2=nQ2)
    mceg_true = MCEG(idis_true, rs=140, tar="p", W2min=10, nx=nx, nQ2=nQ2)
    events_pred = mceg.gen_events(n_events, verb=False)

    events = mceg_true.gen_events(n_events, verb=False)
    evts = _valid_evts(events)
    evts_pred = _valid_evts(events_pred)
    if evts is None or len(evts) == 0:
        raise ValueError("No valid reco events with positive x and Q2.")

    # IMPLEMENTATION: Use numpy.histogram2d with proper binning in log(x), log(Q2) space
    # Create log-space bin edges for better mceg4dis compatibility
    x_min, x_max = 1e-4, 1e-1
    Q2_min, Q2_max = 10.0, min(max_Q2_for_plot, 1000.0)

    # Use nx, nQ2 parameters for consistent binning
    logx_edges = np.linspace(np.log(x_min), np.log(x_max), nx + 1)
    logQ2_edges = np.linspace(np.log(Q2_min), np.log(Q2_max), nQ2 + 1)

    hist = np.histogram2d(
        np.log(evts[:, 0]), np.log(evts[:, 1]), bins=(logx_edges, logQ2_edges)
    )
    true = np.zeros(hist[0].shape)
    reco = np.zeros(hist[0].shape)
    gen = np.zeros(hist[0].shape)
    for i, j in tqdm(
        (a, b) for a in range(hist[1].shape[0] - 1) for b in range(hist[2].shape[0] - 1)
    ):
        if hist[0][i, j] > 0:
            x = np.exp(0.5 * (hist[1][i] + hist[1][i + 1]))
            Q2 = np.exp(0.5 * (hist[2][j] + hist[2][j + 1]))
            tval = idis_true.get_diff_xsec(x, Q2, mceg_true.rs, mceg_true.tar, "xQ2")
            # get_diff_xsec may return (value, clip_alert) or value; handle both
            if isinstance(tval, tuple) or isinstance(tval, list):
                true[i, j] = float(tval[0])
            else:
                true[i, j] = float(tval)

            dx = np.exp(hist[1][i + 1]) - np.exp(hist[1][i])
            dQ2 = np.exp(hist[2][j + 1]) - np.exp(hist[2][j])
            reco[i, j] = hist[0][i, j] / dx / dQ2
            gval = idis.get_diff_xsec(x, Q2, mceg.rs, mceg.tar, "xQ2")
            if isinstance(gval, tuple) or isinstance(gval, list):
                gen[i, j] = float(gval[0])
            else:
                gen[i, j] = float(gval)

    reco *= mceg_true.total_xsec / np.sum(hist[0])
    gen *= mceg.total_xsec / np.sum(
        hist[0]
    )  # Fixed: use hist[0] instead of gen for consistency

    nrows, ncols = 1, 2
    AX = []
    fig = py.figure(figsize=(ncols * 7, nrows * 5))
    ax = py.subplot(nrows, ncols, 1)
    AX.append(ax)
    c = ax.pcolor(hist[1], hist[2], true.T, norm=matplotlib.colors.LogNorm())
    ax = py.subplot(nrows, ncols, 2)
    AX.append(ax)
    c = ax.pcolor(hist[1], hist[2], gen.T, norm=matplotlib.colors.LogNorm())
    for ax in AX:
        ax.tick_params(axis="both", which="major", labelsize=20, direction="in")
        ax.set_ylabel(r"$Q^2$", size=30)
        ax.set_xlabel(r"$x$", size=30)
        # fewer x-ticks for clarity
        ax.set_xticks(np.log([1e-3, 1e-1]))
        ax.set_xticklabels([r"$10^{-3}$", r"$10^{-1}$"])
        ax.set_yticks(np.log([10, 100]))
        ax.set_yticklabels([r"$10$", r"$100$"])
    AX[0].text(0.1, 0.8, r"$\rm True$", transform=AX[0].transAxes, size=30)
    AX[1].text(0.1, 0.8, r"$\rm Gen$", transform=AX[1].transAxes, size=30)
    py.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        py.savefig(
            os.path.join(save_dir, "PDF_2D_distribution_mceg_oldstyle.png"), dpi=300
        )

    nrows, ncols = 1, 3
    AX = []

    fig = py.figure(figsize=(ncols * 6, nrows * 5))
    cmap = "gist_rainbow"

    def _safe_contour_panel(ax, x_edges, y_edges, Z, title=None):
        """Attempt to contour Z on the grid defined by x_edges/y_edges.
        If contouring fails or yields no visible levels, fall back to a pcolor
        visualization with a LogNorm and print diagnostic summary stats.
        Returns the matplotlib artist (ContourSet or QuadMesh) or None."""
        # Prepare masked positive data
        Zp = np.where(np.isfinite(Z) & (Z > 0), Z, np.nan)
        if not np.isfinite(Zp).any():
            ax.text(
                0.5,
                0.5,
                "No positive data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            if title:
                ax.set_title(title)
            return None

        # Compute robust levels
        try:
            levels = safe_log_levels(Zp, n=60)
            if (
                not np.isfinite(levels).all()
                or levels[0] <= 0
                or levels[0] >= levels[-1]
            ):
                raise ValueError("degenerate levels")
        except Exception:
            levels = 10.0 ** np.linspace(-6, 0, 60)

        try:
            norm = matplotlib.colors.LogNorm(vmin=levels[0], vmax=levels[-1])
            cs = ax.contour(
                x_edges[:-1], y_edges[:-1], Zp.T, levels=levels, cmap=cmap, norm=norm
            )
            if getattr(cs, "collections", None) is None or len(cs.collections) == 0:
                raise RuntimeError("empty contour collections")
            if title:
                ax.set_title(title)
            return cs
        except Exception as e:
            # Fallback to pcolor with log norm using robust vmin/vmax
            pos = Zp[np.isfinite(Zp)]
            try:
                vmin = float(np.nanpercentile(pos, 1.0))
                vmax = float(np.nanpercentile(pos, 99.0))
                if not (np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax):
                    raise ValueError("bad vmin/vmax")
            except Exception:
                vmin, vmax = np.nanmin(pos), np.nanmax(pos)
                if not (np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax):
                    vmin, vmax = 1e-12, 1.0

            try:
                pcm = ax.pcolor(
                    x_edges,
                    y_edges,
                    Zp.T,
                    norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                    cmap=cmap,
                )
                # diagnostics: annotate axis and stdout
                summary = dict(
                    min=float(np.nanmin(pos)),
                    p1=float(np.nanpercentile(pos, 1.0)),
                    med=float(np.nanmedian(pos)),
                    p99=float(np.nanpercentile(pos, 99.0)),
                    max=float(np.nanmax(pos)),
                    npos=int(pos.size),
                )
                diag_txt = f"min={summary['min']:.3e}, med={summary['med']:.3e}, max={summary['max']:.3e}, npos={summary['npos']}"
                ax.set_title(diag_txt, fontsize=8)
                print(f"[contour fallback] {diag_txt}; original error: {e}")
                return pcm
            except Exception as ee:
                ax.text(
                    0.5,
                    0.5,
                    f"Contour/pcolor failed",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                print(f"[contour fallback] both contour and pcolor failed: {ee}")
                return None

    ax = py.subplot(nrows, ncols, 1)
    AX.append(ax)
    _safe_contour_panel(ax, hist[1], hist[2], true, title="True")
    ax = py.subplot(nrows, ncols, 2)
    AX.append(ax)
    _safe_contour_panel(ax, hist[1], hist[2], gen, title="Gen")
    for ax in AX:
        ax.tick_params(axis="both", which="major", labelsize=20, direction="in")
        ax.set_ylabel(r"$Q^2$", size=30)
        ax.set_xlabel(r"$x$", size=30)
        # fewer x-ticks for better readability
        ax.set_xticks(np.log([1e-3, 1e-1]))
        ax.set_xticklabels([r"$10^{-3}$", r"$10^{-1}$"])
        ax.set_yticks(np.log([10, 100]))
    ax.set_yticklabels([r"$10$", r"$100$"])
    AX[0].text(0.1, 0.8, r"$\rm True$", transform=AX[0].transAxes, size=30)
    AX[1].text(0.1, 0.8, r"$\rm Gen$", transform=AX[1].transAxes, size=30)
    py.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        py.savefig(
            os.path.join(save_dir, "PDF_2D_distribution_mceg_contour.png"), dpi=300
        )

    # IMPLEMENTATION: Q2 slice plotting with log-scaled axes and error bars as required
    # ===================================================================================
    print(f"🔧 [MCEG4DIS] Implementing Q² slice plotting for mceg4dis compatibility")

    Q2_slices = Q2_slices or [0.5, 1.0, 1.5, 2.0, 10.0, 50.0, 200.0]

    # Define reasonable x and Q2 ranges based on the histogram data
    x_min, x_max = 1e-4, 1e-1
    Q2_min, Q2_max = 10.0, min(max_Q2_for_plot, 1000.0)

    # Filter Q2_slices to be within our data range
    Q2_slices = [q for q in Q2_slices if Q2_min <= q <= Q2_max]

    if not Q2_slices:
        print(
            f"⚠️ [MCEG4DIS] No valid Q2 slices in range [{Q2_min}, {Q2_max}], using default range"
        )
        Q2_slices = [10.0, 50.0, 100.0]

    print(f"✅ [MCEG4DIS] Creating Q² slice plots for Q² values: {Q2_slices}")

    # Extract bin centers for reference
    x_centers = np.exp(0.5 * (hist[1][:-1] + hist[1][1:]))
    Q2_centers = np.exp(0.5 * (hist[2][:-1] + hist[2][1:]))

    # Create Q2 slice plot with log-scaled axes as specified
    fig, ax = py.subplots(figsize=(10, 8))
    color_palette = py.cm.viridis_r(np.linspace(0, 1, len(Q2_slices)))

    # Define x range for slice plotting using logspace as specified
    x_slice_vals = np.logspace(np.log10(x_min), np.log10(x_max), 200)

    for i, Q2_fixed in enumerate(Q2_slices):
        print(f"🔧 [MCEG4DIS] Processing Q² slice = {Q2_fixed}")

        # Compute theoretical values at this Q² slice
        true_slice = np.zeros_like(x_slice_vals)
        pred_slice = np.zeros_like(x_slice_vals)

        for j, x_val in enumerate(x_slice_vals):
            # get_diff_xsec returns (value, clip_alert) — unpack the first element
            try:
                tval, _ = idis_true.get_diff_xsec(
                    x_val, Q2_fixed, mceg_true.rs, mceg_true.tar, "xQ2"
                )
            except Exception:
                # ensure we don't break on unexpected returns
                tval = idis_true.get_diff_xsec(
                    x_val, Q2_fixed, mceg_true.rs, mceg_true.tar, "xQ2"
                )
                if isinstance(tval, tuple) or isinstance(tval, list):
                    tval = tval[0]
            try:
                pval, _ = idis.get_diff_xsec(x_val, Q2_fixed, mceg.rs, mceg.tar, "xQ2")
            except Exception:
                pval = idis.get_diff_xsec(x_val, Q2_fixed, mceg.rs, mceg.tar, "xQ2")
                if isinstance(pval, tuple) or isinstance(pval, list):
                    pval = pval[0]
            # ensure scalars
            true_slice[j] = float(tval)
            pred_slice[j] = float(pval)

        # Plot true curve
        ax.plot(
            x_slice_vals,
            true_slice,
            color=color_palette[i],
            linestyle="-",
            linewidth=2.5,
            label=f"True Q²={Q2_fixed}",
            alpha=0.8,
        )

        # Plot predicted curve
        ax.plot(
            x_slice_vals,
            pred_slice,
            color=color_palette[i],
            linestyle="--",
            linewidth=2,
            label=f"Pred Q²={Q2_fixed}",
            alpha=0.8,
        )

        # Add error bars from statistical uncertainty using reco data
        # Find the Q2 index closest to Q2_fixed for extracting statistical errors
        Q2_idx = np.argmin(np.abs(Q2_centers - Q2_fixed))
        if Q2_idx < len(Q2_centers):
            # Extract reco values and statistical uncertainties for this Q2 slice
            x_err_vals = x_centers
            reco_vals = reco[:, Q2_idx]

            # Compute statistical errors: stat[i,j] = sqrt(hist[0][i,j])/dx/dQ2
            stat_err_vals = np.zeros_like(reco_vals)
            for k in range(len(x_centers)):
                if hist[0][k, Q2_idx] > 0:
                    dx = np.exp(hist[1][k + 1]) - np.exp(hist[1][k])
                    dQ2 = np.exp(hist[2][Q2_idx + 1]) - np.exp(hist[2][Q2_idx])
                    stat_err_vals[k] = np.sqrt(hist[0][k, Q2_idx]) / (dx * dQ2)
                    # Apply same normalization as reco
                    stat_err_vals[k] *= mceg_true.total_xsec / np.sum(hist[0])

            # Only plot error bars where we have significant statistics
            mask = reco_vals > 0
            if np.any(mask):
                ax.errorbar(
                    x_err_vals[mask],
                    reco_vals[mask],
                    yerr=stat_err_vals[mask],
                    color=color_palette[i],
                    fmt="o",
                    markersize=3,
                    alpha=0.6,
                    label=f"Reco±stat Q²={Q2_fixed}",
                )
            mask = reco_vals > 0
            if np.any(mask):
                ax.errorbar(
                    x_err_vals[mask],
                    reco_vals[mask],
                    yerr=stat_err_vals[mask],
                    color=color_palette[i],
                    fmt="o",
                    markersize=3,
                    alpha=0.6,
                    label=f"Reco±stat Q²={Q2_fixed}",
                )

    # Optional: overlay LoTV-derived mean +/- std bands if per-boot posterior samples provided
    if per_boot_posterior_samples is not None:
        try:
            from plotting_UQ_helpers import compute_function_lotv_for_mceg

            # Use a lightweight simulator instance for compute_function_lotv_for_mceg
            try:
                from simulator import MCEGSimulator

                lotv_sim = MCEGSimulator(device=torch.device("cpu"))
            except Exception:
                lotv_sim = simulator if "simulator" in locals() else None
            if lotv_sim is None:
                raise RuntimeError("No MCEG simulator available for LoTV computation")

            lotv = compute_function_lotv_for_mceg(
                lotv_sim,
                per_boot_posterior_samples=per_boot_posterior_samples,
                n_theta_per_boot=lotv_n_theta_per_boot,
                num_events=lotv_num_events,
                nx=lotv_nx,
                nQ2=lotv_nQ2,
                device="cpu",
            )

            mean2d = lotv["mean"]  # (nx_l, nQ2_l)
            total2d = lotv["total_var"]  # (nx_l, nQ2_l)

            # Map lotv bins to our plotting Q2_slices by nearest Q2 center
            # Use lotv bin centers
            lotv_x_centers = np.arange(
                mean2d.shape[0]
            )  # indices; compute real x centers unknown -> use x_centers if sizes match
            if mean2d.shape[0] == x_centers.shape[0]:
                lotv_x = x_centers
            else:
                # fallback: use a linspace over plotting x range
                lotv_x = np.logspace(
                    np.log10(x_centers.min()),
                    np.log10(x_centers.max()),
                    mean2d.shape[0],
                )

            lotv_Q2_centers = np.arange(mean2d.shape[1])
            # If lotv nQ2 matches our Q2_centers, use Q2_centers mapping, else map by index
            if mean2d.shape[1] == Q2_centers.shape[0]:
                lotv_Q2_vals = Q2_centers
            else:
                lotv_Q2_vals = np.linspace(
                    Q2_centers.min(), Q2_centers.max(), mean2d.shape[1]
                )

            # For each Q2 slice, overlay mean +/- sqrt(total_var) on the same axis using bin centers
            for i, Q2_fixed in enumerate(Q2_slices):
                # find nearest lotv Q2 index
                qidx = int(np.argmin(np.abs(lotv_Q2_vals - Q2_fixed)))
                mean_col = mean2d[:, qidx]
                std_col = np.sqrt(total2d[:, qidx])
                # plot as line + band on top of existing Q2 slice plot
                ax.plot(
                    lotv_x,
                    mean_col,
                    color="k",
                    linewidth=1.5,
                    linestyle="-.",
                    alpha=0.9,
                    label=(f"LoTV mean Q²={Q2_fixed}" if i == 0 else None),
                )
                ax.fill_between(
                    lotv_x,
                    mean_col - std_col,
                    mean_col + std_col,
                    color="k",
                    alpha=0.12,
                    label=(f"LoTV ±1σ" if i == 0 else None),
                )
        except Exception as e:
            print(f"⚠️ Could not compute/display LoTV overlay for mceg plots: {e}")

    # Format the plot with log-scaled axes as specified in requirements
    # (Intentionally no LoTV computation here) Keep plotting based on empirical histograms

    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("Differential Cross Section", fontsize=16)
    ax.set_title("PDF Q² Slices - mceg4dis Compatible", fontsize=18)
    ax.set_xscale("log")  # log-scaled axes as required
    ax.set_yscale("log")  # log-scaled axes as required
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    py.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        slice_plot_path = os.path.join(save_dir, "PDF_Q2_slices_mceg4dis.png")
        py.savefig(slice_plot_path, dpi=300, bbox_inches="tight")
        print(f"✅ [MCEG4DIS] Q² slice plot saved to: {slice_plot_path}")

    py.close(fig)
    print(f"✅ [MCEG4DIS] Enhanced mceg4dis-compatible PDF plotting completed")
    print(
        f"✅ [MCEG4DIS] Generated: 2D histograms + Q² slices with log-scaled axes and error bars"
    )

def plot_combined_uncertainty_PDF_distribution(
    model,
    pointnet_model,
    true_params,
    device,
    num_events,
    n_bootstrap,
    laplace_model=None,
    problem="simplified_dis",
    save_dir=None,
):
    """
    Plot PDF distributions with function-level uncertainty quantification combining
    Laplace approximation (model uncertainty) and bootstrapping (data uncertainty).

    **KEY CHANGE**: This function now focuses on uncertainty over the predicted
    functions (u(x), d(x), q(x)) at each x-point, rather than uncertainty over
    the model parameters themselves.

    Function-level uncertainty aggregation:
    1. For each bootstrap sample + Laplace uncertainty:
       - Generates independent event sets from true parameters
       - Predicts parameter distribution (mean ± Laplace std)
       - Evaluates PDF functions f(x|θ) for multiple θ samples from predicted distribution
       - Stores f(x) values at each x-point across all samples

    2. Pointwise uncertainty computation for each x:
       - Aggregate all PDF values f(x) from bootstrap and Laplace samples
       - Compute mean and standard deviation of f(x) at each x
       - total_variance(x) = variance_of_bootstrap_means(x) + mean_laplace_variance(x)
       - Uncertainty bands reflect function uncertainty, not parameter uncertainty

    3. Uncertainty decomposition at each x:
       - Data uncertainty: variance across bootstrap samples of mean PDF predictions
       - Model uncertainty: mean of Laplace-induced variance in PDF at each x
       - Combined: pointwise addition of variance sources

    Args:
        model: Trained model head for parameter prediction
        pointnet_model: Trained PointNet model for latent extraction
        true_params: Fixed true parameter values [tensor of shape (param_dim,)]
        device: Device to run computations on
        num_events: Number of events per bootstrap sample
        n_bootstrap: Number of bootstrap samples to generate
        laplace_model: Fitted Laplace approximation for model uncertainty (optional)
        problem: Problem type ('simplified_dis', 'mceg')
        save_dir: Directory to save plots (required)

    Returns:
        None (saves plots and analysis to save_dir)

    Saves:
        For simplified_dis:
            - function_uncertainty_pdf_up.png: u(x) with function-level uncertainty bands
            - function_uncertainty_pdf_down.png: d(x) with function-level uncertainty bands
            - function_uncertainty_breakdown_up.txt: Pointwise uncertainty breakdown for u(x)
            - function_uncertainty_breakdown_down.txt: Pointwise uncertainty breakdown for d(x)

        Common saves:
            - function_uncertainty_summary.png: Average uncertainty statistics across x
            - function_uncertainty_methodology.txt: Documentation of uncertainty computation

    Example Usage:
        # Simplified DIS with function-level uncertainty
        plot_combined_uncertainty_PDF_distribution(
            model=model,
            pointnet_model=pointnet_model,
            true_params=torch.tensor([2.0, 1.2, 2.0, 1.2]),
            device=device,
            num_events=100000,
            n_bootstrap=50,
            laplace_model=laplace_model,
            problem='simplified_dis',
            save_dir='./plots/function_uncertainty'
        )

    Notes:
        - **MAJOR CHANGE**: Uncertainty now computed over PDF functions at each x, not parameters
        - Provides pointwise uncertainty bands that reflect prediction uncertainty of f(x)
        - Combines data and model uncertainty sources in function space, not parameter space
        - More interpretable uncertainty for PDF predictions and physics applications
        - If laplace_model is None, uses bootstrap-only function-level uncertainty
    """
    if save_dir is None:
        raise ValueError(
            "save_dir must be specified for saving function uncertainty plots"
        )

    # Validate all inputs comprehensively
    validate_combined_uncertainty_inputs(
        model,
        pointnet_model,
        true_params,
        device,
        num_events,
        n_bootstrap,
        problem,
        save_dir,
    )

    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from tqdm import tqdm

    os.makedirs(save_dir, exist_ok=True)

    print(
        f"Starting FUNCTION-LEVEL uncertainty analysis with {n_bootstrap} bootstrap samples..."
    )
    print(
        "🔄 KEY CHANGE: Computing uncertainty over predicted functions f(x), not parameters θ"
    )
    if laplace_model is not None:
        print(
            "  📊 Using Laplace approximation for model uncertainty in function space"
        )
    else:
        print(
            "  ⚠️  No Laplace model provided - using bootstrap-only function uncertainty"
        )

    # Initialize simulator based on problem type
    SimplifiedDIS, MCEGSimulator = get_simulator_module()

    if problem == "simplified_dis":
        if SimplifiedDIS is None:
            raise ImportError(
                "SimplifiedDIS not available - please install required dependencies"
            )
        simulator = SimplifiedDIS(device=torch.device("cpu"))
        param_names = [r"$a_u$", r"$b_u$", r"$a_d$", r"$b_d$"]
    elif problem in ["mceg", "mceg4dis"]:
        if MCEGSimulator is None:
            raise ImportError(
                "MCEGSimulator not available - please install required dependencies"
            )
        simulator = MCEGSimulator(device=torch.device("cpu"))
        param_names = [f"Param {i+1}" for i in range(len(true_params))]
    else:
        raise ValueError(f"Unknown problem type: {problem}")

    # Get feature engineering function
    log_feature_engineering = get_log_feature_engineering()

    model.eval()
    pointnet_model.eval()
    true_params = true_params.to(device)
    n_params = len(true_params)

    # Implement rigorous per-x Law-of-Total-Variance (LoTV) decomposition
    # using the dedicated helper from plotting_UQ_helpers for simplified_dis.

    if problem != "simplified_dis":
        raise NotImplementedError(
            "LoTV plotting is currently implemented for simplified_dis only"
        )

    # Build a callable that produces posterior parameter samples per bootstrap
    # by drawing Gaussian samples from the Laplace predictive at the latent of
    # a bootstrap-generated dataset. This keeps the semantics: within-boot = posterior; between-boot = data.
    def posterior_sampler_callable_factory(max_boot: int):
        def sampler(bootstrap_index: int):
            # Stop after the requested number of bootstraps
            if bootstrap_index >= max_boot:
                return None
            try:
                xs = simulator.sample(true_params.detach().cpu(), num_events)
                xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
                xs_tensor = log_feature_engineering(xs_tensor)
                latent = pointnet_model(xs_tensor.unsqueeze(0))

                # Obtain mean/std with Laplace; fallback to Gaussian head if needed
                mean_params, std_params = get_analytic_uncertainty(
                    model, latent, laplace_model
                )
                mean_params = mean_params.cpu().squeeze(0)
                std_params = std_params.cpu().squeeze(0)

                n_theta = 20
                samples = []
                for _ in range(n_theta):
                    samples.append(
                        (mean_params + std_params * torch.randn_like(mean_params))
                        .cpu()
                        .numpy()
                    )
                return np.stack(samples)
            except Exception:
                return None

        return sampler

    # Prepare simulator and x-grid
    from plotting_UQ_helpers import compute_function_lotv_for_simplified_dis
    x_grid = torch.linspace(1e-3, 1, 500).to(device)

    lotv_results = compute_function_lotv_for_simplified_dis(
        simulator,
        x_grid,
        posterior_sampler_callable=posterior_sampler_callable_factory(n_bootstrap),
        n_theta_per_boot=100,
        device=device,
    )

    # Build empirical ensembles f(x; theta^{(b,s)}) to compute percentile bands as in the methodology
    # We reuse the same posterior sampler to aggregate all draws across bootstraps.
    f_up_all = []
    f_down_all = []
    b_idx = 0
    sampler = posterior_sampler_callable_factory(n_bootstrap)
    while True:
        samples = sampler(b_idx)
        if samples is None:
            break
        samples = np.array(samples)
        if samples.size == 0:
            break
        # Evaluate f for each theta sample at this bootstrap
        for i in range(samples.shape[0]):
            theta = torch.tensor(samples[i], dtype=torch.float32, device=device)
            pdf = simulator.f(x_grid, theta)
            f_up_all.append(pdf["up"].detach().cpu().numpy().ravel())
            f_down_all.append(pdf["down"].detach().cpu().numpy().ravel())
        b_idx += 1
    f_up_all = np.array(f_up_all)  # (N, n_x)
    f_down_all = np.array(f_down_all)

    # Plot median with IQR bands (per-x aggregation) for up/down
    for fn_name, fn_label, color in [
        ("up", "u", "royalblue"),
        ("down", "d", "darkorange"),
    ]:
        # Extract LoTV decomposition for per-x std curves
        total_key = f"total_var_{fn_name}"
        between_key = f"between_var_{fn_name}"
        within_key = f"avg_within_var_{fn_name}"

        std_total = torch.tensor(lotv_results[total_key], dtype=torch.float32).sqrt()
        std_between = torch.tensor(lotv_results[between_key], dtype=torch.float32).sqrt()
        std_within = torch.tensor(lotv_results[within_key], dtype=torch.float32).sqrt()

        # True curve for reference
        simulator.init(true_params.squeeze().cpu())
        true_curve = getattr(simulator, fn_name)(x_grid).detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(
            x_grid.detach().cpu().numpy(),
            true_curve,
            label=rf"True ${fn_label}(x|\theta^*)$",
            color=color,
            linewidth=2.5,
        )

        # Prepare x array for plotting
        x_np = x_grid.detach().cpu().numpy()

        # IQR bands and median from empirical ensembles
        if fn_name == "up" and f_up_all.size > 0:
            p25 = np.nanpercentile(f_up_all, 25, axis=0)
            p50 = np.nanpercentile(f_up_all, 50, axis=0)
            p75 = np.nanpercentile(f_up_all, 75, axis=0)
            ax.fill_between(x_np, p25, p75, color="gray", alpha=0.20)
            ax.plot(x_np, p50, color="black", linestyle="-.", linewidth=1.5, label="Median (empirical)")
        if fn_name == "down" and f_down_all.size > 0:
            p25 = np.nanpercentile(f_down_all, 25, axis=0)
            p50 = np.nanpercentile(f_down_all, 50, axis=0)
            p75 = np.nanpercentile(f_down_all, 75, axis=0)
            ax.fill_between(x_np, p25, p75, color="gray", alpha=0.20)
            ax.plot(x_np, p50, color="black", linestyle="-.", linewidth=1.5, label="Median (empirical)")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(rf"${fn_label}(x|\theta)$")
        ax.set_xlim(1e-3, 1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend(frameon=False)
        # ax.set_title("Function-Space Combined Uncertainty (LoTV)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"function_uncertainty_lotv_{fn_name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Save per-x breakdown file with median and IQR
        breakdown_path = os.path.join(
            save_dir, f"function_uncertainty_lotv_breakdown_{fn_name}.txt"
        )
        
        # Compute median and IQR from empirical ensembles
        if fn_name == "up" and f_up_all.size > 0:
            median_vals = np.nanpercentile(f_up_all, 50, axis=0)
            p25_vals = np.nanpercentile(f_up_all, 25, axis=0)
            p75_vals = np.nanpercentile(f_up_all, 75, axis=0)
        elif fn_name == "down" and f_down_all.size > 0:
            median_vals = np.nanpercentile(f_down_all, 50, axis=0)
            p25_vals = np.nanpercentile(f_down_all, 25, axis=0)
            p75_vals = np.nanpercentile(f_down_all, 75, axis=0)
        else:
            median_vals = np.full(x_grid.numel(), np.nan)
            p25_vals = np.full(x_grid.numel(), np.nan)
            p75_vals = np.full(x_grid.numel(), np.nan)
        
        with open(breakdown_path, "w") as f:
            f.write(f"LoTV per-x breakdown for {fn_name}(x) - Median and IQR aggregation\n")
            f.write("=" * 60 + "\n")
            f.write(
                "Columns: x, median(x), IQR_25(x), IQR_75(x), sqrt(total_var), sqrt(between_var), sqrt(avg_within_var)\n"
            )
            for i in range(x_grid.numel()):
                f.write(
                    f"{x_grid[i].item():.6e} "
                    f"{median_vals[i]:.6e} "
                    f"{p25_vals[i]:.6e} "
                    f"{p75_vals[i]:.6e} "
                    f"{std_total[i].item():.6e} "
                    f"{std_between[i].item():.6e} "
                    f"{std_within[i].item():.6e}\n"
                )

    # Summary plot of decomposition curves and scalar uncertainty (mean_x std)
    try:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for j, fn_name in enumerate(["up", "down"]):
            mean_key = f"mean_{fn_name}"
            total_key = f"total_var_{fn_name}"
            between_key = f"between_var_{fn_name}"
            within_key = f"avg_within_var_{fn_name}"
            std_total = np.sqrt(np.array(lotv_results[total_key]))
            std_between = np.sqrt(np.array(lotv_results[between_key]))
            std_within = np.sqrt(np.array(lotv_results[within_key]))
            ax[j].plot(x_np, std_total, label="Total std. (LoTV)", color="crimson")
            ax[j].plot(x_np, std_between, label="Between std. (bootstrapping)", color="teal")
            ax[j].plot(x_np, std_within, label="AvgWithin std. (LA)", color="purple")
            ax[j].set_xscale("log")
            ax[j].set_xlim(1e-3, 1)
            ax[j].set_xlabel("x")
            ax[j].set_ylabel("std over f(x)")
            ax[j].grid(True, linestyle=":")
            if j == 0:
                ax[j].legend(frameon=False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "function_uncertainty_lotv_summary.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    except Exception:
        pass

    print(f"✅ Combined LoTV function-space uncertainty saved to {save_dir}")
    return lotv_results


def validate_combined_uncertainty_inputs(
    model,
    pointnet_model,
    true_params: torch.Tensor,
    device: torch.device,
    num_events: int,
    n_bootstrap: int,
    problem: str,
    save_dir: str,
) -> bool:
    """
    Validate inputs for combined uncertainty analysis function.

    Args:
        model: Model to validate
        pointnet_model: PointNet model to validate
        true_params: Parameter tensor to validate
        device: Device to validate
        num_events: Number of events to validate
        n_bootstrap: Bootstrap count to validate
        problem: Problem type to validate
        save_dir: Save directory to validate

    Returns:
        bool: True if all inputs are valid

    Raises:
        ValueError: If any input is invalid with descriptive message
    """
    # Validate models
    if model is None:
        raise ValueError("model cannot be None")
    if pointnet_model is None:
        raise ValueError("pointnet_model cannot be None")

    # Validate parameters
    if not isinstance(true_params, torch.Tensor):
        raise ValueError("true_params must be a torch.Tensor")
    if true_params.dim() != 1:
        raise ValueError("true_params must be 1-dimensional tensor")
    if len(true_params) == 0:
        raise ValueError("true_params cannot be empty")

    # Validate device
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device")

    # Validate counts
    if not isinstance(num_events, int) or num_events <= 0:
        raise ValueError("num_events must be a positive integer")
    if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if n_bootstrap > 1000:
        print(
            f"⚠️  Warning: n_bootstrap={n_bootstrap} is quite large and may take significant time"
        )

    # Validate problem type
    valid_problems = ["simplified_dis", "mceg"]
    if problem not in valid_problems:
        raise ValueError(f"problem must be one of {valid_problems}, got '{problem}'")

    # Validate save directory
    if save_dir is None:
        raise ValueError("save_dir cannot be None")
    if not isinstance(save_dir, str):
        raise ValueError("save_dir must be a string")

    # Check parameter dimensions match problem expectations
    expected_dims = {"simplified_dis": 4, "mceg": None}  # Variable

    if problem in expected_dims and expected_dims[problem] is not None:
        if len(true_params) != expected_dims[problem]:
            raise ValueError(
                f"For problem '{problem}', expected {expected_dims[problem]} parameters, got {len(true_params)}"
            )

    return True

def get_parameter_bounds_for_problem(problem):
    """
    Get parameter bounds for different problem types based on Dataset class definitions.

    Parameters:
    -----------
    problem : str
        Problem type ('simplified_dis', 'mceg')

    Returns:
    --------
    torch.Tensor
        Parameter bounds tensor of shape (n_params, 2) where bounds[:, 0] are lower bounds
        and bounds[:, 1] are upper bounds
    """
    if problem == "simplified_dis":
        # From DISDataset class: [[0.0, 5]] * theta_dim for theta_dim=4
        return torch.tensor(
            [
                [-1.0, 0.0],
                [0.0, 5.0],
                [-1.0, 0.0],
                [0.0, 5.0],
            ]
        )
    elif problem in ["mceg", "mceg4dis"]:
        # From MCEGDISDataset class
        return torch.tensor(
            [
                [-1.0, 10.0],
                [0.0, 10.0],
                [-10.0, 10.0],
                [-10.0, 10.0],
            ]
        )
    else:
        raise ValueError(
            f"Unknown problem type: {problem}. Supported: 'simplified_dis', 'mceg'"
        )


def get_simulator_for_problem(problem, device=None):
    """
    Get the appropriate simulator for the given problem type.

    Parameters:
    -----------
    problem : str
        Problem type ('simplified_dis', 'mceg')
    device : torch.device, optional
        Device to run the simulator on

    Returns:
    --------
    simulator
        The appropriate simulator instance
    """
    SimplifiedDIS, MCEGSimulator = get_simulator_module()

    if problem == "simplified_dis":
        if SimplifiedDIS is None:
            raise ImportError("Could not import SimplifiedDIS from simulator module")
        return SimplifiedDIS(device=device)
    elif problem in ["mceg", "mceg4dis"]:
        if MCEGSimulator is None:
            raise ImportError("Could not import MCEGSimulator from simulator module")
        return MCEGSimulator(device=device)
    else:
        raise ValueError(f"Unknown problem type: {problem}")


@torch.no_grad()
def generate_parameter_error_histogram(
    model,
    pointnet_model,
    device,
    n_draws=100,
    n_events=10000,
    problem="simplified_dis",
    laplace_model=None,
    save_path="parameter_error_histogram.png",
    param_names=None,
    return_data=False,
):
    """
    Automatically retrieves parameter bounds, samples parameter sets, generates events,
    runs inference pipeline, and creates publication-ready parameter error histograms.

    This is a comprehensive utility function that benchmarks model accuracy across the
    parameter space with minimal user input.

    ⚠️  **REPRODUCIBILITY WARNING**: This function requires extensive simulation to
    generate parameter error benchmarks. It samples multiple parameter sets and
    generates fresh simulated data for each. Results may vary between runs unless
    random seeds are fixed. For reproducible analysis, consider using precomputed
    validation data via extract_latents_from_data() for latent extraction.

    Parameters:
    -----------
    model : torch.nn.Module
        Trained inference model
    pointnet_model : torch.nn.Module
        Trained PointNet model for latent extraction
    device : torch.device
        Device to run computations on
    n_draws : int, default=100
        Number of parameter sets to sample and evaluate via simulation
    n_events : int, default=10000
        Number of events to generate per parameter set via simulation
    problem : str, default='simplified_dis'
        Problem type ('simplified_dis', 'mceg')
    laplace_model : optional
        Laplace approximation model for uncertainty quantification
    save_path : str, default="parameter_error_histogram.png"
        Path to save the histogram plot
    param_names : list of str, optional
        Parameter names for axis labels. Auto-generated if None.
    return_data : bool, default=False
        If True, return the true and predicted parameter lists

    Returns:
    --------
    None or tuple
        If return_data=True, returns (true_params_list, predicted_params_list)
        Otherwise saves the plot and returns None

    Raises:
    -------
    ImportError
        If required simulator modules are not available
    ValueError
        If problem type is not supported
    RuntimeError
        If model inference fails

    Examples:
    ---------
    >>> generate_parameter_error_histogram(
    ...     model, pointnet_model, device,
    ...     n_draws=200, n_events=50000,
    ...     problem='simplified_dis',
    ...     save_path='simplified_dis_errors.png'
    ... )
    """
    print(f"🎯 Starting parameter error histogram generation...")
    print(f"📊 Problem: {problem}")
    print(f"🎲 Parameter draws: {n_draws} (each requiring simulation)")
    print(f"🔢 Events per draw: {n_events}")
    print(f"⚠️  Note: This process uses extensive simulation for benchmarking")

    # Set models to evaluation mode
    model.eval()
    pointnet_model.eval()
    if laplace_model is not None:
        laplace_model.eval()

    # Get parameter bounds and simulator for the problem
    try:
        theta_bounds = get_parameter_bounds_for_problem(problem).to(device)
        simulator = get_simulator_for_problem(problem, device=device)
        print(f"   ✅ Parameter bounds: {theta_bounds.shape[0]} parameters")
    except Exception as e:
        raise RuntimeError(f"Failed to setup problem '{problem}': {str(e)}")

    # Get feature engineering function
    try:
        log_feature_engineering = get_log_feature_engineering()
        print(f"   ✅ Feature engineering function loaded")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not load log_feature_engineering: {e}")
        print(f"       Using identity function as fallback")
        log_feature_engineering = lambda x: x

    # Storage for results
    true_params_list = []
    predicted_params_list = []
    failed_samples = 0

    print(f"   🔄 Processing {n_draws} parameter samples...")

    # Process each parameter draw
    for i in tqdm(range(n_draws), desc="Generating parameter errors"):
        try:
            # Sample random parameters from bounds
            theta_raw = torch.rand(theta_bounds.shape[0], device=device)
            true_params = (
                theta_raw * (theta_bounds[:, 1] - theta_bounds[:, 0])
                + theta_bounds[:, 0]
            )

            # Generate events using the simulator
            try:
                if problem == "gaussian":
                    xs = simulator.sample(true_params, nevents=n_events)
                else:
                    xs = simulator.sample(true_params, n_events)

                # Handle different output shapes from simulators
                if xs.dim() == 1:
                    xs = xs.unsqueeze(0)
                if xs.shape[0] != n_events:
                    if xs.shape[1] == n_events:
                        xs = xs.t()  # transpose if needed
                    else:
                        # Pad or truncate to get correct number of events
                        if xs.shape[0] > n_events:
                            xs = xs[:n_events]
                        else:
                            # Pad with zeros if needed
                            pad_size = n_events - xs.shape[0]
                            if xs.dim() == 2:
                                padding = torch.zeros(
                                    pad_size, xs.shape[1], device=device
                                )
                            else:
                                padding = torch.zeros(pad_size, device=device)
                            xs = torch.cat([xs, padding], dim=0)

            except Exception as e:
                print(f"   ⚠️  Failed to generate events for sample {i}: {e}")
                failed_samples += 1
                continue

            # Apply feature engineering
            try:
                if problem == "simplified_dis":
                    xs_engineered = log_feature_engineering(xs).float()
                else:
                    xs_engineered = log_feature_engineering(xs).float()
                if not isinstance(xs_engineered, torch.Tensor):
                    xs_engineered = torch.tensor(
                        xs_engineered, device=device, dtype=torch.float32
                    )
                xs_engineered = xs_engineered.to(device)
            except Exception as e:
                print(f"   ⚠️  Feature engineering failed for sample {i}: {e}")
                failed_samples += 1
                continue

            # Extract latent features using PointNet
            try:
                # Add batch dimension if needed
                if xs_engineered.dim() == 2:
                    xs_batch = xs_engineered.unsqueeze(0)  # [1, n_events, n_features]
                else:
                    xs_batch = xs_engineered

                latent = pointnet_model(xs_batch)  # [1, latent_dim]

                if latent.dim() == 1:
                    latent = latent.unsqueeze(0)

            except Exception as e:
                print(f"   ⚠️  PointNet inference failed for sample {i}: {e}")
                failed_samples += 1
                continue

            # Predict parameters using the inference model
            try:
                if hasattr(model, "nll_mode") and model.nll_mode:
                    # Handle NLL mode (returns mean and log_var)
                    predicted_mean, predicted_log_var = model(latent)
                    predicted_params = predicted_mean.squeeze()
                else:
                    # Standard mode
                    predicted_params = model(latent).squeeze()

                # Handle Laplace uncertainty if available
                if laplace_model is not None:
                    try:
                        # Get uncertainty-aware predictions
                        with torch.no_grad():
                            laplace_samples = laplace_model.sample(
                                10, x=latent
                            )  # Sample from posterior
                            predicted_params = laplace_samples.mean(dim=0).squeeze()
                    except Exception as e:
                        print(
                            f"   ⚠️  Laplace inference failed for sample {i}, using standard prediction: {e}"
                        )

            except Exception as e:
                print(f"   ⚠️  Parameter prediction failed for sample {i}: {e}")
                failed_samples += 1
                continue

            # Store results
            true_params_list.append(true_params.detach().cpu())
            predicted_params_list.append(predicted_params.detach().cpu())

        except Exception as e:
            print(f"   ⚠️  Sample {i} failed with unexpected error: {e}")
            failed_samples += 1
            continue

    # Check if we have enough successful samples
    successful_samples = len(true_params_list)
    if successful_samples == 0:
        raise RuntimeError(
            "All parameter samples failed. Check model compatibility and data pipeline."
        )
    elif failed_samples > 0:
        print(
            f"   ⚠️  {failed_samples}/{n_draws} samples failed, proceeding with {successful_samples} successful samples"
        )

    print(f"   ✅ Successfully processed {successful_samples} parameter samples")

    # Generate the histogram plot using existing function
    try:
        plot_parameter_error_histogram(
            true_params_list=true_params_list,
            predicted_params_list=predicted_params_list,
            param_names=param_names,
            save_path=save_path,
            problem=problem,
        )
        print(f"   ✅ Parameter error histogram saved to: {save_path}")
    except Exception as e:
        print(f"   ❌ Failed to create histogram plot: {e}")
        raise

    # Print summary statistics
    if successful_samples > 0:
        true_params_array = torch.stack(true_params_list)
        predicted_params_array = torch.stack(predicted_params_list)

        abs_errors = torch.abs(predicted_params_array - true_params_array)
        rel_errors = abs_errors / (torch.abs(true_params_array) + 1e-8)

        print(f"\n📊 Error Statistics Summary:")
        print(f"   Mean absolute error: {abs_errors.mean():.4f}")
        print(f"   Mean relative error: {rel_errors.mean():.4f}")
        print(f"   Max absolute error: {abs_errors.max():.4f}")
        print(f"   Max relative error: {rel_errors.max():.4f}")

        for i in range(true_params_array.shape[1]):
            param_name = param_names[i] if param_names else f"θ_{i}"
            print(
                f"   {param_name}: MAE={abs_errors[:, i].mean():.4f}, MRE={rel_errors[:, i].mean():.4f}"
            )

    print(f"🎯 Parameter error histogram generation complete!")

    if return_data:
        return true_params_list, predicted_params_list


@torch.no_grad()
def plot_function_error_histogram_mceg(
    model,
    pointnet_model,
    device,
    n_draws=100,
    n_events=10000,
    problem="simplified_dis",  # default preserved
    laplace_model=None,
    save_path="function_error_histogram_mceg.png",
    param_names=None,
    return_data=False,
    nx=30,
    nQ2=20,
    Q2_slices=None,
    use_log_feature_engineering=True,
    verbose=True,
):
    """
    For each sampled parameter set:
      - simulate events (true)
      - infer parameters via the pipeline (PointNet -> model, optional laplace)
      - simulate events from inferred params (predicted)
      - compute log-space 2D histograms (x, Q2) and convert to density (counts / bin_area)
      - for the requested Q2_slices: select nearest Q2 bins and compute abs(true_density - pred_density)
      - per Q2 slice: average the entrywise absolute error across x bins
      - per draw: average the Q2-averaged errors into one scalar
    Finally: plot a histogram of the per-draw scalar errors. Optionally return raw arrays.

    Returns:
      If return_data: (true_params_list, predicted_params_list, per_draw_errors)
      Else: saves plot and returns None.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # sanity prints
    if verbose:
        print(f"🎯 Starting MCEG function error histogram (problem={problem})")
        print(f"🔁 n_draws={n_draws}, n_events={n_events}, bins=(nx={nx}, nQ2={nQ2})")

    model.eval()
    pointnet_model.eval()
    if laplace_model is not None:
        laplace_model.eval()

    # Try to obtain parameter bounds and simulator; provide fallbacks for 'mceg'
    try:
        theta_bounds = get_parameter_bounds_for_problem(problem).to(device)
        simulator = get_simulator_for_problem(problem, device=device)
    except Exception:
        # fallback for mceg or when helper is unavailable
        if problem == "mceg":
            try:
                from simulator import MCEGSimulator

                simulator = MCEGSimulator(device=device)
                # try to obtain bounds if helper exists; else set None
                try:
                    theta_bounds = get_parameter_bounds_for_problem(problem).to(device)
                except Exception:
                    theta_bounds = None
            except Exception as e:
                raise RuntimeError("Could not create MCEG simulator: " + str(e))
        else:
            raise RuntimeError(
                f"Failed to setup problem '{problem}' and no fallback available."
            )

    # Helpers: feature engineering, posterior sampler (if laplace), and density builder
    try:
        from utils import log_feature_engineering
    except Exception:
        # fallback identity if missing
        log_feature_engineering = lambda x: x

    # Posterior sampling wrapper used if laplace_model is provided
    def posterior_sampler(feats, pointnet, model_, laplace_, n_samples=100):
        # Default fallback: sample MAP (repeat) if a real sampler isn't available
        # If user has a specialized sampler,
        # their code should override this name in the module or we will use MAP repeats.
        try:
            # if there's a helper in user's namespace, use it
            from inference_utils import posterior_sampler as _ps

            return _ps(feats, pointnet, model_, laplace_, n_samples=n_samples)
        except Exception:
            # fallback: sample model.mean (or model output) repeated
            with torch.no_grad():
                feats_device = (
                    feats.to(device)
                    if isinstance(feats, torch.Tensor)
                    else torch.tensor(feats, device=device)
                )
                feats_pe = log_feature_engineering(feats_device).float().unsqueeze(0)
                lat = pointnet_model(feats_pe).detach()
                if laplace_ is not None:
                    try:
                        samples = laplace_.sample(
                            n_samples, x=lat
                        )  # shape [n_samples, param_dim]
                        return [s.detach().cpu().numpy() for s in samples]
                    except Exception:
                        pass
                pred = model_(lat)
                try:
                    mean = pred.mean(dim=0).detach().cpu().numpy()
                except Exception:
                    mean = pred.detach().cpu().numpy().ravel()
                return [mean for _ in range(n_samples)]

    # Histogram builder: returns density and bin edges (log-edges) using log variables consistent with collaborator style
    def build_density_from_events(evts_np, nx_local=nx, nQ2_local=nQ2):
        # evts_np: numpy array with columns [x, Q2, ...] or shape [N, 2]
        if evts_np.size == 0:
            hist = np.histogram2d(
                np.array([]), np.array([]), bins=(nx_local, nQ2_local)
            )
            counts = hist[0].astype(float)
            return np.zeros_like(counts), hist[1], hist[2]
        log_x = np.log(evts_np[:, 0])
        log_Q2 = np.log(evts_np[:, 1])
        hist = np.histogram2d(log_x, log_Q2, bins=(nx_local, nQ2_local))
        counts = hist[0].astype(float)
        x_edges = hist[1]
        Q2_edges = hist[2]
        density = np.zeros_like(counts)
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
                    density[i, j] = c / (dx * dQ2)
        # scale to simulator total_xsec if available
        try:
            scale = float(simulator.mceg.total_xsec) / np.sum(counts)
        except Exception:
            scale = 1.0
        density *= scale
        return density, x_edges, Q2_edges

    # Storage
    true_params_list = []
    predicted_params_list = []
    observed_events_list = []  # Store observed events for NPE comparison
    per_draw_scalar_errors = []
    failed = 0

    # We'll build canonical bin edges from the first successful draw so that every draw uses the same grid
    canonical_x_edges = None
    canonical_Q2_edges = None

    # iterate draws
    for draw_idx in tqdm(range(n_draws), desc="computing function errors"):
        try:
            # sample params from bounds if we have them, else random normal fallback
            if theta_bounds is not None:
                theta_raw = torch.rand(theta_bounds.shape[0], device=device)
                true_params = (
                    theta_raw * (theta_bounds[:, 1] - theta_bounds[:, 0])
                    + theta_bounds[:, 0]
                )
            else:
                # fallback: 4-dim standard uniform
                true_params = torch.rand(4, device=device)

            # simulate true events
            try:
                if problem == "gaussian":
                    evts_true = simulator.sample(true_params, nevents=n_events)
                else:
                    # many simulators expect (params, n_events) naming; be permissive
                    try:
                        evts_true = simulator.sample(true_params, n_events)
                    except TypeError:
                        evts_true = simulator.sample(true_params, n_events=n_events)
            except Exception as e:
                # attempt alternate call signatures
                try:
                    evts_true = simulator.sample(n_events, true_params)
                except Exception:
                    if verbose:
                        print(f"   ⚠️  draw {draw_idx}: simulation failed: {e}")
                    failed += 1
                    continue

            # convert to numpy for histogramming
            evts_true_np = (
                evts_true.detach().cpu().numpy()
                if torch.is_tensor(evts_true)
                else np.asarray(evts_true)
            )

            # For the first successful draw, compute canonical edges
            if canonical_x_edges is None or canonical_Q2_edges is None:
                _, canonical_x_edges, canonical_Q2_edges = build_density_from_events(
                    evts_true_np, nx_local=nx, nQ2_local=nQ2
                )

            # Build true density on canonical grid: to guarantee same edges, we feed log-values into numpy.histogram2d directly
            # Recompute histogram with canonical edges to get counts consistent with canonical grid
            log_x = np.log(evts_true_np[:, 0])
            log_Q2 = np.log(evts_true_np[:, 1])
            counts_true, _, _ = np.histogram2d(
                log_x, log_Q2, bins=[canonical_x_edges, canonical_Q2_edges]
            )
            # convert counts -> density using canonical bin areas
            true_density = np.zeros_like(counts_true)
            for i in range(canonical_x_edges.shape[0] - 1):
                for j in range(canonical_Q2_edges.shape[0] - 1):
                    c = counts_true[i, j]
                    if c > 0:
                        xmin = np.exp(canonical_x_edges[i])
                        xmax = np.exp(canonical_x_edges[i + 1])
                        Q2min = np.exp(canonical_Q2_edges[j])
                        Q2max = np.exp(canonical_Q2_edges[j + 1])
                        dx = xmax - xmin
                        dQ2 = Q2max - Q2min
                        true_density[i, j] = c / (dx * dQ2)
            # scale same as build_density_from_events would
            try:
                scale = float(simulator.mceg.total_xsec) / np.sum(counts_true)
            except Exception:
                scale = 1.0
            true_density *= scale

            # inference: feature engineering -> pointnet -> model (optionally laplace)
            try:
                feats = (
                    evts_true.float().to(device)
                    if torch.is_tensor(evts_true)
                    else torch.tensor(evts_true_np, device=device, dtype=torch.float32)
                )
                feats_for_pointnet = log_feature_engineering(feats).float().unsqueeze(0)
                lat = pointnet_model(feats_for_pointnet).detach()
                # model prediction
                if hasattr(model, "nll_mode") and model.nll_mode:
                    pred_mean, _ = model(lat)
                    inferred_theta = pred_mean.squeeze()
                else:
                    outp = model(lat)
                    try:
                        inferred_theta = outp.mean(dim=0).squeeze()
                    except Exception:
                        inferred_theta = outp.squeeze()
                # apply laplace if requested
                if laplace_model is not None:
                    try:
                        # use mean of laplace samples
                        lap_samps = laplace_model.sample(20, x=lat)
                        inferred_theta = lap_samps.mean(dim=0).squeeze()
                    except Exception:
                        pass
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  draw {draw_idx}: inference failed: {e}")
                failed += 1
                continue

            # --- New: compute function error via collaborator PDF evaluation (u - ub at Q2=10)
            # This mirrors the approach used in plot_pdf_uncertainty_mceg and is more
            # principled than histogram-based density comparisons.
            try:
                from simulator import ALPHAS, MELLIN, PDF
            except Exception:
                # If collaborator PDF not available, skip this draw
                if verbose:
                    print(
                        f"   ⚠️  draw {draw_idx}: collaborator PDF components unavailable; skipping draw"
                    )
                failed += 1
                continue

            # fixed evaluation grid and Q2
            x_grid_pdf = np.linspace(0.001, 0.99, 100)
            q2_val_pdf = 10.0

            def eval_pdf_u_minus_ub_local(theta_arr):
                pdf_temp = PDF(MELLIN(npts=8), ALPHAS())
                # prepare parameter array and insert theta values in expected slot
                cpar = pdf_temp.get_current_par_array()[::]
                arr = np.asarray(theta_arr)
                try:
                    cpar[4 : 4 + arr.shape[0]] = arr
                except Exception:
                    try:
                        cpar[4:8] = arr
                    except Exception:
                        cpar[4 : 4 + arr.shape[0]] = arr
                try:
                    pdf_temp.setup(cpar)
                except Exception:
                    try:
                        pdf_temp.setup(arr)
                    except Exception:
                        raise

                vals = []
                for x_val in x_grid_pdf:
                    try:
                        u = pdf_temp.get_xF(float(x_val), q2_val_pdf, "u", evolve=True)
                        ub = pdf_temp.get_xF(
                            float(x_val), q2_val_pdf, "ub", evolve=True
                        )
                        uval = float(u[0]) if hasattr(u, "__len__") else float(u)
                        ubval = float(ub[0]) if hasattr(ub, "__len__") else float(ub)
                        vals.append(uval - ubval)
                    except Exception:
                        vals.append(np.nan)
                return np.asarray(vals)

            # Evaluate true and predicted curves and compute mean absolute error across x
            try:
                true_theta_np = (
                    true_params.detach().cpu().numpy()
                    if torch.is_tensor(true_params)
                    else np.asarray(true_params)
                )
                true_curve = eval_pdf_u_minus_ub_local(true_theta_np)
            except Exception:
                true_curve = np.full_like(x_grid_pdf, np.nan)

            try:
                inferred_theta_cpu = (
                    inferred_theta.detach().cpu().numpy()
                    if torch.is_tensor(inferred_theta)
                    else np.asarray(inferred_theta)
                )
            except Exception:
                inferred_theta_cpu = np.asarray(inferred_theta)

            try:
                pred_curve = eval_pdf_u_minus_ub_local(inferred_theta_cpu)
            except Exception:
                pred_curve = np.full_like(x_grid_pdf, np.nan)

            diff = np.abs(true_curve - pred_curve)
            mask = np.isfinite(diff)
            draw_scalar = float(np.nanmean(diff[mask])) if np.any(mask) else np.nan

            # store
            true_params_list.append(
                true_params.detach().cpu()
                if torch.is_tensor(true_params)
                else torch.tensor(true_params)
            )
            predicted_params_list.append(torch.tensor(inferred_theta_cpu))
            observed_events_list.append(evts_true_np.copy())  # Store observed events
            per_draw_scalar_errors.append(draw_scalar)

        except Exception as e:
            if verbose:
                print(f"   ⚠️ draw {draw_idx} unexpected error: {e}")
            failed += 1
            continue

    successful = len(per_draw_scalar_errors)
    if successful == 0:
        raise RuntimeError("All draws failed; check simulator / models / settings.")

    if failed > 0 and verbose:
        print(
            f"   ⚠️ {failed}/{n_draws} draws failed; proceeding with {successful} results"
        )

    per_draw_array = np.array(per_draw_scalar_errors)
    # remove NaNs if present
    valid_mask = ~np.isnan(per_draw_array)
    per_draw_array = per_draw_array[valid_mask]
    # summary stats
    mean_error = float(np.mean(per_draw_array))
    med_error = float(np.median(per_draw_array))
    std_error = float(np.std(per_draw_array))

    if verbose:
        print(
            f"   ✅ Computed per-draw averaged function errors (n={len(per_draw_array)})."
        )
        print(
            f"      mean={mean_error:.6g}, median={med_error:.6g}, std={std_error:.6g}"
        )
    
    # --- Compute NPE errors if available ---
    npe_errors_array = np.array([])
    try:
        import pickle
        import os
        npe_path = "npe_posterior_mceg.pkl"
        if os.path.exists(npe_path):
            if verbose:
                print(f"Loading NPE posterior from {npe_path}...")
            with open(npe_path, "rb") as f:
                posterior_npe = pickle.load(f)
            
            from simulator import ALPHAS, MELLIN, PDF
            npe_errors_list = []
            
            # Use same true parameters AND observed events as our method
            for idx in range(min(len(true_params_list), len(observed_events_list), n_draws)):
                try:
                    true_params_idx = true_params_list[idx]
                    true_theta_cpu = (
                        true_params_idx.detach().cpu().numpy()
                        if torch.is_tensor(true_params_idx)
                        else np.asarray(true_params_idx)
                    )
                    
                    # Use the SAME observed events that our method used (stored earlier)
                    evts_np = observed_events_list[idx]
                    
                    # Get summary for NPE (2D histogram)
                    try:
                        from sbibm_benchmark_mceg4dis import histogram_summary
                        x_o_summary = histogram_summary(evts_np, nx=30, nQ2=20)
                    except Exception:
                        continue
                    
                    # Sample from NPE posterior
                    try:
                        npe_samples = posterior_npe.sample((1,), x=x_o_summary)
                        pred_theta_npe = npe_samples[0].cpu().numpy()
                    except Exception:
                        continue
                    
                    # Evaluate PDF curves at Q2=10 for true and NPE-predicted parameters
                    x_grid_pdf = np.linspace(0.001, 0.99, 100)
                    q2_val_pdf = 10.0
                    
                    def eval_pdf_u_minus_ub_local(theta_arr):
                        pdf_temp = PDF(MELLIN(npts=8), ALPHAS())
                        cpar = pdf_temp.get_current_par_array()[::]
                        arr = np.asarray(theta_arr)
                        try:
                            cpar[4 : 4 + arr.shape[0]] = arr
                        except Exception:
                            try:
                                cpar[4:8] = arr
                            except Exception:
                                cpar[4 : 4 + arr.shape[0]] = arr
                        try:
                            pdf_temp.setup(cpar)
                        except Exception:
                            try:
                                pdf_temp.setup(arr)
                            except Exception:
                                raise
                        vals = []
                        for x_val in x_grid_pdf:
                            try:
                                u = pdf_temp.get_xF(float(x_val), q2_val_pdf, "u", evolve=True)
                                ub = pdf_temp.get_xF(float(x_val), q2_val_pdf, "ub", evolve=True)
                                uval = float(u[0]) if hasattr(u, "__len__") else float(u)
                                ubval = float(ub[0]) if hasattr(ub, "__len__") else float(ub)
                                vals.append(uval - ubval)
                            except Exception:
                                vals.append(np.nan)
                        return np.asarray(vals)
                    
                    true_curve = eval_pdf_u_minus_ub_local(true_theta_cpu)
                    pred_curve = eval_pdf_u_minus_ub_local(pred_theta_npe)
                    
                    diff = np.abs(true_curve - pred_curve)
                    mask = np.isfinite(diff)
                    draw_scalar = float(np.nanmean(diff[mask])) if np.any(mask) else np.nan
                    
                    if not np.isnan(draw_scalar):
                        npe_errors_list.append(draw_scalar)
                except Exception as e_inner:
                    if verbose:
                        print(f"   ⚠️ NPE draw {idx} failed: {e_inner}")
                    continue
            
            npe_errors_array = np.array(npe_errors_list)
            if verbose and len(npe_errors_array) > 0:
                print(f"   ✓ Computed NPE errors: {len(npe_errors_array)} samples")
                print(f"      NPE mean={np.mean(npe_errors_array):.6g}, median={np.median(npe_errors_array):.6g}")
        else:
            if verbose:
                print(f"   NPE posterior not found at {npe_path}, skipping NPE comparison")
    except Exception as e:
        if verbose:
            print(f"   Could not compute NPE errors: {e}")

    # Plot histogram of per-draw averaged errors (with NPE overlay if available)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute log-spaced bin edges to ensure both histograms are directly comparable
    if len(npe_errors_array) > 0:
        all_errors = np.concatenate([per_draw_array, npe_errors_array])
    else:
        all_errors = per_draw_array
    
    # Ensure all values are positive for log spacing
    min_err = np.min(all_errors[all_errors > 0]) if np.any(all_errors > 0) else 1e-10
    max_err = np.max(all_errors)
    bin_edges = np.logspace(np.log10(min_err), np.log10(max_err), 51)  # 51 edges = 50 bins
    
    ax.hist(per_draw_array, bins=bin_edges, alpha=0.6, label="Ours", color=COLORBLIND_COLORS["blue"])
    
    # Add median lines for Ours
    ax.axvline(
        med_error,
        color=COLORBLIND_COLORS["dark_green"],
        linestyle=":",
        linewidth=3,
        label=f"Ours (Median) = {med_error:.4g}",
        alpha=0.9,
    )
    
    # Overlay NPE histogram if available
    if len(npe_errors_array) > 0:
        ax.hist(npe_errors_array, bins=bin_edges, alpha=0.6, label="NPE", color=COLORBLIND_COLORS["purple"])
        
        # Add statistics for NPE
        npe_mean = np.mean(npe_errors_array)
        npe_std = np.std(npe_errors_array)
        npe_median = np.median(npe_errors_array)
        
        # Add NPE median line
        ax.axvline(
            npe_median,
            color=COLORBLIND_COLORS["purple"],
            linestyle="-.",
            linewidth=3,
            label=f"NPE (Median) = {npe_median:.4g}",
            alpha=0.9,
        )
        
        stats_text = f"Ours: μ={mean_error:.4g}, σ={std_error:.4g}\nNPE: μ={npe_mean:.4g}, σ={npe_std:.4g}"
    else:
        stats_text = f"Ours: μ={mean_error:.4g}, σ={std_error:.4g}"
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    ax.text(
        0.98, 0.97,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=20,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor="gray"),
    )
    
    ax.set_xlabel("Errors", fontsize=26)
    ax.tick_params(axis='x', labelsize=24, which='both')
    ax.set_ylabel("Count", fontsize=26)
    ax.tick_params(axis='y', labelsize=24)
    ax.legend(fontsize=20, loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, which="both")
    
    # Format x-axis labels with 10^x notation
    from matplotlib.ticker import FuncFormatter
    def log_formatter(x, pos):
        if x <= 0:
            return ''
        exponent = np.log10(x)
        if abs(exponent - round(exponent)) < 0.01:  # Close to integer power
            return f'$10^{{{int(round(exponent))}}}$'
        else:
            return f'$10^{{{exponent:.1f}}}$'
    ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))
    
    # ax.set_title(f"Function error histogram (problem={problem})\nmean={mean_error:.3e}, median={med_error:.3e}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    if verbose:
        print(f"   ✅ Saved histogram to: {save_path}")

    if return_data:
        # convert param lists to tensors/arrays trimmed to valid_mask
        true_params_array = (
            torch.stack(true_params_list) if len(true_params_list) > 0 else None
        )
        pred_params_array = (
            torch.stack(predicted_params_list)
            if len(predicted_params_list) > 0
            else None
        )
        # filter per-draw arrays to valid entries (if any NaNs were removed)
        return true_params_array, pred_params_array, per_draw_array

    return None


def plot_function_posterior_from_sbi_samples(
    model,
    pointnet_model,
    sbi_samples,
    true_params,
    device,
    num_events,
    problem,
    save_path,
    rng_seed=None,
    label="SBI Posterior",
):
    """
    aggregation='median',
    n_mc=None,
    rng_seed=None
    Uncertainty is over the distribution of PDFs generated by the SBI samples.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    model.eval()

    # RNG for deterministic subsampling if requested
    rng = None
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    pointnet_model.eval()
    sbi_samples = sbi_samples.to(device)
    if problem == "simplified_dis":
        # Use percentile bands (median, 25-75 IQR, 5-95 outer) and log-scaled y-axis
        simulator = SimplifiedDIS(device=device)
        x_grid = torch.linspace(0.01, 0.99, 100).to(device)
        pdfs_up = []
        pdfs_down = []
        # collect PDFs for each theta sample
        for theta in sbi_samples:
            theta = theta.to(device)
            pdf_dict = simulator.f(x_grid, theta)
            pdfs_up.append(pdf_dict["up"].cpu().numpy())
            pdfs_down.append(pdf_dict["down"].cpu().numpy())
        pdfs_up = np.array(pdfs_up)  # [n_samples, n_x]
        pdfs_down = np.array(pdfs_down)

        # compute median and quantile bands
        median_up = np.median(pdfs_up, axis=0)
        q25_up = np.quantile(pdfs_up, 0.25, axis=0)
        q75_up = np.quantile(pdfs_up, 0.75, axis=0)
        q05_up = np.quantile(pdfs_up, 0.05, axis=0)
        q95_up = np.quantile(pdfs_up, 0.95, axis=0)

        median_down = np.median(pdfs_down, axis=0)
        q25_down = np.quantile(pdfs_down, 0.25, axis=0)
        q75_down = np.quantile(pdfs_down, 0.75, axis=0)
        q05_down = np.quantile(pdfs_down, 0.05, axis=0)
        q95_down = np.quantile(pdfs_down, 0.95, axis=0)

        x = x_grid.cpu().numpy()
        plt.figure(figsize=(12, 5))
    # Compute ground truth PDFs
    true_pdf = simulator.f(x_grid, true_params.to(device))
    true_up = true_pdf["up"].cpu().numpy()
    true_down = true_pdf["down"].cpu().numpy()
    # For plotting, avoid zeros/negatives before applying log scale by flooring
    eps = 1e-12

    plt.subplot(1, 2, 1)
    # IQR 25-75
    plt.fill_between(
        x,
        np.maximum(q25_up, eps),
        np.maximum(q75_up, eps),
        color="C0",
        alpha=0.28,
        label=f"{label} IQR",
    )
    # median
    plt.plot(
        x, np.maximum(median_up, eps), color="C0", linewidth=2, label=f"{label} median"
    )
    # ground truth
    plt.plot(
        x, np.maximum(true_up, eps), "r--", label="Ground Truth up PDF", linewidth=2
    )
    plt.title(f"Function Posterior Up ({label})")
    plt.xlabel("x")
    plt.ylabel("PDF value")
    plt.legend()
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.fill_between(
        x,
        np.maximum(q25_down, eps),
        np.maximum(q75_down, eps),
        color="C1",
        alpha=0.28,
        label=f"{label} IQR",
    )
    plt.plot(
        x,
        np.maximum(median_down, eps),
        color="C1",
        linewidth=2,
        label=f"{label} median",
    )
    plt.plot(
        x, np.maximum(true_down, eps), "r--", label="Ground Truth down PDF", linewidth=2
    )
    plt.title(f"Function Posterior Down ({label})")
    plt.xlabel("x")
    plt.ylabel("PDF value")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Saved SBI function posterior plot: {save_path}")
    # else:
    #     # For other problems, keep previous logic or adapt as needed
    #     pass


def plot_function_posterior_from_multiple_sbi_samples(
    model,
    pointnet_model,
    sbi_samples_list,
    labels,
    true_params,
    device,
    num_events,
    problem,
    save_path,
    figsize=(12, 5),
    true_color=None,
):
    """
    Plot multiple SBI-derived function posteriors on the same figure.

    Each entry in `sbi_samples_list` should be a torch.Tensor of shape [N, param_dim].
    `labels` should be a list of strings of the same length used for the legend.

    This creates a single figure with two subplots (up/down) for `simplified_dis` and
    overlays the mean ± std bands for each SBI method. The ground-truth PDF is drawn
    once as a red dashed line by default.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    model.eval()
    pointnet_model.eval()

    # Colors: reuse the palette defined higher in this module if available
    try:
        palette = [
            COLORBLIND_COLORS["blue"],
            COLORBLIND_COLORS["orange"],
            COLORBLIND_COLORS["green"],
            COLORBLIND_COLORS["purple"],
        ]
    except Exception:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    if true_color is None:
        try:
            true_color = COLORBLIND_COLORS["red"]
        except Exception:
            true_color = "red"

    # Ensure inputs are lists
    if not isinstance(sbi_samples_list, (list, tuple)):
        sbi_samples_list = [sbi_samples_list]
    if labels is None:
        labels = [f"SBI {i+1}" for i in range(len(sbi_samples_list))]

    if problem == "simplified_dis":
        simulator = SimplifiedDIS(device=device)
        x_grid = torch.linspace(0.01, 0.99, 100).to(device)

        # compute median and quantile bands for each method
        medians_up = []
        q25_up = []
        q75_up = []
        q05_up = []
        q95_up = []
        medians_down = []
        q25_down = []
        q75_down = []
        q05_down = []
        q95_down = []
        x = x_grid.cpu().numpy()

        for sbi_samples in sbi_samples_list:
            sbi_samples = sbi_samples.to(device)
            pdfs_up = []
            pdfs_down = []
            for theta in sbi_samples:
                theta = theta.to(device)
                pdf_dict = simulator.f(x_grid, theta)
                pdfs_up.append(pdf_dict["up"].cpu().numpy())
                pdfs_down.append(pdf_dict["down"].cpu().numpy())
            pdfs_up = np.array(pdfs_up)
            pdfs_down = np.array(pdfs_down)
            medians_up.append(np.median(pdfs_up, axis=0))
            q25_up.append(np.quantile(pdfs_up, 0.25, axis=0))
            q75_up.append(np.quantile(pdfs_up, 0.75, axis=0))
            q05_up.append(np.quantile(pdfs_up, 0.05, axis=0))
            q95_up.append(np.quantile(pdfs_up, 0.95, axis=0))

            medians_down.append(np.median(pdfs_down, axis=0))
            q25_down.append(np.quantile(pdfs_down, 0.25, axis=0))
            q75_down.append(np.quantile(pdfs_down, 0.75, axis=0))
            q05_down.append(np.quantile(pdfs_down, 0.05, axis=0))
            q95_down.append(np.quantile(pdfs_down, 0.95, axis=0))

        # compute ground truth once
        true_pdf = simulator.f(x_grid, true_params.to(device))
        true_up = true_pdf["up"].cpu().numpy()
        true_down = true_pdf["down"].cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        # Plot each SBI method using quantile-based bands (median, IQR, and 5-95%)
        for i, (mu, q25, q75, q05, q95) in enumerate(
            zip(medians_up, q25_up, q75_up, q05_up, q95_up)
        ):
            color = palette[i % len(palette)]
            label = labels[i] if i < len(labels) else f"SBI {i+1}"
            ax.plot(x, mu, color=color, label=label + " (median)", linewidth=2)
            # outer 90% band (5-95)
            ax.fill_between(x, q05, q95, color=color, alpha=0.12)
            # inner IQR band (25-75)
            ax.fill_between(x, q25, q75, color=color, alpha=0.28)

        # true line (single)
        ax.plot(
            x, true_up, linestyle="--", color=true_color, linewidth=3, label="True PDF"
        )
        # ax.set_title('Function Posterior Up (combined SBI)')
        ax.set_xlabel("x")
        ax.set_ylabel(r"$u(x)$")
        ax.legend()

        ax = axes[1]
        for i, (mu, q25, q75, q05, q95) in enumerate(
            zip(medians_down, q25_down, q75_down, q05_down, q95_down)
        ):
            color = palette[i % len(palette)]
            label = labels[i] if i < len(labels) else f"SBI {i+1}"
            ax.plot(x, mu, color=color, label=label + " (median)", linewidth=2)
            ax.fill_between(x, q05, q95, color=color, alpha=0.12)
            ax.fill_between(x, q25, q75, color=color, alpha=0.28)
        ax.plot(
            x,
            true_down,
            linestyle="--",
            color=true_color,
            linewidth=3,
            label="True PDF",
        )
        # ax.set_title('Function Posterior Down (combined SBI)')
        ax.set_xlabel("x")
        ax.set_ylabel(r"$d(x)$")
        # ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved combined SBI function posterior plot: {save_path}")
    else:
        # Support mceg / mceg4dis by plotting each SBI method separately (one file per method)
        if problem in ["mceg", "mceg4dis"]:
            print(
                "ℹ️ plot_function_posterior_from_multiple_sbi_samples: creating separate mceg4dis plots for each SBI method"
            )
            # Q2 slices default (kept consistent with other mceg plotting functions)
            Q2_slices = [0.5, 1.0, 1.5, 2.0, 10.0, 50.0, 200.0]
            # x range for slice evaluation (logspace)
            x_min, x_max = 1e-4, 1e-1
            x_slice_vals = np.logspace(np.log10(x_min), np.log10(x_max), 200)

            # set up theory helpers once
            try:
                from simulator import MCEGSimulator
            except Exception:
                MCEGSimulator = None

            # Build true-theory idis object once
            try:
                mellin = MELLIN(npts=8)
                alphaS = ALPHAS()
                eweak = EWEAK()
                pdf_base = PDF(mellin, alphaS)
                # configure PDF with current defaults
                new_cpar_true = pdf_base.get_current_par_array()[::]
                # insert true params into expected slots (match earlier mapping)
                try:
                    new_cpar_true[4:8] = (
                        true_params.cpu().numpy()
                        if isinstance(true_params, torch.Tensor)
                        else np.asarray(true_params)
                    )
                except Exception:
                    # fallback: if sizes mismatch, try to broadcast
                    arr = np.asarray(true_params)
                    new_cpar_true[4 : 4 + len(arr)] = arr
                pdf_base.setup(new_cpar_true)
                idis_true = THEORY(mellin, pdf_base, alphaS, eweak)
            except Exception as e:
                print(f"⚠️ Could not construct theory objects for mceg plotting: {e}")
                idis_true = None

            for sbi_samples, label in zip(sbi_samples_list, labels):
                label_safe = label.replace(" ", "_")
                out_path = save_path.replace(".png", f"_{label_safe}.png")
                print(f"ℹ️ Generating mceg4dis slices for {label} -> {out_path}")

                # limit number of SBI samples to keep runtime reasonable
                sbi_samples = sbi_samples.to("cpu")
                n_samples = (
                    min(200, int(sbi_samples.shape[0]))
                    if sbi_samples.numel() > 0
                    else 0
                )
                if n_samples == 0:
                    print(f"⚠️ No samples provided for {label}; skipping")
                    continue
                samp = sbi_samples[:n_samples]

                # For each Q2 slice, evaluate idis.get_diff_xsec for every theta sample and build quantiles
                for Q2_fixed in Q2_slices:
                    vals = np.zeros((n_samples, x_slice_vals.shape[0]), dtype=float)
                    for i in range(n_samples):
                        theta = samp[i].numpy()
                        try:
                            pdf_temp = PDF(mellin, alphaS)
                            new_cpar = pdf_temp.get_current_par_array()[::]
                            # same mapping as earlier: place theta into 4:8 (adjust if dims differ)
                            try:
                                new_cpar[4:8] = theta
                            except Exception:
                                new_cpar[4 : 4 + len(theta)] = theta
                            pdf_temp.setup(new_cpar)
                            idis_temp = THEORY(mellin, pdf_temp, alphaS, eweak)
                        except Exception as e:
                            print(f"⚠️ Could not build idis for sample {i}: {e}")
                            # fill row with NaNs
                            vals[i, :] = np.nan
                            continue

                        row = np.zeros_like(x_slice_vals)
                        for j, xval in enumerate(x_slice_vals):
                            try:
                                t = idis_temp.get_diff_xsec(
                                    float(xval), float(Q2_fixed), None, None, "xQ2"
                                )
                                if isinstance(t, (tuple, list)):
                                    row[j] = float(t[0])
                                else:
                                    row[j] = float(t)
                            except Exception:
                                row[j] = np.nan
                        vals[i, :] = row

                    # compute quantiles across samples (ignore NaNs)
                    with np.errstate(invalid="ignore"):
                        q05 = np.nanquantile(vals, 0.05, axis=0)
                        q25 = np.nanquantile(vals, 0.25, axis=0)
                        q50 = np.nanquantile(vals, 0.50, axis=0)
                        q75 = np.nanquantile(vals, 0.75, axis=0)
                        q95 = np.nanquantile(vals, 0.95, axis=0)

                    # True curve for this Q2
                    if idis_true is not None:
                        true_curve = np.zeros_like(x_slice_vals)
                        for j, xval in enumerate(x_slice_vals):
                            try:
                                t = idis_true.get_diff_xsec(
                                    float(xval), float(Q2_fixed), None, None, "xQ2"
                                )
                                true_curve[j] = (
                                    float(t[0])
                                    if isinstance(t, (tuple, list))
                                    else float(t)
                                )
                            except Exception:
                                true_curve[j] = np.nan
                    else:
                        true_curve = None

                    # Plot one figure per method per Q2 slice (stack Q2 slices into subplots)
                    # We'll create a multi-panel figure with all Q2_slices plotted (one line per Q2)
                    # but since user asked for separate per-method plots, we save one file per method containing all Q2 slices.
                    fig, ax = plt.subplots(figsize=(10, 7))
                    # Plot median and bands for the method at this Q2
                    ax.plot(
                        x_slice_vals,
                        q50,
                        color=COLORBLIND_COLORS.get("blue", "#1f77b4"),
                        linewidth=2,
                        label=f"{label} median",
                    )
                    ax.fill_between(
                        x_slice_vals,
                        q25,
                        q75,
                        color=COLORBLIND_COLORS.get("blue", "#1f77b4"),
                        alpha=0.25,
                        label="IQR",
                    )
                    ax.fill_between(
                        x_slice_vals,
                        q05,
                        q95,
                        color=COLORBLIND_COLORS.get("blue", "#1f77b4"),
                        alpha=0.12,
                        label="90%",
                    )
                    if true_curve is not None:
                        ax.plot(
                            x_slice_vals,
                            true_curve,
                            linestyle="--",
                            color=COLORBLIND_COLORS.get("red", "red"),
                            linewidth=3,
                            label="True",
                        )
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlabel("x")
                    ax.set_ylabel("Differential Cross Section")
                    ax.set_title(f"{label} - Q²={Q2_fixed}")
                    ax.grid(True, which="both", linestyle=":", alpha=0.3)
                    ax.legend()
                    plt.tight_layout()
                    # Save file: append Q2 value to filename
                    q2_safe = str(Q2_fixed).replace(".", "p")
                    out_file = out_path.replace(".png", f"_Q2_{q2_safe}.png")
                    plt.savefig(out_file, dpi=300, bbox_inches="tight")
                    plt.close(fig)

                print(
                    f"✓ Saved separate mceg4dis plots for {label} (one file per Q2 slice) -> base: {out_path}"
                )
            return
        else:
            # Fallback: call single-sample plotting repeatedly if not simplified_dis
            print(
                "⚠️ plot_function_posterior_from_multiple_sbi_samples: combined plotting only implemented for 'simplified_dis' and 'mceg4dis'. Falling back to individual plots."
            )
            for sbi_samples, label in zip(sbi_samples_list, labels):
                out_path = save_path.replace(".png", f"_{label.replace(' ', '_')}.png")
                plot_function_posterior_from_sbi_samples(
                    model,
                    pointnet_model,
                    sbi_samples,
                    true_params,
                    device,
                    num_events,
                    problem,
                    out_path,
                    label=label,
                )


def plot_function_error_summary_from_sbi_samples(
    sbi_samples_list,
    labels,
    true_params,
    device,
    problem="simplified_dis",
    save_path="sbi_function_error_summary.png",
    x_grid=None,
    our_results_dict=None,
    relative=True,
    aggregation="median",
    n_mc=None,
    rng_seed=None,
):
    """
    Compute and plot function-space average errors from SBI posterior samples for simplified_dis.

    Plots a two-row figure:
      - Top: grouped bar chart of mean absolute error (averaged over x) for each method and for up/down.
      - Bottom: per-x absolute error curves for each method (up and down in separate subplots).

    sbi_samples_list: list of torch.Tensor [N, D]
    labels: list of strings
    true_params: torch.Tensor or array
    device: torch.device or string
    aggregation: 'median' or 'mean' -- how to aggregate per-x errors into a scalar for the top bar chart
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    try:
        from plotting_UQ_helpers import collect_predicted_pdfs_simplified_dis
    except Exception:
        # fallback: use local collector defined near end of file
        def collect_predicted_pdfs_simplified_dis(param_samples, device):
            from simulator import SimplifiedDIS

            simulator = SimplifiedDIS(device=device)
            x_grid_t = torch.linspace(0.01, 0.99, 100).to(device)
            pdfs_up, pdfs_down = [], []
            for theta in param_samples:
                theta = theta.to(device)
                pdfd = simulator.f(x_grid_t, theta)
                pdfs_up.append(pdfd["up"].cpu().numpy())
                pdfs_down.append(pdfd["down"].cpu().numpy())
            return np.array(pdfs_up), np.array(pdfs_down)

    # Immediate debug banner so callers/CI can't miss that we've entered this function
    try:
        import sys

        print("[debug-entry] plot_function_error_summary_from_sbi_samples invoked")
        print(
            f"  - incoming sbi_samples_list type: {type(sbi_samples_list)}, length: {len(sbi_samples_list) if hasattr(sbi_samples_list, '__len__') else 'unknown'}"
        )
        print(f"  - incoming labels: {labels}")
        sys.stdout.flush()
    except Exception:
        # Best-effort: don't fail if printing diagnostics fails
        pass

    # Defaults
    if x_grid is None:
        x_grid = np.linspace(0.01, 0.99, 100)

    # RNG for optional subsampling of ensembles
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = None

    # Ensure lists
    if not isinstance(sbi_samples_list, (list, tuple)):
        sbi_samples_list = [sbi_samples_list]
    if labels is None:
        labels = [f"SBI {i+1}" for i in range(len(sbi_samples_list))]

    # Compute true PDFs on grid
    if problem != "simplified_dis":
        raise ValueError(
            "plot_function_error_summary_from_sbi_samples currently supports simplified_dis only"
        )
    from simulator import SimplifiedDIS

    simulator = SimplifiedDIS(device=device)
    x_t = torch.tensor(x_grid, dtype=torch.float32).to(device)
    true_pdf = simulator.f(x_t, true_params.to(device))
    true_up = true_pdf["up"].cpu().numpy()
    true_down = true_pdf["down"].cpu().numpy()

    # Diagnostic: print what our_results_dict contains (types and shapes)
    try:
        if our_results_dict is None:
            print("[debug] our_results_dict is None")
        else:
            print(f"[debug] our_results_dict keys: {list(our_results_dict.keys())}")
            for k, v in our_results_dict.items():
                print(f"[debug] our_results_dict[{k}] type={type(v)}")
                if isinstance(v, dict):
                    for sub in [
                        "pdfs_up",
                        "pdfs_down",
                        "mean_up",
                        "mean_down",
                        "unc_up",
                        "unc_down",
                    ]:
                        val = v.get(sub, None)
                        if val is None:
                            print(f"    - {sub}: None")
                            continue
                        # try to get length/shape safely
                        try:
                            if hasattr(val, "shape"):
                                print(
                                    f"    - {sub}: type={type(val)}, shape={getattr(val,'shape', None)}"
                                )
                            else:
                                arr = np.asarray(val)
                                print(
                                    f"    - {sub}: converted ndarray shape={arr.shape}, dtype={arr.dtype}"
                                )
                        except Exception as e:
                            print(
                                f"    - {sub}: (could not inspect) {type(val)} -> {e}"
                            )
                else:
                    print(f"    - value: {v}")
    except Exception as e:
        print(f"[debug] Could not inspect our_results_dict: {e}")

    # We'll collect results keyed by method label (with aggregation suffix) so we can order them consistently
    method_map_avg_up = {}
    method_map_avg_down = {}
    method_map_per_x_up = {}
    method_map_per_x_down = {}

    # If user provided 'our' approaches, combine Laplace and Bootstrap into a single ensemble
    extra_labels = []
    if our_results_dict is not None and isinstance(our_results_dict, dict):
        # Collect ensemble PDFs from Laplace and Bootstrap, concatenate them
        combined_pdfs_up = []
        combined_pdfs_down = []
        
        for k in ["Laplace", "Bootstrap"]:
            vals = our_results_dict.get(k, {}) or {}
            pup = (
                np.array(vals.get("pdfs_up", []))
                if isinstance(vals, dict)
                else np.array([])
            )
            pdn = (
                np.array(vals.get("pdfs_down", []))
                if isinstance(vals, dict)
                else np.array([])
            )
            
            # Only add if we have valid ensemble data
            if pup.size > 0 and pdn.size > 0 and pup.ndim == 2 and pdn.ndim == 2:
                combined_pdfs_up.append(pup)
                combined_pdfs_down.append(pdn)
        
        # If we collected any ensembles, concatenate and generate Median-only rows (no mean)
        if combined_pdfs_up:
            combined_pup = np.concatenate(combined_pdfs_up, axis=0)  # [N_total, n_x]
            combined_pdn = np.concatenate(combined_pdfs_down, axis=0)
            
            # Generate only Median aggregation for the combined ensemble (removed Mean)
            label_with_agg = "Combined (Median)"
            extra_labels.append(label_with_agg)
            
            # Compute per-x median
            central_up = np.median(combined_pup, axis=0)
            central_down = np.median(combined_pdn, axis=0)

            # Compute per-x squared errors (MSE per-x)
            err_up = (central_up - true_up) ** 2
            err_dn = (central_down - true_down) ** 2

            method_map_per_x_up[label_with_agg] = err_up
            method_map_per_x_down[label_with_agg] = err_dn
            method_map_avg_up[label_with_agg] = float(np.nanmean(err_up))
            method_map_avg_down[label_with_agg] = float(np.nanmean(err_dn))

    # Now process SBI sample methods with both mean and median aggregations
    sbi_labels = []
    for sbi_samples, label in zip(sbi_samples_list, labels):
        # Normalize samples to a CPU tensor and optionally subsample to n_mc
        if isinstance(sbi_samples, torch.Tensor):
            sbi_cpu = sbi_samples.cpu()
        else:
            sbi_cpu = torch.tensor(sbi_samples, dtype=torch.float32)

        if n_mc is not None and sbi_cpu.shape[0] > n_mc:
            if rng is not None:
                idxs = rng.choice(sbi_cpu.shape[0], size=n_mc, replace=False)
            else:
                idxs = np.arange(n_mc)
            try:
                sbi_cpu = sbi_cpu[idxs]
            except Exception:
                sbi_cpu = sbi_cpu[torch.from_numpy(np.array(idxs, dtype=np.int64))]

        pdfs_up, pdfs_down = collect_predicted_pdfs_simplified_dis(
            sbi_cpu.to(device), device
        )
        if pdfs_up.size == 0 or pdfs_down.size == 0:
            # Only Median aggregation (removed Mean)
            label_with_agg = f"{label} (Median)"
            sbi_labels.append(label_with_agg)
            method_map_avg_up[label_with_agg] = np.nan
            method_map_avg_down[label_with_agg] = np.nan
            method_map_per_x_up[label_with_agg] = np.full_like(x_grid, np.nan)
            method_map_per_x_down[label_with_agg] = np.full_like(x_grid, np.nan)
            continue

        # Generate only Median rows for each SBI method (removed Mean)
        label_with_agg = f"{label} (Median)"
        sbi_labels.append(label_with_agg)
        
        # Compute per-x median
        central_up = np.median(pdfs_up, axis=0)
        central_down = np.median(pdfs_down, axis=0)

        # Use per-x squared error (MSE per-x)
        abs_err_up = (central_up - true_up) ** 2
        abs_err_down = (central_down - true_down) ** 2

        method_map_per_x_up[label_with_agg] = abs_err_up
        method_map_per_x_down[label_with_agg] = abs_err_down
        method_map_avg_up[label_with_agg] = float(np.nanmean(abs_err_up))
        method_map_avg_down[label_with_agg] = float(np.nanmean(abs_err_down))

    # Build combined labels (SBI first, then our approaches if present)
    final_labels = sbi_labels
    if extra_labels:
        final_labels += extra_labels
    # Debug: what final labels will be plotted and current method map keys
    try:
        print(f"[debug] final_labels: {final_labels}")
        for lbl in final_labels:
            mu_up = method_map_avg_up.get(lbl, np.nan)
            mu_dn = method_map_avg_down.get(lbl, np.nan)
            perx_up = np.asarray(
                method_map_per_x_up.get(lbl, np.full_like(x_grid, np.nan))
            )
            perx_dn = np.asarray(
                method_map_per_x_down.get(lbl, np.full_like(x_grid, np.nan))
            )
            print(
                f"[debug] label={lbl}: avg_up={mu_up}, avg_down={mu_dn}, perx_up_finite={np.sum(np.isfinite(perx_up))}/{perx_up.size}, perx_dn_finite={np.sum(np.isfinite(perx_dn))}/{perx_dn.size}"
            )
    except Exception:
        pass

    # Assemble ordered arrays matching final_labels
    method_avg_up = [method_map_avg_up.get(lbl, np.nan) for lbl in final_labels]
    method_avg_down = [method_map_avg_down.get(lbl, np.nan) for lbl in final_labels]
    per_x_errors_up = [
        method_map_per_x_up.get(lbl, np.full_like(x_grid, np.nan))
        for lbl in final_labels
    ]
    per_x_errors_down = [
        method_map_per_x_down.get(lbl, np.full_like(x_grid, np.nan))
        for lbl in final_labels
    ]

    n_methods = len(final_labels)
    ind = np.arange(n_methods)
    width = 0.35

    fig = plt.figure(figsize=(10, 8))
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=1)
    ax1 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)

    # Top: grouped bar chart (clearer aesthetics)
    # Use hatch and edgecolor for clarity and print relative (%) if requested
    def fmt_bar_values(vals):
        # For MSE plotting we display raw numeric values (no percent conversion)
        return [v if (v is not None and not np.isnan(v)) else np.nan for v in vals]

    up_vals = fmt_bar_values(method_avg_up + ([] if not extra_labels else []))
    down_vals = fmt_bar_values(method_avg_down + ([] if not extra_labels else []))

    # Bars with edge and hatching to be clear in grayscale/print
    b1 = ax0.bar(
        ind - width / 2,
        method_avg_up[:n_methods],
        width,
        label="Up",
        color=COLORBLIND_COLORS.get("blue", "#1f77b4"),
        edgecolor="k",
        hatch="",
    )
    b2 = ax0.bar(
        ind + width / 2,
        method_avg_down[:n_methods],
        width,
        label="Down",
        color=COLORBLIND_COLORS.get("orange", "#ff7f0e"),
        edgecolor="k",
        hatch="",
    )
    ax0.set_xticks(ind)
    for i in range(len(final_labels)):
        if "Combined_LOTV" in final_labels[i]:
            final_labels[i] = final_labels[i].replace("Combined_LOTV", "Combined")
        elif final_labels[i] == "Wasserstein MCABC (Mean)":
            final_labels[i] = "MCABC-W (Mean)"
        elif final_labels[i] == "Wasserstein MCABC (Median)":
            final_labels[i] = "MCABC-W (Median)"
    ax0.set_xticklabels(final_labels, rotation=45, ha="right", fontsize=11)
    # label reflects aggregation choice
    # Top bar label: Mean MSE (no longer using aggregation or relative flags)
    ax0.set_ylabel("Mean MSE")
    ax0.set_title("Average Function-space MSE (Mean over x)", fontsize=16)
    ax0.legend()
    ax0.grid(True, alpha=0.25)
    # Annotate bar values on top for clarity (use up/down pairs)
    # Prepare display values (convert to percent if relative)
    disp_up = method_avg_up[:n_methods]
    disp_down = method_avg_down[:n_methods]
    max_val = (
        np.nanmax(np.concatenate([np.array(method_avg_up), np.array(method_avg_down)]))
        if (method_avg_up or method_avg_down)
        else 1.0
    )
    y_offset = 0.01 * max(1.0, max_val)

    for i in range(n_methods):
        try:
            rect_up = b1[i]
            rect_dn = b2[i]
        except Exception:
            continue
        h_up = disp_up[i] if i < len(disp_up) else np.nan
        h_dn = disp_down[i] if i < len(disp_down) else np.nan
        # Display numeric aggregate (MSE) values on bars
        # txt_up = f"{h_up:.3e}" if not np.isnan(h_up) else ""
        # txt_dn = f"{h_dn:.3e}" if not np.isnan(h_dn) else ""
        # if txt_up:
        #     ax0.text(rect_up.get_x() + rect_up.get_width()/2., rect_up.get_height() + y_offset, txt_up, ha='center', va='bottom', fontsize=15)
        # if txt_dn:
        #     ax0.text(rect_dn.get_x() + rect_dn.get_width()/2., rect_dn.get_height() + y_offset, txt_dn, ha='center', va='bottom', fontsize=15)

    # Bottom: per-x curves (plot Up and Down separately)
    import matplotlib

    # Create exactly n_methods distinct colors using expanded colorblind-friendly palette
    cb_palette = [
        COLORBLIND_COLORS["blue"],
        COLORBLIND_COLORS["orange"],
        COLORBLIND_COLORS["green"],
        COLORBLIND_COLORS["red"],
        COLORBLIND_COLORS["yellow"],
        COLORBLIND_COLORS["gray"],
        COLORBLIND_COLORS["black"],
        COLORBLIND_COLORS["sky_blue"],
        COLORBLIND_COLORS["blueish_green"],
        COLORBLIND_COLORS["light_blue"],
        COLORBLIND_COLORS["light_orange"],
        COLORBLIND_COLORS["light_gray"],
    ]
    # Extend palette by cycling if needed
    colors = [cb_palette[i % len(cb_palette)] for i in range(n_methods)]

    handles = []
    labels_legend = []

    # Diagnostic: report per-label central source (ensemble/lotv/none)
    per_label_source = {}
    for lbl in final_labels:
        src = "none"
        if isinstance(our_results_dict, dict) and lbl in our_results_dict:
            vals = our_results_dict.get(lbl) or {}
            pup = (
                np.array(vals.get("pdfs_up", []))
                if isinstance(vals, dict)
                else np.array([])
            )
            mu_u = (
                np.array(vals.get("mean_up", []))
                if isinstance(vals, dict)
                else np.array([])
            )
            if pup.size > 0:
                src = "ensemble"
            elif mu_u.size > 0:
                src = "lotv_mean"
        per_label_source[lbl] = src
    print(f"[debug] per_label_source: {per_label_source}")

    for i, label in enumerate(final_labels):
        try:
            e_up = np.asarray(per_x_errors_up[i], dtype=float)
            e_dn = np.asarray(per_x_errors_down[i], dtype=float)
        except Exception:
            continue
        # Skip plotting if both arrays are entirely non-finite
        if not (np.any(np.isfinite(e_up)) or np.any(np.isfinite(e_dn))):
            continue
        color = colors[i % len(colors)]
        # If relative, show percent on the per-x plot for readability
        # Show per-x MSE (squared error) curves
        plot_e_up = np.where(np.isfinite(e_up), e_up, np.nan)
        plot_e_dn = np.where(np.isfinite(e_dn), e_dn, np.nan)
        ylabel = "MSE"

        # Plot only finite segments to avoid matplotlib quietly skipping or making empty legend entries
        # Up (dashed)
        if np.any(np.isfinite(plot_e_up)):
            ax1.plot(
                x_grid,
                plot_e_up,
                color=color,
                linestyle="--",
                linewidth=2.0,
                alpha=0.95,
            )
        # Down (solid)
        if np.any(np.isfinite(plot_e_dn)):
            ax1.plot(
                x_grid,
                plot_e_dn,
                color=color,
                linestyle="-",
                linewidth=2.0,
                alpha=0.9,
            )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("x")
    ax1.set_ylabel("MSE (per x)")
    ax1.set_title(
        "Per-x MSE Curves", fontsize=16
    )
    ax1.grid(True, which="both", linestyle=":", alpha=0.3)
    
    # Create efficient legend: approach colors + line style meanings
    # Approach entries (colored lines)
    approach_handles = []
    approach_labels = []
    for i, label in enumerate(final_labels):
        color = colors[i % len(colors)]
        approach_handles.append(plt.Line2D([0], [0], color=color, linewidth=2.0))
        approach_labels.append(label)
    
    # Line style meaning entries (black lines with different styles)
    linestyle_handles = [
        plt.Line2D([0], [0], color='black', linewidth=2.0, linestyle='--', label='Up quark (u)'),
        plt.Line2D([0], [0], color='black', linewidth=2.0, linestyle='-', label='Down quark (d)')
    ]
    
    # Combine legends: both in bottom left, side-by-side with proper spacing
    ax1_legend1 = ax1.legend(approach_handles, approach_labels, loc="lower left", fontsize=14, title="Approach", frameon=True, bbox_to_anchor=(0.0, 0.0))
    ax1_legend2 = ax1.legend(linestyle_handles, ['Up quark (u)', 'Down quark (d)'], loc="lower left", fontsize=14, title="Quark type", frameon=True, bbox_to_anchor=(0.35, 0.0))
    ax1.add_artist(ax1_legend1)  # Add first legend back so both are displayed

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved SBI function error summary: {save_path}")

def save_function_UQ_metrics_table_simplified_dis(
    save_path, true_params, device, results_dict, aggregation="mean"
):
    """
    Save a LaTeX-compatible table of function error and uncertainty metrics for each UQ/SBI approach.
    Each row: [Approach]
    Columns: [Up MSE, Up Bias, Up Unc., Down MSE, Down Bias, Down Unc.]
    results_dict: {
        "Laplace": {"pdfs_up": [...], "pdfs_down": [...]},
        "Bootstrap": {"pdfs_up": [...], "pdfs_down": [...]},
        "Combined": {"pdfs_up": [...], "pdfs_down": [...]},
        "SNPE": {"pdfs_up": [...], "pdfs_down": [...]},
        "MCABC": {"pdfs_up": [...], "pdfs_down": [...]},
        "Wasserstein MCABC": {"pdfs_up": [...], "pdfs_down": [...]},
    }
    """
    from simulator import SimplifiedDIS

    simulator = SimplifiedDIS(device=device)
    x_grid = torch.linspace(0.01, 0.99, 100).to(device)
    true_pdf = simulator.f(x_grid, true_params.to(device))
    true_up = true_pdf["up"].cpu().numpy()
    true_down = true_pdf["down"].cpu().numpy()

    rows = []
    for approach, vals in results_dict.items():
        pdfs_up = np.array(vals.get("pdfs_up", []))  # shape [N, 100]
        pdfs_down = np.array(vals.get("pdfs_down", []))  # shape [N, 100]
        # If the entry provides full ensembles (pdfs_up/pdf_down), compute metrics from ensembles
        if pdfs_up.size > 0 and pdfs_down.size > 0:
            # central function: follow aggregation choice to be consistent with plotting
            if aggregation == "median":
                central_up = np.median(pdfs_up, axis=0)
                central_down = np.median(pdfs_down, axis=0)
            else:
                central_up = np.mean(pdfs_up, axis=0)
                central_down = np.mean(pdfs_down, axis=0)

            # Up metrics computed from central function and ensemble spread
            up_mse = np.mean((central_up - true_up) ** 2)
            up_bias = np.mean(central_up - true_up)
            up_unc = np.mean(np.std(pdfs_up, axis=0))
            # Down metrics
            down_mse = np.mean((central_down - true_down) ** 2)
            down_bias = np.mean(central_down - true_down)
            down_unc = np.mean(np.std(pdfs_down, axis=0))
            rows.append(
                [
                    approach,
                    f"{up_mse:.4g}",
                    f"{up_bias:.4g}",
                    f"{up_unc:.4g}",
                    f"{down_mse:.4g}",
                    f"{down_bias:.4g}",
                    f"{down_unc:.4g}",
                ]
            )
            continue

    header = [
        "Approach",
        "Up MSE",
        "Up Bias",
        "Up Unc.",
        "Down MSE",
        "Down Bias",
        "Down Unc.",
    ]
    latex_lines = []
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")
    for row in rows:
        latex_lines.append(" & ".join(row) + " \\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")

    with open(save_path, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"Saved function UQ metrics table to {save_path}")