"""
Enhanced Plotting Utilities for Uncertainty Quantification in PDF Parameter Inference

This module provides publication-ready plotting functions for visualizing uncertainty
in PDF parameter inference, supporting multiple problem types including simplified_dis,
realistic_dis, and mceg4dis. All plots are designed to be beautiful, clear, and suitable 
for publication.

Problem Type Support:
====================

- simplified_dis: 1D PDF inputs (x only) with up/down quark distributions
- realistic_dis: 2D PDF inputs (x, Q2) with realistic DIS structure functions  
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
   density visualization, mean/median indicators, and comprehensive statistics.

3. generate_parameter_error_histogram:
   **NEW STANDALONE UTILITY** - Automated parameter error benchmarking function
   that handles the complete workflow from parameter sampling to histogram plotting:
   - Automatically retrieves parameter bounds for different problem types
   - Uniformly samples parameter sets from bounds
   - Generates events using appropriate simulators
   - Runs complete inference pipeline (featurize → PointNet → predict)
   - Creates publication-ready error histograms with comprehensive statistics
   - Supports optional Laplace uncertainty quantification
   - Minimal user input required - just provide models and settings

Enhanced Functions:
==================

1. plot_event_histogram_simplified_DIS:
   - Now provides both scatter plots and true 2D histograms
   - Log-scale colorbars with proper normalization
   - Clear axis labels and mathematical notation
   - Multiple visualization modes: 'scatter', 'histogram', or 'both'

2. plot_params_distribution_single:
   - Publication-ready styling with colorblind-friendly colors
   - Confidence interval visualization (±1σ, ±2σ)
   - Professional typography and mathematical notation
   - Statistical text boxes with mean/std information
   - Adaptive subplot layout for any number of parameters

3. plot_PDF_distribution_single:
   - Enhanced uncertainty visualization with multiple confidence levels
   - Beautiful color schemes for different PDF functions
   - Error statistics and comparison with true functions
   - Professional mathematical notation and legends

Aesthetic Features:
==================

All plotting functions now include:
- Colorblind-friendly color palettes (verified with colorbrewer)
- Professional mathematical notation with LaTeX formatting
- Publication-ready typography and layout
- Proper gridlines, tick marks, and spacing
- Comprehensive legends and statistical annotations
- High-DPI output (300 DPI) suitable for publication
- Consistent styling across all plot types

Color Schemes:
=============

COLORBLIND_COLORS: Main palette safe for colorblind users
UNCERTAINTY_COLORS: Specific colors for uncertainty visualization  
PDF_FUNCTION_COLORS: Colors for different PDF functions (u, d, q)

Usage Examples:
==============

# Parameter error analysis
plot_parameter_error_histogram(
    true_params_list=[true_params1, true_params2, ...],
    predicted_params_list=[pred_params1, pred_params2, ...],
    save_path="param_errors.png",
    problem='simplified_dis'
)

# Function error analysis  
plot_function_error_histogram(
    true_function_values_list=[true_vals1, true_vals2, ...],
    predicted_function_values_list=[pred_vals1, pred_vals2, ...],
    function_names=['u(x)', 'd(x)'],
    save_path="function_errors.png"
)

# Enhanced event visualization
plot_event_histogram_simplified_DIS(
    model, pointnet_model, true_params, device,
    plot_type='both',  # Show both scatter and histogram
    save_path="events_both.png"
)

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
    n_draws=100, n_events=10000,
    problem='simplified_dis',
    save_path="automated_param_errors.png"
)

# NEW: With custom settings and data return
true_params, pred_params = generate_parameter_error_histogram(
    model, pointnet_model, device,
    n_draws=200, n_events=50000,
    problem='realistic_dis',
    laplace_model=laplace_model,
    param_names=['logA0', 'delta', 'a', 'b', 'c', 'd'],
    return_data=True
)

Dependencies:
============

Required:
- matplotlib
- numpy
- torch

Optional (with fallbacks):
- seaborn (for enhanced color palettes)
- scipy (for advanced statistics)

Note: All functions include fallback implementations when optional dependencies
are not available, ensuring robust operation in any environment.

Version: Enhanced for publication-ready output
Author: Enhanced plotting utilities for PDFParameterInference
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import os
import pylab as py
from tqdm import tqdm
from typing import Optional, Callable

# Try to import datasets, but don't fail if not available
try:
    from datasets import *
except ImportError:
    pass  # Will handle missing dataset classes in get_datasets() function

# Set publication-ready plotting style
plt.style.use('default')  # Start with clean default style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.figsize': [10, 8],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
})

# Define colorblind-friendly color palette
COLORBLIND_COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e', 
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf',
    'dark_blue': '#0c2c84',
    'dark_orange': '#cc5500',
    'dark_green': '#006400'
}

# Enhanced color schemes for specific plot types
UNCERTAINTY_COLORS = {
    'model': COLORBLIND_COLORS['blue'],
    'data': COLORBLIND_COLORS['orange'], 
    'combined': COLORBLIND_COLORS['purple'],
    'true': COLORBLIND_COLORS['dark_green'],
    'predicted': COLORBLIND_COLORS['red']
}

PDF_FUNCTION_COLORS = {
    'up': COLORBLIND_COLORS['blue'],
    'down': COLORBLIND_COLORS['orange'],
    'q': COLORBLIND_COLORS['green']
}

def setup_publication_axes(ax, xlabel="", ylabel="", title="", legend=True, grid=True):
    """
    Apply consistent publication-ready styling to matplotlib axes.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to style
    xlabel : str
        X-axis label with LaTeX formatting
    ylabel : str  
        Y-axis label with LaTeX formatting
    title : str
        Plot title
    legend : bool
        Whether to show legend if present
    grid : bool
        Whether to show grid
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
    
    # Enhanced tick styling
    ax.tick_params(which='both', direction='in', labelsize=10)
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)
    
    # Grid styling
    if grid:
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Legend styling
    if legend and ax.get_legend():
        leg = ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        leg.get_frame().set_alpha(0.9)

def add_statistics_box(ax, data, position='top_left', format_str='.3f'):
    """
    Add a statistics text box to a plot.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the box to
    data : array-like
        Data to compute statistics for
    position : str
        Where to place the box ('top_left', 'top_right', 'bottom_left', 'bottom_right')
    format_str : str
        Format string for numbers
    """
    mean_val = np.mean(data)
    std_val = np.std(data)
    median_val = np.median(data)
    
    stats_text = f'μ = {mean_val:{format_str}}\nσ = {std_val:{format_str}}\nMedian = {median_val:{format_str}}'
    
    # Position mapping
    positions = {
        'top_left': (0.02, 0.98, 'top', 'left'),
        'top_right': (0.98, 0.98, 'top', 'right'), 
        'bottom_left': (0.02, 0.02, 'bottom', 'left'),
        'bottom_right': (0.98, 0.02, 'bottom', 'right')
    }
    
    x, y, va, ha = positions.get(position, positions['top_left'])
    
    ax.text(x, y, stats_text, transform=ax.transAxes,
           verticalalignment=va, horizontalalignment=ha, fontsize=9,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def create_error_summary_plot(
    errors_dict, 
    titles_dict=None,
    save_path="error_summary.png",
    figsize=(15, 10)
):
    """
    Create a comprehensive error summary plot with multiple error types.
    
    Parameters:
    -----------
    errors_dict : dict
        Dictionary with error type as key and error arrays as values
    titles_dict : dict, optional
        Dictionary with error type as key and plot titles as values
    save_path : str
        Path to save the summary plot
    figsize : tuple
        Figure size
    """
    n_error_types = len(errors_dict)
    cols = min(3, n_error_types)
    rows = (n_error_types + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_error_types == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    colors = [COLORBLIND_COLORS['blue'], COLORBLIND_COLORS['orange'], 
              COLORBLIND_COLORS['green'], COLORBLIND_COLORS['purple'],
              COLORBLIND_COLORS['brown'], COLORBLIND_COLORS['pink']]
    
    for i, (error_type, errors) in enumerate(errors_dict.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        color = colors[i % len(colors)]
        
        # Create histogram
        counts, bins, patches = ax.hist(
            errors, bins=30, alpha=0.7, color=color,
            edgecolor='white', linewidth=0.5, density=True
        )
        
        # Add mean and median lines
        mean_err = np.mean(errors)
        median_err = np.median(errors)
        ax.axvline(mean_err, color='red', linestyle='--', alpha=0.8, 
                  linewidth=2, label=f'Mean: {mean_err:.4f}')
        ax.axvline(median_err, color='purple', linestyle=':', alpha=0.8,
                  linewidth=2, label=f'Median: {median_err:.4f}')
        
        # Styling
        title = titles_dict.get(error_type, error_type) if titles_dict else error_type
        setup_publication_axes(ax, xlabel='Error Value', ylabel='Density', title=title)
        add_statistics_box(ax, errors, position='top_right')
        
    # Hide extra subplots
    for i in range(n_error_types, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Error Analysis Summary', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Import optional dependencies with fallbacks
try:
    import umap.umap_ as umap
except ImportError:
    umap = None

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
except ImportError:
    TSNE = None
    PCA = None

# Import simulator and other modules only when needed
def get_simulator_module():
    try:
        from simulator import SimplifiedDIS, RealisticDIS, MCEGSimulator
        return SimplifiedDIS, RealisticDIS, MCEGSimulator
    except ImportError:
        return None, None, None

def get_advanced_feature_engineering():
    try:
        from PDF_learning import advanced_feature_engineering
        return advanced_feature_engineering
    except ImportError:
        return lambda x: x  # fallback identity function

def get_plotting_driver_UQ():
    try:
        from plotting_driver_UQ import reload_pointnet
        return reload_pointnet
    except ImportError:
        return None

def get_datasets():
    try:
        import datasets
        return datasets
    except ImportError:
        return None

# ===========================
# Robust Precomputed Data Loading
# ===========================

def generate_precomputed_data_if_needed(problem, num_samples, num_events, n_repeat=2, output_dir="precomputed_data"):
    """
    Check if precomputed data exists for the given parameters, and generate it if not found.
    
    This function enforces exact matching: only files with the exact parameters 
    (problem, num_samples, num_events, n_repeat) will be accepted. If not found,
    it automatically generates the required data using generate_data_for_problem.
    
    Args:
        problem: Problem type ('gaussian', 'simplified_dis', 'realistic_dis', 'mceg')
        num_samples: Number of theta parameter samples
        num_events: Number of events per simulation
        n_repeat: Number of repeated simulations per theta
        output_dir: Directory where precomputed data should be stored
    
    Returns:
        str: Path to the data directory
        
    Raises:
        RuntimeError: If precomputed data support is not available or generation fails
    """
    print("🔍 PRECOMPUTED DATA DIAGNOSTIC:")
    print(f"   Looking for problem: '{problem}' with ns={num_samples}, ne={num_events}, nr={n_repeat}")
    print(f"   Data directory: '{output_dir}'")
    
    # Check precomputed data availability
    try:
        from precomputed_datasets import PrecomputedDataset
        PRECOMPUTED_AVAILABLE = True
    except ImportError:
        PRECOMPUTED_AVAILABLE = False
        
    if not PRECOMPUTED_AVAILABLE:
        print("   ✗ Precomputed data support not available. Please check precomputed_datasets.py")
        raise RuntimeError("Precomputed data support not available. Please check precomputed_datasets.py")
    
    # Check if data already exists
    import os
    expected_filename = f"{problem}_ns{num_samples}_ne{num_events}_nr{n_repeat}.npz"
    exact_file_path = os.path.join(output_dir, expected_filename)
    print(f"   Required exact file: '{exact_file_path}'")
    
    if os.path.exists(exact_file_path):
        print(f"   ✓ Found exact matching precomputed data: {exact_file_path}")
        return output_dir
    else:
        print(f"📊 Precomputed data not found for {problem} with ns={num_samples}, ne={num_events}, nr={n_repeat}")
        print("🚀 Generating precomputed data automatically...")
        
        try:
            # Import and run the data generation function
            from generate_precomputed_data import generate_data_for_problem
            import torch
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def get_robust_validation_data(args, problem, device, num_samples=1000):
    """
    Get validation data using robust precomputed data loading.
    
    This is a convenience wrapper around load_validation_dataset_batch that provides
    a simpler interface for getting validation data with automatic precomputed data
    generation if needed.
    
    Args:
        args: Argument namespace with validation parameters (val_samples, num_events, etc.)
        problem: Problem type string
        device: PyTorch device
        num_samples: Number of validation samples to load
        
    Returns:
        tuple: (thetas, xs) where both are already on the specified device
               xs is already feature engineered and ready for PointNet input
               
    Raises:
        RuntimeError: If validation data cannot be loaded or generated
    """
    return load_validation_dataset_batch(args, problem, device, num_samples)

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
        print(f"🔧 [FIX] get_analytic_uncertainty: Laplace model provided")
        print(f"🔧 [FIX] Laplace model type: {type(laplace_model)}")
        print(f"🔧 [FIX] Latent embedding shape: {latent_embedding.shape}")
        
        with torch.no_grad():
            # --- Path 1: predictive_distribution(x) -> distribution with .loc and .scale
            pred_dist_fn = getattr(laplace_model, "predictive_distribution", None)
            if callable(pred_dist_fn):
                try:
                    dist = pred_dist_fn(latent_embedding)
                    mean_params = dist.loc
                    std_params  = dist.scale
                    print(f"✅ [FIXED] Path 1 SUCCESS - Laplace uncertainty working!")
                    return mean_params.cpu(), std_params.cpu()
                except Exception as e:
                    print(f"❌ [DEBUG] Path 1 FAILED: {type(e).__name__}: {e}")

            # --- Path 2: calling the object sometimes returns (mean, var)
            try:
                out = laplace_model(latent_embedding, joint=False)
                if isinstance(out, tuple) and len(out) == 2:
                    pred_mean, pred_var = out
                    if pred_var.dim() == 3:
                        pred_std = torch.sqrt(torch.diagonal(pred_var, dim1=-2, dim2=-1))
                    else:
                        pred_std = torch.sqrt(pred_var.clamp_min(0))
                    print(f"✅ [FIXED] Path 2 SUCCESS - Laplace uncertainty working!")
                    return pred_mean.cpu(), pred_std.cpu()
                else:
                    print(f"❌ [DEBUG] Path 2 - output not a 2-tuple: {type(out)}")
            except Exception as e:
                print(f"❌ [DEBUG] Path 2 FAILED: {type(e).__name__}: {e}")

            # --- Path 3: predict(..., pred_type='glm', link_approx='mc')
            predict_fn = getattr(laplace_model, "predict", None)
            if callable(predict_fn):
                try:
                    pred = predict_fn(latent_embedding, pred_type='glm', link_approx='mc', n_samples=200)
                    if isinstance(pred, tuple) and len(pred) == 2:
                        mean, var = pred
                        std = torch.sqrt(var.clamp_min(0))
                        print(f"✅ [FIXED] Path 3 SUCCESS - Laplace uncertainty working!")
                        return mean.cpu(), std.cpu()
                    if hasattr(pred, "loc") and hasattr(pred, "scale"):
                        print(f"✅ [FIXED] Path 3 SUCCESS - Laplace uncertainty working!")
                        return pred.loc.cpu(), pred.scale.cpu()
                    else:
                        print(f"❌ [DEBUG] Path 3 - predict output lacks expected attributes")
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
        sel_stds  = torch.exp(0.5 * logvars[torch.arange(b), idx])
        return sel_means.cpu(), sel_stds.cpu()
    else:  # deterministic
        pred_mean = output
        pred_std  = torch.zeros_like(pred_mean)
        return pred_mean.cpu(), pred_std.cpu()


def get_gaussian_samples(model, latent_embedding, n_samples=100, laplace_model=None):
    """
    DEPRECATED: Use get_analytic_uncertainty for improved speed and accuracy.
    
    Legacy function for generating parameter samples from model uncertainty.
    When laplace_model is provided, converts analytic uncertainty to samples
    for backward compatibility. This function is maintained for compatibility
    but should be replaced with analytic methods in new code.
    
    Args:
        model: Neural network model (head)
        latent_embedding: Input latent embedding tensor
        n_samples: Number of samples to generate (for backward compatibility)
        laplace_model: Fitted Laplace approximation object
        
    Returns:
        torch.Tensor: Generated samples [n_samples, param_dim]
        
    Note: This function now uses analytic uncertainty internally and converts
    to samples, providing the same interface but with improved accuracy.
    """
    # For backward compatibility, convert analytic uncertainty to samples
    mean_params, std_params = get_analytic_uncertainty(model, latent_embedding, laplace_model)
    
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
    problem='simplified_dis'
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
        Problem type ('simplified_dis', 'realistic_dis', 'mceg', 'mceg4dis')
        
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
    SimplifiedDIS, RealisticDIS, MCEGSimulator = get_simulator_module()
    if problem == 'realistic_dis':
        simulator = RealisticDIS(torch.device('cpu'))
    elif problem in ['mceg', 'mceg4dis']:
        simulator = MCEGSimulator(torch.device('cpu'))
    else:
        simulator = SimplifiedDIS(torch.device('cpu'))

    advanced_feature_engineering = get_advanced_feature_engineering()
    
    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    if problem not in ['mceg', 'mceg4dis']:
        xs_tensor = advanced_feature_engineering(xs_tensor)
    else:
        from utils import log_feature_engineering
        xs_tensor = log_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    if laplace_model is not None:
        # Use analytic uncertainty propagation (delta method)
        mean_params, std_params = get_analytic_uncertainty(model, latent_embedding, laplace_model)
        mean_params = mean_params.cpu().squeeze(0)
        std_params = std_params.cpu().squeeze(0)
        use_analytic = True
        uncertainty_label = "Analytic (Laplace)"
    else:
        # Fallback to MC sampling for backward compatibility
        samples = get_gaussian_samples(model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model).cpu()
        mean_params = torch.mean(samples, dim=0)
        std_params = torch.std(samples, dim=0)
        use_analytic = False
        uncertainty_label = "Monte Carlo"

    n_params = true_params.size(0)
    
    # Set parameter names with proper mathematical notation
    if problem == 'simplified_dis':
        param_names = [r'$a_u$', r'$b_u$', r'$a_d$', r'$b_d$']
    elif problem == 'realistic_dis':
        param_names = [r'$\log A_0$', r'$\delta$', r'$a$', r'$b$', r'$c$', r'$d$']
    elif problem in ['mceg', 'mceg4dis']:
        param_names = [r'$\mu_1$', r'$\mu_2$', r'$\sigma_1$', r'$\sigma_2$']
    else:
        param_names = [f'$\\theta_{{{i+1}}}$' for i in range(n_params)]
    
    # Set up color palette
    base_colors = [COLORBLIND_COLORS['blue'], COLORBLIND_COLORS['orange'], 
                   COLORBLIND_COLORS['green'], COLORBLIND_COLORS['purple'],
                   COLORBLIND_COLORS['brown'], COLORBLIND_COLORS['pink']]
    
    # Create subplots with proper sizing
    cols = min(n_params, 4)  # Max 4 columns
    rows = (n_params + cols - 1) // cols  # Ceiling division
    figsize = (5 * cols, 4 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_params == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    # Hide extra subplots if needed
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    # Prepare SBI data for proper color cycling
    all_samples = [samples] if not use_analytic else []
    all_labels = [uncertainty_label]
    
    if compare_with_sbi and sbi_posteriors is not None:
        all_samples.extend(sbi_posteriors)
        if sbi_labels is not None:
            all_labels.extend(sbi_labels)
        else:
            all_labels.extend([f'SBI Method {i+1}' for i in range(len(sbi_posteriors))])

    for i in range(n_params):
        ax = axes[i]
        
        if use_analytic:
            # Plot analytic Gaussian distribution with enhanced aesthetics
            mu = mean_params[i].item()
            sigma = std_params[i].item()
            
            # Create x range around the mean (show ±4 standard deviations)
            x_range = max(4 * sigma, 0.1 * abs(mu))  # Ensure minimum range
            x_vals = torch.linspace(mu - x_range, mu + x_range, 1000)
            
            # Compute Gaussian PDF
            gaussian_pdf = torch.exp(-0.5 * ((x_vals - mu) / sigma) ** 2) / (sigma * torch.sqrt(2 * torch.tensor(torch.pi)))
            
            # Plot the analytic Gaussian with enhanced styling
            ax.plot(x_vals.numpy(), gaussian_pdf.numpy(), color=base_colors[0], linewidth=2.5, 
                   label=f'Posterior ({uncertainty_label})', alpha=0.9, zorder=3)
            ax.fill_between(x_vals.numpy(), 0, gaussian_pdf.numpy(), 
                          color=base_colors[0], alpha=0.25, zorder=2)
            
            # Add confidence intervals
            for n_sigma, alpha, label in [(1, 0.4, r'$\pm 1\sigma$'), (2, 0.2, r'$\pm 2\sigma$')]:
                lower, upper = mu - n_sigma * sigma, mu + n_sigma * sigma
                ax.axvspan(lower, upper, alpha=alpha, color=base_colors[0], zorder=1, 
                          label=label if i == 0 else "")
            
            # Set appropriate x limits
            ax.set_xlim(mu - x_range, mu + x_range)
            
        else:
            # Plot histogram from MC samples with enhanced styling
            param_vals = [s[:, i].numpy() for s in all_samples]
            xmin = min([v.min() for v in param_vals])
            xmax = max([v.max() for v in param_vals])
            padding = 0.1 * (xmax - xmin)
            xmin -= padding
            xmax += padding
            
            # Plot main posterior
            n, bins, patches = ax.hist(samples[:, i].numpy(), bins=30, alpha=0.7, density=True, 
                                     color=base_colors[0], label=uncertainty_label,
                                     edgecolor='white', linewidth=0.5)
            ax.set_xlim(xmin, xmax)

        # Add SBI comparison if requested
        if compare_with_sbi and sbi_posteriors is not None and sbi_labels is not None:
            for j, sbi_samples in enumerate(sbi_posteriors):
                color_idx = (j + 1) % len(base_colors)
                label = sbi_labels[j] if j < len(sbi_labels) else f"SBI {j+1}"
                ax.hist(
                    sbi_samples[:, i].detach().cpu().numpy(),
                    bins=30, alpha=0.6, density=True,
                    color=base_colors[color_idx],
                    label=label,
                    edgecolor='white',
                    linewidth=0.5
                )
        
        # Add true value line with enhanced styling
        true_val = true_params[i].item()
        ax.axvline(true_val, color=COLORBLIND_COLORS['red'], linestyle='--', 
                  linewidth=2.5, label='True Value', alpha=0.9, zorder=4)
        
        # Enhanced axis styling
        ax.set_title(f'Parameter {param_names[i]}', fontsize=14, pad=15, fontweight='bold')
        ax.set_xlabel(f'{param_names[i]}', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.tick_params(which='both', direction='in', labelsize=10)
        
        # Add legend only to first subplot to avoid clutter
        if i == 0: 
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        
        # Add statistics text box
        if use_analytic:
            stats_text = f'μ = {mu:.3f}\nσ = {sigma:.3f}'
        else:
            sample_mean = torch.mean(samples[:, i]).item()
            sample_std = torch.std(samples[:, i]).item()
            stats_text = f'μ = {sample_mean:.3f}\nσ = {sample_std:.3f}'
            
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Add overall title
    method_str = "Analytic Laplace" if use_analytic else "Monte Carlo"
    fig.suptitle(f'Parameter Posterior Distributions ({method_str} Uncertainty)', 
                fontsize=16, y=0.98, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_PDF_distribution_single(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=200,  # bump a bit if you like smoother quantiles
    laplace_model=None,
    problem='simplified_dis',
    Q2_slices=None,
    save_dir=None,
    save_path="pdf_distribution.png"
):
    """
    Create publication-ready PDF distribution plots with function-level uncertainty quantification.
    
    This function generates beautiful, clear plots showing posterior uncertainty over predicted 
    functions (PDFs), providing more interpretable uncertainty visualization than parameter-only plots.
    
    **ENHANCED APPROACH**: Uncertainty is computed over the predicted functions f(x), not just 
    parameter uncertainty. For each parameter sample θ drawn from the posterior, we evaluate 
    f(x|θ) at each x-point and then compute pointwise statistics (median, IQR) of the function values.

    ⚠️  **REPRODUCIBILITY WARNING**: This function requires simulation to generate event data
    for latent extraction from true parameters. Results may vary between runs unless random 
    seeds are fixed. For reproducible latent extraction, prefer using precomputed data via
    extract_latents_from_data() where possible.

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
        Number of Monte Carlo samples for uncertainty estimation
    laplace_model : object, optional
        Fitted Laplace approximation object for analytic uncertainty
    problem : str
        Problem type ('simplified_dis', 'realistic_dis', 'mceg')
    Q2_slices : list of float, optional
        Q² values for realistic_dis problem (ignored for simplified_dis)
    save_dir : str, optional
        Directory to save plots (if None, uses current directory)
    save_path : str
        Base name for saved plots
        
    Returns:
    --------
    None
        Saves publication-ready plots to specified paths
        
    Method:
    -------
    1. Extract latent representation from events generated with true parameters (via simulation)
    2. Sample parameters from posterior (Laplace if available, otherwise model intrinsic)
    3. For each parameter sample θ_i: evaluate f(x|θ_i) at each x in evaluation grid
    4. Compute pointwise median and quantiles of f(x) across all parameter samples
    5. Plot uncertainty bands reflecting function uncertainty at each x-point

    Features:
    ---------
    - Colorblind-friendly color palette
    - Professional mathematical notation
    - Clear uncertainty bands with IQR visualization
    - Proper log-scale handling
    - Statistical annotations and legends
    """
    model.eval()
    pointnet_model.eval()
    
    print(f"📊 Generating PDF distribution plot for {problem}")
    print(f"🎲 Simulating events from true parameters for latent extraction")
    
    # Get simulators with fallback
    SimplifiedDIS, RealisticDIS, MCEGSimulator = get_simulator_module()
    if problem == 'realistic_dis':
        simulator = RealisticDIS(torch.device('cpu'))
    elif problem in ['mceg', 'mceg4dis']:
        simulator = MCEGSimulator(torch.device('cpu'))
    else:
        simulator = SimplifiedDIS(torch.device('cpu'))

    advanced_feature_engineering = get_advanced_feature_engineering()
    
    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    if problem not in ['mceg', 'mceg4dis']:
        xs_tensor = advanced_feature_engineering(xs_tensor)
    else:
        from utils import log_feature_engineering
        xs_tensor = log_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # --- Enhanced Sampling Strategy ---
    if laplace_model is not None:
        samples = get_gaussian_samples(
            model,
            latent_embedding,
            n_samples=n_mc,
            laplace_model=laplace_model
        ).cpu()
        uncertainty_method = "Laplace Posterior"
        label_curve = "Median (Analytic Uncertainty)"
        label_band  = "IQR (Function Uncertainty)"
    else:
        samples = get_gaussian_samples(
            model,
            latent_embedding,
            n_samples=n_mc,
            laplace_model=None
        ).cpu()
        uncertainty_method = "Monte Carlo"
        label_curve = "Median (MC Uncertainty)"
        label_band  = "IQR (Function Uncertainty)"

    if problem == 'simplified_dis':
        x_vals = torch.linspace(0.001, 1, 500).to(device)  # Start slightly above 0 for log scale
        
        # Enhanced color scheme
        function_colors = {
            'up': COLORBLIND_COLORS['blue'],
            'down': COLORBLIND_COLORS['orange']
        }
        
        for fn_name, fn_label, _ in [("up", "u", None), ("down", "d", None)]:
            color = function_colors[fn_name]
            
            # Evaluate function for each sampled parameter vector
            fn_vals_all = []
            for i in range(samples.shape[0]):
                simulator.init(samples[i])
                fn = getattr(simulator, fn_name)
                fn_vals_all.append(fn(x_vals).unsqueeze(0))

            fn_stack = torch.cat(fn_vals_all, dim=0)  # [n_mc, 500]
            median_vals = fn_stack.median(dim=0).values.detach().cpu()
            lower_bounds = torch.quantile(fn_stack, 0.25, dim=0).detach().cpu()
            upper_bounds = torch.quantile(fn_stack, 0.75, dim=0).detach().cpu()
            
            # Additional confidence levels
            p05_bounds = torch.quantile(fn_stack, 0.05, dim=0).detach().cpu()
            p95_bounds = torch.quantile(fn_stack, 0.95, dim=0).detach().cpu()

            # True curve
            simulator.init(true_params.squeeze())
            true_vals = getattr(simulator, fn_name)(x_vals).detach().cpu()

            # Create enhanced plot
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Plot true function with enhanced styling
            ax.plot(x_vals.detach().cpu(), true_vals, 
                   label=fr"True ${fn_label}(x|\theta^*)$", 
                   color=COLORBLIND_COLORS['dark_green'], 
                   linewidth=3, alpha=0.9, zorder=3)
            
            # Plot predicted median with enhanced styling
            ax.plot(
                x_vals.detach().cpu(),
                median_vals,
                linestyle='-',
                label=fr"{label_curve} ${fn_label}(x)$",
                color=color,
                linewidth=2.5,
                alpha=0.9,
                zorder=2
            )
            
            # Plot uncertainty bands with multiple confidence levels
            ax.fill_between(
                x_vals.detach().cpu(),
                p05_bounds,
                p95_bounds,
                color=color,
                alpha=0.15,
                label="90% Confidence",
                zorder=0
            )
            
            ax.fill_between(
                x_vals.detach().cpu(),
                lower_bounds,
                upper_bounds,
                color=color,
                alpha=0.3,
                label="IQR (25%-75%)",
                zorder=1
            )

            # Enhanced axis styling
            ax.set_xlabel(r"$x$", fontsize=14)
            ax.set_ylabel(fr"${fn_label}(x|\theta)$", fontsize=14)
            ax.set_xlim(1e-3, 1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.3)
            ax.tick_params(which='both', direction='in', labelsize=11)
            
            # Enhanced legend
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, loc='best')
            
            # Enhanced title with method information
            ax.set_title(f"PDF Function Uncertainty: {fn_name.title()} Distribution\n"
                        f"Method: {uncertainty_method} ({n_mc} samples)", 
                        fontsize=14, pad=20, fontweight='bold')
            
            # Add statistical information box
            mean_error = torch.mean(torch.abs(median_vals - true_vals)).item()
            max_error = torch.max(torch.abs(median_vals - true_vals)).item()
            stats_text = f'Mean |Error|: {mean_error:.4f}\nMax |Error|: {max_error:.4f}'
            
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
                   verticalalignment='bottom', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            out_path = f"{save_dir}/{fn_name}_enhanced.png" if save_dir else f"{fn_name}_enhanced.png"
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    elif problem == 'realistic_dis':
        x_range = (1e-3, 0.9)
        x_vals = torch.linspace(x_range[0], x_range[1], 500).to(device)
        Q2_slices = Q2_slices or [2.0, 10.0, 50.0, 200.0]
        color_palette = plt.cm.viridis_r(np.linspace(0, 1, len(Q2_slices)))
        
        for i, Q2_fixed in enumerate(Q2_slices):
            Q2_vals = torch.full_like(x_vals, Q2_fixed).to(device)
            
            if use_analytic:
                # Compute PDF using analytic uncertainty
                simulator.init(mean_params)
                mean_q = simulator.q(x_vals, Q2_vals).detach().cpu()
                
                # Approximate uncertainty bounds
                param_std_norm = torch.norm(std_params).item()
                uncertainty_factor = 2.0 * param_std_norm
                
                lower_q = mean_q * (1 - uncertainty_factor)
                upper_q = mean_q * (1 + uncertainty_factor)
                lower_q = torch.clamp(lower_q, min=0.0)
                
            else:
                # MC sampling approach (legacy)
                q_vals_all = []
                for j in range(n_mc):
                    simulator.init(samples[j])
                    q_vals = simulator.q(x_vals, Q2_vals)
                    q_vals_all.append(q_vals.unsqueeze(0))
                q_stack = torch.cat(q_vals_all, dim=0)
                mean_q = torch.median(q_stack, dim=0).values.detach().cpu()
                lower_q = torch.quantile(q_stack, 0.25, dim=0).detach().cpu()
                upper_q = torch.quantile(q_stack, 0.75, dim=0).detach().cpu()

            # Compute true values
            simulator.init(true_params.squeeze())
            true_q = simulator.q(x_vals, Q2_vals).detach().cpu()

            # Create plot
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(x_vals.detach().cpu(), true_q, color=color_palette[i], linewidth=2.5,
                    label=fr"True $q(x,\ Q^2={Q2_fixed})$")
            
            if use_analytic:
                ax.plot(x_vals.detach().cpu(), mean_q, linestyle='--', color="crimson", linewidth=2,
                        label=fr"MAP $\hat{{q}}(x,\ Q^2={Q2_fixed})$ (Analytic)")
                ax.fill_between(x_vals.detach().cpu(), lower_q, upper_q, 
                               color="crimson", alpha=0.2, label="95% Analytic CI")
            else:
                ax.plot(x_vals.detach().cpu(), mean_q, linestyle='--', color="crimson", linewidth=2,
                        label=fr"Median $\hat{{q}}(x,\ Q^2={Q2_fixed})$ (MC)")
                ax.fill_between(x_vals.detach().cpu(), lower_q, upper_q, color="crimson", alpha=0.2)

            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$q(x,\ Q^2)$")
            ax.set_xlim(x_range)
            ax.set_xscale("log")
            ax.set_title(fr"$q(x)$ at $Q^2 = {Q2_fixed}\ \mathrm{{GeV}}^2$")
            ax.legend(frameon=False)
            ax.grid(True, which="both", linestyle=":", linewidth=0.5)
            plt.tight_layout()
            path = f"{save_dir}/q_Q2_{int(Q2_fixed)}.png" if save_dir else f"q_Q2_{int(Q2_fixed)}.png"
            plt.savefig(path, dpi=300)
            plt.close(fig)

def plot_PDF_distribution_single_same_plot(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,  # Kept for backward compatibility but ignored when using analytic uncertainty
    laplace_model=None,
    problem='simplified_dis',
    Q2_slices=None,
    plot_IQR=False,
    save_path="pdf_overlay.png"
):
    """
    Plot PDF distributions on the same plot using analytic Laplace uncertainty propagation.
    
    When laplace_model is provided, uses analytic uncertainty propagation to 
    compute error bands instead of Monte Carlo sampling for improved speed and accuracy.
    """
    model.eval()
    pointnet_model.eval()
    simulator = RealisticDIS(torch.device('cpu')) if problem == 'realistic_dis' else SimplifiedDIS(torch.device('cpu'))

    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    if problem not in ['mceg', 'mceg4dis']:
        xs_tensor = advanced_feature_engineering(xs_tensor)
    else:
        # FIXED: Apply log_feature_engineering for mceg/mceg4dis to match training
        from utils import log_feature_engineering
        xs_tensor = log_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    if laplace_model is not None:
        # Use analytic uncertainty propagation (delta method)
        mean_params, std_params = get_analytic_uncertainty(model, latent_embedding, laplace_model)
        mean_params = mean_params.cpu().squeeze(0)
        std_params = std_params.cpu().squeeze(0)
        use_analytic = True
    else:
        # Fallback to MC sampling for backward compatibility
        samples = get_gaussian_samples(model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model).cpu()
        use_analytic = False

    if problem == 'realistic_dis':
        x_range = (1e-3, 0.9)
        x_vals = torch.linspace(*x_range, 500).to(device)
        Q2_slices = Q2_slices or [1.0, 1.5, 2.0, 10.0, 50.0]
        fig, ax = plt.subplots(figsize=(8, 6))
        color_palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(Q2_slices)))

        for i, Q2_fixed in enumerate(Q2_slices):
            Q2_vals = torch.full_like(x_vals, Q2_fixed).to(device)
            
            if use_analytic:
                # Compute PDF using analytic uncertainty
                simulator.init(mean_params)
                mean_q = simulator.q(x_vals, Q2_vals).detach().cpu()
                
                # Approximate uncertainty bounds
                param_std_norm = torch.norm(std_params).item()
                uncertainty_factor = 2.0 * param_std_norm
                
                lower_q = mean_q * (1 - uncertainty_factor)
                upper_q = mean_q * (1 + uncertainty_factor)
                lower_q = torch.clamp(lower_q, min=0.0)
                
            else:
                # MC sampling approach (legacy)
                q_vals_all = []
                for j in range(n_mc):
                    simulator.init(samples[j])
                    q_vals = simulator.q(x_vals, Q2_vals)
                    q_vals_all.append(q_vals.unsqueeze(0))
                q_stack = torch.cat(q_vals_all, dim=0)
                mean_q = q_stack.median(dim=0).values.detach().cpu()
                lower_q = torch.quantile(q_stack, 0.25, dim=0).detach().cpu()
                upper_q = torch.quantile(q_stack, 0.75, dim=0).detach().cpu()

            # Compute true values
            simulator.init(true_params.squeeze())
            true_q = simulator.q(x_vals, Q2_vals).detach().cpu()

            # Plot true and predicted values
            ax.plot(x_vals.detach().cpu(), true_q, color=color_palette[i], linewidth=2,
                    label=fr"True $q(x,\ Q^2={Q2_fixed})$")
            
            if use_analytic:
                ax.plot(x_vals.detach().cpu(), mean_q, linestyle='--', color=color_palette[i], linewidth=1.8,
                        label=fr"MAP $\hat{{q}}(x,\ Q^2={Q2_fixed})$ (Analytic)")
            else:
                ax.plot(x_vals.detach().cpu(), mean_q, linestyle='--', color=color_palette[i], linewidth=1.8,
                        label=fr"Median $\hat{{q}}(x,\ Q^2={Q2_fixed})$ (MC)")
            
            # Add uncertainty bands if requested
            if plot_IQR or use_analytic:
                label_suffix = "95% Analytic CI" if use_analytic else "IQR"
                ax.fill_between(x_vals.detach().cpu(), lower_q, upper_q,
                                color=color_palette[i], alpha=0.2)
                
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$q(x,\ Q^2)$")
        uncertainty_type = "Analytic Laplace" if use_analytic else "MC Sampling"
        ax.set_title(fr"Posterior over $q(x, Q^2)$ at Multiple $Q^2$ Slices ({uncertainty_type})")
        ax.set_xscale("log")
        ax.set_xlim(x_range)
        ax.grid(True, which='both', linestyle=':', linewidth=0.6)
        ax.legend(loc="best", frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

def plot_parameter_error_histogram(
    true_params_list,
    predicted_params_list,
    param_names=None,
    save_path="parameter_error_histogram.png",
    problem='simplified_dis'
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
        Problem type ('simplified_dis' or 'realistic_dis') for default param names
        
    Returns:
    --------
    None
        Saves the plot to save_path
    """
    # Convert to numpy if needed and compute errors
    if isinstance(true_params_list[0], torch.Tensor):
        true_params_list = [p.detach().cpu().numpy() for p in true_params_list]
    if isinstance(predicted_params_list[0], torch.Tensor):
        predicted_params_list = [p.detach().cpu().numpy() for p in predicted_params_list]
        
    true_params = np.array(true_params_list)  # Shape: (n_samples, n_params)
    predicted_params = np.array(predicted_params_list)
    
    # Compute parameter errors
    param_errors = predicted_params - true_params  # Shape: (n_samples, n_params)
    relative_errors = param_errors / (true_params + 1e-8)  # Avoid division by zero
    
    n_params = param_errors.shape[1]
    
    # Set default parameter names
    if param_names is None:
        if problem == 'simplified_dis':
            param_names = [r'$a_u$', r'$b_u$', r'$a_d$', r'$b_d$']
        elif problem == 'realistic_dis':
            param_names = [r'$\log A_0$', r'$\delta$', r'$a$', r'$b$', r'$c$', r'$d$']
        else:
            param_names = [f'$\\theta_{{{i}}}$' for i in range(n_params)]
    
    # Create subplots
    fig, axes = plt.subplots(2, n_params, figsize=(4*n_params, 10))
    if n_params == 1:
        axes = axes.reshape(2, 1)
    
    colors_list = [COLORBLIND_COLORS['blue'], COLORBLIND_COLORS['orange'], 
                   COLORBLIND_COLORS['green'], COLORBLIND_COLORS['red']]
    
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
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add vertical line at zero
        ax_abs.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='True Value')
        
        # Statistics text
        mean_err = np.mean(param_errors[:, i])
        std_err = np.std(param_errors[:, i])
        ax_abs.text(0.02, 0.98, f'μ = {mean_err:.3f}\nσ = {std_err:.3f}', 
                   transform=ax_abs.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_abs.set_xlabel(f'Error in {param_names[i]}')
        ax_abs.set_ylabel('Frequency')
        ax_abs.set_title(f'Absolute Error: {param_names[i]}')
        ax_abs.grid(True, alpha=0.3)
        ax_abs.legend()
        
        # Relative errors (bottom row)
        ax_rel = axes[1, i]
        counts, bins, patches = ax_rel.hist(
            relative_errors[:, i] * 100,  # Convert to percentage
            bins=n_bins, 
            alpha=0.7, 
            color=color,
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add vertical line at zero
        ax_rel.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='True Value')
        
        # Statistics text
        mean_rel_err = np.mean(relative_errors[:, i]) * 100
        std_rel_err = np.std(relative_errors[:, i]) * 100
        ax_rel.text(0.02, 0.98, f'μ = {mean_rel_err:.1f}%\nσ = {std_rel_err:.1f}%', 
                   transform=ax_rel.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_rel.set_xlabel(f'Relative Error in {param_names[i]} (%)')
        ax_rel.set_ylabel('Frequency')
        ax_rel.set_title(f'Relative Error: {param_names[i]}')
        ax_rel.grid(True, alpha=0.3)
        ax_rel.legend()
    
    plt.suptitle('Parameter Error Analysis', fontsize=18, y=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_function_error_histogram(
    true_function_values_list,
    predicted_function_values_list,
    function_names=None,
    save_path="function_error_histogram.png",
    bins=50
):
    """
    Create publication-ready histograms of average entrywise function value errors.
    
    Parameters:
    -----------
    true_function_values_list : list of numpy arrays
        List of true function evaluations, each array of shape (n_points,) or (n_points, n_functions)
    predicted_function_values_list : list of numpy arrays
        List of predicted function evaluations, same shape as true_function_values_list
    function_names : list of str, optional
        Function names for labels (e.g., ['u(x)', 'd(x)'])
    save_path : str
        Path to save the histogram plot
    bins : int
        Number of histogram bins
        
    Returns:
    --------
    None
        Saves the plot to save_path
    """
    # Convert to numpy if needed
    if isinstance(true_function_values_list[0], torch.Tensor):
        true_function_values_list = [f.detach().cpu().numpy() for f in true_function_values_list]
    if isinstance(predicted_function_values_list[0], torch.Tensor):
        predicted_function_values_list = [f.detach().cpu().numpy() for f in predicted_function_values_list]
    
    # Ensure consistent shapes
    true_vals = np.array(true_function_values_list)
    pred_vals = np.array(predicted_function_values_list)
    
    if true_vals.ndim == 2:
        # Single function case: (n_samples, n_points) -> treat as single function
        true_vals = true_vals[:, :, np.newaxis]  # (n_samples, n_points, 1)
        pred_vals = pred_vals[:, :, np.newaxis]
    
    n_samples, n_points, n_functions = true_vals.shape
    
    # Set default function names
    if function_names is None:
        if n_functions == 2:
            function_names = [r'$u(x)$', r'$d(x)$']
        else:
            function_names = [f'$f_{{{i}}}(x)$' for i in range(n_functions)]
    
    # Compute average entrywise errors for each sample and function
    abs_errors = np.abs(pred_vals - true_vals)  # (n_samples, n_points, n_functions)
    avg_abs_errors = np.mean(abs_errors, axis=1)  # (n_samples, n_functions)
    
    rel_errors = abs_errors / (np.abs(true_vals) + 1e-8)  # Relative errors
    avg_rel_errors = np.mean(rel_errors, axis=1)  # (n_samples, n_functions)
    
    # Create subplots
    fig, axes = plt.subplots(2, n_functions, figsize=(6*n_functions, 10))
    if n_functions == 1:
        axes = axes.reshape(2, 1)
    
    colors_list = [COLORBLIND_COLORS['blue'], COLORBLIND_COLORS['orange'], 
                   COLORBLIND_COLORS['green'], COLORBLIND_COLORS['purple']]
    
    for i in range(n_functions):
        color = colors_list[i % len(colors_list)]
        
        # Absolute errors (top row)
        ax_abs = axes[0, i]
        counts, bins_abs, patches = ax_abs.hist(
            avg_abs_errors[:, i], 
            bins=bins, 
            alpha=0.7, 
            color=color,
            edgecolor='white',
            linewidth=0.5,
            density=True  # Normalize to show probability density
        )
        
        # Add statistics
        mean_abs = np.mean(avg_abs_errors[:, i])
        std_abs = np.std(avg_abs_errors[:, i])
        median_abs = np.median(avg_abs_errors[:, i])
        
        ax_abs.axvline(mean_abs, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_abs:.4f}')
        ax_abs.axvline(median_abs, color='purple', linestyle=':', alpha=0.8, linewidth=2, label=f'Median: {median_abs:.4f}')
        
        ax_abs.text(0.98, 0.98, f'μ = {mean_abs:.4f}\nσ = {std_abs:.4f}', 
                   transform=ax_abs.transAxes, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_abs.set_xlabel(f'Average Absolute Error in {function_names[i]}')
        ax_abs.set_ylabel('Probability Density')
        ax_abs.set_title(f'Distribution of Errors: {function_names[i]}')
        ax_abs.grid(True, alpha=0.3)
        ax_abs.legend()
        
        # Relative errors (bottom row)
        ax_rel = axes[1, i]
        counts, bins_rel, patches = ax_rel.hist(
            avg_rel_errors[:, i] * 100,  # Convert to percentage
            bins=bins, 
            alpha=0.7, 
            color=color,
            edgecolor='white',
            linewidth=0.5,
            density=True
        )
        
        # Add statistics
        mean_rel = np.mean(avg_rel_errors[:, i]) * 100
        std_rel = np.std(avg_rel_errors[:, i]) * 100
        median_rel = np.median(avg_rel_errors[:, i]) * 100
        
        ax_rel.axvline(mean_rel, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_rel:.2f}%')
        ax_rel.axvline(median_rel, color='purple', linestyle=':', alpha=0.8, linewidth=2, label=f'Median: {median_rel:.2f}%')
        
        ax_rel.text(0.98, 0.98, f'μ = {mean_rel:.2f}%\nσ = {std_rel:.2f}%', 
                   transform=ax_rel.transAxes, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_rel.set_xlabel(f'Average Relative Error in {function_names[i]} (%)')
        ax_rel.set_ylabel('Probability Density')
        ax_rel.set_title(f'Distribution of Relative Errors: {function_names[i]}')
        ax_rel.grid(True, alpha=0.3)
        ax_rel.legend()
    
    plt.suptitle('Function Value Error Analysis', fontsize=18, y=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_event_histogram_simplified_DIS(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,  # Kept for backward compatibility but ignored when using analytic uncertainty
    laplace_model=None,
    num_events=100000,
    save_path="event_histogram_simplified.png",
    problem='simplified_dis',
    plot_type='both',  # 'scatter', 'histogram', or 'both'
    bins=50,
    figsize=(15, 8)
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
        Problem type ('simplified_dis', 'realistic_dis', 'mceg')
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
    SimplifiedDIS, RealisticDIS, MCEGSimulator = get_simulator_module()
    if problem in ['mceg', 'mceg4dis']:
        simulator = MCEGSimulator(torch.device('cpu'))
    elif problem == 'realistic_dis':
        simulator = RealisticDIS(torch.device('cpu'))
    else:
        simulator = SimplifiedDIS(torch.device('cpu'))
    
    advanced_feature_engineering = get_advanced_feature_engineering()
    
    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), num_events).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    
    if problem not in ['mceg', 'mceg4dis']:
        xs_tensor = advanced_feature_engineering(xs_tensor)
    else:
        from utils import log_feature_engineering
        xs_tensor = log_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    if laplace_model is not None:
        # Use analytic uncertainty propagation (delta method)
        mean_params, std_params = get_analytic_uncertainty(model, latent_embedding, laplace_model)
        predicted_params = mean_params.cpu().squeeze(0)
        use_analytic = True
    else:
        # Fallback to MC sampling for backward compatibility
        samples = get_gaussian_samples(model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model).cpu()
        predicted_params = torch.median(samples, dim=0).values
        use_analytic = False

    # Generate events using the predicted parameters
    generated_events = simulator.sample(predicted_params.detach().cpu(), num_events).to(device)
    
    # Convert to numpy for plotting
    true_events_np = xs.detach().cpu().numpy()
    generated_events_np = generated_events.detach().cpu().numpy()
    
    # Determine number of subplots based on plot_type
    if plot_type == 'both':
        fig, axes = plt.subplots(2, 2, figsize=(figsize[0], figsize[1]*1.2))
        ax_true_scatter, ax_gen_scatter = axes[0, 0], axes[0, 1]
        ax_true_hist, ax_gen_hist = axes[1, 0], axes[1, 1]
    elif plot_type in ['scatter', 'histogram']:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_true, ax_gen = axes[0], axes[1]
    else:
        raise ValueError("plot_type must be 'scatter', 'histogram', or 'both'")
    
    method_label = "MAP (Analytic)" if use_analytic else "Median (MC)"
    
    # Color scheme
    true_color = COLORBLIND_COLORS['cyan']
    gen_color = COLORBLIND_COLORS['orange']
    
    # Helper function to set log scales and labels
    def setup_axes(ax, title, is_generated=False):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=14, pad=10)
        if problem == 'simplified_dis':
            if is_generated:
                ax.set_xlabel(fr"$x_{{u}} \sim u(x|\hat{{\theta}})$ ({method_label})", fontsize=12)
                ax.set_ylabel(fr"$x_{{d}} \sim d(x|\hat{{\theta}})$ ({method_label})", fontsize=12)
            else:
                ax.set_xlabel(r"$x_{u} \sim u(x|\theta^{*})$", fontsize=12)
                ax.set_ylabel(r"$x_{d} \sim d(x|\theta^{*})$", fontsize=12)
        else:
            ax.set_xlabel("$x$", fontsize=12)
            ax.set_ylabel("$Q^2$", fontsize=12)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(which='both', direction='in')
    
    # Plot scatter plots if requested
    if plot_type in ['scatter', 'both']:
        if plot_type == 'both':
            ax_true_scat, ax_gen_scat = ax_true_scatter, ax_gen_scatter
        else:
            ax_true_scat, ax_gen_scat = ax_true, ax_gen
            
        # True events scatter
        ax_true_scat.scatter(true_events_np[:, 0], true_events_np[:, 1], 
                           color=true_color, alpha=0.3, s=1.5, edgecolors='none')
        setup_axes(ax_true_scat, r"$\Xi_{\theta^{*}}$ (True Parameters) - Scatter", False)
        
        # Generated events scatter  
        ax_gen_scat.scatter(generated_events_np[:, 0], generated_events_np[:, 1], 
                          color=gen_color, alpha=0.3, s=1.5, edgecolors='none')
        setup_axes(ax_gen_scat, fr"$\Xi_{{\hat{{\theta}}}}$ ({method_label}) - Scatter", True)
    
    # Plot 2D histograms if requested
    if plot_type in ['histogram', 'both']:
        if plot_type == 'both':
            ax_true_hist_ax, ax_gen_hist_ax = ax_true_hist, ax_gen_hist
        else:
            ax_true_hist_ax, ax_gen_hist_ax = ax_true, ax_gen
        
        # Create log-spaced bins for better visualization
        x_min = min(np.min(true_events_np[:, 0]), np.min(generated_events_np[:, 0]))
        x_max = max(np.max(true_events_np[:, 0]), np.max(generated_events_np[:, 0]))
        y_min = min(np.min(true_events_np[:, 1]), np.min(generated_events_np[:, 1]))
        y_max = max(np.max(true_events_np[:, 1]), np.max(generated_events_np[:, 1]))
        
        # Add small margin in log space
        x_margin = (np.log10(x_max) - np.log10(x_min)) * 0.05
        y_margin = (np.log10(y_max) - np.log10(y_min)) * 0.05
        
        x_bins = np.logspace(np.log10(x_min) - x_margin, np.log10(x_max) + x_margin, bins)
        y_bins = np.logspace(np.log10(y_min) - y_margin, np.log10(y_max) + y_margin, bins)
        
        # True events histogram
        hist_true, _, _ = np.histogram2d(true_events_np[:, 0], true_events_np[:, 1], bins=[x_bins, y_bins])
        hist_true = hist_true.T  # Transpose for correct orientation
        
        # Only show non-zero bins
        hist_true_masked = np.ma.masked_where(hist_true == 0, hist_true)
        
        im_true = ax_true_hist_ax.pcolormesh(x_bins, y_bins, hist_true_masked, 
                                           cmap='Blues', norm=colors.LogNorm(vmin=1))
        setup_axes(ax_true_hist_ax, r"$\Xi_{\theta^{*}}$ (True Parameters) - 2D Histogram", False)
        
        # Add colorbar for true events
        cbar_true = plt.colorbar(im_true, ax=ax_true_hist_ax, fraction=0.046, pad=0.04)
        cbar_true.set_label('Event Count', fontsize=11)
        cbar_true.ax.tick_params(labelsize=10)
        
        # Generated events histogram
        hist_gen, _, _ = np.histogram2d(generated_events_np[:, 0], generated_events_np[:, 1], bins=[x_bins, y_bins])
        hist_gen = hist_gen.T  # Transpose for correct orientation
        
        # Only show non-zero bins
        hist_gen_masked = np.ma.masked_where(hist_gen == 0, hist_gen)
        
        im_gen = ax_gen_hist_ax.pcolormesh(x_bins, y_bins, hist_gen_masked, 
                                         cmap='Oranges', norm=colors.LogNorm(vmin=1))
        setup_axes(ax_gen_hist_ax, fr"$\Xi_{{\hat{{\theta}}}}$ ({method_label}) - 2D Histogram", True)
        
        # Add colorbar for generated events
        cbar_gen = plt.colorbar(im_gen, ax=ax_gen_hist_ax, fraction=0.046, pad=0.04)
        cbar_gen.set_label('Event Count', fontsize=11)
        cbar_gen.ax.tick_params(labelsize=10)
    
    # Add overall title
    if plot_type == 'both':
        fig.suptitle('Event Distribution Analysis: Scatter & Histogram Views', fontsize=16, y=0.95)
    elif plot_type == 'scatter':
        fig.suptitle('Event Distribution Analysis: Scatter View', fontsize=16, y=0.95)
    else:
        fig.suptitle('Event Distribution Analysis: Histogram View', fontsize=16, y=0.95)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_loss_curves(loss_dir='.', save_path='loss_plot.png', show_plot=False, nll_loss=False):
    contrastive_path = os.path.join(loss_dir, 'loss_contrastive.npy')
    regression_path = os.path.join(loss_dir, 'loss_regression.npy')
    total_path = os.path.join(loss_dir, 'loss_total.npy')
    contrastive_loss = np.load(contrastive_path)
    regression_loss = np.load(regression_path)
    total_loss = np.load(total_path)
    epochs = np.arange(1, len(contrastive_loss) + 1)
    loss_type = "NLL" if nll_loss else "MSE"
    regression_label = f'Regression Loss ({loss_type}, scaled)'
    title = f'Training Loss Components Over Epochs ({loss_type} Regression)'
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, contrastive_loss, label='Contrastive Loss', linewidth=2)
    plt.plot(epochs, regression_loss, label=regression_label, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    if show_plot: plt.show()
    plt.close()
    epochs = np.arange(1, len(total_loss) + 1)
    total_title = f'Training Loss Over Epochs ({loss_type} Regression)'
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, total_loss, label='Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(total_title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_PDF_learning.png', dpi=300)
    if show_plot: plt.show()
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, total_loss, label='Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Training Loss Over Epochs (Log Scale, {loss_type} Regression)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('log_loss_PDF_learning.png', dpi=300)
    if show_plot: plt.show()
    plt.close()


def plot_latents(latents, params, method='umap', param_idx=0, title=None, save_path=None):
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(latents)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(emb[:,0], emb[:,1], c=params[:,param_idx], cmap='viridis', s=30)
    plt.xlabel(f"{method.upper()} dim 1")
    plt.ylabel(f"{method.upper()} dim 2")
    plt.title(title or f"Latent space ({method.upper()}) colored by param {param_idx}")
    plt.colorbar(scatter, label=f"Parameter {param_idx}")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

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
    val_samples = getattr(args, 'val_samples', 1000)
    
    # Construct validation data path
    val_data_dir = getattr(args, 'precomputed_data_dir', 'precomputed_data')
    
    # Ensure precomputed data exists, generate if needed
    try:
        print(f"📂 Loading validation batch for {problem} with {num_samples} samples...")
        generate_precomputed_data_if_needed(
            problem=problem, 
            num_samples=val_samples, 
            num_events=args.num_events, 
            n_repeat=1,  # Validation data typically uses n_repeat=1
            output_dir=val_data_dir
        )
        
        # Load precomputed validation dataset
        from precomputed_datasets import PrecomputedDataset
        
        val_dataset = PrecomputedDataset(
            val_data_dir, 
            problem, 
            shuffle=False,
            exact_ns=val_samples, 
            exact_ne=args.num_events, 
            exact_nr=1
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
        
        print(f"✅ Loaded validation batch: thetas.shape={thetas.shape}, xs.shape={xs.shape}")  
        print(f"ℹ️  Note: Data is already feature engineered - do not apply engineering again")
        return thetas, xs
        
    except Exception as e:
        print(f"❌ Could not load precomputed validation dataset: {e}")
        raise RuntimeError(f"Failed to load validation data for {problem}. "
                         f"Precomputed data loading failed: {e}. "
                         f"Please check precomputed_datasets.py and generate_precomputed_data.py")


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
    print(f"ℹ️  Using precomputed data - feature engineering already applied during generation")
    
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

def plot_latents_umap(latents, params, color_mode='single', param_idx=0, method='umap', save_path=None, show=True):
    """
    Plot latent vectors (n_samples x latent_dim) reduced to 2D via UMAP or t-SNE,
    colored by parameters (n_samples x param_dim).
    """
    # Reduce latents to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(latents)

    # Determine coloring
    if color_mode == 'single':
        color = params[:, param_idx]
        label = f"Parameter {param_idx}"
    elif color_mode == 'mean':
        color = np.mean(params, axis=1)
        label = "Mean parameter"
    elif color_mode == 'pca':
        pca = PCA(n_components=1)
        color = pca.fit_transform(params).flatten()
        label = "First principal component of parameters"
    else:
        raise ValueError("color_mode must be 'single', 'mean', or 'pca'")

    # Plot
    plt.figure(figsize=(8,6))
    sc = plt.scatter(emb[:,0], emb[:,1], c=color, cmap='viridis', s=30)
    plt.xlabel(f"{method.upper()} dim 1")
    plt.ylabel(f"{method.upper()} dim 2")
    plt.title(f"Latent space ({method.upper()}), colored by {label}")
    plt.colorbar(sc, label=label)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

def plot_latents_all_params(latents, params, method='umap', save_path=None, show=True):
    """
    Plot latent vectors (n_samples x latent_dim) reduced to 2D via UMAP or t-SNE,
    with one subplot per parameter dimension.
    """
    # Reduce latents to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(latents)

    n_params = params.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))

    for i in range(n_params):
        ax = axes[i] if n_params > 1 else axes
        color = params[:, i]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=color, cmap='viridis', s=30)
        ax.set_xlabel(f"{method.upper()} dim 1")
        ax.set_ylabel(f"{method.upper()} dim 2")
        ax.set_title(f"Colored by Parameter {i}")
        plt.colorbar(sc, ax=ax, label=f"Parameter {i}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
def _bin_edges_log(evts_list, nx_bins=50, nQ2_bins=50, x_min=1e-4, x_max=1e-1, Q2_min=10.0, Q2_max=1e3):
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
        xs.append(E[:,0])
        Q2s.append(E[:,1])
    if xs:
        x_min = max(x_min, np.nanmax([np.nanmin(a[a>0]) for a in xs]))
        x_max = min(x_max, np.nanmax([np.nanmax(a) for a in xs]))
    if Q2s:
        Q2_min = max(Q2_min, np.nanmax([np.nanmin(a[a>0]) for a in Q2s]))
        Q2_max = min(Q2_max, np.nanmax([np.nanmax(a) for a in Q2s]))
    logx_edges  = np.linspace(np.log(x_min),  np.log(x_max),  nx_bins+1)
    logQ2_edges = np.linspace(np.log(Q2_min), np.log(Q2_max), nQ2_bins+1)
    return logx_edges, logQ2_edges

def _hist2d_density_log(evts, logx_edges, logQ2_edges, total_xsec=None):
    """
    Histogram in (log x, log Q2), convert to differential rate by dividing by bin area (dx*dQ2).
    Optionally scale to match total cross section like your original code.
    """
    if evts is None or len(evts) == 0:
        H = np.zeros((len(logx_edges)-1, len(logQ2_edges)-1), dtype=float)
        return H, (logx_edges, logQ2_edges)

    H, xedges, q2edges = np.histogram2d(np.log(evts[:,0]), np.log(evts[:,1]),
                                        bins=(logx_edges, logQ2_edges))
    # Convert counts to density via (dx*dQ2)
    # Precompute dx, dQ2 on linear scale for each bin
    dx  = np.exp(xedges[1:]) - np.exp(xedges[:-1])          # (nx,)
    dQ2 = np.exp(q2edges[1:]) - np.exp(q2edges[:-1])        # (nQ2,)
    area = dx[:, None] * dQ2[None, :]
    density = np.divide(H, area, where=(area>0))
    if total_xsec is not None and H.sum() > 0:
        density *= total_xsec / H.sum()
    return density, (xedges, q2edges)

def _theory_grid(idis, xedges, q2edges, rs, tar, mode='xQ2'):
    """
    Evaluate theory on bin centers defined by (xedges, q2edges).
    """
    nx  = len(xedges)-1
    nQ2 = len(q2edges)-1
    out = np.zeros((nx, nQ2), dtype=float)
    # Bin centers in linear space
    x_centers  = np.exp(0.5*(xedges[:-1]  + xedges[1:]))
    q2_centers = np.exp(0.5*(q2edges[:-1] + q2edges[1:]))

    for i in tqdm(range(nx), desc="theory x bins", leave=False):
        x = x_centers[i]
        for j in range(nQ2):
            Q2 = q2_centers[j]
            out[i, j] = idis.get_diff_xsec(x, Q2, rs, tar, mode)
    return out

def _theory_grid_masked(idis, xedges, q2edges, rs, tar, mode, occupancy_counts):
    nx, nQ2 = len(xedges)-1, len(q2edges)-1
    out = np.zeros((nx, nQ2), dtype=float)
    x_centers  = np.exp(0.5*(xedges[:-1]  + xedges[1:]))
    q2_centers = np.exp(0.5*(q2edges[:-1] + q2edges[1:]))
    for i in range(nx):
        for j in range(nQ2):
            if occupancy_counts[i, j] > 0:      # <-- workbook’s guard
                val = idis.get_diff_xsec(x_centers[i], q2_centers[j], rs, tar, mode)
                out[i, j] = _to_scalar_xsec(val, mode_hint=mode)
    return out

def _to_scalar_xsec(v, mode_hint='xQ2'):
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
    if ev is None: return None
    ev = np.asarray(ev)
    m = np.isfinite(ev).all(axis=1) & (ev[:,0] > 0) & (ev[:,1] > 0)
    return ev[m]

def safe_log_levels(A, n=60, lo_pct=1.0, hi_pct=99.0, default=(1e-6, 1.0)):
    A = np.asarray(A, dtype=float)

    # keep only positive finite values
    A = np.where(np.isfinite(A) & (A > 0), A, np.nan)

    # pick percentiles to avoid extreme outliers
    vmin = np.nanpercentile(A, lo_pct)
    vmax = np.nanpercentile(A, hi_pct)

    # fallback if bad or empty
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0 or vmin >= vmax:
        vmin, vmax = default

    # log-spaced levels, same as your inline construction
    levels = 10**np.linspace(np.log10(vmin), np.log10(vmax), n)

    return levels

from typing import Optional, Tuple

@torch.no_grad()
def _simulate_pdf_curve_from_theta(
    simulator,
    theta: torch.Tensor,
    num_events: int,
    x_range: Tuple[float, float],
    bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate events under `theta` and return a density curve over x via a shared histogram.
    Returns (x_centers [B], pdf_values [B]).
    """
    xs = simulator.sample(theta.detach().cpu().float(), num_events)  # shape (N, 2) or (N, ...)
    x = np.asarray(xs)[:, 0]  # assume x is first column
    H, edges = np.histogram(x, bins=bins, range=x_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, H

def _to_SD(theta_samples: torch.Tensor) -> Tuple[int, int]:
    """Return (S, D). Accepts [S,D] or [B,S,D] -> flattens across B."""
    if theta_samples.dim() == 3:
        B, S, D = theta_samples.shape
        return B * S, D
    elif theta_samples.dim() == 2:
        S, D = theta_samples.shape
        return S, D
    else:
        raise ValueError(f"theta_samples must be [S,D] or [B,S,D], got {tuple(theta_samples.shape)}")

@torch.no_grad()
def plot_PDF_distribution_single_same_plot_from_theta_samples(
    simulator,
    theta_samples: torch.Tensor,   # [S,D] or [B,S,D] on any device
    true_params: Optional[torch.Tensor],
    device: torch.device,
    num_events_per_theta: int = 5000,
    x_range: Tuple[float, float] = (0.0, 1.0),
    bins: int = 100,
    quantiles = (5, 25, 50, 75, 95),
    overlay_point_estimate: bool = True,
    point_estimate: str = "mean",  # "mean" or "median"
    save_path: str = "pdf_overlay_flow.png",
    title: Optional[str] = None,
):
    """
    Builds posterior bands over the induced PDF(x) using simulator + theta_samples.
    - Posterior bands: 5–95% and 25–75% + median curve.
    - Optional overlays: truth curve (true_params) and a point-estimate curve (mean/median θ).
    """
    # Sanitize shapes
    if theta_samples.dim() == 3:
        B, S, D = theta_samples.shape
        thetas = theta_samples.reshape(B * S, D)
    else:
        thetas = theta_samples
        S, D = thetas.shape

    # Precompute a common x-grid (by simulating once with the first theta)
    x_centers_ref, _ = _simulate_pdf_curve_from_theta(
        simulator, thetas[0].to(device), max(2000, num_events_per_theta // 5), x_range, bins
    )
    # Simulate all θ-samples → stack PDFs
    pdf_mat = []
    for s in range(thetas.shape[0]):
        _, H = _simulate_pdf_curve_from_theta(
            simulator, thetas[s].to(device), num_events_per_theta, x_range, bins
        )
        pdf_mat.append(H)
    pdf_mat = np.stack(pdf_mat, axis=0)  # [S_total, BINS]

    # Quantile bands
    qdict = {q: np.quantile(pdf_mat, q/100.0, axis=0) for q in quantiles}

    # Optional truth curve
    truth_curve = None
    if true_params is not None:
        x_t, H_t = _simulate_pdf_curve_from_theta(
            simulator, true_params.to(device), num_events_per_theta * 5, x_range, bins
        )
        truth_curve = (x_t, H_t)

    # Optional point-estimate curve
    pe_curve = None
    if overlay_point_estimate:
        if point_estimate == "mean":
            theta_pe = thetas.mean(dim=0)
        elif point_estimate == "median":
            theta_pe = thetas.median(dim=0).values
        else:
            raise ValueError("point_estimate must be 'mean' or 'median'")
        x_pe, H_pe = _simulate_pdf_curve_from_theta(
            simulator, theta_pe.to(device), num_events_per_theta * 2, x_range, bins
        )
        pe_curve = (x_pe, H_pe)

    # Plot
    fig = plt.figure(figsize=(7, 4.5))
    ax = plt.gca()

    # Shaded bands
    if 95 in qdict and 5 in qdict:
        ax.fill_between(x_centers_ref, qdict[5], qdict[95], alpha=0.20, label="90% band")
    if 75 in qdict and 25 in qdict:
        ax.fill_between(x_centers_ref, qdict[25], qdict[75], alpha=0.35, label="50% band")

    # Median curve
    if 50 in qdict:
        ax.plot(x_centers_ref, qdict[50], linewidth=2.0, label="Posterior median")

    # Point estimate curve
    if pe_curve is not None:
        ax.plot(pe_curve[0], pe_curve[1], linestyle="--", linewidth=1.8,
                label=f"Posterior {point_estimate}")

    # Truth
    if truth_curve is not None:
        ax.plot(truth_curve[0], truth_curve[1], linewidth=2.0, label="Truth")

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    if title: ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_PDF_distribution_single_same_plot_mceg(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,                    # kept for backward compatibility
    laplace_model=None,
    problem='simplified_dis',
    Q2_slices=None,              # list of Q² values to show; if None we auto-pick
    plot_IQR=False,              # used only for MC mode (overlay disabled by default)
    save_dir=None,
    nx=100,
    nQ2=100,
    n_events=1000000,
    max_Q2_for_plot=100.0,
):
    """
    Reproduce the 'true vs reconstructed' plot style:
      - 2D histogram in (log x, log Q²) for reconstructed with error bars (Poisson)
      - 'true' curve from simulator.q at bin centers
      - OPTIONAL model overlay (MAP dashed) ONLY when laplace_model is provided
    """
    # -------- setup ----------
    model.eval()
    pointnet_model.eval()
    if problem in ['mceg', 'mceg4dis']:
        simulator = MCEGSimulator(torch.device('cpu'))
    elif problem == 'realistic_dis':
        simulator = RealisticDIS(torch.device('cpu'))
    elif problem == 'simplified_dis':
        simulator = SimplifiedDIS(torch.device('cpu'))
    # simulator = RealisticDIS(torch.device('cpu')) if problem == 'realistic_dis' else SimplifiedDIS(torch.device('cpu'))

    # Make sure params live on correct device
    true_params = true_params.to(device)

    # -------- initialize theory components ----------

    mellin = MELLIN(npts=8)
    alphaS = ALPHAS()
    eweak  = EWEAK()
    pdf    = PDF(mellin, alphaS)

    # -------- sample events for the reconstructed histogram ----------
    with torch.no_grad():
        events = simulator.sample(true_params.detach().cpu(), n_events)  # expected shape (N, 2) = [x, Q2]
    events = np.asarray(events)
    x_ev  = events[:, 0]
    Q2_ev = events[:, 1]

    xs_tensor = torch.tensor(events, dtype=torch.float32, device=device)
    
    print(f"🔧 [FIX] mceg feature engineering - problem: {problem}")
    print(f"🔧 [FIX] Raw events shape: {xs_tensor.shape}")
    
    if problem not in ['mceg', 'mceg4dis']:
        xs_tensor = advanced_feature_engineering(xs_tensor)
        print(f"🔧 [FIX] After advanced_feature_engineering: {xs_tensor.shape}")
    else:
        # FIXED: Apply log_feature_engineering for mceg/mceg4dis to match training
        from utils import log_feature_engineering
        xs_tensor = log_feature_engineering(xs_tensor)
        print(f"✅ [FIXED] mceg/mceg4dis: Applied log_feature_engineering - shape: {xs_tensor.shape}")
    
    print(f"🔧 [FIX] Final tensor shape for PointNet: {xs_tensor.shape}")
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))
    print(f"🔧 [FIX] Latent embedding shape: {latent_embedding.shape}")
    
    # Now prediction should work correctly with proper input dimensions
    theta_pred = model(latent_embedding).cpu().squeeze(0).detach()
    print(f"✅ [FIXED] Predicted parameters: {theta_pred}")

    new_cpar = pdf.get_current_par_array()[::]
    # Assume parameters are only corresponding to 'uv1' parameters
    if not isinstance(theta_pred, torch.Tensor):
        new_cpar[4:8] = theta_pred
    else:
        new_cpar[4:8] = theta_pred.cpu().numpy()  # Update uv1 parameters
    pdf.setup(new_cpar)
    idis = THEORY(mellin, pdf, alphaS, eweak)
    new_cpar_true = pdf.get_current_par_array()[::]
    new_cpar_true[4:8] = true_params.cpu().numpy() if isinstance(true_params, torch.Tensor) else true_params
    pdf_true = PDF(mellin, alphaS)
    pdf_true.setup(new_cpar_true)
    idis_true = THEORY(mellin, pdf_true, alphaS, eweak)
    mceg=MCEG(idis,rs=140,tar='p',W2min=10,nx=nx,nQ2=nQ2) 
    mceg_true = MCEG(idis_true,rs=140,tar='p',W2min=10,nx=nx,nQ2=nQ2)
    events_pred = mceg.gen_events(n_events,verb=False)

    events = mceg_true.gen_events(n_events,verb=False)
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
    
    hist = np.histogram2d(np.log(evts[:,0]), np.log(evts[:,1]), bins=(logx_edges, logQ2_edges))
    true=np.zeros(hist[0].shape)
    reco=np.zeros(hist[0].shape)
    gen=np.zeros(hist[0].shape)
    for i,j in tqdm((a,b) for a in range(hist[1].shape[0]-1) 
                        for b in range(hist[2].shape[0]-1)):
        if hist[0][i,j]>0: 
            x=np.exp(0.5*(hist[1][i]+hist[1][i+1]))
            Q2=np.exp(0.5*(hist[2][j]+hist[2][j+1]))
            true[i,j],_=idis_true.get_diff_xsec(x,Q2,mceg_true.rs,mceg_true.tar,'xQ2')
            
            dx=np.exp(hist[1][i+1])-np.exp(hist[1][i])
            dQ2=np.exp(hist[2][j+1])-np.exp(hist[2][j])
            reco[i,j]=hist[0][i,j]/dx/dQ2
            gen[i,j],_=idis.get_diff_xsec(x,Q2,mceg.rs,mceg.tar,'xQ2')

    reco*=mceg_true.total_xsec/np.sum(hist[0])
    gen*=mceg.total_xsec/np.sum(hist[0])  # Fixed: use hist[0] instead of gen for consistency


    nrows,ncols=1,3; AX=[]
    fig = py.figure(figsize=(ncols*6,nrows*5))
    ax=py.subplot(nrows,ncols,1);AX.append(ax)
    c=ax.pcolor(hist[1],hist[2],reco.T, norm=matplotlib.colors.LogNorm())
    ax=py.subplot(nrows,ncols,2);AX.append(ax)
    c=ax.pcolor(hist[1],hist[2],true.T, norm=matplotlib.colors.LogNorm())
    ax=py.subplot(nrows,ncols,3);AX.append(ax)
    c=ax.pcolor(hist[1],hist[2],gen.T, norm=matplotlib.colors.LogNorm())
    for ax in AX:
        ax.tick_params(axis='both', which='major', labelsize=20,direction='in')
        ax.set_ylabel(r'$Q^2$',size=30)
        ax.set_xlabel(r'$x$',size=30)
        ax.set_xticks(np.log([1e-4,1e-3,1e-2,1e-1]))
        ax.set_xticklabels([r'$0.0001$',r'$0.001$',r'$0.01$',r'$0.1$'])
        ax.set_yticks(np.log([10,100,1000]))
        ax.set_yticklabels([r'$10$',r'$100$',r'$1000$']);
    AX[0].text(0.1,0.8,r'$\rm Reco$',transform=AX[0].transAxes,size=30)
    AX[1].text(0.1,0.8,r'$\rm True$',transform=AX[1].transAxes,size=30)
    AX[2].text(0.1,0.8,r'$\rm Gen$',transform=AX[2].transAxes,size=30)
    py.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        py.savefig(os.path.join(save_dir, 'PDF_2D_distribution_mceg_oldstyle.png'), dpi=300)
    
    nrows,ncols=1,3; AX=[]

    fig = py.figure(figsize=(ncols*6,nrows*5))
    cmap='gist_rainbow'

    ax=py.subplot(nrows,ncols,1);AX.append(ax)
    levels=10**np.linspace( np.log10(np.amin(reco[reco>0])),np.log10(np.amax(reco)),60)
    cs = ax.contour(hist[1][:-1],hist[2][:-1],reco.T,levels=levels,cmap=cmap,norm=colors.LogNorm())
    ax=py.subplot(nrows,ncols,2);AX.append(ax)
    cs = ax.contour(hist[1][:-1],hist[2][:-1],true.T,levels=levels,cmap=cmap,norm=colors.LogNorm())
    ax=py.subplot(nrows,ncols,3);AX.append(ax)
    cs = ax.contour(hist[1][:-1],hist[2][:-1],gen.T,levels=levels,cmap=cmap,norm=colors.LogNorm())
    for ax in AX:
        ax.tick_params(axis='both', which='major', labelsize=20,direction='in')
        ax.set_ylabel(r'$Q^2$',size=30)
        ax.set_xlabel(r'$x$',size=30)
        ax.set_xticks(np.log([1e-4,1e-3,1e-2,1e-1]))
        ax.set_xticklabels([r'$0.0001$',r'$0.001$',r'$0.01$',r'$0.1$'])
        ax.set_yticks(np.log([10,100,1000]))
        ax.set_yticklabels([r'$10$',r'$100$',r'$1000$']);
    AX[0].text(0.1,0.8,r'$\rm Reco$',transform=AX[0].transAxes,size=30)
    AX[1].text(0.1,0.8,r'$\rm True$',transform=AX[1].transAxes,size=30)
    AX[2].text(0.1,0.8,r'$\rm Gen$',transform=AX[2].transAxes,size=30)
    py.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        py.savefig(os.path.join(save_dir, 'PDF_2D_distribution_mceg_contour.png'), dpi=300)

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
        print(f"⚠️ [MCEG4DIS] No valid Q2 slices in range [{Q2_min}, {Q2_max}], using default range")
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
            true_slice[j] = idis_true.get_diff_xsec(x_val, Q2_fixed, mceg_true.rs, mceg_true.tar, 'xQ2')
            pred_slice[j] = idis.get_diff_xsec(x_val, Q2_fixed, mceg.rs, mceg.tar, 'xQ2')
        
        # Plot true curve
        ax.plot(x_slice_vals, true_slice, 
               color=color_palette[i], linestyle='-', linewidth=2.5,
               label=f'True Q²={Q2_fixed}', alpha=0.8)
        
        # Plot predicted curve
        ax.plot(x_slice_vals, pred_slice,
               color=color_palette[i], linestyle='--', linewidth=2,
               label=f'Pred Q²={Q2_fixed}', alpha=0.8)
        
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
                    dx = np.exp(hist[1][k+1]) - np.exp(hist[1][k])
                    dQ2 = np.exp(hist[2][Q2_idx+1]) - np.exp(hist[2][Q2_idx])
                    stat_err_vals[k] = np.sqrt(hist[0][k, Q2_idx]) / (dx * dQ2)
                    # Apply same normalization as reco
                    stat_err_vals[k] *= mceg_true.total_xsec / np.sum(hist[0])
            
            # Only plot error bars where we have significant statistics
            mask = reco_vals > 0
            if np.any(mask):
                ax.errorbar(x_err_vals[mask], reco_vals[mask], yerr=stat_err_vals[mask],
                           color=color_palette[i], fmt='o', markersize=3, alpha=0.6,
                           label=f'Reco±stat Q²={Q2_fixed}')
    
    # Format the plot with log-scaled axes as specified in requirements
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('Differential Cross Section', fontsize=16)
    ax.set_title('PDF Q² Slices - mceg4dis Compatible', fontsize=18)
    ax.set_xscale('log')  # log-scaled axes as required
    ax.set_yscale('log')  # log-scaled axes as required  
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    py.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        slice_plot_path = os.path.join(save_dir, 'PDF_Q2_slices_mceg4dis.png')
        py.savefig(slice_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ [MCEG4DIS] Q² slice plot saved to: {slice_plot_path}")
    
    py.close(fig)
    print(f"✅ [MCEG4DIS] Enhanced mceg4dis-compatible PDF plotting completed")
    print(f"✅ [MCEG4DIS] Generated: 2D histograms + Q² slices with log-scaled axes and error bars")


#     # 2) Compute histogram once, vectorize the density (no per-bin loop needed)
#     hist = np.histogram2d(np.log(evts_reco[:,0]), np.log(evts_reco[:,1]), bins=(50,50))
#     H = hist[0]                    # (nx, nQ2)
#     logx_edges = hist[1]
#     logQ2_edges = hist[2]
#     dx  = np.diff(np.exp(logx_edges))    # (nx,)
#     dQ2 = np.diff(np.exp(logQ2_edges))    # (nQ2,)
#     # avoid division by zero
#     dx  = np.where(dx  > 0, dx,  np.nan)
#     dQ2 = np.where(dQ2 > 0, dQ2, np.nan)
#     reco = H / (dx[:,None] * dQ2[None,:])

#     # scale (guard sum==0)
#     Hsum = H.sum()
#     if Hsum > 0:
#         reco *= (mceg_true.total_xsec / Hsum) 

#     gen_hist = np.histogram2d(np.log(evts_pred[:,0]), np.log(evts_pred[:,1]), bins=(logx_edges, logQ2_edges))
#     gen = gen_hist[0]                    # (nx, nQ2)
#     gen_Hsum = gen.sum()
#     gen = gen / (dx[:,None] * dQ2[None,:])
#     if gen_Hsum > 0:
#         gen *= (mceg.total_xsec / gen_Hsum)
    

#     # 3) Fill "true" and "gen" safely; define x,Q2 for every bin center
#     true = np.zeros_like(reco, dtype=float)
#     xc   = np.exp(0.5*(logx_edges[:-1] + logx_edges[1:]))   # (nx,)
#     Q2c  = np.exp(0.5*(logQ2_edges[:-1] + logQ2_edges[1:])) # (nQ2,)

#     for i in range(len(xc)):
#         for j in range(len(Q2c)):
#             x  = float(xc[i]); Q2 = float(Q2c[j])
#             # Evaluate theory everywhere, but guard exceptions/negatives
#             try:
#                 tval, _ = idis_true.get_diff_xsec(x, Q2, mceg_true.rs, mceg_true.tar, 'xQ2')
#             except Exception:
#                 tval = np.nan
#             true[i,j] = tval if np.isfinite(tval) and tval >= 0 else np.nan

#     # 4) Safe levels + consistent LogNorm bounds
#     levels = safe_log_levels(reco, n=60)

#     # --- Pseudocolor plots (with explicit vmin/vmax) ---
#     fig = py.figure(figsize=(18,5)); AX=[]
#     ax=py.subplot(1,3,1); AX.append(ax)
#     c=ax.pcolor(logx_edges, logQ2_edges, np.where(reco>0, reco, np.nan).T,
#                 norm=colors.LogNorm())
#     ax=py.subplot(1,3,2); AX.append(ax)
#     c=ax.pcolor(logx_edges, logQ2_edges, np.where(true>0, true, np.nan).T,
#                 norm=colors.LogNorm())
#     ax=py.subplot(1,3,3); AX.append(ax)
#     c=ax.pcolor(logx_edges, logQ2_edges, np.where(gen>0, gen, np.nan).T,
#                norm=colors.LogNorm())

    
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         py.savefig(os.path.join(save_dir, 'PDF_2D_distribution_mceg_pcolor.png'), dpi=300)
# # --- Panels: Gen / Reco / True, styled like your target snippet ---
#     nrows, ncols = 1, 3
#     fig = py.figure(figsize=(ncols*6, nrows*5))
#     AX = []
#     cmap = 'gist_rainbow'

#     # Use your log-binned edges (swap to hist[1], hist[2] if that's what you actually have)
#     x_edges = logx_edges     # or: hist[1]
#     y_edges = logQ2_edges    # or: hist[2]

#     # Robust log-spaced levels from reco (fallback if helper isn't defined)
#     try:
#         levels = safe_log_levels(reco, n=60, lo_pct=1.0, hi_pct=99.0, default=(1e-6, 1.0))
#     except NameError:
#         reco_pos = reco[np.isfinite(reco) & (reco > 0)]
#         if reco_pos.size == 0:
#             # harmless default if reco has no positives
#             levels = 10.0 ** np.linspace(-6, 0, 60)
#         else:
#             vmin = np.percentile(reco_pos, 1.0)
#             vmax = np.percentile(reco_pos, 99.0)
#             vmin = max(vmin, 1e-12)
#             vmax = max(vmax, vmin * 10)
#             levels = 10.0 ** np.linspace(np.log10(vmin), np.log10(vmax), 60)

#     def _contour_panel(ax, Z, title):
#         Zp = np.where(np.isfinite(Z) & (Z > 0), Z, np.nan)
#         if np.isfinite(Zp).any():
#             cs = ax.contour(x_edges[:-1], y_edges[:-1], Zp.T,
#                             levels=levels, cmap=cmap, norm=colors.LogNorm())
#             ax.set_title(title, fontsize=18)
#             return cs
#         else:
#             ax.text(0.5, 0.5, f'No positive data for {title}',
#                     ha='center', va='center', transform=ax.transAxes)
#             return None

#     # Create panels (keep "Generated (Ours)")
#     ax = py.subplot(nrows, ncols, 1); AX.append(ax); cs_gen  = _contour_panel(ax, gen,  'Generated (Ours)')
#     ax = py.subplot(nrows, ncols, 2); AX.append(ax); cs_reco = _contour_panel(ax, reco, 'Reco')
#     ax = py.subplot(nrows, ncols, 3); AX.append(ax); cs_true = _contour_panel(ax, true, 'True')

#     # Shared styling like your example
#     for ax in AX:
#         ax.tick_params(axis='both', which='major', labelsize=20, direction='in')
#         ax.set_xlabel(r'$x$',  size=30)
#         ax.set_ylabel(r'$Q^2$', size=30)
#         ax.set_xticks(np.log([1e-4, 1e-3, 1e-2, 1e-1]))
#         ax.set_xticklabels([r'$0.0001$', r'$0.001$', r'$0.01$', r'$0.1$'])
#         ax.set_yticks(np.log([10, 100, 1000]))
#         ax.set_yticklabels([r'$10$', r'$100$', r'$1000$'])

#     # One colorbar for whichever panel rendered last successfully
#     for cs in (cs_true, cs_reco, cs_gen):
#         if cs is not None:
#             cbar = fig.colorbar(cs, ax=AX, fraction=0.02, pad=0.02)
#             cbar.ax.tick_params(labelsize=16)
#             break

#     py.tight_layout()
#     # 6) Shared cosmetics
#     for ax in AX:
#         ax.tick_params(axis='both', which='major', labelsize=20, direction='in')
#         ax.set_ylabel(r'$Q^2$', size=30)
#         ax.set_xlabel(r'$x$', size=30)
#         ax.set_xticks(np.log([1e-4,1e-3,1e-2,1e-1]))
#         ax.set_xticklabels([r'$0.0001$',r'$0.001$',r'$0.01$',r'$0.1$'])
#         ax.set_yticks(np.log([10,100,1000]))
#         ax.set_yticklabels([r'$10$',r'$100$',r'$1000$'])
#     py.tight_layout()
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         py.savefig(os.path.join(save_dir, 'PDF_2D_distribution_mceg.png'), dpi=300)



    # HERE




    # H_reco, xedges, q2edges = np.histogram2d(np.log(evts_reco[:,0]), np.log(evts_reco[:,1]),
    #                                         bins=(logx_edges, logQ2_edges))
    # H_reco*=mceg.total_xsec/np.sum(H_reco)  # scale to total xsec
    # true_density = _theory_grid_masked(idis_true, xedges, q2edges, mceg_true.rs, mceg_true.tar,
    #                                 'xQ2', occupancy_counts=H_reco.astype(int))

    # # Predicted θ̂ events (right)
    # pred_density, _ = _hist2d_density_log(
    #     evts_pred, logx_edges, logQ2_edges,
    #     total_xsec=mceg_pred.total_xsec if 'mceg_pred' in globals() else None
    # )

    # # ---------- Plot: top row pcolor, bottom row contour ----------
    # # Order: True (left), Reco (middle), Pred (right)
    # panels = [
    #     ("True", true_density),
    #     ("Reco", reco_density),
    #     ("Pred", pred_density),
    # ]

    # # Compute shared contour levels (log-spaced) over all three, ignoring zeros
    # all_vals = np.concatenate([p[1].ravel() for p in panels])
    # all_vals = all_vals[all_vals > 0]
    # vmin = np.percentile(all_vals, 5) if all_vals.size else 1e-20
    # vmax = np.percentile(all_vals, 99.5) if all_vals.size else 1.0
    # # levels = np.geomspace(max(vmin, 1e-30), vmax, 12)
    # levels=10**np.linspace( np.log10(np.amin(H_reco[H_reco>0])),np.log10(np.amax(H_reco)),60)

    # fig = plt.figure(figsize=(18, 10))
    # AX = []
    # for col, (title, D) in enumerate(panels, start=1):
    #     # Top: heatmap
    #     ax = plt.subplot(2, 3, col); AX.append(ax)
    #     c = ax.pcolor(xedges, q2edges, D.T, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    #     ax.set_title(title, fontsize=18)
    #     # Bottom: contours
    #     ax2 = plt.subplot(2, 3, 3+col); AX.append(ax2)
    #     cs = ax2.contour(xedges[:-1], q2edges[:-1], D.T, levels=levels, norm=matplotlib.colors.LogNorm())
    #     ax2.clabel(cs, inline=True, fontsize=8)
    #     ax2.set_title(f"{title} (contours)", fontsize=16)

    # # Shared axis cosmetics
    # for ax in AX:
    #     ax.tick_params(axis='both', which='major', labelsize=12, direction='in')
    #     ax.set_xlabel(r'$x$', size=14)
    #     ax.set_ylabel(r'$Q^2$', size=14)
    #     ax.set_xticks(np.log([1e-4, 1e-3, 1e-2, 1e-1]))
    #     ax.set_xticklabels([r'$0.0001$', r'$0.001$', r'$0.01$', r'$0.1$'])
    #     ax.set_yticks(np.log([10, 100, 1000]))
    #     ax.set_yticklabels([r'$10$', r'$100$', r'$1000$'])

    # # Colorbar for the heatmaps (top row)
    # cbar_ax = fig.add_axes([0.92, 0.56, 0.015, 0.32])
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    #                                         cmap=plt.get_cmap()),
    #             cax=cbar_ax, label=r'd$\sigma$/d$x$d$Q^2$')
    # plt.tight_layout(rect=[0,0,0.9,1])
    # plt.show()
    # plt.savefig(save_path, dpi=300)

from typing import Optional, Tuple, List, Callable, Dict, Union

# ---------------------------
# CNF helper: sample θ | latent
# ---------------------------
@torch.no_grad()
def cnf_sample_theta(
    model,
    cond_latent: torch.Tensor,     # shape [1, L] or [B, L]
    n_samples: int,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    """
    Adapter for your CNF. Expects model.sample(n, cond=latent) -> [n, D] OR [B, n, D].
    Handles both [1,L] and [B,L] latents; returns [n,D] if B==1 else [B,n,D].
    """
    cond_latent = cond_latent.to(device)
    if cond_latent.dim() == 1:
        cond_latent = cond_latent.unsqueeze(0)  # [1, L]
    B = cond_latent.shape[0]

    thetas = []
    remaining = n_samples
    while remaining > 0:
        m = min(batch_size, remaining)
        # ---- EDIT HERE if your sampler uses a different API ----
        # Common patterns:
        #   samples = model.sample(m, cond=cond_latent)            # -> [B, m, D]
        #   samples = model.sample(m, condition=cond_latent)
        #   samples = model.generate(m, context=cond_latent)
        samples = model.sample(m, cond=cond_latent)  # <-- align to your CNF
        # --------------------------------------------------------
        if samples.dim() == 2:          # [m, D] (implies B==1)
            samples = samples.unsqueeze(0)  # [1, m, D]
        thetas.append(samples)           # [B, m, D]
        remaining -= m

    thetas = torch.cat(thetas, dim=1)    # [B, n, D]
    return thetas.squeeze(0) if B == 1 else thetas


# ---------------------------
# Latent extraction from events
# ---------------------------
@torch.no_grad()
def make_latent_from_true_params(
    simulator,
    pointnet_model,
    true_params: torch.Tensor,
    num_events: int,
    device: torch.device,
    feature_fn: Callable,  # e.g., advanced_feature_engineering
) -> torch.Tensor:
    """
    Simulate events at θ*, featurize, embed with PointNet -> latent [1, L].
    """
    xs = simulator.sample(true_params.detach().cpu(), num_events)
    xs = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_feat = feature_fn(xs)                 # your advanced_feature_engineering
    pointnet_model.eval()
    latent = pointnet_model(xs_feat.unsqueeze(0))  # [1, L]
    return latent


# ---------------------------
# Utility: compute bands over f(x | θ) by sampling θ
# ---------------------------
@torch.no_grad()
def function_bands_over_theta_samples(
    eval_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    # eval_fn(x_grid, theta) -> [|x|]  ; theta shape [D]
    theta_samples: torch.Tensor,  # [S, D]
    x_grid: torch.Tensor,         # [X]
    q_low: float = 0.25,
    q_high: float = 0.75,
    q_mid: float = 0.50,
    device: Optional[torch.device] = None,
    chunk: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (low, median, high) each of shape [X].
    eval_fn must set the simulator to θ and return f(x|θ) at x_grid.
    """
    device = device or x_grid.device
    S = theta_samples.shape[0]
    outs = []

    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        thetas = theta_samples[s0:s1].to(device)
        vals = []
        for t in thetas:
            vals.append(eval_fn(x_grid, t).unsqueeze(0))  # [1, X]
        outs.append(torch.cat(vals, dim=0))  # [s1-s0, X]
    stack = torch.cat(outs, dim=0)  # [S, X]

    low = torch.quantile(stack, q_low, dim=0)
    mid = torch.quantile(stack, q_mid, dim=0)
    high = torch.quantile(stack, q_high, dim=0)
    return low, mid, high


# ---------------------------
# Chi-squared helper
# ---------------------------
def compute_chisq_statistic(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    Unweighted (per-point) chi-squared-like discrepancy.
    Customize with experimental variances/weights if you have them.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(((y_pred - y_true) / denom)**2))


# =========================================================
# 1) Single-instance bands: simplified_dis (u,d) and realistic_dis (q at Q2 slices)
# =========================================================
@torch.no_grad()
def plot_PDF_distribution_single_CNF(
    model,                    # CNF
    pointnet_model,           # feature encoder
    true_params: torch.Tensor,
    device: torch.device,
    feature_fn: Callable,     # advanced_feature_engineering
    simulator,                # SimplifiedDIS(...) or RealisticDIS(...)
    n_theta: int = 512,       # # θ samples from CNF
    n_events_for_latent: int = 100_000,
    problem: str = 'simplified_dis',
    x_range: Tuple[float, float] = (1e-3, 1.0),
    nx: int = 500,
    Q2_slices: Optional[List[float]] = None,
    save_dir: Optional[str] = None,
):
    """
    Build posterior bands for u(x|θ), d(x|θ) (simplified) or q(x,Q^2|θ) (realistic).
    """
    model.eval()
    pointnet_model.eval()

    # Make latent from θ* data
    latent = make_latent_from_true_params(
        simulator, pointnet_model, true_params.to(device),
        num_events=n_events_for_latent, device=device, feature_fn=feature_fn
    )  # [1, L]

    # Sample θ ~ CNF(·|latent)
    theta_samples = cnf_sample_theta_SimpleCNF(
        cnf=model,                    # your SimpleCNF (or DDP-wrapped)
        cond_latent=latent,           # [1,L] or [B,L] from PointNet
        n_samples=100,            # e.g., 512
        device=device,
        batch_size=2048,              # tune for your GPU
    )

    # x-grid
    x_lo, x_hi = x_range
    x_vals = torch.logspace(np.log10(x_lo), np.log10(x_hi), nx, device=device) if x_lo > 0 else torch.linspace(x_lo, x_hi, nx, device=device)

    if problem == 'simplified_dis':
        # Small closures to evaluate u/d at given θ
        def eval_up(x, theta):
            simulator.init(theta)             # set θ
            return simulator.up(x)            # [X]

        def eval_down(x, theta):
            simulator.init(theta)
            return simulator.down(x)

        # Bands
        up_lo, up_mid, up_hi   = function_bands_over_theta_samples(eval_up,   theta_samples, x_vals, device=device)
        dn_lo, dn_mid, dn_hi   = function_bands_over_theta_samples(eval_down, theta_samples, x_vals, device=device)

        # Truth
        simulator.init(true_params.to(device).squeeze())
        up_true = simulator.up(x_vals)
        dn_true = simulator.down(x_vals)

        for (mid, lo, hi, truth, name, color) in [
            (up_mid, up_lo, up_hi, up_true, "up",  "royalblue"),
            (dn_mid, dn_lo, dn_hi, dn_true, "down","darkorange"),
        ]:
            fig, ax = plt.subplots(figsize=(7,5))
            ax.plot(x_vals.detach().cpu(), truth.detach().cpu(), label=fr"True ${name}(x\mid\theta^*)$", linewidth=2)
            ax.plot(x_vals.detach().cpu(), mid.detach().cpu(),   linestyle='--', label=fr"Median $\hat{{{name}}}(x)$", linewidth=2)
            ax.fill_between(x_vals.detach().cpu(), lo.detach().cpu(), hi.detach().cpu(), alpha=0.30, label="IQR")
            ax.set_xscale("log")
            ax.set_xlim(x_range)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(fr"${name}(x\mid\theta)$")
            ax.grid(True, which='both', linestyle=':', linewidth=0.5)
            ax.legend(frameon=False)
            plt.tight_layout()
            out = f"{save_dir}/{name}.png" if save_dir else f"{name}.png"
            plt.savefig(out, dpi=200)
            plt.close(fig)

    elif problem == 'realistic_dis':
        Q2_slices = Q2_slices or [2.0, 10.0, 50.0, 200.0]

        def eval_q(x, theta, Q2_val: float):
            simulator.init(theta)
            Q2v = torch.full_like(x, float(Q2_val))
            return simulator.q(x, Q2v)        # [X]

        for Q2_fixed in Q2_slices:
            def eval_q_fixed(x, theta):
                return eval_q(x, theta, Q2_fixed)

            lo, mid, hi = function_bands_over_theta_samples(eval_q_fixed, theta_samples, x_vals, device=device)

            simulator.init(true_params.to(device).squeeze())
            true_q = eval_q(x_vals, true_params.to(device).squeeze(), Q2_fixed)

            fig, ax = plt.subplots(figsize=(7,5))
            ax.plot(x_vals.detach().cpu(), true_q.detach().cpu(), linewidth=2.5,
                    label=fr"True $q(x, Q^2={Q2_fixed})$")
            ax.plot(x_vals.detach().cpu(), mid.detach().cpu(), linestyle='--', linewidth=2,
                    label=fr"Median $\hat{{q}}(x, Q^2={Q2_fixed})$")
            ax.fill_between(x_vals.detach().cpu(), lo.detach().cpu(), hi.detach().cpu(), alpha=0.25, label="IQR")

            ax.set_xscale("log")
            ax.set_xlim(x_range)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$q(x,Q^2)$")
            ax.set_title(fr"$q(x)$ at $Q^2={Q2_fixed}\,\mathrm{{GeV}}^2$")
            ax.grid(True, which="both", linestyle=":", linewidth=0.5)
            ax.legend(frameon=False)
            plt.tight_layout()
            out = f"{save_dir}/q_Q2_{int(Q2_fixed)}.png" if save_dir else f"q_Q2_{int(Q2_fixed)}.png"
            plt.savefig(out, dpi=200)
            plt.close(fig)

    else:
        raise ValueError("problem must be 'simplified_dis' or 'realistic_dis'")


# =========================================================
# 2) Multi-instance evaluation with χ², using CNF sampling for θ
# =========================================================
@torch.no_grad()
def evaluate_over_n_parameters_CNF(
    model, pointnet_model,
    n: int = 100,
    num_events: int = 100_000,
    device: Optional[torch.device] = None,
    problem: str = 'simplified_dis',
    feature_fn: Callable = None,       # advanced_feature_engineering
    simulator = None,
    n_theta_per_case: int = 512,
    save_dir=None
):
    """
    Like your previous evaluator, but:
      - samples θ from the CNF posterior (conditioned on latent from true events),
      - builds predictive curves by averaging f(x|θ_s) over θ_s,
      - computes |θ_pred - θ_true| / |θ_true| via posterior mean θ (optional).
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert feature_fn is not None, "Pass your advanced_feature_engineering as feature_fn."

    if problem == 'simplified_dis':
        param_dim = 4
        x_grid = torch.logspace(-3, np.log10(0.999), 1000, device=device)
    elif problem == 'realistic_dis':
        param_dim = 6 if "realistic" in problem else 4  # adjust as needed
        x_grid = torch.logspace(-3, np.log10(0.9), 800, device=device)
    else:
        raise ValueError("problem must be 'simplified_dis' or 'realistic_dis'")

    all_errors = []
    chi2_up = []
    chi2_down = []

    for _ in range(n):
        true_params = torch.empty(param_dim).uniform_(0.0, 5.0).to(device)

        # latent from θ* data
        latent = make_latent_from_true_params(
            simulator, pointnet_model, true_params, num_events=num_events,
            device=device, feature_fn=feature_fn
        )  # [1, L]

        # θ samples from CNF
        theta_samples = cnf_sample_theta_SimpleCNF(
        cnf=model,                    # your SimpleCNF (or DDP-wrapped)
        cond_latent=latent,           # [1,L] or [B,L] from PointNet
        n_samples=100,            # e.g., 512
        device=device,
        batch_size=2048,              # tune for your GPU
        )

        # (Optional) parameter error vs posterior mean θ
        theta_mean = theta_samples.mean(dim=0)
        rel_err = torch.abs(theta_mean - true_params) / (true_params.abs() + 1e-8)
        all_errors.append(rel_err.detach().cpu())

        if problem == 'simplified_dis':
            # Predictive mean curves for up/down by averaging over θ samples
            vals_up = []
            vals_dn = []
            for t in theta_samples:
                simulator.init(t)
                vals_up.append(simulator.up(x_grid).unsqueeze(0))
                simulator.init(t)
                vals_dn.append(simulator.down(x_grid).unsqueeze(0))
            pred_up = torch.cat(vals_up, dim=0).mean(dim=0)  # [X]
            pred_dn = torch.cat(vals_dn, dim=0).mean(dim=0)  # [X]

            simulator.init(true_params)
            true_up = simulator.up(x_grid)
            true_dn = simulator.down(x_grid)

            chi2_up.append(compute_chisq_statistic(true_up.cpu().numpy(), pred_up.cpu().numpy()))
            chi2_down.append(compute_chisq_statistic(true_dn.cpu().numpy(), pred_dn.cpu().numpy()))

        elif problem == 'realistic_dis':
            # If you want χ² for q, pick a Q2 grid or average across slices
            Q2_slices = [2.0, 10.0, 50.0, 200.0]
            chis = []
            for Q2_fixed in Q2_slices:
                vals = []
                Q2v = torch.full_like(x_grid, float(Q2_fixed))
                for t in theta_samples:
                    simulator.init(t)
                    vals.append(simulator.q(x_grid, Q2v).unsqueeze(0))
                pred_q = torch.cat(vals, dim=0).mean(dim=0)

                simulator.init(true_params)
                true_q = simulator.q(x_grid, Q2v)
                chis.append(compute_chisq_statistic(true_q.cpu().numpy(), pred_q.cpu().numpy()))
            # store mean across slices for convenience
            chi2_up.append(float(np.mean(chis)))   # reuse arrays for convenience
            chi2_down.append(float(np.std(chis)))  # e.g., store std separately
        else:
            raise ValueError

    all_errors = torch.stack(all_errors).numpy()
    chi2_up = np.array(chi2_up)
    chi2_down = np.array(chi2_down)

    # --- Plots for diagnostics ---
    fig, axes = plt.subplots(1, param_dim, figsize=(4*param_dim, 4))
    if param_dim == 1:
        axes = [axes]
    for i in range(param_dim):
        axes[i].hist(all_errors[:, i], bins=50, alpha=0.8)
        axes[i].set_title(f'Parameter {i+1} Relative Error')
        axes[i].set_xlabel('Relative Error')
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_dir + "/error_distributions_CNF.png", dpi=200)
    plt.close(fig)

    if problem == 'simplified_dis':
        print(f"Median Chi² up:   {np.median(chi2_up):.4f} ± {chi2_up.std():.4f}")
        print(f"Median Chi² down: {np.median(chi2_down):.4f} ± {chi2_down.std():.4f}")

        chi2_up_clip = np.percentile(chi2_up, 99)
        chi2_down_clip = np.percentile(chi2_down, 99)

        plt.figure(figsize=(10,5))
        plt.hist(chi2_up[chi2_up < chi2_up_clip], bins=50, alpha=0.6, label='Chi² Up')
        plt.hist(chi2_down[chi2_down < chi2_down_clip], bins=50, alpha=0.6, label='Chi² Down')
        plt.legend()
        plt.title("Chi-Square Statistic Distribution (Clipped at 99th percentile)")
        plt.xlabel("Chi-Square")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(save_dir + "/chisq_distributions_CNF_clipped.png", dpi=200)
        plt.close()
    else:
        print("Stored mean/std χ² over Q² slices in chi2_up/chi2_down arrays (rename if desired).")

@torch.no_grad()
def cnf_sample_theta_SimpleCNF(
    cnf,                    # instance of SimpleCNF or DDP-wrapped
    cond_latent: torch.Tensor,   # [L], [1,L], or [B,L]
    n_samples: int,
    device: torch.device,
    batch_size: int = 2048,      # split sampling if n is large
) -> torch.Tensor:
    """
    Samples theta ~ p_cnf(theta | context) using your SimpleCNF's base and inverse.
    Returns:
      - [n_samples, D] if B == 1
      - [B, n_samples, D] if B > 1
    """
    # unwrap DDP if needed
    cnf_module = cnf.module if hasattr(cnf, "module") else cnf
    cnf_module.eval()

    cond_latent = cond_latent.to(device)
    if cond_latent.dim() == 1:
        cond_latent = cond_latent.unsqueeze(0)   # [1, L]
    B, L = cond_latent.shape

    theta_dim = cnf_module.theta_dim
    base_mean   = cnf_module.base_mean.to(device)      # [D]
    base_logstd = cnf_module.base_logstd.to(device)    # [D]
    base_std    = base_logstd.exp()                    # [D]

    out_list = []

    # We’ll generate in chunks (m per chunk) to control memory.
    remaining = n_samples
    while remaining > 0:
        m = min(batch_size, remaining)

        # Base samples: for each context, draw m z's.
        # Shape we want before inverse:
        #   if B == 1: [m, D] with context broadcasted to [m, L]
        #   if B > 1:  [B*m, D] with context repeated to [B*m, L]
        if B == 1:
            z = base_mean + base_std * torch.randn(m, theta_dim, device=device)  # [m, D]
            ctx = cond_latent.repeat(m, 1)                                       # [m, L]
            theta_chunk, _ = cnf_module.inverse(z, ctx)                          # [m, D]
            out_list.append(theta_chunk.unsqueeze(0))                            # [1, m, D]
        else:
            z = base_mean + base_std * torch.randn(B * m, theta_dim, device=device)  # [B*m, D]
            ctx = cond_latent.repeat_interleave(m, dim=0)                            # [B*m, L]
            theta_chunk, _ = cnf_module.inverse(z, ctx)                               # [B*m, D]
            theta_chunk = theta_chunk.view(B, m, theta_dim)                           # [B, m, D]
            out_list.append(theta_chunk)

        remaining -= m

    # Concatenate along the sample dimension
    if B == 1:
        # out_list: [ [1,m1,D], [1,m2,D], ... ] -> [1, n, D] -> squeeze batch
        theta = torch.cat(out_list, dim=1).squeeze(0)   # [n_samples, D]
    else:
        # out_list: [ [B,m1,D], [B,m2,D], ... ] -> [B, n, D]
        theta = torch.cat(out_list, dim=1)              # [B, n_samples, D]

    return theta


def plot_bootstrap_PDF_distribution(
    model,
    pointnet_model,
    true_params,
    device,
    num_events,
    n_bootstrap,
    problem='simplified_dis',
    save_dir=None,
    Q2_slices=None
):
    """
    Bootstrap uncertainty visualization with function-level uncertainty focus.
    
    **REFACTORED & UPDATED**: This function emphasizes uncertainty over the predicted PDF 
    functions f(x) at each x-point, complementing the main combined uncertainty 
    analysis but focusing specifically on data uncertainty via bootstrap resampling.
    
    ⚠️  **REPRODUCIBILITY WARNING**: This function requires simulation for uncertainty 
    estimation (by design for bootstrap analysis). Each bootstrap sample generates 
    new simulated events to capture data uncertainty. Results may vary between runs 
    unless random seeds are fixed. For reproducible latent extraction in other 
    contexts, prefer functions that use precomputed data.
    
    For each bootstrap sample:
    - Generates independent event sets from true parameters via simulation
    - Applies appropriate feature engineering based on problem type  
    - Extracts latent representations and predicts parameters  
    - Evaluates PDF functions f(x|θ) using predicted parameters
    - Aggregates function values pointwise to compute uncertainty at each x
    
    The uncertainty bands show variability in the predicted PDF functions due to
    finite event samples (data uncertainty), providing interpretable confidence
    intervals on the PDF predictions themselves.
    
    Args:
        model: Trained model head for parameter prediction
        pointnet_model: Trained PointNet model for latent extraction
        true_params: Fixed true parameter values [tensor of shape (param_dim,)]
        device: Device to run computations on
        num_events: Number of events per bootstrap sample
        n_bootstrap: Number of bootstrap samples to generate
        problem: Problem type ('simplified_dis', 'realistic_dis', 'mceg')
        save_dir: Directory to save plots (required)
        Q2_slices: List of Q2 values for realistic_dis problem
        
    Returns:
        None (saves plots to save_dir)
        
    Saves:
        - bootstrap_pdf_median_up.png: u(x) with function-level uncertainty bands
        - bootstrap_pdf_median_down.png: d(x) with function-level uncertainty bands  
        - bootstrap_pdf_Q2_{value}.png: q(x) at fixed Q2 with function uncertainty
        - bootstrap_param_histograms.png: Parameter distribution histograms (diagnostic)
        
    Example Usage:
        # For simplified DIS problem with function-level uncertainty
        plot_bootstrap_PDF_distribution(
            model=model,
            pointnet_model=pointnet_model, 
            true_params=torch.tensor([2.0, 1.2, 2.0, 1.2]),
            device=device,
            num_events=100000,
            n_bootstrap=50,
            problem='simplified_dis',
            save_dir='./plots/bootstrap'
        )
        
        # For realistic DIS with custom Q2 slices
        plot_bootstrap_PDF_distribution(
            model=model,
            pointnet_model=pointnet_model,
            true_params=torch.tensor([1.0, 0.1, 0.7, 3.0, 0.0, 0.0]),
            device=device, 
            num_events=50000,
            n_bootstrap=30,
            problem='realistic_dis',
            save_dir='./plots/bootstrap',
            Q2_slices=[2.0, 10.0, 50.0]
        )
    """
    if save_dir is None:
        raise ValueError("save_dir must be specified for saving bootstrap plots")
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🎯 Starting bootstrap PDF analysis with {n_bootstrap} samples...")
    print(f"⚠️  Note: Bootstrap analysis requires simulation for uncertainty estimation")
    print(f"📊 Each bootstrap sample generates {num_events} new events for data uncertainty")
    
    # Initialize simulator based on problem type
    SimplifiedDIS, RealisticDIS, MCEGSimulator = get_simulator_module()
    
    if problem == 'realistic_dis':
        if RealisticDIS is None:
            raise ImportError("RealisticDIS not available - please install required dependencies") 
        simulator = RealisticDIS(device=torch.device('cpu'))
        param_names = [r'$\log A_0$', r'$\delta$', r'$a$', r'$b$', r'$c$', r'$d$']
    elif problem == 'simplified_dis':
        if SimplifiedDIS is None:
            raise ImportError("SimplifiedDIS not available - please install required dependencies")
        simulator = SimplifiedDIS(device=torch.device('cpu'))
        param_names = [r'$a_u$', r'$b_u$', r'$a_d$', r'$b_d$']
    elif problem in ['mceg', 'mceg4dis']:
        if MCEGSimulator is None:
            raise ImportError("MCEGSimulator not available - please install required dependencies")
        simulator = MCEGSimulator(device=torch.device('cpu'))
        param_names = [f'Param {i+1}' for i in range(len(true_params))]
    else:
        raise ValueError(f"Unknown problem type: {problem}. Supported: 'simplified_dis', 'realistic_dis', 'mceg', 'mceg4dis'")
    
    model.eval()
    pointnet_model.eval()
    true_params = true_params.to(device)
    
    # Storage for bootstrap results
    bootstrap_params = []
    bootstrap_pdfs = {}  # Will store PDFs for each function/Q2 slice
    
    print("🔄 Generating bootstrap samples via simulation...")
    print(f"📈 This generates fresh simulated data for each of {n_bootstrap} bootstrap samples")
    for i in range(n_bootstrap):
        if (i + 1) % 10 == 0:
            print(f"  ✓ Bootstrap sample {i+1}/{n_bootstrap}")
        
        # Generate independent event set via simulation (required for bootstrap uncertainty)
        with torch.no_grad():
            # Generate events directly from simulator
            xs = simulator.sample(true_params.detach().cpu(), num_events)
            xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
            
            # Apply feature engineering based on problem type
            advanced_feature_engineering = get_advanced_feature_engineering()
            if problem not in ['mceg', 'mceg4dis']:
                xs_tensor = advanced_feature_engineering(xs_tensor)
            else:
                # For mceg/mceg4dis, apply log feature engineering as used in training
                from utils import log_feature_engineering
                xs_flat = xs_tensor.view(-1, xs_tensor.shape[-1])
                xs_tensor = log_feature_engineering(xs_flat)
            
            # Extract latent embedding using PointNet
            latent = pointnet_model(xs_tensor.unsqueeze(0))
            
            # Predict parameters from latent
            predicted_params = model(latent).cpu().squeeze(0)  # [param_dim]
            bootstrap_params.append(predicted_params)
            
            # Compute PDFs for this parameter set
            simulator.init(predicted_params.detach().cpu())
            
            if problem == 'simplified_dis':
                # Compute up and down PDFs
                x_vals = torch.linspace(1e-3, 1, 500)
                
                for fn_name in ['up', 'down']:
                    fn = getattr(simulator, fn_name)
                    pdf_vals = fn(x_vals)
                    
                    if fn_name not in bootstrap_pdfs:
                        bootstrap_pdfs[fn_name] = []
                    bootstrap_pdfs[fn_name].append(pdf_vals.detach().cpu())
                    
            elif problem == 'realistic_dis':
                # Compute PDFs at different Q2 slices
                Q2_slices = Q2_slices or [2.0, 10.0, 50.0, 200.0]
                x_vals = torch.linspace(1e-3, 0.9, 500)
                
                for Q2_fixed in Q2_slices:
                    Q2_vals = torch.full_like(x_vals, Q2_fixed)
                    q_vals = simulator.q(x_vals, Q2_vals)
                    
                    q_key = f'q_Q2_{Q2_fixed}'
                    if q_key not in bootstrap_pdfs:
                        bootstrap_pdfs[q_key] = []
                    bootstrap_pdfs[q_key].append(q_vals.detach().cpu())
    
    # Convert to tensors for easier manipulation
    bootstrap_params = torch.stack(bootstrap_params)  # [n_bootstrap, param_dim]
    
    for key in bootstrap_pdfs:
        bootstrap_pdfs[key] = torch.stack(bootstrap_pdfs[key])  # [n_bootstrap, n_points]
    
    print("Computing statistics and creating plots...")
    
    # Plot parameter histograms
    n_params = bootstrap_params.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    if n_params == 1:
        axes = [axes]
    
    for i in range(n_params):
        predicted_vals = bootstrap_params[:, i].numpy()
        
        # Plot histogram of predicted parameters
        axes[i].hist(predicted_vals, bins=20, alpha=0.6, density=True, 
                    color='skyblue', label=f'Bootstrap Predictions')
        
        # Add true value line
        true_val = true_params[i].item()
        axes[i].axvline(true_val, color='red', linestyle='--', linewidth=2, 
                       label='True Value')
        
        # Add statistics
        mean_pred = np.mean(predicted_vals)
        std_pred = np.std(predicted_vals)
        axes[i].axvline(mean_pred, color='green', linestyle=':', linewidth=1.5,
                       label=f'Mean: {mean_pred:.3f}')
        
        axes[i].set_title(f'{param_names[i]}\nBias: {mean_pred - true_val:.3f}, Std: {std_pred:.3f}')
        axes[i].set_xlabel('Parameter Value')
        axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bootstrap_param_histograms.png"), dpi=300)
    plt.close(fig)
    
    # Plot PDF distributions with uncertainty
    if problem == 'simplified_dis':
        x_vals = torch.linspace(1e-3, 1, 500)
        
        for fn_name, fn_label, color in [("up", "u", "royalblue"), ("down", "d", "darkorange")]:
            if fn_name in bootstrap_pdfs:
                pdf_stack = bootstrap_pdfs[fn_name]  # [n_bootstrap, n_points]
                
                # Compute statistics
                median_vals = torch.median(pdf_stack, dim=0).values
                std_vals = torch.std(pdf_stack, dim=0)
                lower_bounds = median_vals - std_vals
                upper_bounds = median_vals + std_vals
                
                # Compute true PDF
                simulator.init(true_params.squeeze().cpu())
                true_vals = getattr(simulator, fn_name)(x_vals)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot true PDF
                ax.plot(x_vals.numpy(), true_vals.numpy(), 
                       label=fr"True ${fn_label}(x|\theta^*)$", 
                       color=color, linewidth=2.5)
                
                # Plot bootstrap median and uncertainty
                ax.plot(x_vals.numpy(), median_vals.numpy(),
                       linestyle='--', label=fr"Bootstrap Median ${fn_label}(x)$",
                       color="crimson", linewidth=2)
                
                ax.fill_between(x_vals.numpy(), lower_bounds.numpy(), upper_bounds.numpy(),
                               color="crimson", alpha=0.3, 
                               label=fr"±1STD Function Uncertainty (Bootstrap)")
                
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(fr"${fn_label}(x|\theta)$")
                ax.set_xlim(1e-3, 1)
                ax.set_xscale("log")
                ax.grid(True, which='both', linestyle=':', linewidth=0.5)
                ax.legend(frameon=False)
                ax.set_title(f"Function-Level Bootstrap Uncertainty: {fn_name.title()} PDF\n({n_bootstrap} bootstrap samples)")
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"bootstrap_pdf_median_{fn_name}.png"), dpi=300)
                plt.close(fig)
                
    elif problem == 'realistic_dis':
        Q2_slices = Q2_slices or [2.0, 10.0, 50.0, 200.0]
        x_vals = torch.linspace(1e-3, 0.9, 500)
        color_palette = plt.cm.viridis_r(np.linspace(0, 1, len(Q2_slices)))
        
        for i, Q2_fixed in enumerate(Q2_slices):
            q_key = f'q_Q2_{Q2_fixed}'
            if q_key in bootstrap_pdfs:
                pdf_stack = bootstrap_pdfs[q_key]  # [n_bootstrap, n_points]
                
                # Compute statistics
                median_vals = torch.median(pdf_stack, dim=0).values
                std_vals = torch.std(pdf_stack, dim=0)
                lower_bounds = median_vals - std_vals
                upper_bounds = median_vals + std_vals
                
                # Compute true PDF
                simulator.init(true_params.squeeze().cpu())
                Q2_vals = torch.full_like(x_vals, Q2_fixed)
                true_vals = simulator.q(x_vals, Q2_vals)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot true PDF
                ax.plot(x_vals.numpy(), true_vals.numpy(),
                       color=color_palette[i], linewidth=2.5,
                       label=fr"True $q(x,\ Q^2={Q2_fixed})$")
                
                # Plot bootstrap median and uncertainty
                ax.plot(x_vals.numpy(), median_vals.numpy(),
                       linestyle='--', label=fr"Bootstrap Median $q(x)$",
                       color="crimson", linewidth=2)
                
                ax.fill_between(x_vals.numpy(), lower_bounds.numpy(), upper_bounds.numpy(),
                               color="crimson", alpha=0.3,
                               label=fr"±1STD Function Uncertainty (Bootstrap)")
                
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(fr"$q(x, Q^2={Q2_fixed})$")
                ax.set_xlim(1e-3, 0.9)
                ax.set_xscale("log")
                ax.grid(True, which='both', linestyle=':', linewidth=0.5)
                ax.legend(frameon=False)
                ax.set_title(f"Function-Level Bootstrap Uncertainty: $Q^2={Q2_fixed}$ GeV²\n({n_bootstrap} bootstrap samples)")
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"bootstrap_pdf_Q2_{Q2_fixed}.png"), dpi=300)
                plt.close(fig)
    
    print(f"✅ Bootstrap analysis complete! Results saved to {save_dir}")
    print(f"   - Generated {n_bootstrap} bootstrap samples")
    print(f"   - Parameter histograms: bootstrap_param_histograms.png")
    if problem == 'simplified_dis':
        print(f"   - PDF plots: bootstrap_pdf_median_up.png, bootstrap_pdf_median_down.png")
    elif problem == 'realistic_dis':
        print(f"   - PDF plots: bootstrap_pdf_Q2_{{value}}.png for each Q² slice")


def plot_combined_uncertainty_PDF_distribution(
    model,
    pointnet_model,
    true_params,
    device,
    num_events,
    n_bootstrap,
    laplace_model=None,
    problem='simplified_dis',
    save_dir=None,
    Q2_slices=None
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
        problem: Problem type ('simplified_dis', 'realistic_dis', 'mceg')
        save_dir: Directory to save plots (required)
        Q2_slices: List of Q2 values for realistic_dis problem (optional)
        
    Returns:
        None (saves plots and analysis to save_dir)
        
    Saves:
        For simplified_dis:
            - function_uncertainty_pdf_up.png: u(x) with function-level uncertainty bands
            - function_uncertainty_pdf_down.png: d(x) with function-level uncertainty bands
            - function_uncertainty_breakdown_up.txt: Pointwise uncertainty breakdown for u(x)
            - function_uncertainty_breakdown_down.txt: Pointwise uncertainty breakdown for d(x)
            
        For realistic_dis:
            - function_uncertainty_pdf_Q2_{value}.png: q(x) at fixed Q2 with uncertainty
            - function_uncertainty_breakdown_Q2_{value}.txt: Pointwise uncertainty breakdown
            
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
        raise ValueError("save_dir must be specified for saving function uncertainty plots")
    
    # Validate all inputs comprehensively
    validate_combined_uncertainty_inputs(
        model, pointnet_model, true_params, device, num_events, n_bootstrap, problem, save_dir
    )
    
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting FUNCTION-LEVEL uncertainty analysis with {n_bootstrap} bootstrap samples...")
    print("🔄 KEY CHANGE: Computing uncertainty over predicted functions f(x), not parameters θ")
    if laplace_model is not None:
        print("  📊 Using Laplace approximation for model uncertainty in function space")
    else:
        print("  ⚠️  No Laplace model provided - using bootstrap-only function uncertainty")
    
    # Initialize simulator based on problem type
    SimplifiedDIS, RealisticDIS, MCEGSimulator = get_simulator_module()
    
    if problem == 'realistic_dis':
        if RealisticDIS is None:
            raise ImportError("RealisticDIS not available - please install required dependencies")
        simulator = RealisticDIS(device=torch.device('cpu'))
        param_names = [r'$\log A_0$', r'$\delta$', r'$a$', r'$b$', r'$c$', r'$d$']
    elif problem == 'simplified_dis':
        if SimplifiedDIS is None:
            raise ImportError("SimplifiedDIS not available - please install required dependencies")
        simulator = SimplifiedDIS(device=torch.device('cpu'))
        param_names = [r'$a_u$', r'$b_u$', r'$a_d$', r'$b_d$']
    elif problem in ['mceg', 'mceg4dis']:
        if MCEGSimulator is None:
            raise ImportError("MCEGSimulator not available - please install required dependencies")
        simulator = MCEGSimulator(device=torch.device('cpu'))
        param_names = [f'Param {i+1}' for i in range(len(true_params))]
    else:
        raise ValueError(f"Unknown problem type: {problem}")
    
    # Get feature engineering function
    advanced_feature_engineering = get_advanced_feature_engineering()
    
    model.eval()
    pointnet_model.eval()
    true_params = true_params.to(device)
    n_params = len(true_params)
    
    # **NEW APPROACH**: Storage for function-level uncertainty
    # We collect ALL function evaluations f(x|θ) from both bootstrap + Laplace sampling
    function_samples = {}  # Will store all f(x) samples for each function/Q2 slice
    
    # Number of parameter samples per bootstrap iteration (for model uncertainty)
    n_laplace_samples = 20  # Sample multiple θ from each Laplace posterior for each bootstrap
    
    print("Generating bootstrap samples and evaluating functions at each x...")
    for i in tqdm(range(n_bootstrap), desc="Bootstrap samples"):
        # Generate independent event set from true parameters
        with torch.no_grad():
            # Simulate events using true parameters
            xs = simulator.sample(true_params.detach().cpu(), num_events)
            xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
            
            # Apply feature engineering based on problem type
            if problem not in ['mceg', 'mceg4dis']:
                xs_tensor = advanced_feature_engineering(xs_tensor)
            else:
                xs_tensor = log_feature_engineering(xs_tensor)
            
            # Extract latent embedding using PointNet
            latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))
            
            # Get parameter distribution from model + Laplace
            if laplace_model is not None:
                # Use analytic uncertainty (model uncertainty via Laplace)
                mean_params, std_params = get_analytic_uncertainty(
                    model, latent_embedding, laplace_model
                )
                mean_params = mean_params.cpu().squeeze(0)  # [param_dim]
                std_params = std_params.cpu().squeeze(0)    # [param_dim]
                
                # Sample multiple parameter sets from the Laplace posterior for this bootstrap
                param_samples = []
                for _ in range(n_laplace_samples):
                    # Sample θ ~ N(mean_params, diag(std_params²))
                    theta_sample = mean_params + std_params * torch.randn_like(mean_params)
                    param_samples.append(theta_sample)
                param_samples = torch.stack(param_samples)  # [n_laplace_samples, param_dim]
                
            else:
                # Fallback: use deterministic prediction, still create "samples" for consistent processing
                with torch.no_grad():
                    output = model(latent_embedding)
                if isinstance(output, tuple) and len(output) == 2:  # Gaussian head
                    mean_params, logvars = output
                    std_params = torch.exp(0.5 * logvars)
                    mean_params = mean_params.cpu().squeeze(0)
                    std_params = std_params.cpu().squeeze(0)
                    
                    # Sample from model's intrinsic uncertainty
                    param_samples = []
                    for _ in range(n_laplace_samples):
                        theta_sample = mean_params + std_params * torch.randn_like(mean_params)
                        param_samples.append(theta_sample)
                    param_samples = torch.stack(param_samples)
                else:  # Deterministic
                    mean_params = output.cpu().squeeze(0)
                    # Create identical "samples" for consistent processing
                    param_samples = mean_params.unsqueeze(0).repeat(n_laplace_samples, 1)
            
            # **CORE CHANGE**: Evaluate functions f(x|θ) for all θ samples
            if problem == 'simplified_dis':
                # Set up x-grid for evaluation
                x_vals = torch.linspace(1e-3, 1, 500)
                
                for fn_name in ['up', 'down']:
                    if fn_name not in function_samples:
                        function_samples[fn_name] = {'x_vals': x_vals, 'all_samples': []}
                    
                    # Evaluate f(x|θ) for each θ sample from this bootstrap iteration
                    for theta in param_samples:
                        simulator.init(theta.detach().cpu())
                        fn = getattr(simulator, fn_name)
                        pdf_vals = fn(x_vals).detach().cpu()  # [n_x_points]
                        function_samples[fn_name]['all_samples'].append(pdf_vals)
                    
            elif problem == 'realistic_dis':
                # Set up x-grid and Q2 slices for evaluation
                Q2_slices = Q2_slices or [2.0, 10.0, 50.0, 200.0]
                x_vals = torch.linspace(1e-3, 0.9, 500)
                
                for Q2_fixed in Q2_slices:
                    q_key = f'q_Q2_{Q2_fixed}'
                    if q_key not in function_samples:
                        function_samples[q_key] = {'x_vals': x_vals, 'Q2': Q2_fixed, 'all_samples': []}
                    
                    Q2_vals = torch.full_like(x_vals, Q2_fixed)
                    
                    # Evaluate q(x|θ) for each θ sample from this bootstrap iteration
                    for theta in param_samples:
                        simulator.init(theta.detach().cpu())
                        q_vals = simulator.q(x_vals, Q2_vals).detach().cpu()  # [n_x_points]
                        function_samples[q_key]['all_samples'].append(q_vals)
    
    # Convert function samples to tensors for pointwise statistics
    print("Computing pointwise function uncertainty statistics...")
    for key in function_samples:
        # Stack all function evaluations: [n_bootstrap * n_laplace_samples, n_x_points]
        function_samples[key]['all_samples'] = torch.stack(function_samples[key]['all_samples'])
        
        n_total_samples, n_x_points = function_samples[key]['all_samples'].shape
        print(f"  Function {key}: {n_total_samples} total samples across {n_x_points} x-points")
    
    # **NEW APPROACH**: Pointwise uncertainty decomposition
    # For each x-point, we now have n_bootstrap * n_laplace_samples function values
    # We can compute mean, std, and decompose uncertainty sources pointwise
    
    # Save methodology documentation
    methodology_path = os.path.join(save_dir, "function_uncertainty_methodology.txt")
    with open(methodology_path, 'w') as f:
        f.write("Function-Level Uncertainty Quantification Methodology\n")
        f.write("=" * 60 + "\n\n")
        f.write("KEY CHANGE: This analysis computes uncertainty over the predicted FUNCTIONS f(x),\n")
        f.write("not over the model parameters θ. This provides more interpretable uncertainty\n") 
        f.write("for PDF predictions and physics applications.\n\n")
        f.write("Method:\n")
        f.write("1. For each bootstrap iteration:\n")
        f.write(f"   - Generate {num_events} events from true parameters\n")
        f.write("   - Extract latent representation via PointNet\n")
        f.write("   - Predict parameter distribution θ ~ N(mean, STD²) via model + Laplace\n")
        f.write(f"   - Sample {n_laplace_samples} parameter sets from θ ~ N(mean, STD²)\n")
        f.write("   - Evaluate f(x|θ) for each θ sample at each x-point\n\n")
        f.write("2. Aggregate uncertainty pointwise:\n")
        f.write("   - Collect all f(x) values at each x from all bootstrap + Laplace samples\n")
        f.write("   - Compute mean and standard deviation of f(x) at each x\n")
        f.write("   - Uncertainty bands reflect variation in predicted function, not parameters\n\n")
        f.write("3. Uncertainty sources:\n")
        f.write("   - Data uncertainty: variation due to finite event samples (bootstrap)\n")
        f.write("   - Model uncertainty: variation due to parameter posterior (Laplace)\n")
        f.write("   - Combined pointwise: total_variance(x) = var_bootstrap(x) + var_laplace(x)\n\n")
        f.write(f"Configuration:\n")
        f.write(f"Problem: {problem}\n")
        f.write(f"True parameters: {true_params.cpu().numpy()}\n")
        f.write(f"Bootstrap samples: {n_bootstrap}\n")
        f.write(f"Events per sample: {num_events}\n")
        f.write(f"Laplace samples per bootstrap: {n_laplace_samples}\n")
        f.write(f"Total function evaluations: {n_bootstrap * n_laplace_samples}\n")
        f.write(f"Laplace model: {'Available' if laplace_model is not None else 'Not available'}\n\n")
    
    print("Computing function-level uncertainty statistics and creating plots...")
    
    # Create plots for each function with pointwise uncertainty bands
    if problem == 'simplified_dis':
        for fn_name, fn_label, color in [("up", "u", "royalblue"), ("down", "d", "darkorange")]:
            if fn_name in function_samples:
                data = function_samples[fn_name]
                x_vals = data['x_vals']
                all_samples = data['all_samples']  # [n_total_samples, n_x_points]
                
                # **POINTWISE UNCERTAINTY COMPUTATION**
                mean_pdf = all_samples.mean(dim=0)        # Mean f(x) at each x [n_x_points]
                std_pdf = all_samples.std(dim=0)          # Std f(x) at each x [n_x_points] 
                
                # Uncertainty bands (±1 standard deviation)
                lower_bound = mean_pdf - std_pdf
                upper_bound = mean_pdf + std_pdf
                
                # Compute true PDF for comparison
                simulator.init(true_params.squeeze().cpu())
                true_pdf = getattr(simulator, fn_name)(x_vals).detach().cpu()
                
                # Create main plot with function-level uncertainty
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot true PDF
                ax.plot(x_vals.numpy(), true_pdf.numpy(), 
                       label=fr"True ${fn_label}(x|\theta^*)$", 
                       color=color, linewidth=2.5)
                
                # Plot mean prediction
                ax.plot(x_vals.numpy(), mean_pdf.numpy(),
                       linestyle='--', label=fr"Mean Prediction ${fn_label}(x)$",
                       color="crimson", linewidth=2)
                
                # Function-level uncertainty band
                ax.fill_between(x_vals.numpy(), lower_bound.numpy(), upper_bound.numpy(),
                               color="crimson", alpha=0.3, 
                               label=fr"±1STD Function Uncertainty")
                
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(fr"${fn_label}(x|\theta)$")
                ax.set_xlim(1e-3, 1)
                ax.set_xscale("log")
                ax.grid(True, which='both', linestyle=':', linewidth=0.5)
                ax.legend(frameon=False)
                ax.set_title(f"Function-Level Uncertainty: {fn_name.title()} PDF\n"
                           f"({n_bootstrap} bootstrap × {n_laplace_samples} Laplace samples)")
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"function_uncertainty_pdf_{fn_name}.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Save pointwise uncertainty breakdown
                breakdown_path = os.path.join(save_dir, f"function_uncertainty_breakdown_{fn_name}.txt")
                with open(breakdown_path, 'w') as f:
                    f.write(f"Pointwise Function Uncertainty Breakdown: {fn_name}(x)\n")
                    f.write("=" * 50 + "\n\n")
                    f.write("This file contains pointwise uncertainty statistics for the predicted\n")
                    f.write(f"function {fn_name}(x) at each x-point in the evaluation grid.\n\n")
                    f.write("Columns:\n")
                    f.write("x: x-coordinate\n")
                    f.write("true_f(x): true function value\n") 
                    f.write("mean_f(x): mean predicted function value across all samples\n")
                    f.write("std_f(x): standard deviation of predicted function value\n")
                    f.write("bias_f(x): mean_f(x) - true_f(x)\n")
                    f.write("rel_uncertainty: std_f(x) / |mean_f(x)|\n\n")
                    f.write(f"{'x':>12s} {'true_f(x)':>12s} {'mean_f(x)':>12s} {'std_f(x)':>12s} {'bias_f(x)':>12s} {'rel_unc':>12s}\n")
                    f.write("-" * 80 + "\n")
                    
                    for i, x_val in enumerate(x_vals):
                        true_val = true_pdf[i].item()
                        mean_val = mean_pdf[i].item()
                        std_val = std_pdf[i].item()
                        bias_val = mean_val - true_val
                        rel_unc = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else float('inf')
                        
                        f.write(f"{x_val.item():12.6e} {true_val:12.6e} {mean_val:12.6e} "
                               f"{std_val:12.6e} {bias_val:12.6e} {rel_unc:12.6f}\n")
                
                print(f"  ✅ Function uncertainty analysis saved for {fn_name}(x)")
                
    elif problem == 'realistic_dis':
        Q2_slices = Q2_slices or [2.0, 10.0, 50.0, 200.0]
        color_palette = plt.cm.viridis_r(np.linspace(0, 1, len(Q2_slices)))
        
        for i, Q2_fixed in enumerate(Q2_slices):
            q_key = f'q_Q2_{Q2_fixed}'
            if q_key in function_samples:
                data = function_samples[q_key]
                x_vals = data['x_vals'] 
                all_samples = data['all_samples']  # [n_total_samples, n_x_points]
                
                # **POINTWISE UNCERTAINTY COMPUTATION**
                mean_pdf = all_samples.mean(dim=0)        # Mean q(x) at each x [n_x_points]
                std_pdf = all_samples.std(dim=0)          # Std q(x) at each x [n_x_points]
                
                # Uncertainty bands (±1 standard deviation)
                lower_bound = mean_pdf - std_pdf
                upper_bound = mean_pdf + std_pdf
                
                # Compute true PDF for comparison
                simulator.init(true_params.squeeze().cpu())
                Q2_vals = torch.full_like(x_vals, Q2_fixed)
                true_pdf = simulator.q(x_vals, Q2_vals).detach().cpu()
                
                # Create main plot with function-level uncertainty
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot true PDF
                ax.plot(x_vals.numpy(), true_pdf.numpy(),
                       color=color_palette[i], linewidth=2.5,
                       label=fr"True $q(x,\ Q^2={Q2_fixed})$")
                
                # Plot mean prediction
                ax.plot(x_vals.numpy(), mean_pdf.numpy(),
                       linestyle='--', label=fr"Mean Prediction $q(x)$",
                       color="crimson", linewidth=2)
                
                # Function-level uncertainty band
                ax.fill_between(x_vals.numpy(), lower_bound.numpy(), upper_bound.numpy(),
                               color="crimson", alpha=0.3,
                               label=fr"±1STD Function Uncertainty")
                
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(fr"$q(x, Q^2={Q2_fixed})$")
                ax.set_xlim(1e-3, 0.9)
                ax.set_xscale("log")
                ax.grid(True, which='both', linestyle=':', linewidth=0.5)
                ax.legend(frameon=False)
                ax.set_title(f"Function-Level Uncertainty: PDF at $Q^2={Q2_fixed}$ GeV²\n"
                           f"({n_bootstrap} bootstrap × {n_laplace_samples} Laplace samples)")
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"function_uncertainty_pdf_Q2_{Q2_fixed}.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Save pointwise uncertainty breakdown
                breakdown_path = os.path.join(save_dir, f"function_uncertainty_breakdown_Q2_{Q2_fixed}.txt")
                with open(breakdown_path, 'w') as f:
                    f.write(f"Pointwise Function Uncertainty Breakdown: q(x, Q²={Q2_fixed})\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("This file contains pointwise uncertainty statistics for the predicted\n")
                    f.write(f"function q(x, Q²={Q2_fixed}) at each x-point in the evaluation grid.\n\n")
                    f.write("Columns:\n")
                    f.write("x: x-coordinate\n")
                    f.write("true_q(x): true function value\n")
                    f.write("mean_q(x): mean predicted function value across all samples\n")
                    f.write("std_q(x): standard deviation of predicted function value\n")
                    f.write("bias_q(x): mean_q(x) - true_q(x)\n")
                    f.write("rel_uncertainty: std_q(x) / |mean_q(x)|\n\n")
                    f.write(f"{'x':>12s} {'true_q(x)':>12s} {'mean_q(x)':>12s} {'std_q(x)':>12s} {'bias_q(x)':>12s} {'rel_unc':>12s}\n")
                    f.write("-" * 80 + "\n")
                    
                    for j, x_val in enumerate(x_vals):
                        true_val = true_pdf[j].item()
                        mean_val = mean_pdf[j].item()
                        std_val = std_pdf[j].item()
                        bias_val = mean_val - true_val
                        rel_unc = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else float('inf')
                        
                        f.write(f"{x_val.item():12.6e} {true_val:12.6e} {mean_val:12.6e} "
                               f"{std_val:12.6e} {bias_val:12.6e} {rel_unc:12.6f}\n")
                
                print(f"  ✅ Function uncertainty analysis saved for Q²={Q2_fixed}")
    
    # Create summary statistics plot across all x-points
    print("Creating summary uncertainty analysis...")
    
    # Compute average statistics across all functions/x-points
    all_relative_uncertainties = []
    all_absolute_uncertainties = []
    function_names = []
    
    for key in function_samples:
        data = function_samples[key]
        all_samples = data['all_samples']
        mean_vals = all_samples.mean(dim=0)
        std_vals = all_samples.std(dim=0)
        
        # Compute relative uncertainties (avoiding division by zero)
        rel_unc = std_vals / torch.clamp(torch.abs(mean_vals), min=1e-10)
        
        all_relative_uncertainties.append(rel_unc.mean().item())  # Average over x
        all_absolute_uncertainties.append(std_vals.mean().item())  # Average over x
        function_names.append(key)
    
    # Summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Relative uncertainty
    ax1.bar(range(len(function_names)), all_relative_uncertainties, 
            color='lightcoral', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Function')
    ax1.set_ylabel('Average Relative Uncertainty')
    ax1.set_title('Average Relative Function Uncertainty')
    ax1.set_xticks(range(len(function_names)))
    ax1.set_xticklabels(function_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Absolute uncertainty  
    ax2.bar(range(len(function_names)), all_absolute_uncertainties,
            color='lightblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Function')
    ax2.set_ylabel('Average Absolute Uncertainty')
    ax2.set_title('Average Absolute Function Uncertainty')
    ax2.set_xticks(range(len(function_names)))
    ax2.set_xticklabels(function_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "function_uncertainty_summary.png"), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Function-level uncertainty analysis complete! Results saved to {save_dir}")
    print(f"   📊 MAJOR CHANGE: Uncertainty now computed over predicted FUNCTIONS f(x), not parameters θ")
    print(f"   📈 Generated {n_bootstrap * n_laplace_samples} total function evaluations")
    print(f"   📄 Methodology documentation: function_uncertainty_methodology.txt")
    print(f"   📋 Summary statistics: function_uncertainty_summary.png")
    if problem == 'simplified_dis':
        print(f"   📊 Function plots: function_uncertainty_pdf_up.png, function_uncertainty_pdf_down.png")
        print(f"   📝 Pointwise breakdowns: function_uncertainty_breakdown_up.txt, function_uncertainty_breakdown_down.txt")
    elif problem == 'realistic_dis':
        print(f"   📊 Function plots: function_uncertainty_pdf_Q2_{{value}}.png for each Q² slice") 
        print(f"   📝 Pointwise breakdowns: function_uncertainty_breakdown_Q2_{{value}}.txt for each Q² slice")
    
    # Return summary statistics for potential programmatic use
    return {
        'problem': problem,
        'n_bootstrap': n_bootstrap,
        'n_laplace_samples': n_laplace_samples,
        'total_function_evaluations': n_bootstrap * n_laplace_samples,
        'function_names': function_names,
        'average_relative_uncertainties': all_relative_uncertainties,
        'average_absolute_uncertainties': all_absolute_uncertainties,
        'true_params': true_params,
        'methodology': 'function_level_uncertainty'
    }


def plot_uncertainty_decomposition_comparison(
    true_params: torch.Tensor,
    bootstrap_only_results: Dict,
    laplace_only_results: Dict, 
    combined_results: Dict,
    save_dir: str,
    param_names: List[str] = None
):
    """
    Create comparison plots showing different uncertainty quantification methods.
    
    This helper function creates side-by-side comparisons of uncertainty estimates
    from different methods: bootstrap-only, Laplace-only, and combined approaches.
    
    Args:
        true_params: True parameter values [param_dim]
        bootstrap_only_results: Results dict from bootstrap-only analysis
        laplace_only_results: Results dict from Laplace-only analysis  
        combined_results: Results dict from combined uncertainty analysis
        save_dir: Directory to save comparison plots
        param_names: Parameter name labels for plots
        
    Returns:
        None (saves comparison plots to save_dir)
        
    Saves:
        - uncertainty_method_comparison.png: Side-by-side uncertainty comparison
        - uncertainty_correlation_analysis.png: Correlation between methods
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(save_dir, exist_ok=True)
    
    n_params = len(true_params)
    param_names = param_names or [f'Param {i+1}' for i in range(n_params)]
    
    # Extract uncertainties from each method
    methods = ['Bootstrap Only', 'Laplace Only', 'Combined']
    results = [bootstrap_only_results, laplace_only_results, combined_results]
    
    # Method comparison plot
    fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 8))
    if n_params == 1:
        axes = axes.reshape(2, 1)
    
    colors = ['lightblue', 'orange', 'green']
    
    for i in range(n_params):
        # Top row: Uncertainty magnitude comparison
        ax_top = axes[0, i]
        
        uncertainties = []
        for result in results:
            if 'total_uncertainty' in result:
                uncertainties.append(result['total_uncertainty'][i].item())
            elif 'data_uncertainty' in result:
                uncertainties.append(result['data_uncertainty'][i].item())
            else:
                uncertainties.append(0.0)  # fallback
        
        bars = ax_top.bar(methods, uncertainties, color=colors, alpha=0.7, edgecolor='black')
        ax_top.set_title(f'{param_names[i]} Uncertainty Comparison')
        ax_top.set_ylabel('Standard Deviation')
        ax_top.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, uncertainties):
            ax_top.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(uncertainties)*0.01,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Bottom row: Bias comparison
        ax_bottom = axes[1, i]
        
        true_val = true_params[i].item()
        biases = []
        for result in results:
            if 'mean_predictions' in result:
                bias = (result['mean_predictions'][i] - true_val).item()
                biases.append(bias)
            else:
                biases.append(0.0)  # fallback
        
        bars = ax_bottom.bar(methods, biases, color=colors, alpha=0.7, edgecolor='black')
        ax_bottom.axhline(0, color='red', linestyle='--', alpha=0.7)
        ax_bottom.set_title(f'{param_names[i]} Bias Comparison')
        ax_bottom.set_ylabel('Bias (Predicted - True)')
        ax_bottom.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, biases):
            y_pos = bar.get_height() + max(abs(min(biases)), max(biases))*0.01 if val >= 0 else bar.get_height() - max(abs(min(biases)), max(biases))*0.01
            ax_bottom.text(bar.get_x() + bar.get_width()/2, y_pos,
                          f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "uncertainty_method_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Uncertainty method comparison saved to {save_dir}/uncertainty_method_comparison.png")


def validate_combined_uncertainty_inputs(
    model,
    pointnet_model, 
    true_params: torch.Tensor,
    device: torch.device,
    num_events: int,
    n_bootstrap: int,
    problem: str,
    save_dir: str
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
        print(f"⚠️  Warning: n_bootstrap={n_bootstrap} is quite large and may take significant time")
    
    # Validate problem type
    valid_problems = ['simplified_dis', 'realistic_dis', 'mceg']
    if problem not in valid_problems:
        raise ValueError(f"problem must be one of {valid_problems}, got '{problem}'")
    
    # Validate save directory
    if save_dir is None:
        raise ValueError("save_dir cannot be None")
    if not isinstance(save_dir, str):
        raise ValueError("save_dir must be a string")
    
    # Check parameter dimensions match problem expectations
    expected_dims = {
        'simplified_dis': 4,
        'realistic_dis': 6, 
        'mceg': None  # Variable
    }
    
    if problem in expected_dims and expected_dims[problem] is not None:
        if len(true_params) != expected_dims[problem]:
            raise ValueError(f"For problem '{problem}', expected {expected_dims[problem]} parameters, got {len(true_params)}")
    
    return True


def plot_uncertainty_vs_events(
    model,
    pointnet_model, 
    true_params,
    device,
    event_counts=None,
    n_bootstrap=20,
    laplace_model=None,
    problem='simplified_dis',
    save_dir=None,
    Q2_slices=None,
    fixed_x_values=None
):
    """
    Plot uncertainty quantification consistency: how uncertainty bands shrink 
    as the number of events (data) increases.
    
    This function demonstrates the fundamental principle that uncertainty should 
    decrease as more data becomes available. For each event count, it runs the 
    full uncertainty quantification pipeline and shows how both bootstrap 
    (data uncertainty) and Laplace (model uncertainty) behave with varying 
    amounts of training data.
    
    **Key Consistency Check**: Uncertainty should generally decrease as N_events 
    increases, following statistical scaling laws (roughly proportional to 1/√N).
    
    Args:
        model: Trained model head for parameter prediction
        pointnet_model: Trained PointNet model for latent extraction
        true_params: Fixed true parameter values [tensor of shape (param_dim,)]
        device: Device to run computations on
        event_counts: List of event counts to test [default: [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]]
        n_bootstrap: Number of bootstrap samples per event count
        laplace_model: Fitted Laplace approximation for model uncertainty (optional)
        problem: Problem type ('simplified_dis', 'realistic_dis', 'mceg')
        save_dir: Directory to save plots (required)
        Q2_slices: List of Q2 values for realistic_dis problem (optional)
        fixed_x_values: List of x values to track uncertainty (optional)
        
    Returns:
        Dict: Summary statistics and scaling analysis results
        
    Saves:
        - uncertainty_vs_events_scaling.png: Overall uncertainty scaling plot
        - uncertainty_vs_events_by_function.png: Per-function uncertainty scaling  
        - uncertainty_vs_events_fixed_x.png: Uncertainty at fixed x values
        - uncertainty_scaling_analysis.txt: Statistical analysis of scaling behavior
        
    Example Usage:
        # Test uncertainty scaling for simplified DIS
        scaling_results = plot_uncertainty_vs_events(
            model=model,
            pointnet_model=pointnet_model,
            true_params=torch.tensor([2.0, 1.2, 2.0, 1.2]),
            device=device,
            event_counts=[1000, 5000, 10000, 50000, 100000],
            n_bootstrap=30,
            laplace_model=laplace_model,
            problem='simplified_dis',
            save_dir='./plots/scaling_analysis'
        )
        
        # Test with specific x values and Q2 slices for realistic DIS
        scaling_results = plot_uncertainty_vs_events(
            model=model,
            pointnet_model=pointnet_model, 
            true_params=torch.tensor([1.0, 0.1, 0.7, 3.0, 0.0, 0.0]),
            device=device,
            event_counts=[5000, 25000, 100000, 500000],
            n_bootstrap=25,
            problem='realistic_dis',
            save_dir='./plots/scaling_analysis',
            Q2_slices=[2.0, 10.0, 50.0],
            fixed_x_values=[0.01, 0.1, 0.5]
        )
    """
    if save_dir is None:
        raise ValueError("save_dir must be specified for saving uncertainty scaling plots")
    
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Default event counts spanning realistic range
    if event_counts is None:
        event_counts = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    
    print(f"🔄 Testing uncertainty scaling across {len(event_counts)} event counts...")
    print(f"   Event counts: {event_counts}")
    print(f"   Bootstrap samples per count: {n_bootstrap}")
    if laplace_model is not None:
        print("   Using Laplace approximation for model uncertainty")
    else:
        print("   Using bootstrap-only uncertainty (no Laplace)")
    
    # Initialize simulator
    SimplifiedDIS, RealisticDIS, MCEGSimulator = get_simulator_module()
    
    if problem == 'realistic_dis':
        if RealisticDIS is None:
            raise ImportError("RealisticDIS not available - please install required dependencies")
        simulator = RealisticDIS(device=torch.device('cpu'))
        param_names = [r'$\log A_0$', r'$\delta$', r'$a$', r'$b$', r'$c$', r'$d$']
    elif problem == 'simplified_dis':
        if SimplifiedDIS is None:
            raise ImportError("SimplifiedDIS not available - please install required dependencies")
        simulator = SimplifiedDIS(device=torch.device('cpu'))
        param_names = [r'$a_u$', r'$b_u$', r'$a_d$', r'$b_d$']
    elif problem in ['mceg', 'mceg4dis']:
        if MCEGSimulator is None:
            raise ImportError("MCEGSimulator not available - please install required dependencies")
        simulator = MCEGSimulator(device=torch.device('cpu'))
        param_names = [f'Param {i+1}' for i in range(len(true_params))]
    else:
        raise ValueError(f"Unknown problem type: {problem}")
    
    # Get feature engineering function
    advanced_feature_engineering = get_advanced_feature_engineering()
    
    model.eval()
    pointnet_model.eval()
    true_params = true_params.to(device)
    
    # Storage for scaling analysis results
    scaling_results = {
        'event_counts': event_counts,
        'problem': problem,
        'n_bootstrap': n_bootstrap,
        'true_params': true_params.cpu().numpy(),
        'param_names': param_names,
        'function_uncertainties': {},  # {function_name: [uncertainties_per_event_count]}
        'parameter_uncertainties': [],  # [param_uncertainties_per_event_count]
        'fixed_x_uncertainties': {},   # {x_value: {function: [uncertainties_per_event_count]}}
        'laplace_available': laplace_model is not None
    }
    
    # Set up fixed x values for tracking
    if fixed_x_values is None:
        fixed_x_values = [0.01, 0.1, 0.5] if problem == 'simplified_dis' else [0.01, 0.1, 0.5]
    
    for x_val in fixed_x_values:
        scaling_results['fixed_x_uncertainties'][x_val] = {}
    
    print("Running uncertainty analysis for each event count...")
    
    for i, num_events in enumerate(tqdm(event_counts, desc="Event counts")):
        print(f"\n  📊 Event count: {num_events:,}")
        
        # Storage for this event count
        function_uncertainties_this_count = {}
        param_uncertainties_this_count = []
        fixed_x_uncertainties_this_count = {x: {} for x in fixed_x_values}
        
        # Run bootstrap analysis for this event count
        bootstrap_params = []
        bootstrap_pdfs = {}
        
        for j in tqdm(range(n_bootstrap), desc=f"Bootstrap (N={num_events})", leave=False):
            # Generate events with this count
            with torch.no_grad():
                xs = simulator.sample(true_params.detach().cpu(), int(num_events))
                xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
                
                # Apply feature engineering based on problem type
                if problem not in ['mceg', 'mceg4dis']:
                    xs_tensor = advanced_feature_engineering(xs_tensor)
                else:
                    xs_tensor = log_feature_engineering(xs_tensor)
                
                # Extract latent embedding
                latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))
                
                # Get parameter prediction
                if laplace_model is not None:
                    # Use analytic uncertainty (includes model uncertainty)
                    mean_params, std_params = get_analytic_uncertainty(
                        model, latent_embedding, laplace_model
                    )
                    predicted_params = mean_params.cpu().squeeze(0)  # [param_dim]
                    param_std = std_params.cpu().squeeze(0)          # [param_dim]
                else:
                    # Fallback to model output only
                    with torch.no_grad():
                        output = model(latent_embedding)
                    if isinstance(output, tuple) and len(output) == 2:  # Gaussian head
                        mean_params, logvars = output
                        predicted_params = mean_params.cpu().squeeze(0)
                        param_std = torch.exp(0.5 * logvars).cpu().squeeze(0)
                    else:  # Deterministic
                        predicted_params = output.cpu().squeeze(0)
                        param_std = torch.zeros_like(predicted_params)
                
                bootstrap_params.append(predicted_params)
                
                # Compute PDFs for this parameter set
                simulator.init(predicted_params.detach().cpu())
                
                if problem == 'simplified_dis':
                    # Evaluate up and down PDFs
                    x_vals = torch.linspace(1e-3, 1, 500)
                    
                    for fn_name in ['up', 'down']:
                        fn = getattr(simulator, fn_name)
                        pdf_vals = fn(x_vals).detach().cpu()
                        
                        if fn_name not in bootstrap_pdfs:
                            bootstrap_pdfs[fn_name] = []
                        bootstrap_pdfs[fn_name].append(pdf_vals)
                        
                        # Evaluate at fixed x values
                        for x_fixed in fixed_x_values:
                            x_tensor = torch.tensor([x_fixed])
                            pdf_at_x = fn(x_tensor).item()
                            if fn_name not in fixed_x_uncertainties_this_count[x_fixed]:
                                fixed_x_uncertainties_this_count[x_fixed][fn_name] = []
                            fixed_x_uncertainties_this_count[x_fixed][fn_name].append(pdf_at_x)
                            
                elif problem == 'realistic_dis':
                    # Evaluate q PDFs at different Q2 slices
                    Q2_slices = Q2_slices or [2.0, 10.0, 50.0, 200.0]
                    x_vals = torch.linspace(1e-3, 0.9, 500)
                    
                    for Q2_fixed in Q2_slices:
                        Q2_vals = torch.full_like(x_vals, Q2_fixed)
                        q_vals = simulator.q(x_vals, Q2_vals).detach().cpu()
                        
                        q_key = f'q_Q2_{Q2_fixed}'
                        if q_key not in bootstrap_pdfs:
                            bootstrap_pdfs[q_key] = []
                        bootstrap_pdfs[q_key].append(q_vals)
                        
                        # Evaluate at fixed x values
                        for x_fixed in fixed_x_values:
                            x_tensor = torch.tensor([x_fixed])
                            Q2_tensor = torch.tensor([Q2_fixed])
                            q_at_x = simulator.q(x_tensor, Q2_tensor).item()
                            if q_key not in fixed_x_uncertainties_this_count[x_fixed]:
                                fixed_x_uncertainties_this_count[x_fixed][q_key] = []
                            fixed_x_uncertainties_this_count[x_fixed][q_key].append(q_at_x)
        
        # Convert bootstrap results to tensors and compute uncertainties
        bootstrap_params = torch.stack(bootstrap_params)  # [n_bootstrap, param_dim]
        
        # Parameter-level uncertainty
        param_uncertainties_this_count = torch.std(bootstrap_params, dim=0).numpy()
        scaling_results['parameter_uncertainties'].append(param_uncertainties_this_count)
        
        # Function-level uncertainty  
        for key in bootstrap_pdfs:
            pdf_stack = torch.stack(bootstrap_pdfs[key])  # [n_bootstrap, n_points]
            # Average standard deviation across x-points for this function
            avg_std = torch.std(pdf_stack, dim=0).mean().item()
            
            if key not in function_uncertainties_this_count:
                function_uncertainties_this_count[key] = avg_std
            if key not in scaling_results['function_uncertainties']:
                scaling_results['function_uncertainties'][key] = []
            scaling_results['function_uncertainties'][key].append(avg_std)
        
        # Fixed x uncertainty
        for x_val in fixed_x_values:
            for func_key in fixed_x_uncertainties_this_count[x_val]:
                values = fixed_x_uncertainties_this_count[x_val][func_key]
                uncertainty = np.std(values)
                
                if func_key not in scaling_results['fixed_x_uncertainties'][x_val]:
                    scaling_results['fixed_x_uncertainties'][x_val][func_key] = []
                scaling_results['fixed_x_uncertainties'][x_val][func_key].append(uncertainty)
        
        print(f"    Parameter uncertainties: {param_uncertainties_this_count}")
        print(f"    Function uncertainties: {function_uncertainties_this_count}")
    
    print("\n📈 Creating uncertainty scaling plots...")
    
    # Plot 1: Overall uncertainty scaling (log-log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Parameter uncertainty scaling
    param_uncertainties = np.array(scaling_results['parameter_uncertainties'])  # [n_event_counts, n_params]
    
    for i, param_name in enumerate(param_names):
        uncertainties = param_uncertainties[:, i]
        ax1.loglog(event_counts, uncertainties, 'o-', label=param_name, linewidth=2, markersize=6)
    
    # Add theoretical 1/sqrt(N) scaling line
    theoretical_scaling = uncertainties[0] * np.sqrt(event_counts[0] / np.array(event_counts))
    ax1.loglog(event_counts, theoretical_scaling, 'k--', alpha=0.7, linewidth=2, 
               label=r'$\propto 1/\sqrt{N}$ (theoretical)')
    
    ax1.set_xlabel('Number of Events')
    ax1.set_ylabel('Parameter Uncertainty (std)')
    ax1.set_title('Parameter Uncertainty vs. Event Count')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Function uncertainty scaling
    colors = plt.cm.tab10(np.linspace(0, 1, len(scaling_results['function_uncertainties'])))
    for i, (func_name, uncertainties) in enumerate(scaling_results['function_uncertainties'].items()):
        ax2.loglog(event_counts, uncertainties, 'o-', color=colors[i], 
                   label=func_name, linewidth=2, markersize=6)
    
    # Add theoretical scaling line for functions
    if len(scaling_results['function_uncertainties']) > 0:
        first_func_uncertainties = list(scaling_results['function_uncertainties'].values())[0]
        theoretical_func_scaling = first_func_uncertainties[0] * np.sqrt(event_counts[0] / np.array(event_counts))
        ax2.loglog(event_counts, theoretical_func_scaling, 'k--', alpha=0.7, linewidth=2,
                   label=r'$\propto 1/\sqrt{N}$ (theoretical)')
    
    ax2.set_xlabel('Number of Events')
    ax2.set_ylabel('Function Uncertainty (avg std)')
    ax2.set_title('Function Uncertainty vs. Event Count')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "uncertainty_vs_events_scaling.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot 2: Uncertainty at fixed x values
    if fixed_x_values and scaling_results['fixed_x_uncertainties']:
        n_x_vals = len(fixed_x_values)
        fig, axes = plt.subplots(1, n_x_vals, figsize=(5 * n_x_vals, 5))
        if n_x_vals == 1:
            axes = [axes]
        
        for i, x_val in enumerate(fixed_x_values):
            ax = axes[i]
            
            for func_name, uncertainties in scaling_results['fixed_x_uncertainties'][x_val].items():
                if uncertainties:  # Check if we have data
                    ax.loglog(event_counts, uncertainties, 'o-', label=func_name, 
                             linewidth=2, markersize=6)
            
            # Add theoretical scaling
            if scaling_results['fixed_x_uncertainties'][x_val]:
                first_uncertainties = list(scaling_results['fixed_x_uncertainties'][x_val].values())[0]
                if first_uncertainties:
                    theoretical = first_uncertainties[0] * np.sqrt(event_counts[0] / np.array(event_counts))
                    ax.loglog(event_counts, theoretical, 'k--', alpha=0.7, linewidth=2,
                             label=r'$\propto 1/\sqrt{N}$')
            
            ax.set_xlabel('Number of Events')
            ax.set_ylabel('Uncertainty at x')
            ax.set_title(f'Uncertainty at x = {x_val}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "uncertainty_vs_events_fixed_x.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Save detailed analysis
    analysis_path = os.path.join(save_dir, "uncertainty_scaling_analysis.txt")
    with open(analysis_path, 'w') as f:
        f.write("Uncertainty Quantification Scaling Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write("This analysis demonstrates the consistency of uncertainty quantification\n")
        f.write("by showing how uncertainty bands shrink as the number of events increases.\n\n")
        f.write("THEORY: For well-behaved statistical estimators, uncertainty should scale\n")
        f.write("approximately as 1/√N where N is the number of data points (events).\n\n")
        f.write(f"Configuration:\n")
        f.write(f"Problem: {problem}\n")
        f.write(f"True parameters: {true_params.cpu().numpy()}\n")
        f.write(f"Event counts tested: {event_counts}\n")
        f.write(f"Bootstrap samples per count: {n_bootstrap}\n")
        f.write(f"Laplace uncertainty: {'Available' if laplace_model is not None else 'Not available'}\n\n")
        
        f.write("Parameter Uncertainty Results:\n")
        f.write("-" * 30 + "\n")
        for i, param_name in enumerate(param_names):
            f.write(f"{param_name}:\n")
            uncertainties = param_uncertainties[:, i]
            for j, (count, unc) in enumerate(zip(event_counts, uncertainties)):
                f.write(f"  {count:>8,} events: {unc:.6f}\n")
            
            # Compute scaling exponent via linear regression in log space
            log_counts = np.log(event_counts)
            log_uncertainties = np.log(uncertainties)
            slope, intercept = np.polyfit(log_counts, log_uncertainties, 1)
            f.write(f"  Scaling exponent: {slope:.3f} (ideal: -0.5)\n")
            f.write(f"  R² fit quality: {np.corrcoef(log_counts, log_uncertainties)[0,1]**2:.3f}\n\n")
        
        f.write("Function Uncertainty Results:\n")
        f.write("-" * 30 + "\n")
        for func_name, uncertainties in scaling_results['function_uncertainties'].items():
            f.write(f"{func_name}:\n")
            for count, unc in zip(event_counts, uncertainties):
                f.write(f"  {count:>8,} events: {unc:.6f}\n")
            
            # Compute scaling exponent
            log_counts = np.log(event_counts)
            log_uncertainties = np.log(uncertainties)
            slope, intercept = np.polyfit(log_counts, log_uncertainties, 1)
            f.write(f"  Scaling exponent: {slope:.3f} (ideal: -0.5)\n")
            f.write(f"  R² fit quality: {np.corrcoef(log_counts, log_uncertainties)[0,1]**2:.3f}\n\n")
        
        if fixed_x_values:
            f.write("Fixed X-Value Uncertainty Results:\n")
            f.write("-" * 35 + "\n")
            for x_val in fixed_x_values:
                f.write(f"At x = {x_val}:\n")
                for func_name, uncertainties in scaling_results['fixed_x_uncertainties'][x_val].items():
                    if uncertainties:
                        f.write(f"  {func_name}:\n")
                        for count, unc in zip(event_counts, uncertainties):
                            f.write(f"    {count:>8,} events: {unc:.6f}\n")
                        
                        # Compute scaling exponent
                        log_counts = np.log(event_counts)
                        log_uncertainties = np.log(uncertainties)
                        slope, intercept = np.polyfit(log_counts, log_uncertainties, 1)
                        f.write(f"    Scaling exponent: {slope:.3f} (ideal: -0.5)\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("- Scaling exponents close to -0.5 indicate proper statistical behavior\n")
        f.write("- R² values close to 1.0 indicate consistent power-law scaling\n") 
        f.write("- Deviations may indicate systematic effects or insufficient bootstrap samples\n")
        f.write("- This analysis validates the consistency of the uncertainty quantification method\n")
    
    print(f"✅ Uncertainty scaling analysis complete! Results saved to {save_dir}")
    print(f"   📊 Main scaling plot: uncertainty_vs_events_scaling.png")
    if fixed_x_values:
        print(f"   📍 Fixed x analysis: uncertainty_vs_events_fixed_x.png")
    print(f"   📄 Detailed analysis: uncertainty_scaling_analysis.txt")
    
    return scaling_results


def plot_uncertainty_at_fixed_x(
    scaling_results,
    x_values=None,
    save_dir=None,
    comparison_functions=None
):
    """
    Create detailed plots showing uncertainty at specific x values as a function 
    of the number of events.
    
    This function takes the results from plot_uncertainty_vs_events and creates
    focused visualizations of how uncertainty behaves at specific x-coordinates.
    This is particularly useful for understanding if uncertainty scaling is 
    consistent across different regions of the PDF.
    
    Args:
        scaling_results: Results dictionary from plot_uncertainty_vs_events
        x_values: List of x values to plot (uses those from scaling_results if None)
        save_dir: Directory to save plots (uses scaling_results save_dir if None)
        comparison_functions: List of function names to compare (optional)
        
    Returns:
        None (saves plots to save_dir)
        
    Saves:
        - uncertainty_fixed_x_comparison.png: Comparison across functions at each x
        - uncertainty_fixed_x_scaling_quality.png: Quality metrics for scaling fits
        
    Example Usage:
        # After running plot_uncertainty_vs_events
        scaling_results = plot_uncertainty_vs_events(...)
        
        # Create detailed fixed-x analysis
        plot_uncertainty_at_fixed_x(
            scaling_results=scaling_results,
            x_values=[0.01, 0.1, 0.5],
            comparison_functions=['up', 'down']  # for simplified_dis
        )
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    if 'fixed_x_uncertainties' not in scaling_results:
        print("⚠️  No fixed x uncertainty data found in scaling_results")
        return
    
    if save_dir is None:
        save_dir = "./plots/scaling_analysis"  # fallback
    os.makedirs(save_dir, exist_ok=True)
    
    event_counts = scaling_results['event_counts']
    fixed_x_data = scaling_results['fixed_x_uncertainties']
    
    if x_values is None:
        x_values = list(fixed_x_data.keys())
    
    if not x_values:
        print("⚠️  No x values to plot")
        return
    
    # Determine functions to plot
    all_functions = set()
    for x_val in x_values:
        if x_val in fixed_x_data:
            all_functions.update(fixed_x_data[x_val].keys())
    
    if comparison_functions is not None:
        all_functions = [f for f in comparison_functions if f in all_functions]
    else:
        all_functions = sorted(list(all_functions))
    
    print(f"📍 Creating fixed-x uncertainty plots for x = {x_values}")
    print(f"   Functions: {all_functions}")
    
    # Plot 1: Uncertainty comparison at each x value
    n_x = len(x_values)
    fig, axes = plt.subplots(1, n_x, figsize=(5 * n_x, 5))
    if n_x == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_functions)))
    
    for i, x_val in enumerate(x_values):
        ax = axes[i]
        
        if x_val in fixed_x_data:
            for j, func_name in enumerate(all_functions):
                if func_name in fixed_x_data[x_val]:
                    uncertainties = fixed_x_data[x_val][func_name]
                    if uncertainties:
                        ax.loglog(event_counts, uncertainties, 'o-', 
                                 color=colors[j], label=func_name,
                                 linewidth=2, markersize=6)
            
            # Add theoretical 1/sqrt(N) line
            if fixed_x_data[x_val] and all_functions:
                first_func = all_functions[0]
                if first_func in fixed_x_data[x_val] and fixed_x_data[x_val][first_func]:
                    first_uncertainties = fixed_x_data[x_val][first_func]
                    theoretical = first_uncertainties[0] * np.sqrt(event_counts[0] / np.array(event_counts))
                    ax.loglog(event_counts, theoretical, 'k--', alpha=0.7, linewidth=2,
                             label=r'$\propto 1/\sqrt{N}$')
        
        ax.set_xlabel('Number of Events')
        ax.set_ylabel('Uncertainty')
        ax.set_title(f'Uncertainty at x = {x_val}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "uncertainty_fixed_x_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot 2: Scaling quality metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compute scaling exponents and R² values
    scaling_exponents = {}
    r_squared_values = {}
    
    for x_val in x_values:
        if x_val in fixed_x_data:
            scaling_exponents[x_val] = {}
            r_squared_values[x_val] = {}
            
            for func_name in all_functions:
                if func_name in fixed_x_data[x_val] and fixed_x_data[x_val][func_name]:
                    uncertainties = fixed_x_data[x_val][func_name]
                    
                    # Linear regression in log space
                    log_counts = np.log(event_counts)
                    log_uncertainties = np.log(uncertainties)
                    slope, intercept = np.polyfit(log_counts, log_uncertainties, 1)
                    r_squared = np.corrcoef(log_counts, log_uncertainties)[0, 1] ** 2
                    
                    scaling_exponents[x_val][func_name] = slope
                    r_squared_values[x_val][func_name] = r_squared
    
    # Plot scaling exponents
    x_positions = np.arange(len(x_values))
    width = 0.8 / len(all_functions) if all_functions else 0.8
    
    for j, func_name in enumerate(all_functions):
        exponents = [scaling_exponents.get(x_val, {}).get(func_name, np.nan) for x_val in x_values]
        offset = (j - len(all_functions)/2 + 0.5) * width
        ax1.bar(x_positions + offset, exponents, width, label=func_name, color=colors[j], alpha=0.7)
    
    ax1.axhline(-0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Ideal (-0.5)')
    ax1.set_xlabel('x value')
    ax1.set_ylabel('Scaling Exponent')
    ax1.set_title('Uncertainty Scaling Exponents')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'{x:.2f}' for x in x_values])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot R² values
    for j, func_name in enumerate(all_functions):
        r_squared_vals = [r_squared_values.get(x_val, {}).get(func_name, np.nan) for x_val in x_values]
        offset = (j - len(all_functions)/2 + 0.5) * width
        ax2.bar(x_positions + offset, r_squared_vals, width, label=func_name, color=colors[j], alpha=0.7)
    
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Perfect (1.0)')
    ax2.set_xlabel('x value')
    ax2.set_ylabel('R² (Fit Quality)')
    ax2.set_title('Scaling Fit Quality')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'{x:.2f}' for x in x_values])
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "uncertainty_fixed_x_scaling_quality.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Fixed-x uncertainty analysis complete!")
    print(f"   📊 Comparison plot: uncertainty_fixed_x_comparison.png") 
    print(f"   📈 Quality metrics: uncertainty_fixed_x_scaling_quality.png")


def plot_summary_uncertainty_scaling(
    scaling_results,
    save_dir=None,
    include_theoretical_comparison=True,
    aggregation_method='mean'
):
    """
    Create a summary plot showing how average uncertainty decreases with 
    increasing number of events, demonstrating consistency of uncertainty 
    quantification.
    
    This function creates the key summary visualization requested in the problem 
    statement: a line chart or log-log plot showing how average uncertainty 
    decreases with increasing data, with annotations highlighting consistency.
    
    Args:
        scaling_results: Results dictionary from plot_uncertainty_vs_events
        save_dir: Directory to save plots (uses scaling_results info if None)
        include_theoretical_comparison: Whether to overlay theoretical 1/√N scaling
        aggregation_method: How to aggregate uncertainty across functions/parameters
                           ('mean', 'median', 'max', 'rms')
        
    Returns:
        Dict: Summary statistics and consistency metrics
        
    Saves:
        - uncertainty_scaling_summary.png: Main summary plot (log-log)
        - uncertainty_scaling_linear.png: Linear scale version  
        - uncertainty_consistency_metrics.txt: Quantitative consistency analysis
        
    Example Usage:
        # After running uncertainty vs events analysis
        scaling_results = plot_uncertainty_vs_events(...)
        
        # Create summary plots
        summary_metrics = plot_summary_uncertainty_scaling(
            scaling_results=scaling_results,
            include_theoretical_comparison=True,
            aggregation_method='mean'
        )
        
        print(f"Overall consistency score: {summary_metrics['consistency_score']:.3f}")
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    if save_dir is None:
        save_dir = "./plots/scaling_analysis"  # fallback
    os.makedirs(save_dir, exist_ok=True)
    
    event_counts = np.array(scaling_results['event_counts'])
    
    print(f"📈 Creating summary uncertainty scaling plots...")
    print(f"   Aggregation method: {aggregation_method}")
    print(f"   Theoretical comparison: {include_theoretical_comparison}")
    
    # Aggregate parameter uncertainties
    param_uncertainties = np.array(scaling_results['parameter_uncertainties'])  # [n_counts, n_params]
    
    if aggregation_method == 'mean':
        agg_param_uncertainty = np.mean(param_uncertainties, axis=1)
    elif aggregation_method == 'median':
        agg_param_uncertainty = np.median(param_uncertainties, axis=1)
    elif aggregation_method == 'max':
        agg_param_uncertainty = np.max(param_uncertainties, axis=1)
    elif aggregation_method == 'rms':
        agg_param_uncertainty = np.sqrt(np.mean(param_uncertainties**2, axis=1))
    else:
        agg_param_uncertainty = np.mean(param_uncertainties, axis=1)  # fallback
    
    # Aggregate function uncertainties
    function_uncertainties = scaling_results['function_uncertainties']
    if function_uncertainties:
        func_unc_matrix = np.array(list(function_uncertainties.values())).T  # [n_counts, n_functions]
        
        if aggregation_method == 'mean':
            agg_func_uncertainty = np.mean(func_unc_matrix, axis=1)
        elif aggregation_method == 'median':
            agg_func_uncertainty = np.median(func_unc_matrix, axis=1)
        elif aggregation_method == 'max':
            agg_func_uncertainty = np.max(func_unc_matrix, axis=1)
        elif aggregation_method == 'rms':
            agg_func_uncertainty = np.sqrt(np.mean(func_unc_matrix**2, axis=1))
        else:
            agg_func_uncertainty = np.mean(func_unc_matrix, axis=1)  # fallback
    else:
        agg_func_uncertainty = None
    
    # Theoretical 1/√N scaling
    if include_theoretical_comparison:
        # Normalize to first data point
        param_theoretical = agg_param_uncertainty[0] * np.sqrt(event_counts[0] / event_counts)
        if agg_func_uncertainty is not None:
            func_theoretical = agg_func_uncertainty[0] * np.sqrt(event_counts[0] / event_counts)
    
    # Create main summary plot (log-log)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot aggregated uncertainties
    ax.loglog(event_counts, agg_param_uncertainty, 'o-', color='royalblue', 
              linewidth=3, markersize=8, label=f'{aggregation_method.title()} Parameter Uncertainty')
    
    if agg_func_uncertainty is not None:
        ax.loglog(event_counts, agg_func_uncertainty, 's-', color='darkorange',
                  linewidth=3, markersize=8, label=f'{aggregation_method.title()} Function Uncertainty')
    
    # Add theoretical scaling
    if include_theoretical_comparison:
        ax.loglog(event_counts, param_theoretical, '--', color='royalblue', alpha=0.7,
                  linewidth=2, label=r'Parameter $\propto 1/\sqrt{N}$')
        if agg_func_uncertainty is not None:
            ax.loglog(event_counts, func_theoretical, '--', color='darkorange', alpha=0.7,
                      linewidth=2, label=r'Function $\propto 1/\sqrt{N}$')
    
    ax.set_xlabel('Number of Events', fontsize=14)
    ax.set_ylabel('Average Uncertainty', fontsize=14)
    ax.set_title('Uncertainty Quantification Consistency:\nUncertainty Decreases with Increasing Data', 
                 fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Add consistency annotations
    if include_theoretical_comparison:
        # Compute how well data follows theoretical scaling
        log_counts = np.log(event_counts)
        
        # Parameter scaling analysis
        log_param_unc = np.log(agg_param_uncertainty)
        param_slope, param_intercept = np.polyfit(log_counts, log_param_unc, 1)
        param_r2 = np.corrcoef(log_counts, log_param_unc)[0, 1] ** 2
        
        # Function scaling analysis
        if agg_func_uncertainty is not None:
            log_func_unc = np.log(agg_func_uncertainty)
            func_slope, func_intercept = np.polyfit(log_counts, log_func_unc, 1)
            func_r2 = np.corrcoef(log_counts, log_func_unc)[0, 1] ** 2
        else:
            func_slope, func_r2 = np.nan, np.nan
        
        # Add text annotations
        textstr = f'Parameter Scaling:\n  Exponent: {param_slope:.3f} (ideal: -0.5)\n  R²: {param_r2:.3f}'
        if not np.isnan(func_slope):
            textstr += f'\n\nFunction Scaling:\n  Exponent: {func_slope:.3f} (ideal: -0.5)\n  R²: {func_r2:.3f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        # Add consistency indicator
        param_consistency = max(0, 1 - abs(param_slope + 0.5) / 0.5)  # How close to -0.5
        if not np.isnan(func_slope):
            func_consistency = max(0, 1 - abs(func_slope + 0.5) / 0.5)
            overall_consistency = (param_consistency + func_consistency) / 2
        else:
            overall_consistency = param_consistency
        
        # Color-coded consistency indicator
        if overall_consistency > 0.8:
            consistency_color = 'green'
            consistency_text = 'EXCELLENT'
        elif overall_consistency > 0.6:
            consistency_color = 'orange'
            consistency_text = 'GOOD'
        else:
            consistency_color = 'red'
            consistency_text = 'POOR'
        
        ax.text(0.95, 0.05, f'Consistency: {consistency_text}\n({overall_consistency:.1%})', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                horizontalalignment='right', verticalalignment='bottom',
                color=consistency_color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "uncertainty_scaling_summary.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create linear scale version
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(event_counts, agg_param_uncertainty, 'o-', color='royalblue',
            linewidth=3, markersize=8, label=f'{aggregation_method.title()} Parameter Uncertainty')
    
    if agg_func_uncertainty is not None:
        ax.plot(event_counts, agg_func_uncertainty, 's-', color='darkorange',
                linewidth=3, markersize=8, label=f'{aggregation_method.title()} Function Uncertainty')
    
    if include_theoretical_comparison:
        ax.plot(event_counts, param_theoretical, '--', color='royalblue', alpha=0.7,
                linewidth=2, label=r'Parameter $\propto 1/\sqrt{N}$')
        if agg_func_uncertainty is not None:
            ax.plot(event_counts, func_theoretical, '--', color='darkorange', alpha=0.7,
                    linewidth=2, label=r'Function $\propto 1/\sqrt{N}$')
    
    ax.set_xlabel('Number of Events', fontsize=14)
    ax.set_ylabel('Average Uncertainty', fontsize=14)
    ax.set_title('Uncertainty Quantification Consistency (Linear Scale):\nUncertainty Decreases with Increasing Data', 
                 fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "uncertainty_scaling_linear.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save quantitative consistency analysis
    consistency_path = os.path.join(save_dir, "uncertainty_consistency_metrics.txt")
    with open(consistency_path, 'w') as f:
        f.write("Uncertainty Quantification Consistency Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write("This analysis quantifies how well the uncertainty quantification method\n")
        f.write("follows expected statistical scaling laws, demonstrating consistency.\n\n")
        f.write("EXPECTED BEHAVIOR: Uncertainty should scale as 1/√N where N is the\n")
        f.write("number of events (data points). This corresponds to a slope of -0.5\n")
        f.write("in log-log space.\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"Problem: {scaling_results['problem']}\n")
        f.write(f"Event counts: {event_counts.tolist()}\n")
        f.write(f"Aggregation method: {aggregation_method}\n")
        f.write(f"Bootstrap samples: {scaling_results['n_bootstrap']}\n\n")
        
        if include_theoretical_comparison:
            f.write("Parameter Uncertainty Analysis:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Scaling exponent: {param_slope:.4f} (ideal: -0.5)\n")
            f.write(f"Deviation from ideal: {abs(param_slope + 0.5):.4f}\n")
            f.write(f"R² (fit quality): {param_r2:.4f}\n")
            f.write(f"Consistency score: {param_consistency:.3f} (0=poor, 1=perfect)\n\n")
            
            if not np.isnan(func_slope):
                f.write("Function Uncertainty Analysis:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Scaling exponent: {func_slope:.4f} (ideal: -0.5)\n")
                f.write(f"Deviation from ideal: {abs(func_slope + 0.5):.4f}\n")
                f.write(f"R² (fit quality): {func_r2:.4f}\n")
                f.write(f"Consistency score: {func_consistency:.3f} (0=poor, 1=perfect)\n\n")
                
                f.write(f"Overall Consistency Score: {overall_consistency:.3f}\n\n")
            else:
                f.write(f"Overall Consistency Score: {param_consistency:.3f}\n\n")
        
        f.write("Uncertainty Values by Event Count:\n")
        f.write("-" * 35 + "\n")
        f.write(f"{'Events':>10s} {'Param Unc':>12s}")
        if agg_func_uncertainty is not None:
            f.write(f" {'Func Unc':>12s}")
        f.write("\n")
        
        for i, count in enumerate(event_counts):
            f.write(f"{count:>10,} {agg_param_uncertainty[i]:>12.6f}")
            if agg_func_uncertainty is not None:
                f.write(f" {agg_func_uncertainty[i]:>12.6f}")
            f.write("\n")
        
        f.write("\nINTERPRETATION:\n")
        f.write("- Consistency scores > 0.8: Excellent scaling behavior\n")
        f.write("- Consistency scores 0.6-0.8: Good scaling behavior\n")
        f.write("- Consistency scores < 0.6: Poor scaling, may indicate issues\n")
        f.write("- R² > 0.9 indicates strong linear relationship in log space\n")
        f.write("- Scaling exponents close to -0.5 indicate proper statistical behavior\n")
    
    # Prepare return dictionary
    summary_metrics = {
        'event_counts': event_counts,
        'aggregated_param_uncertainty': agg_param_uncertainty,
        'aggregated_func_uncertainty': agg_func_uncertainty,
        'aggregation_method': aggregation_method
    }
    
    if include_theoretical_comparison:
        summary_metrics.update({
            'param_scaling_exponent': param_slope,
            'param_r_squared': param_r2,
            'param_consistency_score': param_consistency,
            'overall_consistency_score': overall_consistency
        })
        
        if not np.isnan(func_slope):
            summary_metrics.update({
                'func_scaling_exponent': func_slope,
                'func_r_squared': func_r2,
                'func_consistency_score': func_consistency
            })
    
    print(f"✅ Summary uncertainty scaling analysis complete!")
    print(f"   📊 Log-log summary: uncertainty_scaling_summary.png")
    print(f"   📈 Linear summary: uncertainty_scaling_linear.png")
    print(f"   📄 Consistency metrics: uncertainty_consistency_metrics.txt")
    
    if include_theoretical_comparison:
        print(f"   🎯 Overall consistency score: {overall_consistency:.3f}")
        if overall_consistency > 0.8:
            print("   ✅ EXCELLENT: Uncertainty scaling follows expected statistical behavior")
        elif overall_consistency > 0.6:
            print("   ⚠️  GOOD: Uncertainty scaling is mostly consistent")
        else:
            print("   ❌ POOR: Uncertainty scaling deviates significantly from expected behavior")

def get_parameter_bounds_for_problem(problem):
    """
    Get parameter bounds for different problem types based on Dataset class definitions.
    
    Parameters:
    -----------
    problem : str
        Problem type ('simplified_dis', 'realistic_dis', 'mceg', 'gaussian')
        
    Returns:
    --------
    torch.Tensor
        Parameter bounds tensor of shape (n_params, 2) where bounds[:, 0] are lower bounds
        and bounds[:, 1] are upper bounds
    """
    if problem == 'simplified_dis':
        # From DISDataset class: [[0.0, 5]] * theta_dim for theta_dim=4
        return torch.tensor([[0.0, 5.0]] * 4)
    elif problem == 'realistic_dis':
        # From RealisticDISDataset class
        return torch.tensor([
            [-2.0, 2.0],   # logA0
            [-1.0, 1.0],   # delta
            [0.0, 5.0],    # a
            [0.0, 10.0],   # b
            [-5.0, 5.0],   # c
            [-5.0, 5.0],   # d
        ])
    elif problem in ['mceg', 'mceg4dis']:
        # From MCEGDISDataset class
        return torch.tensor([
            [-1.0, 10.0],
            [0.0, 10.0],
            [-10.0, 10.0],
            [-10.0, 10.0],
        ])
    elif problem == 'gaussian':
        # From Gaussian2DDataset class
        return torch.tensor([
            [-2.0, 2.0],   # mu_x
            [-2.0, 2.0],   # mu_y
            [0.5, 2.0],    # sigma_x
            [0.5, 2.0],    # sigma_y
            [-0.8, 0.8],   # rho
        ])
    else:
        raise ValueError(f"Unknown problem type: {problem}. Supported: 'simplified_dis', 'realistic_dis', 'mceg', 'gaussian'")

def get_simulator_for_problem(problem, device=None):
    """
    Get the appropriate simulator for the given problem type.
    
    Parameters:
    -----------
    problem : str
        Problem type ('simplified_dis', 'realistic_dis', 'mceg', 'gaussian')
    device : torch.device, optional
        Device to run the simulator on
        
    Returns:
    --------
    simulator
        The appropriate simulator instance
    """
    SimplifiedDIS, RealisticDIS, MCEGSimulator = get_simulator_module()
    
    if problem == 'simplified_dis':
        if SimplifiedDIS is None:
            raise ImportError("Could not import SimplifiedDIS from simulator module")
        return SimplifiedDIS(device=device)
    elif problem == 'realistic_dis':
        if RealisticDIS is None:
            raise ImportError("Could not import RealisticDIS from simulator module")
        return RealisticDIS(device=device, smear=True, smear_std=0.05)
    elif problem in ['mceg', 'mceg4dis']:
        if MCEGSimulator is None:
            raise ImportError("Could not import MCEGSimulator from simulator module")
        return MCEGSimulator(device=device)
    elif problem == 'gaussian':
        try:
            from simulator import Gaussian2DSimulator
            return Gaussian2DSimulator(device=device)
        except ImportError:
            raise ImportError("Could not import Gaussian2DSimulator from simulator module")
    else:
        raise ValueError(f"Unknown problem type: {problem}")

@torch.no_grad()
def generate_parameter_error_histogram(
    model, 
    pointnet_model, 
    device,
    n_draws=100,
    n_events=10000,
    problem='simplified_dis',
    laplace_model=None,
    save_path="parameter_error_histogram.png",
    param_names=None,
    return_data=False
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
        Trained inference model (InferenceNet)
    pointnet_model : torch.nn.Module
        Trained PointNet model for latent extraction
    device : torch.device
        Device to run computations on
    n_draws : int, default=100
        Number of parameter sets to sample and evaluate via simulation
    n_events : int, default=10000
        Number of events to generate per parameter set via simulation
    problem : str, default='simplified_dis'
        Problem type ('simplified_dis', 'realistic_dis', 'mceg', 'gaussian')
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
    
    >>> # With Laplace uncertainty
    >>> generate_parameter_error_histogram(
    ...     model, pointnet_model, device,
    ...     laplace_model=laplace_model,
    ...     problem='realistic_dis',
    ...     save_path='realistic_dis_errors.png'
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
        advanced_feature_engineering = get_advanced_feature_engineering()
        print(f"   ✅ Feature engineering function loaded")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not load advanced_feature_engineering: {e}")
        print(f"       Using identity function as fallback")
        advanced_feature_engineering = lambda x: x
    
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
            true_params = (theta_raw * (theta_bounds[:, 1] - theta_bounds[:, 0]) + 
                          theta_bounds[:, 0])
            
            # Generate events using the simulator
            try:
                if problem == 'gaussian':
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
                                padding = torch.zeros(pad_size, xs.shape[1], device=device)
                            else:
                                padding = torch.zeros(pad_size, device=device)
                            xs = torch.cat([xs, padding], dim=0)
                
            except Exception as e:
                print(f"   ⚠️  Failed to generate events for sample {i}: {e}")
                failed_samples += 1
                continue
            
            # Apply feature engineering
            try:
                if problem == 'simplified_dis':
                    xs_engineered = advanced_feature_engineering(xs)
                else:
                    xs_engineered = log_feature_engineering(xs)
                if not isinstance(xs_engineered, torch.Tensor):
                    xs_engineered = torch.tensor(xs_engineered, device=device, dtype=torch.float32)
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
                if hasattr(model, 'nll_mode') and model.nll_mode:
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
                            laplace_samples = laplace_model.sample(10, x=latent)  # Sample from posterior
                            predicted_params = laplace_samples.mean(dim=0).squeeze()
                    except Exception as e:
                        print(f"   ⚠️  Laplace inference failed for sample {i}, using standard prediction: {e}")
                
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
        raise RuntimeError("All parameter samples failed. Check model compatibility and data pipeline.")
    elif failed_samples > 0:
        print(f"   ⚠️  {failed_samples}/{n_draws} samples failed, proceeding with {successful_samples} successful samples")
    
    print(f"   ✅ Successfully processed {successful_samples} parameter samples")
    
    # Generate the histogram plot using existing function
    try:
        plot_parameter_error_histogram(
            true_params_list=true_params_list,
            predicted_params_list=predicted_params_list,
            param_names=param_names,
            save_path=save_path,
            problem=problem
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
            print(f"   {param_name}: MAE={abs_errors[:, i].mean():.4f}, MRE={rel_errors[:, i].mean():.4f}")
    
    print(f"🎯 Parameter error histogram generation complete!")
    
    if return_data:
        return true_params_list, predicted_params_list