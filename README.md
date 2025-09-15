# Contrastive Learning for Parton Distribution Function Inference

This repository contains code for an AI-driven approach for learning parameters from event data in quantum chromodynamics.

Training takes place in two stages. First, a PointNet-style embedding of simulated data is learned through contrastive learning. The idea here is to learn an embedding which preserves distances in the space of event data. Then, an embedding-to-parameters network is trained.

## NEW: Precomputed Data Pipeline

The repository now supports **precomputed datasets** for faster, more reproducible training:

```bash
# Generate precomputed datasets
python generate_precomputed_data.py --problems gaussian simplified_dis --num_samples 10000

# Train with precomputed data (faster, reproducible)
python end_to_end.py --problem gaussian --use_precomputed --precomputed_data_dir precomputed_data

# Train with on-the-fly generation (original method, still supported)
python end_to_end.py --problem gaussian --num_samples 10000 --num_events 1000
```

**Benefits of precomputed data:**
- ✅ **Faster training**: No simulation overhead during training iterations
- ✅ **Reproducible**: Identical datasets across experiments
- ✅ **Debuggable**: Fixed data enables easier debugging
- ✅ **All problems supported**: gaussian, simplified_dis, realistic_dis, mceg
- ✅ **Distributed training**: Automatic data splitting across ranks

See [PRECOMPUTED_DATA_README.md](PRECOMPUTED_DATA_README.md) for detailed documentation.

## How to train:

### Option 1: With Precomputed Data (Recommended)

1. **Generate datasets**: `python generate_precomputed_data.py --problems gaussian simplified_dis --num_samples 10000`
2. **Train models**: `python end_to_end.py --problem gaussian --use_precomputed`

### Option 2: Original On-the-Fly Method

1. Run `python cl.py`. This will automatically save the PointNet embedding model.
2. Run `python PDF_learning.py` (basic) or `python PDF_learning_UQ.py` (with Laplace uncertainty). This will automatically save the embedding-to-parameters model.

## Plotting and Uncertainty Quantification:

### Basic Plotting:
- Run `python plotting_driver.py` for basic parameter and error plotting.

### Advanced Uncertainty Quantification:
- Run `python plotting_driver_UQ.py` for **function-level uncertainty quantification** using analytic Laplace approximation.
- **KEY FEATURE**: Now reports uncertainty over the **predicted functions** (u(x), d(x), q(x)) at each x-point, not just model parameters.
- Uses **pointwise uncertainty aggregation** where total_variance(x) = variance_bootstrap(x) + variance_laplace(x).
- Automatically detects and uses Laplace models when available, falls back to Monte Carlo when not.

### NEW: Uncertainty Scaling Analysis:
- **Demonstrates UQ consistency** by showing how uncertainty bands shrink as the number of events increases
- Run `python example_uncertainty_scaling.py --mock_mode` to see uncertainty scaling validation
- **Key functions**: `plot_uncertainty_vs_events()`, `plot_summary_uncertainty_scaling()`, `plot_uncertainty_at_fixed_x()`
- **Validates 1/√N scaling**: Shows uncertainty decreases appropriately with more data
- **Comprehensive analysis**: Parameter uncertainties, function uncertainties, and fixed x-value tracking
- **Quantitative metrics**: Consistency scores, scaling exponents, and R² quality measures
- See `UNCERTAINTY_SCALING_README.md` for detailed documentation and usage examples

**Function-Level Uncertainty Quantification:**
- **What changed**: Instead of reporting uncertainty over parameters θ, the system now computes uncertainty over the induced PDF functions f(x|θ) at each x-point.
- **Why this matters**: Function-level uncertainty is more interpretable for physics applications and directly quantifies prediction uncertainty for the PDFs themselves.
- **Method**: For each bootstrap sample, multiple parameter samples θ are drawn from the posterior, f(x|θ) is evaluated at each x, and pointwise statistics (mean ± std) are computed across all function evaluations.
- **Uncertainty combination**: Data uncertainty (bootstrap variance) and model uncertainty (Laplace variance) are combined pointwise: total_variance(x) = var_bootstrap(x) + var_laplace(x).

**Key Features:**
- **Function-level uncertainty**: Reports uncertainty of predicted PDFs f(x) at each x, not just parameters θ
- **Pointwise aggregation**: Combines uncertainty sources in function space, providing interpretable uncertainty bands
- **Analytic uncertainty**: Uses Laplace approximation with delta method for speed and accuracy
- **Automatic detection**: Finds and loads Laplace models automatically  
- **Backward compatibility**: Falls back to Monte Carlo when Laplace unavailable
- **Multiple architectures**: Supports MLP, Transformer, Gaussian, and Multimodal heads

**Usage Examples:**
```bash
# Plot with function-level uncertainty (recommended):
python plotting_driver_UQ.py --arch gaussian --problem simplified_dis

# Combined uncertainty analysis with detailed function-level breakdown:
python example_combined_uncertainty_usage.py --problem simplified_dis --n_bootstrap 50

# Plot all architectures with function uncertainty:
python plotting_driver_UQ.py --arch all --latent_dim 1024

# Fast fallback for missing Laplace models:
python plotting_driver_UQ.py --arch mlp --n_mc 50
```

**Output Files for Function-Level Uncertainty:**
- `function_uncertainty_pdf_{function}.png`: PDF plots with function-level uncertainty bands
- `function_uncertainty_breakdown_{function}.txt`: Pointwise uncertainty statistics for each x
- `function_uncertainty_summary.png`: Summary of uncertainty across all functions
- `function_uncertainty_methodology.txt`: Detailed explanation of the uncertainty computation method

## Dependencies:

This repository requires relatively few dependencies:
- Up-to-date PyTorch
- NumPy and Matplotlib
- For uncertainty quantification: `laplace-torch` library
- MCEG simulator usage is limited to those who have permission to access the relevant repository.
