# Contrastive Learning for Parton Distribution Function Inference

This repository contains code for an AI-driven approach for learning parameters from event data in quantum chromodynamics.

Training takes place in two stages. First, a PointNet-style embedding of simulated data is learned through contrastive learning. The idea here is to learn an embedding which preserves distances in the space of event data. Then, an embedding-to-parameters network is trained.

## How to train:

1. Run `python cl.py`. This will automatically save the PointNet embedding model.
2. Run `python PDF_learning.py` (basic) or `python PDF_learning_UQ.py` (with Laplace uncertainty). This will automatically save the embedding-to-parameters model.

## Plotting and Uncertainty Quantification:

### Basic Plotting:
- Run `python plotting_driver.py` for basic parameter and error plotting.

### Advanced Uncertainty Quantification:
- Run `python plotting_driver_UQ.py` for **analytic uncertainty propagation** using Laplace approximation.
- Uses the **delta method** for fast, accurate uncertainty quantification instead of Monte Carlo sampling.
- Automatically detects and uses Laplace models when available, falls back to Monte Carlo when not.

**Key Features:**
- **Analytic uncertainty**: Uses Laplace approximation with delta method for speed and accuracy
- **Automatic detection**: Finds and loads Laplace models automatically  
- **Backward compatibility**: Falls back to Monte Carlo when Laplace unavailable
- **Multiple architectures**: Supports MLP, Transformer, Gaussian, and Multimodal heads

**Usage Examples:**
```bash
# Plot with analytic uncertainty (recommended):
python plotting_driver_UQ.py --arch gaussian --problem simplified_dis

# Plot all architectures:
python plotting_driver_UQ.py --arch all --latent_dim 1024

# Fast fallback for missing Laplace models:
python plotting_driver_UQ.py --arch mlp --n_mc 50
```

## Dependencies:

This repository requires relatively few dependencies:
- Up-to-date PyTorch
- NumPy and Matplotlib
- For uncertainty quantification: `laplace-torch` library
- MCEG simulator usage is limited to those who have permission to access the relevant repository.
