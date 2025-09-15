# generate_parameter_error_histogram Function

## Overview

The `generate_parameter_error_histogram` function is a comprehensive utility that automates the entire workflow for benchmarking model accuracy across the parameter space. This standalone function handles everything from parameter sampling to generating publication-ready error histograms.

## Key Features

✅ **Automatic parameter bounds retrieval** - No need to manually specify bounds for different problem types  
✅ **Complete inference pipeline** - Handles event generation, feature engineering, and model inference  
✅ **Publication-ready plots** - Creates beautiful histograms with comprehensive error statistics  
✅ **Robust error handling** - Graceful fallbacks and detailed error reporting  
✅ **Optional uncertainty quantification** - Supports Laplace uncertainty if available  
✅ **Minimal user input** - Just specify models, device, and basic settings  

## Function Signature

```python
generate_parameter_error_histogram(
    model,                    # Trained inference model (InferenceNet)
    pointnet_model,          # Trained PointNet model
    device,                  # PyTorch device (cpu/cuda)
    n_draws=100,             # Number of parameter sets to sample
    n_events=10000,          # Number of events per parameter set
    problem='simplified_dis', # Problem type
    laplace_model=None,      # Optional Laplace uncertainty model
    save_path="parameter_error_histogram.png",
    param_names=None,        # Custom parameter names
    return_data=False        # Return raw data for analysis
)
```

## Supported Problem Types

| Problem | Parameters | Bounds | Description |
|---------|-----------|--------|-------------|
| `simplified_dis` | 4 | [0, 5] for all | Simplified DIS problem |
| `realistic_dis` | 6 | Custom bounds | Realistic DIS with 6 parameters |
| `mceg` | 4 | Custom bounds | MCEG simulator |
| `gaussian` | 5 | Custom bounds | 2D Gaussian simulator |

## Usage Examples

### Basic Usage
```python
from plotting_UQ_utils import generate_parameter_error_histogram

# Minimal usage - just provide models and device
generate_parameter_error_histogram(
    model=inference_model,
    pointnet_model=pointnet_model,
    device=device,
    problem='simplified_dis',
    save_path='param_errors.png'
)
```

### Advanced Usage with Custom Settings
```python
# Custom settings and parameter names
true_params, pred_params = generate_parameter_error_histogram(
    model=inference_model,
    pointnet_model=pointnet_model,
    device=device,
    n_draws=200,              # More samples for better statistics
    n_events=50000,           # More events per sample
    problem='realistic_dis',
    laplace_model=laplace_model,  # Include uncertainty
    param_names=['logA₀', 'δ', 'a', 'b', 'c', 'd'],  # Custom labels
    save_path='realistic_dis_errors.png',
    return_data=True          # Get data for further analysis
)

# Analyze the returned data
import torch
true_params_tensor = torch.stack(true_params)
pred_params_tensor = torch.stack(pred_params)
abs_errors = torch.abs(pred_params_tensor - true_params_tensor)
print(f"Mean absolute error: {abs_errors.mean():.4f}")
```

## Workflow

The function automatically handles the complete workflow:

1. **Parameter Bounds Retrieval** - Gets appropriate bounds for the problem type
2. **Parameter Sampling** - Uniformly samples N parameter sets from bounds
3. **Event Generation** - Uses the appropriate simulator (SimplifiedDIS, RealisticDIS, etc.)
4. **Feature Engineering** - Applies advanced feature engineering to events
5. **Latent Extraction** - Uses PointNet to extract latent features
6. **Parameter Prediction** - Uses inference model to predict parameters
7. **Error Computation** - Calculates absolute and relative errors
8. **Visualization** - Creates publication-ready histograms with statistics

## Output

The function generates:
- **Publication-ready histogram plot** showing parameter errors
- **Comprehensive error statistics** (MAE, MRE, per-parameter breakdown)
- **Optional data return** for further analysis

## Error Handling

The function includes robust error handling:
- Graceful fallbacks for missing dependencies
- Detailed error messages and progress reporting  
- Continues with successful samples if some fail
- Validates model compatibility and data pipeline

## Tips for Production Use

- **Use larger sample sizes**: `n_draws=100-1000` for production analysis
- **Increase events per sample**: `n_events=10k-100k` for more stable results
- **Enable Laplace uncertainty**: If available, provides uncertainty quantification
- **Use return_data=True**: For further statistical analysis beyond the plot
- **Custom parameter names**: For better plot readability and publication

## Requirements

- PyTorch models in evaluation mode
- Compatible device specification
- Appropriate simulators available for the problem type
- matplotlib, numpy, torch, tqdm dependencies

## Integration

This function integrates seamlessly with the existing plotting ecosystem in `plotting_UQ_utils.py` and uses the existing `plot_parameter_error_histogram` function for the actual plotting, ensuring consistent styling and functionality.