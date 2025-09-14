# Uncertainty Scaling Analysis Functions

This document describes the new plotting functions added to `plotting_UQ_utils.py` that demonstrate the consistency of uncertainty quantification by showing how uncertainty bands shrink as the number of events (data) increases.

## Overview

The new functions address a key requirement in uncertainty quantification: demonstrating that uncertainties decrease appropriately as more data becomes available. This is fundamental to validating that your UQ method is statistically sound and follows expected scaling laws.

### Key Principle

For well-behaved statistical estimators, uncertainty should scale approximately as **1/√N** where N is the number of data points (events). This relationship indicates that:
- Doubling the data reduces uncertainty by ~30%
- Increasing data 10× reduces uncertainty by ~68%
- This scaling demonstrates statistical consistency

## New Functions

### 1. `plot_uncertainty_vs_events()`

**Main function** that runs the full uncertainty quantification pipeline across multiple event counts and analyzes scaling behavior.

```python
def plot_uncertainty_vs_events(
    model,
    pointnet_model, 
    true_params,
    device,
    event_counts=None,           # [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
    n_bootstrap=20,
    laplace_model=None,
    problem='simplified_dis',
    save_dir=None,
    Q2_slices=None,
    fixed_x_values=None
):
```

**What it does:**
- Tests uncertainty quantification across multiple event counts
- Runs bootstrap sampling for each event count
- Evaluates both parameter and function-level uncertainties
- Tracks uncertainty at specific x values
- Generates comprehensive scaling analysis

**Outputs:**
- `uncertainty_vs_events_scaling.png`: Main scaling plot (log-log)
- `uncertainty_vs_events_fixed_x.png`: Uncertainty at fixed x values
- `uncertainty_scaling_analysis.txt`: Detailed statistical analysis

### 2. `plot_uncertainty_at_fixed_x()`

**Detailed analysis** of uncertainty behavior at specific x-coordinates.

```python
def plot_uncertainty_at_fixed_x(
    scaling_results,
    x_values=None,
    save_dir=None,
    comparison_functions=None
):
```

**What it does:**
- Creates focused plots for specific x values
- Compares uncertainty across different PDF functions
- Analyzes scaling quality metrics (R², exponents)
- Shows consistency across different regions of the PDF

**Outputs:**
- `uncertainty_fixed_x_comparison.png`: Comparison across functions
- `uncertainty_fixed_x_scaling_quality.png`: Quality metrics

### 3. `plot_summary_uncertainty_scaling()`

**Summary visualization** showing overall uncertainty scaling consistency.

```python
def plot_summary_uncertainty_scaling(
    scaling_results,
    save_dir=None,
    include_theoretical_comparison=True,
    aggregation_method='mean'      # 'mean', 'median', 'max', 'rms'
):
```

**What it does:**
- Creates the key summary plot requested in the problem statement
- Shows average uncertainty vs. number of events
- Overlays theoretical 1/√N scaling for comparison
- Provides quantitative consistency scores
- Annotates plots with scaling quality metrics

**Outputs:**
- `uncertainty_scaling_summary.png`: Main summary (log-log)
- `uncertainty_scaling_linear.png`: Linear scale version
- `uncertainty_consistency_metrics.txt`: Quantitative analysis

## Usage Examples

### Basic Usage

```python
from plotting_UQ_utils import plot_uncertainty_vs_events

# Run scaling analysis for simplified DIS
scaling_results = plot_uncertainty_vs_events(
    model=your_model,
    pointnet_model=your_pointnet,
    true_params=torch.tensor([2.0, 1.2, 2.0, 1.2]),
    device=device,
    event_counts=[1000, 5000, 10000, 50000, 100000],
    n_bootstrap=30,
    laplace_model=your_laplace_model,  # optional
    problem='simplified_dis',
    save_dir='./plots/scaling_analysis'
)
```

### Complete Analysis Pipeline

```python
from plotting_UQ_utils import (
    plot_uncertainty_vs_events,
    plot_uncertainty_at_fixed_x,
    plot_summary_uncertainty_scaling
)

# 1. Main scaling analysis
scaling_results = plot_uncertainty_vs_events(
    model=model,
    pointnet_model=pointnet_model,
    true_params=true_params,
    device=device,
    event_counts=[1000, 5000, 10000, 50000, 100000],
    n_bootstrap=25,
    save_dir='./plots/scaling'
)

# 2. Detailed fixed-x analysis  
plot_uncertainty_at_fixed_x(
    scaling_results=scaling_results,
    x_values=[0.01, 0.1, 0.5],
    save_dir='./plots/scaling'
)

# 3. Summary with consistency metrics
summary_metrics = plot_summary_uncertainty_scaling(
    scaling_results=scaling_results,
    save_dir='./plots/scaling',
    aggregation_method='mean'
)

print(f"Consistency score: {summary_metrics['overall_consistency_score']:.3f}")
```

### Command Line Usage

```bash
# Run example with mock data (no models required)
python example_uncertainty_scaling.py --mock_mode

# Real analysis with custom parameters
python example_uncertainty_scaling.py \
    --problem simplified_dis \
    --event_counts 1000,5000,10000,50000 \
    --n_bootstrap 25 \
    --save_dir ./plots/my_scaling_analysis

# Realistic DIS with specific Q2 slices
python example_uncertainty_scaling.py \
    --problem realistic_dis \
    --Q2_slices 2.0,10.0,50.0 \
    --fixed_x_values 0.01,0.1,0.5
```

## Interpreting Results

### Consistency Scores

The functions provide quantitative consistency scores:

- **> 0.8**: Excellent scaling behavior
- **0.6-0.8**: Good scaling behavior  
- **< 0.6**: Poor scaling (may indicate issues)

### Scaling Exponents

- **Ideal**: -0.5 (perfect 1/√N scaling)
- **Acceptable**: -0.4 to -0.6 (close to ideal)
- **Concerning**: Outside this range

### R² Values

- **> 0.9**: Strong linear relationship in log space
- **0.8-0.9**: Good linear relationship
- **< 0.8**: Weak relationship (may indicate problems)

## Key Features

### 1. Demonstrates Statistical Consistency
- Shows uncertainty shrinks as data increases
- Validates 1/√N scaling behavior
- Provides quantitative consistency metrics

### 2. Multiple Uncertainty Sources
- Bootstrap uncertainty (data uncertainty)
- Laplace uncertainty (model uncertainty)
- Combined uncertainty analysis

### 3. Function-Level Focus
- Emphasizes uncertainty over predicted PDFs f(x)
- More interpretable than parameter-only uncertainty
- Pointwise uncertainty at each x-coordinate

### 4. Comprehensive Analysis
- Parameter and function uncertainties
- Fixed x-value tracking
- Scaling quality metrics
- Detailed documentation

### 5. Robust Implementation
- Handles missing dependencies gracefully
- Works with or without Laplace models
- Extensive error checking and validation
- Mock data mode for testing

## Integration with Existing Code

These functions integrate seamlessly with the existing uncertainty quantification framework:

```python
# Use with existing plotting drivers
from plotting_driver_UQ import reload_model, reload_pointnet
from plotting_UQ_utils import plot_uncertainty_vs_events

# Load your trained models
model = reload_model(model_path, device)
pointnet_model = reload_pointnet(pointnet_path, device)

# Run scaling analysis
scaling_results = plot_uncertainty_vs_events(
    model=model,
    pointnet_model=pointnet_model,
    true_params=your_true_params,
    device=device,
    problem='simplified_dis',
    save_dir='./plots/scaling'
)
```

## Files Generated

Each analysis generates a comprehensive set of outputs:

### Main Plots
- `uncertainty_vs_events_scaling.png`: Core scaling visualization
- `uncertainty_scaling_summary.png`: Summary plot with annotations
- `uncertainty_fixed_x_comparison.png`: Fixed x-value analysis

### Quality Analysis
- `uncertainty_fixed_x_scaling_quality.png`: Scaling quality metrics
- `uncertainty_scaling_linear.png`: Linear scale version

### Documentation
- `uncertainty_scaling_analysis.txt`: Detailed statistical analysis
- `uncertainty_consistency_metrics.txt`: Quantitative consistency report

## Dependencies

The functions work with the existing codebase dependencies:
- PyTorch
- Matplotlib  
- NumPy
- SciPy
- tqdm

Optional dependencies are handled gracefully:
- Laplace models (falls back to bootstrap-only)
- Simulator modules (can use mock data)

## Testing

Run the included tests to validate functionality:

```bash
# Test core mathematical functions
python test_uncertainty_minimal.py

# Test with mock data (full pipeline)
python example_uncertainty_scaling.py --mock_mode
```

This demonstrates the functions work correctly and can be used to validate your installation.