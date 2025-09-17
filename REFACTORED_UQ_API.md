# Updated UQ Plotting API

The UQ plotting functions in `uq_plotting_demo.py` have been refactored to support a consistent, generator-style API that matches the plotting workflow functions in `plotting_driver_UQ.py`.

## New API Usage

All functions now accept these standard arguments:

```python
from uq_plotting_demo import (
    plot_parameter_uncertainty,
    plot_function_uncertainty,
    plot_bootstrap_uncertainty,
    plot_combined_uncertainty_decomposition,
    plot_uncertainty_scaling
)

# Common arguments for all functions
model = your_parameter_prediction_model
pointnet_model = your_pointnet_feature_extractor
true_params = torch.tensor([2.0, 1.2, 2.0, 1.2])  # Ground truth parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_events = 10000  # Number of events to generate
problem = 'simplified_dis'  # Problem type
save_dir = "plots"  # Output directory

# 1. Parameter uncertainty
plot_parameter_uncertainty(
    model=model,
    pointnet_model=pointnet_model,
    true_params=true_params,
    device=device,
    num_events=num_events,
    problem=problem,
    save_dir=save_dir
)

# 2. Function uncertainty  
plot_function_uncertainty(
    model=model,
    pointnet_model=pointnet_model,
    true_params=true_params,
    device=device,
    num_events=num_events,
    problem=problem,
    save_dir=save_dir,
    laplace_model=laplace_model  # Optional for analytic uncertainty
)

# 3. Bootstrap uncertainty
plot_bootstrap_uncertainty(
    model=model,
    pointnet_model=pointnet_model,
    true_params=true_params,
    device=device,
    num_events=num_events,
    n_bootstrap=50,
    problem=problem,
    save_dir=save_dir
)

# 4. Combined uncertainty decomposition
plot_combined_uncertainty_decomposition(
    model=model,
    pointnet_model=pointnet_model,
    true_params=true_params,
    device=device,
    num_events=num_events,
    n_bootstrap=30,
    problem=problem,
    save_dir=save_dir,
    laplace_model=laplace_model  # Optional
)

# 5. Uncertainty scaling
plot_uncertainty_scaling(
    model=model,
    pointnet_model=pointnet_model,
    true_params=true_params,
    device=device,
    event_counts=[1000, 5000, 10000, 50000],
    n_bootstrap=20,
    problem=problem,
    save_dir=save_dir
)
```

## Backward Compatibility

The functions still support the original API for existing code:

```python
# Legacy API still works
simulator = SimplifiedDIS()
true_theta = torch.tensor([2.0, 1.2, 2.0, 1.2])
observed_data = simulator.sample(true_theta, 2000)

plot_parameter_uncertainty(
    simulator=simulator,
    true_theta=true_theta,
    observed_data=observed_data,
    save_dir="plots"
)
```

## Key Benefits

1. **Consistent API**: All functions now use the same argument pattern as other plotting workflow functions
2. **Generator-style**: Functions generate their own simulation data internally  
3. **Flexible**: Support for different problem types ('simplified_dis', 'realistic_dis', 'gaussian', 'mceg')
4. **Backward Compatible**: Existing code continues to work unchanged
5. **Laplace Support**: Optional integration with Laplace approximation for analytic uncertainty
6. **Extensible**: Easy to add new problem types and simulators

## Integration with plotting_driver_UQ.py

The functions are now fully compatible with the plotting workflow in `plotting_driver_UQ.py` and can be called using the same argument patterns as other plotting functions.