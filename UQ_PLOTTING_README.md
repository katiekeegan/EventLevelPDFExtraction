# Enhanced Uncertainty Quantification Plotting Demo

This module provides a comprehensive suite of uncertainty quantification plots using **only simulator data**. It demonstrates parameter-space uncertainty, function-space uncertainty, bootstrap uncertainty, combined uncertainty decomposition, and uncertainty scaling with number of events.

## 🎯 Features

### Complete UQ Analysis Suite
- **Parameter-space uncertainty**: Posterior distributions of inferred parameters
- **Function-space (predictive) uncertainty**: Uncertainty propagation to function predictions  
- **Bootstrap/data uncertainty**: Parametric bootstrap analysis of sampling uncertainty
- **Combined uncertainty decomposition**: Variance decomposition into model vs. data components
- **Uncertainty scaling**: How uncertainty decreases with increasing data size

### Publication-Ready Outputs
- High-quality plots (300 DPI) with professional styling
- Colorblind-friendly color palettes
- Mathematical notation and statistical annotations
- LaTeX descriptions explaining uncertainty computation for each plot
- Self-contained: no external data dependencies

## 📦 Requirements

```bash
pip install torch matplotlib numpy scipy tqdm
```

## 🚀 Quick Start

### Basic Usage

```bash
# Run the complete demonstration
python uq_plotting_demo.py
```

This generates all uncertainty quantification plots in the `plots/` directory, with each plot accompanied by a LaTeX description file.

### Output Structure

```
plots/
├── parameter_uncertainty.png     # Parameter posterior distributions  
├── parameter_uncertainty.tex     # LaTeX explanation
├── function_uncertainty.png      # Function-space uncertainty bands
├── function_uncertainty.tex      # LaTeX explanation  
├── bootstrap_uncertainty.png     # Bootstrap parameter distributions
├── bootstrap_uncertainty.tex     # LaTeX explanation
├── uncertainty_decomposition.png # Variance decomposition analysis
├── uncertainty_decomposition.tex # LaTeX explanation
├── uncertainty_scaling.png       # Uncertainty vs. number of events
├── uncertainty_scaling.tex       # LaTeX explanation
└── gaussian_demo/                # Bonus: Gaussian2D demonstration
    ├── parameter_uncertainty.png
    └── parameter_uncertainty.tex
```

## 🔬 Simulator Interface

The module includes two example simulators that demonstrate the required interface:

### SimplifiedDIS Simulator
```python
from uq_plotting_demo import SimplifiedDIS

# Initialize simulator
simulator = SimplifiedDIS(device='cpu', smear=True, smear_std=0.02)

# Define parameters: [au, bu, ad, bd] for up/down quark PDFs
theta = torch.tensor([2.0, 1.2, 2.0, 1.2])

# Generate events
events = simulator.sample(theta, n_events=1000)

# Evaluate PDF functions
x = torch.linspace(0.01, 0.99, 100)
functions = simulator.f(x, theta)  # Returns {'up': ..., 'down': ...}
```

### Gaussian2D Simulator
```python
from uq_plotting_demo import Gaussian2DSimulator

# Initialize simulator  
simulator = Gaussian2DSimulator(device='cpu')

# Define parameters: [mu_x, mu_y, sigma_x, sigma_y, rho]
theta = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.3])

# Generate events
events = simulator.sample(theta, n_events=1000)

# Evaluate function (1D marginal for plotting)
x = torch.linspace(-3, 3, 100)
f_vals = simulator.f(x, theta)
```

## 🔧 Adapting to Your Simulator

To use with your own simulator, ensure it implements this interface:

```python
class YourSimulator:
    def sample(self, theta, n_events):
        """
        Generate simulated events.
        
        Args:
            theta: Parameter tensor of shape (n_params,)
            n_events: Number of events to generate
            
        Returns:
            Tensor of shape (n_events, n_features)
        """
        pass
        
    def f(self, x, theta):
        """
        Evaluate functions f(x|theta) for uncertainty propagation.
        
        Args:
            x: Input tensor of shape (n_points,)
            theta: Parameter tensor of shape (n_params,)
            
        Returns:
            Dict with function values or single tensor
        """
        pass
```

Then simply replace the simulator in the demo:

```python
# Replace this line in uq_plotting_demo.py
simulator = YourSimulator()

# Update parameter bounds in posterior_sampler() function
theta_bounds = [(low1, high1), (low2, high2), ...]  # Your parameter ranges

# Update parameter names for plots
param_names = [r'$\theta_1$', r'$\theta_2$', ...]  # Your parameter names
```

## 📊 Individual Plot Functions

You can also use individual plotting functions:

```python
from uq_plotting_demo import (
    plot_parameter_uncertainty,
    plot_function_uncertainty,  
    plot_bootstrap_uncertainty,
    plot_combined_uncertainty_decomposition,
    plot_uncertainty_scaling
)

# Generate parameter uncertainty plot
posterior_samples = plot_parameter_uncertainty(
    simulator=simulator,
    true_theta=true_theta,
    observed_data=observed_data,
    save_dir="my_plots"
)

# Generate function uncertainty plot  
plot_function_uncertainty(
    simulator=simulator,
    posterior_samples=posterior_samples,
    true_theta=true_theta,
    save_dir="my_plots"
)
```

## 🧮 Mathematical Background

### Parameter-Space Uncertainty
Shows posterior distribution $p(\theta|\mathcal{D})$ computed via:
$$p(\theta|\mathcal{D}) \propto p(\mathcal{D}|\theta) p(\theta)$$

### Function-Space Uncertainty  
Propagates parameter uncertainty to function predictions:
$$p(f(x)|\mathcal{D}) = \int p(f(x)|\theta) p(\theta|\mathcal{D}) d\theta$$

### Bootstrap Uncertainty
Parametric bootstrap procedure:
1. Generate $B$ datasets $\{\mathcal{D}_b\}$ with same true parameters
2. Estimate $\hat{\theta}_b$ for each dataset  
3. Analyze distribution of $\{\hat{\theta}_b\}$

### Uncertainty Decomposition
Decomposes total variance as:
$$\text{Var}_{\text{total}}[f(x)] = \mathbb{E}_b[\text{Var}_{\theta|b}[f(x|\theta)]] + \text{Var}_b[\mathbb{E}_{\theta|b}[f(x|\theta)]]$$

### Uncertainty Scaling
Demonstrates theoretical $1/\sqrt{N}$ scaling of estimation uncertainty with sample size.

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python test_uq_plotting_demo.py
```

## 📝 LaTeX Integration

Each plot comes with a detailed LaTeX description saved as a `.tex` file. These can be directly included in your papers:

```latex
\input{plots/parameter_uncertainty.tex}
\begin{figure}
    \centering
    \includegraphics[width=\textwidth]{plots/parameter_uncertainty.png}
    \caption{Parameter-space uncertainty analysis.}
\end{figure}
```

## 🎨 Customization

### Styling
The module uses publication-ready defaults but can be customized:

```python
# Modify the COLORS dictionary for custom color schemes
COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    # ... add your colors
}

# Modify matplotlib rcParams for custom styling
plt.rcParams.update({
    'font.size': 14,          # Increase font size
    'figure.dpi': 300,        # High-resolution output
    # ... other customizations
})
```

### Plot Parameters
Key parameters can be adjusted in each function:

```python
# Number of posterior samples for speed vs. accuracy tradeoff
posterior_samples = posterior_sampler(simulator, data, bounds, n_samples=1000)

# Bootstrap iterations for precision vs. speed
bootstrap_estimates = plot_bootstrap_uncertainty(
    simulator, true_theta, n_bootstrap=100  # Increase for better statistics
)

# Event counts for scaling analysis  
plot_uncertainty_scaling(
    simulator, true_theta, 
    event_counts=[100, 500, 1000, 5000, 10000]  # Customize range
)
```

## 📖 Example Applications

### High-Energy Physics
- PDF parameter inference (as demonstrated)
- Cross-section measurements with systematic uncertainties
- Detector calibration uncertainty propagation

### Bayesian Inference
- Parameter estimation with model comparison
- Predictive uncertainty in surrogate models
- Bootstrap validation of MCMC results

### Machine Learning
- Neural network uncertainty quantification
- Hyperparameter optimization uncertainty
- Model selection uncertainty analysis

## 🤝 Contributing

To extend this module:

1. Add new simulator classes following the established interface
2. Create new plotting functions following the existing pattern:
   - Generate plot with matplotlib
   - Save high-quality PNG 
   - Create informative LaTeX description
   - Use consistent styling and error handling

3. Update tests and documentation

## 📄 Citation

If you use this module in your research, please cite:

```bibtex
@software{uq_plotting_demo,
    title={Enhanced Uncertainty Quantification Plotting Demo},
    author={PDFParameterInference Contributors},
    year={2024},
    url={https://github.com/katiekeegan/PDFParameterInference}
}
```

## 📞 Support

For questions or issues:
- Check the test suite: `python test_uq_plotting_demo.py`
- Review the LaTeX descriptions for mathematical details
- Examine the source code for implementation details
- Open an issue in the repository

---

**🎯 Key Benefits:**
- ✅ **Self-contained**: No external data dependencies
- ✅ **Production-ready**: Publication-quality plots and documentation  
- ✅ **Extensible**: Easy to adapt for new simulators and use cases
- ✅ **Educational**: Comprehensive LaTeX explanations of UQ concepts
- ✅ **Efficient**: Optimized for practical computational constraints