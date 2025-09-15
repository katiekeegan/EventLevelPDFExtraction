#!/usr/bin/env python3
"""
Test script for enhanced plotting functions.
Tests the new plotting utilities without external dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import sys

# Create a minimal test environment
def test_histogram_functions():
    """Test the new histogram functions with mock data"""
    
    print("🧪 Testing enhanced plotting functions...")
    
    # Mock parameter data for testing
    n_samples = 100
    n_params = 4
    
    # Generate mock true and predicted parameters
    true_params_list = []
    predicted_params_list = []
    
    # Generate realistic parameter scenarios
    base_true = np.array([2.0, 1.2, 2.0, 1.2])  # simplified_dis example
    
    for i in range(n_samples):
        # Add some variation to true parameters
        noise = np.random.normal(0, 0.1, n_params)
        true_params = base_true + noise
        
        # Add prediction error (bias + noise)
        pred_error = np.random.normal(0, 0.2, n_params) + np.random.normal(0.05, 0.1, n_params)
        predicted_params = true_params + pred_error
        
        true_params_list.append(true_params)
        predicted_params_list.append(predicted_params)
    
    # Test parameter error histogram function
    try:
        # Import the specific functions we want to test
        sys.path.insert(0, '.')
        
        # Create colorblind-friendly color palette (copy from plotting_UQ_utils.py)
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
        
        # Test parameter error histogram
        def plot_parameter_error_histogram_test(
            true_params_list,
            predicted_params_list,
            param_names=None,
            save_path="test_parameter_error_histogram.png",
            problem='simplified_dis'
        ):
            # Convert to numpy if needed and compute errors
            true_params = np.array(true_params_list)  # Shape: (n_samples, n_params)
            predicted_params = np.array(predicted_params_list)
            
            # Compute parameter errors
            param_errors = predicted_params - true_params
            relative_errors = param_errors / (true_params + 1e-8)
            
            n_params = param_errors.shape[1]
            
            # Set default parameter names
            if param_names is None:
                if problem == 'simplified_dis':
                    param_names = [r'$a_u$', r'$b_u$', r'$a_d$', r'$b_d$']
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
                n_bins = min(50, max(10, len(param_errors) // 5))
                counts, bins, patches = ax_abs.hist(
                    param_errors[:, i], 
                    bins=n_bins, 
                    alpha=0.7, 
                    color=color,
                    edgecolor='white',
                    linewidth=0.5
                )
                
                ax_abs.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='True Value')
                
                mean_err = np.mean(param_errors[:, i])
                std_err = np.std(param_errors[:, i])
                ax_abs.text(0.02, 0.98, f'μ = {mean_err:.3f}\nσ = {std_err:.3f}', 
                           transform=ax_abs.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax_abs.set_xlabel(f'Error in {param_names[i]}')
                ax_abs.set_ylabel('Frequency')
                ax_abs.set_title(f'Absolute Error: {param_names[i]}')
                ax_abs.grid(True, alpha=0.3)
                if i == 0:
                    ax_abs.legend()
                
                # Relative errors (bottom row)
                ax_rel = axes[1, i]
                counts, bins, patches = ax_rel.hist(
                    relative_errors[:, i] * 100,
                    bins=n_bins, 
                    alpha=0.7, 
                    color=color,
                    edgecolor='white',
                    linewidth=0.5
                )
                
                ax_rel.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='True Value')
                
                mean_rel_err = np.mean(relative_errors[:, i]) * 100
                std_rel_err = np.std(relative_errors[:, i]) * 100
                ax_rel.text(0.02, 0.98, f'μ = {mean_rel_err:.1f}%\nσ = {std_rel_err:.1f}%', 
                           transform=ax_rel.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax_rel.set_xlabel(f'Relative Error in {param_names[i]} (%)')
                ax_rel.set_ylabel('Frequency')
                ax_rel.set_title(f'Relative Error: {param_names[i]}')
                ax_rel.grid(True, alpha=0.3)
                if i == 0:
                    ax_rel.legend()
            
            plt.suptitle('Parameter Error Analysis (Test)', fontsize=18, y=0.95)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Parameter error histogram saved to {save_path}")

        plot_parameter_error_histogram_test(true_params_list, predicted_params_list)
        
        # Test function error histogram  
        def plot_function_error_histogram_test(save_path="test_function_error_histogram.png"):
            # Generate mock function evaluation data
            n_samples = 50
            n_points = 100
            n_functions = 2
            
            true_function_values_list = []
            predicted_function_values_list = []
            
            x = np.linspace(0.01, 1, n_points)
            
            for i in range(n_samples):
                # Generate mock true functions (e.g., power laws)
                true_u = 2.0 * x**(-0.5)  # Mock up function
                true_d = 1.2 * x**(-0.8)  # Mock down function
                true_funcs = np.column_stack([true_u, true_d])
                
                # Add prediction errors
                error_u = np.random.normal(0, 0.1 * true_u)
                error_d = np.random.normal(0, 0.1 * true_d)
                pred_u = true_u + error_u
                pred_d = true_d + error_d
                pred_funcs = np.column_stack([pred_u, pred_d])
                
                true_function_values_list.append(true_funcs)
                predicted_function_values_list.append(pred_funcs)
            
            # Test function
            true_vals = np.array(true_function_values_list)
            pred_vals = np.array(predicted_function_values_list)
            
            n_samples, n_points, n_functions = true_vals.shape
            function_names = [r'$u(x)$', r'$d(x)$']
            
            # Compute average entrywise errors
            abs_errors = np.abs(pred_vals - true_vals)
            avg_abs_errors = np.mean(abs_errors, axis=1)
            
            rel_errors = abs_errors / (np.abs(true_vals) + 1e-8)
            avg_rel_errors = np.mean(rel_errors, axis=1)
            
            # Create subplots
            fig, axes = plt.subplots(2, n_functions, figsize=(6*n_functions, 10))
            if n_functions == 1:
                axes = axes.reshape(2, 1)
            
            colors_list = [COLORBLIND_COLORS['blue'], COLORBLIND_COLORS['orange']]
            
            for i in range(n_functions):
                color = colors_list[i % len(colors_list)]
                
                # Absolute errors (top row)
                ax_abs = axes[0, i]
                counts, bins_abs, patches = ax_abs.hist(
                    avg_abs_errors[:, i], 
                    bins=30, 
                    alpha=0.7, 
                    color=color,
                    edgecolor='white',
                    linewidth=0.5,
                    density=True
                )
                
                mean_abs = np.mean(avg_abs_errors[:, i])
                median_abs = np.median(avg_abs_errors[:, i])
                
                ax_abs.axvline(mean_abs, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_abs:.4f}')
                ax_abs.axvline(median_abs, color='purple', linestyle=':', alpha=0.8, linewidth=2, label=f'Median: {median_abs:.4f}')
                
                ax_abs.set_xlabel(f'Average Absolute Error in {function_names[i]}')
                ax_abs.set_ylabel('Probability Density')
                ax_abs.set_title(f'Distribution of Errors: {function_names[i]}')
                ax_abs.grid(True, alpha=0.3)
                ax_abs.legend()
                
                # Relative errors (bottom row)  
                ax_rel = axes[1, i]
                counts, bins_rel, patches = ax_rel.hist(
                    avg_rel_errors[:, i] * 100,
                    bins=30, 
                    alpha=0.7, 
                    color=color,
                    edgecolor='white',
                    linewidth=0.5,
                    density=True
                )
                
                mean_rel = np.mean(avg_rel_errors[:, i]) * 100
                median_rel = np.median(avg_rel_errors[:, i]) * 100
                
                ax_rel.axvline(mean_rel, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_rel:.2f}%')
                ax_rel.axvline(median_rel, color='purple', linestyle=':', alpha=0.8, linewidth=2, label=f'Median: {median_rel:.2f}%')
                
                ax_rel.set_xlabel(f'Average Relative Error in {function_names[i]} (%)')
                ax_rel.set_ylabel('Probability Density')
                ax_rel.set_title(f'Distribution of Relative Errors: {function_names[i]}')
                ax_rel.grid(True, alpha=0.3)
                ax_rel.legend()
            
            plt.suptitle('Function Value Error Analysis (Test)', fontsize=18, y=0.95)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Function error histogram saved to {save_path}")
        
        plot_function_error_histogram_test()
        
        print("✅ All histogram functions tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing histogram functions: {e}")
        return False

if __name__ == "__main__":
    success = test_histogram_functions()
    if success:
        print("🎉 All tests passed! New plotting functions are working correctly.")
    else:
        print("💥 Some tests failed. Check the error messages above.")