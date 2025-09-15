#!/usr/bin/env python3
"""
Usage examples for enhanced plotting utilities in PDFParameterInference.

This file demonstrates how to use the new and enhanced plotting functions
for creating publication-ready visualizations of uncertainty quantification
in PDF parameter inference.

Run this file to see examples of all the enhanced plotting capabilities.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# Ensure we can import the enhanced plotting functions
try:
    from plotting_UQ_utils import (
        plot_parameter_error_histogram,
        plot_function_error_histogram,
        plot_event_histogram_simplified_DIS,
        plot_params_distribution_single,
        plot_PDF_distribution_single
    )
    print("✅ Successfully imported enhanced plotting functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure matplotlib, numpy, and torch are installed")
    exit(1)

def generate_mock_data():
    """Generate mock data for demonstration purposes"""
    print("📊 Generating mock data for plotting demonstrations...")
    
    # Mock parameter data
    n_samples = 100
    n_params = 4
    base_true_params = np.array([2.0, 1.2, 2.0, 1.2])  # simplified_dis example
    
    true_params_list = []
    predicted_params_list = []
    
    for i in range(n_samples):
        # Add variation to true parameters
        noise = np.random.normal(0, 0.05, n_params)
        true_params = base_true_params + noise
        
        # Add prediction errors
        prediction_bias = np.random.normal(0.02, 0.08, n_params)
        prediction_noise = np.random.normal(0, 0.15, n_params)
        predicted_params = true_params + prediction_bias + prediction_noise
        
        true_params_list.append(true_params)
        predicted_params_list.append(predicted_params)
    
    # Mock function evaluation data
    n_points = 200
    n_functions = 2
    x = np.linspace(0.001, 1, n_points)
    
    true_function_values_list = []
    predicted_function_values_list = []
    
    for i in range(n_samples):
        # Generate mock PDFs (power laws with noise)
        true_u = 2.0 * x**(-0.5) + np.random.normal(0, 0.1 * x**(-0.5))
        true_d = 1.2 * x**(-0.8) + np.random.normal(0, 0.1 * x**(-0.8))
        true_funcs = np.column_stack([true_u, true_d])
        
        # Add prediction errors
        pred_u = true_u + np.random.normal(0, 0.2 * true_u)
        pred_d = true_d + np.random.normal(0, 0.2 * true_d)
        pred_funcs = np.column_stack([pred_u, pred_d])
        
        true_function_values_list.append(true_funcs)
        predicted_function_values_list.append(pred_funcs)
    
    return {
        'true_params_list': true_params_list,
        'predicted_params_list': predicted_params_list,
        'true_function_values_list': true_function_values_list,
        'predicted_function_values_list': predicted_function_values_list
    }

def demonstrate_parameter_error_plots(data):
    """Demonstrate parameter error histogram plotting"""
    print("📈 Creating parameter error histograms...")
    
    try:
        plot_parameter_error_histogram(
            true_params_list=data['true_params_list'],
            predicted_params_list=data['predicted_params_list'],
            save_path="example_parameter_errors.png",
            problem='simplified_dis'
        )
        print("✅ Parameter error histogram saved to example_parameter_errors.png")
        
    except Exception as e:
        print(f"❌ Error creating parameter error plot: {e}")

def demonstrate_function_error_plots(data):
    """Demonstrate function error histogram plotting"""
    print("📊 Creating function error histograms...")
    
    try:
        plot_function_error_histogram(
            true_function_values_list=data['true_function_values_list'],
            predicted_function_values_list=data['predicted_function_values_list'],
            function_names=[r'$u(x)$', r'$d(x)$'],
            save_path="example_function_errors.png"
        )
        print("✅ Function error histogram saved to example_function_errors.png")
        
    except Exception as e:
        print(f"❌ Error creating function error plot: {e}")

def demonstrate_style_consistency():
    """Demonstrate the consistent styling across different plot types"""
    print("🎨 Creating style consistency demonstration...")
    
    # Import the color schemes
    try:
        from plotting_UQ_utils import COLORBLIND_COLORS, UNCERTAINTY_COLORS, PDF_FUNCTION_COLORS
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Color palette demonstration
        ax = axes[0, 0]
        colors = list(COLORBLIND_COLORS.values())[:8]
        labels = list(COLORBLIND_COLORS.keys())[:8]
        x = np.arange(len(colors))
        bars = ax.bar(x, np.ones(len(colors)), color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title('Colorblind-Friendly Palette')
        ax.set_ylabel('Intensity')
        
        # Uncertainty visualization demonstration
        ax = axes[0, 1]
        x = np.linspace(0, 10, 100)
        y_true = np.sin(x)
        y_pred = y_true + 0.1 * np.random.normal(0, 1, len(x))
        uncertainty = 0.2 * np.abs(y_true) + 0.1
        
        ax.plot(x, y_true, color=UNCERTAINTY_COLORS['true'], 
               linewidth=2.5, label='True Function')
        ax.plot(x, y_pred, color=UNCERTAINTY_COLORS['predicted'], 
               linewidth=2, label='Predicted')
        ax.fill_between(x, y_pred - uncertainty, y_pred + uncertainty,
                       color=UNCERTAINTY_COLORS['combined'], alpha=0.3,
                       label='Uncertainty Band')
        ax.set_title('Uncertainty Visualization Style')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mathematical notation demonstration
        ax = axes[1, 0]
        x = np.logspace(-3, 0, 100)
        for i, (func_name, color) in enumerate(PDF_FUNCTION_COLORS.items()):
            if func_name in ['up', 'down']:
                y = (i + 1) * x**(-(i + 1)*0.5)
                ax.plot(x, y, color=color, linewidth=2.5, 
                       label=f'${func_name}(x|\\theta)$')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'PDF Value')
        ax.set_title('Mathematical Notation Style')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        # Statistics annotation demonstration
        ax = axes[1, 1]
        data = np.random.normal(100, 15, 1000)
        counts, bins, patches = ax.hist(data, bins=30, alpha=0.7, 
                                       color=COLORBLIND_COLORS['blue'],
                                       edgecolor='white', linewidth=0.5)
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.axvline(mean_val, color=COLORBLIND_COLORS['red'], 
                  linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        
        # Statistics text box
        stats_text = f'μ = {mean_val:.2f}\nσ = {std_val:.2f}\nN = {len(data)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
        
        ax.set_title('Statistical Annotations Style')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Enhanced Plotting Style Consistency', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig('style_consistency_demo.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Style consistency demo saved to style_consistency_demo.png")
        
    except Exception as e:
        print(f"❌ Error creating style demo: {e}")

def main():
    """Main demonstration function"""
    print("🎯 Enhanced Plotting Utilities Demonstration")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("plotting_examples", exist_ok=True)
    os.chdir("plotting_examples")
    
    # Generate mock data
    data = generate_mock_data()
    
    # Demonstrate different plotting capabilities
    demonstrate_parameter_error_plots(data)
    demonstrate_function_error_plots(data)
    demonstrate_style_consistency()
    
    print("\n🎉 Demonstration complete!")
    print("Generated files:")
    print("  - example_parameter_errors.png: Parameter error analysis")
    print("  - example_function_errors.png: Function error analysis") 
    print("  - style_consistency_demo.png: Style and color scheme demo")
    print("\nAll plots feature:")
    print("  ✅ Colorblind-friendly color palettes")
    print("  ✅ Publication-ready typography")
    print("  ✅ Professional mathematical notation")
    print("  ✅ Statistical annotations and legends")
    print("  ✅ High-resolution output (300 DPI)")

if __name__ == "__main__":
    main()