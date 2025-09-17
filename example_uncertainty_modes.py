#!/usr/bin/env python3
"""
Example demonstrating the enhanced uncertainty modes in plotting functions.

This script shows how to use the new 'mode' parameter to visualize different
types of uncertainty: parameter, bootstrap, and combined.
"""

import torch
import numpy as np
import os
from simulator import SimplifiedDIS
from uq_plotting_demo import plot_parameter_uncertainty, plot_function_uncertainty

def example_uncertainty_modes():
    """Demonstrate the three uncertainty modes with minimal working example."""
    print("🎯 Demonstrating Enhanced Uncertainty Modes")
    print("=" * 50)
    
    # Create example directory
    os.makedirs("uncertainty_mode_examples", exist_ok=True)
    
    # Example parameters
    device = torch.device('cpu')
    true_params = torch.tensor([1.5, 2.0, 1.2, 1.8], device=device)
    
    print("\n📋 Example Usage:")
    print("""
# 1. Parameter uncertainty (default - backward compatible)
plot_parameter_uncertainty(
    model=model, pointnet_model=pointnet_model, true_params=true_params,
    device=device, mode='parameter'  # or omit mode for default
)

# 2. Bootstrap uncertainty (data/aleatoric uncertainty)  
plot_parameter_uncertainty(
    model=model, pointnet_model=pointnet_model, true_params=true_params,
    device=device, mode='bootstrap', n_bootstrap=30
)

# 3. Combined uncertainty (parameter + bootstrap)
plot_parameter_uncertainty(
    model=model, pointnet_model=pointnet_model, true_params=true_params,
    device=device, mode='combined', n_bootstrap=30
)

# Same modes available for function uncertainty:
plot_function_uncertainty(..., mode='parameter')  # Default
plot_function_uncertainty(..., mode='bootstrap', n_bootstrap=30)
plot_function_uncertainty(..., mode='combined', n_bootstrap=30)
    """)
    
    print("\n🔍 Mode Descriptions:")
    print("• parameter:  Uses posterior samples from Laplace model (single dataset)")
    print("• bootstrap:  Generates B independent datasets, shows spread of estimates") 
    print("• combined:   Aggregates both bootstrap and parameter uncertainties")
    
    print("\n💾 Output files will be saved with mode-specific names:")
    print("• parameter_uncertainty_parameter.png")
    print("• parameter_uncertainty_bootstrap.png") 
    print("• parameter_uncertainty_combined.png")
    print("• function_uncertainty_parameter.png")
    print("• function_uncertainty_bootstrap.png")
    print("• function_uncertainty_combined.png")
    
    print("\n✅ The implementation maintains full backward compatibility:")
    print("   Existing code without 'mode' parameter will work unchanged!")
    
    print(f"\n📁 For a working example, see: {os.path.abspath('test_uncertainty_modes.py')}")

if __name__ == "__main__":
    example_uncertainty_modes()