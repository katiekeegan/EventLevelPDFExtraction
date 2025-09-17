#!/usr/bin/env python3
"""
Test script for enhanced uncertainty modes in plotting functions.
"""

import torch
import numpy as np
import os
import tempfile
from simulator import SimplifiedDIS
from uq_plotting_demo import plot_parameter_uncertainty, plot_function_uncertainty

def test_uncertainty_modes():
    """Test all uncertainty modes work correctly."""
    print("🧪 Testing enhanced uncertainty modes...")
    
    # Set up test environment
    device = torch.device('cpu')  # Use CPU for testing
    true_params = torch.tensor([1.5, 2.0, 1.2, 1.8], device=device)
    
    # Create simplified models for testing (mock implementations)
    class MockPointNet(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 64)
    
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(64, 4)
        
        def forward(self, x):
            return self.fc(x) + torch.randn(x.shape[0], 4) * 0.1
    
    class MockLaplaceModel:
        def __call__(self, x, joint=False):
            # Mock Laplace posterior sampling
            if joint:
                # Return mean and covariance
                mean = true_params + torch.randn(4) * 0.1
                cov = torch.eye(4) * 0.1
                return mean, cov
            else:
                # Just return mean
                return true_params + torch.randn(4) * 0.1
    
    # Initialize models
    pointnet_model = MockPointNet()
    model = MockModel()
    laplace_model = MockLaplaceModel()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"   Saving test plots to: {temp_dir}")
        
        # Test parameter uncertainty modes
        for mode in ['parameter', 'bootstrap', 'combined']:
            print(f"   Testing parameter uncertainty mode: {mode}")
            try:
                plot_parameter_uncertainty(
                    model=model,
                    pointnet_model=pointnet_model,
                    true_params=true_params,
                    device=device,
                    num_events=100,  # Small for fast testing
                    problem='simplified_dis',
                    save_dir=temp_dir,
                    n_mc=20,
                    laplace_model=laplace_model,
                    mode=mode,
                    n_bootstrap=5  # Small for fast testing
                )
                print(f"      ✅ Parameter uncertainty mode '{mode}' successful")
            except Exception as e:
                print(f"      ❌ Parameter uncertainty mode '{mode}' failed: {e}")
                return False
        
        # Test function uncertainty modes  
        for mode in ['parameter', 'bootstrap', 'combined']:
            print(f"   Testing function uncertainty mode: {mode}")
            try:
                plot_function_uncertainty(
                    model=model,
                    pointnet_model=pointnet_model,
                    true_params=true_params,
                    device=device,
                    num_events=100,  # Small for fast testing
                    problem='simplified_dis',
                    save_dir=temp_dir,
                    n_mc=20,
                    laplace_model=laplace_model,
                    mode=mode,
                    n_bootstrap=5  # Small for fast testing
                )
                print(f"      ✅ Function uncertainty mode '{mode}' successful")
            except Exception as e:
                print(f"      ❌ Function uncertainty mode '{mode}' failed: {e}")
                return False
        
        # Test invalid mode
        try:
            plot_parameter_uncertainty(
                model=model,
                pointnet_model=pointnet_model,
                true_params=true_params,
                device=device,
                mode='invalid_mode',
                save_dir=temp_dir
            )
            print("      ❌ Invalid mode should have raised ValueError")
            return False
        except ValueError:
            print("      ✅ Invalid mode correctly raises ValueError")
        except Exception as e:
            print(f"      ❌ Invalid mode raised unexpected error: {e}")
            return False
        
        # Test backward compatibility (default mode)
        try:
            plot_parameter_uncertainty(
                model=model,
                pointnet_model=pointnet_model,
                true_params=true_params,
                device=device,
                num_events=100,
                save_dir=temp_dir,
                laplace_model=laplace_model
                # mode parameter omitted - should default to 'parameter'
            )
            print("      ✅ Backward compatibility (default mode) successful")
        except Exception as e:
            print(f"      ❌ Backward compatibility failed: {e}")
            return False
    
    print("🎉 All uncertainty mode tests passed!")
    return True

if __name__ == "__main__":
    success = test_uncertainty_modes()
    exit(0 if success else 1)