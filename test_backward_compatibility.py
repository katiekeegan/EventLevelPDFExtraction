#!/usr/bin/env python3
"""
Test backward compatibility - ensure existing code works unchanged.
"""

import torch
import tempfile
import os
from uq_plotting_demo import plot_parameter_uncertainty, plot_function_uncertainty

def test_backward_compatibility():
    """Test that existing function calls work without mode parameter."""
    print("🔄 Testing backward compatibility...")
    
    # Mock setup
    device = torch.device('cpu')
    true_params = torch.tensor([1.5, 2.0, 1.2, 1.8], device=device)
    
    class MockPointNet(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 64)
    
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(64, 4)
        
        def forward(self, x):
            return self.fc(x)
    
    class MockLaplaceModel:
        def __call__(self, x, joint=False):
            if joint:
                return true_params, torch.eye(4) * 0.1
            return true_params
    
    pointnet_model = MockPointNet()
    model = MockModel()
    laplace_model = MockLaplaceModel()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test 1: Function call without mode parameter (should default to 'parameter')
        try:
            plot_parameter_uncertainty(
                model=model,
                pointnet_model=pointnet_model,
                true_params=true_params,
                device=device,
                num_events=50,
                save_dir=temp_dir,
                laplace_model=laplace_model
                # NO mode parameter - testing default behavior
            )
            print("   ✅ plot_parameter_uncertainty without mode parameter works")
        except Exception as e:
            print(f"   ❌ plot_parameter_uncertainty failed: {e}")
            return False
        
        # Test 2: Function uncertainty without mode parameter
        try:
            plot_function_uncertainty(
                model=model,
                pointnet_model=pointnet_model,
                true_params=true_params,
                device=device,
                num_events=50,
                save_dir=temp_dir,
                laplace_model=laplace_model
                # NO mode parameter - testing default behavior
            )
            print("   ✅ plot_function_uncertainty without mode parameter works")
        except Exception as e:
            print(f"   ❌ plot_function_uncertainty failed: {e}")
            return False
        
        # Verify default files are created
        param_file = os.path.join(temp_dir, "parameter_uncertainty_parameter.png")
        func_file = os.path.join(temp_dir, "function_uncertainty_parameter.png")
        
        if os.path.exists(param_file):
            print("   ✅ Default parameter uncertainty plot created")
        else:
            print("   ❌ Default parameter uncertainty plot not found")
            return False
            
        if os.path.exists(func_file):
            print("   ✅ Default function uncertainty plot created")
        else:
            print("   ❌ Default function uncertainty plot not found")  
            return False
    
    print("🎉 Backward compatibility test passed!")
    return True

if __name__ == "__main__":
    success = test_backward_compatibility()
    exit(0 if success else 1)