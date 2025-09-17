#!/usr/bin/env python3
"""
Visual demonstration of uncertainty modes - creates actual plots to show differences.
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from simulator import SimplifiedDIS
from uq_plotting_demo import plot_parameter_uncertainty, plot_function_uncertainty

def create_visual_demo():
    """Create visual demonstration plots showing the difference between modes."""
    print("📸 Creating visual demonstration of uncertainty modes...")
    
    # Create output directory
    demo_dir = "uncertainty_modes_demo"
    os.makedirs(demo_dir, exist_ok=True)
    
    # Set up realistic test environment
    device = torch.device('cpu')
    true_params = torch.tensor([1.5, 2.0, 1.2, 1.8], device=device)
    
    # Create more realistic mock models
    class MockPointNet(torch.nn.Module):
        def forward(self, x):
            # Add some meaningful variation based on input
            batch_size = x.shape[0]
            features = torch.randn(batch_size, 64)
            # Add some input dependency
            if x.ndim > 1:
                mean_features = torch.mean(x.view(batch_size, -1), dim=1, keepdim=True)
                features = features + 0.1 * mean_features.expand(-1, 64)
            return features
    
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(64, 32)
            self.fc2 = torch.nn.Linear(32, 4)
            self.activation = torch.nn.ReLU()
        
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.fc2(x)
            # Add some realistic parameter-dependent noise
            noise = torch.randn_like(x) * 0.15
            return x + noise
    
    class MockLaplaceModel:
        def __call__(self, x, joint=False):
            batch_size = x.shape[0] if hasattr(x, 'shape') else 1
            # Create more realistic parameter-dependent behavior
            if joint:
                # Vary mean slightly based on input features
                base_mean = true_params.clone()
                if x.ndim > 1:
                    feature_effect = torch.mean(x) * 0.1
                    base_mean = base_mean + feature_effect
                
                # Realistic covariance with some correlations
                cov = torch.tensor([
                    [0.15, 0.02, 0.01, 0.005],
                    [0.02, 0.20, 0.015, 0.01],
                    [0.01, 0.015, 0.12, 0.008],
                    [0.005, 0.01, 0.008, 0.18]
                ])
                return base_mean, cov
            else:
                return true_params + torch.randn(4) * 0.1
    
    # Initialize models with some trained-like weights
    pointnet_model = MockPointNet()
    model = MockModel()
    laplace_model = MockLaplaceModel()
    
    # Set models to eval mode
    pointnet_model.eval()
    model.eval()
    
    # Create plots for each mode
    modes_to_test = ['parameter', 'bootstrap', 'combined']
    
    print(f"   Saving demonstration plots to: {os.path.abspath(demo_dir)}")
    
    for mode in modes_to_test:
        print(f"   Creating {mode} mode plots...")
        
        # Create parameter uncertainty plot
        plot_parameter_uncertainty(
            model=model,
            pointnet_model=pointnet_model,
            true_params=true_params,
            device=device,
            num_events=200,  # Moderate size for realistic demo
            problem='simplified_dis',
            save_dir=demo_dir,
            n_mc=50,  # More samples for smoother distributions
            laplace_model=laplace_model,
            mode=mode,
            n_bootstrap=15  # Moderate bootstrap size
        )
        
        # Create function uncertainty plot
        plot_function_uncertainty(
            model=model,
            pointnet_model=pointnet_model,
            true_params=true_params,
            device=device,
            num_events=200,
            problem='simplified_dis',
            save_dir=demo_dir,
            n_mc=50,
            laplace_model=laplace_model,
            mode=mode,
            n_bootstrap=15
        )
    
    print(f"\n✅ Visual demonstration complete!")
    print(f"📁 Check the '{demo_dir}' directory for the generated plots:")
    
    # List generated files
    for mode in modes_to_test:
        param_file = f"parameter_uncertainty_{mode}.png"
        func_file = f"function_uncertainty_{mode}.png"
        print(f"   • {param_file}")
        print(f"   • {func_file}")
    
    print(f"\n🔍 Compare the plots to see how different uncertainty modes show:")
    print("   • Parameter mode: Posterior uncertainty from single dataset")
    print("   • Bootstrap mode: Variation across multiple datasets") 
    print("   • Combined mode: Total uncertainty (wider distributions)")

if __name__ == "__main__":
    create_visual_demo()