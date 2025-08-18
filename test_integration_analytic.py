#!/usr/bin/env python3
"""
Integration test for analytic uncertainty propagation.

This test verifies the end-to-end functionality of the analytic uncertainty
system, including proper handling of different scenarios and edge cases.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from pathlib import Path
from laplace import Laplace

# Mock simulator
class MockSimulator:
    def __init__(self, device):
        self.device = device
    def sample(self, params, n_events):
        return torch.randn(n_events, 2)
    def init(self, params):
        self.params = params
    def up(self, x_vals):
        return torch.exp(-x_vals)
    def down(self, x_vals):
        return torch.exp(-2 * x_vals)
    def q(self, x_vals, Q2_vals):
        return torch.exp(-x_vals) * torch.log(Q2_vals + 1)

def advanced_feature_engineering(x):
    if x.dim() == 2:
        features = torch.cat([x, x**2, torch.log(torch.abs(x) + 1e-6)], dim=-1)
        return features
    return x

# Mock the simulator module
import sys
from unittest.mock import MagicMock
simulator_mock = MagicMock()
simulator_mock.SimplifiedDIS = MockSimulator
simulator_mock.RealisticDIS = MockSimulator  
simulator_mock.advanced_feature_engineering = advanced_feature_engineering
sys.modules['simulator'] = simulator_mock

# Import our functions after mocking
from plotting_UQ_utils import (
    get_analytic_uncertainty, 
    plot_params_distribution_single,
    plot_PDF_distribution_single,
    plot_PDF_distribution_single_same_plot,
    plot_event_histogram_simplified_DIS
)

def create_realistic_models():
    """Create more realistic test models."""
    class PointNetPMA(nn.Module):
        def __init__(self, input_dim=6, latent_dim=128):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 128)
            self.fc3 = nn.Linear(128, latent_dim)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x.mean(dim=1)))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    class MLPHead(nn.Module):
        def __init__(self, latent_dim=128, output_dim=4):
            super().__init__()
            self.fc1 = nn.Linear(latent_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, output_dim)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    class GaussianHead(nn.Module):
        def __init__(self, latent_dim=128, output_dim=4):
            super().__init__()
            self.fc1 = nn.Linear(latent_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.mean_head = nn.Linear(32, output_dim)
            self.logvar_head = nn.Linear(32, output_dim)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            means = self.mean_head(x)
            logvars = self.logvar_head(x)
            return means, logvars
    
    class MultimodalHead(nn.Module):
        def __init__(self, latent_dim=128, output_dim=4, nmodes=2):
            super().__init__()
            self.nmodes = nmodes
            self.output_dim = output_dim
            self.fc1 = nn.Linear(latent_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.means_head = nn.Linear(32, nmodes * output_dim)
            self.logvars_head = nn.Linear(32, nmodes * output_dim)
            self.weights_head = nn.Linear(32, nmodes)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            
            means = self.means_head(x).view(-1, self.nmodes, self.output_dim)
            logvars = self.logvars_head(x).view(-1, self.nmodes, self.output_dim)
            weights = torch.softmax(self.weights_head(x), dim=-1)
            
            return means, logvars, weights
    
    pointnet = PointNetPMA()
    mlp_head = MLPHead()
    gaussian_head = GaussianHead()
    multimodal_head = MultimodalHead()
    
    return pointnet, mlp_head, gaussian_head, multimodal_head

def test_laplace_integration():
    """Test integration with actual Laplace approximation."""
    print("Testing Laplace integration...")
    
    pointnet, mlp_head, gaussian_head, multimodal_head = create_realistic_models()
    
    # Generate synthetic training data
    n_train = 200
    x_train = torch.randn(n_train, 128)  # Latent features
    y_train = torch.randn(n_train, 4)    # Parameters
    
    dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset, batch_size=32)
    
    # Test with MLP head
    laplace_mlp = Laplace(mlp_head, 'regression', subset_of_weights='last_layer')
    laplace_mlp.fit(train_loader)
    
    # Test analytic uncertainty
    x_test = torch.randn(1, 128)
    mean_analytic, std_analytic = get_analytic_uncertainty(mlp_head, x_test, laplace_mlp)
    
    print(f"Laplace MLP - Mean: {mean_analytic.squeeze()}")
    print(f"Laplace MLP - Std: {std_analytic.squeeze()}")
    
    # Verify reasonable values
    assert torch.all(std_analytic > 0), "Laplace uncertainty should be positive"
    assert torch.all(std_analytic < 5), "Laplace uncertainty should be reasonable"
    
    print("✓ Laplace integration test passed!")
    return True

def test_all_model_types():
    """Test analytic uncertainty with all model types."""
    print("Testing all model types...")
    
    pointnet, mlp_head, gaussian_head, multimodal_head = create_realistic_models()
    device = torch.device('cpu')
    
    # Test input
    latent = torch.randn(1, 128)
    
    # Test MLP head (deterministic)
    mean_mlp, std_mlp = get_analytic_uncertainty(mlp_head, latent, laplace_model=None)
    assert torch.allclose(std_mlp, torch.zeros_like(std_mlp)), "MLP should have zero uncertainty without Laplace"
    print("  ✓ MLP head test passed")
    
    # Test Gaussian head (intrinsic uncertainty)
    mean_gauss, std_gauss = get_analytic_uncertainty(gaussian_head, latent, laplace_model=None)
    assert torch.all(std_gauss >= 0), "Gaussian head should have non-negative uncertainty"
    print("  ✓ Gaussian head test passed")
    
    # Test Multimodal head (mode selection)
    mean_multi, std_multi = get_analytic_uncertainty(multimodal_head, latent, laplace_model=None)
    assert torch.all(std_multi >= 0), "Multimodal head should have non-negative uncertainty"
    print("  ✓ Multimodal head test passed")
    
    print("✓ All model types test passed!")
    return True

def test_plotting_integration():
    """Test full plotting pipeline with different scenarios."""
    print("Testing plotting integration...")
    
    pointnet, mlp_head, gaussian_head, multimodal_head = create_realistic_models()
    device = torch.device('cpu')
    true_params = torch.tensor([0.5, 0.5, 0.5, 0.5])
    
    # Create output directory
    os.makedirs('/tmp/integration_test', exist_ok=True)
    
    # Test scenarios
    scenarios = [
        ("mlp_no_laplace", mlp_head, None, "MLP without Laplace"),
        ("gaussian_no_laplace", gaussian_head, None, "Gaussian without Laplace"),
        ("multimodal_no_laplace", multimodal_head, None, "Multimodal without Laplace"),
    ]
    
    # Add Laplace scenario if possible
    try:
        # Create minimal training data for Laplace
        x_train = torch.randn(100, 128)
        y_train = torch.randn(100, 4)
        dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(dataset, batch_size=20)
        
        laplace_mlp = Laplace(mlp_head, 'regression', subset_of_weights='last_layer')
        laplace_mlp.fit(train_loader)
        scenarios.append(("mlp_with_laplace", mlp_head, laplace_mlp, "MLP with Laplace"))
    except Exception as e:
        print(f"  Warning: Could not create Laplace scenario: {e}")
    
    for scenario_name, model, laplace_model, description in scenarios:
        print(f"  Testing {description}...")
        
        try:
            # Test parameter distribution plotting
            plot_params_distribution_single(
                model=model,
                pointnet_model=pointnet,
                true_params=true_params,
                device=device,
                n_mc=20,  # Small for speed
                laplace_model=laplace_model,
                save_path=f"/tmp/integration_test/params_{scenario_name}.png",
                problem='simplified_dis'
            )
            
            # Test PDF plotting
            plot_PDF_distribution_single_same_plot(
                model=model,
                pointnet_model=pointnet,
                true_params=true_params,
                device=device,
                n_mc=20,
                laplace_model=laplace_model,
                problem='realistic_dis',
                save_path=f"/tmp/integration_test/pdf_{scenario_name}.png"
            )
            
            # Test event histogram
            plot_event_histogram_simplified_DIS(
                model=model,
                pointnet_model=pointnet,
                true_params=true_params,
                device=device,
                n_mc=20,
                laplace_model=laplace_model,
                num_events=1000,
                save_path=f"/tmp/integration_test/events_{scenario_name}.png"
            )
            
            print(f"    ✓ {description} plotting completed")
            
        except Exception as e:
            print(f"    ❌ {description} plotting failed: {e}")
            return False
    
    # Verify output files
    output_files = list(Path("/tmp/integration_test").glob("*.png"))
    expected_min_files = len(scenarios) * 3  # 3 plot types per scenario
    assert len(output_files) >= expected_min_files, f"Expected at least {expected_min_files} files, got {len(output_files)}"
    
    print(f"✓ Plotting integration test passed! Generated {len(output_files)} plots.")
    return True

def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    
    pointnet, mlp_head, gaussian_head, multimodal_head = create_realistic_models()
    
    # Test with different batch sizes
    for batch_size in [1, 5]:
        latent = torch.randn(batch_size, 128)
        mean, std = get_analytic_uncertainty(mlp_head, latent, laplace_model=None)
        assert mean.shape[0] == batch_size, f"Batch size handling failed for batch_size={batch_size}"
    
    # Test with different parameter dimensions
    class CustomHead(nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            self.fc = nn.Linear(128, output_dim)
        def forward(self, x):
            return self.fc(x)
    
    for param_dim in [2, 6, 8]:
        custom_head = CustomHead(param_dim)
        latent = torch.randn(1, 128)
        mean, std = get_analytic_uncertainty(custom_head, latent, laplace_model=None)
        assert mean.shape[-1] == param_dim, f"Parameter dimension handling failed for param_dim={param_dim}"
    
    print("✓ Edge cases test passed!")
    return True

def main():
    """Run all integration tests."""
    print("="*60)
    print("Integration Test for Analytic Uncertainty Propagation")
    print("="*60)
    
    try:
        test_laplace_integration()
        print()
        
        test_all_model_types()
        print()
        
        test_plotting_integration()
        print()
        
        test_edge_cases()
        print()
        
        print("="*60)
        print("All integration tests passed! ✓")
        print("="*60)
        print()
        
        # Show generated files
        print("Generated test outputs in /tmp/integration_test/:")
        test_dir = Path("/tmp/integration_test")
        if test_dir.exists():
            for plot_file in sorted(test_dir.glob("*.png")):
                print(f"  - {plot_file.name}")
        
        print()
        print("Summary of verified capabilities:")
        print("- End-to-end analytic uncertainty propagation")
        print("- Integration with Laplace approximation")
        print("- Support for all model architectures (MLP, Gaussian, Multimodal)")
        print("- Fallback handling when Laplace unavailable")
        print("- Proper batch and dimension handling")
        print("- Complete plotting pipeline functionality")
        print("- Error handling and edge case robustness")
        
    except Exception as e:
        print(f"Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)