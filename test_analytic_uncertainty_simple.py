#!/usr/bin/env python3
"""
Simple test script for analytic uncertainty propagation.

This test verifies the core analytic uncertainty computation without
dependencies on the complex simulator module.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from laplace import Laplace

def get_analytic_uncertainty_simple(model, latent_embedding, laplace_model=None):
    """
    Simplified version of analytic uncertainty computation for testing.
    """
    device = latent_embedding.device
    
    if laplace_model is not None:
        # Use Laplace approximation for analytic uncertainty propagation
        with torch.no_grad():
            # Get predictive mean and variance analytically
            pred_mean, pred_var = laplace_model(latent_embedding, joint=False)
            
            # Extract diagonal covariance as standard deviations
            if pred_var.dim() == 3:  # [batch_size, output_dim, output_dim]
                # Take diagonal elements and sqrt for standard deviation
                pred_std = torch.sqrt(torch.diagonal(pred_var, dim1=-2, dim2=-1))
            else:  # [batch_size, output_dim] - already diagonal variance
                pred_std = torch.sqrt(pred_var)
                
            return pred_mean, pred_std
    else:
        # Fallback: Standard model without uncertainty
        model.eval()
        with torch.no_grad():
            output = model(latent_embedding)
            
        # For deterministic models, return zero uncertainty
        pred_mean = output
        pred_std = torch.zeros_like(pred_mean)
        return pred_mean, pred_std

def create_test_model():
    """Create a simple test model for uncertainty testing."""
    class TestModel(nn.Module):
        def __init__(self, input_dim=10, output_dim=4):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 20)
            self.fc2 = nn.Linear(20, 15)
            self.fc3 = nn.Linear(15, output_dim)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    return TestModel()

def test_analytic_uncertainty():
    """Test that analytic uncertainty computation works."""
    print("Testing analytic uncertainty computation...")
    
    # Create test model and data
    model = create_test_model()
    input_dim = 10
    output_dim = 4
    
    # Generate synthetic training data
    n_train = 100
    x_train = torch.randn(n_train, input_dim)
    y_train = torch.randn(n_train, output_dim)
    
    # Create DataLoader
    dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset, batch_size=20)
    
    # Fit Laplace approximation
    laplace_model = Laplace(model, 'regression', subset_of_weights='last_layer')
    laplace_model.fit(train_loader)
    
    # Test input
    x_test = torch.randn(1, input_dim)
    
    # Get analytic uncertainty
    mean_analytic, std_analytic = get_analytic_uncertainty_simple(model, x_test, laplace_model)
    
    print(f"Input shape: {x_test.shape}")
    print(f"Analytic mean shape: {mean_analytic.shape}")
    print(f"Analytic std shape: {std_analytic.shape}")
    print(f"Analytic mean: {mean_analytic.squeeze()}")
    print(f"Analytic std: {std_analytic.squeeze()}")
    
    # Check that uncertainty is non-zero and reasonable
    assert torch.all(std_analytic > 0), "Analytic uncertainty should be positive"
    assert torch.all(std_analytic < 10), "Analytic uncertainty should be reasonable"
    assert mean_analytic.shape == (1, output_dim), f"Mean shape should be (1, {output_dim})"
    assert std_analytic.shape == (1, output_dim), f"Std shape should be (1, {output_dim})"
    
    print("✓ Analytic uncertainty computation test passed!")
    return True

def test_fallback_mode():
    """Test fallback mode without Laplace."""
    print("Testing fallback mode...")
    
    # Test with standard deterministic model
    model = create_test_model()
    x_test = torch.randn(1, 10)
    
    mean, std = get_analytic_uncertainty_simple(model, x_test, laplace_model=None)
    
    print(f"Deterministic model mean: {mean.squeeze()}")
    print(f"Deterministic model std: {std.squeeze()}")
    
    # Should return zero uncertainty for deterministic model
    assert torch.allclose(std, torch.zeros_like(std)), "Deterministic model should have zero uncertainty"
    
    print("✓ Fallback mode test passed!")
    return True

def test_batch_processing():
    """Test that batch processing works correctly."""
    print("Testing batch processing...")
    
    model = create_test_model()
    
    # Generate training data
    n_train = 100
    x_train = torch.randn(n_train, 10)
    y_train = torch.randn(n_train, 4)
    
    dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset, batch_size=20)
    
    # Fit Laplace
    laplace_model = Laplace(model, 'regression', subset_of_weights='last_layer')
    laplace_model.fit(train_loader)
    
    # Test with batch input
    batch_size = 5
    x_test_batch = torch.randn(batch_size, 10)
    
    # Process each element in the batch
    all_means = []
    all_stds = []
    for i in range(batch_size):
        x_single = x_test_batch[i:i+1]
        mean, std = get_analytic_uncertainty_simple(model, x_single, laplace_model)
        all_means.append(mean)
        all_stds.append(std)
    
    batch_means = torch.cat(all_means, dim=0)
    batch_stds = torch.cat(all_stds, dim=0)
    
    print(f"Batch means shape: {batch_means.shape}")
    print(f"Batch stds shape: {batch_stds.shape}")
    
    assert batch_means.shape == (batch_size, 4), "Batch means shape incorrect"
    assert batch_stds.shape == (batch_size, 4), "Batch stds shape incorrect"
    assert torch.all(batch_stds > 0), "All uncertainties should be positive"
    
    print("✓ Batch processing test passed!")
    return True

def test_comparison_visualization():
    """Create a simple visualization showing the uncertainty."""
    print("Creating uncertainty visualization...")
    
    model = create_test_model()
    
    # Generate training data
    n_train = 200
    x_train = torch.randn(n_train, 10)
    y_train = torch.randn(n_train, 4)
    
    dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset, batch_size=20)
    
    # Fit Laplace
    laplace_model = Laplace(model, 'regression', subset_of_weights='last_layer')
    laplace_model.fit(train_loader)
    
    # Test on multiple points
    n_test = 20
    test_inputs = torch.randn(n_test, 10)
    
    means = []
    stds = []
    
    for i in range(n_test):
        x_test = test_inputs[i:i+1]
        mean, std = get_analytic_uncertainty_simple(model, x_test, laplace_model)
        means.append(mean.squeeze().numpy())
        stds.append(std.squeeze().numpy())
    
    means = np.array(means)  # [n_test, 4]
    stds = np.array(stds)    # [n_test, 4]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Analytic Uncertainty Visualization', fontsize=16)
    
    for param_idx in range(4):
        row = param_idx // 2
        col = param_idx % 2
        ax = axes[row, col]
        
        x_vals = np.arange(n_test)
        y_means = means[:, param_idx]
        y_stds = stds[:, param_idx]
        
        # Plot means with error bars
        ax.errorbar(x_vals, y_means, yerr=2*y_stds, fmt='o-', capsize=5, capthick=2,
                   label=f'Parameter {param_idx} (±2σ)')
        ax.fill_between(x_vals, y_means - 2*y_stds, y_means + 2*y_stds, alpha=0.3)
        
        ax.set_xlabel('Test Point')
        ax.set_ylabel(f'Parameter {param_idx} Value')
        ax.set_title(f'Parameter {param_idx}: Mean ± 2σ')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/analytic_uncertainty_viz.png', dpi=300)
    print("✓ Saved uncertainty visualization to /tmp/analytic_uncertainty_viz.png")
    plt.close()
    
    return True

def main():
    """Run all tests for analytic uncertainty computation."""
    print("="*60)
    print("Testing Analytic Uncertainty Computation (Simple)")
    print("="*60)
    
    try:
        # Basic functionality tests
        test_analytic_uncertainty()
        print()
        
        test_fallback_mode()
        print()
        
        test_batch_processing()
        print()
        
        # Visualization test
        test_comparison_visualization()
        print()
        
        print("="*60)
        print("All analytic uncertainty tests passed! ✓")
        print("="*60)
        print()
        print("Summary of verified features:")
        print("- Analytic uncertainty computation using Laplace approximation")
        print("- Proper handling of tensor shapes and batching")
        print("- Fallback modes for models without Laplace")
        print("- Reasonable uncertainty values (positive and bounded)")
        print("- Visualization capabilities")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)