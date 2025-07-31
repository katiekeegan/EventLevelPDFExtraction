#!/usr/bin/env python3
"""
Simple test script to validate the NLL loss enhancement functionality.
Tests both MSE and NLL modes to ensure backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import InferenceNet
from utils import gaussian_nll_loss

def test_inference_net_modes():
    """Test that InferenceNet works in both MSE and NLL modes."""
    print("Testing InferenceNet modes...")
    
    # Test parameters
    batch_size = 4
    embedding_dim = 128
    output_dim = 6
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, embedding_dim)
    
    # Test MSE mode (original behavior)
    print("  Testing MSE mode...")
    mse_net = InferenceNet(embedding_dim, output_dim, nll_mode=False)
    mse_output = mse_net(dummy_input)
    assert mse_output.shape == (batch_size, output_dim), f"MSE mode output shape mismatch: {mse_output.shape}"
    print(f"    MSE mode output shape: {mse_output.shape} ✓")
    
    # Test NLL mode
    print("  Testing NLL mode...")
    nll_net = InferenceNet(embedding_dim, output_dim, nll_mode=True)
    means, log_vars = nll_net(dummy_input)
    assert means.shape == (batch_size, output_dim), f"NLL mode means shape mismatch: {means.shape}"
    assert log_vars.shape == (batch_size, output_dim), f"NLL mode log_vars shape mismatch: {log_vars.shape}"
    assert torch.all(log_vars >= -10) and torch.all(log_vars <= 10), "Log variances not properly clamped"
    print(f"    NLL mode means shape: {means.shape} ✓")
    print(f"    NLL mode log_vars shape: {log_vars.shape} ✓")
    print(f"    Log-variance range: [{log_vars.min():.2f}, {log_vars.max():.2f}] ✓")


def test_gaussian_nll_loss():
    """Test the Gaussian NLL loss function."""
    print("Testing Gaussian NLL loss...")
    
    # Create test data
    batch_size = 4
    output_dim = 6
    
    # Predictions: means and log-variances
    means = torch.randn(batch_size, output_dim)
    log_vars = torch.randn(batch_size, output_dim) * 0.5  # Small log-variances
    
    # Ground truth
    targets = torch.randn(batch_size, output_dim)
    
    # Compute loss
    loss = gaussian_nll_loss(means, log_vars, targets)
    
    assert loss.item() > 0, "NLL loss should be positive"
    assert torch.isfinite(loss), "NLL loss should be finite"
    print(f"    NLL loss value: {loss.item():.4f} ✓")
    
    # Test that loss decreases when predictions get better
    better_means = targets + torch.randn_like(targets) * 0.01  # Very close to targets
    better_loss = gaussian_nll_loss(better_means, log_vars, targets)
    assert better_loss < loss, "Loss should decrease with better predictions"
    print(f"    Better predictions give lower loss: {better_loss.item():.4f} < {loss.item():.4f} ✓")


def test_backward_compatibility():
    """Test that the new code doesn't break existing functionality."""
    print("Testing backward compatibility...")
    
    # Parameters matching typical usage
    embedding_dim = 1024
    output_dim = 4  # simplified_dis problem
    batch_size = 8
    
    # Create models in both modes
    old_style_net = InferenceNet(embedding_dim, output_dim, nll_mode=False)
    new_style_net = InferenceNet(embedding_dim, output_dim, nll_mode=True)
    
    # Test input
    dummy_input = torch.randn(batch_size, embedding_dim)
    true_params = torch.randn(batch_size, output_dim)
    
    # Test MSE mode still works
    pred_params = old_style_net(dummy_input)
    mse_loss = F.mse_loss(pred_params, true_params)
    assert torch.isfinite(mse_loss), "MSE loss should be finite"
    print(f"    MSE loss computes correctly: {mse_loss.item():.4f} ✓")
    
    # Test NLL mode works
    means, log_vars = new_style_net(dummy_input)
    nll_loss = gaussian_nll_loss(means, log_vars, true_params)
    assert torch.isfinite(nll_loss), "NLL loss should be finite"
    print(f"    NLL loss computes correctly: {nll_loss.item():.4f} ✓")


def test_parameter_ranges():
    """Test parameter ranges for different problems."""
    print("Testing parameter normalization ranges...")
    
    # Test parameter bounds
    simplified_mins = torch.tensor([0.0, 0.0, 0.0, 0.0])
    simplified_maxs = torch.tensor([5.0, 5.0, 5.0, 5.0])
    
    realistic_mins = torch.tensor([-2.0, -1.0, 0.0, 0.0, -5.0, -5.0])
    realistic_maxs = torch.tensor([2.0, 1.0, 5.0, 10.0, 5.0, 5.0])
    
    # Test that bounds are reasonable
    assert len(simplified_mins) == 4, "Simplified DIS should have 4 parameters"
    assert len(realistic_mins) == 6, "Realistic DIS should have 6 parameters"
    
    # Test normalization works
    test_params = torch.tensor([1.0, 2.0, 3.0, 4.0])
    normalized = (test_params - simplified_mins) / (simplified_maxs - simplified_mins)
    assert torch.all(normalized >= 0) and torch.all(normalized <= 1), "Normalized params should be in [0,1]"
    print(f"    Parameter normalization works correctly ✓")


if __name__ == "__main__":
    print("="*60)
    print("Testing NLL Loss Enhancement")
    print("="*60)
    
    try:
        test_inference_net_modes()
        print()
        
        test_gaussian_nll_loss()
        print()
        
        test_backward_compatibility()
        print()
        
        test_parameter_ranges()
        print()
        
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)