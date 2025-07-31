#!/usr/bin/env python3
"""
Integration test for the NLL loss enhancement that tests the actual training loop logic
without requiring all the heavy dependencies.
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import InferenceNet
from utils import gaussian_nll_loss

def simulate_training_step(use_nll_loss=False, problem="simplified_dis"):
    """
    Simulate one training step to test the loss computation logic.
    """
    print(f"Testing training step with NLL loss: {use_nll_loss}, Problem: {problem}")
    
    # Parameters
    batch_size = 8
    embedding_dim = 128
    
    if problem == "simplified_dis":
        output_dim = 4
        param_mins = torch.tensor([0.0, 0.0, 0.0, 0.0])
        param_maxs = torch.tensor([5.0, 5.0, 5.0, 5.0])
    else:  # realistic_dis
        output_dim = 6
        param_mins = torch.tensor([-2.0, -1.0, 0.0, 0.0, -5.0, -5.0])
        param_maxs = torch.tensor([2.0, 1.0, 5.0, 10.0, 5.0, 5.0])
    
    # Create model and dummy data
    model = InferenceNet(embedding_dim, output_dim, nll_mode=use_nll_loss)
    latent_embeddings = torch.randn(batch_size, embedding_dim)
    true_params = torch.rand(batch_size, output_dim) * (param_maxs - param_mins) + param_mins
    
    # Forward pass
    if use_nll_loss:
        # NLL mode
        means, log_vars = model(latent_embeddings)
        assert means.shape == (batch_size, output_dim), f"Means shape mismatch: {means.shape}"
        assert log_vars.shape == (batch_size, output_dim), f"Log-vars shape mismatch: {log_vars.shape}"
        
        # Normalize for loss computation
        normalized_means = (means - param_mins) / (param_maxs - param_mins)
        normalized_true = (true_params - param_mins) / (param_maxs - param_mins)
        
        # Compute NLL loss
        loss = gaussian_nll_loss(normalized_means, log_vars, normalized_true)
        print(f"  NLL loss: {loss.item():.4f}")
        
    else:
        # MSE mode  
        pred_params = model(latent_embeddings)
        assert pred_params.shape == (batch_size, output_dim), f"Pred params shape mismatch: {pred_params.shape}"
        
        # Normalize for loss computation
        normalized_pred = (pred_params - param_mins) / (param_maxs - param_mins)
        normalized_true = (true_params - param_mins) / (param_maxs - param_mins)
        
        # Compute MSE loss
        loss = F.mse_loss(normalized_pred, normalized_true)
        print(f"  MSE loss: {loss.item():.4f}")
    
    # Test that loss is finite and can be backpropagated
    assert torch.isfinite(loss), "Loss should be finite"
    loss.backward()
    
    # Check that gradients were computed
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains non-finite values"
        # It's ok if some params don't have gradients (e.g., in one mode vs another)
    
    print(f"  ✓ Loss computation and backpropagation successful")
    return loss.item()


def test_consistency():
    """Test that the same input produces consistent results across runs."""
    print("Testing result consistency...")
    
    # Test parameters
    batch_size = 4
    embedding_dim = 64
    output_dim = 4
    
    # Create fixed input
    latent_embeddings = torch.randn(batch_size, embedding_dim)
    
    # Test MSE mode consistency
    model_mse = InferenceNet(embedding_dim, output_dim, nll_mode=False)
    model_mse.eval()  # Set to eval mode to disable dropout
    
    with torch.no_grad():
        out1 = model_mse(latent_embeddings)
        out2 = model_mse(latent_embeddings)
        assert torch.allclose(out1, out2), "MSE mode should be consistent"
    print("  ✓ MSE mode consistency verified")
    
    # Test NLL mode consistency
    model_nll = InferenceNet(embedding_dim, output_dim, nll_mode=True)
    model_nll.eval()  # Set to eval mode to disable dropout
    
    with torch.no_grad():
        means1, log_vars1 = model_nll(latent_embeddings)
        means2, log_vars2 = model_nll(latent_embeddings)
        assert torch.allclose(means1, means2), "NLL mode means should be consistent"
        assert torch.allclose(log_vars1, log_vars2), "NLL mode log-vars should be consistent"
    print("  ✓ NLL mode consistency verified")


def test_parameter_bounds():
    """Test that the parameter normalization works correctly for both problems."""
    print("Testing parameter normalization bounds...")
    
    # Test simplified_dis bounds
    param_mins = torch.tensor([0.0, 0.0, 0.0, 0.0])
    param_maxs = torch.tensor([5.0, 5.0, 5.0, 5.0])
    
    # Test edge cases
    test_params = torch.stack([param_mins, param_maxs, (param_mins + param_maxs) / 2])
    normalized = (test_params - param_mins) / (param_maxs - param_mins)
    
    expected = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0], 
                            [0.5, 0.5, 0.5, 0.5]])
    
    assert torch.allclose(normalized, expected), "Simplified DIS normalization failed"
    print("  ✓ Simplified DIS parameter normalization correct")
    
    # Test realistic_dis bounds
    param_mins = torch.tensor([-2.0, -1.0, 0.0, 0.0, -5.0, -5.0])
    param_maxs = torch.tensor([2.0, 1.0, 5.0, 10.0, 5.0, 5.0])
    
    test_params = torch.stack([param_mins, param_maxs])
    normalized = (test_params - param_mins) / (param_maxs - param_mins)
    
    expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    
    assert torch.allclose(normalized, expected), "Realistic DIS normalization failed"
    print("  ✓ Realistic DIS parameter normalization correct")


if __name__ == "__main__":
    print("="*70)
    print("Integration Test for NLL Loss Enhancement")
    print("="*70)
    
    try:
        # Test training steps for all combinations
        simulate_training_step(use_nll_loss=False, problem="simplified_dis")
        simulate_training_step(use_nll_loss=True, problem="simplified_dis") 
        simulate_training_step(use_nll_loss=False, problem="realistic_dis")
        simulate_training_step(use_nll_loss=True, problem="realistic_dis")
        print()
        
        test_consistency()
        print()
        
        test_parameter_bounds()
        print()
        
        print("="*70)
        print("All integration tests passed! ✓")
        print("The NLL loss enhancement is working correctly.")
        print("="*70)
        
    except Exception as e:
        print(f"Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)