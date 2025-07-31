#!/usr/bin/env python3
"""
Test the command-line argument parsing for the --nll-loss flag.
"""

import argparse
import sys

def test_argparse():
    """Test that the new --nll-loss argument is properly parsed."""
    
    # Create the same parser as in PDF_learning.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--gpus", type=int, default=1)  # Use 1 instead of torch.cuda.device_count()
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument(
        "--problem",
        type=str,
        default="simplified_dis",
        choices=["simplified_dis", "realistic_dis"],
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_events", type=int, default=100000)
    parser.add_argument("--nll-loss", action="store_true", 
                       help="Use Gaussian negative log-likelihood loss with mean and variance prediction")
    
    # Test default behavior (MSE mode)
    print("Testing default behavior (MSE mode)...")
    args_mse = parser.parse_args([])
    assert not getattr(args_mse, 'nll_loss', False), "Default should be MSE mode"
    print(f"  nll_loss flag: {getattr(args_mse, 'nll_loss', False)} ✓")
    
    # Test with --nll-loss flag
    print("Testing with --nll-loss flag...")
    args_nll = parser.parse_args(["--nll-loss"])
    assert getattr(args_nll, 'nll_loss', False), "Should be True when flag is set"
    print(f"  nll_loss flag: {getattr(args_nll, 'nll_loss', False)} ✓")
    
    # Test with other arguments
    print("Testing with mixed arguments...")
    args_mixed = parser.parse_args(["--nll-loss", "--epochs", "500", "--problem", "realistic_dis"])
    assert getattr(args_mixed, 'nll_loss', False), "nll_loss should be True"
    assert args_mixed.epochs == 500, "epochs should be 500"
    assert args_mixed.problem == "realistic_dis", "problem should be realistic_dis"
    print(f"  nll_loss flag: {getattr(args_mixed, 'nll_loss', False)} ✓")
    print(f"  epochs: {args_mixed.epochs} ✓")
    print(f"  problem: {args_mixed.problem} ✓")

if __name__ == "__main__":
    print("="*60)
    print("Testing Command-Line Argument Parsing")
    print("="*60)
    
    try:
        test_argparse()
        print()
        print("="*60)
        print("Command-line parsing tests passed! ✓")  
        print("="*60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)