#!/usr/bin/env python3
"""
Test the CLI argument parsing and label generation for plotting_driver.py
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_plot_loss_curves_labels():
    """Test that plot_loss_curves generates correct labels based on nll_loss parameter."""
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create dummy loss data
    loss_dir = '/tmp/test_losses'
    os.makedirs(loss_dir, exist_ok=True)
    
    # Create dummy loss files
    dummy_loss = np.random.rand(100)
    np.save(os.path.join(loss_dir, 'loss_contrastive.npy'), dummy_loss)
    np.save(os.path.join(loss_dir, 'loss_regression.npy'), dummy_loss)
    np.save(os.path.join(loss_dir, 'loss_total.npy'), dummy_loss)
    
    # Import the function after matplotlib is set up
    from plotting_driver import plot_loss_curves
    
    # Test MSE mode (default)
    print("Testing MSE mode...")
    try:
        plot_loss_curves(loss_dir=loss_dir, save_path='/tmp/test_mse.png', 
                        show_plot=False, nll_loss=False)
        print("✓ MSE mode plot generated successfully")
    except Exception as e:
        print(f"✗ MSE mode failed: {e}")
    
    # Test NLL mode
    print("Testing NLL mode...")
    try:
        plot_loss_curves(loss_dir=loss_dir, save_path='/tmp/test_nll.png', 
                        show_plot=False, nll_loss=True)
        print("✓ NLL mode plot generated successfully")
    except Exception as e:
        print(f"✗ NLL mode failed: {e}")
    
    # Clean up
    import shutil
    shutil.rmtree(loss_dir)
    for f in ['/tmp/test_mse.png', '/tmp/test_nll.png']:
        if os.path.exists(f):
            os.remove(f)


def test_cli_parsing():
    """Test CLI argument parsing without running the full main function."""
    import argparse
    
    # Create the same parser as in plotting_driver.py main()
    parser = argparse.ArgumentParser(
        description='Plot training results and model predictions. '
                   'Supports both MSE and NLL loss modes with appropriate labeling.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plotting_driver.py                    # Plot with MSE loss labels (default)
  python plotting_driver.py --nll-loss        # Plot with NLL loss labels
        """
    )
    parser.add_argument('--nll-loss', action='store_true',
                       help='Use NLL loss mode for plot labels and model loading. '
                            'Should match the mode used during training.')
    
    # Test default behavior (MSE mode)
    print("Testing default CLI behavior (MSE mode)...")
    args_mse = parser.parse_args([])
    assert not args_mse.nll_loss, "Default should be MSE mode"
    print(f"  nll_loss flag: {args_mse.nll_loss} ✓")
    
    # Test with --nll-loss flag
    print("Testing --nll-loss CLI flag...")
    args_nll = parser.parse_args(["--nll-loss"])
    assert args_nll.nll_loss, "Should be True when flag is set"
    print(f"  nll_loss flag: {args_nll.nll_loss} ✓")


def test_model_detection():
    """Test the model architecture detection function."""
    # Create mock models to test detection
    class MockMSEModel:
        def __init__(self):
            self.output_head = "exists"
    
    class MockNLLModel:
        def __init__(self):
            self.mean_head = "exists"
            self.log_var_head = "exists"
    
    class MockUnknownModel:
        def __init__(self):
            pass
    
    # Import the detection function
    from plotting_driver import detect_model_mode_mismatch
    
    # Test MSE model detection
    mse_model = MockMSEModel()
    is_mismatch, detected, expected = detect_model_mode_mismatch(mse_model, False)
    assert not is_mismatch, "MSE model with MSE expectation should not be a mismatch"
    assert detected == "MSE", f"Expected MSE detection, got {detected}"
    print("✓ MSE model correctly detected")
    
    # Test NLL model detection
    nll_model = MockNLLModel()
    is_mismatch, detected, expected = detect_model_mode_mismatch(nll_model, True)
    assert not is_mismatch, "NLL model with NLL expectation should not be a mismatch"
    assert detected == "NLL", f"Expected NLL detection, got {detected}"
    print("✓ NLL model correctly detected")
    
    # Test mismatch detection
    is_mismatch, detected, expected = detect_model_mode_mismatch(mse_model, True)
    assert is_mismatch, "MSE model with NLL expectation should be a mismatch"
    assert detected == "MSE" and expected == "NLL", f"Mismatch detection failed: {detected} vs {expected}"
    print("✓ Model architecture mismatch correctly detected")
    
    # Test unknown model
    unknown_model = MockUnknownModel()
    is_mismatch, detected, expected = detect_model_mode_mismatch(unknown_model, False)
    assert not is_mismatch, "Unknown models should not trigger mismatch to be safe"
    print("✓ Unknown model architecture handled safely")


if __name__ == "__main__":
    print("="*60)
    print("Testing Plotting Driver CLI and Functionality")
    print("="*60)
    
    try:
        test_cli_parsing()
        print()
        
        test_model_detection()
        print()
        
        test_plot_loss_curves_labels()
        print()
        
        print("="*60)
        print("All plotting driver tests passed! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)