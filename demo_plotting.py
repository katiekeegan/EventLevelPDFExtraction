#!/usr/bin/env python3
"""
Demonstration script showing the new plotting_driver.py functionality.
This creates sample plots showing the difference between MSE and NLL modes.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def demo_plot_loss_curves(loss_dir='.', save_path='demo_loss_plot.png', nll_loss=False):
    """
    Demo version of plot_loss_curves with the new NLL support.
    """
    # Create sample loss data
    epochs = np.arange(1, 101)
    contrastive_loss = 2.0 * np.exp(-epochs/30) + 0.1 + 0.05 * np.random.rand(100)
    regression_loss = 1.5 * np.exp(-epochs/20) + 0.05 + 0.03 * np.random.rand(100)
    
    # Determine loss type labels - this is the key new functionality
    loss_type = "NLL" if nll_loss else "MSE"
    regression_label = f'Regression Loss ({loss_type}, scaled)'
    title = f'Training Loss Components Over Epochs ({loss_type} Regression)'
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, contrastive_loss, label='Contrastive Loss', linewidth=2, color='blue')
    plt.plot(epochs, regression_loss, label=regression_label, linewidth=2, color='orange')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.close()

def demo_model_architecture_detection():
    """Demonstrate the model architecture detection functionality."""
    
    # Mock model classes
    class MockMSEModel:
        def __init__(self):
            self.output_head = "Linear layer for direct parameter prediction"
    
    class MockNLLModel:
        def __init__(self):
            self.mean_head = "Linear layer for parameter means"
            self.log_var_head = "Linear layer for parameter log-variances"
    
    # Detection function (copied from plotting_driver.py)
    def detect_model_mode_mismatch(model, expected_nll_mode):
        has_mean_head = hasattr(model, 'mean_head')
        has_log_var_head = hasattr(model, 'log_var_head')
        has_output_head = hasattr(model, 'output_head')
        
        model_is_nll = has_mean_head and has_log_var_head and not has_output_head
        model_is_mse = has_output_head and not has_mean_head and not has_log_var_head
        
        if not (model_is_nll or model_is_mse):
            return False, "Unknown", "Unknown"
        
        detected_mode_str = "NLL" if model_is_nll else "MSE"
        expected_mode_str = "NLL" if expected_nll_mode else "MSE"
        
        is_mismatch = model_is_nll != expected_nll_mode
        
        return is_mismatch, detected_mode_str, expected_mode_str
    
    print("Model Architecture Detection Demo:")
    print("=" * 40)
    
    # Test scenarios
    scenarios = [
        (MockMSEModel(), False, "MSE model with MSE plotting mode"),
        (MockNLLModel(), True, "NLL model with NLL plotting mode"),
        (MockMSEModel(), True, "MSE model with NLL plotting mode (MISMATCH)"),
        (MockNLLModel(), False, "NLL model with MSE plotting mode (MISMATCH)"),
    ]
    
    for model, expected_nll, description in scenarios:
        is_mismatch, detected, expected = detect_model_mode_mismatch(model, expected_nll)
        
        print(f"\n{description}:")
        print(f"  Detected model type: {detected}")
        print(f"  Expected plotting mode: {expected}")
        
        if is_mismatch:
            print(f"  ⚠️  MISMATCH DETECTED!")
            print(f"     Consider using {'--nll-loss' if detected == 'NLL' else 'no --nll-loss flag'}")
        else:
            print(f"  ✓  No mismatch - labels will be accurate")

def demo_cli_modes():
    """Demonstrate different CLI usage modes."""
    
    print("\nCLI Usage Demonstration:")
    print("=" * 40)
    
    print("\n1. Default mode (MSE):")
    print("   $ python plotting_driver.py")
    print("   → Plots will show 'Regression Loss (MSE, scaled)'")
    
    print("\n2. NLL mode:")
    print("   $ python plotting_driver.py --nll-loss")
    print("   → Plots will show 'Regression Loss (NLL, scaled)'")
    
    print("\n3. Help:")
    print("   $ python plotting_driver.py --help")
    print("   → Shows usage information and examples")

def main():
    """Run the demonstration."""
    print("=" * 60)
    print("Plotting Driver NLL Support Demonstration")
    print("=" * 60)
    
    # Create output directory
    demo_dir = "/tmp/plotting_demo"
    os.makedirs(demo_dir, exist_ok=True)
    
    # Generate demo plots
    print("\nGenerating demo plots...")
    demo_plot_loss_curves(save_path=os.path.join(demo_dir, "loss_mse_mode.png"), nll_loss=False)
    demo_plot_loss_curves(save_path=os.path.join(demo_dir, "loss_nll_mode.png"), nll_loss=True)
    
    print("\nPlots generated:")
    print(f"  MSE mode: {demo_dir}/loss_mse_mode.png")
    print(f"  NLL mode: {demo_dir}/loss_nll_mode.png")
    
    # Demonstrate model detection
    demo_model_architecture_detection()
    
    # Show CLI usage
    demo_cli_modes()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)
    
    print("\nKey improvements implemented:")
    print("✓ Added --nll-loss CLI flag with argparse")
    print("✓ Dynamic plot labels based on loss mode")
    print("✓ Model architecture mismatch detection and warnings")
    print("✓ Backward compatibility (default MSE mode)")
    print("✓ Comprehensive help text and usage examples")
    print("✓ Logging of which loss mode is being visualized")

if __name__ == "__main__":
    main()