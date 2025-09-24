#!/usr/bin/env python3
"""
Example usage script for the combined uncertainty plotting function.

This script demonstrates how to integrate plot_combined_uncertainty_PDF_distribution
into the existing CLI workflow and plotting drivers.

Usage examples:
    # Basic usage with simplified DIS
    python example_combined_uncertainty_usage.py --problem simplified_dis --arch mlp
    
    # With custom parameters
    python example_combined_uncertainty_usage.py --problem realistic_dis --arch transformer --n_bootstrap 100
    
    # Test mode (using mock data)
    python example_combined_uncertainty_usage.py --test_mode
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the new function and existing utilities
from plotting_UQ_utils import (
    plot_combined_uncertainty_PDF_distribution,
    plot_uncertainty_decomposition_comparison, 
    validate_combined_uncertainty_inputs
)

# Existing imports for integration (commented out since we can't test with real models)
# from plotting_driver_UQ import reload_model, reload_pointnet
# from simulator import SimplifiedDIS, RealisticDIS, MCEGSimulator


def create_example_cli_integration():
    """
    Example of how to integrate combined uncertainty into existing CLI drivers.
    
    This function shows the integration pattern that would be added to 
    plotting_driver_UQ.py or similar existing CLI scripts.
    """
    
    parser = argparse.ArgumentParser(
        description="Combined Uncertainty Analysis for PDF Parameter Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Simplified DIS with MLP architecture
    python {script} --problem simplified_dis --arch mlp --n_bootstrap 50
    
    # Realistic DIS with Transformer and custom parameters
    python {script} --problem realistic_dis --arch transformer --n_bootstrap 30 --num_events 200000
    
    # Quick test with fewer bootstrap samples
    python {script} --problem simplified_dis --arch mlp --n_bootstrap 10 --quick_test
        """.format(script=os.path.basename(__file__))
    )
    
    # Core arguments
    parser.add_argument('--problem', type=str, default='simplified_dis',
                       choices=['simplified_dis', 'realistic_dis', 'mceg', 'mceg4dis'],
                       help='Problem type for analysis')
    
    parser.add_argument('--arch', type=str, default='mlp',
                       choices=['mlp', 'transformer', 'gaussian', 'multimodal'],
                       help='Model architecture to analyze')
    
    # Uncertainty analysis parameters
    parser.add_argument('--n_bootstrap', type=int, default=50,
                       help='Number of bootstrap samples for data uncertainty')
    
    parser.add_argument('--num_events', type=int, default=100000,
                       help='Number of events per bootstrap sample')
    
    # Model and data parameters
    parser.add_argument('--latent_dim', type=int, default=1024,
                       help='Latent dimension of PointNet embeddings')
    
    parser.add_argument('--param_dim', type=int, default=None,
                       help='Parameter dimension (auto-detected from problem if not specified)')
    
    # True parameters for analysis
    parser.add_argument('--true_params', type=float, nargs='+', default=None,
                       help='True parameter values for analysis (problem-specific defaults used if not specified)')
    
    # Q2 slices for realistic DIS
    parser.add_argument('--Q2_slices', type=float, nargs='+', default=None,
                       help='Q2 values for realistic DIS analysis (default: [2.0, 10.0, 50.0, 200.0])')
    
    # Output options
    parser.add_argument('--save_dir', type=str, default='./plots/combined_uncertainty',
                       help='Directory to save combined uncertainty plots')
    
    parser.add_argument('--experiment_dir', type=str, default='./experiments',
                       help='Directory containing trained models and Laplace approximations')
    
    # Testing and debugging
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode with mock models (for development)')
    
    parser.add_argument('--quick_test', action='store_true',
                       help='Use reduced parameters for quick testing')
    
    parser.add_argument('--no_laplace', action='store_true',
                       help='Skip Laplace model loading (bootstrap-only uncertainty)')
    
    return parser


def get_default_true_params(problem):
    """Get reasonable default true parameters for each problem type."""
    defaults = {
        'simplified_dis': torch.tensor([2.0, 1.2, 2.0, 1.2]),  # [a_u, b_u, a_d, b_d]
        'realistic_dis': torch.tensor([1.0, 0.1, 0.7, 3.0, 0.0, 0.0]),  # [log A_0, delta, a, b, c, d]
        'mceg': torch.tensor([1.0, 2.0, 3.0, 4.0]),  # Example parameters
        'mceg4dis': torch.tensor([1.0, 2.0, 3.0, 4.0])  # Same as mceg - 2D PDF inputs (x, Q2)
    }
    return defaults.get(problem, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def get_default_param_dim(problem):
    """Get default parameter dimension for each problem type."""
    dims = {
        'simplified_dis': 4,
        'realistic_dis': 6,
        'mceg': 4,
        'mceg4dis': 4  # Same as mceg - 4 parameters for 2D PDF inputs
    }
    return dims.get(problem, 4)


def run_combined_uncertainty_analysis(args):
    """
    Main function to run combined uncertainty analysis.
    
    This would be integrated into the existing plotting driver workflow.
    """
    
    print(f"Running combined uncertainty analysis for {args.problem} with {args.arch} architecture")
    print(f"Bootstrap samples: {args.n_bootstrap}, Events per sample: {args.num_events}")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get default parameters if not specified
    if args.param_dim is None:
        args.param_dim = get_default_param_dim(args.problem)
    
    if args.true_params is None:
        true_params = get_default_true_params(args.problem)
    else:
        true_params = torch.tensor(args.true_params)
    
    print(f"True parameters: {true_params.numpy()}")
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.test_mode:
        print("⚠️  Running in test mode with mock models")
        run_test_mode(args, device, true_params)
    else:
        print("Running with real models (requires trained models and dependencies)")
        # This is where real model loading would happen
        print("❌ Real model mode not implemented - requires trained models")
        print("   Use --test_mode for structure validation")
        return False
    
    return True


def run_test_mode(args, device, true_params):
    """
    Run combined uncertainty analysis in test mode with mock models.
    
    This demonstrates the function call structure without requiring real models.
    """
    from unittest.mock import Mock, MagicMock
    
    print("Creating mock models for testing...")
    
    # Create mock models that return reasonable shapes
    mock_model = Mock()
    mock_pointnet = Mock()
    mock_laplace = Mock() if not args.no_laplace else None
    
    # Mock the model outputs to return tensors of correct shape
    def mock_model_call(latent):
        batch_size = latent.shape[0] if hasattr(latent, 'shape') else 1
        return torch.randn(batch_size, args.param_dim)
    
    def mock_pointnet_call(events):
        batch_size = events.shape[0] if hasattr(events, 'shape') else 1
        return torch.randn(batch_size, args.latent_dim)
    
    mock_model.side_effect = mock_model_call
    mock_pointnet.side_effect = mock_pointnet_call
    
    # Add eval() method
    mock_model.eval = Mock(return_value=mock_model)
    mock_pointnet.eval = Mock(return_value=mock_pointnet)
    
    print("Testing input validation...")
    
    # Test the validation function first
    try:
        validate_combined_uncertainty_inputs(
            mock_model, mock_pointnet, true_params, device,
            args.num_events, args.n_bootstrap, args.problem, args.save_dir
        )
        print("✅ Input validation passed")
    except Exception as e:
        print(f"❌ Input validation failed: {e}")
        return False
    
    print("Validation successful! Function structure is ready for integration.")
    print(f"Would call plot_combined_uncertainty_PDF_distribution with:")
    print(f"  - problem: {args.problem}")
    print(f"  - n_bootstrap: {args.n_bootstrap}")
    print(f"  - num_events: {args.num_events}")
    print(f"  - save_dir: {args.save_dir}")
    print(f"  - laplace_model: {'Available' if mock_laplace else 'None'}")
    
    # Note: We don't actually call the function since it would require real simulators
    # But the structure is validated and ready for integration
    
    return True


def main():
    """Main entry point for the combined uncertainty analysis example."""
    
    parser = create_example_cli_integration()
    args = parser.parse_args()
    
    # Apply quick test settings if requested
    if args.quick_test:
        args.n_bootstrap = min(args.n_bootstrap, 10)
        args.num_events = min(args.num_events, 10000)
        print("Quick test mode: reducing parameters for fast execution")
    
    try:
        success = run_combined_uncertainty_analysis(args)
        
        if success:
            print("\n✅ Combined uncertainty analysis setup completed successfully!")
            print(f"   Integration ready for {args.problem} with {args.arch} architecture")
            print(f"   Use this pattern in plotting_driver_UQ.py for full functionality")
        else:
            print("\n❌ Analysis setup failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())