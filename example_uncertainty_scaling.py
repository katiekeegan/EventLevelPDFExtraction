#!/usr/bin/env python3
"""
Example usage script demonstrating the new uncertainty scaling analysis functions.

This script shows how to use the new plotting functions to demonstrate 
consistency of uncertainty quantification for simplified_dis PDFs by 
showing how uncertainty bands shrink as the number of events increases.

Usage:
    python example_uncertainty_scaling.py --problem simplified_dis --save_dir ./plots/scaling
    python example_uncertainty_scaling.py --problem realistic_dis --event_counts 1000,5000,10000,50000
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_example_usage():
    """
    Demonstrate the new uncertainty scaling analysis functions.
    
    This function shows how to integrate the new uncertainty vs events
    plotting functions into your workflow to validate UQ consistency.
    """
    
    parser = argparse.ArgumentParser(
        description="Uncertainty Scaling Analysis for PDF Parameter Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic scaling analysis for simplified DIS
    python example_uncertainty_scaling.py --problem simplified_dis
    
    # Custom event counts and parameters
    python example_uncertainty_scaling.py --problem simplified_dis \\
        --event_counts 1000,5000,10000,50000,100000 \\
        --n_bootstrap 25 \\
        --save_dir ./plots/custom_scaling
    
    # Realistic DIS with specific Q2 slices
    python example_uncertainty_scaling.py --problem realistic_dis \\
        --Q2_slices 2.0,10.0,50.0 \\
        --fixed_x_values 0.01,0.1,0.5
        
    # Test with mock data (no real models needed)
    python example_uncertainty_scaling.py --mock_mode
        """
    )
    
    parser.add_argument('--problem', type=str, default='simplified_dis',
                       choices=['simplified_dis', 'realistic_dis', 'mceg'],
                       help='Problem type to analyze')
    
    parser.add_argument('--event_counts', type=str, 
                       default='1000,5000,10000,50000,100000',
                       help='Comma-separated list of event counts to test')
    
    parser.add_argument('--n_bootstrap', type=int, default=20,
                       help='Number of bootstrap samples per event count')
    
    parser.add_argument('--save_dir', type=str, default='./plots/uncertainty_scaling',
                       help='Directory to save analysis results')
    
    parser.add_argument('--fixed_x_values', type=str, default='0.01,0.1,0.5',
                       help='Comma-separated list of x values to track')
    
    parser.add_argument('--Q2_slices', type=str, default='2.0,10.0,50.0,200.0',
                       help='Comma-separated list of Q2 values (for realistic_dis)')
    
    parser.add_argument('--mock_mode', action='store_true',
                       help='Run with mock data instead of real models')
    
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Parse lists from strings
    event_counts = [int(x.strip()) for x in args.event_counts.split(',')]
    fixed_x_values = [float(x.strip()) for x in args.fixed_x_values.split(',')]
    Q2_slices = [float(x.strip()) for x in args.Q2_slices.split(',')]
    
    device = torch.device(args.device)
    
    print("🔬 Uncertainty Quantification Scaling Analysis")
    print("=" * 50)
    print(f"Problem: {args.problem}")
    print(f"Event counts: {event_counts}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Save directory: {args.save_dir}")
    print(f"Mock mode: {args.mock_mode}")
    print()
    
    if args.mock_mode:
        run_mock_scaling_analysis(
            problem=args.problem,
            event_counts=event_counts,
            n_bootstrap=args.n_bootstrap,
            save_dir=args.save_dir,
            fixed_x_values=fixed_x_values,
            Q2_slices=Q2_slices,
            device=device
        )
    else:
        run_real_scaling_analysis(
            problem=args.problem,
            event_counts=event_counts,
            n_bootstrap=args.n_bootstrap,
            save_dir=args.save_dir,
            fixed_x_values=fixed_x_values,
            Q2_slices=Q2_slices,
            device=device
        )


def run_mock_scaling_analysis(problem, event_counts, n_bootstrap, save_dir, 
                             fixed_x_values, Q2_slices, device):
    """
    Run scaling analysis with mock data to demonstrate functionality.
    
    This is useful for testing and demonstrating the functions without
    requiring trained models or large datasets.
    """
    
    print("🧪 Running mock scaling analysis...")
    
    # Generate mock scaling results that follow 1/sqrt(N) scaling
    n_params = 4 if problem == 'simplified_dis' else 6
    param_names = (['a_u', 'b_u', 'a_d', 'b_d'] if problem == 'simplified_dis' 
                  else ['log_A0', 'delta', 'a', 'b', 'c', 'd'])
    
    scaling_results = generate_mock_scaling_results(
        problem=problem,
        event_counts=event_counts,
        n_params=n_params,
        param_names=param_names,
        n_bootstrap=n_bootstrap,
        fixed_x_values=fixed_x_values,
        Q2_slices=Q2_slices if problem == 'realistic_dis' else None
    )
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        from plotting_UQ_utils import (
            plot_uncertainty_at_fixed_x,
            plot_summary_uncertainty_scaling
        )
        
        print("\n📍 Creating fixed-x uncertainty analysis...")
        plot_uncertainty_at_fixed_x(
            scaling_results=scaling_results,
            x_values=fixed_x_values,
            save_dir=save_dir,
            comparison_functions=['up', 'down'] if problem == 'simplified_dis' else None
        )
        
        print("\n📈 Creating summary scaling analysis...")
        summary_metrics = plot_summary_uncertainty_scaling(
            scaling_results=scaling_results,
            save_dir=save_dir,
            include_theoretical_comparison=True,
            aggregation_method='mean'
        )
        
        print("\n✅ Mock scaling analysis complete!")
        print(f"📊 Consistency score: {summary_metrics.get('overall_consistency_score', 'N/A'):.3f}")
        print(f"📂 Results saved to: {save_dir}")
        
        # List generated files
        print("\n📁 Generated files:")
        for file in sorted(os.listdir(save_dir)):
            if file.endswith(('.png', '.txt')):
                print(f"   • {file}")
        
        return summary_metrics
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed")
        return None
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_real_scaling_analysis(problem, event_counts, n_bootstrap, save_dir,
                             fixed_x_values, Q2_slices, device):
    """
    Run scaling analysis with real trained models.
    
    This would be used in practice with actual trained PointNet and
    parameter prediction models.
    """
    
    print("🔧 Running real scaling analysis...")
    print("⚠️  This requires trained models and proper setup")
    
    try:
        # Attempt to load real models (this would need to be adapted to your setup)
        from plotting_driver_UQ import reload_model, reload_pointnet
        
        # Example model loading (adapt to your model paths)
        model_path = f"./models/{problem}_model.pt" 
        pointnet_path = f"./models/{problem}_pointnet.pt"
        
        if not (os.path.exists(model_path) and os.path.exists(pointnet_path)):
            print(f"❌ Model files not found:")
            print(f"   Expected: {model_path}")
            print(f"   Expected: {pointnet_path}")
            print("   Running mock analysis instead...")
            return run_mock_scaling_analysis(
                problem, event_counts, n_bootstrap, save_dir,
                fixed_x_values, Q2_slices, device
            )
        
        # Load models
        model = reload_model(model_path, device)
        pointnet_model = reload_pointnet(pointnet_path, device)
        
        # Set true parameters (example values)
        if problem == 'simplified_dis':
            true_params = torch.tensor([2.0, 1.2, 2.0, 1.2], device=device)
        elif problem == 'realistic_dis':
            true_params = torch.tensor([1.0, 0.1, 0.7, 3.0, 0.0, 0.0], device=device)
        else:
            true_params = torch.randn(4, device=device)  # fallback
        
        # Try to load Laplace model if available
        laplace_model = None
        laplace_path = f"./models/{problem}_laplace.pt"
        if os.path.exists(laplace_path):
            try:
                laplace_model = torch.load(laplace_path, map_location=device)
                print("✅ Loaded Laplace model for enhanced uncertainty")
            except Exception as e:
                print(f"⚠️  Could not load Laplace model: {e}")
        
        # Import the main scaling function
        from plotting_UQ_utils import (
            plot_uncertainty_vs_events,
            plot_uncertainty_at_fixed_x,
            plot_summary_uncertainty_scaling
        )
        
        print(f"\n🔄 Running uncertainty vs events analysis...")
        print(f"   This may take several minutes for large event counts...")
        
        # Main scaling analysis
        scaling_results = plot_uncertainty_vs_events(
            model=model,
            pointnet_model=pointnet_model,
            true_params=true_params,
            device=device,
            event_counts=event_counts,
            n_bootstrap=n_bootstrap,
            laplace_model=laplace_model,
            problem=problem,
            save_dir=save_dir,
            Q2_slices=Q2_slices if problem == 'realistic_dis' else None,
            fixed_x_values=fixed_x_values
        )
        
        print("\n📍 Creating detailed fixed-x analysis...")
        plot_uncertainty_at_fixed_x(
            scaling_results=scaling_results,
            x_values=fixed_x_values,
            save_dir=save_dir
        )
        
        print("\n📈 Creating summary analysis...")
        summary_metrics = plot_summary_uncertainty_scaling(
            scaling_results=scaling_results,
            save_dir=save_dir,
            include_theoretical_comparison=True,
            aggregation_method='mean'
        )
        
        print("\n✅ Real scaling analysis complete!")
        print_analysis_summary(summary_metrics, save_dir)
        
        return summary_metrics
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Falling back to mock analysis...")
        return run_mock_scaling_analysis(
            problem, event_counts, n_bootstrap, save_dir,
            fixed_x_values, Q2_slices, device
        )
    except Exception as e:
        print(f"❌ Real analysis failed: {e}")
        print("   Falling back to mock analysis...")
        return run_mock_scaling_analysis(
            problem, event_counts, n_bootstrap, save_dir,
            fixed_x_values, Q2_slices, device
        )


def generate_mock_scaling_results(problem, event_counts, n_params, param_names,
                                 n_bootstrap, fixed_x_values, Q2_slices=None):
    """
    Generate realistic mock scaling results that follow 1/sqrt(N) behavior.
    """
    
    base_param_uncertainty = 0.1
    base_func_uncertainty = 0.05
    
    scaling_results = {
        'event_counts': event_counts,
        'problem': problem,
        'n_bootstrap': n_bootstrap,
        'true_params': np.random.uniform(1.0, 3.0, n_params),
        'param_names': param_names,
        'function_uncertainties': {},
        'parameter_uncertainties': [],
        'fixed_x_uncertainties': {},
        'laplace_available': False
    }
    
    # Determine function names based on problem
    if problem == 'simplified_dis':
        func_names = ['up', 'down']
    elif problem == 'realistic_dis':
        func_names = [f'q_Q2_{Q2}' for Q2 in Q2_slices] if Q2_slices else ['q_Q2_2.0', 'q_Q2_10.0']
    else:
        func_names = ['pdf_1', 'pdf_2']
    
    # Generate scaling data
    for i, N in enumerate(event_counts):
        # Parameter uncertainties with 1/sqrt(N) scaling + realistic noise
        noise_scale = 0.02 * base_param_uncertainty
        param_unc = (base_param_uncertainty / np.sqrt(N / 1000) + 
                    np.random.normal(0, noise_scale, n_params))
        param_unc = np.maximum(param_unc, 0.001)  # Ensure positive
        scaling_results['parameter_uncertainties'].append(param_unc)
        
        # Function uncertainties
        for func_name in func_names:
            if func_name not in scaling_results['function_uncertainties']:
                scaling_results['function_uncertainties'][func_name] = []
            
            func_noise = 0.01 * base_func_uncertainty
            func_unc = (base_func_uncertainty / np.sqrt(N / 1000) + 
                       np.random.normal(0, func_noise))
            func_unc = max(func_unc, 0.001)
            scaling_results['function_uncertainties'][func_name].append(func_unc)
        
        # Fixed x uncertainties
        for x_val in fixed_x_values:
            if x_val not in scaling_results['fixed_x_uncertainties']:
                scaling_results['fixed_x_uncertainties'][x_val] = {}
            
            for func_name in func_names:
                if func_name not in scaling_results['fixed_x_uncertainties'][x_val]:
                    scaling_results['fixed_x_uncertainties'][x_val][func_name] = []
                
                # Add some x-dependence to make it more realistic
                x_factor = 1.0 + 0.5 * x_val  # uncertainty increases with x
                fixed_noise = 0.005 * base_func_uncertainty
                fixed_unc = (base_func_uncertainty * x_factor / np.sqrt(N / 1000) + 
                            np.random.normal(0, fixed_noise))
                fixed_unc = max(fixed_unc, 0.001)
                scaling_results['fixed_x_uncertainties'][x_val][func_name].append(fixed_unc)
    
    return scaling_results


def print_analysis_summary(summary_metrics, save_dir):
    """Print a summary of the analysis results."""
    
    print(f"📊 Analysis Summary:")
    print(f"   Directory: {save_dir}")
    
    if summary_metrics and 'overall_consistency_score' in summary_metrics:
        score = summary_metrics['overall_consistency_score']
        print(f"   Overall consistency: {score:.3f}")
        
        if score > 0.8:
            print("   ✅ EXCELLENT: Uncertainty scaling is highly consistent")
        elif score > 0.6:
            print("   ⚠️  GOOD: Uncertainty scaling is mostly consistent")
        else:
            print("   ❌ POOR: Uncertainty scaling has significant deviations")
            
        if 'param_scaling_exponent' in summary_metrics:
            exp = summary_metrics['param_scaling_exponent']
            print(f"   Parameter scaling: {exp:.3f} (ideal: -0.5)")
            
        if 'func_scaling_exponent' in summary_metrics:
            exp = summary_metrics['func_scaling_exponent']
            print(f"   Function scaling: {exp:.3f} (ideal: -0.5)")
    
    # List key output files
    key_files = [
        'uncertainty_scaling_summary.png',
        'uncertainty_consistency_metrics.txt',
        'uncertainty_fixed_x_comparison.png'
    ]
    
    print("\n📁 Key output files:")
    for file in key_files:
        path = os.path.join(save_dir, file)
        if os.path.exists(path):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (not found)")


if __name__ == "__main__":
    try:
        create_example_usage()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)