#!/usr/bin/env python3
"""
Minimal test for the new uncertainty scaling functions.
Tests just the core plotting logic without external dependencies.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def test_core_functions():
    """Test the core mathematical and plotting logic"""
    
    print("🧪 Testing uncertainty scaling core functions...")
    
    # Mock scaling results data structure
    event_counts = [1000, 5000, 10000, 50000]
    n_params = 4
    param_names = ['a_u', 'b_u', 'a_d', 'b_d']
    
    # Generate mock uncertainty data that follows 1/sqrt(N) scaling with some noise
    scaling_results = {
        'event_counts': event_counts,
        'problem': 'simplified_dis',
        'n_bootstrap': 10,
        'true_params': np.array([2.0, 1.2, 2.0, 1.2]),
        'param_names': param_names,
        'function_uncertainties': {},
        'parameter_uncertainties': [],
        'fixed_x_uncertainties': {},
        'laplace_available': False
    }
    
    # Generate realistic uncertainty scaling data
    base_param_uncertainty = 0.1
    base_func_uncertainty = 0.05
    
    for i, N in enumerate(event_counts):
        # Parameter uncertainties (1/sqrt(N) + noise)
        param_unc = base_param_uncertainty / np.sqrt(N / 1000) + np.random.normal(0, 0.001, n_params)
        param_unc = np.maximum(param_unc, 0.001)  # Ensure positive
        scaling_results['parameter_uncertainties'].append(param_unc)
        
        # Function uncertainties
        for func_name in ['up', 'down']:
            if func_name not in scaling_results['function_uncertainties']:
                scaling_results['function_uncertainties'][func_name] = []
            
            func_unc = base_func_uncertainty / np.sqrt(N / 1000) + np.random.normal(0, 0.0005)
            func_unc = max(func_unc, 0.001)
            scaling_results['function_uncertainties'][func_name].append(func_unc)
        
        # Fixed x uncertainties
        for x_val in [0.1, 0.5]:
            if x_val not in scaling_results['fixed_x_uncertainties']:
                scaling_results['fixed_x_uncertainties'][x_val] = {}
            
            for func_name in ['up', 'down']:
                if func_name not in scaling_results['fixed_x_uncertainties'][x_val]:
                    scaling_results['fixed_x_uncertainties'][x_val][func_name] = []
                
                fixed_unc = base_func_uncertainty / np.sqrt(N / 1000) + np.random.normal(0, 0.0005)
                fixed_unc = max(fixed_unc, 0.001)
                scaling_results['fixed_x_uncertainties'][x_val][func_name].append(fixed_unc)
    
    save_dir = "/tmp/test_minimal_uncertainty"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Test plot_uncertainty_at_fixed_x function directly
        print("\n🔬 Testing plot_uncertainty_at_fixed_x...")
        
        from plotting_UQ_utils import plot_uncertainty_at_fixed_x
        
        plot_uncertainty_at_fixed_x(
            scaling_results=scaling_results,
            x_values=[0.1, 0.5],
            save_dir=save_dir,
            comparison_functions=['up', 'down']
        )
        
        print("✅ plot_uncertainty_at_fixed_x completed successfully")
        
        # Test plot_summary_uncertainty_scaling function directly
        print("\n🔬 Testing plot_summary_uncertainty_scaling...")
        
        from plotting_UQ_utils import plot_summary_uncertainty_scaling
        
        summary_metrics = plot_summary_uncertainty_scaling(
            scaling_results=scaling_results,
            save_dir=save_dir,
            include_theoretical_comparison=True,
            aggregation_method='mean'
        )
        
        print("✅ plot_summary_uncertainty_scaling completed successfully")
        print(f"   Summary metrics keys: {list(summary_metrics.keys())}")
        
        if 'overall_consistency_score' in summary_metrics:
            print(f"   📊 Consistency score: {summary_metrics['overall_consistency_score']:.3f}")
            if 'param_scaling_exponent' in summary_metrics:
                print(f"   📈 Parameter scaling exponent: {summary_metrics['param_scaling_exponent']:.3f} (ideal: -0.5)")
        
        # Check generated files
        print("\n📁 Generated files:")
        for file in os.listdir(save_dir):
            if file.endswith(('.png', '.txt')):
                print(f"   ✅ {file}")
                
        print(f"\n🎉 Core function tests passed! Check results in {save_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mathematical_correctness():
    """Test the mathematical correctness of scaling analysis"""
    
    print("\n🧮 Testing mathematical correctness...")
    
    # Generate perfect 1/sqrt(N) data
    event_counts = np.array([1000, 2000, 4000, 8000, 16000])
    base_uncertainty = 0.1
    perfect_uncertainties = base_uncertainty / np.sqrt(event_counts / 1000)
    
    # Test scaling exponent calculation
    log_counts = np.log(event_counts)
    log_uncertainties = np.log(perfect_uncertainties)
    slope, intercept = np.polyfit(log_counts, log_uncertainties, 1)
    
    print(f"   Perfect 1/sqrt(N) data scaling exponent: {slope:.6f}")
    print(f"   Deviation from ideal (-0.5): {abs(slope + 0.5):.6f}")
    
    if abs(slope + 0.5) < 0.001:
        print("   ✅ Mathematical scaling calculation is correct")
        return True
    else:
        print("   ❌ Mathematical scaling calculation has issues")
        return False

if __name__ == "__main__":
    # Test mathematical correctness first
    math_ok = test_mathematical_correctness()
    
    # Test core functions
    functions_ok = test_core_functions()
    
    if math_ok and functions_ok:
        print("\n✅ All minimal tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)