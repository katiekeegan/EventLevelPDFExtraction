#!/usr/bin/env python3
"""
Test script for the new uncertainty scaling plotting functions.

This script creates mock data to test the new plotting functions without
requiring trained models or real data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List

# Mock classes for testing
class MockModel:
    """Mock model for testing"""
    def __init__(self, param_dim=4):
        self.param_dim = param_dim
        
    def eval(self):
        pass
        
    def __call__(self, latent_embedding):
        # Return mock parameter predictions with some noise
        batch_size = latent_embedding.shape[0]
        means = torch.randn(batch_size, self.param_dim) * 0.1 + torch.tensor([2.0, 1.2, 2.0, 1.2])
        logvars = torch.ones(batch_size, self.param_dim) * (-2)  # small variance
        return means, logvars

class MockPointNet:
    """Mock PointNet for testing"""
    def __init__(self, latent_dim=128):
        self.latent_dim = latent_dim
        
    def eval(self):
        pass
        
    def __call__(self, xs_tensor):
        # Return mock latent embedding
        batch_size = xs_tensor.shape[0]
        return torch.randn(batch_size, self.latent_dim)

class MockSimulator:
    """Mock simplified DIS simulator for testing"""
    def __init__(self, device=None):
        self.device = device or torch.device('cpu')
        self.au, self.bu, self.ad, self.bd = 2.0, 1.2, 2.0, 1.2
        
    def init(self, params):
        self.au, self.bu, self.ad, self.bd = [float(p) for p in params]
        
    def sample(self, params, nevents=1000):
        # Return mock event data
        return torch.rand(nevents, 2) * 0.8 + 0.1  # events in [0.1, 0.9]
        
    def up(self, x):
        return torch.pow(x, self.au) * torch.pow(1 - x, self.bu)
        
    def down(self, x):
        return torch.pow(x, self.ad) * torch.pow(1 - x, self.bd)

def mock_advanced_feature_engineering(xs_tensor):
    """Mock feature engineering function"""
    # Just return the input with some additional features
    batch_size, n_events, n_features = xs_tensor.shape
    additional_features = torch.randn(batch_size, n_events, 2)
    return torch.cat([xs_tensor, additional_features], dim=-1)

def test_uncertainty_scaling_functions():
    """Test the new uncertainty scaling functions with mock data"""
    
    print("🧪 Testing uncertainty scaling functions with mock data...")
    
    # Setup
    device = torch.device('cpu')
    true_params = torch.tensor([2.0, 1.2, 2.0, 1.2])
    save_dir = "/tmp/test_uncertainty_scaling"
    os.makedirs(save_dir, exist_ok=True)
    
    # Mock models
    model = MockModel(param_dim=4)
    pointnet_model = MockPointNet(latent_dim=128)
    
    # Mock event counts (smaller for testing)
    event_counts = [100, 500, 1000, 5000, 10000]
    
    print(f"   Testing with event counts: {event_counts}")
    print(f"   Save directory: {save_dir}")
    
    try:
        # Import the new functions
        from plotting_UQ_utils import (
            plot_uncertainty_vs_events,
            plot_uncertainty_at_fixed_x, 
            plot_summary_uncertainty_scaling
        )
        
        print("✅ Successfully imported new functions")
        
        # Test 1: Main uncertainty vs events function
        print("\n🔬 Test 1: plot_uncertainty_vs_events")
        
        # Mock the simulator import in the function
        import sys
        sys.modules['simulator'] = type('MockModule', (), {
            'SimplifiedDIS': MockSimulator,
            'RealisticDIS': MockSimulator, 
            'MCEGSimulator': MockSimulator
        })()
        
        # Mock the feature engineering import
        sys.modules['PDF_learning'] = type('MockModule', (), {
            'advanced_feature_engineering': mock_advanced_feature_engineering
        })()
        
        scaling_results = plot_uncertainty_vs_events(
            model=model,
            pointnet_model=pointnet_model,
            true_params=true_params,
            device=device,
            event_counts=event_counts,
            n_bootstrap=5,  # Small for testing
            laplace_model=None,  # No Laplace for simple test
            problem='simplified_dis',
            save_dir=save_dir,
            fixed_x_values=[0.1, 0.5]
        )
        
        print("✅ plot_uncertainty_vs_events completed successfully")
        print(f"   Generated results keys: {list(scaling_results.keys())}")
        
        # Test 2: Fixed x analysis
        print("\n🔬 Test 2: plot_uncertainty_at_fixed_x")
        
        plot_uncertainty_at_fixed_x(
            scaling_results=scaling_results,
            x_values=[0.1, 0.5],
            save_dir=save_dir,
            comparison_functions=['up', 'down']
        )
        
        print("✅ plot_uncertainty_at_fixed_x completed successfully")
        
        # Test 3: Summary scaling analysis
        print("\n🔬 Test 3: plot_summary_uncertainty_scaling")
        
        summary_metrics = plot_summary_uncertainty_scaling(
            scaling_results=scaling_results,
            save_dir=save_dir,
            include_theoretical_comparison=True,
            aggregation_method='mean'
        )
        
        print("✅ plot_summary_uncertainty_scaling completed successfully")
        print(f"   Summary metrics keys: {list(summary_metrics.keys())}")
        
        # Check generated files
        print("\n📁 Generated files:")
        for file in os.listdir(save_dir):
            if file.endswith(('.png', '.txt')):
                print(f"   ✅ {file}")
        
        print(f"\n🎉 All tests passed! Check results in {save_dir}")
        
        # Display some key metrics
        if 'overall_consistency_score' in summary_metrics:
            print(f"   📊 Mock consistency score: {summary_metrics['overall_consistency_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_uncertainty_scaling_functions()
    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Tests failed!")
        exit(1)