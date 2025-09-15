#!/usr/bin/env python3
"""
Test script for the new generate_parameter_error_histogram function.
This test should work in the repository environment with proper mocking.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import tempfile

# Mock simulators to avoid dependency issues
class MockSimplifiedDIS:
    def __init__(self, device=None):
        self.device = device or torch.device('cpu')
        
    def sample(self, params, nevents=1000):
        # Return mock events that will work with feature engineering
        return torch.randn(nevents, 2, device=self.device)

def mock_get_simulator_module():
    return MockSimplifiedDIS, None, None

def mock_get_advanced_feature_engineering():
    """Mock feature engineering that produces expected features"""
    def mock_feature_engineering(x):
        # Convert 2 features to 4 features for compatibility
        log_x = torch.log(torch.abs(x) + 1e-8)
        x_squared = x ** 2
        return torch.cat([log_x, x_squared], dim=-1)
    return mock_feature_engineering

# Patch the import functions
try:
    import plotting_UQ_utils as puu
    puu.get_simulator_module = mock_get_simulator_module
    puu.get_advanced_feature_engineering = mock_get_advanced_feature_engineering
    
    from plotting_UQ_utils import generate_parameter_error_histogram, get_parameter_bounds_for_problem
except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)

class TestGenerateParameterErrorHistogram:
    """Test class for the generate_parameter_error_histogram function"""
    
    def create_mock_models(self, latent_dim=64, output_dim=4):
        """Create mock models for testing"""
        
        class MockPointNet(torch.nn.Module):
            def __init__(self, input_dim=4, latent_dim=64):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_dim, 32)
                self.fc2 = torch.nn.Linear(32, latent_dim)
                
            def forward(self, x):
                x_pooled = x.mean(dim=1)
                x = torch.relu(self.fc1(x_pooled))
                return self.fc2(x)
        
        class MockInferenceNet(torch.nn.Module):
            def __init__(self, latent_dim=64, output_dim=4):
                super().__init__()
                self.fc = torch.nn.Linear(latent_dim, output_dim)
                self.nll_mode = False
                
            def forward(self, z):
                return self.fc(z)
        
        pointnet = MockPointNet(input_dim=4, latent_dim=latent_dim)
        inference = MockInferenceNet(latent_dim=latent_dim, output_dim=output_dim)
        
        # Initialize with small weights
        with torch.no_grad():
            for param in pointnet.parameters():
                param.data.normal_(0, 0.1)
            for param in inference.parameters():
                param.data.normal_(0, 0.1)
        
        return pointnet, inference
    
    def test_parameter_bounds_retrieval(self):
        """Test that parameter bounds are correctly retrieved"""
        bounds = get_parameter_bounds_for_problem('simplified_dis')
        assert bounds.shape == (4, 2), f"Expected (4, 2), got {bounds.shape}"
        assert torch.all(bounds[:, 0] >= 0), "Lower bounds should be >= 0"
        assert torch.all(bounds[:, 1] > bounds[:, 0]), "Upper bounds should be > lower bounds"
        
        bounds = get_parameter_bounds_for_problem('realistic_dis')
        assert bounds.shape == (6, 2), f"Expected (6, 2), got {bounds.shape}"
        
        bounds = get_parameter_bounds_for_problem('mceg')
        assert bounds.shape == (4, 2), f"Expected (4, 2), got {bounds.shape}"
    
    def test_basic_functionality(self):
        """Test basic functionality of the function"""
        device = torch.device('cpu')
        pointnet_model, inference_model = self.create_mock_models()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                true_params, pred_params = generate_parameter_error_histogram(
                    model=inference_model,
                    pointnet_model=pointnet_model,
                    device=device,
                    n_draws=5,
                    n_events=100,
                    problem='simplified_dis',
                    save_path=tmp_file.name,
                    return_data=True
                )
                
                # Validate results
                assert len(true_params) == len(pred_params)
                assert len(true_params) > 0
                assert true_params[0].shape[0] == 4  # simplified_dis has 4 parameters
                assert os.path.exists(tmp_file.name)
                assert os.path.getsize(tmp_file.name) > 1000  # Plot should be reasonably sized
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_different_problem_types(self):
        """Test function with different problem types"""
        device = torch.device('cpu')
        
        problems = [
            ('simplified_dis', 4),
            ('realistic_dis', 6),
            ('mceg', 4),
        ]
        
        for problem, n_params in problems:
            pointnet_model, inference_model = self.create_mock_models(
                latent_dim=64, 
                output_dim=n_params
            )
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                try:
                    true_params, pred_params = generate_parameter_error_histogram(
                        model=inference_model,
                        pointnet_model=pointnet_model,
                        device=device,
                        n_draws=3,
                        n_events=50,
                        problem=problem,
                        save_path=tmp_file.name,
                        return_data=True
                    )
                    
                    assert len(true_params) > 0, f"No samples generated for {problem}"
                    assert true_params[0].shape[0] == n_params, f"Wrong parameter count for {problem}"
                    assert os.path.exists(tmp_file.name), f"Plot not created for {problem}"
                    
                finally:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
    
    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        device = torch.device('cpu')
        pointnet_model, inference_model = self.create_mock_models()
        
        # Test invalid problem type
        try:
            generate_parameter_error_histogram(
                model=inference_model,
                pointnet_model=pointnet_model,
                device=device,
                n_draws=1,
                n_events=10,
                problem='invalid_problem',
                save_path='/tmp/test_invalid.png'
            )
            assert False, "Should have raised an error for invalid problem type"
        except (ValueError, RuntimeError):
            pass  # Expected behavior
    
    def test_without_return_data(self):
        """Test function without returning data"""
        device = torch.device('cpu')
        pointnet_model, inference_model = self.create_mock_models()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                result = generate_parameter_error_histogram(
                    model=inference_model,
                    pointnet_model=pointnet_model,
                    device=device,
                    n_draws=3,
                    n_events=50,
                    problem='simplified_dis',
                    save_path=tmp_file.name,
                    return_data=False
                )
                
                assert result is None  # Should not return data
                assert os.path.exists(tmp_file.name)
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)

def run_tests():
    """Run all tests manually (since pytest might not be available)"""
    test_instance = TestGenerateParameterErrorHistogram()
    
    tests = [
        ('Parameter bounds retrieval', test_instance.test_parameter_bounds_retrieval),
        ('Basic functionality', test_instance.test_basic_functionality),
        ('Different problem types', test_instance.test_different_problem_types),
        ('Error handling', test_instance.test_error_handling),
        ('Without return data', test_instance.test_without_return_data),
    ]
    
    print("🧪 Running tests for generate_parameter_error_histogram...")
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n   Testing: {test_name}")
            test_func()
            print(f"   ✅ {test_name} passed")
            passed += 1
        except Exception as e:
            print(f"   ❌ {test_name} failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results:")
    print(f"   Total: {passed + failed}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    return passed, failed

if __name__ == "__main__":
    run_tests()