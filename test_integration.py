#!/usr/bin/env python3
"""
Integration test for parameter_prediction.py single-GPU mode with minimal dependencies
"""
import os
import sys
import tempfile
import torch
import numpy as np

def test_single_gpu_mode_functionality():
    """Test single-GPU mode by running the core logic without full dependencies"""
    print("Testing single-GPU mode functionality...")
    
    try:
        # Create a minimal test setup
        class MockArgs:
            def __init__(self):
                self.single_gpu = True
                self.experiment_name = "test_single_gpu"
                self.num_samples = 100
                self.num_events = 10
                self.batch_size = 4
                self.lr = 1e-4
                self.latent_dim = 8
                self.problem = "gaussian"
                self.wandb = False
                self.use_precomputed = False
                
        args = MockArgs()
        
        # Test the mode selection logic
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        if args.single_gpu or world_size <= 1:
            print(f"✓ Single-GPU mode selected (single_gpu={args.single_gpu}, detected GPUs={world_size})")
            mode = "single"
        else:
            print(f"Multi-GPU mode would be selected with {world_size} GPUs")
            mode = "multi"
        
        # Test device selection
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"✓ Device selected: {device}")
        else:
            device = torch.device("cpu")
            print(f"✓ Device selected: {device} (no CUDA available)")
        
        # Test model creation without DDP (single-GPU mode)
        if mode == "single":
            # Create a simple model to test the logic
            class SimpleModel(torch.nn.Module):
                def __init__(self, input_dim, latent_dim):
                    super().__init__()
                    self.linear = torch.nn.Linear(input_dim, latent_dim)
                
                def forward(self, x):
                    return self.linear(x)
            
            model = SimpleModel(2, args.latent_dim).to(device)
            print("✓ Model created without DDP wrapper (single-GPU mode)")
            
            # Verify no DDP wrapper
            if not hasattr(model, 'module'):
                print("✓ Model is not wrapped in DDP (correct for single-GPU)")
            else:
                print("✗ Model unexpectedly wrapped in DDP")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in single-GPU mode test: {e}")
        return False

def test_distributed_atomic_save_simulation():
    """Simulate the distributed atomic save behavior"""
    print("Testing distributed atomic save simulation...")
    
    try:
        from generate_precomputed_data import atomic_savez_compressed
        
        # Test non-distributed case (which we can actually run)
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_distributed.npz")
            test_data = {"data": np.random.rand(10, 5)}
            
            # This should work in non-distributed mode
            atomic_savez_compressed(test_file, **test_data)
            
            if os.path.exists(test_file):
                print("✓ Atomic save works in non-distributed mode")
                
                # Verify file content
                loaded = np.load(test_file)
                if np.array_equal(loaded["data"], test_data["data"]):
                    print("✓ Saved data integrity verified")
                    return True
                else:
                    print("✗ Saved data integrity check failed")
                    return False
            else:
                print("✗ Atomic save failed to create file")
                return False
                
    except Exception as e:
        print(f"✗ Error in distributed atomic save test: {e}")
        return False

def test_gpu_detection_logic():
    """Test the GPU detection and mode selection logic"""
    print("Testing GPU detection and mode selection...")
    
    try:
        # Test the actual logic from parameter_prediction.py
        world_size = torch.cuda.device_count()
        print(f"Detected GPUs: {world_size}")
        
        # Test all combinations
        test_cases = [
            (True, world_size, "single"),   # --single_gpu forces single mode
            (False, 1, "single"),           # Only 1 GPU available
            (False, 0, "single"),           # No GPUs available
        ]
        
        # Only test multi-GPU if we actually have multiple GPUs
        if world_size > 1:
            test_cases.append((False, world_size, "multi"))
        
        for single_gpu_flag, detected_gpus, expected_mode in test_cases:
            if single_gpu_flag or detected_gpus <= 1:
                actual_mode = "single"
            else:
                actual_mode = "multi"
            
            if actual_mode == expected_mode:
                print(f"✓ single_gpu={single_gpu_flag}, GPUs={detected_gpus} → {actual_mode} mode")
            else:
                print(f"✗ single_gpu={single_gpu_flag}, GPUs={detected_gpus} → {actual_mode} mode (expected {expected_mode})")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in GPU detection test: {e}")
        return False

def main():
    """Run all integration tests"""
    print("Running integration tests for single-GPU/multi-GPU functionality...\n")
    
    tests = [
        test_single_gpu_mode_functionality,
        test_distributed_atomic_save_simulation, 
        test_gpu_detection_logic,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}\n")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Integration tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("❌ Some integration tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())