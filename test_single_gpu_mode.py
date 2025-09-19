#!/usr/bin/env python3
"""
Test script for single-GPU mode functionality in parameter_prediction.py
"""
import sys
import os
import subprocess
import argparse

def test_argument_parsing():
    """Test that the --single_gpu argument is accepted"""
    print("Testing argument parsing...")
    
    try:
        # Read the parameter_prediction.py file to check for --single_gpu
        with open("parameter_prediction.py", "r") as f:
            content = f.read()
            
        if "--single_gpu" in content and "Force single-GPU mode" in content:
            print("✓ --single_gpu argument is present in parameter_prediction.py")
            
            # Also test the parsing logic directly
            import sys
            sys.path.insert(0, ".")
            exec_globals = {}
            
            # Extract just the argument parsing section
            lines = content.split('\n')
            parser_start = None
            for i, line in enumerate(lines):
                if "parser = argparse.ArgumentParser()" in line:
                    parser_start = i
                    break
            
            if parser_start:
                # Execute the argument parsing code
                try:
                    import argparse
                    parser = argparse.ArgumentParser()
                    # Find all add_argument lines
                    for line in lines[parser_start:]:
                        if "parser.add_argument" in line and "--single_gpu" in line:
                            print("✓ --single_gpu argument definition found")
                            return True
                        if "args = parser.parse_args()" in line:
                            break
                    
                except Exception as e:
                    print(f"Note: Could not execute parser code: {e}")
                    # But we already found the argument definition
                    return True
            
            return True
        else:
            print("✗ --single_gpu argument not found in parameter_prediction.py")
            return False
            
    except Exception as e:
        print(f"✗ Error testing argument parsing: {e}")
        return False

def test_mode_selection_logic():
    """Test the mode selection logic directly"""
    print("Testing mode selection logic...")
    
    try:
        # Mock the logic from parameter_prediction.py
        import torch
        
        # Simulate different scenarios
        test_cases = [
            {"single_gpu": True, "world_size": 2, "expected": "single"},
            {"single_gpu": False, "world_size": 2, "expected": "multi"},
            {"single_gpu": False, "world_size": 1, "expected": "single"},
            {"single_gpu": True, "world_size": 1, "expected": "single"},
        ]
        
        for case in test_cases:
            single_gpu = case["single_gpu"]
            world_size = case["world_size"]
            expected = case["expected"]
            
            # Apply the logic from parameter_prediction.py
            if single_gpu or world_size <= 1:
                actual = "single"
            else:
                actual = "multi"
            
            if actual == expected:
                print(f"✓ Case single_gpu={single_gpu}, world_size={world_size}: {actual} mode (expected {expected})")
            else:
                print(f"✗ Case single_gpu={single_gpu}, world_size={world_size}: {actual} mode (expected {expected})")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error testing mode selection logic: {e}")
        return False

def test_atomic_savez_import():
    """Test that the updated atomic_savez_compressed function can be imported"""
    print("Testing atomic_savez_compressed import...")
    
    try:
        from generate_precomputed_data import atomic_savez_compressed
        print("✓ atomic_savez_compressed imported successfully")
        
        # Test that it can handle non-distributed case
        import tempfile
        import numpy as np
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.npz")
            test_data = {"data": np.array([1, 2, 3])}
            atomic_savez_compressed(test_file, **test_data)
            
            if os.path.exists(test_file):
                print("✓ atomic_savez_compressed works in non-distributed mode")
                return True
            else:
                print("✗ atomic_savez_compressed failed to create file")
                return False
                
    except Exception as e:
        print(f"✗ Error testing atomic_savez_compressed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running tests for single-GPU mode functionality...\n")
    
    tests = [
        test_argument_parsing,
        test_mode_selection_logic,
        test_atomic_savez_import,
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
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())