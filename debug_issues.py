#!/usr/bin/env python3
"""
Debug script to investigate CUDA multiprocessing and precomputed data issues.
"""

import torch
import torch.multiprocessing as mp
import os
import sys
import glob
from torch.utils.data import DataLoader

print("=== DIAGNOSTIC SCRIPT FOR CUDA MULTIPROCESSING AND PRECOMPUTED DATA ISSUES ===")
print()

# Issue 1: Check multiprocessing start method
print("1. MULTIPROCESSING START METHOD DIAGNOSTICS:")
print(f"   Current start method: {mp.get_start_method()}")
print(f"   Available start methods: {mp.get_all_start_methods()}")
print(f"   Recommended for CUDA: 'spawn'")
print()

# Issue 2: Check CUDA availability and initialization
print("2. CUDA DIAGNOSTICS:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA device count: {torch.cuda.device_count()}")
    print(f"   Current CUDA device: {torch.cuda.current_device()}")
    print(f"   CUDA initialized: {torch.cuda.is_initialized()}")
print()

# Issue 3: Check precomputed data detection
print("3. PRECOMPUTED DATA DETECTION DIAGNOSTICS:")
print()

# Import filter function if available
try:
    from precomputed_datasets import filter_valid_precomputed_files
    PRECOMPUTED_AVAILABLE = True
    print("   ✓ precomputed_datasets module imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import precomputed_datasets: {e}")
    PRECOMPUTED_AVAILABLE = False
    
    # Define fallback
    def filter_valid_precomputed_files(file_list):
        valid_files = []
        for file_path in file_list:
            filename = os.path.basename(file_path)
            if filename.endswith('.npz') and '.tmp' not in filename:
                valid_files.append(file_path)
        return valid_files

print()

# Check each problem type
data_dir = "precomputed_data"
problems = ["mceg", "gaussian", "simplified_dis", "realistic_dis"]

for problem in problems:
    print(f"   Problem: '{problem}'")
    
    # Check pattern matching
    pattern = os.path.join(data_dir, f"{problem}_*.npz")
    all_files = sorted(glob.glob(pattern))
    print(f"     Pattern '{pattern}' found {len(all_files)} files: {all_files}")
    
    # Apply filtering
    valid_files = filter_valid_precomputed_files(all_files)
    print(f"     After filtering: {len(valid_files)} valid files: {valid_files}")
    
    if all_files and not valid_files:
        temp_files = [f for f in all_files if f not in valid_files]
        print(f"     ⚠️  All files filtered out as temporary: {temp_files}")
    elif valid_files:
        print(f"     ✓ Valid precomputed data found")
    else:
        print(f"     ✗ No precomputed data found")
    print()

# Issue 4: DataLoader worker diagnostics
print("4. DATALOADER WORKER DIAGNOSTICS:")
print()

# Create a minimal dataset for testing
class TestDataset:
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # This would normally trigger CUDA operations in worker
        return torch.tensor([idx], dtype=torch.float32)

try:
    test_dataset = TestDataset()
    print(f"   Created test dataset with {len(test_dataset)} items")
    
    # Test with workers=0 (main process)
    print("   Testing DataLoader with num_workers=0 (main process)...")
    loader_no_workers = DataLoader(test_dataset, batch_size=2, num_workers=0)
    sample = next(iter(loader_no_workers))
    print(f"     ✓ No workers: Success, sample shape: {sample.shape}")
    
    # Test with workers=1 (subprocess)
    print("   Testing DataLoader with num_workers=1 (subprocess)...")
    try:
        loader_with_workers = DataLoader(test_dataset, batch_size=2, num_workers=1)
        sample = next(iter(loader_with_workers))
        print(f"     ✓ With workers: Success, sample shape: {sample.shape}")
    except Exception as e:
        print(f"     ✗ With workers: Failed with error: {e}")
        print(f"       This suggests a multiprocessing start method issue")
    
except Exception as e:
    print(f"   ✗ DataLoader test failed: {e}")

print()

print("=== RECOMMENDATIONS ===")
print()

if mp.get_start_method() != 'spawn':
    print("🔧 ISSUE 1 - CUDA Multiprocessing:")
    print("   The current multiprocessing start method is not 'spawn'.")
    print("   CUDA requires 'spawn' method to avoid re-initialization errors.")
    print("   Add this at the start of your script:")
    print("   mp.set_start_method('spawn', force=True)")
    print()

# Check if we found precomputed data issues
has_data_issues = False
for problem in problems:
    pattern = os.path.join(data_dir, f"{problem}_*.npz")
    all_files = sorted(glob.glob(pattern))
    valid_files = filter_valid_precomputed_files(all_files)
    if all_files and not valid_files:
        has_data_issues = True
        break

if has_data_issues:
    print("🔧 ISSUE 2 - Precomputed Data Detection:")
    print("   Some precomputed files exist but are being filtered out.")
    print("   Check if files have .tmp in their names or other issues.")
    print("   Add better logging to see exactly why files are rejected.")
    print()

print("🔧 GENERAL RECOMMENDATIONS:")
print("   1. Set multiprocessing start method to 'spawn' before any CUDA operations")
print("   2. Add diagnostic logging to show which files are found/filtered")
print("   3. Consider reducing num_workers or using num_workers=0 if issues persist")
print("   4. Ensure CUDA operations only happen after worker initialization")
print()

print("=== END DIAGNOSTICS ===")