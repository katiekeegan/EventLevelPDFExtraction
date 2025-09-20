#!/usr/bin/env python3
"""
Script to fix precomputed data files with .tmp extensions.

This script helps fix the common issue where precomputed data files have .tmp 
in their names, causing them to be filtered out as temporary/incomplete files.
"""

import os
import glob
import shutil

def fix_tmp_files(data_dir="precomputed_data"):
    """
    Find and fix precomputed data files with .tmp in their names.
    
    Args:
        data_dir: Directory containing precomputed data files
    """
    print("🔧 FIXING PRECOMPUTED DATA FILES WITH .tmp EXTENSIONS")
    print(f"   Scanning directory: {data_dir}")
    print()
    
    if not os.path.exists(data_dir):
        print(f"   ❌ Directory {data_dir} does not exist")
        return
    
    # Find all .npz files with .tmp in their names
    pattern = os.path.join(data_dir, "*.tmp.npz")
    tmp_files = glob.glob(pattern)
    
    print(f"   Found {len(tmp_files)} files with .tmp extensions:")
    
    if not tmp_files:
        print("   ✅ No .tmp files found - no action needed")
        return
    
    for tmp_file in tmp_files:
        filename = os.path.basename(tmp_file)
        print(f"   📁 Processing: {filename}")
        
        # Generate the fixed filename by removing .tmp
        if '.tmp.npz' in filename:
            fixed_filename = filename.replace('.tmp.npz', '.npz')
        elif '.tmp' in filename:
            # Handle other .tmp patterns 
            fixed_filename = filename.replace('.tmp', '')
        else:
            print(f"      ⚠️  File doesn't contain .tmp: {filename}")
            continue
        
        fixed_path = os.path.join(data_dir, fixed_filename)
        
        # Check if the fixed file already exists
        if os.path.exists(fixed_path):
            print(f"      ⚠️  Target file already exists: {fixed_filename}")
            print(f"      💡 Consider removing the temporary file manually")
            continue
        
        # Rename the file
        try:
            shutil.move(tmp_file, fixed_path)
            print(f"      ✅ Renamed to: {fixed_filename}")
        except Exception as e:
            print(f"      ❌ Failed to rename: {e}")
    
    print()
    print("🔍 VERIFICATION:")
    
    # Verify the results
    from parameter_prediction import filter_valid_precomputed_files
    
    problems = ["mceg", "gaussian", "simplified_dis", "realistic_dis"]
    for problem in problems:
        pattern = os.path.join(data_dir, f"{problem}_*.npz")
        all_files = sorted(glob.glob(pattern))
        valid_files = filter_valid_precomputed_files(all_files)
        
        if all_files:
            print(f"   Problem '{problem}': {len(valid_files)} valid out of {len(all_files)} total files")
            if len(valid_files) < len(all_files):
                invalid_files = [f for f in all_files if f not in valid_files]
                print(f"      ⚠️  Still invalid: {invalid_files}")
    
    print()
    print("✅ Fix completed!")

if __name__ == "__main__":
    fix_tmp_files()