# PrecomputedDataset Exact Parameter Matching Fix

## Problem Summary

The `PrecomputedDataset` class was loading and attempting to concatenate ALL .npz files for a problem (e.g., `mceg_*.npz`), even when only files with exact requested parameters (ns, ne, nr) were required. This led to dimension mismatch errors when files had different `num_events`, `n_repeat`, or other parameters.

**Example of the problem:**
```
Files in directory:
- mceg_ns1000_ne5000_nr1.npz  (events shape: [1000, 1, 5000, 6])
- mceg_ns2000_ne10000_nr1.npz (events shape: [2000, 1, 10000, 6])

Old behavior: Load both files and try to concatenate
Result: ValueError - dimension mismatch (5000 vs 10000 events)
```

## Solution Implemented

### 1. Parameter Parsing
Added `parse_filename_parameters()` function to extract parameters from filenames:
```python
parse_filename_parameters("mceg_ns1000_ne5000_nr1.npz")
# Returns: {'problem': 'mceg', 'ns': 1000, 'ne': 5000, 'nr': 1}
```

### 2. Exact Parameter Matching
Added `filter_files_by_exact_parameters()` to filter files by exact parameter values:
```python
filter_files_by_exact_parameters(files, required_ns=1000, required_ne=5000, required_nr=1)
# Returns only files with exactly matching parameters
```

### 3. Enhanced PrecomputedDataset
Updated `PrecomputedDataset` to accept exact parameter specifications:
```python
# NEW: Load only files with exact parameters
dataset = PrecomputedDataset(data_dir, "mceg", 
                           exact_ns=1000, exact_ne=5000, exact_nr=1)

# OLD: Load all mceg_*.npz files (could cause dimension errors)
dataset = PrecomputedDataset(data_dir, "mceg")
```

### 4. Integration with parameter_prediction.py
Updated the training pipeline to use exact parameter matching:
```python
# Before
train_dataset = PrecomputedDataset(train_data_dir, args.problem, shuffle=True)

# After  
train_dataset = PrecomputedDataset(train_data_dir, args.problem, shuffle=True,
                                 exact_ns=args.num_samples, 
                                 exact_ne=args.num_events, 
                                 exact_nr=1)
```

## Key Features

### ✅ Exact Parameter Matching
- Only loads files with exactly matching (ns, ne, nr) parameters
- Prevents dimension mismatch errors
- Ensures users get exactly the data they requested

### ✅ Comprehensive Error Handling
- Clear error messages when exact matches aren't found
- Shows available files and their parameters for debugging
- Helpful suggestions for fixing issues

### ✅ Robust Logging
```
🔍 FILTERING FOR EXACT PARAMETERS: ns=1000, ne=5000, nr=1
     ✓ MATCH: 'mceg_ns1000_ne5000_nr1.npz' - ns=1000, ne=5000, nr=1
     ✗ NO MATCH: 'mceg_ns2000_ne10000_nr1.npz' - ns=2000, ne=10000, nr=1
📊 EXACT MATCH RESULT: 1 files match out of 2 total
```

### ✅ Dimension Compatibility Validation
- Detects incompatible dimensions before concatenation
- Prevents silent data corruption
- Provides detailed error information

### ✅ Backward Compatibility
- When no exact parameters specified, behaves as before
- Enhanced error detection and messaging
- Existing code continues to work with better error handling

## Usage Examples

### Training with Specific Parameters
```python
# Generate or ensure data exists for specific parameters
data_dir = generate_precomputed_data_if_needed(
    problem="mceg", 
    num_samples=1000, 
    num_events=5000, 
    n_repeat=1
)

# Load dataset with exact parameter matching
dataset = PrecomputedDataset(data_dir, "mceg",
                           exact_ns=1000, exact_ne=5000, exact_nr=1)
```

### Single File Mode
```python
# Load a specific file
single_file = "precomputed_data/mceg_ns1000_ne5000_nr1.npz"
dataset = PrecomputedDataset(single_file, "mceg")

# Optionally verify parameters match expectations
dataset = PrecomputedDataset(single_file, "mceg",
                           exact_ns=1000, exact_ne=5000, exact_nr=1)
```

### Directory Mode with Multiple Compatible Files
```python
# This would work if all files have compatible dimensions
dataset = PrecomputedDataset("precomputed_data", "mceg")

# This guarantees only exact matches are loaded
dataset = PrecomputedDataset("precomputed_data", "mceg",
                           exact_ns=1000, exact_ne=5000, exact_nr=1)
```

## Error Messages

### When Exact Match Not Found
```
FileNotFoundError: No precomputed data files found with exact parameters: ns=1000, ne=5000, nr=1
Expected file: mceg_ns1000_ne5000_nr1.npz
Searched in: precomputed_data
Available files: ['mceg_ns2000_ne10000_nr1.npz']
Hint: None of the available files match the exact required parameters.
```

### When Dimensions Are Incompatible
```
ValueError: Cannot concatenate files with incompatible dimensions.
This typically happens when files have different num_events, n_repeat, or feature dimensions.
Use exact parameter matching (exact_ns, exact_ne, exact_nr) to load only compatible files.

Files loaded:
  - mceg_ns1000_ne5000_nr1.npz: ns=1000, ne=5000, nr=1
    thetas: (1000, 4), events: (1000, 1, 5000, 6)
  - mceg_ns2000_ne10000_nr1.npz: ns=2000, ne=10000, nr=1  
    thetas: (2000, 4), events: (2000, 1, 10000, 6)
```

## Migration Guide

### For Existing Code
Most existing code will continue to work, but may now raise more informative errors:

```python
# This may now fail with better error messages
dataset = PrecomputedDataset("precomputed_data", "mceg")

# Fix by adding exact parameters
dataset = PrecomputedDataset("precomputed_data", "mceg",
                           exact_ns=1000, exact_ne=5000, exact_nr=1)
```

### For parameter_prediction.py
The integration is automatic - the training pipeline now uses exact parameter matching by default based on the command-line arguments.

## Testing

The fix includes comprehensive testing:
- Parameter parsing validation
- Exact matching logic verification  
- Dimension compatibility checking
- Integration testing with realistic scenarios
- Backward compatibility verification

## Benefits

1. **🚫 No More Dimension Errors**: Eliminates the primary source of dimension mismatch errors
2. **🎯 Precise Data Loading**: Users get exactly the data they request
3. **🔍 Better Debugging**: Clear error messages and detailed logging
4. **📊 Reproducible Results**: Exact parameter matching ensures consistent datasets
5. **🔄 Backward Compatible**: Existing code continues to work with enhanced error handling
6. **🛡️ Data Integrity**: Prevents accidental mixing of incompatible datasets

This fix solves the core issue described in the problem statement while maintaining full backward compatibility and adding robust error handling and logging capabilities.