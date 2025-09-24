# Investigation Report: mceg/mceg4dis Issues in plotting_driver_UQ.py

## Executive Summary

This report documents the investigation of two critical issues in the PDF Parameter Inference plotting system for mceg/mceg4dis problem types:

1. **Laplace Model Failures**: Analytic uncertainty propagation fails and falls back to Monte Carlo
2. **Missing Q² Slice Plotting**: Expected Q² slice curves are not generated in mceg plotting functions

## Investigation Findings

### Issue 1: Laplace Model Failures for mceg/mceg4dis

**Root Cause Identified**: Feature Engineering Inconsistency

**Problem Description**:
The mceg/mceg4dis problem types suffer from a critical mismatch between training and inference feature engineering:

- **During Training** (reload_pointnet function):
  - Uses `log_feature_engineering` from `utils.py`
  - Transforms 2D input (x, Q²) → 6D features
  - PointNet model is trained to expect 6D input

- **During Inference** (plotting functions):
  - Uses NO feature engineering (`xs_tensor = xs_tensor`)
  - Keeps 2D input (x, Q²) unchanged
  - PointNet receives 2D input instead of expected 6D

**Consequences**:
1. Input dimension mismatch causes Laplace model API calls to fail
2. All analytic uncertainty propagation falls back to Monte Carlo sampling
3. Predictions may be incorrect due to the feature engineering mismatch

**Evidence**:
- Lines 139-147 in `plotting_driver_UQ.py` show training uses `log_feature_engineering`
- Lines 2365-2368 in `plotting_UQ_utils.py` show inference skips feature engineering
- Multiple debugging statements added to trace this issue

**Failure Path Analysis**:
The `get_analytic_uncertainty` function has three fallback paths:
1. `predictive_distribution()` method
2. Direct call with `joint=False` 
3. `predict()` method with GLM approximation

All paths fail due to the input dimension mismatch, causing fallback to standard model prediction.

### Issue 2: Missing Q² Slice Plotting

**Root Cause Identified**: Incomplete Implementation

**Problem Description**:
The `plot_PDF_distribution_single_same_plot_mceg` function only generates 2D histograms but completely lacks Q² slice visualization.

**Expected Behavior** (from reference notebook):
- Multiple 1D curves showing PDF behavior at different fixed Q² values
- Each curve shows x vs PDF value for a specific Q² slice  
- Common Q² slices: [0.5, 1.0, 1.5, 2.0, 10.0, 50.0, 200.0]
- Should display uncertainty bands when Laplace model is available

**Current Behavior**:
- Only 2D histogram visualization (reco, true, gen)
- No 1D Q² slice curves generated
- `Q2_slices` parameter is accepted but completely ignored
- Missing integration with uncertainty quantification

**Evidence**:
- Function ends at line 2491 without implementing Q² slice plotting
- Commented incomplete code suggests development was started but never finished
- No Q² slice extraction or 1D plotting logic exists

## Diagnosis Summary

### Primary Issues Identified:

1. **Feature Engineering Mismatch** (Laplace Failures):
   - **Severity**: Critical - Breaks analytic uncertainty propagation
   - **Location**: Inconsistency between training and inference pipelines
   - **Impact**: Forces expensive Monte Carlo fallback, reduces accuracy

2. **Missing Q² Slice Implementation** (Plotting):
   - **Severity**: High - Missing expected visualization functionality  
   - **Location**: `plot_PDF_distribution_single_same_plot_mceg` function
   - **Impact**: Users don't get expected Q² slice curves from reference notebook

### Secondary Issues:

3. **Inadequate Error Handling**: 
   - Silent failures in Laplace API calls
   - No clear diagnostic information when issues occur

4. **Documentation Gaps**:
   - Feature engineering requirements not clearly documented
   - Expected plotting outputs not specified

## Recommended Next Steps

### For Laplace Issues:
1. **Fix Feature Engineering Consistency**:
   - Apply `log_feature_engineering` in all mceg plotting functions
   - Ensure 2D→6D transformation matches training pipeline
   - Verify PointNet input dimensions are consistent

2. **Improve Error Diagnostics**:
   - Add better logging for Laplace model loading failures
   - Provide clear messages when feature engineering mismatches occur

### For Q² Slice Plotting:
1. **Implement Q² Slice Extraction**:
   - Extract 1D slices from existing 2D histogram data
   - Create plotting function for Q² slice curves
   - Integrate with uncertainty quantification when Laplace available

2. **Add Reference Notebook Compatibility**:
   - Match expected output format from mceg4dis reference notebook
   - Ensure consistent styling and Q² slice selection

## Code Changes Made

This investigation added comprehensive debugging statements and comments to:

- `plotting_UQ_utils.py`: Enhanced `get_analytic_uncertainty` with detailed diagnostic logging
- `plotting_UQ_utils.py`: Added feature engineering mismatch warnings in mceg plotting function  
- `plotting_UQ_utils.py`: Documented missing Q² slice implementation with detailed analysis
- `plotting_driver_UQ.py`: Added mceg-specific Laplace diagnostic logging
- `plotting_driver_UQ.py`: Documented critical feature engineering bug in `reload_pointnet`

## Reference Materials

- Reference notebook: https://github.com/quantom-collab/mceg4dis/blob/main/01_get_started.ipynb
- Expected Q² slice visualization patterns
- Feature engineering functions: `utils.log_feature_engineering` vs `simulator.advanced_feature_engineering`

---
*Investigation completed as part of diagnostic task - no code fixes implemented per requirements*