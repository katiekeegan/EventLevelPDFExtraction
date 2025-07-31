# NLL Loss Enhancement - Usage Guide

## Overview

The PDF_learning.py script now supports a new `--nll-loss` command-line flag that enables Gaussian negative log-likelihood (NLL) loss mode as an alternative to the default MSE loss.

## Usage

### Default Mode (MSE Loss)
```bash
python PDF_learning.py --epochs 1000 --problem simplified_dis
```

### NLL Loss Mode 
```bash
python PDF_learning.py --nll-loss --epochs 1000 --problem simplified_dis
```

### Combined with Other Options
```bash
python PDF_learning.py --nll-loss --epochs 500 --problem realistic_dis --latent_dim 512 --gpus 2
```

## What Changes in NLL Mode

1. **Model Architecture**: InferenceNet outputs both parameter means and log-variances instead of just parameter values

2. **Loss Function**: Uses Gaussian NLL loss instead of MSE loss:
   - NLL = 0.5 * (log(σ²) + (x-μ)²/σ²) 
   - Provides uncertainty estimates through predicted variances

3. **Numerical Stability**: Log-variances are clamped to [-10, 10] range to prevent numerical issues

## Backward Compatibility

- When `--nll-loss` is NOT used, the system behaves exactly as before (MSE loss)
- All existing command-line arguments work the same way
- No breaking changes to existing functionality

## Technical Details

- **Means**: Predicted parameter values (same as original MSE mode)
- **Log-variances**: Predicted uncertainty for each parameter (for numerical stability)
- **Parameter Normalization**: Applied consistently in both modes for fair comparison
- **Training**: Both single-GPU and multi-GPU training paths supported

## Testing

Run the included tests to verify functionality:
```bash
python test_nll_loss.py      # Core functionality tests
python test_cli_args.py      # Command-line argument tests  
python test_integration.py   # Training loop integration tests
```

All tests should pass, confirming both modes work correctly.