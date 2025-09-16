# Train-Test Split Implementation for parameter_prediction.py

## Summary

This implementation adds a clear train-test split to `parameter_prediction.py` with the following key features:

### 1. **Clear Train-Test Split**
- The `num_samples` parameter now only affects the training dataset size
- Added `--val_samples` parameter (default: 1000) for validation dataset size
- Training and validation datasets are created separately and do not overlap

### 2. **Validation Dataset**
- 1,000 validation samples by default (configurable via `--val_samples`)
- Validation data drawn from the same simulator/problem as training data
- Ensures no overlap between training and validation sets

### 3. **Enhanced Training Loop**
- Modified `train_joint()` function to accept both train and validation dataloaders
- Validation evaluation performed every epoch
- Training and validation losses logged side-by-side each epoch
- Model switches to eval mode during validation, then back to train mode

### 4. **Logging and Monitoring**
- Both training and validation losses logged to console each epoch
- WandB integration updated to log both `train_mse_loss` and `val_mse_loss`
- Format: `Epoch XXX | Train Loss: X.XXXXXX | Val Loss: X.XXXXXX`

### 5. **Code Organization**
- New `create_train_val_datasets()` function handles dataset creation
- Supports both precomputed and on-the-fly data generation
- Works with all existing problem types (simplified_dis, realistic_dis, mceg, gaussian)

## Key Changes Made

### Modified Files:
1. **parameter_prediction.py** - Main implementation with train-test split
2. **test_parameter_prediction_full.py** - Full test implementation
3. **simple_simulator.py** - Simple simulators for testing
4. **test_parameter_prediction.py** - Basic functionality tests
5. **test_comprehensive.py** - Comprehensive validation tests

### New Command Line Arguments:
- `--val_samples`: Number of validation samples (default: 1000)
- Enhanced help text for `--num_samples` to clarify it's for training only

### Function Signatures Changed:
```python
# Old:
train_joint(model, param_prediction_model, dataloader, ...)

# New:
train_joint(model, param_prediction_model, train_dataloader, val_dataloader, ...)
```

## Usage Examples

```bash
# Basic usage with default 1000 validation samples
python parameter_prediction.py --num_samples 5000 --num_epochs 100

# Custom validation set size
python parameter_prediction.py --num_samples 5000 --val_samples 2000 --num_epochs 100

# Different problem types
python parameter_prediction.py --problem gaussian --num_samples 3000 --val_samples 1500
```

## Output Example

```
Starting training with 5000 training samples and 1000 validation samples per rank
Epoch   1 | Train Loss: 2.345678 | Val Loss: 2.123456
Epoch   2 | Train Loss: 1.876543 | Val Loss: 1.765432
Epoch   3 | Train Loss: 1.543210 | Val Loss: 1.456789
...
```

## Validation Tests

All tests pass and validate:
- ✅ Clear train-test split (num_samples only for training)
- ✅ Configurable validation samples (default 1000)
- ✅ Separate train and validation data loading
- ✅ Validation evaluation during training
- ✅ Both training and validation loss logged each epoch
- ✅ Non-overlapping datasets from same simulator
- ✅ Works with different problem types

## Implementation Details

### Dataset Separation
- Training dataset uses `args.num_samples`
- Validation dataset uses `args.val_samples` (default 1000)
- Both datasets use the same simulator but with different random seeds
- Ensures statistical independence between train and validation sets

### Memory Efficiency
- Both datasets use the same DataLoader configuration
- Validation evaluation uses `torch.no_grad()` for memory efficiency
- Model properly switches between train/eval modes

### Distributed Training Support
- Maintains compatibility with multi-GPU training
- Both training and validation datasets support distributed sampling
- Logging only occurs on rank 0 to avoid duplicate output

The implementation successfully meets all requirements while maintaining backward compatibility and the existing codebase structure.