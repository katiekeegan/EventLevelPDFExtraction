# Precomputed Data Pipeline for PDF Parameter Inference

This document describes the new precomputed data pipeline that allows for faster, more reproducible training by generating datasets in advance rather than on-the-fly during training.

## Overview

The precomputed data pipeline addresses several issues with the original on-the-fly data generation:

- **Reproducibility**: Datasets are generated once and reused, ensuring consistent training data across runs
- **Performance**: Training iterations are faster since no simulation is required during training
- **Debugging**: Easier to debug with fixed datasets
- **Experimentation**: Compare different models on exactly the same data

## Quick Start

### 1. Generate Precomputed Data

```bash
# Generate data for Gaussian problem
python generate_precomputed_data.py --problems gaussian --num_samples 10000 --num_events 1000

# Generate data for all problems
python generate_precomputed_data.py --problems gaussian simplified_dis realistic_dis mceg --num_samples 5000 --num_events 1000

# Generate data with custom parameters
python generate_precomputed_data.py \
    --problems simplified_dis \
    --num_samples 20000 \
    --num_events 2000 \
    --n_repeat 3 \
    --output_dir /path/to/data
```

### 2. Train with Precomputed Data

```bash
# Train using precomputed data
python end_to_end.py \
    --problem gaussian \
    --use_precomputed \
    --precomputed_data_dir precomputed_data \
    --batch_size 128 \
    --num_epochs 100

# Train using original on-the-fly generation (for comparison)
python end_to_end.py \
    --problem gaussian \
    --num_samples 10000 \
    --num_events 1000 \
    --batch_size 128 \
    --num_epochs 100
```

## Data Generation

### Command Line Options

```bash
python generate_precomputed_data.py --help
```

Key arguments:
- `--problems`: Which problems to generate data for (`gaussian`, `simplified_dis`, `realistic_dis`, `mceg`)
- `--num_samples`: Number of theta parameter samples to generate
- `--num_events`: Number of events per simulation
- `--n_repeat`: Number of repeated simulations per theta (for data augmentation)
- `--output_dir`: Directory to save .npz files
- `--device`: Computation device (`cpu`, `cuda`, `auto`)

### Generated File Format

Data is saved in compressed NumPy format (`.npz`) with the following structure:

```
{problem}_ns{num_samples}_ne{num_events}_nr{n_repeat}.npz
├── thetas: [num_samples, theta_dim] - Parameter vectors
├── events: [num_samples, n_repeat, num_events, feature_dim] - Event data
├── problem: Problem name (string)
├── num_samples: Number of samples
├── num_events: Number of events per simulation
├── n_repeat: Number of repetitions
├── theta_shape: Shape of theta array
└── events_shape: Shape of events array
```

### Problem-Specific Details

#### Gaussian (problem='gaussian')
- **Theta dimensions**: 5 (mu_x, mu_y, sigma_x, sigma_y, rho)
- **Event dimensions**: 2 (x, y coordinates)
- **Parameter ranges**: 
  - mu_x, mu_y: [-2, 2]
  - sigma_x, sigma_y: [0.5, 2.0]
  - rho: [-0.8, 0.8]

#### Simplified DIS (problem='simplified_dis')
- **Theta dimensions**: 4 (au, bu, ad, bd)
- **Event dimensions**: 3 (x, Q2, F2) after feature engineering
- **Parameter ranges**: [0, 5] for all parameters

#### Realistic DIS (problem='realistic_dis')
- **Theta dimensions**: 6 (logA0, delta, a, b, c, d)
- **Event dimensions**: 3 (x, Q2, F2) after feature engineering
- **Parameter ranges**:
  - logA0: [-2, 2]
  - delta: [-1, 1]
  - a: [0, 5]
  - b: [0, 10]
  - c, d: [-5, 5]

#### MCEG (problem='mceg')
- **Theta dimensions**: 4
- **Event dimensions**: 2 after feature engineering
- **Parameter ranges**: 
  - First param: [-1, 10]
  - Second param: [0, 10]
  - Third, fourth: [-10, 10]

## Training with Precomputed Data

### Dataset Classes

The pipeline provides two main dataset classes:

- `PrecomputedDataset`: Standard dataset for single-process training
- `DistributedPrecomputedDataset`: Automatically splits data across distributed training ranks

### Usage in Training Scripts

```python
from precomputed_datasets import create_precomputed_dataloader

# Create dataloader for precomputed data
dataloader = create_precomputed_dataloader(
    data_dir='precomputed_data',
    problem='gaussian',
    batch_size=32,
    shuffle=True,
    rank=0,
    world_size=1
)

# Use in training loop
for theta, events in dataloader:
    # theta: [batch_size, theta_dim]
    # events: [batch_size, n_repeat, num_events, feature_dim]
    # ... training code ...
```

### Distributed Training

The precomputed data pipeline fully supports distributed training:

```python
# Automatically splits data across ranks
dataloader = create_precomputed_dataloader(
    data_dir='precomputed_data',
    problem='simplified_dis', 
    batch_size=64,
    rank=local_rank,
    world_size=world_size
)
```

Each rank gets a different subset of the data, ensuring no duplication while maintaining load balance.

## Backward Compatibility

The enhanced training script maintains full backward compatibility:

- **Default behavior**: Still uses on-the-fly generation unless `--use_precomputed` is specified
- **Same API**: All existing command line arguments work unchanged
- **Same results**: Training with precomputed data should give equivalent results to on-the-fly generation

## Performance Comparison

### Benefits of Precomputed Data

1. **Faster training iterations**: No simulation overhead during training
2. **Reproducible experiments**: Identical data across runs
3. **Better resource utilization**: Separate data generation from training
4. **Easier debugging**: Fixed datasets enable better error analysis

### When to Use Each Approach

**Use precomputed data when**:
- Running multiple experiments on the same problem
- Need reproducible results
- Training time is more important than storage space
- Debugging training issues

**Use on-the-fly generation when**:
- Experimenting with different simulation parameters
- Storage space is limited
- Need maximum data diversity (different samples each epoch)

## File Organization

Recommended directory structure:

```
project/
├── precomputed_data/          # Generated datasets
│   ├── gaussian_ns10000_ne1000_nr2.npz
│   ├── simplified_dis_ns10000_ne1000_nr2.npz
│   └── ...
├── experiments/               # Training outputs
│   ├── gaussian_precomputed_experiment1/
│   └── gaussian_onthefly_experiment1/
└── scripts/
    ├── generate_precomputed_data.py
    ├── end_to_end.py
    └── precomputed_datasets.py
```

## Troubleshooting

### Common Issues

1. **Import errors for simulators**
   - The data generation script includes fallback minimal simulators
   - Warning message indicates when fallbacks are used
   - Generated data is still valid for training

2. **CUDA out of memory during generation**
   - Use `--device cpu` to generate on CPU
   - Reduce `--num_samples` or `--num_events`
   - Generate data in smaller batches

3. **No data files found**
   - Check that file names match expected pattern
   - Verify `--precomputed_data_dir` path is correct
   - Ensure data generation completed successfully

4. **Dimension mismatches**
   - Verify problem type matches between generation and training
   - Check that feature engineering is consistent

### Debugging Commands

```bash
# Check generated data
python -c "
import numpy as np
data = np.load('precomputed_data/gaussian_ns1000_ne500_nr2.npz')
print('Keys:', list(data.keys()))
print('Thetas shape:', data['thetas'].shape)
print('Events shape:', data['events'].shape)
"

# Test dataset loading
python -c "
from precomputed_datasets import PrecomputedDataset
dataset = PrecomputedDataset('precomputed_data', 'gaussian')
print('Dataset length:', len(dataset))
print('Metadata:', dataset.get_metadata())
"
```

## Future Enhancements

Potential improvements to the precomputed data pipeline:

1. **Incremental generation**: Add new samples to existing datasets
2. **Data validation**: Verify precomputed data matches on-the-fly generation
3. **Compression options**: Different storage formats (HDF5, PyTorch tensors)
4. **Metadata tracking**: Store generation parameters and git hashes
5. **Automatic caching**: Generate data automatically when missing

## Examples

See the `examples/` directory for complete usage examples:

- `example_data_generation.py`: Generate datasets for all problems
- `example_training_comparison.py`: Compare precomputed vs on-the-fly training
- `example_distributed_training.py`: Multi-GPU training with precomputed data