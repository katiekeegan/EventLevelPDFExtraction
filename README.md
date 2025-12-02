# Event-Level ML-Driven Extraction of Parton Distribution Functions with Uncertainty Quantification

This repository implements the full, end-to-end pipeline from the paper “Event-Level ML-Driven Extraction of Parton Distribution Functions with Uncertainty Quantification.”

It is a scalable, amortized, simulation-based inference (SBI) framework that directly consumes whole sets of Deep Inelastic Scattering (DIS) events—without histogramming—to infer parameters of a chosen Parton Distribution Function (PDF) parametrization. PDFs are a specific class of Quantum Correlation Functions (QCFs) characterizing hadronic structure; our method targets PDF extraction while remaining conceptually applicable to event-level inference of related QCFs. The pipeline includes full uncertainty quantification (UQ) with Laplace approximation and bootstrap-based data uncertainty, multi-GPU training, on-the-fly simulation or precomputed datasets, and comprehensive evaluation against standard SBI baselines.


## Overview

- Problem: Infer PDF parameters describing momentum-fraction distributions of partons (quarks, gluons) from DIS events.
- Problem: Infer PDF parameters describing momentum-fraction distributions of partons (quarks, gluons) from DIS events. PDFs belong to the broader family of QCFs (Quantum Correlation Functions) used to encode fundamental properties of hadronic systems.
- Limitation of prior approaches: Event-level data are reduced to histograms/cross-sections; classical SBI often requires per-observation conditioning and retraining, limiting scalability.
- Core idea: Amortized, event-level ML. Treat each event set X(θ) of size ~10^4 as a permutation-invariant set. Embed sets with a PointNet-PMA encoder, then predict θ with a simple MLP head. Avoids histogramming entirely.
- Uncertainty: Epistemic via Laplace approximation around the last layer; data uncertainty via parametric bootstrap (re-simulation at the same θ). Combine via the law of total variance.
- Scalability: Chunked processing of large event sets and multi-GPU training; supports on-GPU simulation where available.


## Pipeline at a Glance

Simulation → Event sets X(θ) → PointNet-PMA encoder (chunked) → Latent vector z → MLP head → θ̂ → UQ (Laplace + bootstrap) → PDF reconstruction and plots.

- Simulation: Generate event sets on-the-fly or load precomputed sets.
- Encoding: `ChunkedPointNetPMA` maps a large set of events to a fixed-dimensional latent vector z, using a tiny point encoder per chunk and Pooling-by-Multihead-Attention (PMA) over chunk summaries.
- Regression head: `MLPHead` (or optional Transformer head) predicts the parameter vector θ.
- UQ: Laplace approx on the final layer + bootstrap resimulation for data uncertainty; combine via LoTV.
- Outputs: Parameters, posterior samples, PDF functions and bands, decomposition plots, and baselines comparisons.


## Repository Structure

- `parameter_prediction.py` — Train the end-to-end model: PointNet-PMA encoder + MLP head. Supports precomputed data, multi-GPU, and W&B logging.
- `datasets.py` — Iterable datasets for different simulators; on-the-fly sampling, feature engineering, and distributed sharding.
- `simulator.py` — Simulators for: Gaussian 2D proxy, simplified DIS, realistic DIS (x, Q^2), and optional MCEG-based generator.
- `models.py` — PointNet/PMA encoders, chunked event-set encoder (`ChunkedPointNetPMA`), MLP/Transformer heads, and utility regressors.
- `plotting_driver_UQ.py` — End-to-end plotting driver with Laplace-based analytic UQ and bootstrap; produces figures and tables.
- `generate_precomputed_data.py` — Utilities to pre-generate exact-matching datasets for training/validation.
- `precomputed_datasets.py` — Loaders for `.npz` precomputed data (exact parameter matching, distributed-friendly). 
- `plotting_UQ_utils.py`, `plotting_UQ_helpers.py` — UQ utilities: posterior sampling, LoTV decomposition, figure generation.
- `run_sbi_simplified_baseline.py`, `sbibm_benchmark*.py` — SNPE and ABC baselines.
- `experiments/` — Training artifacts per run (models, checkpoints, plots).


## Installation and Environment

We provide both a conda `environment.yml` and a pip `requirements.txt`. Python 3.11 is recommended.

### Option A: Conda environment

```bash
# Create and activate
conda env create -f environment.yml
conda activate simformer-env

# Install PyTorch wheel suited to your setup
# CPU-only:
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.1.0
# For CUDA (example for CUDA 11.8):
# pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.1.0
```

### Option B: Pip venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# For CUDA, install torch from the PyTorch index matching your CUDA version
```

Notes:
- GPU strongly recommended for training; multi-GPU supported via PyTorch DDP.
- Optional MCEG components require additional physics dependencies in `mceg4dis/`.


## Data and Simulators

Two settings are supported:

1) Proxy 1-D problem (events contain x only): `SimplifiedDIS` with 4-parameter ansatz [au, bu, ad, bd], events transformed to features. 
2) Realistic 2-D problem (events contain (x, Q^2)): `RealisticDIS` and optional `MCEGSimulator` producing (x, Q^2) samples under a 4-parameter PDF ansatz.

- On-the-fly simulation: datasets sample θ uniformly within bounds and simulate `num_events` per θ, possibly repeated (n_repeat) for bootstraps.
- Precomputed datasets: exact-matching `.npz` files can be generated and reused; training and validation datasets are separated.

Expected shapes:
- Training batch from `DataLoader`: `(theta, x_sets)`
  - `theta`: shape (B, Dθ)
  - `x_sets`: shape (B, n_repeat, num_events, F)

Feature engineering:
- We use log/advanced feature engineering to stabilize inputs; see `utils.py` and dataset classes for the exact transforms.


## Model Architecture

PointNet-PMA encoder (chunked):
- `ChunkedPointNetPMA` in `models.py` handles very large event sets.
- Steps:
  - Pad and split events into fixed-size chunks (K, e.g., 4096).
  - Encode points in each chunk with a tiny MLP (`SmallPointEncoder`).
  - Pool chunk summaries with PMA: learnable seed vectors attend over chunk summaries using multihead attention.
  - Flatten seeds and project to latent `z` of dimension `latent_dim`.

Head:
- `MLPHead(latent_dim, out_dim)` maps z → θ.
- A `TransformerHead` is available for ablations.

Training:
- Joint training of encoder and head with Adam, gradient clipping, and AMP scaler.
- Loss: Unnormalized MSE in original parameter units.
- Parameter bounds are printed for diagnostics; inputs are normalized internally for stability where needed.


## Uncertainty Quantification (UQ)

- Epistemic (model) uncertainty: Laplace approximation around the final layer (subset_of_weights=last_layer, hessian=kron). Saved and reused for inference.
- Data uncertainty: Parametric bootstrap—simulate multiple event sets at the same θ, re-embed and re-predict.
- Combined uncertainty: Law of Total Variance (LoTV) to decompose posterior vs. data contributions.
- Utilities: posterior sampling, analytic propagation (delta method), function-space uncertainty (PDF bands), and decomposition plots.


## How to Train from Scratch

Minimal single-GPU run (simplified DIS):

```bash
python parameter_prediction.py \
  --problem simplified_dis \
  --num_samples 10000 \
  --val_samples 1000 \
  --num_events 10000 \
  --latent_dim 512 \
  --batch_size 32 \
  --num_epochs 1000 \
  --lr 1e-4 \
  --dataloader_workers 0 \
  --single_gpu
```

- Output dir: `experiments/{problem}_latent{latent}_ns_{num_samples}_ne_{num_events}_parameter_predidction/`
  - Saves `final_model.pth` (PointNet encoder) and `final_params_model.pth` (MLP head).
  - Periodic checkpoints `model_epoch_*.pth`, `params_model_epoch_*.pth`.
- Precomputed data: add `--use_precomputed --precomputed_data_dir precomputed_data`. If exact files are missing, they will be generated via `generate_precomputed_data.py`.
- Multi-GPU: omit `--single_gpu` and run on N GPUs; script will spawn DDP ranks automatically. Consider `--dataloader_workers 1` or more; if you encounter CUDA fork errors, set it to 0.

MCEG or realistic DIS:

```bash
# Realistic DIS (x, Q^2)
python parameter_prediction.py --problem realistic_dis --num_events 100000 --latent_dim 512 --single_gpu

# MCEG-based DIS (requires mceg4dis deps)
python parameter_prediction.py --problem mceg --num_events 100000 --latent_dim 1024 --single_gpu
```

W&B logging:
- Enable with `--wandb`. If initialization fails, training continues without logging.


## Inference and Plotting (Paper Figures)

Use the UQ plotting driver to reproduce figures with analytic Laplace propagation and bootstrap:

```bash
python plotting_driver_UQ.py \
  --arch all \
  --latent_dim 1024 \
  --param_dim 4 \
  --problem simplified_dis \
  --num_events 100000 \
  --n_bootstrap 100
```

- Produces: parameter distributions, PDF function posteriors, uncertainty bands, UMAP/TSNE latent visualizations, error summaries, and LaTeX metrics tables under `experiments/.../plots_*`.
- For MCEG-like problems use `--problem mceg4dis` (2D inputs); driver handles Q^2 slices and SBI overlays.
- The driver attempts to load saved Laplace objects; if missing, it can fit a Laplace approx on extracted latents.

Specific examples:

```bash
# Simplified DIS with analytic UQ (Laplace), plus SBI comparisons
python plotting_driver_UQ.py --arch mlp --latent_dim 512 --param_dim 4 --problem simplified_dis --num_events 100000

# MCEG4DIS (2D inputs: x, Q^2) with combined uncertainty plots and SBI overlays
python plotting_driver_UQ.py --arch transformer --latent_dim 1024 --param_dim 4 --problem mceg4dis --num_events 100000
```

Input/Output contract for plotting driver:
- Loads trained models from `experiments/` (PointNet encoder `final_model.pth`, head `final_params_model.pth`).
- Optional SBI baseline samples from `samples_snpe*.txt`, `samples_wasserstein*.txt`, `samples_mmd*.txt`.
- Writes plots and tables to `experiments/.../plots_{arch}/`.

## Cluster/SLURM usage and practical defaults

SLURM batch scripts used in production runs are provided in the repo and reflect tested defaults on GPU clusters:

- `mceg_cl.sh` (simplified DIS, single GPU):
  - `--problem simplified_dis --latent_dim 512 --num_events 10000 --num_samples 10000`
  - `--batch_size 32 --num_epochs 1000 --num_repeat 1 --lr 5e-5 --use_precomputed --single_gpu`

- `mceg_cl_ne_100000.sh` (MCEG, multi-GPU-ready parameters, using precomputed data):
  - `--problem mceg --latent_dim 1024 --num_events 100000 --num_samples 10000`
  - `--batch_size 64 --num_epochs 2000 --use_precomputed`

- `mceg_parameter_prediction.sh` and `mceg_parameter_prediction_ne_100000.sh` (cluster jobs with W&B logging):
  - `--latent_dim 1024 --num_events 10000 --num_samples 10000 --batch_size 64 --num_epochs 2000 --use_precomputed --wandb`

These scripts also set SLURM headers and environment modules (e.g., `pytorch/2.6.0`) and export `WANDB_ENTITY`. Adapt module loads to your site; the Python entrypoint is always `parameter_prediction.py`.

Tip: when using multiple GPUs, omit `--single_gpu`. For DataLoader stability on clusters, start with `--dataloader_workers 0` if you see CUDA fork warnings.


## Baselines and Benchmarks

- SNPE (sbi): `run_sbi_simplified_baseline.py` and `sbibm_benchmark*.py`
- MCABC with L2 distance; Wasserstein ABC
- Baselines use histogramming and per-dataset retraining; our method operates directly on event sets and amortizes inference.
- Reproduction files: `samples_snpe*.txt`, `samples_wasserstein*.txt`, `samples_mmd*.txt` used by the plotting driver for overlays.


## Data Formats and Chunking Details

- Event set X(θ): `(num_events, F)` features per repetition; aggregated in the dataloader as `(B, n_repeat, num_events, F)`.
- Chunking (`ChunkedPointNetPMA`): pad to a multiple of K, reshape to `(B * n_chunks, K, F)`, encode per chunk, aggregate via PMA with a small number of seeds, then project to latent.
Typical defaults (informed by the provided SLURM jobs):
- Simplified DIS (`mceg_cl.sh`):
  - `latent_dim`: 512
  - `num_events`: 10,000
  - `num_samples`: 10,000
  - `batch_size`: 32
  - `num_epochs`: 1,000
  - `num_repeat`: 1
  - `lr`: 5e-5
  - `use_precomputed`: true
  - `single_gpu`: true
- MCEG (`mceg_cl_ne_100000.sh`):
  - `latent_dim`: 1,024
  - `num_events`: 100,000
  - `num_samples`: 10,000
  - `batch_size`: 64
  - `num_epochs`: 2,000
  - `use_precomputed`: true
- General encoder/head internals (from `models.py` and `parameter_prediction.py`):
  - `ChunkedPointNetPMA` with PMA and small per-chunk encoder
  - `chunk_size`: 4,096 (padding to multiples in forward)
  - `chunk_latent`: 128 for MCEG; 32 for simplified/realistic DIS
  - `num_seeds`: 8
  - `num_heads`: 4
  - Head: `MLPHead(embedding_dim=latent_dim, out_dim=Dθ)`

Hyperparameters (see `parameter_prediction.py`) and practical values:
- Flags: `--num_samples`, `--val_samples`, `--num_events`, `--num_repeat`, `--batch_size`, `--num_epochs`, `--lr`, `--latent_dim`, `--dataloader_workers`, `--use_precomputed`, `--single_gpu`.
- Problems: `simplified_dis`, `realistic_dis`, `mceg`, `gaussian`.
- Typical values (from the job scripts): see the list above; adapt to your hardware.


## Uncertainty Visualization and LoTV

The repository provides:
- Posterior parameter uncertainty via Laplace approx (analytic delta method).
- Data (bootstrap) uncertainty by repeating simulation at fixed θ.
- Combined uncertainty using the law of total variance with function-space decomposition.
- Plots: parameter histograms, function-space bands, PDF overlays at multiple Q^2 slices, error histograms, UMAP/TSNE of latents.


## Extending the Framework

Add a new PDF parametrization or simulator:

1) Define the simulator in `simulator.py`:
   - Implement `.sample(theta, n_events)` returning an `(n_events, D_event)` tensor.
   - Implement a `.f(x, theta)` (or equivalent) to evaluate the PDF/cross-section for plotting.
   - Provide sensible parameter bounds.

2) Add a dataset in `datasets.py`:
   - Create an `IterableDataset` that samples θ within bounds, simulates events, applies feature engineering, and yields `(theta, x_sets)`.
   - Ensure shapes match `(n_repeat, num_events, F)` for each sample.

3) Update model input dimension:
   - Determine feature dimension F after engineering; set `input_dim` when constructing `ChunkedPointNetPMA` in `parameter_prediction.py`.

4) Plotting support:
   - Add handling to `plotting_driver_UQ.py` and helpers for the new simulator’s function evaluation and plots.

Contract summary:
- Inputs: set of events `x ∈ R^{N×F}` (per repetition).
- Outputs: θ̂ ∈ R^{Dθ}; posterior samples for UQ; PDF functions and UQ bands.
- Error modes: non-finite loss, CUDA multiprocessing fork issues; set `--dataloader_workers 0` or enforce spawn, and check bounds.


## Reproducibility Notes

- Multi-GPU training uses PyTorch DDP; the script sets `spawn` to avoid CUDA in forked subprocess errors.
- Validation uses separate datasets (`--val_samples`) and original-unit MSE.
- Laplace models saved when available; plotting driver can refit if missing.
- Experiment naming encodes settings; plots and tables are written under the same directory.


## How to Run Baselines

SNPE / ABC baseline samples are provided and used by the plotting driver. To regenerate:

```bash
# Example: run SNPE baseline for simplified DIS
python run_sbi_simplified_baseline.py --num_events 10000 --num_samples 1000
# Generated samples are read by plotting_driver_UQ.py
```

Consult the `sbibm_benchmark*.py` scripts for benchmark-specific configurations.


## GPU Requirements

- NVIDIA GPU and recent CUDA recommended for large event sets; CPU-only is possible but slower.
- For multi-GPU, ensure `nccl` backend availability and adequate GPU memory; training scales with `num_events`, `latent_dim`, and chunk sizes.

## License

This project is licensed under the terms in `LICENSE`.


## Acknowledgments

- The MCEG components depend on external physics packages in `mceg4dis/`.