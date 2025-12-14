#!/usr/bin/env python3
"""
Script to generate precomputed datasets for PDF Parameter Inference training.

This script generates training data (theta parameters and corresponding simulated features)
in advance and saves them to disk in .npz format for faster, more reproducible training.
"""

import argparse
import os

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable


import warnings

# Handle the MCEG import issue gracefully
try:
    from simulator import (MCEGSimulator,
                           SimplifiedDIS)
    from utils import improved_feature_engineering, log_feature_engineering

    FULL_SIMULATORS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Full simulators not available: {e}. Using minimal implementations.")
    FULL_SIMULATORS_AVAILABLE = False


def atomic_savez_compressed(filepath, **kwargs):
    import os
    import time

    try:
        import torch.distributed as dist

        distributed_available = dist.is_available() and dist.is_initialized()
    except (ImportError, RuntimeError):
        distributed_available = False

    if distributed_available:
        rank = dist.get_rank()
        process_id = os.getpid()
        print(
            f"[atomic_savez_compressed] Rank {rank}, PID {process_id}: Starting save to {filepath}"
        )

        if rank == 0:
            # Only rank 0 performs the save and rename operation
            # Remove .npz extension for temp file since np.savez_compressed adds it
            base_path = filepath[:-4] if filepath.endswith(".npz") else filepath
            tmpfile = base_path + ".tmp"
            print(
                f"[atomic_savez_compressed] Rank {rank}, PID {process_id}: Saving to temp file: {tmpfile}"
            )
            try:
                np.savez_compressed(tmpfile, **kwargs)
                # np.savez_compressed adds .npz extension
                actual_tmpfile = tmpfile + ".npz"
            except Exception as e:
                print(
                    f"[atomic_savez_compressed] Rank {rank}, PID {process_id}: Failed to save temp file {tmpfile}: {e}"
                )
                raise
            if not os.path.exists(actual_tmpfile):
                print(
                    f"[atomic_savez_compressed] Rank {rank}, PID {process_id}: Temp file {actual_tmpfile} was not created!"
                )
                raise FileNotFoundError(
                    f"Temp file {actual_tmpfile} not found after save."
                )
            os.replace(actual_tmpfile, filepath)
            print(
                f"[atomic_savez_compressed] Rank {rank}, PID {process_id}: Renamed {actual_tmpfile} to {filepath}"
            )

        # All ranks wait at barrier
        print(
            f"[atomic_savez_compressed] Rank {rank}, PID {process_id}: Waiting at barrier"
        )
        dist.barrier()

        if rank != 0:
            # Other ranks poll for the file to appear
            print(
                f"[atomic_savez_compressed] Rank {rank}, PID {process_id}: Polling for file {filepath}"
            )
            max_wait_time = 300  # 5 minutes
            wait_interval = 1.0  # Check every second
            total_waited = 0

            while not os.path.exists(filepath) and total_waited < max_wait_time:
                time.sleep(wait_interval)
                total_waited += wait_interval
                if total_waited % 10 == 0:  # Log every 10 seconds
                    print(
                        f"[atomic_savez_compressed] Rank {rank}, PID {process_id}: Still waiting for {filepath} (waited {total_waited}s)"
                    )

            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"[atomic_savez_compressed] Rank {rank}, PID {process_id}: File {filepath} not found after waiting {max_wait_time}s"
                )

            print(
                f"[atomic_savez_compressed] Rank {rank}, PID {process_id}: File {filepath} found after {total_waited}s"
            )
    else:
        # Non-distributed case - behave as before
        process_id = os.getpid()
        print(
            f"[atomic_savez_compressed] PID {process_id}: Saving to {filepath} (non-distributed)"
        )
        # Remove .npz extension for temp file since np.savez_compressed adds it
        base_path = filepath[:-4] if filepath.endswith(".npz") else filepath
        tmpfile = base_path + ".tmp"
        print(
            f"[atomic_savez_compressed] PID {process_id}: Saving to temp file: {tmpfile}"
        )
        try:
            np.savez_compressed(tmpfile, **kwargs)
            # np.savez_compressed adds .npz extension
            actual_tmpfile = tmpfile + ".npz"
        except Exception as e:
            print(
                f"[atomic_savez_compressed] PID {process_id}: Failed to save temp file {tmpfile}: {e}"
            )
            raise
        if not os.path.exists(actual_tmpfile):
            print(
                f"[atomic_savez_compressed] PID {process_id}: Temp file {actual_tmpfile} was not created!"
            )
            raise FileNotFoundError(f"Temp file {actual_tmpfile} not found after save.")
        os.replace(actual_tmpfile, filepath)
        print(
            f"[atomic_savez_compressed] PID {process_id}: Renamed {actual_tmpfile} to {filepath}"
        )

def get_simulators_and_utils():
    """Get simulators and utility functions, falling back to minimal versions if needed."""
    if FULL_SIMULATORS_AVAILABLE:
        return (
            SimplifiedDIS,
            MCEGSimulator,
            log_feature_engineering,
        )
    else:
        raise RuntimeError("Full simulators not availabl. Aborting precomputed data generation.")


def generate_theta_samples(problem, num_samples, device):
    """Generate theta parameter samples for a given problem."""
    if problem == "gaussian":
        # [mu_x, mu_y, sigma_x, sigma_y, rho]
        theta_bounds = torch.tensor(
            [
                [-2.0, 2.0],  # mu_x
                [-2.0, 2.0],  # mu_y
                [0.5, 2.0],  # sigma_x
                [0.5, 2.0],  # sigma_y
                [-0.8, 0.8],  # rho
            ],
            device=device,
        )
        theta_dim = 5
    elif problem == "simplified_dis":
        # [au, bu, ad, bd]
        theta_bounds = torch.tensor(
            [
                [-1.0, 0.0],
                [0.0, 5.0],
                [-1.0, 0.0],
                [0.0, 5.0],
            ],
            device=device,
        )
        theta_dim = 4
    elif problem == "mceg":
        # MCEG parameters
        theta_bounds = torch.tensor(
            [
                [-1.0, 10.0],
                [0.0, 10.0],
                [-10.0, 10.0],
                [-10.0, 10.0],
            ],
            device=device,
        )
        theta_dim = 4
    else:
        raise ValueError(f"Unknown problem: {problem}")

    # Generate random samples within bounds
    thetas = torch.rand(num_samples, theta_dim, device=device)
    thetas = thetas * (theta_bounds[:, 1] - theta_bounds[:, 0]) + theta_bounds[:, 0]

    return thetas


def generate_data_for_problem(
    problem, num_samples, num_events, n_repeat, device, output_dir
):
    """Generate and save data for a specific problem."""

    # Get appropriate simulators and utilities
    (
        SimplifiedDISSim,
        MCEGSim,
        log_feat_eng
    ) = get_simulators_and_utils()

    # Create simulator
    if problem == "simplified_dis":
        simulator = SimplifiedDISSim(device=device)
        feature_engineering = log_feat_eng
    elif problem == "mceg":
        simulator = MCEGSim(device=device)
        feature_engineering = log_feat_eng
    else:
        raise ValueError(f"Unknown problem: {problem}")

    print(f"Generating data for {problem}...")
    print(f"  num_samples: {num_samples}")
    print(f"  num_events: {num_events}")
    print(f"  n_repeat: {n_repeat}")
    print(f"  device: {device}")

    # Generate theta samples
    thetas = generate_theta_samples(problem, num_samples, device)
    print(f"  theta shape: {thetas.shape}")

    # Generate corresponding event data
    all_theta_data = []
    all_event_data = []

    for i in tqdm(range(num_samples), desc=f"Generating {problem} data"):
        theta = thetas[i]

        # Generate n_repeat simulations for this theta
        event_sets = []
        for _ in range(n_repeat):
            try:
                # Sample events
                events = simulator.sample(theta, num_events)

                # Apply feature engineering if specified
                if feature_engineering is not None:
                    events = feature_engineering(events)

                # Ensure events is on CPU for storage
                events = events.cpu()
                event_sets.append(events)

            except Exception as e:
                print(f"Warning: Failed to generate data for theta {i}: {e}")
                # Generate dummy data to maintain consistency
                if problem == "gaussian":
                    dummy_events = torch.randn(num_events, 2)
                elif problem == "mceg":
                    dummy_events = torch.randn(num_events, 2)
                else:
                    dummy_events = torch.randn(num_events, 3)
                event_sets.append(dummy_events)

        # Stack the repeated simulations
        stacked_events = torch.stack(
            event_sets, dim=0
        )  # [n_repeat, num_events, feature_dim]

        all_theta_data.append(theta.cpu())
        all_event_data.append(stacked_events)

    # Convert to tensors
    all_thetas = torch.stack(all_theta_data, dim=0)  # [num_samples, theta_dim]
    all_events = torch.stack(
        all_event_data, dim=0
    )  # [num_samples, n_repeat, num_events, feature_dim]

    print(f"  Final theta shape: {all_thetas.shape}")
    print(f"  Final events shape: {all_events.shape}")

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{problem}_ns{num_samples}_ne{num_events}_nr{n_repeat}.npz"
    filepath = os.path.join(output_dir, filename)

    # Save metadata as separate arrays to avoid pickle issues
    atomic_savez_compressed(
        filepath,
        thetas=all_thetas.numpy(),
        events=all_events.numpy(),
        problem=np.array([problem], dtype="U20"),
        num_samples=np.array([num_samples]),
        num_events=np.array([num_events]),
        n_repeat=np.array([n_repeat]),
        theta_shape=np.array(all_thetas.shape),
        events_shape=np.array(all_events.shape),
    )

    print(f"  Saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Generate precomputed datasets for PDF Parameter Inference"
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=["simplified_dis"],
        choices=["simplified_dis", "mceg"],
        help="Problems to generate data for",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of theta parameter samples to generate",
    )
    parser.add_argument(
        "--num_events", type=int, default=1000, help="Number of events per simulation"
    )
    parser.add_argument(
        "--n_repeat",
        type=int,
        default=2,
        help="Number of repeated simulations per theta",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="precomputed_data",
        help="Directory to save precomputed datasets",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for generation (cpu, cuda, auto)",
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Generating data for problems: {args.problems}")

    # Generate data for each problem
    generated_files = []
    for problem in args.problems:
        try:
            filepath = generate_data_for_problem(
                problem,
                args.num_samples,
                args.num_events,
                args.n_repeat,
                device,
                args.output_dir,
            )
            generated_files.append(filepath)
        except Exception as e:
            print(f"Error generating data for {problem}: {e}")

    print(f"\nGenerated {len(generated_files)} datasets:")
    for filepath in generated_files:
        print(f"  {filepath}")

    print("\nDatasets are ready for training!")


if __name__ == "__main__":
    main()
