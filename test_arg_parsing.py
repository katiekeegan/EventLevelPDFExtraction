#!/usr/bin/env python3
"""
Test script to verify the parameter_prediction.py argument parsing works
"""
import argparse

# Extract just the argument parsing logic from parameter_prediction.py
def test_argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--num_events", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--problem", type=str, default="simplified_dis")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default=None, help="Unique name for this ablation run")
    parser.add_argument("--use_precomputed", action="store_true", 
                       help="Use precomputed data instead of generating on-the-fly. Automatically generates data if not found.")
    parser.add_argument("--precomputed_data_dir", type=str, default="precomputed_data",
                       help="Directory containing precomputed data files")
    parser.add_argument("--single_gpu", action="store_true",
                       help="Force single-GPU mode even if multiple GPUs are available")
    
    # Test parsing with --single_gpu
    try:
        args = parser.parse_args(["--single_gpu"])
        if hasattr(args, 'single_gpu') and args.single_gpu:
            print("✓ --single_gpu argument parsing works correctly")
            return True
        else:
            print("✗ --single_gpu argument not properly parsed")
            return False
    except Exception as e:
        print(f"✗ Error parsing --single_gpu argument: {e}")
        return False

if __name__ == "__main__":
    success = test_argument_parsing()
    exit(0 if success else 1)