#!/usr/bin/env python3
"""
Simple test to show the help output with the new --nll-loss argument.
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description="PDF Parameter Inference Training")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument(
        "--problem",
        type=str,
        default="simplified_dis",
        choices=["simplified_dis", "realistic_dis"],
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_events", type=int, default=100000)
    parser.add_argument("--nll-loss", action="store_true", 
                       help="Use Gaussian negative log-likelihood loss with mean and variance prediction")
    
    parser.print_help()

if __name__ == "__main__":
    main()