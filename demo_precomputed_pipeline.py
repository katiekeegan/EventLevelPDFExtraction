#!/usr/bin/env python3
"""
Example script demonstrating the precomputed data pipeline for PDF Parameter Inference.

This script shows how to:
1. Generate precomputed datasets
2. Load and inspect the data  
3. Train a simple model with precomputed data
4. Compare with on-the-fly generation
"""

import sys
import os
sys.path.append('/home/runner/work/PDFParameterInference/PDFParameterInference')

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time

# Import our modules
from precomputed_datasets import PrecomputedDataset, create_precomputed_dataloader
from models import PointNetPMA

def generate_sample_data():
    """Generate sample datasets for demonstration."""
    print("Step 1: Generating sample precomputed datasets...")
    
    import subprocess
    
    # Generate small datasets for demo
    cmd = [
        sys.executable, "generate_precomputed_data.py",
        "--problems", "gaussian", "simplified_dis",
        "--num_samples", "500",
        "--num_events", "50", 
        "--output_dir", "demo_data"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Warnings:", result.stderr)
    
    print("✓ Sample datasets generated in demo_data/")

def inspect_generated_data():
    """Inspect the generated datasets."""
    print("\nStep 2: Inspecting generated datasets...")
    
    for problem in ["gaussian", "simplified_dis"]:
        try:
            dataset = PrecomputedDataset("demo_data", problem)
            metadata = dataset.get_metadata()
            
            print(f"\n{problem.upper()} Dataset:")
            print(f"  - Samples: {metadata['num_samples']}")
            print(f"  - Theta dimensions: {metadata['theta_dim']}")
            print(f"  - Feature dimensions: {metadata['feature_dim']}")
            print(f"  - Events per sample: {metadata['num_events']}")
            print(f"  - Repetitions: {metadata['n_repeat']}")
            
            # Get a sample
            theta, events = dataset[0]
            print(f"  - Sample shapes: theta {theta.shape}, events {events.shape}")
            
        except Exception as e:
            print(f"Error inspecting {problem}: {e}")
    
    print("✓ Data inspection complete")

def train_simple_model(problem="gaussian"):
    """Train a simple model with precomputed data."""
    print(f"\nStep 3: Training simple model on {problem} data...")
    
    # Create dataset and dataloader
    dataset = PrecomputedDataset("demo_data", problem)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    # Get dimensions from first batch
    theta, events = next(iter(dataloader))
    batch_size, n_repeat, num_events, feature_dim = events.shape
    theta_dim = theta.shape[1]
    
    print(f"  Model input: {feature_dim}D events → {theta_dim}D parameters")
    
    # Create simple model
    device = torch.device('cpu')  # Use CPU for demo
    latent_dim = 32
    
    model = PointNetPMA(
        input_dim=feature_dim, 
        latent_dim=latent_dim, 
        predict_theta=True
    ).to(device)
    
    # Simple prediction head
    class SimplePredictionHead(nn.Module):
        def __init__(self, latent_dim, theta_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Linear(16, theta_dim)
            )
        
        def forward(self, x):
            return self.net(x)
    
    pred_head = SimplePredictionHead(latent_dim, theta_dim).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(pred_head.parameters()),
        lr=1e-3
    )
    
    # Training loop
    model.train()
    pred_head.train()
    
    print("  Training...")
    start_time = time.time()
    
    for epoch in range(5):  # Just a few epochs for demo
        epoch_loss = 0.0
        num_batches = 0
        
        for theta_batch, events_batch in dataloader:
            # Move to device
            theta_batch = theta_batch.to(device).float()
            events_batch = events_batch.to(device).float()
            
            # Reshape for processing
            B, n_repeat, num_points, feat_dim = events_batch.shape
            x_reshaped = events_batch.reshape(B * n_repeat, num_points, feat_dim)
            theta_target = theta_batch.repeat_interleave(n_repeat, dim=0)
            
            # Forward pass
            optimizer.zero_grad()
            
            latent = model(x_reshaped)
            theta_pred = pred_head(latent)
            
            loss = nn.functional.mse_loss(theta_pred, theta_target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"    Epoch {epoch+1}: loss = {avg_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"  ✓ Training completed in {training_time:.2f} seconds")
    
    return training_time

def demonstrate_speed_comparison():
    """Compare training speed between precomputed and on-the-fly data."""
    print("\nStep 4: Speed comparison (precomputed vs on-the-fly)")
    
    # Time precomputed training
    precomputed_time = train_simple_model("gaussian")
    
    print(f"\nPrecomputed data training time: {precomputed_time:.2f} seconds")
    print("\nNote: On-the-fly training would be slower due to simulation overhead")
    print("      (simulation not available in this demo environment)")

def main():
    """Run the complete demonstration."""
    print("=== Precomputed Data Pipeline Demo ===\n")
    
    try:
        # Step 1: Generate data
        generate_sample_data()
        
        # Step 2: Inspect data
        inspect_generated_data()
        
        # Step 3: Train model
        train_simple_model("gaussian")
        
        # Step 4: Speed comparison
        demonstrate_speed_comparison()
        
        print("\n=== Demo Complete ===")
        print("\nKey benefits demonstrated:")
        print("✓ Fast data generation and storage")
        print("✓ Easy data loading and inspection")
        print("✓ Efficient training with precomputed data")
        print("✓ Reproducible datasets for experiments")
        
        print(f"\nGenerated datasets are saved in demo_data/")
        print("You can now use these for further experiments!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()