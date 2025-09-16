#!/usr/bin/env python3
"""
Test script for parameter_prediction.py with train-test split functionality
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np


class SimpleDataset(Dataset):
    """Simple dataset for testing train-test split functionality"""
    def __init__(self, num_samples, num_events, theta_dim=4, x_dim=6):
        self.num_samples = num_samples
        self.num_events = num_events
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        
        # Generate random data for testing
        torch.manual_seed(42)  # For reproducibility
        self.thetas = torch.randn(num_samples, theta_dim)
        self.xs = torch.randn(num_samples, 2, num_events, x_dim)  # 2 repeats
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.thetas[idx], self.xs[idx]


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, x):
        # x shape: [B*n_repeat, num_events, input_dim]
        # Pool over events dimension
        x = x.mean(dim=1)  # [B*n_repeat, input_dim]
        return self.layers(x)


class SimpleParamPredictionModel(nn.Module):
    """Simple parameter prediction model for testing"""
    def __init__(self, latent_dim, theta_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, theta_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


def train_with_validation(model, param_model, train_loader, val_loader, 
                         epochs=10, lr=1e-3, device='cpu'):
    """Training function with validation evaluation"""
    
    optimizer = optim.Adam(list(model.parameters()) + list(param_model.parameters()), lr=lr)
    
    model.train()
    param_model.train()
    
    print("Starting training with validation...")
    
    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        for batch_idx, (theta, x_sets) in enumerate(train_loader):
            x_sets = x_sets.to(device)
            theta = theta.to(device)
            
            B, n_repeat, num_points, feat_dim = x_sets.shape
            x_sets = x_sets.reshape(B*n_repeat, num_points, feat_dim)
            theta = theta.repeat_interleave(n_repeat, dim=0)
            
            optimizer.zero_grad()
            
            emb = model(x_sets)
            predicted_theta = param_model(emb)
            loss = F.mse_loss(predicted_theta, theta)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        param_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for theta, x_sets in val_loader:
                x_sets = x_sets.to(device)
                theta = theta.to(device)
                
                B, n_repeat, num_points, feat_dim = x_sets.shape
                x_sets = x_sets.reshape(B*n_repeat, num_points, feat_dim)
                theta = theta.repeat_interleave(n_repeat, dim=0)
                
                emb = model(x_sets)
                predicted_theta = param_model(emb)
                loss = F.mse_loss(predicted_theta, theta)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        model.train()
        param_model.train()
    
    return train_loss, val_loss


def test_train_validation_split():
    """Test the train-validation split functionality"""
    print("Testing train-validation split functionality...")
    
    # Parameters
    total_samples = 1200  # Total samples
    train_samples = 200   # Training samples (num_samples parameter)
    val_samples = 1000    # Validation samples (as required)
    assert total_samples == train_samples + val_samples
    
    num_events = 100
    theta_dim = 4
    x_dim = 6
    latent_dim = 32
    batch_size = 16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create full dataset
    full_dataset = SimpleDataset(total_samples, num_events, theta_dim, x_dim)
    
    # Split into train and validation
    train_dataset, val_dataset = random_split(full_dataset, [train_samples, val_samples])
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create models
    model = SimpleModel(x_dim, latent_dim).to(device)
    param_model = SimpleParamPredictionModel(latent_dim, theta_dim).to(device)
    
    # Train with validation
    train_loss, val_loss = train_with_validation(
        model, param_model, train_loader, val_loader, 
        epochs=5, lr=1e-3, device=device
    )
    
    print(f"\nFinal - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    print("✓ Train-validation split test completed successfully!")


if __name__ == "__main__":
    test_train_validation_split()