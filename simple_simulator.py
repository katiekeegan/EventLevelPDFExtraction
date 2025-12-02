"""
Simple simulators for testing parameter prediction without external dependencies
"""

import numpy as np
import torch


class SimpleGaussian2DSimulator:
    """Simple 2D Gaussian simulator"""

    def __init__(self, device=None):
        self.device = device or torch.device("cpu")

    def sample(self, theta, nevents=100):
        """
        Sample from 2D Gaussian
        theta: [mu_x, mu_y, sigma_x, sigma_y, rho]
        """
        mu_x, mu_y, sigma_x, sigma_y, rho = theta

        # Create covariance matrix
        cov = torch.tensor(
            [
                [sigma_x**2, rho * sigma_x * sigma_y],
                [rho * sigma_x * sigma_y, sigma_y**2],
            ],
            device=self.device,
        )

        mean = torch.tensor([mu_x, mu_y], device=self.device)

        # Sample from multivariate normal
        dist = torch.distributions.MultivariateNormal(mean, cov)
        samples = dist.sample((nevents,))

        return samples


class SimpleSimplifiedDIS:
    """Simple DIS simulator for testing"""

    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        self.clip_alert = False

    def sample(self, theta, nevents=100):
        """
        Simple DIS simulation
        theta: [au, bu, ad, bd] parameters
        """
        au, bu, ad, bd = theta

        # Create simple features based on theta
        x = torch.rand(nevents, device=self.device)

        # Simple physics-inspired features
        f1 = au * torch.exp(-bu * x) + torch.randn(nevents, device=self.device) * 0.1
        f2 = ad * torch.exp(-bd * x) + torch.randn(nevents, device=self.device) * 0.1
        f3 = (au + ad) * x * (1 - x) + torch.randn(nevents, device=self.device) * 0.1
        f4 = (
            torch.sin(torch.pi * x) * (au - ad)
            + torch.randn(nevents, device=self.device) * 0.1
        )
        f5 = x**2 * bu + torch.randn(nevents, device=self.device) * 0.1
        f6 = (1 - x) ** 2 * bd + torch.randn(nevents, device=self.device) * 0.1

        features = torch.stack([f1, f2, f3, f4, f5, f6], dim=1)
        return features


def log_feature_engineering(x):
    """Simple feature engineering for testing"""
    # Apply log transformation where applicable, add small epsilon to avoid log(0)
    epsilon = 1e-8
    return torch.log(torch.abs(x) + epsilon)


def advanced_feature_engineering(x):
    """Simple advanced feature engineering for testing"""
    return log_feature_engineering(x)
