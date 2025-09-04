import torch
from torch.distributions import Uniform, Distribution
import numpy as np

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.integrate import quad,fixed_quad
import os, sys
sys.path.append(os.path.abspath("mceg4dis"))

#--matplotlib
import matplotlib
from matplotlib.lines import Line2D
matplotlib.rc('text',usetex=True)
import pylab as py
from matplotlib import colors
import matplotlib.gridspec as gridspec

import params as par 
import cfg
from alphaS  import ALPHAS
from eweak   import EWEAK
from pdf     import PDF
from mellin  import MELLIN
from idis    import THEORY
from mceg    import MCEG

class Gaussian2DSimulator:
    """
    Unimodal 2D Gaussian simulator with 1D parameter vector input.
    Parameter vector: [mu_x, mu_y, sigma_x, sigma_y, rho]
    """
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")

    def sample(self, theta, nevents=1000):
        """
        theta: torch.tensor of shape (5,) -- [mu_x, mu_y, sigma_x, sigma_y, rho]
        Returns: torch.tensor of shape (nevents, 2)
        """
        mu_x, mu_y, sigma_x, sigma_y, rho = theta
        mean = torch.tensor([mu_x, mu_y], device=self.device)
        cov = torch.tensor([
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2]
        ], device=self.device)
        samples = torch.distributions.MultivariateNormal(mean, cov).sample((nevents,))
        return samples

class SimplifiedDIS:
    def __init__(self, device=None, smear=False, smear_std=0.05):
        self.device = device
        self.smear = smear
        self.smear_std = smear_std
        self.Nu = 1
        self.Nd = 2
        self.au, self.bu, self.ad, self.bd = None, None, None, None

    def init(self, params):
        self.au, self.bu, self.ad, self.bd = [
            torch.tensor(p, device=self.device) if not torch.is_tensor(p) else p.to(self.device)
            for p in params
        ]

    def up(self, x):
        return self.Nu * (x ** self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        return self.Nd * (x ** self.ad) * ((1 - x) ** self.bd)

    def sample(self, params, nevents=1):
        self.init(params)
        eps = 1e-6
        rand = lambda: torch.clamp(torch.rand(nevents, device=self.device), min=eps, max=1 - eps)
        smear_noise = lambda s: s + torch.randn_like(s) * (self.smear_std * s) if self.smear else s

        xs_p, xs_n = rand(), rand()
        sigma_p = smear_noise(4 * self.up(xs_p) + self.down(xs_p))
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0, posinf=1e8, neginf=0.0)
        sigma_n = smear_noise(4 * self.down(xs_n) + self.up(xs_n))
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0, posinf=1e8, neginf=0.0)
        return torch.stack([sigma_p, sigma_n], dim=-1)

class MCEGSimulator:
    def __init__(self, device=None):
        self.device = device
        # Initialize MCEG Sampler with default parameters
        self.mellin = MELLIN(npts=8)
        self.alphaS = ALPHAS()
        self.eweak  = EWEAK()
        self.pdf    = PDF(self.mellin, self.alphaS)
        self.idis   = THEORY(self.mellin, self.pdf, self.alphaS, self.eweak)

    def init(self, params):
        # Take in new parameters and update MCEG class
        new_cpar = self.pdf.get_current_par_array()[::]
        # Assume parameters are only corresponding to 'uv1' parameters
        if not isinstance(params, torch.Tensor):
            new_cpar[4:8] = params
        else:
            new_cpar[4:8] = params.cpu().numpy()  # Update uv1 parameters
        self.pdf.setup(new_cpar)
        self.idis = THEORY(self.mellin, self.pdf, self.alphaS, self.eweak)

    def sample(self, params, nevents=1):
        assert nevents > 0, "Number of events must be positive"
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, dtype=torch.float32, device=self.device)
        self.init(params)  # Take in new parameters
        # Initialize Monte Carlo Event Generator
        mceg = MCEG(self.idis, rs=140, tar='p', W2min=10, nx=30, nQ2=20)
        # TODO: negative probabilities may arise with certain parameters.
        # Find a way to work around this (maybe work directly with idis if there is a bug in mceg.gen_events)
        samples = torch.tensor(mceg.gen_events(nevents, verb=False)).to(self.device)
        self.clip_alert = mceg.clip_alert
        return samples

def up(x, params):
    return (x ** params[0]) * ((1 - x) ** params[1])

def down(x, params):
    return (x ** params[2]) * ((1 - x) ** params[3])

def advanced_feature_engineering(xs_tensor):
    log_features = torch.log1p(xs_tensor)
    symlog_features = torch.sign(xs_tensor) * torch.log1p(xs_tensor.abs())

    ratio_features = []
    diff_features = []
    for i in range(xs_tensor.shape[-1]):
        for j in range(i + 1, xs_tensor.shape[-1]):
            ratio = xs_tensor[..., i] / (xs_tensor[..., j] + 1e-8)
            ratio_features.append(torch.log1p(ratio.abs()).unsqueeze(-1))
            diff = torch.log1p(xs_tensor[..., i]) - torch.log1p(xs_tensor[..., j])
            diff_features.append(diff.unsqueeze(-1))

    ratio_features = torch.cat(ratio_features, dim=-1)
    diff_features = torch.cat(diff_features, dim=-1)
    return torch.cat([log_features, symlog_features, ratio_features, diff_features], dim=-1)
    
class RealisticDIS:
    def __init__(self, device=None, smear=True, smear_std=0.05):
        self.device = device or torch.device("cpu")
        self.smear = smear
        self.smear_std = smear_std
        self.Q0_squared = 1.0  # GeV^2 reference scale
        self.params = None

    def __call__(self, params, nevents=1000):
        return self.sample(params, nevents)

    def init(self, params):
        # Accepts raw list or tensor of 6 params: [logA0, delta, a, b, c, d]
        p = torch.tensor(params, dtype=torch.float32, device=self.device)
        self.logA0 = p[0]
        self.delta = p[1]
        self.a = p[2]
        self.b = p[3]
        self.c = p[4]
        self.d = p[5]

    def q(self, x, Q2):
        A0 = torch.exp(self.logA0)
        scale_factor = (Q2 / self.Q0_squared).clamp(min=1e-6)
        A_Q2 = A0 * scale_factor ** self.delta
        shape = x.clamp(min=1e-6, max=1.0)**self.a * (1 - x.clamp(min=0.0, max=1.0))**self.b
        poly = 1 + self.c * x + self.d * x**2
        shape = shape * poly.clamp(min=1e-6)  # avoid negative polynomial tail
        return A_Q2 * shape

    def F2(self, x, Q2):
        return x * self.q(x, Q2)

    def sample(self, params, nevents=1000, x_range=(1e-3, 0.9), Q2_range=(1.0, 1000.0)):
        self.init(params)

        # Sample x ~ Uniform, Q2 ~ LogUniform
        x = torch.rand(nevents, device=self.device) * (x_range[1] - x_range[0]) + x_range[0]
        logQ2 = torch.rand(nevents, device=self.device) * (
            np.log10(Q2_range[1]) - np.log10(Q2_range[0])
        ) + np.log10(Q2_range[0])
        Q2 = 10 ** logQ2

        f2 = self.F2(x, Q2)

        if self.smear:
            noise = torch.randn_like(f2) * (self.smear_std * f2)
            f2 = f2 + noise
            f2f = f2.clamp(min=1e-6)

        return torch.stack([x, Q2, f2], dim=1)  # shape: [nevents, 3]