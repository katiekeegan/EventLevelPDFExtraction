import os
# Basic imports that should always work
import sys

import numpy as np
import pandas as pd
import torch
from scipy.integrate import fixed_quad, quad
from torch.distributions import Distribution, Uniform
from tqdm import tqdm
from utils import log_feature_engineering

np.random.seed(42)
torch.manual_seed(42)

# Try to import optional dependencies for advanced simulators
try:
    sys.path.append(os.path.abspath("mceg4dis"))

    # --matplotlib
    import matplotlib
    from matplotlib.lines import Line2D

    matplotlib.rc("text", usetex=True)
    import cfg
    import matplotlib.gridspec as gridspec
    import params as par
    import pylab as py
    from alphaS import ALPHAS
    from eweak import EWEAK
    from idis import THEORY
    from matplotlib import colors
    from mceg import MCEG
    from mellin import MELLIN
    from pdf import PDF

    HAS_MCEG_DEPS = True
except ImportError:
    HAS_MCEG_DEPS = False


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
            (
                torch.tensor(p, device=self.device)
                if not torch.is_tensor(p)
                else p.to(self.device)
            )
            for p in params
        ]

    def up(self, x):
        return self.Nu * (x**self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        return self.Nd * (x**self.ad) * ((1 - x) ** self.bd)

    def f(self, x, theta):
        """Evaluate PDF functions f(x|theta). Returns dict with 'up' and 'down'."""
        self.init(theta)
        return {"up": self.up(x), "down": self.down(x)}

    def sample(self, params, n_events=1000):
        self.init(params)
        eps = 1e-6
        rand = lambda: torch.clamp(
            torch.rand(n_events, device=self.device), min=eps, max=1 - eps
        )
        smear_noise = lambda s: (
            s + torch.randn_like(s) * (self.smear_std * s) if self.smear else s
        )

        xs_p, xs_n = rand(), rand()
        sigma_p = smear_noise(4 * self.up(xs_p) + self.down(xs_p))
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0, posinf=1e8, neginf=0.0)
        sigma_n = smear_noise(4 * self.down(xs_n) + self.up(xs_n))
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0, posinf=1e8, neginf=0.0)
        return torch.stack([sigma_p, sigma_n], dim=-1)


# MCEGSimulator: Only available if MCEG dependencies are installed
if HAS_MCEG_DEPS:

    class MCEGSimulator:
        def __init__(self, device=None):
            self.device = device
            # Initialize MCEG Sampler with default parameters
            self.mellin = MELLIN(npts=8)
            self.alphaS = ALPHAS()
            self.eweak = EWEAK()
            self.pdf = PDF(self.mellin, self.alphaS)
            self.idis = THEORY(self.mellin, self.pdf, self.alphaS, self.eweak)

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
            self.mceg = MCEG(self.idis, rs=140, tar="p", W2min=10, nx=30, nQ2=20)

        def sample(self, params, n_events=1000):
            assert n_events > 0, "Number of events must be positive"
            if not isinstance(params, torch.Tensor):
                params = torch.tensor(params, dtype=torch.float32, device=self.device)
            self.init(params)  # Take in new parameters
            # Initialize Monte Carlo Event Generator
            mceg = MCEG(self.idis, rs=140, tar="p", W2min=10, nx=30, nQ2=20)
            # TODO: negative probabilities may arise with certain parameters.
            # Find a way to work around this (maybe work directly with idis if there is a bug in mceg.gen_events)

            samples = torch.tensor(mceg.gen_events(n_events + 1000, verb=False)).to(
                self.device
            )
            random_indices = torch.randperm(samples.size(0))[:n_events]
            samples = samples[random_indices]
            self.clip_alert = mceg.clip_alert
            self.mceg = mceg
            return samples

else:
    MCEGSimulator = None


def up(x, params):
    return (x ** params[0]) * ((1 - x) ** params[1])


def down(x, params):
    return (x ** params[2]) * ((1 - x) ** params[3])
