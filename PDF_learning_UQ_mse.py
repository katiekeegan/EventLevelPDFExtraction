"""
PDF_learning_UQ_mse.py - Enhanced training with simulator MSE loss

This file extends PDF_learning_UQ.py with a new loss function that computes
MSE between simulator outputs for predicted and true parameters, rather than
direct parameter MSE or log-relative PDF discrepancy.

Key Features:
- pdf_theta_mse_loss: MSE between simulator outputs  
- pdf_theta_mse_loss_batched: Optimized batched version
- train_mse_simulator: Training function using simulator MSE loss
- Full Laplace approximation compatibility
- Original pdf_theta_loss preserved for analysis
- Command line switching between loss types

Usage:
    # Train with simulator MSE loss - MLP
    python PDF_learning_UQ_mse.py --arch mlp --use_simulator_loss
    
    # Train with simulator MSE loss - Transformer
    python PDF_learning_UQ_mse.py --arch transformer --use_simulator_loss --nevents_loss 2000
    
    # Standard training (same as original)
    python PDF_learning_UQ_mse.py --arch all

The implementation focuses on the "simplified_dis" problem as requested but
supports other problems as well.
"""

import os
import warnings
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, random_split

from datasets import *
from models import *
from simulator import *
from utils import *

import os, json, torch

from torch.autograd import Function

# Laplace-torch for Laplace approximation
from laplace import Laplace

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# --- Add this at the top of your file with other imports ---
import torch.nn as nn
import torch
def save_laplace(la, output_dir, filename="laplace_mlp_state.pt",
                 likelihood="regression",
                 subset_of_weights="last_layer",
                 hessian_structure="kron",
                 extra_meta=None):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    meta = {
        "likelihood": likelihood,
        "subset_of_weights": subset_of_weights,
        "hessian_structure": hessian_structure,
    }
    if extra_meta:  # e.g. {"temperature": 1.0, "sigma_noise": 0.1}
        meta.update(extra_meta)
    torch.save({"laplace_state": la.state_dict(), "meta": meta}, path)
    return path

def load_laplace(make_model_fn, ckpt_path, device="cpu"):
    """
    make_model_fn() -> your base model instance with MAP weights already loaded.
    """
    from laplace.laplace import Laplace
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = make_model_fn().to(device)
    meta = ckpt["meta"]
    la = Laplace(model,
                 likelihood=meta["likelihood"],
                 subset_of_weights=meta["subset_of_weights"],
                 hessian_structure=meta["hessian_structure"])
    la.load_state_dict(ckpt["laplace_state"])
    return la  # Fixed: was 'las' instead of 'la'
class MeanOnlyWrapper(nn.Module):
    def __init__(self, gaussian_model: nn.Module):
        super().__init__()
        self.model = gaussian_model  # forward returns (mean, log_std)
    def forward(self, x):
        mean, _ = self.model(x)
        return mean

def last_linear(module: nn.Module) -> nn.Linear:
    # find the last nn.Linear actually present in the wrapped model
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    assert last is not None, "No Linear layer found in mean path."
    return last

# --- Custom Gaussian NLL loss for Laplace ---
def gaussian_nll_loss(output, target):
    """
    output: [batch, 2*param_dim] = [means, logvars]
    target: [batch, param_dim]
    """
    param_dim = target.shape[-1]
    means = output[:, :param_dim]
    logvars = output[:, param_dim:]
    var = torch.exp(logvars)
    # Standard NLL for diagonal Gaussians
    nll = 0.5 * logvars + 0.5 * ((target - means) ** 2) / var
    return nll.sum(dim=-1).mean()

def _get_pdf_accessor(simulator):
    """
    Return f(theta, x, Q2, flav) -> tensor (len(x),) of xF(x,Q2,flav), robust to
    scalar-only PDF implementations. It tries vector x once; on failure, it loops scalars.
    """
    # discover a callable that computes xF
    if hasattr(simulator, 'get_xF'):
        raw = simulator.get_xF
        need_init = hasattr(simulator, 'init')
    elif hasattr(simulator, 'pdf') and hasattr(simulator.pdf, 'get_xF'):
        raw = simulator.pdf.get_xF
        need_init = hasattr(simulator, 'init')
    elif hasattr(simulator, 'get_pdf_xF'):
        raw = simulator.get_pdf_xF
        need_init = hasattr(simulator, 'init')
    else:
        return None

    def f(theta, x, Q2, flav):
        # ensure params are set
        if need_init:
            simulator.init(theta)

        # numpy view of x (1D)
        x_np = x.detach().cpu().numpy().astype(float, copy=False)

        # 1) try a single vector call
        try:
            vals = raw(x_np, float(Q2), flav)
            vals_np = np.asarray(vals)
            # expect shape (nx,) (or broadcastable). If clearly wrong, fallback.
            if vals_np.shape == x_np.shape:
                return torch.as_tensor(vals_np, dtype=torch.float32, device=x.device)
        except Exception:
            pass

        # 2) scalar fallback (robust path)
        out = np.empty_like(x_np, dtype=float)
        for i, xi in enumerate(x_np):
            out[i] = raw(float(xi), float(Q2), flav)
        return torch.as_tensor(out, dtype=torch.float32, device=x.device)

    return f

def numpy_xF(simulator, theta, x, q2, flav):
    """
    Robust accessor for xF(x, Q2, flav) that works whether the simulator
    only supports scalar x or not. Keeps mellin.invert unchanged.
    """
    # Ensure params are set
    if hasattr(simulator, 'init'):
        simulator.init(theta)

    x_arr = np.asarray(x)

    # Fast path: scalar x
    if x_arr.ndim == 0:
        return simulator.pdf.get_xF(float(x_arr), float(q2), flav)

    # General path: loop scalars to avoid broadcasting inside mellin.invert
    flat = x_arr.reshape(-1)
    out = np.empty_like(flat, dtype=np.float64)
    for i, xi in enumerate(flat):
        out[i] = simulator.pdf.get_xF(float(xi), float(q2), flav)
    return out.reshape(x_arr.shape)

class NumpySimXFFunction(Function):
    @staticmethod
    def forward(ctx, theta, x, q2, flav, simulator, eps_scale: float):
        """
        theta: (P,) tensor, requires_grad=True
        x:     (nx,) tensor
        q2:    float or 0-dim tensor
        flav:  whatever the simulator expects (str/int)
        simulator: your MCEGSimulator instance
        eps_scale: relative step size for finite-diff per parameter
        """
        device, dtype = theta.device, theta.dtype

        # --- detach for NumPy forward (we will supply gradients manually) ---
        theta_np = theta.detach().cpu().numpy().astype(np.float64, copy=True)
        x_np     = x.detach().cpu().numpy().astype(np.float64, copy=False)

        vals_np  = np.asarray(numpy_xF(simulator, theta_np, x_np, float(q2), flav))
        vals     = torch.from_numpy(vals_np).to(device=device, dtype=dtype)

        # save for backward
        ctx.simulator = simulator
        ctx.theta_np  = theta_np
        ctx.x_np      = x_np
        ctx.q2        = float(q2)
        ctx.flav      = flav
        ctx.eps_scale = float(eps_scale)

        return vals

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (nx,) tensor
        returns gradients for (theta, x, q2, flav, simulator, eps_scale).
        Only theta gets a gradient; others are None.
        """
        simulator = ctx.simulator
        theta_np  = ctx.theta_np
        x_np      = ctx.x_np
        q2        = ctx.q2
        flav      = ctx.flav
        eps_scale = ctx.eps_scale

        P   = theta_np.shape[0]
        nx  = x_np.shape[0]

        # bring grad_output to cpu numpy
        g = grad_output.detach().cpu().numpy().astype(np.float64, copy=False)

        # per-parameter central differences, compute VJP = (J^T g)
        grad_theta = np.zeros((P,), dtype=np.float64)
        # step size per param: eps = eps_scale * max(1, |theta_j|)
        eps_vec = eps_scale * np.maximum(1.0, np.abs(theta_np))

        # cache baseline if you want to do forward-diff; for central diff we don’t need it
        for j in range(P):
            eps = eps_vec[j]
            th_plus  = theta_np.copy(); th_plus[j]  += eps
            th_minus = theta_np.copy(); th_minus[j] -= eps

            f_plus  = np.asarray(numpy_xF(simulator, th_plus,  x_np, q2, flav), dtype=np.float64)
            f_minus = np.asarray(numpy_xF(simulator, th_minus, x_np, q2, flav), dtype=np.float64)
            df_dthj = (f_plus - f_minus) / (2.0 * eps)          # shape (nx,)
            grad_theta[j] = (df_dthj * g).sum()                 # inner product => scalar

        grad_theta_t = torch.from_numpy(grad_theta).to(grad_output.device, dtype=grad_output.dtype)

        # no grads for x, q2, flav, simulator, eps_scale
        return grad_theta_t, None, None, None, None, None

# convenience function
def torch_xF(theta, x, q2, flav, simulator, eps_scale=1e-4):
    return NumpySimXFFunction.apply(theta, x, q2, flav, simulator, float(eps_scale))

# --- NEW: PDF/Simulator-based MSE loss function ---
def pdf_theta_mse_loss(theta_pred, theta_true, simulator, problem="simplified_dis", nevents=1000):
    """
    Compute MSE between up() and down() outputs for fixed xs under predicted and true parameters.
    
    This loss function compares the PDF up() and down() function outputs at fixed x values
    for predicted vs true parameters, providing a direct comparison of PDF shapes.
    
    Args:
        theta_pred: Predicted parameters, shape (batch_size, param_dim)
        theta_true: True parameters, shape (batch_size, param_dim)
        simulator: Simulator instance (SimplifiedDIS, RealisticDIS, etc.)
        problem: Problem type for parameter bounds and normalization
        nevents: Number of fixed x points to evaluate (used as nx_points)
        
    Returns:
        MSE loss between up() and down() outputs for fixed x values
    """
    batch_size = theta_pred.shape[0]
    
    # Generate fixed x values for evaluation
    eps = 1e-6
    nx_points = nevents  # Reuse nevents parameter as number of x points
    x_values = torch.linspace(eps, 1 - eps, nx_points, device=theta_pred.device)
    
    # Handle different simulator types
    if hasattr(simulator, 'up') and hasattr(simulator, 'down'):
        # SimplifiedDIS case
        pred_up_outputs = []
        pred_down_outputs = []
        true_up_outputs = []
        true_down_outputs = []
        
        for i in range(batch_size):
            # Evaluate up() and down() for predicted parameters
            simulator.init(theta_pred[i])
            pred_up = simulator.up(x_values)
            pred_down = simulator.down(x_values)
            pred_up_outputs.append(pred_up)
            pred_down_outputs.append(pred_down)
            
            # Evaluate up() and down() for true parameters  
            simulator.init(theta_true[i])
            true_up = simulator.up(x_values)
            true_down = simulator.down(x_values)
            true_up_outputs.append(true_up)
            true_down_outputs.append(true_down)
        
        # Stack into tensors
        pred_up_outputs = torch.stack(pred_up_outputs)  # (batch_size, nx_points)
        pred_down_outputs = torch.stack(pred_down_outputs)  # (batch_size, nx_points)
        true_up_outputs = torch.stack(true_up_outputs)  # (batch_size, nx_points)
        true_down_outputs = torch.stack(true_down_outputs)  # (batch_size, nx_points)
        
        # Compute MSE between up() and down() outputs
        mse_up = F.mse_loss(pred_up_outputs, true_up_outputs)
        mse_down = F.mse_loss(pred_down_outputs, true_down_outputs)
        mse_loss = mse_up + mse_down
        
    else:
        # pick a sane nx (don’t tie it to nevents if you’re not generating events)
        nx = max(128, min(1024, int(nevents)))  # or just a constant like 256
        x = torch.linspace(x_min + eps, x_max - eps, nx, device=device)

        if not Q2_slices:
            Q2_slices = [4.0, 10.0, 100.0]

        batch_losses = []
        for i in range(batch_size):
            per_pred, per_true = [], []
            for q2 in Q2_slices:
                for flav in flavors:
                    vals_pred = torch_xF(theta_pred[i], x, q2, flav, simulator)     # differentiable via custom backward
                    vals_true = torch_xF(theta_true[i].detach(), x, q2, flav, simulator).detach()
                    per_pred.append(vals_pred)
                    per_true.append(vals_true)

            pred_mat = torch.stack(per_pred)   # (n_curves, nx), requires_grad=True
            true_mat = torch.stack(per_true)   # (n_curves, nx)
            mse_i = torch.nn.functional.mse_loss(pred_mat, true_mat)
            batch_losses.append(mse_i)

        mse_loss = torch.stack(batch_losses).mean()
    
    return mse_loss

# --- NEW: Vectorized/batched version for better performance ---
def pdf_theta_mse_loss_batched(theta_pred, theta_true, simulator, problem="simplified_dis", nevents=1000, flavors=("u"),     Q2_slices=None,          # e.g. [par.mc2+1e-2, 10.0, 100.0]
    x_min=1e-4,
    x_max=0.9):
    """
    Compute MSE between up() and down() outputs for fixed xs with optimized batching.
    
    This version processes in chunks to manage memory efficiently while comparing
    up() and down() outputs at fixed x values.
    """
    batch_size = theta_pred.shape[0]
    
    # Generate fixed x values for evaluation
    eps = 1e-6
    nx_points = nevents  # Reuse nevents parameter as number of x points
    x_values = torch.linspace(eps, 1 - eps, nx_points, device=theta_pred.device)
    
    # Handle different simulator types
    if hasattr(simulator, 'up') and hasattr(simulator, 'down'):
        # SimplifiedDIS case with chunked processing
        pred_up_outputs = []
        pred_down_outputs = []
        true_up_outputs = []
        true_down_outputs = []
        
        # Process in smaller chunks to manage memory
        chunk_size = min(4, batch_size)  # Process 4 samples at a time to avoid memory issues
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            
            # Generate for this chunk
            chunk_pred_up = []
            chunk_pred_down = []
            chunk_true_up = []
            chunk_true_down = []
            
            for j in range(i, end_idx):
                # Evaluate up() and down() for predicted parameters
                simulator.init(theta_pred[j])
                pred_up = simulator.up(x_values)
                pred_down = simulator.down(x_values)
                chunk_pred_up.append(pred_up)
                chunk_pred_down.append(pred_down)
                
                # Evaluate up() and down() for true parameters  
                simulator.init(theta_true[j])
                true_up = simulator.up(x_values)
                true_down = simulator.down(x_values)
                chunk_true_up.append(true_up)
                chunk_true_down.append(true_down)
            
            pred_up_outputs.extend(chunk_pred_up)
            pred_down_outputs.extend(chunk_pred_down)
            true_up_outputs.extend(chunk_true_up)
            true_down_outputs.extend(chunk_true_down)
        
        # Stack and compute loss
        pred_up_outputs = torch.stack(pred_up_outputs)  # (batch_size, nx_points)
        pred_down_outputs = torch.stack(pred_down_outputs)  # (batch_size, nx_points)
        true_up_outputs = torch.stack(true_up_outputs)  # (batch_size, nx_points)
        true_down_outputs = torch.stack(true_down_outputs)  # (batch_size, nx_points)
        
        # Compute MSE between up() and down() outputs
        mse_up = F.mse_loss(pred_up_outputs, true_up_outputs)
        mse_down = F.mse_loss(pred_down_outputs, true_down_outputs)
        mse_loss = mse_up + mse_down
        
    else:
        # pick a sane nx (don’t tie it to nevents if you’re not generating events)
        nx = max(128, min(1024, int(nevents)))  # or just a constant like 256
        x = torch.linspace(x_min + eps, x_max - eps, nx, device=device)

        if not Q2_slices:
            Q2_slices = [4.0, 10.0, 100.0]

        batch_losses = []
        for i in range(batch_size):
            per_pred, per_true = [], []
            for q2 in Q2_slices:
                for flav in flavors:
                    vals_pred = torch_xF(theta_pred[i], x, q2, flav, simulator)     # differentiable via custom backward
                    vals_true = torch_xF(theta_true[i].detach(), x, q2, flav, simulator).detach()
                    per_pred.append(vals_pred)
                    per_true.append(vals_true)

            pred_mat = torch.stack(per_pred)   # (n_curves, nx), requires_grad=True
            true_mat = torch.stack(per_true)   # (n_curves, nx)
            mse_i = torch.nn.functional.mse_loss(pred_mat, true_mat)
            batch_losses.append(mse_i)

        mse_loss = torch.stack(batch_losses).mean()
    
    return mse_loss

# --- Original pdf_theta_loss function for analysis/evaluation ---
def pdf_theta_loss(theta_pred, theta_true, simulator, problem="simplified_dis", x_vals=None):
    """
    Original log-relative PDF discrepancy loss function.
    Kept for analysis and evaluation purposes.
    """
    if x_vals is None:
        x_vals = torch.linspace(0.01, 0.99, 100, device=theta_pred.device)
    
    batch_size = theta_pred.shape[0]
    total_loss = 0.0
    
    for i in range(batch_size):
        if problem == "simplified_dis":
            # Compute PDFs for predicted parameters
            simulator.init(theta_pred[i])
            pred_up = simulator.up(x_vals)
            pred_down = simulator.down(x_vals)
            
            # Compute PDFs for true parameters
            simulator.init(theta_true[i])
            true_up = simulator.up(x_vals)
            true_down = simulator.down(x_vals)
            
            # Log-relative discrepancy
            eps = 1e-8
            up_loss = torch.mean(torch.abs(torch.log(pred_up + eps) - torch.log(true_up + eps)))
            down_loss = torch.mean(torch.abs(torch.log(pred_down + eps) - torch.log(true_down + eps)))
            
            total_loss += (up_loss + down_loss) / 2
        
    return total_loss / batch_size

# --- Wrapper for Laplace fitting ---
class GaussianLaplaceWrapper(nn.Module):
    def __init__(self, gaussian_head):
        super().__init__()
        self.gaussian_head = gaussian_head
    def forward(self, x):
        means, logvars = self.gaussian_head(x)
        # Concatenate for Laplace: [means, logvars]
        return torch.cat([means, logvars], dim=-1)


class MixtureLaplaceWrapper(nn.Module):
    def __init__(self, mixture_head, laplace_mode="means"):
        super().__init__()
        self.mixture_head = mixture_head
        self.laplace_mode = laplace_mode  # "means" or "means_logvars_weights"
    def forward(self, x):
        means, logvars, weights = self.mixture_head(x)
        batch_size, nmodes, param_dim = means.shape
        means_flat = means.view(batch_size, nmodes * param_dim)
        if self.laplace_mode == "means":
            return means_flat
        elif self.laplace_mode == "means_logvars_weights":
            logvars_flat = logvars.view(batch_size, nmodes * param_dim)
            weights_flat = weights.view(batch_size, nmodes)
            # Concatenate all outputs
            return torch.cat([means_flat, logvars_flat, weights_flat], dim=-1)
        else:
            raise ValueError(f"Unknown laplace_mode: {self.laplace_mode}")

class MLPHead(nn.Module):
    def __init__(self, embedding_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class TransformerHead(nn.Module):
    def __init__(self, embedding_dim, out_dim, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(embedding_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(128, out_dim)
    def forward(self, x):
        x = self.embedding(x).unsqueeze(0)
        x = self.transformer(x).squeeze(0)
        return self.fc(x)

class GaussianHead(nn.Module):
    def __init__(self, embedding_dim, param_dim, multimodal=False, nmodes=2):
        super().__init__()
        self.multimodal = multimodal
        self.nmodes = nmodes
        self.embed = nn.Linear(embedding_dim, 128)
        if multimodal:
            self.means = nn.Linear(128, param_dim * nmodes)
            self.logvars = nn.Linear(128, param_dim * nmodes)
            self.weights = nn.Linear(128, nmodes)
        else:
            self.means = nn.Linear(128, param_dim)
            self.logvars = nn.Linear(128, param_dim)
    def forward(self, x):
        x = F.relu(self.embed(x))
        if self.multimodal:
            batch = x.shape[0]
            means = self.means(x).view(batch, self.nmodes, -1)
            logvars = self.logvars(x).view(batch, self.nmodes, -1)
            weights = torch.softmax(self.weights(x), -1)
            return means, logvars, weights
        else:
            return self.means(x), self.logvars(x)

def fit_laplace(model, train_loader, regression=True):
    lap = Laplace(model, 'regression' if regression else 'classification', subset_of_weights='last_layer')
    lap.fit(train_loader)
    lap.optimize_prior_precision()
    return lap

def train_standard(model, train_loader, val_loader, device, epochs=100, lr=1e-4):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler()
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for latent, target in train_loader:
            latent, target = latent.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(dtype=torch.float16):
                # print(f"latent shape: {latent.shape}, target shape: {target.shape}")
                pred = model(latent)
                loss = F.mse_loss(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for latent, target in val_loader:
                latent, target = latent.to(device), target.to(device)
                pred = model(latent)
                val_loss += F.mse_loss(pred, target).item()
        val_loss /= len(val_loader)
        print(f"[Standard NN] Epoch {epoch} Train {epoch_loss/len(train_loader):.4f} Val {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_standard_model.pth")
    return model

# --- NEW: Training function with MSE simulator loss ---
def train_mse_simulator(model, train_loader, val_loader, device, simulator, problem="simplified_dis", 
                       epochs=100, lr=1e-4, nevents=1000, use_simulator_loss=True):
    """
    Train model using MSE loss between simulator outputs.
    
    Args:
        model: Neural network model to train
        train_loader: Training data loader
        val_loader: Validation data loader  
        device: Device to train on
        simulator: Simulator instance for computing loss
        problem: Problem type ("simplified_dis", etc.)
        epochs: Number of training epochs
        lr: Learning rate
        nevents: Number of events for simulator loss computation
        use_simulator_loss: If True, use simulator MSE loss; if False, use parameter MSE
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler()
    best_val_loss = float('inf')
    
    print(f"Training with {'simulator MSE' if use_simulator_loss else 'parameter MSE'} loss...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for latent, target in train_loader:
            latent, target = latent.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with amp.autocast(dtype=torch.float16):
                pred = model(latent)
                
                if use_simulator_loss:
                    # Use simulator-based MSE loss
                    loss = pdf_theta_mse_loss_batched(pred, target, simulator, problem, nevents)
                else:
                    # Use standard parameter MSE loss
                    loss = F.mse_loss(pred, target)

                print(loss)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for latent, target in val_loader:
                latent, target = latent.to(device), target.to(device)
                pred = model(latent)
                
                if use_simulator_loss:
                    # Use simulator-based MSE loss for validation too
                    val_loss += pdf_theta_mse_loss_batched(pred, target, simulator, problem, nevents).item()
                else:
                    # Use standard parameter MSE loss
                    val_loss += F.mse_loss(pred, target).item()
        
        val_loss /= len(val_loader)
        loss_type = "Sim-MSE" if use_simulator_loss else "Param-MSE"
        print(f"[{loss_type}] Epoch {epoch} Train {epoch_loss/len(train_loader):.4f} Val {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_mse_simulator_model_{problem}.pth")
    
    return model

def train_gaussian(model, train_loader, val_loader, device, epochs=100, lr=1e-4, multimodal=False, nmodes=2):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler()
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for latent, target in train_loader:
            latent, target = latent.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(dtype=torch.float16):
                if multimodal:
                    means, logvars, weights = model(latent)
                    # Mixture of Gaussians NLL loss
                    loss = 0.0
                    for i in range(latent.shape[0]):
                        sample = target[i]
                        mode_lls = []
                        for m in range(nmodes):
                            mean = means[i, m]
                            var = torch.exp(logvars[i, m])
                            dist = Normal(mean, torch.sqrt(var))
                            mode_lls.append(dist.log_prob(sample).sum())
                        log_mix = torch.log(weights[i] + 1e-8)
                        mix_ll = torch.logsumexp(torch.stack(mode_lls) + log_mix, dim=0)
                        loss += -mix_ll
                    loss = loss / latent.shape[0]
                else:
                    means, logvars = model(latent)
                    var = torch.exp(logvars)
                    dist = Normal(means, torch.sqrt(var))
                    loss = -dist.log_prob(target).sum(dim=-1).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for latent, target in val_loader:
                latent, target = latent.to(device), target.to(device)
                if multimodal:
                    means, logvars, weights = model(latent)
                    loss = 0.0
                    for i in range(latent.shape[0]):
                        sample = target[i]
                        mode_lls = []
                        for m in range(nmodes):
                            mean = means[i, m]
                            var = torch.exp(logvars[i, m])
                            dist = Normal(mean, torch.sqrt(var))
                            mode_lls.append(dist.log_prob(sample).sum())
                        log_mix = torch.log(weights[i] + 1e-8)
                        mix_ll = torch.logsumexp(torch.stack(mode_lls) + log_mix, dim=0)
                        loss += -mix_ll
                    loss = loss / latent.shape[0]
                else:
                    means, logvars = model(latent)
                    var = torch.exp(logvars)
                    dist = Normal(means, torch.sqrt(var))
                    loss = -dist.log_prob(target).sum(dim=-1).mean()
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"[Gaussian Model] Epoch {epoch} Train {epoch_loss/len(train_loader):.4f} Val {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_gaussian_model.pth")
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--problem", type=str, default="simplified_dis", choices=["simplified_dis", "realistic_dis", "mceg"])
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=4000)
    parser.add_argument("--num_events", type=int, default=100000)
    parser.add_argument("--cl_model", type=str, default="final_model.pth", help="Path to pre-trained PointNetPMA model (assumes same experiment directory).")
    parser.add_argument("--arch", type=str, default="all", choices=["mlp", "transformer", "gaussian", "multimodal", "all"])
    parser.add_argument("--use_simulator_loss", action="store_true", help="Use simulator MSE loss instead of parameter MSE")
    parser.add_argument("--nevents_loss", type=int, default=1000, help="Number of events for simulator loss computation")
    args = parser.parse_args()

    # Add MSE suffix to output directory when using simulator loss
    suffix = "_mse" if args.use_simulator_loss else ""
    output_dir = os.path.join("experiments", f"{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thetas, xs = generate_data(args.num_samples, args.num_events, problem=args.problem, device=device)
    if args.problem == "simplified_dis":
        input_dim = 6
    elif args.problem == "realistic_dis":
        input_dim = 12
    elif args.problem == "mceg":
        input_dim = 2
    param_dim = thetas.size(-1)

    # Load PointNetPMA encoder
    pointnet_model = PointNetPMA(input_dim=input_dim, latent_dim=args.latent_dim, predict_theta=True)
    state_dict = torch.load(output_dir + '/' + args.cl_model, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    pointnet_model.load_state_dict(state_dict)
    pointnet_model.eval()
    pointnet_model.to(device)
    if args.problem in ["simplified_dis", "realistic_dis"]:
        xs_tensor_engineered = log_feature_engineering(xs)
    else:
        xs_tensor_engineered = xs

    # Initialize PointNetEmbedding model (do this once)
    input_dim = xs_tensor_engineered.shape[-1]
    print(f"[precompute] Input dimension: {input_dim}")
    print(f"xs_tensor_engineered shape: {xs_tensor_engineered.shape}")
    
    latent_path = os.path.join(output_dir, "latents.h5")
    precompute_latents_to_disk(
        pointnet_model, xs_tensor_engineered, thetas, latent_path, args.latent_dim, chunk_size=64
    )
    del xs_tensor_engineered
    del xs

    dataset = H5Dataset(latent_path)
    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=EventDataset.collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=EventDataset.collate_fn)

    # Initialize simulator for loss computation
    if args.problem == "simplified_dis":
        simulator = SimplifiedDIS(device=device)
    elif args.problem == "realistic_dis":
        simulator = RealisticDIS(device=device)
    else:
        simulator = MCEGSimulator(device=device)

    # Train and Laplace all heads with option for simulator MSE loss
    if args.arch in ["mlp", "all"]:
        print("Training MLP with simulator MSE loss..." if args.use_simulator_loss else "Training MLP...")
        mlp_head = MLPHead(args.latent_dim, param_dim)
        
        if args.use_simulator_loss:
            trained_mlp = train_mse_simulator(
                mlp_head, train_loader, val_loader, device, simulator, 
                problem=args.problem, epochs=args.epochs, nevents=args.nevents_loss,
                use_simulator_loss=True
            )
            torch.save(trained_mlp.state_dict(), os.path.join(output_dir, "mlp_head_mse_final.pth"))
        else:
            trained_mlp = train_standard(mlp_head, train_loader, val_loader, device, epochs=args.epochs)
            torch.save(trained_mlp.state_dict(), os.path.join(output_dir, "mlp_head_final.pth"))
        
        # Laplace approximation (works with both loss types)
        from laplace.laplace import Laplace
        lap_mlp = Laplace(trained_mlp, 'regression',
                        subset_of_weights='last_layer',
                        hessian_structure='kron')
        lap_mlp.fit(train_loader)
        
        filename = "laplace_mlp_mse.pt" if args.use_simulator_loss else "laplace_mlp.pt"
        print(f"Saving Laplace MLP to {filename}")
        save_laplace(lap_mlp, output_dir,
                    filename=filename,
                    likelihood="regression",
                    subset_of_weights="last_layer",
                    hessian_structure="kron")

    if args.arch in ["transformer", "all"]:
        print("Training Transformer with simulator MSE loss..." if args.use_simulator_loss else "Training Transformer...")
        transformer_head = TransformerHead(args.latent_dim, param_dim)
        
        if args.use_simulator_loss:
            trained_transformer = train_mse_simulator(
                transformer_head, train_loader, val_loader, device, simulator,
                problem=args.problem, epochs=args.epochs, nevents=args.nevents_loss,
                use_simulator_loss=True
            )
            torch.save(trained_transformer.state_dict(), os.path.join(output_dir, "transformer_head_mse_final.pth"))
        else:
            trained_transformer = train_standard(transformer_head, train_loader, val_loader, device, epochs=args.epochs)
            torch.save(trained_transformer.state_dict(), os.path.join(output_dir, "transformer_head_final.pth"))
        
        from laplace import Laplace
        lap_transformer = Laplace(trained_transformer, 'regression', subset_of_weights='last_layer', hessian_structure='kron')
        lap_transformer.fit(train_loader)
        
        filename = "laplace_transformer_mse.pt" if args.use_simulator_loss else "laplace_transformer.pt"
        print(f"Saving Laplace Transformer to {filename}")
        save_laplace(lap_transformer, output_dir,
                    filename=filename,
                    likelihood="regression",
                    subset_of_weights="last_layer",
                    hessian_structure="kron")

    if args.arch in ["gaussian", "all"]:
        print("Training Gaussian (NLL loss, with Laplace)...")
        from laplace import Laplace
        # Note: Gaussian models use NLL loss, not simulator MSE loss
        gaussian_head = GaussianHead(args.latent_dim, param_dim).to(device)
        trained_gaussian = train_gaussian(gaussian_head, train_loader, val_loader, device, epochs=args.epochs)
        torch.save(trained_gaussian.state_dict(), os.path.join(output_dir, "gaussian_head_final.pth"))

    if args.arch in ["multimodal", "all"]:
        print("Training Mixture of Gaussians (NLL loss, with Laplace)...")
        multimodal_head = GaussianHead(args.latent_dim, param_dim, multimodal=True, nmodes=2)
        trained_multimodal = train_gaussian(
            multimodal_head, train_loader, val_loader, device, epochs=args.epochs, multimodal=True, nmodes=2
        )
        torch.save(trained_multimodal.state_dict(), os.path.join(output_dir, "multimodal_head_final.pth"))
        print("Mixture of Gaussians model trained and saved (NLL loss).")

    print("All models trained and saved.")
    print(f"Models saved to: {output_dir}")
    
    if args.use_simulator_loss:
        print("\n--- USAGE NOTES ---")
        print("Models were trained using MSE loss between simulator outputs.")
        print("For analysis/evaluation, you can use the original pdf_theta_loss function")
        print("by calling pdf_theta_loss(theta_pred, theta_true, simulator, problem).")
        print("This computes log-relative PDF discrepancy for comparison purposes.")
        print("Laplace approximation is compatible with both training approaches.")

if __name__ == "__main__":
    main()