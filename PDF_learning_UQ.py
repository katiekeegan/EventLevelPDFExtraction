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
    return las
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
class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

class _Coupling(nn.Module):
    """
    Affine coupling: splits theta by mask; s,t depend on (theta_masked, z).
    """
    def __init__(self, theta_dim, z_dim, hidden=256, scale_clamp=5.0, mask=None):
        super().__init__()
        self.theta_dim = theta_dim
        self.z_dim = z_dim
        self.scale_clamp = scale_clamp
        if mask is None:
            raise ValueError("mask is required")
        self.register_buffer("mask", mask.float())  # [D]

        in_dim = theta_dim + z_dim  # we concat masked theta and z
        self.s_net = _MLP(in_dim, theta_dim, hidden)
        self.t_net = _MLP(in_dim, theta_dim, hidden)

    def forward(self, theta, z, inverse=False):
        """
        theta: [B,D], z: [B,Z]
        Returns: theta_out, log_det (per-sample) for the direction used.
        """
        m = self.mask
        theta_masked = theta * m  # [B,D]
        h = torch.cat([theta_masked, z], dim=-1)  # [B, D+Z]
        s = self.s_net(h)
        t = self.t_net(h)
        # stable scale
        s = torch.tanh(s) * self.scale_clamp  # [-clamp, clamp]

        if not inverse:
            # y = x_masked + (1-m) * [ (x * exp(s)) + t ]
            y = theta_masked + (1 - m) * (theta * torch.exp(s) + t)
            log_det = ((1 - m) * s).sum(dim=-1)  # [B]
            return y, log_det
        else:
            # x = y_masked + (1-m) * [ (y - t) * exp(-s) ]
            x = theta_masked + (1 - m) * ((theta - t) * torch.exp(-s))
            log_det = -((1 - m) * s).sum(dim=-1)  # [B]
            return x, log_det

class ConditionalRealNVP(nn.Module):
    """
    Conditional flow p_theta(. | z). Base = N(0,I) in D=param_dim.
    """
    def __init__(self, z_dim, param_dim, n_layers=8, hidden=256, scale_clamp=5.0):
        super().__init__()
        self.z_dim = z_dim
        self.param_dim = param_dim

        # alternating binary masks
        masks = []
        for i in range(n_layers):
            mask = torch.zeros(param_dim)
            mask[i % 2 :: 2] = 1.0
            masks.append(mask)
        self.layers = nn.ModuleList([
            _Coupling(param_dim, z_dim, hidden=hidden, scale_clamp=scale_clamp, mask=m)
            for m in masks
        ])
        self.base_mean = nn.Parameter(torch.zeros(param_dim), requires_grad=False)
        self.base_logstd = nn.Parameter(torch.zeros(param_dim), requires_grad=False)

    def forward(self, z, n_samples=None):
        """
        Sampling: draw theta ~ p(.|z).
        z: [B,Z]; if n_samples is None, returns one sample per z -> [B,D]
        else returns [B,n_samples,D]
        """
        if n_samples is None:
            eps = torch.randn(z.size(0), self.param_dim, device=z.device, dtype=z.dtype)
            return self._transform(eps, z, inverse=True)[0]  # [B,D]
        else:
            B = z.size(0)
            eps = torch.randn(B * n_samples, self.param_dim, device=z.device, dtype=z.dtype)
            z_rep = z.repeat_interleave(n_samples, dim=0)                      # [B*n,Dz]
            theta, _ = self._transform(eps, z_rep, inverse=True)
            return theta.view(B, n_samples, self.param_dim)                    # [B,S,D]

    def log_prob(self, theta, z):
        """
        theta: [B,D], z: [B,Z] -> log p(theta|z) [B]
        """
        u, logdet = self._transform(theta, z, inverse=False)   # to base
        log_pu = -0.5 * (((u - self.base_mean) ** 2) * torch.exp(-2*self.base_logstd)).sum(dim=-1)
        log_pu += -0.5 * self.param_dim * math.log(2 * math.pi) - self.base_logstd.sum()
        return log_pu + logdet

    def _transform(self, theta, z, inverse=False):
        """
        Run through all coupling layers; accumulate logdet.
        Direction depends on inverse flag.
        """
        logdet = torch.zeros(theta.size(0), device=theta.device, dtype=theta.dtype)
        if not inverse:
            h = theta
            for layer in self.layers:
                h, ld = layer(h, z, inverse=False)
                logdet = logdet + ld
            return h, logdet
        else:
            h = theta
            for layer in reversed(self.layers):
                h, ld = layer(h, z, inverse=True)
                logdet = logdet + ld
            return h, logdet
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

def train_flow(flow, train_loader, val_loader, device, epochs=200, lr=2e-4):
    flow = flow.to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    scaler = amp.GradScaler()
    best = float('inf')

    for ep in range(epochs):
        flow.train()
        tr = 0.0
        for z, theta in train_loader:
            z, theta = z.to(device), theta.to(device)
            opt.zero_grad(set_to_none=True)
            with amp.autocast(dtype=torch.float16):
                nll = -flow.log_prob(theta, z).mean()
            scaler.scale(nll).backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tr += nll.item()
        tr /= len(train_loader)

        # val NLL
        flow.eval()
        va = 0.0
        with torch.no_grad():
            for z, theta in val_loader:
                z, theta = z.to(device), theta.to(device)
                va += (-flow.log_prob(theta, z).mean()).item()
        va /= len(val_loader)
        print(f"[Flow] Epoch {ep:03d} | Train NLL {tr:.4f} | Val NLL {va:.4f}")

        if va < best:
            best = va
            torch.save(flow.state_dict(), "best_flow.pth")
    return flow

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
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class TransformerHead(nn.Module):
    def __init__(self, embedding_dim, out_dim, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(embedding_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=nhead, dropout=dropout)
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--problem", type=str, default="simplified_dis", choices=["simplified_dis", "realistic_dis", "mceg"])
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=4000)
    parser.add_argument("--num_events", type=int, default=100000)
    parser.add_argument("--cl_model", type=str, default="final_model.pth", help="Path to pre-trained PointNetPMA model (assumes same experiment directory).")
    parser.add_argument("--arch", type=str, default="all", choices=["mlp", "transformer", "gaussian", "multimodal", "all"])
    args = parser.parse_args()

    output_dir = os.path.join("experiments", f"{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thetas, xs = generate_data(args.num_samples, 10000, problem=args.problem, device=device)
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

    # latent_path = "simplified_dis_latent_features.h5"
    # if not os.path.exists(latent_path):
    if args.problem != "mceg":
        xs_tensor_engineered = log_feature_engineering(xs)
    else:
        xs_tensor_engineered = xs


    # Initialize PointNetEmbedding model (do this once)
    input_dim = xs_tensor_engineered.shape[-1]
    print(f"[precompute] Input dimension: {input_dim}")
    print(f"xs_tensor_engineered shape: {xs_tensor_engineered.shape}")
    latent_path = os.path.join(output_dir, f"{args.problem}_latent_features.h5")
    # def precompute_latents_to_disk(pointnet_model, xs_tensor, thetas, output_path, chunk_size=4):
    precompute_latents_to_disk(
        pointnet_model, xs_tensor_engineered, thetas, latent_path, args.latent_dim, chunk_size=8
    )
    del xs_tensor_engineered
    del xs

    dataset = H5Dataset(latent_path)
    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=EventDataset.collate_fn)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=EventDataset.collate_fn)

    # Train and Laplace all heads
    if args.arch in ["mlp", "all"]:
        print("Training MLP...")
        mlp_head = MLPHead(args.latent_dim, param_dim)
        trained_mlp = train_standard(mlp_head, train_loader, val_loader, device, epochs=args.epochs)
        torch.save(trained_mlp.state_dict(), os.path.join(output_dir, "mlp_head_final.pth"))
        from laplace.laplace import Laplace

        lap_mlp = Laplace(trained_mlp, 'regression',
                        subset_of_weights='last_layer',
                        hessian_structure='kron')
        lap_mlp.fit(train_loader)
        # after fitting:
        save_laplace(lap_mlp, output_dir,
                    filename="laplace_mlp.pt",
                    likelihood="regression",
                    subset_of_weights="last_layer",
                    hessian_structure="kron")

    if args.arch in ["transformer", "all"]:
        print("Training Transformer...")
        transformer_head = TransformerHead(args.latent_dim, param_dim)
        trained_transformer = train_standard(transformer_head, train_loader, val_loader, device, epochs=args.epochs)
        torch.save(trained_transformer.state_dict(), os.path.join(output_dir, "transformer_head_final.pth"))
        from laplace import Laplace
        lap_transformer = Laplace(trained_transformer, 'regression', subset_of_weights='last_layer', hessian_structure='kron')
        lap_transformer.fit(train_loader)
        # after fitting:
        save_laplace(lap_transformer, output_dir,
                    filename="laplace_transformer.pt",
                    likelihood="regression",
                    subset_of_weights="last_layer",
                    hessian_structure="kron")

    if args.arch in ["flow", "all"]:
        print("Training Conditional RealNVP flow...")
        flow = ConditionalRealNVP(z_dim=args.latent_dim, param_dim=param_dim,
                                n_layers=8, hidden=256, scale_clamp=5.0).to(device)
        trained_flow = train_flow(flow, train_loader, val_loader, device,
                                epochs=args.epochs, lr=2e-4)
        torch.save(trained_flow.state_dict(), os.path.join(output_dir, "flow_final.pth"))


    # if args.arch in ["gaussian", "all"]:
    #     print("Training Gaussian (NLL loss, with Laplace)...")
    #     from laplace import Laplace
    #     # after training + saving the gaussian head
    #     gaussian_head = GaussianHead(args.latent_dim, param_dim).to(device)
    #     trained_gaussian = train_gaussian(gaussian_head, train_loader, val_loader, device, epochs=args.epochs)
    #     torch.save(trained_gaussian.state_dict(), os.path.join(output_dir, "gaussian_head_final.pth"))

    # for n, p in trained_gaussian.named_parameters():
    #     if "logstd" in n or "log_std" in n or "sigma" in n:
    #         p.requires_grad_(False)

    # wrapped_mean_model = MeanOnlyWrapper(trained_gaussian).to(device).eval()

    # # Choose the last *mean* layer explicitly:
    # mean_last = last_linear(wrapped_mean_model)  # if your architecture mixes paths,
    #                                             # replace with a direct handle to the mean head's final Linear

    # lap_gaussian = Laplace(
    #     wrapped_mean_model,
    #     likelihood='regression',
    #     subset_of_weights='last_layer',
    #     hessian_structure='kron',
    #     last_layer=mean_last,              # <-- critical line
    # )

    # lap_gaussian.fit(train_loader)
    # lap_gaussian.optimize_prior_precision(method='marglik')

    # torch.save(lap_gaussian.state_dict(), os.path.join(output_dir, "laplace_gaussian_state.pth"))

    # if args.arch in ["multimodal", "all"]:
    #     print("Training Mixture of Gaussians (NLL loss, with Laplace)...")
    #     multimodal_head = GaussianHead(args.latent_dim, param_dim, multimodal=True, nmodes=2)
    #     trained_multimodal = train_gaussian(
    #         multimodal_head, train_loader, val_loader, device, epochs=args.epochs, multimodal=True, nmodes=2
    #     )
    #     torch.save(trained_multimodal.state_dict(), os.path.join(output_dir, "multimodal_head_final.pth"))
    #     print("Mixture of Gaussians model trained and saved (NLL loss).")

        # # Fit Laplace and save Laplace object
        # from laplace import Laplace
        # lap_multimodal = Laplace(trained_multimodal, 'regression', subset_of_weights='all')
        # lap_multimodal.fit(train_loader)
        # laplace_path = os.path.join(output_dir, "laplace_multimodal.pt")
        # torch.save(lap_multimodal.state_dict(), laplace_path)  # <--- FIXED
        # print(f"Laplace approximation fitted and saved to {laplace_path}")

    print("All models trained and saved (NLL loss, Laplace fits saved).")
if __name__ == "__main__":
    main()
