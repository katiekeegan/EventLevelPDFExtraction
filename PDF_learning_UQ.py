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

# Laplace-torch for Laplace approximation
from laplace import Laplace

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# --- Add this at the top of your file with other imports ---
import torch.nn as nn
import torch

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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--problem", type=str, default="simplified_dis", choices=["simplified_dis", "realistic_dis"])
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_events", type=int, default=100000)
    parser.add_argument("--arch", type=str, default="all", choices=["mlp", "transformer", "gaussian", "multimodal", "all"])
    args = parser.parse_args()

    output_dir = os.path.join("experiments", f"{args.problem}_latent_{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thetas, xs = generate_data(args.num_samples, args.num_events, problem=args.problem, device=device)
    input_dim = 6 if args.problem == "simplified_dis" else 12
    param_dim = thetas.size(-1)

    # Load PointNetPMA encoder
    pointnet_model = PointNetPMA(input_dim=input_dim, latent_dim=args.latent_dim, predict_theta=True)
    state_dict = torch.load("experiments/simplified_dis_latent1024_ns_1000_ne_100000/final_model.pth", map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    pointnet_model.load_state_dict(state_dict)
    pointnet_model.eval()
    pointnet_model.to(device)

    latent_path = "latent_features.h5"
    if not os.path.exists(latent_path):
        xs_tensor_engineered = log_feature_engineering(xs)

        # Initialize PointNetEmbedding model (do this once)
        input_dim = xs_tensor_engineered.shape[-1]
        print(f"[precompute] Input dimension: {input_dim}")
        print(f"xs_tensor_engineered shape: {xs_tensor_engineered.shape}")
        precompute_latents_to_disk(
            pointnet_model, xs_tensor_engineered, latent_path, chunk_size=8
        )
        del xs_tensor_engineered
        del xs

    dataset = H5Dataset(latent_path)
    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=EventDataset.collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=EventDataset.collate_fn)

    # Train and Laplace all heads
    if args.arch in ["mlp", "all"]:
        print("Training MLP...")
        mlp_head = MLPHead(args.latent_dim, param_dim)
        trained_mlp = train_standard(mlp_head, train_loader, val_loader, device, epochs=args.epochs)
        lap_mlp = fit_laplace(trained_mlp, train_loader, regression=True)
        torch.save(trained_mlp.state_dict(), os.path.join(output_dir, "mlp_head_final.pth"))
        torch.save(lap_mlp.state_dict(), os.path.join(output_dir, "laplace_mlp_state.pt"))

    if args.arch in ["transformer", "all"]:
        print("Training Transformer...")
        transformer_head = TransformerHead(args.latent_dim, param_dim)
        trained_transformer = train_standard(transformer_head, train_loader, val_loader, device, epochs=args.epochs)
        lap_transformer = fit_laplace(trained_transformer, train_loader, regression=True)
        torch.save(trained_transformer.state_dict(), os.path.join(output_dir, "transformer_head_final.pth"))
        torch.save(lap_transformer.state_dict(), os.path.join(output_dir, "laplace_transformer_state.pt"))
    # --- Main patch for Laplace fitting and saving ---
    if args.arch in ["gaussian", "all"]:
        print("Training Gaussian (NLL loss, with Laplace)...")
        gaussian_head = GaussianHead(args.latent_dim, param_dim)
        trained_gaussian = train_gaussian(
            gaussian_head, train_loader, val_loader, device, epochs=args.epochs
        )
        # Save the trained Gaussian head
        torch.save(trained_gaussian.state_dict(), os.path.join(output_dir, "gaussian_head_final.pth"))
        print("Gaussian model trained and saved (NLL loss).")

        # Fit Laplace and save Laplace object
        from laplace import Laplace
        lap_gaussian = Laplace(trained_gaussian, 'regression', subset_of_weights='all')
        lap_gaussian.fit(train_loader)
        laplace_path = os.path.join(output_dir, "laplace_gaussian.pt")
        torch.save(lap_gaussian.state_dict(), laplace_path)
        print(f"Laplace approximation fitted and saved to {laplace_path}")

    if args.arch in ["multimodal", "all"]:
        print("Training Mixture of Gaussians (NLL loss, with Laplace)...")
        multimodal_head = GaussianHead(args.latent_dim, param_dim, multimodal=True, nmodes=2)
        trained_multimodal = train_gaussian(
            multimodal_head, train_loader, val_loader, device, epochs=args.epochs, multimodal=True, nmodes=2
        )
        torch.save(trained_multimodal.state_dict(), os.path.join(output_dir, "multimodal_head_final.pth"))
        print("Mixture of Gaussians model trained and saved (NLL loss).")

        # Fit Laplace and save Laplace object
        from laplace import Laplace
        lap_multimodal = Laplace(trained_multimodal, 'regression', subset_of_weights='all')
        lap_multimodal.fit(train_loader)
        laplace_path = os.path.join(output_dir, "laplace_multimodal.pt")
        torch.save(lap_multimodal.state_dict(), laplace_path)
        print(f"Laplace approximation fitted and saved to {laplace_path}")


    print("All models trained and saved (NLL loss, no Laplace applied).")
if __name__ == "__main__":
    main()