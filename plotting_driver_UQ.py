"""
PDF Parameter Inference Plotting Driver with Analytic Laplace Uncertainty

This script reloads and generates plots for each model architecture with
ANALYTIC uncertainty propagation using Laplace approximation (delta method).

Key Features:
- Uses analytic uncertainty propagation instead of Monte Carlo sampling
- Supports all architectures: MLP, Transformer, Gaussian, and Multimodal
- Automatically detects and loads Laplace approximations when available
- Falls back to Monte Carlo for backward compatibility when Laplace unavailable
- Faster and more accurate uncertainty quantification

Uncertainty Methods:
- When Laplace model available: Uses analytic delta method for uncertainty
- When Laplace unavailable: Falls back to Monte Carlo sampling
- For Gaussian heads: Uses intrinsic uncertainty from predicted variances

Usage:
    # Use analytic uncertainty (recommended - requires Laplace models):
    python plotting_driver_UQ.py --arch gaussian --latent_dim 1024 --param_dim 4 --problem simplified_dis
    
    # Plot all architectures with analytic uncertainty:
    python plotting_driver_UQ.py --arch all --latent_dim 1024 --param_dim 4 --problem simplified_dis
    
    # Reduce n_mc for faster fallback when Laplace unavailable:
    python plotting_driver_UQ.py --arch mlp --n_mc 50 --latent_dim 1024 --param_dim 4
"""

import torch
import os
import numpy as np
from PDF_learning import *
from simulator import *
from models import *
from plotting_UQ_utils import *
from datasets import *
from PDF_learning_UQ import *

# utils_laplace.py
import os, torch
from laplace.laplace import Laplace

def build_head(arch, latent_dim, param_dim, device, nmodes=None):
    if arch == "mlp":
        return MLPHead(latent_dim, param_dim).to(device)
    if arch == "transformer":
        return TransformerHead(latent_dim, param_dim).to(device)
    raise ValueError(f"Unknown arch: {arch}")

def _ensure_dir(path): Path(path).mkdir(parents=True, exist_ok=True)

def plot_1d_marginals(samples, outdir, prefix, theta_gt=None):
    """
    samples: [S, D] tensor/cpu-np
    theta_gt: [D] or None
    """
    X = samples.detach().cpu().numpy() if torch.is_tensor(samples) else samples
    S, D = X.shape
    cols = min(4, D)
    rows = int(math.ceil(D / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0*cols, 2.4*rows), squeeze=False)
    for d in range(D):
        r, c = divmod(d, cols)
        ax = axes[r][c]
        ax.hist(X[:, d], bins=60, density=True)
        ax.set_title(f"$\\theta_{d}$")
        if theta_gt is not None:
            g = theta_gt[d].item() if torch.is_tensor(theta_gt) else float(theta_gt[d])
            ax.axvline(g, linestyle="--")
    # hide empties
    for k in range(D, rows*cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_1d_marginals.png"), dpi=200)
    plt.close(fig)

def plot_pairwise_2d(samples, outdir, prefix, max_pairs=6, theta_gt=None):
    """
    samples: [S, D]
    Plots up to max_pairs unique 2D pairs.
    """
    X = samples.detach().cpu().numpy() if torch.is_tensor(samples) else samples
    S, D = X.shape
    pairs = []
    for i in range(D):
        for j in range(i+1, D):
            pairs.append((i, j))
    pairs = pairs[:max_pairs]
    cols = min(3, len(pairs)) or 1
    rows = int(math.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 3.2*rows), squeeze=False)
    for k, (i, j) in enumerate(pairs):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        ax.scatter(X[:, i], X[:, j], s=4, alpha=0.25)
        ax.set_xlabel(f"$\\theta_{i}$")
        ax.set_ylabel(f"$\\theta_{j}$")
        if theta_gt is not None:
            ax.scatter([theta_gt[i].item() if torch.is_tensor(theta_gt) else float(theta_gt[i])],
                       [theta_gt[j].item() if torch.is_tensor(theta_gt) else float(theta_gt[j])],
                       s=40, marker="x")
    # hide empties
    for k in range(len(pairs), rows*cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_pairs2d.png"), dpi=200)
    plt.close(fig)

def plot_corner(samples, outpath, labels=None, max_dim=6):
    """
    Quick corner-like plot for up to max_dim dims.
    """
    X = samples.detach().cpu().numpy() if torch.is_tensor(samples) else samples
    S, D = X.shape
    Dv = min(D, max_dim)
    fig, axes = plt.subplots(Dv, Dv, figsize=(2.2*Dv, 2.2*Dv))
    for i in range(Dv):
        for j in range(Dv):
            ax = axes[i, j]
            if i == j:
                ax.hist(X[:, j], bins=50, density=True)
                if labels: ax.set_xlabel(labels[j])
            elif i > j:
                ax.scatter(X[:, j], X[:, i], s=2, alpha=0.2)
                if labels and i == Dv-1: ax.set_xlabel(labels[j])
                if labels and j == 0:    ax.set_ylabel(labels[i])
            else:
                ax.axis('off')
    fig.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def _flow_checkpoint_present(path_or_dir: str) -> bool:
    """
    Return True if a flow checkpoint exists at the exact path OR anywhere in the directory.
    """
    if not FLOW_CLASS_OK:
        return False
    if not path_or_dir:
        return False
    if os.path.isfile(path_or_dir):
        return ("flow" in os.path.basename(path_or_dir).lower()) or path_or_dir.endswith(".pth")
    if os.path.isdir(path_or_dir):
        matches = glob.glob(os.path.join(path_or_dir, "*flow*.pth")) + \
                  glob.glob(os.path.join(path_or_dir, "*flow*/*.pth"))
        return len(matches) > 0
    # path doesn’t exist yet; conservatively return False
    return False

def load_latents_and_optional_theta(args, device):
    z = None; theta = None
    if args.latent_npy:
        z = torch.from_numpy(np.load(args.latent_npy)).float().to(device)
    if args.theta_npy:
        theta = torch.from_numpy(np.load(args.theta_npy)).float().to(device)
    return z, theta
HEAD_PTH = {
    "mlp": "mlp_head_final.pth",
    "transformer": "transformer_head_final.pth",
}

LAPLACE_CANDIDATES = lambda arch: [
    f"laplace_{arch}.pt",
    f"laplace_{arch}.ckpt",
    # f"laplace_{arch}_state.pt",
    # historical fallbacks (optional):
    # "laplace_mlp.pt" if arch == "mlp" else None,
    # "laplace_transformer.pt" if arch == "transformer" else None,
]

def make_model(experiment_dir, arch, device, args):
    m = build_head(arch, args.latent_dim, args.param_dim, device, nmodes=getattr(args, "nmodes", None))
    map_path = os.path.join(experiment_dir, HEAD_PTH[arch])
    if not os.path.exists(map_path):
        print(f"⚠️  Expected MAP weights not found for {arch}: {map_path}")
        return m.eval()  # return uninitialized head so Laplace still constructs
    try:
        state = torch.load(map_path, map_location=device)
        m.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # Helpful debug: show missing/unexpected keys and continue with strict=False
        print("⚠️  Strict load failed for", arch, "->", map_path)
        print(e)
        missing, unexpected = [], []
        try:
            # parse message lightly
            msg = str(e)
            print("Attempting strict=False load as fallback.")
        except Exception:
            pass
        m.load_state_dict(state, strict=False)
    return m.eval()

def _finalize_device_eval(la, device):
    if hasattr(la, "model") and la.model is not None:
        la.model.to(device).eval()
    return la

def load_model(arch, checkpoint, latent_dim, param_dim, device):
    if arch == "gaussian":
        model = GaussianHead(latent_dim, param_dim, multimodal=False).to(device)
    elif arch == "multimodal":
        # nmodes was set during training; the head stores its layers, so loading works
        model = GaussianHead(latent_dim, param_dim, multimodal=True).to(device)
    elif arch == "flow":
        model = ConditionalRealNVP(z_dim=latent_dim, param_dim=param_dim,
                                   n_layers=8, hidden=256, scale_clamp=5.0).to(device)
    else:
        raise ValueError(f"Unknown arch {arch}")
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

@torch.no_grad()
def draw_posterior_samples(model, arch, z, nsamples):
    """
    z: [B, Dz]  -> returns [B, nsamples, Dtheta]
    """
    if arch in ("gaussian", "multimodal"):
        return sample_gaussian_posterior(model, z, nsamples=nsamples,
                                         multimodal=(arch == "multimodal"))
    elif arch == "flow":
        # flow forward() already does conditional sampling
        return model(z, n_samples=nsamples)   # [B, S, D]
    else:
        raise ValueError(arch)

    
def reload_model(arch, latent_dim, param_dim, experiment_dir, device, multimodal=False, nmodes=2):
    """
    Reloads model from checkpoint for the specified architecture.
    """
    if arch == "mlp":
        checkpoint_path = os.path.join(experiment_dir, "mlp_head_final.pth")
        model = MLPHead(latent_dim, param_dim).to(device)
    elif arch == "transformer":
        checkpoint_path = os.path.join(experiment_dir, "transformer_head_final.pth")
        model = TransformerHead(latent_dim, param_dim).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found for {arch}: {checkpoint_path}")
        return None
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def reload_pointnet(experiment_dir, latent_dim, device, cl_model='final_model.pth', problem='simplified_dis'):
    """
    Reloads PointNet model from the experiment directory.
    """
    # pointnet_path = os.path.join(experiment_dir, "final_model.pth")
    pointnet_path = os.path.join(experiment_dir, cl_model)
    # Dummy input for input_dim inference
    xs_dummy = np.random.randn(100, 2)
    xs_dummy_tensor = torch.tensor(xs_dummy, dtype=torch.float32)
    if problem != 'mceg':
        input_dim = advanced_feature_engineering(xs_dummy_tensor).shape[-1]
    else:
        input_dim = xs_dummy_tensor.shape[-1]
    pointnet_model = PointNetPMA(input_dim=input_dim, latent_dim=latent_dim).to(device)
    state_dict = torch.load(pointnet_path, map_location=device)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    pointnet_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    pointnet_model.eval()
    return pointnet_model
def load_laplace(make_model_fn, ckpt_path, device="cpu",
                 default_likelihood="regression",
                 default_subset="last_layer",
                 default_hessian="kron"):
    if not os.path.exists(ckpt_path):
        return None
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Case A: dict with 'laplace_state' [+ optional 'meta']
    if isinstance(obj, dict) and "laplace_state" in obj:
        meta = obj.get("meta", {
            "likelihood": default_likelihood,
            "subset_of_weights": default_subset,
            "hessian_structure": default_hessian,
        })
        model = make_model_fn()  # this already .to(device).eval()
        la = Laplace(model,
                     meta["likelihood"],
                     subset_of_weights=meta["subset_of_weights"],
                     hessian_structure=meta["hessian_structure"])
        la.load_state_dict(obj["laplace_state"])
        return _finalize_device_eval(la, device)

    # Case B: plain state_dict (no meta)
    if isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values()):
        model = make_model_fn()
        la = Laplace(model,
                     default_likelihood,
                     subset_of_weights=default_subset,
                     hessian_structure=default_hessian)
        la.load_state_dict(obj)
        return _finalize_device_eval(la, device)

    # Case C: pickled Laplace object
    if hasattr(obj, "model"):
        la = obj
        return _finalize_device_eval(la, device)

    print(f"⚠️  Unrecognized Laplace checkpoint format: {ckpt_path}")
    return None

def _get_checkpoint_arg(args):
    # Works even if old scripts forget to add --checkpoint
    return getattr(args, "checkpoint", "") or ""

def load_laplace_if_available(experiment_dir, arch, device, args):
    # try several filenames in order
    for name in filter(None, LAPLACE_CANDIDATES(arch)):
        ckpt_path = os.path.join(experiment_dir, name)
        if not os.path.exists(ckpt_path):
            continue
        def make_model_fn():
            return make_model(experiment_dir, arch, device, args)
        la = load_laplace(make_model_fn, ckpt_path, device=device,
                          default_likelihood="regression",
                          default_subset="last_layer",
                          default_hessian="kron")
        if la is not None:
            print(f"✓ Loaded Laplace from {ckpt_path}")
            return la
    print(f"ℹ️  No Laplace checkpoint found for {arch}. Tried: {', '.join([p for p in LAPLACE_CANDIDATES(arch) if p])}")
    return None

import os, glob, math, torch
try:
    from PDF_learning_UQ import ConditionalRealNVP
    _FLOW_CLASS_OK = True
except Exception:
    _FLOW_CLASS_OK = False

def _flow_ckpt_present(path_or_dir: str) -> bool:
    if not _FLOW_CLASS_OK or not path_or_dir:
        return False
    if os.path.isfile(path_or_dir):
        # accept any .pth; prefer names containing 'flow'
        return path_or_dir.endswith(".pth")
    if os.path.isdir(path_or_dir):
        cands = glob.glob(os.path.join(path_or_dir, "*flow*.pth")) or \
                glob.glob(os.path.join(path_or_dir, "*.pth"))
        return len(cands) > 0
    return False

def _pick_ckpt_for_arch(arch: str, path_or_dir: str) -> str:
    if os.path.isfile(path_or_dir):  # user passed an exact file
        return path_or_dir
    if not os.path.isdir(path_or_dir):
        return path_or_dir  # may be empty; your loader can handle
    patt = {
        "flow":       "*flow*.pth",
        "multimodal": "*multimodal*.pth",
        "gaussian":   "*gaussian*.pth",
    }.get(arch, "*.pth")
    cands = glob.glob(os.path.join(path_or_dir, patt)) or \
            glob.glob(os.path.join(path_or_dir, "*.pth"))
    if not cands:
        return path_or_dir
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate plots with analytic Laplace uncertainty propagation. '
                   'Uses delta method for fast, accurate uncertainty quantification.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Uncertainty Methods:
  When Laplace models are available, uses analytic uncertainty propagation 
  via delta method for improved speed and accuracy. Falls back to Monte 
  Carlo sampling when Laplace models are not found.

Examples:
  # Plot with analytic uncertainty (recommended):
  python plotting_driver_UQ.py --arch gaussian --problem simplified_dis
  
  # Plot all architectures:
  python plotting_driver_UQ.py --arch all --latent_dim 512
  
  # Faster fallback for missing Laplace models:
  python plotting_driver_UQ.py --arch mlp --n_mc 50
        """
    )
    parser.add_argument('--arch', type=str, default='all',
                        help='Which architecture to plot: mlp, transformer, gaussian, multimodal, or all')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension of the model')
    parser.add_argument('--param_dim', type=int, default=4,
                        help='Parameter dimension (4 for simplified_dis, 6 for realistic_dis)')
    parser.add_argument('--problem', type=str, default='simplified_dis',
                        help='Problem type: simplified_dis or realistic_dis')
    parser.add_argument('--nmodes', type=int, default=2,
                        help='Number of modes for multimodal architecture')
    parser.add_argument('--n_mc', type=int, default=100,
                        help='Number of MC samples (used only when Laplace unavailable)')
    parser.add_argument("--num_samples", type=int, default=4000)
    parser.add_argument('--num_events', type=int, default=100000,
                        help='Number of events for simulation')
    parser.add_argument('--true_params', type=float, nargs='+', default=None,
                        help='True parameter values for plotting, e.g. --true_params 0.5 0.5 0.5 0.5')
    parser.add_argument("--cl_model", type=str, default="final_model.pth", help="Path to pre-trained PointNetPMA model (assumes same experiment directory).")
    parser.add_argument("--checkpoint", type=str, default="", 
                help="(Optional) Path to a flow/gaussian/multimodal .pth file or a directory containing them.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_dir = f"experiments/{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}"
    experiment_dir_pointnet = f"experiments/{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}"

    # Which architectures to plot?
    archs = []
    if args.arch == "all":
        archs = ["flow","mlp", "transformer"]
    else:
        archs = [args.arch]

    # True parameters for plotting
    if args.true_params is not None:
        true_params = torch.tensor(args.true_params, dtype=torch.float32)
    else:
        if args.problem == 'realistic_dis':
            true_params = torch.tensor([1.0, 0.1, 0.7, 3.0, 0.0, 0.0], dtype=torch.float32)
        else:
            true_params = torch.tensor([-7.10000000e-01, 3.48000000e+00, 1.34000000e+00, 2.33000000e+01], dtype=torch.float32)

    # Load PointNet once (shared across all heads)
    pointnet_model = reload_pointnet(experiment_dir_pointnet, args.latent_dim, device, args.cl_model, args.problem)
    param_ranges = [np.linspace(0.0, 5.0, 10) for _ in range(4)]
    params_grid = np.array(np.meshgrid(*param_ranges)).reshape(4, -1).T

    # Extract latents and parameters
    latents, thetas = extract_latents_from_data(pointnet_model, args, args.problem, device)
    plot_latents_umap(latents, thetas, color_mode='single', param_idx=0, method='umap', save_path=os.path.join(experiment_dir, "umap_single.png"))
    # plot_latents_umap(latents, params, color_mode='single', param_idx=0, method='umap')
    plot_latents_umap(latents, thetas, color_mode='mean', method='umap', save_path=os.path.join(experiment_dir, "umap_mean.png"))
    plot_latents_umap(latents, thetas, color_mode='pca', method='tsne', save_path=os.path.join(experiment_dir, "tsne_pca.png"))
    plot_latents_all_params(latents, thetas, method='umap', save_path=os.path.join(experiment_dir, "umap_all_params.png"))
    for arch in archs:
        print(f"\n=== Arch: {arch} ===")
        if arch == "flow":
            # look for a flow_model checkpoint in the experiment directory
            exp_dir = experiment_dir   # or however you’re already pointing to the directory
            ckpt = os.path.join(exp_dir, "flow_final.pth")
           
            if _FLOW_CLASS_OK and os.path.isfile(ckpt):
                plot_dir = os.path.join(experiment_dir, f"plots_{arch}")
                os.makedirs(plot_dir, exist_ok=True)
                print(f"[flow] Loading flow model from {ckpt}")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                flow = ConditionalRealNVP(
                    z_dim=args.latent_dim,
                    param_dim=args.param_dim,
                    n_layers=8, hidden=256, scale_clamp=5.0
                ).to(device)
                state = torch.load(ckpt, map_location=device)
                flow.load_state_dict(state)
                flow.eval()

                # your existing z-loading logic
                # z = torch.from_numpy(np.load(args.latent_npy)).float().to(device)
                z = torch.tensor(latents).to(device)
                if args.problem == 'mceg':
                    simulator = MCEGSimulator(torch.device('cpu'))
                elif args.problem == 'realistic_dis':
                    simulator = RealisticDIS(torch.device('cpu'))
                elif args.problem == 'simplified_dis':
                    simulator = SimplifiedDIS(torch.device('cpu'))
                # simulator = RealisticDIS(torch.device('cpu')) if problem == 'realistic_dis' else SimplifiedDIS(torch.device('cpu'))

                # Make sure params live on correct device
                true_params = true_params.to(device)
                # -------- sample events for the reconstructed histogram ----------
                with torch.no_grad():
                    xs = simulator.sample(true_params.detach().cpu(), args.num_events)  # expected shape (N, 2) = [x, Q2]
                xs_tensor = torch.tensor(xs, dtype=torch.float32).to(device)
                if args.problem != 'mceg':
                    xs_featurized = advanced_feature_engineering(xs_tensor)
                else:
                    xs_featurized = xs_tensor
                latent_samples = pointnet_model(xs_featurized.unsqueeze(0)).to(device)  # [N, Dz]
                z = latent_samples
                with torch.no_grad():
                    theta_samples = flow(z, n_samples=100)  # [B, S, D]

                agg = theta_samples.reshape(-1, theta_samples.shape[-1]).cpu().numpy()
                plot_1d_marginals(agg, plot_dir, "flow")
                plot_pairwise_2d(agg, plot_dir, "flow")
                plot_corner(agg, os.path.join(plot_dir, "flow_corner.png"),
                            labels=[f"$\\theta_{i}$" for i in range(min(args.param_dim,6))], max_dim=6)
                # Build posterior bands over PDF(x) using SimplifiedDIS
                if args.problem == 'simplified_dis':
                    plot_PDF_distribution_single_same_plot_from_theta_samples(
                        simulator=simulator,                         # SimplifiedDIS(torch.device('cpu')) per your code
                        theta_samples=theta_samples,                 # [B,S,D] is fine
                        true_params=true_params,                     # tensor [D] or None
                        device=device,
                        num_events_per_theta=max(2000, args.num_events // 2),  # tune for speed/quality
                        x_range=(0.0, 1.0),
                        bins=100,
                        quantiles=(5, 25, 50, 75, 95),
                        overlay_point_estimate=True,
                        point_estimate="mean",
                        save_path=os.path.join(plot_dir, "pdf_overlay_flow.png"),
                        title="Posterior PDF bands (SimplifiedDIS, flow θ-samples)"
                    )
        else:
            print(f"\n==== Plotting for architecture: {arch.upper()} ====")
            multimodal = (arch == "multimodal")
            model = reload_model(arch, args.latent_dim, args.param_dim, experiment_dir, device,
                                multimodal=multimodal, nmodes=args.nmodes)
            if model is None:
                print(f"Model for architecture '{arch}' not found, skipping.")
                continue
            plot_dir = os.path.join(experiment_dir, f"plots_{arch}")
            os.makedirs(plot_dir, exist_ok=True)
            # Attempt to load Laplace approximation
            laplace_model = load_laplace_if_available(experiment_dir, arch, device,args)

            if laplace_model is not None:
                # quick preview latent (batch=1)
                latent_preview = torch.zeros(1, args.latent_dim, device=device)
                with torch.no_grad():
                    param_mean, param_std = get_analytic_uncertainty(model, latent_preview, laplace_model=laplace_model)
                print("Output parameter mean (preview):", param_mean.squeeze(0).cpu().numpy())
                print("Output parameter std  (preview):", param_std.squeeze(0).cpu().numpy())
            # Run all plotting functions, passing laplace_model if available
            plot_params_distribution_single(
                model=model,
                pointnet_model=pointnet_model,
                true_params=true_params,
                device=device,
                n_mc=args.n_mc,
                laplace_model=laplace_model,
                compare_with_sbi=False,
                problem=args.problem,
                save_path=os.path.join(plot_dir, "params_distribution.png")
            )
            if args.problem != 'mceg':
                plot_PDF_distribution_single_same_plot(
                    model=model,
                    pointnet_model=pointnet_model,
                    true_params=true_params,
                    device=device,
                    n_mc=args.n_mc,
                    laplace_model=laplace_model,
                    problem=args.problem,
                    save_path=os.path.join(plot_dir, "pdf_overlay.png")
                )
            elif args.problem == 'mceg':
                plot_PDF_distribution_single_same_plot_mceg(
                    model=model,
                    pointnet_model=pointnet_model,
                    true_params=true_params,
                    device=device,
                    n_mc=args.n_mc,
                    laplace_model=laplace_model,
                    problem=args.problem,
                    save_dir=plot_dir
                )
            plot_PDF_distribution_single(
                model=model,
                pointnet_model=pointnet_model,
                true_params=true_params,
                device=device,
                n_mc=args.n_mc,
                laplace_model=laplace_model,
                problem=args.problem,
                Q2_slices=[0.5, 1.0, 1.5, 2.0, 10.0, 50.0, 200.0],
                save_dir=plot_dir
            )
            if args.problem == 'simplified_dis':
                plot_event_histogram_simplified_DIS(
                    model=model,
                    pointnet_model=pointnet_model,
                    true_params=true_params,
                    device=device,
                    n_mc=args.n_mc,
                    laplace_model=laplace_model,
                    num_events=args.num_events,
                    save_path=os.path.join(plot_dir, "event_histogram_simplified.png")
                )
            print(f"✅ Finished plotting for {arch} (plots in {plot_dir})")

if __name__ == "__main__":
    main()