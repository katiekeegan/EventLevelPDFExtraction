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
from cnf import *

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

def reload_model(arch, latent_dim, param_dim, experiment_dir, device, multimodal=False, nmodes=2):
    """
    Reloads model from checkpoint for the specified architecture.
    """
    if arch == "mlp":
        checkpoint_path = os.path.join(experiment_dir, "mlp_head_final.pth")
        model = MLPHead(latent_dim, param_dim).to(device)
    elif arch == "transformer":
        checkpoint_path = os.path.join(experiment_dir, "final_params_model.pth")
        model = TransformerHead(latent_dim, param_dim).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found for {arch}: {checkpoint_path}")
        return None
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
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
    pointnet_model = PointNetPMA(input_dim=input_dim, latent_dim=latent_dim, predict_theta=False).to(device)
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
    parser.add_argument('--n_bootstrap', type=int, default=100,
                    help='Number of bootstrap samples for uncertainty analysis')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_dir = f"experiments/{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}_parameter_predidction"
    experiment_dir_pointnet = f"experiments/{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}_parameter_predidction"

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
        elif args.problem == 'mceg':
            true_params = torch.tensor([-7.10000000e-01, 3.48000000e+00, 1.34000000e+00, 2.33000000e+01], dtype=torch.float32)
        elif args.problem == 'simplified_dis':
            true_params = torch.tensor([2.0, 1.2, 2.0, 1.2], dtype=torch.float32)
            

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
        # --- Fallback: Fit Laplace if not found ---
        if laplace_model is None and arch in ("mlp", "transformer"):
            print(f"ℹ️ No Laplace checkpoint found for {arch}, fitting Laplace from model weights...")

            fit_z = torch.tensor(latents[:500], dtype=torch.float32, device=device)
            test_out = model(fit_z)
            print("Model output shape:", test_out.shape)
            fit_theta = torch.tensor(thetas[:500], dtype=torch.float32, device=device)
            print(f"fit_z.shape: {fit_z.shape}, fit_theta.shape: {fit_theta.shape}")
            # For multi-output regression, Laplace expects targets of shape [N, D]
            # If fit_theta shape is [N], expand dims
            if fit_theta.ndim == 1:
                fit_theta = fit_theta.unsqueeze(1)
            # If fit_theta shape is [N, D], it's fine

            try:
                from laplace import Laplace
                laplace_model = Laplace(
                    model, 
                    likelihood='regression', 
                    subset_of_weights='last_layer', 
                    hessian_structure='kron'
                )
                import traceback
                try:
                    # Create a dataset of pairs
                    train_dataset = TensorDataset(fit_z, fit_theta)

                    # Create a DataLoader (mini-batch size of 64 is typical; adjust as needed)
                    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

                    laplace_model.fit(train_loader)
                    print(f"✓ Fitted Laplace approximation for {arch} using {fit_z.shape[0]} samples.")
                except Exception as e:
                    print("❌ Could not fit Laplace approximation:", e)
                    traceback.print_exc()
                    laplace_model = None
            except Exception as e:
                print("❌ Could not fit Laplace approximation:", e)
                laplace_model = None
        if laplace_model is not None:
            # quick preview latent (batch=1)
            latent_preview = torch.zeros(1, args.latent_dim, device=device)
            with torch.no_grad():
                param_mean, param_std = get_analytic_uncertainty(model, latent_preview, laplace_model=laplace_model)
            print("Output parameter mean (preview):", param_mean.squeeze(0).cpu().numpy())
            print("Output parameter std  (preview):", param_std.squeeze(0).cpu().numpy())

        # Add to plotting workflow  
        plot_bootstrap_PDF_distribution(
            model=model,
            pointnet_model=pointnet_model,
            true_params=true_params,
            device=device,
            num_events=args.num_events,
            n_bootstrap=args.n_bootstrap,
            problem=args.problem,
            save_dir=plot_dir
        )
        # Simplified DIS with combined uncertainty
        plot_combined_uncertainty_PDF_distribution(
            model=model,
            pointnet_model=pointnet_model,
            true_params=true_params,
            device=device,
            num_events=args.num_events,
            n_bootstrap=args.n_bootstrap,
            laplace_model=laplace_model,
            problem=args.problem,
            save_dir=plot_dir
        )
        scaling_results = plot_uncertainty_vs_events(
            model=model,
            pointnet_model=pointnet_model,
            true_params=true_params,
            device=device,
            event_counts=[1000, 5000, 10000, 50000, 100000],
            n_bootstrap=args.n_bootstrap,
            problem=args.problem,
            save_dir=plot_dir
        )
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
        generate_parameter_error_histogram(
            model=model,
            pointnet_model=pointnet_model,
            device=device,
            n_draws=100,
            n_events=args.num_events,
            problem=args.problem,
            save_path=plot_dir + 'param_errors.png'
        )

        # Enhanced event visualization with both views
        plot_event_histogram_simplified_DIS(
            model, pointnet_model, true_params, device,
            plot_type='both',  # Shows both scatter and 2D histogram
            save_path="events_enhanced.png"
        )

        # Publication-ready parameter distributions
        plot_params_distribution_single(
            model, pointnet_model, true_params, device,
            laplace_model=laplace_model,  # For analytic uncertainty
            save_path="param_distributions.png"
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