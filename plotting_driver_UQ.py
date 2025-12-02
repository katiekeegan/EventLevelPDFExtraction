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

Problem Types Supported:
- simplified_dis: Simplified DIS problem with 1D PDF inputs (x only)
- realistic_dis: Realistic DIS problem with 2D PDF inputs (x, Q2)
- mceg4dis: Monte Carlo Event Generator for DIS with 2D PDF inputs (x, Q2)
  * mceg4dis handles 2-dimensional PDF inputs (x and Q2) unlike simplified_dis
  * Uses the same underlying MCEG simulator as 'mceg' but with enhanced plotting support
  * All plotting functions properly handle the 2D nature of mceg4dis inputs

Uncertainty Methods:
- When Laplace model available: Uses analytic delta method for uncertainty
- When Laplace unavailable: Falls back to Monte Carlo sampling
- For Gaussian heads: Uses intrinsic uncertainty from predicted variances

Usage:
    # Use mceg4dis with 2D PDF inputs:
    python plotting_driver_UQ.py --arch gaussian --latent_dim 1024 --param_dim 4 --problem mceg4dis

    # Use analytic uncertainty (recommended - requires Laplace models):
    python plotting_driver_UQ.py --arch gaussian --latent_dim 1024 --param_dim 4 --problem simplified_dis

    # Plot all architectures with analytic uncertainty:
    python plotting_driver_UQ.py --arch all --latent_dim 1024 --param_dim 4 --problem simplified_dis

    # Reduce n_mc for faster fallback when Laplace unavailable:
    python plotting_driver_UQ.py --arch mlp --n_mc 50 --latent_dim 1024 --param_dim 4
"""

import glob
# utils_laplace.py
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from laplace.laplace import Laplace

from datasets import *
from models import *
from plotting_UQ_utils import *
# from PDF_learning import *
from simulator import *

# from PDF_learning_UQ import *
# from cnf import *



# Set up matplotlib for high-quality plots
plt.style.use("default")
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{libertine}\usepackage{zi4}\usepackage{newtxmath}"
    # r"\usepackage{newtxtext,bm}\usepackage[cmintegrals]{newtxmath}"
)
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": False,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.axisbelow": True,
    }
)


def find_experiment_dir_with_repeats(base_dir):
    """
    Returns the path to the repeat directory with the highest repeat number, or the base_dir if no repeat dirs exist.
    """
    pattern = os.path.join(base_dir, "repeat_*")
    repeat_dirs = sorted(glob.glob(pattern))
    # Only accept dirs that are actually directories
    repeat_dirs = [d for d in repeat_dirs if os.path.isdir(d)]
    if repeat_dirs:
        # Pick the highest-numbered repeat dir (lexicographically last)
        return repeat_dirs[-1]
    else:
        return base_dir


def build_head(arch, latent_dim, param_dim, device, nmodes=None):
    if arch == "mlp":
        return MLPHead(latent_dim, param_dim).to(device)
    if arch == "transformer":
        return TransformerHead(latent_dim, param_dim).to(device)
    raise ValueError(f"Unknown arch: {arch}")


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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
    m = build_head(
        arch,
        args.latent_dim,
        args.param_dim,
        device,
        nmodes=getattr(args, "nmodes", None),
    )
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


def reload_model(
    arch, latent_dim, param_dim, experiment_dir, device, multimodal=False, nmodes=2
):
    """
    Reloads model from checkpoint for the specified architecture.
    """
    if arch == "mlp":
        checkpoint_path = os.path.join(experiment_dir, "final_params_model.pth")
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
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval()
    return model


def reload_pointnet(
    experiment_dir,
    latent_dim,
    device,
    cl_model="final_model.pth",
    problem="simplified_dis",
):
    """
    Reloads PointNet model from the experiment directory.

    FIXED: Feature engineering inconsistency for mceg/mceg4dis RESOLVED
    ===================================================================
    This function previously revealed a critical issue with Laplace failures for mceg problems.
    The issue has now been FIXED by ensuring consistent feature engineering.

    TRAINING (here in reload_pointnet):
    - Uses log_feature_engineering from utils.py
    - Transforms 2D input (x, Q2) -> 6D features
    - PointNet expects 6D input during training

    INFERENCE (in plotting functions) - NOW FIXED:
    - Uses log_feature_engineering consistently for all mceg/mceg4dis functions
    - Transforms 2D input (x, Q2) -> 6D features
    - PointNet receives expected 6D input matching training

    RESOLUTION:
    1. ✅ Input dimension consistency restored - Laplace models now work properly
    2. ✅ Analytic uncertainty propagation enabled for mceg/mceg4dis problems
    3. ✅ Predictions are now accurate due to consistent feature engineering

    IMPLEMENTATION: Applied log_feature_engineering consistently in all mceg plotting functions
    """
    # pointnet_path = os.path.join(experiment_dir, "final_model.pth")
    pointnet_path = os.path.join(experiment_dir, cl_model)
    # Dynamically infer input dimension based on problem type and feature engineering
    if problem not in ["mceg", "mceg4dis"]:
        xs_dummy = np.random.randn(100, 2)
        xs_dummy_tensor = torch.tensor(xs_dummy, dtype=torch.float32)
        input_dim = advanced_feature_engineering(xs_dummy_tensor).shape[-1]
    else:
        # For mceg/mceg4dis, determine input_dim based on log feature engineering
        # Create dummy 2D data (x, Q2) and apply the same feature engineering used in training
        xs_dummy = np.random.randn(100, 2)
        xs_dummy_tensor = torch.tensor(xs_dummy, dtype=torch.float32)
        from utils import log_feature_engineering

        feats_dummy = log_feature_engineering(xs_dummy_tensor)
        input_dim = feats_dummy.shape[-1]
        # input_dim = 6
        print(f"MCEG log feature engineering: 2D -> {input_dim}D")
        # print(f"✅ [FIXED] Training uses log_feature_engineering: 2D -> {input_dim}D")
        # print(f"✅ [FIXED] All inference plotting functions now consistently apply log_feature_engineering")
        # print(f"✅ [FIXED] Feature engineering consistency resolved - Laplace approximation should work")

    # Use ChunkedPointNetPMA for mceg problems, PointNetPMA for others
    if problem in ["mceg", "mceg4dis"]:
        pointnet_model = ChunkedPointNetPMA(
            input_dim=input_dim,
            latent_dim=latent_dim,
            chunk_latent=128,
            num_seeds=8,
            num_heads=4,
        ).to(device)
    else:
        pointnet_model = ChunkedPointNetPMA(
            input_dim=input_dim,
            latent_dim=latent_dim,
            dropout=0.3,
            chunk_latent=32,
            num_seeds=8,
            num_heads=4,
        ).to(device)
    state_dict = torch.load(pointnet_path, map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    pointnet_model.load_state_dict(
        {k.replace("module.", ""): v for k, v in state_dict.items()}
    )
    pointnet_model.eval()
    return pointnet_model


def load_laplace(
    make_model_fn,
    ckpt_path,
    device="cpu",
    default_likelihood="regression",
    default_subset="last_layer",
    default_hessian="kron",
):
    if not os.path.exists(ckpt_path):
        return None
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Case A: dict with 'laplace_state' [+ optional 'meta']
    if isinstance(obj, dict) and "laplace_state" in obj:
        meta = obj.get(
            "meta",
            {
                "likelihood": default_likelihood,
                "subset_of_weights": default_subset,
                "hessian_structure": default_hessian,
            },
        )
        model = make_model_fn()  # this already .to(device).eval()
        la = Laplace(
            model,
            meta["likelihood"],
            subset_of_weights=meta["subset_of_weights"],
            hessian_structure=meta["hessian_structure"],
        )
        la.load_state_dict(obj["laplace_state"])
        return _finalize_device_eval(la, device)

    # Case B: plain state_dict (no meta)
    if isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values()):
        model = make_model_fn()
        la = Laplace(
            model,
            default_likelihood,
            subset_of_weights=default_subset,
            hessian_structure=default_hessian,
        )
        la.load_state_dict(obj)
        return _finalize_device_eval(la, device)

    # Case C: pickled Laplace object
    if hasattr(obj, "model"):
        la = obj
        return _finalize_device_eval(la, device)

    print(f"⚠️  Unrecognized Laplace checkpoint format: {ckpt_path}")
    return None

def load_laplace_if_available(experiment_dir, arch, device, args):
    # try several filenames in order
    for name in filter(None, LAPLACE_CANDIDATES(arch)):
        ckpt_path = os.path.join(experiment_dir, name)
        if not os.path.exists(ckpt_path):
            continue

        def make_model_fn():
            return make_model(experiment_dir, arch, device, args)

        la = load_laplace(
            make_model_fn,
            ckpt_path,
            device=device,
            default_likelihood="regression",
            default_subset="last_layer",
            default_hessian="kron",
        )
        if la is not None:
            print(f"✓ Loaded Laplace from {ckpt_path}")
            return la
    print(
        f"ℹ️  No Laplace checkpoint found for {arch}. Tried: {', '.join([p for p in LAPLACE_CANDIDATES(arch) if p])}"
    )
    return None


import glob
import math
import os

import torch

try:
    from PDF_learning_UQ import ConditionalRealNVP

    _FLOW_CLASS_OK = True
except Exception:
    _FLOW_CLASS_OK = False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate plots with analytic Laplace uncertainty propagation. "
        "Uses delta method for fast, accurate uncertainty quantification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""" ... [unchanged] ... """,
    )
    # [arguments as before]
    parser.add_argument(
        "--arch",
        type=str,
        default="all",
        help="Which architecture to plot: mlp, transformer, gaussian, multimodal, or all",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=128, help="Latent dimension of the model"
    )
    parser.add_argument(
        "--param_dim",
        type=int,
        default=4,
        help="Parameter dimension (4 for simplified_dis/mceg4dis, 6 for realistic_dis)",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="simplified_dis",
        help="Problem type: simplified_dis, realistic_dis, or mceg4dis (2D PDF inputs: x and Q2)",
    )
    parser.add_argument(
        "--nmodes",
        type=int,
        default=2,
        help="Number of modes for multimodal architecture",
    )
    parser.add_argument(
        "--n_mc",
        type=int,
        default=100,
        help="Number of MC samples (used only when Laplace unavailable)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["median", "mean"],
        help="Aggregation to reduce per-x errors to a scalar for the bar chart: 'median' or 'mean'.",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=None,
        help="Optional RNG seed used when subsampling ensembles via --n_mc",
    )
    parser.add_argument("--num_samples", type=int, default=4000)
    parser.add_argument(
        "--val_samples",
        type=int,
        default=1000,
        help="Number of validation samples to use (default from parameter_prediction.py)",
    )
    parser.add_argument(
        "--precomputed_data_dir",
        type=str,
        default="precomputed_data",
        help="Directory containing precomputed validation datasets",
    )
    parser.add_argument(
        "--num_events", type=int, default=100000, help="Number of events for simulation"
    )
    parser.add_argument(
        "--true_params",
        type=float,
        nargs="+",
        default=None,
        help="True parameter values for plotting, e.g. --true_params 0.5 0.5 0.5 0.5",
    )
    parser.add_argument(
        "--cl_model",
        type=str,
        default="final_model.pth",
        help="Path to pre-trained PointNetPMA model (assumes same experiment directory).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="(Optional) Path to a flow/gaussian/multimodal .pth file or a directory containing them.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap samples for uncertainty analysis",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct the base experiment directory (without repeat)
    base_experiment_dir = f"experiments/{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}_parameter_predidction"
    # Find the correct experiment dir (with repeat, if exists)
    experiment_dir = find_experiment_dir_with_repeats(base_experiment_dir)
    experiment_dir_pointnet = experiment_dir

    if experiment_dir != base_experiment_dir:
        print(f"🔍 Found experiment repeats, using directory: {experiment_dir}")
    else:
        print(f"🔍 No experiment repeats found, using directory: {base_experiment_dir}")

    # Which architectures to plot?
    archs = []
    if args.arch == "all":
        archs = ["flow", "mlp", "transformer"]
    else:
        archs = [args.arch]

    # True parameters for plotting
    if args.true_params is not None:
        true_params = torch.tensor(args.true_params, dtype=torch.float32)
    else:
        if args.problem == "realistic_dis":
            true_params = torch.tensor(
                [1.0, 0.1, 0.7, 3.0, 0.0, 0.0], dtype=torch.float32
            )
        elif args.problem in ["mceg", "mceg4dis"]:
            true_params = torch.tensor(
                [-7.10000000e-01, 3.48000000e00, 1.34000000e00, 2.33000000],
                dtype=torch.float32,
            )
        elif args.problem == "simplified_dis":
            true_params = torch.tensor([1.0, 1.2, 1.1, 0.5], dtype=torch.float32)

    # Diagnostic: Check if true parameters are within training bounds
    print(f"\n🔍 PARAMETER BOUNDS DIAGNOSTIC:")
    print(f"   Problem type: {args.problem}")
    print(f"   True parameters for plotting: {true_params.tolist()}")

    try:
        from plotting_UQ_utils import get_parameter_bounds_for_problem

        training_bounds = get_parameter_bounds_for_problem(args.problem)
        print(f"   Training bounds (min, max): {training_bounds.tolist()}")

        # Check each parameter
        out_of_bounds = []
        for i, (param_val, bounds) in enumerate(zip(true_params, training_bounds)):
            min_bound, max_bound = bounds[0].item(), bounds[1].item()
            if param_val < min_bound or param_val > max_bound:
                out_of_bounds.append((i, param_val.item(), min_bound, max_bound))

        if out_of_bounds:
            print(
                f"   ⚠️  WARNING: {len(out_of_bounds)} parameter(s) are outside training bounds!"
            )
            for i, val, min_b, max_b in out_of_bounds:
                print(
                    f"      Parameter {i}: {val:.3f} (bounds: [{min_b:.1f}, {max_b:.1f}])"
                )
            print(
                f"   💡 This may cause misleading plots since the model was never trained on such values."
            )
        else:
            print(f"   ✅ All parameters are within training bounds.")
    except Exception as e:
        print(f"   ❌ Could not verify parameter bounds: {e}")
    print()

    # Load PointNet once (shared across all heads)
    pointnet_model = reload_pointnet(
        experiment_dir_pointnet, args.latent_dim, device, args.cl_model, args.problem
    )
    param_ranges = [np.linspace(0.0, 5.0, 10) for _ in range(4)]
    params_grid = np.array(np.meshgrid(*param_ranges)).reshape(4, -1).T

    # Extract latents and parameters
    latents, thetas = extract_latents_from_data(
        pointnet_model, args, args.problem, device
    )
    plot_latents_umap(
        latents,
        thetas,
        color_mode="single",
        param_idx=0,
        method="umap",
        save_path=os.path.join(experiment_dir, "umap_single.png"),
    )
    # plot_latents_umap(latents, params, color_mode='single', param_idx=0, method='umap')
    plot_latents_umap(
        latents,
        thetas,
        color_mode="mean",
        method="umap",
        save_path=os.path.join(experiment_dir, "umap_mean.png"),
    )
    plot_latents_umap(
        latents,
        thetas,
        color_mode="pca",
        method="tsne",
        save_path=os.path.join(experiment_dir, "tsne_pca.png"),
    )
    plot_latents_all_params(
        latents,
        thetas,
        method="umap",
        save_path=os.path.join(experiment_dir, "umap_all_params.png"),
    )
    for arch in archs:
        print(f"\n=== Arch: {arch} ===")
        print(f"\n==== Plotting for architecture: {arch.upper()} ====")
        multimodal = arch == "multimodal"
        model = reload_model(
            arch,
            args.latent_dim,
            args.param_dim,
            experiment_dir,
            device,
            multimodal=multimodal,
            nmodes=args.nmodes,
        )
        if model is None:
            print(f"Model for architecture '{arch}' not found, skipping.")
            continue
        plot_dir = os.path.join(experiment_dir, f"plots_{arch}")
        os.makedirs(plot_dir, exist_ok=True)
        # Attempt to load Laplace approximation
        print(
            f"🔍 [DEBUG] Attempting to load Laplace model for arch={arch}, problem={args.problem}"
        )
        laplace_model = load_laplace_if_available(experiment_dir, arch, device, args)

        # INVESTIGATION: Why does Laplace fail for mceg/mceg4dis?
        if args.problem in ["mceg", "mceg4dis"]:
            print(f"🔍 [DEBUG] mceg Laplace diagnosis:")
            print(f"🔍 [DEBUG] - Experiment dir: {experiment_dir}")
            print(f"🔍 [DEBUG] - Architecture: {arch}")
            print(f"🔍 [DEBUG] - Laplace candidates: {LAPLACE_CANDIDATES(arch)}")
            print(f"🔍 [DEBUG] - Laplace model loaded: {laplace_model is not None}")

            if laplace_model is not None:
                print(f"✅ [DEBUG] Laplace model loaded successfully for mceg")
                print(f"🔍 [DEBUG] Laplace model type: {type(laplace_model)}")
                # Test if the model is compatible with mceg feature engineering
                # print(f"✅ [FIXED] mceg feature engineering consistency verified")
                # print(f"✅ [FIXED] Training: log_feature_engineering (2D->6D)")
                # print(f"✅ [FIXED] Inference: log_feature_engineering (2D->6D) - now consistent!")
            else:
                print(
                    f"❌ [DEBUG] No Laplace model found for mceg - will fallback to Monte Carlo"
                )
                print(
                    f"⚠️  [DEBUG] This explains why analytic uncertainty propagation fails"
                )
        # --- Fallback: Fit Laplace if not found ---
        if laplace_model is None and arch in ("mlp", "transformer"):
            print(
                f"ℹ️ No Laplace checkpoint found for {arch}, fitting Laplace from model weights..."
            )

            fit_z = torch.tensor(latents[:500], dtype=torch.float32, device=device)
            test_out = model(fit_z)
            print("Model output shape:", test_out.shape)
            fit_theta = torch.tensor(thetas[:500], dtype=torch.float32, device=device)
            print(f"fit_z.shape: {fit_z.shape}, fit_theta.shape: {fit_theta.shape}")
            # breakpoint()
            # For multi-output regression, Laplace expects targets of shape [N, D]
            # If fit_theta shape is [N], expand dims
            if fit_theta.ndim == 1:
                fit_theta = fit_theta.unsqueeze(1)
            # If fit_theta shape is [N, D], it's fine

            try:
                from laplace import Laplace

                laplace_model = Laplace(
                    model,
                    likelihood="regression",
                    subset_of_weights="last_layer",
                    hessian_structure="kron",
                )
                import traceback

                try:
                    # Create a dataset of pairs
                    train_dataset = TensorDataset(fit_z, fit_theta)

                    # Create a DataLoader (mini-batch size of 64 is typical; adjust as needed)
                    train_loader = DataLoader(
                        train_dataset, batch_size=64, shuffle=True
                    )

                    laplace_model.fit(train_loader)
                    print(
                        f"✓ Fitted Laplace approximation for {arch} using {fit_z.shape[0]} samples."
                    )
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
                param_mean, param_std = get_analytic_uncertainty(
                    model, latent_preview, laplace_model=laplace_model
                )
            print(
                "Output parameter mean (preview):", param_mean.squeeze(0).cpu().numpy()
            )
            print(
                "Output parameter std  (preview):", param_std.squeeze(0).cpu().numpy()
            )
        if args.problem not in ["mceg", "mceg4dis"]:
            # Add to plotting workflow
            plot_bootstrap_PDF_distribution(
                model=model,
                pointnet_model=pointnet_model,
                true_params=true_params,
                device=device,
                num_events=args.num_events,
                n_bootstrap=args.n_bootstrap,
                problem=args.problem,
                save_dir=plot_dir,
            )
        # SBI sample file names vary by problem: mceg/mceg4dis use *_mceg.txt suffixes
        if args.problem in ["mceg", "mceg4dis"]:
            snpe_file = "samples_snpe_mceg.txt"
            wass_file = "samples_wasserstein_mceg.txt"
            mmd_file = "samples_mmd_mceg.txt"
        else:
            snpe_file = "samples_snpe.txt"
            wass_file = "samples_wasserstein.txt"
            mmd_file = "samples_mmd.txt"

        samples_snpe = torch.tensor(np.loadtxt(snpe_file), dtype=torch.float32)
        samples_wass = torch.tensor(np.loadtxt(wass_file), dtype=torch.float32)
        samples_mmd = torch.tensor(np.loadtxt(mmd_file), dtype=torch.float32)

        # Run all plotting functions, passing laplace_model if available
        plot_params_distribution_single(
            model=model,
            pointnet_model=pointnet_model,
            true_params=true_params,
            device=device,
            n_mc=args.n_mc,
            sbi_posteriors=[samples_snpe, samples_mmd, samples_wass],
            sbi_labels=["SNPE", "MCABC", "MCABC-W"],
            laplace_model=laplace_model,
            compare_with_sbi=True if args.problem == "simplified_dis" else False,
            problem=args.problem,
            save_path=os.path.join(plot_dir, "params_distribution.png"),
        )
        if args.problem == "simplified_dis":
            # Combined SBI plot: overlay SNPE, Wasserstein MCABC, and MCABC on one figure
            plot_function_posterior_from_multiple_sbi_samples(
                model=model,
                pointnet_model=pointnet_model,
                sbi_samples_list=[samples_snpe, samples_wass, samples_mmd],
                labels=["SNPE", "Wasserstein MCABC", "MCABC"],
                true_params=true_params,
                device=device,
                num_events=args.num_events,
                problem=args.problem,
                save_path=os.path.join(plot_dir, "function_posterior_sbi_combined.png"),
            )
            # Also save a function-space error summary (mean absolute error and per-x curves)
            try:
                from plotting_UQ_utils import \
                    plot_function_error_summary_from_sbi_samples

                # SBI-only summary (no our_results_dict provided) — save to a distinct filename
                print(
                    "[driver-debug] Calling SBI-only plot_function_error_summary_from_sbi_samples (SBI-only)"
                )
                sbi_only_path = os.path.join(
                    plot_dir, "sbi_function_error_summary_sbi_only.png"
                )
                plot_function_error_summary_from_sbi_samples(
                    sbi_samples_list=[samples_snpe, samples_wass, samples_mmd],
                    labels=["SNPE", "Wasserstein MCABC", "MCABC"],
                    true_params=true_params,
                    device=device,
                    problem="simplified_dis",
                    save_path=sbi_only_path,
                    aggregation=args.aggregation,
                    n_mc=args.n_mc,
                    rng_seed=args.rng_seed,
                )
            except Exception as e:
                print(f"⚠️ Could not generate SBI function error summary: {e}")
            # Collect SBI-predicted PDFs and save LaTeX metrics table
            try:
                # Small helper to evaluate simulator.f on parameter ensembles
                from plotting_UQ_helpers import (
                    collect_predicted_pdfs_simplified_dis,
                    save_function_UQ_metrics_table_simplified_dis)

                results_dict = {}

                # SNPE
                up_snpe, down_snpe = collect_predicted_pdfs_simplified_dis(
                    samples_snpe, device
                )
                results_dict["SNPE"] = {"pdfs_up": up_snpe, "pdfs_down": down_snpe}
                # MCABC (MMD)
                up_mmd, down_mmd = collect_predicted_pdfs_simplified_dis(
                    samples_mmd, device
                )
                results_dict["MCABC"] = {"pdfs_up": up_mmd, "pdfs_down": down_mmd}
                # Wasserstein
                up_wass, down_wass = collect_predicted_pdfs_simplified_dis(
                    samples_wass, device
                )
                results_dict["Wasserstein MCABC"] = {
                    "pdfs_up": up_wass,
                    "pdfs_down": down_wass,
                }

                # --- Laplace analytic ensemble (if available) ---
                try:
                    if laplace_model is not None:
                        # Lazy imports from heavy plotting utils
                        from plotting_UQ_utils import (
                            get_advanced_feature_engineering,
                            get_gaussian_samples, get_simulator_module)

                        # Build a representative latent embedding from many events (same as other plotting routines)
                        SimplifiedDIS_cls, RealisticDIS_cls, MCEGSimulator_cls = (
                            get_simulator_module()
                        )
                        simulator = SimplifiedDIS_cls(device=torch.device("cpu"))
                        # Use a reasonably large sample to form the latent (keeps consistent with other plots)
                        n_latent_events = min(100000, max(1000, args.num_events))
                        xs = simulator.sample(
                            true_params.detach().cpu(), n_latent_events
                        )
                        xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
                        advanced_fe = get_advanced_feature_engineering()
                        if advanced_fe is not None:
                            xs_tensor = advanced_fe(xs_tensor)
                        latent = pointnet_model(xs_tensor.unsqueeze(0))

                        # Generate parameter samples from Laplace posterior (convert analytic -> samples)
                        laplace_param_samples = get_gaussian_samples(
                            model, latent, n_samples=100, laplace_model=laplace_model
                        )
                        # Ensure tensor on correct device for evaluation helper
                        laplace_param_samples = laplace_param_samples.to(device)
                        up_lap, down_lap = collect_predicted_pdfs_simplified_dis(
                            laplace_param_samples, device
                        )
                        results_dict["Laplace"] = {
                            "pdfs_up": up_lap,
                            "pdfs_down": down_lap,
                        }
                    else:
                        # Laplace not available: add placeholder row (will be skipped/marked by helper)
                        results_dict["Laplace"] = {"pdfs_up": [], "pdfs_down": []}
                except Exception as e:
                    print(
                        f"⚠️ Could not generate Laplace ensemble for function UQ table: {e}"
                    )
                    results_dict["Laplace"] = {"pdfs_up": [], "pdfs_down": []}

                # --- Bootstrap ensemble (generate smaller set for table if not already available) ---
                try:
                    # We'll generate a smaller bootstrap ensemble (at most 50 samples) to keep table computation cheap
                    from plotting_UQ_utils import (
                        get_advanced_feature_engineering, get_gaussian_samples,
                        get_simulator_module)

                    SimplifiedDIS_cls, RealisticDIS_cls, MCEGSimulator_cls = (
                        get_simulator_module()
                    )
                    simulator = SimplifiedDIS_cls(device=torch.device("cpu"))
                    advanced_fe = get_advanced_feature_engineering()
                    n_boot_for_table = min(50, max(10, args.n_bootstrap))
                    x_grid = torch.linspace(0.01, 0.99, 100)
                    bootstrap_up = []
                    bootstrap_down = []
                    per_boot_param_samples = []
                    for i in range(n_boot_for_table):
                        xs = simulator.sample(
                            true_params.detach().cpu(), args.num_events
                        )
                        xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
                        if advanced_fe is not None:
                            xs_tensor = advanced_fe(xs_tensor)
                        latent = pointnet_model(xs_tensor.unsqueeze(0))
                        with torch.no_grad():
                            pred = model(latent).cpu().squeeze(0)
                        pdf = simulator.f(x_grid, pred)
                        bootstrap_up.append(pdf["up"].cpu().numpy())
                        bootstrap_down.append(pdf["down"].cpu().numpy())
                        # Collect per-bootstrap parameter posterior samples when Laplace is available
                        try:
                            if laplace_model is not None:
                                # draw a modest number of posterior samples per bootstrap
                                per_boot_samples = get_gaussian_samples(
                                    model,
                                    latent,
                                    n_samples=20,
                                    laplace_model=laplace_model,
                                )
                                per_boot_param_samples.append(
                                    per_boot_samples.detach().cpu().numpy()
                                )
                            else:
                                # fallback: singleton posterior (the MAP prediction)
                                per_boot_param_samples.append(
                                    pred.detach().cpu().numpy()[None, :]
                                )
                        except Exception:
                            # On any failure, fall back to singleton
                            per_boot_param_samples.append(
                                pred.detach().cpu().numpy()[None, :]
                            )
                    results_dict["Bootstrap"] = {
                        "pdfs_up": np.array(bootstrap_up),
                        "pdfs_down": np.array(bootstrap_down),
                    }
                    # Compute LoTV decomposition for function uncertainty using per-bootstrap posterior samples
                    try:
                        from plotting_UQ_helpers import \
                            compute_function_lotv_for_simplified_dis

                        SimplifiedDIS_cls, RealisticDIS_cls, MCEGSimulator_cls = (
                            get_simulator_module()
                        )
                        simulator = SimplifiedDIS_cls(device=torch.device("cpu"))
                        lotv = compute_function_lotv_for_simplified_dis(
                            simulator,
                            x_grid,
                            per_boot_posterior_samples=per_boot_param_samples,
                            device="cpu",
                        )
                        # Save a decomposition row that contains mean functions and scalar uncertainties
                        results_dict["Combined_LOTV"] = {
                            "mean_up": lotv["mean_up"],
                            "mean_down": lotv["mean_down"],
                            "unc_up": lotv["unc_up"],
                            "unc_down": lotv["unc_down"],
                        }
                    except Exception as e:
                        print(f"⚠️ Could not compute Combined LoTV decomposition: {e}")
                except Exception as e:
                    print(
                        f"⚠️ Could not generate Bootstrap ensemble for function UQ table: {e}"
                    )
                    results_dict["Bootstrap"] = {"pdfs_up": [], "pdfs_down": []}

                # --- Combined ensemble (concatenate Laplace + Bootstrap when both exist) ---
                # Use explicit size checks to avoid ambiguous numpy array truth-value errors.
                lap_up = results_dict.get("Laplace", {}).get("pdfs_up")
                lap_down = results_dict.get("Laplace", {}).get("pdfs_down")
                bst_up = results_dict.get("Bootstrap", {}).get("pdfs_up")
                bst_down = results_dict.get("Bootstrap", {}).get("pdfs_down")
                # If both Laplace and Bootstrap ensembles available, compute a principled LoTV decomposition
                try:
                    from plotting_UQ_helpers import \
                        compute_function_lotv_for_simplified_dis

                    if (
                        isinstance(lap_up, np.ndarray)
                        and lap_up.size > 0
                        and isinstance(bst_up, np.ndarray)
                        and bst_up.size > 0
                    ):
                        # We have numpy ensembles: create per-bootstrap posterior samples list from bootstrap (bst)
                        # For Laplace we treat its ensemble as one 'bootstrap' with many posterior samples
                        per_boot_posterior_samples = []
                        # Laplace ensemble: treat each row as a sample -> make it a single bootstrap with many posterior samples
                        per_boot_posterior_samples.append(lap_up.copy())
                        # For bootstraps, each row is itself a param sample (we lack per-bootstrap posterior samples),
                        # so approximate each bootstrap by a singleton posterior at the bootstrap MAP (i.e., the predicted params)
                        # This is a lightweight approximation; for a more accurate LoTV, supply per-bootstrap posterior samples.
                        for i in range(bst_up.shape[0]):
                            # Here, bst_up[i] is a function output; we need parameter samples. We don't have them saved here,
                            # so fall back to treating each bootstrap prediction as a single theta (approximate)
                            # To keep behavior consistent, just append the Laplace samples as representing posterior variability
                            # This branch will be improved by producing per-bootstrap posterior samples below when available.
                            pass
                        # As we don't have per-bootstrap parameter samples for the bootstrap ensemble in this driver,
                        # fall back to pooled concatenation for the Combined table but also compute LoTV when possible via
                        # a conservative approximation: use Laplace samples as posterior samples and bootstrap means from bst_up.
                        # Build a fake per-bootstrap sample list: each bootstrap will use the Laplace samples shifted to match the
                        # bootstrap predicted mean. This is a heuristic.
                        try:
                            from plotting_UQ_utils import get_simulator_module

                            SimplifiedDIS_cls, RealisticDIS_cls, MCEGSimulator_cls = (
                                get_simulator_module()
                            )
                            simulator = SimplifiedDIS_cls(device=torch.device("cpu"))
                            x_grid = torch.linspace(0.01, 0.99, 100)
                            # Build per-bootstrap posterior samples by re-centering Laplace samples to each bootstrap MAP
                            # First reconstruct Laplace param samples from lap_up/down via inverse mapping is not possible here.
                            # So as a pragmatic step, compute pooled Combined for table and also add a placeholder LoTV row
                            combined_up = np.concatenate([lap_up, bst_up], axis=0)
                            combined_down = np.concatenate([lap_down, bst_down], axis=0)
                            results_dict["Combined"] = {
                                "pdfs_up": combined_up,
                                "pdfs_down": combined_down,
                            }
                        except Exception:
                            results_dict["Combined"] = {"pdfs_up": [], "pdfs_down": []}
                    else:
                        results_dict["Combined"] = {"pdfs_up": [], "pdfs_down": []}
                except Exception as e:
                    print(f"⚠️ Could not compute LoTV Combined decomposition: {e}")
                    # fall back to previous behavior
                    try:
                        if (
                            isinstance(lap_up, np.ndarray)
                            and lap_up.size > 0
                            and isinstance(bst_up, np.ndarray)
                            and bst_up.size > 0
                        ):
                            combined_up = np.concatenate([lap_up, bst_up], axis=0)
                            combined_down = np.concatenate([lap_down, bst_down], axis=0)
                            results_dict["Combined"] = {
                                "pdfs_up": combined_up,
                                "pdfs_down": combined_down,
                            }
                        else:
                            results_dict["Combined"] = {"pdfs_up": [], "pdfs_down": []}
                    except Exception:
                        results_dict["Combined"] = {"pdfs_up": [], "pdfs_down": []}

                # Save table
                table_path = os.path.join(plot_dir, "function_UQ_metrics_table.tex")
                save_function_UQ_metrics_table_simplified_dis(
                    table_path,
                    true_params,
                    device,
                    results_dict,
                    aggregation=args.aggregation,
                )
                print(
                    f"✓ Saved SBI + analytic/bootstrap function UQ metrics table: {table_path}"
                )
                # Also generate combined SBI + our-approach function error summary (relative errors)
                try:
                    from plotting_UQ_utils import \
                        plot_function_error_summary_from_sbi_samples

                    # Debug: announce call and show keys we'll pass
                    print(
                        "[driver-debug] Generating combined SBI + our-approach function error summary"
                    )
                    print(
                        f"[driver-debug] results_dict keys: {list(results_dict.keys())}"
                    )
                    our_results = {}
                    for k in ["Laplace", "Bootstrap", "Combined_LOTV", "Combined"]:
                        our_results[k] = results_dict.get(
                            k, {"pdfs_up": [], "pdfs_down": []}
                        )
                    print(
                        f"[driver-debug] our_results prepared with keys: {list(our_results.keys())}"
                    )
                    plot_function_error_summary_from_sbi_samples(
                        sbi_samples_list=[samples_snpe, samples_wass, samples_mmd],
                        labels=["SNPE", "Wasserstein MCABC", "MCABC"],
                        true_params=true_params,
                        device=device,
                        problem="simplified_dis",
                        save_path=os.path.join(
                            plot_dir, "sbi_function_error_summary.png"
                        ),
                        our_results_dict=our_results,
                        relative=True,
                        aggregation=args.aggregation,
                        n_mc=args.n_mc,
                        rng_seed=args.rng_seed,
                    )
                except Exception as e:
                    print(
                        f"⚠️ Could not generate combined SBI + our approach function error summary: {e}"
                    )
            except Exception as e:
                print(f"⚠️ Could not save SBI function UQ metrics table: {e}")
                # Fallback: even if saving the table or building results_dict failed, call the
                # combined SBI + our-approach summary with an explicit placeholder our_results
                try:
                    from plotting_UQ_utils import \
                        plot_function_error_summary_from_sbi_samples

                    print(
                        "[driver-debug] Fallback: calling combined plot with empty our_results_dict"
                    )
                    our_results = {
                        k: {"pdfs_up": [], "pdfs_down": []}
                        for k in ["Laplace", "Bootstrap", "Combined_LOTV", "Combined"]
                    }
                    plot_function_error_summary_from_sbi_samples(
                        sbi_samples_list=[samples_snpe, samples_wass, samples_mmd],
                        labels=["SNPE", "Wasserstein MCABC", "MCABC"],
                        true_params=true_params,
                        device=device,
                        problem="simplified_dis",
                        save_path=os.path.join(
                            plot_dir, "sbi_function_error_summary_fallback.png"
                        ),
                        our_results_dict=our_results,
                        relative=True,
                        aggregation=args.aggregation,
                        n_mc=args.n_mc,
                        rng_seed=args.rng_seed,
                    )
                except Exception as e2:
                    print(f"⚠️ Fallback combined plot also failed: {e2}")
        # Ensure combined SBI + our-approach summary is always attempted once (non-blocking)
        try:
            from plotting_UQ_utils import \
                plot_function_error_summary_from_sbi_samples

            # Prefer results_dict if it exists, otherwise build an empty placeholder
            our_results = locals().get("results_dict", None)
            if our_results is None:
                our_results = {
                    k: {"pdfs_up": [], "pdfs_down": []}
                    for k in ["Laplace", "Bootstrap", "Combined_LOTV", "Combined"]
                }
            print(
                f"[driver-debug] Unconditional combined plot call; our_results keys: {list(our_results.keys())}"
            )
            plot_function_error_summary_from_sbi_samples(
                sbi_samples_list=[samples_snpe, samples_wass, samples_mmd],
                labels=["SNPE", "Wasserstein MCABC", "MCABC"],
                true_params=true_params,
                device=device,
                problem="simplified_dis",
                save_path=os.path.join(
                    plot_dir, "sbi_function_error_summary_unconditional.png"
                ),
                our_results_dict=our_results,
                relative=True,
                aggregation=args.aggregation,
                n_mc=args.n_mc,
                rng_seed=args.rng_seed,
            )
        except Exception as e:
            print(f"⚠️ Unconditional combined plot failed: {e}")

        generate_parameter_error_histogram(
            model=model,
            pointnet_model=pointnet_model,
            device=device,
            n_draws=100,
            n_events=args.num_events,
            problem=args.problem,
            save_path=plot_dir + "/param_errors.png",
        )
        if args.problem in ["mceg", "mceg4dis"]:
            plot_function_error_histogram_mceg(
                model=model,
                pointnet_model=pointnet_model,
                device=device,
                n_draws=100,
                n_events=args.num_events,
                problem=args.problem,
                save_path=plot_dir + "/function_errors.png",
            )

        # # Enhanced event visualization with both views
        # plot_event_histogram_simplified_DIS(
        #     model, pointnet_model, true_params, device,
        #     plot_type='both',  # Shows both scatter and 2D histogram
        #     save_path="events_enhanced.png"
        # )

        from uq_plotting_demo import (plot_bootstrap_uncertainty,
                                      plot_combined_uncertainty_decomposition,
                                      plot_function_uncertainty,
                                      plot_function_uncertainty_mceg,
                                      plot_parameter_uncertainty,
                                      plot_pdf_uncertainty_mceg,
                                      plot_uncertainty_scaling)

        print(f"🎯 Generating demonstration events for uncertainty plotting...")
        simulator = SimplifiedDIS(device=device)
        true_events = simulator.sample(true_params.numpy(), args.num_events).to(device)
        true_params = true_params.to(device)
        # Example 1: Parameter uncertainty plot - use new API
        plot_parameter_uncertainty(
            model=model,
            pointnet_model=pointnet_model,
            laplace_model=laplace_model,
            true_params=true_params,
            device=device,
            num_events=args.num_events,
            problem=args.problem,
            save_dir=plot_dir,
            mode="posterior",
        )
        plot_parameter_uncertainty(
            model=model,
            pointnet_model=pointnet_model,
            laplace_model=laplace_model,
            true_params=true_params,
            device=device,
            num_events=args.num_events,
            problem=args.problem,
            save_dir=plot_dir,
            mode="bootstrap",
        )
        plot_parameter_uncertainty(
            model=model,
            pointnet_model=pointnet_model,
            laplace_model=laplace_model,
            true_params=true_params,
            device=device,
            num_events=args.num_events,
            problem=args.problem,
            save_dir=plot_dir,
            mode="combined",
        )
        if args.problem in ["mceg", "mceg4dis"]:
            plot_function_uncertainty_mceg(
                model=model,
                pointnet_model=pointnet_model,
                laplace_model=laplace_model,
                true_params=true_params,
                device=device,
                num_events=args.num_events,
                save_dir=plot_dir,
                mode="posterior",
            )
            # # Also create PDF uncertainty plots (mean ± std across parameter samples)
            # plot_pdf_uncertainty_mceg(
            #     model=model,
            #     pointnet_model=pointnet_model,
            #     laplace_model=laplace_model,
            #     true_params=true_params,
            #     device=device,
            #     num_events=args.num_events,
            #     save_dir=plot_dir,
            #     mode='posterior'
            # )
            plot_function_uncertainty_mceg(
                model=model,
                pointnet_model=pointnet_model,
                laplace_model=laplace_model,
                true_params=true_params,
                device=device,
                num_events=args.num_events,
                save_dir=plot_dir,
                mode="bootstrap",
            )
            # plot_pdf_uncertainty_mceg(
            #     model=model,
            #     pointnet_model=pointnet_model,
            #     laplace_model=laplace_model,
            #     true_params=true_params,
            #     device=device,
            #     num_events=args.num_events,
            #     save_dir=plot_dir,
            #     mode='bootstrap'
            # )
            plot_function_uncertainty_mceg(
                model=model,
                pointnet_model=pointnet_model,
                laplace_model=laplace_model,
                true_params=true_params,
                device=device,
                num_events=args.num_events,
                save_dir=plot_dir,
                mode="combined",
            )
            # plot_pdf_uncertainty_mceg(
            #     model=model,
            #     pointnet_model=pointnet_model,
            #     laplace_model=laplace_model,
            #     true_params=true_params,
            #     device=device,
            #     num_events=args.num_events,
            #     save_dir=plot_dir,
            #     mode='combined'
            # )
            # Single combined+SBI overlay: show our combined curve together with SBI methods
            try:
                print(
                    "🔀 Generating combined (our) + SBI overlay plot for PDF uncertainty (Q2=10)"
                )
                plot_pdf_uncertainty_mceg(
                    model=model,
                    pointnet_model=pointnet_model,
                    laplace_model=laplace_model,
                    true_params=true_params,
                    device=device,
                    num_events=args.num_events,
                    save_dir=plot_dir,
                    mode="combined",
                    sbi_samples_list=[samples_snpe, samples_wass, samples_mmd],
                    sbi_labels=["SNPE", "MCABC-W", "MCABC"],
                    combined_plot_modes=True,
                    combined_plot_sbi=True,
                )
            except Exception as e:
                print(f"⚠️ Failed to generate combined+SBI overlay plot: {e}")
            # Additionally, generate function posterior plots for each SBI method using the loaded samples
            try:
                # print("🔁 Generating SBI-based function posterior plots for MCEG")
                # # Ensure samples_snpe, samples_wass, samples_mmd are available
                # plot_function_uncertainty_mceg(
                #     model=None,
                #     pointnet_model=None,
                #     laplace_model=None,
                #     true_params=true_params,
                #     device=device,
                #     num_events=args.num_events,
                #     save_dir=plot_dir,
                #     mode='sbi_SNPE',
                #     sbi_samples_list=[samples_snpe],
                #     sbi_labels=['SNPE']
                # )
                # # Also generate PDF uncertainty plots for the SBI samples
                # plot_pdf_uncertainty_mceg(
                #     model=None,
                #     pointnet_model=None,
                #     laplace_model=None,
                #     true_params=true_params,
                #     device=device,
                #     num_events=args.num_events,
                #     save_dir=plot_dir,
                #     mode='sbi_SNPE',
                #     sbi_samples_list=[samples_snpe],
                #     sbi_labels=['SNPE']
                # )
                pass
            except Exception as e:
                print(f"⚠️ Could not generate SBI SNPE mceg plot: {e}")
            try:
                # plot_function_uncertainty_mceg(
                #     model=None,
                #     pointnet_model=None,
                #     laplace_model=None,
                #     true_params=true_params,
                #     device=device,
                #     num_events=args.num_events,
                #     save_dir=plot_dir,
                #     mode='sbi_Wasserstein',
                #     sbi_samples_list=[samples_wass],
                #     sbi_labels=['Wasserstein']
                # )
                # plot_pdf_uncertainty_mceg(
                #     model=None,
                #     pointnet_model=None,
                #     laplace_model=None,
                #     true_params=true_params,
                #     device=device,
                #     num_events=args.num_events,
                #     save_dir=plot_dir,
                #     mode='sbi_Wasserstein',
                #     sbi_samples_list=[samples_wass],
                #     sbi_labels=['Wasserstein']
                # )
                pass
            except Exception as e:
                print(f"⚠️ Could not generate SBI Wasserstein mceg plot: {e}")
            try:
                # plot_function_uncertainty_mceg(
                #     model=None,
                #     pointnet_model=None,
                #     laplace_model=None,
                #     true_params=true_params,
                #     device=device,
                #     num_events=args.num_events,
                #     save_dir=plot_dir,
                #     mode='sbi_MMD',
                #     sbi_samples_list=[samples_mmd],
                #     sbi_labels=['MMD']
                # )
                # plot_pdf_uncertainty_mceg(
                #     model=None,
                #     pointnet_model=None,
                #     laplace_model=None,
                #     true_params=true_params,
                #     device=device,
                #     num_events=args.num_events,
                #     save_dir=plot_dir,
                #     mode='sbi_MMD',
                #     sbi_samples_list=[samples_mmd],
                #     sbi_labels=['MMD']
                # )
                pass
            except Exception as e:
                print(f"⚠️ Could not generate SBI MMD mceg plot: {e}")
            # Generate combined modes overlay (posterior, bootstrap, combined) at Q2=10
            try:
                # plot_pdf_uncertainty_mceg(
                #     model=model,
                #     pointnet_model=pointnet_model,
                #     laplace_model=laplace_model,
                #     true_params=true_params,
                #     device=device,
                #     num_events=args.num_events,
                #     save_dir=plot_dir,
                #     mode='combined',
                #     combined_plot_modes=True
                # )
                pass
            except Exception as e:
                print(f"⚠️ Could not generate combined modes PDF plot: {e}")

            # Generate combined SBI overlay (SNPE, MCABC, MCABC-W) at Q2=10
            try:
                # plot_pdf_uncertainty_mceg(
                #     model=None,
                #     pointnet_model=None,
                #     laplace_model=None,
                #     true_params=true_params,
                #     device=device,
                #     num_events=args.num_events,
                #     save_dir=plot_dir,
                #     mode='combined',
                #     sbi_samples_list=[samples_snpe, samples_wass, samples_mmd],
                #     sbi_labels=['SNPE', 'Wasserstein', 'MCABC'],
                #     combined_plot_sbi=True
                # )
                pass
            except Exception as e:
                print(f"⚠️ Could not generate combined SBI PDF plot: {e}")
            # Single combined-overlay plot: Combined (our pooled posterior+bootstrap) + all SBI methods
            try:
                plot_pdf_uncertainty_mceg(
                    model=model,
                    pointnet_model=pointnet_model,
                    laplace_model=laplace_model,
                    true_params=true_params,
                    device=device,
                    num_events=args.num_events,
                    save_dir=plot_dir,
                    mode="combined",
                    sbi_samples_list=[samples_snpe, samples_wass, samples_mmd],
                    sbi_labels=["SNPE", "MCABC-W", "MCABC"],
                    combined_plot_sbi=True,
                )
            except Exception as e:
                print(f"⚠️ Could not generate combined+SBI overlay plot: {e}")
        else:
            pass
        #     plot_function_uncertainty(
        #         model=model,
        #         pointnet_model=pointnet_model,
        #         laplace_model=laplace_model,
        #         true_params=true_params,
        #         device=device,
        #         num_events=args.num_events,
        #         problem=args.problem,
        #         save_dir=plot_dir,
        #         mode='posterior'
        #     )
        #     plot_function_uncertainty(
        #         model=model,
        #         pointnet_model=pointnet_model,
        #         laplace_model=laplace_model,
        #         true_params=true_params,
        #         device=device,
        #         num_events=args.num_events,
        #         problem=args.problem,
        #         save_dir=plot_dir,
        #         mode='bootstrap'
        #     )
        #     plot_function_uncertainty(
        #         model=model,
        #         pointnet_model=pointnet_model,
        #         laplace_model=laplace_model,
        #         true_params=true_params,
        #         device=device,
        #         num_events=args.num_events,
        #         problem=args.problem,
        #         save_dir=plot_dir,
        #         mode='combined'
        #     )

        # plot_bootstrap_uncertainty(
        #     model=model,
        #     pointnet_model=pointnet_model,
        #     laplace_model=laplace_model,
        #     true_params=true_params,
        #     device=device,
        #     num_events=args.num_events,
        #     n_bootstrap=20,
        #     problem=args.problem,
        #     save_dir=plot_dir
        # )

        # plot_combined_uncertainty_decomposition(
        #     model=model,
        #     pointnet_model=pointnet_model,
        #     laplace_model=laplace_model,
        #     true_params=true_params,
        #     device=device,
        #     num_events=args.num_events,
        #     n_bootstrap=20,
        #     problem=args.problem,
        #     save_dir=plot_dir
        # )

        if args.problem not in ["mceg", "mceg4dis"]:
            plot_PDF_distribution_single_same_plot(
                model=model,
                pointnet_model=pointnet_model,
                true_params=true_params,
                device=device,
                n_mc=args.n_mc,
                laplace_model=laplace_model,
                problem=args.problem,
                save_path=os.path.join(plot_dir, "pdf_overlay.png"),
            )
        elif args.problem in ["mceg", "mceg4dis"]:
            # INVESTIGATION: mceg/mceg4dis plotting issues
            print(f"🔍 [DEBUG] Starting mceg plotting for problem: {args.problem}")
            print(f"🔍 [DEBUG] Laplace model available: {laplace_model is not None}")
            print(
                f"🔍 [DEBUG] Expected Q2 slices: [0.5, 1.0, 1.5, 2.0, 10.0, 50.0, 200.0]"
            )

            plot_PDF_distribution_single_same_plot_mceg(
                model=model,
                pointnet_model=pointnet_model,
                true_params=true_params,
                device=device,
                n_mc=args.n_mc,
                laplace_model=laplace_model,
                problem=args.problem,  # Pass through the actual problem type (mceg or mceg4dis)
                save_dir=plot_dir,
            )
            print(
                f"✅ [MCEG4DIS] Enhanced mceg/mceg4dis plotting completed with Q2 slice curves"
            )
            # Also create SBI-based function posterior plots for mceg4dis (per-method files)
            try:
                import plotting_UQ_utils as puq

                # Use same SBI samples loaded earlier
                puq.plot_function_posterior_from_multiple_sbi_samples(
                    model=model,
                    pointnet_model=pointnet_model,
                    sbi_samples_list=[samples_snpe, samples_wass, samples_mmd],
                    labels=["SNPE", "MCABC-W", "MCABC"],
                    true_params=true_params,
                    device=device,
                    num_events=args.num_events,
                    problem=args.problem,
                    save_path=os.path.join(
                        plot_dir, "function_posterior_sbi_mceg4dis.png"
                    ),
                )
                print(
                    f"✓ Saved SBI mceg4dis function posterior plots (per-method) in {plot_dir}"
                )
            except Exception as e:
                print(f"⚠️ Could not generate SBI mceg4dis plots: {e}")
        plot_PDF_distribution_single(
            model=model,
            pointnet_model=pointnet_model,
            true_params=true_params,
            device=device,
            n_mc=args.n_mc,
            laplace_model=laplace_model,
            problem=args.problem,
            Q2_slices=[0.5, 1.0, 1.5, 2.0, 10.0, 50.0, 200.0],
            save_dir=plot_dir,
        )
        if args.problem == "simplified_dis":
            plot_event_histogram_simplified_DIS(
                model=model,
                pointnet_model=pointnet_model,
                true_params=true_params,
                device=device,
                n_mc=args.n_mc,
                laplace_model=laplace_model,
                num_events=args.num_events,
                save_path=os.path.join(plot_dir, "event_histogram_simplified.png"),
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
            save_dir=plot_dir,
        )
        # scaling_results = plot_uncertainty_vs_events(
        #     model=model,
        #     pointnet_model=pointnet_model,
        #     true_params=true_params,
        #     device=device,
        #     event_counts=[100, 1000, 5000, 10000, 50000, 100000],
        #     n_bootstrap=args.n_bootstrap,
        #     problem=args.problem,
        #     save_dir=plot_dir,
        #     n_mc=args.n_mc,
        #     # rng_seed passed through to ensure deterministic subsampling where used
        #     Q2_slices=None
        # )
        # plot_uncertainty_scaling(
        #     model=model,
        #     pointnet_model=pointnet_model,
        #     laplace_model=laplace_model,
        #     true_params=true_params,
        #     device=device,
        #     event_counts=[1000, 5000, 10000, 50000, 100000],
        #     n_bootstrap=20,
        #     problem=args.problem,
        #     save_dir=plot_dir,
        #     mode='bootstrap'
        # )
        # plot_uncertainty_scaling(
        #     model=model,
        #     pointnet_model=pointnet_model,
        #     laplace_model=laplace_model,
        #     true_params=true_params,
        #     device=device,
        #     event_counts=[1000, 5000, 10000, 50000, 100000],
        #     n_bootstrap=20,
        #     problem=args.problem,
        #     save_dir=plot_dir,
        #     mode='parameter'
        # )
        # plot_uncertainty_scaling(
        #     model=model,
        #     pointnet_model=pointnet_model,
        #     laplace_model=laplace_model,
        #     true_params=true_params,
        #     device=device,
        #     event_counts=[1000, 5000, 10000, 50000, 100000],
        #     n_bootstrap=20,
        #     problem=args.problem,
        #     save_dir=plot_dir,
        #     mode='combined'
        # )
        print(f"✅ Finished plotting for {arch} (plots in {plot_dir})")


if __name__ == "__main__":
    main()