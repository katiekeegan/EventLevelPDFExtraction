"""
PDF Parameter Inference Plotting Driver (Reload Per-Architecture Models)

This script reloads and generates plots for each model architecture:
MLP, Transformer, Gaussian, and Multimodal. It takes the same CLI arguments
(latent_dim, param_dim, problem, etc.) as your PDF_learning script, finds the
corresponding experiment directories and checkpoints, and saves plots for each.

Usage:
    python plotting_driver_UQ_reload.py --arch mlp --latent_dim 1024 --param_dim 4 --problem simplified_dis ...
    python plotting_driver_UQ_reload.py --arch gaussian --latent_dim 1024 --param_dim 4 --problem simplified_dis ...
    python plotting_driver_UQ_reload.py --arch multimodal --latent_dim 1024 --param_dim 4 --problem simplified_dis ...
    python plotting_driver_UQ_reload.py --arch transformer --latent_dim 1024 --param_dim 4 --problem simplified_dis ...
    python plotting_driver_UQ_reload.py --arch all --latent_dim 1024 --param_dim 4 --problem simplified_dis ...
"""

import torch
import os
import numpy as np
from PDF_learning import *
from simulator import *
from models import *
from plotting_UQ_utils import *
from PDF_learning_UQ import *

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
    elif arch == "gaussian":
        checkpoint_path = os.path.join(experiment_dir, "gaussian_head_final.pth")
        model = GaussianHead(latent_dim, param_dim).to(device)
    elif arch == "multimodal":
        checkpoint_path = os.path.join(experiment_dir, "multimodal_head_final.pth")
        model = GaussianHead(latent_dim, param_dim, multimodal=True, nmodes=nmodes).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found for {arch}: {checkpoint_path}")
        return None
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def reload_pointnet(experiment_dir, latent_dim, device):
    """
    Reloads PointNet model from the experiment directory.
    """
    pointnet_path = os.path.join(experiment_dir, "final_model.pth")
    # Dummy input for input_dim inference
    xs_dummy = np.random.randn(100, 2)
    xs_dummy_tensor = torch.tensor(xs_dummy, dtype=torch.float32)
    input_dim = advanced_feature_engineering(xs_dummy_tensor).shape[-1]
    pointnet_model = PointNetPMA(input_dim=input_dim, latent_dim=latent_dim).to(device)
    state_dict = torch.load(pointnet_path, map_location=device)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    pointnet_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    pointnet_model.eval()
    return pointnet_model

def load_laplace_if_available(experiment_dir, arch, model, device):
    """
    Attempt to load Laplace approximation for the specified model/arch.
    Returns laplace_model or None.
    """
    laplace_path = os.path.join(experiment_dir, f"laplace_{arch}.pt")
    if not os.path.exists(laplace_path):
        return None
    try:
        from laplace import Laplace
    except ImportError:
        print("Laplace library not installed, proceeding without Laplace uncertainty.")
        return None
    laplace_model = Laplace(model, 'regression', subset_of_weights='all')
    state_dict = torch.load(laplace_path, map_location=device)
    laplace_model.load_state_dict(state_dict)
    print(f"✔ Loaded Laplace for {arch} from {laplace_path}")
    return laplace_model

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Reload models for each architecture and generate plots. Arguments should match PDF_learning.py.',
    )
    parser.add_argument('--arch', type=str, default='all',
                        help='Which architecture to plot: mlp, transformer, gaussian, multimodal, or all')
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--param_dim', type=int, default=4)
    parser.add_argument('--problem', type=str, default='simplified_dis')
    parser.add_argument('--nmodes', type=int, default=2)
    parser.add_argument('--n_mc', type=int, default=100)
    parser.add_argument('--num_events', type=int, default=100000)
    parser.add_argument('--true_params', type=float, nargs='+', default=None,
                        help='True parameter values for plotting, e.g. --true_params 0.5 0.5 0.5 0.5')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_dir = f"experiments/{args.problem}_latent_{args.latent_dim}_ns_1000_ne_{args.num_events}"
    experiment_dir_pointnet = f"experiments/{args.problem}_latent{args.latent_dim}_ns_1000_ne_{args.num_events}"

    # Which architectures to plot?
    archs = []
    if args.arch == "all":
        archs = ["mlp", "transformer", "gaussian", "multimodal"]
    else:
        archs = [args.arch]

    # True parameters for plotting
    if args.true_params is not None:
        true_params = torch.tensor(args.true_params, dtype=torch.float32)
    else:
        if args.problem == 'realistic_dis':
            true_params = torch.tensor([1.0, 0.1, 0.7, 3.0, 0.0, 0.0], dtype=torch.float32)
        else:
            true_params = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)

    # Load PointNet once (shared across all heads)
    pointnet_model = reload_pointnet(experiment_dir_pointnet, args.latent_dim, device)

    for arch in archs:
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
        laplace_model = load_laplace_if_available(experiment_dir, arch, model, device)

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