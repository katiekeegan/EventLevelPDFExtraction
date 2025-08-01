"""
PDF Parameter Inference Plotting Driver

This script generates plots for training results and model predictions with support
for both MSE and NLL loss modes. 

Key Features:
- CLI support with --nll-loss flag for appropriate labeling
- Dynamic plot labels that reflect the loss type (MSE vs NLL)
- Model architecture mismatch detection and warnings
- Backward compatibility (defaults to MSE mode)

Usage:
    python plotting_driver.py                # MSE mode (default)
    python plotting_driver.py --nll-loss     # NLL mode
    python plotting_driver.py --help         # Show help

The script automatically detects if a model was trained with a different loss mode
than the one specified for plotting and issues warnings to ensure accurate labeling.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from torch.utils.data import DataLoader
from PDF_learning import *
from simulator import *
from models import *
from torch.distributions import *
import os
from simulator import SimplifiedDIS, up, down, advanced_feature_engineering, RealisticDIS
import scipy
plt.style.use("seaborn-v0_8-muted")  # Or 'default', or 'seaborn-whitegrid'
mpl.rcParams.update({
    # "font.family": "serif",           # For LaTeX-like look
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,             # Set to True if using full LaTeX (requires setup)
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
})
def plot_loss_curves(loss_dir='.', save_path='loss_plot.png', show_plot=True, nll_loss=False):
    """
    Plots training loss components from .npy files in the given directory.

    Args:
        loss_dir (str): Path to the directory containing the loss .npy files.
        save_path (str): Path to save the output plot image.
        show_plot (bool): Whether to display the plot interactively.
        nll_loss (bool): Whether the regression loss is NLL (True) or MSE (False).
                        Affects plot labels and titles.
    """
    # Build full paths
    # total_path = os.path.join(loss_dir, 'loss_total.npy')
    contrastive_path = os.path.join(loss_dir, 'loss_contrastive.npy')
    regression_path = os.path.join(loss_dir, 'loss_regression.npy')

    # Load data
    contrastive_loss = np.load(contrastive_path)
    regression_loss = np.load(regression_path)
    # breakpoint()

    # Plotting
    plt.figure(figsize=(8, 6))
    epochs = np.arange(1, len(contrastive_loss) + 1)

    # Determine loss type labels
    loss_type = "NLL" if nll_loss else "MSE"
    regression_label = f'Regression Loss ({loss_type}, scaled)'
    title = f'Training Loss Components Over Epochs ({loss_type} Regression)'
    
    # plt.plot(epochs, total_loss, label='Total Loss', linewidth=2)
    plt.plot(epochs, contrastive_loss, label='Contrastive Loss', linewidth=2)
    plt.plot(epochs, regression_loss, label=regression_label, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    if show_plot:
        plt.show()
    plt.close()

    total_path = os.path.join(loss_dir, 'loss_total.npy')
    total_loss = np.load(total_path)
    # Plotting
    plt.figure(figsize=(8, 6))
    epochs = np.arange(1, len(total_loss) + 1)

    total_title = f'Training Loss Over Epochs ({loss_type} Regression)'
    
    plt.plot(epochs, total_loss, label='Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(total_title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('loss_PDF_learning.png', dpi=300)
    if show_plot:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    epochs = np.arange(1, len(total_loss) + 1)

    plt.plot(epochs, total_loss, label='Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Training Loss Over Epochs (Log Scale, {loss_type} Regression)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('log_loss_PDF_learning.png', dpi=300)
    if show_plot:
        plt.show()
    plt.close()

def compute_chisq_statistic(true_function, predicted_function):
    """
    Computes the Chi-square statistic between the true function and the predicted function.
    
    Args:
        true_function: The true function values (observed data).
        predicted_function: The predicted function values (expected data).
        
    Returns:
        The Chi-square statistic.
    """
    # Compute Chi-square statistic
    chisq = np.sum(((true_function - predicted_function) ** 2) / (predicted_function + 1e-10))  # Adding small value to avoid division by zero
    return chisq

def evaluate_over_n_parameters(model, pointnet_model, n=100, num_events=100000, device=None, problem='simplified_dis'):
    """
    Evaluate the model over n true parameter samples and compute errors and chi-squared statistics.
    Args:
        model: The trained model to evaluate.
        pointnet_model: The PointNet model for feature extraction.
        n (int): Number of samples to evaluate.
        num_events (int): Number of events to simulate for each sample.
        device: Device to run the evaluation on (CPU or GPU).
    """     
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if problem == 'realistic_dis':
        simulator = RealisticDIS(torch.device('cpu'))
        param_dim = 6
    elif problem == 'mceg':
        simulator = MCEGSimulator(torch.device('cpu'))
        param_dim = 4
    elif problem == 'simplified_dis':
        simulator = SimplifiedDIS(torch.device('cpu'))
        param_dim = 4

    all_errors = []
    chi2_up = []
    chi2_down = []

    for i in range(n):
        # === 1. Sample true parameters from a known range (e.g., Uniform[1, 10]) ===
        true_params = torch.FloatTensor(param_dim).uniform_(0.0, 5.0).to(device)

        # === 2. Generate data from simulator ===
        xs = simulator.sample(true_params.detach().cpu(), num_events).to(device)
        xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
        xs_feat = advanced_feature_engineering(xs_tensor)
        latent = pointnet_model(xs_feat.unsqueeze(0))

        # === 3. Sample predicted parameters ===
        # breakpoint()
        samples = sample_with_mc_dropout(model, torch.tensor(latent).unsqueeze(0), 1000).squeeze()  # (1000, 4)
        predicted = torch.mean(samples, dim=0)

        # === 4. Compute relative error ===
        error = torch.abs(predicted - true_params) / (true_params + 1e-8)
        all_errors.append(error.detach().cpu())

        # === 5. Evaluate Chi-squared statistics ===
        x_grid = torch.linspace(0.01, 0.99, 1000).to(device)  # Avoid 0/1
        pred_up = torch.mean(torch.stack([up(x_grid.detach().cpu(), p.detach().cpu()) for p in samples]), dim=0)
        pred_down = torch.mean(torch.stack([down(x_grid.detach().cpu(), p.detach().cpu()) for p in samples]), dim=0)
        true_up = up(x_grid, true_params)
        true_down = down(x_grid, true_params)

        chi2_up.append(compute_chisq_statistic(true_up.detach().cpu().numpy(), pred_up.detach().cpu().numpy()))
        chi2_down.append(compute_chisq_statistic(true_down.detach().cpu().numpy(), pred_down.detach().cpu().numpy()))

    all_errors = torch.stack(all_errors).numpy()

    # === 6. Plot parameter-wise error distribution ===
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        axes[i].hist(all_errors[:, i], bins=50, alpha=0.7)
        axes[i].set_title(f'Parameter {i+1} Relative Error')
        axes[i].set_xlabel('|θ_pred - θ_true| / |θ_true|')
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("error_distributions.png")
    plt.show()

    # === 7. Print and plot chi-squared ===
    chi2_up = np.array(chi2_up)
    chi2_down = np.array(chi2_down)
    print(f"Median Chi² up: {np.median(chi2_up):.4f} ± {chi2_up.std():.4f}")
    print(f"Median Chi² down: {np.median(chi2_down):.4f} ± {chi2_down.std():.4f}")

    chi2_up_clip = np.percentile(chi2_up, 99)
    chi2_down_clip = np.percentile(chi2_down, 99)
    
    plt.figure(figsize=(10, 5))
    plt.hist(chi2_up[chi2_up < chi2_up_clip], bins=50, alpha=0.6, label='Chi² Up')
    plt.hist(chi2_down[chi2_down < chi2_down_clip], bins=50, alpha=0.6, label='Chi² Down')
    plt.legend()
    plt.title("Chi² Statistic Distribution (Clipped at 99th percentile)")
    plt.xlabel("Chi²")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("chisq_distributions_clipped.png")
    plt.show()

def enable_dropout(model):
    """Enable dropout layers during evaluation"""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def sample_with_mc_dropout(model, latent_embedding, n_samples=100):
    model.train()  # Enable dropout
    samples = []
    for _ in range(n_samples):
        output = model(latent_embedding)
        # Handle NLL mode (output is a tuple: (means, log_vars))
        if isinstance(output, tuple):
            means, log_vars = output
            samples.append(means)  # Only append means for parameter sampling
        else:
            samples.append(output)
    return torch.stack(samples)

def plot_event_histogram_simplified_DIS(model, pointnet_model, true_params, device, n_mc=100, num_events=100000, save_path="event_histogram_simplified.png"):

    model.eval()
    pointnet_model.eval()
    # Choose simulator
    simulator = SimplifiedDIS(torch.device('cpu'))
    # Simulate true events
    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), num_events).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # Monte Carlo sampling
    samples = sample_with_mc_dropout(model, latent_embedding, n_samples=n_mc).squeeze()

    if torch.any(torch.isnan(samples)):
        print("NaNs detected in samples!")
        return

    mode_params = torch.median(samples, dim=0).values  # median for robustness

    if torch.any(torch.isnan(mode_params)):
        print("NaNs detected in mode parameters!")
        return

    generated_events = simulator.sample(mode_params.detach().cpu(), num_events).to(device)
    true_events_np = xs.detach().cpu().numpy()
    generated_events_np = generated_events.detach().cpu().numpy()

    if np.any(np.isnan(generated_events_np)) or np.any(np.isnan(true_events_np)):
        print("NaNs detected in the events!")
        return

    print(f"Shape of true_events_np: {true_events_np.shape}")
    print(f"Shape of generated_events_np: {generated_events_np.shape}")

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].scatter(true_events_np[:, 0], true_events_np[:, 1], color='turquoise', alpha=0.2)
    axs[0].set_title(r"$\Xi_{\theta^{*}}$")
    axs[0].set_xlabel(r"$x_{u} \sim u(x|\theta^{*})$")
    axs[0].set_ylabel(r"$x_{d} \sim d(x|\theta^{*})$")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")

    axs[1].scatter(generated_events_np[:, 0], generated_events_np[:, 1], color='darkorange', alpha=0.2)
    axs[1].set_title(r"$\Xi_{\hat{\theta}}$")
    axs[1].set_xlabel(r"$x_{u} \sim u(x|\hat{\theta})$")
    axs[1].set_ylabel(r"$x_{d} \sim d(x|\hat{\theta})$")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_event_scatter_3d(model, pointnet_model, true_params, device, n_mc=100, num_events=100000, problem='realistic_dis', save_path="event_scatter_3d.png"):
    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()
    pointnet_model.eval()

    # Choose simulator
    simulator = RealisticDIS(smear=False) if problem == 'realistic_dis' else SimplifiedDIS(torch.device('cpu'))

    # Simulate true events
    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), num_events).to(device)
    xs_tensor = advanced_feature_engineering(xs.clone()).to(device)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # Monte Carlo dropout posterior samples
    samples = sample_with_mc_dropout(model, latent_embedding, n_samples=n_mc).squeeze()
    if torch.any(torch.isnan(samples)):
        print("NaNs detected in samples!")
        return
    mode_params = torch.median(samples, dim=0).values
    generated_events = simulator.sample(mode_params.detach().cpu(), num_events).to(device)

    # Unpack and convert to NumPy
    x_true, Q2_true, F2_true = xs[:, 0].detach().cpu().numpy(), xs[:, 1].detach().cpu().numpy(), xs[:, 2].detach().cpu().numpy()
    x_gen, Q2_gen, F2_gen = generated_events[:, 0].detach().cpu().numpy(), generated_events[:, 1].detach().cpu().numpy(), generated_events[:, 2].detach().cpu().numpy()

    # Scatterplot helper
    def make_scatter(ax, x, Q2, F2, title):
        sc = ax.scatter(x, Q2, c=F2, cmap='viridis', norm=plt.LogNorm(), s=3, alpha=0.5, edgecolor='none')
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$Q^2$")
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, label="$F_2$")

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    make_scatter(axs[0], x_true, Q2_true, F2_true, r"True Events: $\Xi_{\theta^*}$")
    make_scatter(axs[1], x_gen, Q2_gen, F2_gen, r"Generated Events: $\Xi_{\hat{\theta}}$")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_params_distribution_single(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,
    compare_with_sbi=False,
    sbi_posteriors=None,  # list of tensors
    sbi_labels=None,      # list of strings
    save_path="Dist.png",
    problem = 'simplified_dis'  # 'simplified_dis' or 'realistic_dis'
):
    model.eval()
    pointnet_model.eval()
    if problem == 'realistic_dis':
        simulator = RealisticDIS(torch.device('cpu'))
    elif problem == 'simplified_dis':
        simulator = SimplifiedDIS(torch.device('cpu'))

    true_params = true_params.to(device)

    # Simulate data + feature engineering
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # MC Dropout samples
    samples = sample_with_mc_dropout(model, latent_embedding, n_samples=n_mc).squeeze()

    # Combine all posterior samples for consistent x-limits
    all_samples = [samples.detach().cpu()]
    if compare_with_sbi and sbi_posteriors is not None:
        all_samples.extend([s.detach().cpu() for s in sbi_posteriors])
    
    n_params = true_params.size(0)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

    if n_params == 1:
        axes = [axes]

    colors = ['skyblue', 'orange', 'green', 'purple', 'gray']
    if problem == 'simplified_dis':
        param_names = [r'$a_u$', r'$b_u$', r'$a_d$', r'$b_d$']
    elif problem == 'realistic_dis':
        param_names = [r'$\log A_0$', r'$\delta$', r'$a$', r'$b$', r'$c$', r'$d$']

    for i in range(n_params):
        # Compute global min/max across all samples for this parameter
        param_vals = [s[:, i].detach().numpy() for s in all_samples]
        xmin = min([v.min() for v in param_vals])
        xmax = max([v.max() for v in param_vals])
        padding = 0.05 * (xmax - xmin)
        xmin -= padding
        xmax += padding

        # MC Dropout
        axes[i].hist(samples[:, i].detach().cpu().numpy(), bins=20, alpha=0.6, density=True, color=colors[0], label='MC Samples')

        # SBI posteriors
        if compare_with_sbi and sbi_posteriors is not None and sbi_labels is not None:
            for j, sbi_samples in enumerate(sbi_posteriors):
                label = sbi_labels[j] if j < len(sbi_labels) else f"SBI {j}"
                axes[i].hist(
                    sbi_samples[:, i].detach().cpu().numpy(),
                    bins=20, alpha=0.4, density=True,
                    color=colors[(j + 1) % len(colors)],
                    label=label
                )

        # True value
        axes[i].axvline(true_params[i].item(), color='red', linestyle='dashed', label='True Value')
        axes[i].set_xlim(xmin, xmax)
        axes[i].set_title(f'{param_names[i]}')
        if i == 0: 
            axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_sbi_posteriors_only(
    true_params,
    sbi_posteriors,   # list of tensors
    sbi_labels=None,  # list of strings
    save_path="SBI_Dist.png"
):
    n_params = true_params.size(0)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

    if n_params == 1:
        axes = [axes]

    colors = ['orange', 'green', 'purple', 'gray', 'cyan']

    # Compute global x-limits across all posteriors
    all_samples = [s.detach().cpu() for s in sbi_posteriors]
    
    for i in range(n_params):
        param_vals = [s[:, i].numpy() for s in all_samples]
        xmin = min([v.min() for v in param_vals])
        xmax = max([v.max() for v in param_vals])
        padding = 0.05 * (xmax - xmin)
        xmin -= padding
        xmax += padding

        for j, sbi_samples in enumerate(all_samples):
            label = sbi_labels[j] if sbi_labels and j < len(sbi_labels) else f"SBI {j}"
            axes[i].hist(
                sbi_samples[:, i].numpy(),
                bins=20,
                alpha=0.6,
                density=True,
                color=colors[j % len(colors)],
                label=label
            )

        axes[i].axvline(true_params[i].item(), color='red', linestyle='dashed', label='True Value')
        axes[i].set_xlim(xmin, xmax)
        axes[i].set_title(f'Param {i+1}')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_PDF_distribution_single(model, pointnet_model, true_params, device, n_mc=100,
                                 problem='simplified_dis', Q2_slices=None,
                                 save_dir=None, save_path="pdf_distribution.png"):
    model.eval()
    pointnet_model.eval()
    
    simulator = RealisticDIS(torch.device('cpu')) if problem == 'realistic_dis' else SimplifiedDIS(torch.device('cpu'))

    # Feature extraction
    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # Monte Carlo sampling
    samples = sample_with_mc_dropout(model, latent_embedding, n_samples=n_mc).squeeze()

    if problem == 'simplified_dis':
        x_vals = torch.linspace(0, 1, 500).to(device)

        for fn_name, fn_label, color in [("up", "u", "royalblue"), ("down", "d", "darkorange")]:
            fn_vals_all = []
            for i in range(n_mc):
                simulator.init(samples[i])
                fn = getattr(simulator, fn_name)
                fn_vals_all.append(fn(x_vals).unsqueeze(0))

            fn_stack = torch.cat(fn_vals_all, dim=0)
            median_vals = fn_stack.median(dim=0).values
            lower = torch.quantile(fn_stack, 0.25, dim=0)
            upper = torch.quantile(fn_stack, 0.75, dim=0)

            simulator.init(true_params.squeeze())
            true_vals = getattr(simulator, fn_name)(x_vals)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(x_vals.detach().cpu(), true_vals.detach().cpu(), label=fr"True ${fn_label}(x|\theta^*)$", color=color, linewidth=2)
            ax.plot(x_vals.detach().cpu(), median_vals.detach().cpu(), linestyle='--', label=fr"Median ${fn_label}(x|\hat{{\theta}})$", color="crimson", linewidth=2)
            ax.fill_between(x_vals.detach().cpu(), lower.detach().cpu(), upper.detach().cpu(), color="crimson", alpha=0.3, label="IQR")

            ax.set_xlabel(r"$x$")
            ax.set_ylabel(fr"${fn_label}(x|\theta)$")
            ax.set_xlim(1e-3, 1)
            ax.set_xscale("log")
            ax.grid(True, which='both', linestyle=':', linewidth=0.5)
            ax.legend(frameon=False)
            plt.tight_layout()
            out_path = f"{save_dir}/{fn_name}.png" if save_dir else f"{fn_name}.png"
            plt.savefig(out_path)
            plt.close(fig)

    elif problem == 'realistic_dis':
        x_range = (1e-3, 0.9)
        x_vals = torch.linspace(x_range[0], x_range[1], 500).to(device)
        Q2_slices = Q2_slices or [2.0, 10.0, 50.0, 200.0]

        color_palette = plt.cm.viridis_r(np.linspace(0, 1, len(Q2_slices)))
        for i, Q2_fixed in enumerate(Q2_slices):
            Q2_vals = torch.full_like(x_vals, Q2_fixed).to(device)
            q_vals_all = []

            for j in range(n_mc):
                simulator.init(samples[j])
                q_vals = simulator.q(x_vals, Q2_vals)
                q_vals_all.append(q_vals.unsqueeze(0))

            q_stack = torch.cat(q_vals_all, dim=0)
            median_q = torch.median(q_stack, dim=0).values
            lower_q = torch.quantile(q_stack, 0.25, dim=0)
            upper_q = torch.quantile(q_stack, 0.75, dim=0)

            simulator.init(true_params.squeeze())
            true_q = simulator.q(x_vals, Q2_vals)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(x_vals.detach().cpu(), true_q.detach().cpu(), color=color_palette[i], linewidth=2.5,
                    label=fr"True $q(x,\ Q^2={Q2_fixed})$")
            ax.plot(x_vals.detach().cpu(), median_q.detach().cpu(), linestyle='--', color="crimson", linewidth=2,
                    label=fr"Median $\hat{{q}}(x,\ Q^2={Q2_fixed})$")
            ax.fill_between(x_vals.detach().cpu(), lower_q.detach().cpu(), upper_q.detach().cpu(), color="crimson", alpha=0.2)

            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$q(x,\ Q^2)$")
            ax.set_xlim(x_range)
            ax.set_xscale("log")
            ax.set_title(fr"$q(x)$ at $Q^2 = {Q2_fixed}\ \mathrm{{GeV}}^2$")
            ax.legend(frameon=False)
            ax.grid(True, which="both", linestyle=":", linewidth=0.5)
            plt.tight_layout()
            path = f"{save_dir}/q_Q2_{int(Q2_fixed)}.png" if save_dir else f"q_Q2_{int(Q2_fixed)}.png"
            plt.savefig(path)
            plt.close(fig)

def plot_PDF_distribution_single_same_plot(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,
    problem='simplified_dis',
    Q2_slices=None,
    plot_IQR=False,
    save_path="pdf_overlay.png"
):
    model.eval()
    pointnet_model.eval()

    simulator = RealisticDIS(torch.device('cpu')) if problem == 'realistic_dis' else SimplifiedDIS(torch.device('cpu'))

    # Feature extraction
    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    samples = sample_with_mc_dropout(model, latent_embedding, n_samples=n_mc).squeeze()

    if problem == 'realistic_dis':
        x_range = (1e-3, 0.9)
        x_vals = torch.linspace(*x_range, 500).to(device)

        Q2_slices = Q2_slices or [1.0, 1.5, 2.0, 10.0, 50.0]
        fig, ax = plt.subplots(figsize=(8, 6))
        color_palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(Q2_slices)))

        for i, Q2_fixed in enumerate(Q2_slices):
            Q2_vals = torch.full_like(x_vals, Q2_fixed).to(device)
            q_vals_all = []

            for j in range(n_mc):
                simulator.init(samples[j])
                q_vals = simulator.q(x_vals, Q2_vals)
                q_vals_all.append(q_vals.unsqueeze(0))

            q_stack = torch.cat(q_vals_all, dim=0)
            median_q = q_stack.median(dim=0).values
            lower_q = torch.quantile(q_stack, 0.25, dim=0)
            upper_q = torch.quantile(q_stack, 0.75, dim=0)

            simulator.init(true_params.squeeze())
            true_q = simulator.q(x_vals, Q2_vals)

            # Plotting curves
            ax.plot(x_vals.detach().cpu(), true_q.detach().cpu(), color=color_palette[i], linewidth=2,
                    label=fr"True $q(x,\ Q^2={Q2_fixed})$")
            ax.plot(x_vals.detach().cpu(), median_q.detach().cpu(), linestyle='--', color=color_palette[i], linewidth=1.8,
                    label=fr"Median $\hat{{q}}(x,\ Q^2={Q2_fixed})$")

            if plot_IQR:
                ax.fill_between(x_vals.detach().cpu(), lower_q.detach().cpu(), upper_q.detach().cpu(),
                                color=color_palette[i], alpha=0.2)

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$q(x,\ Q^2)$")
        ax.set_title(r"Posterior over $q(x, Q^2)$ at Multiple $Q^2$ Slices")
        ax.set_xscale("log")
        ax.set_xlim(x_range)
        ax.grid(True, which='both', linestyle=':', linewidth=0.6)
        ax.legend(loc="best", frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

# Load the model and data
def load_model_and_data(model_dir, num_samples=100, num_events=10000, problem='simplified_dis', device=None, nll_mode=False):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = os.path.join(model_dir + '_inference', 'final_inference_net.pth')
    pointnet_model_path = os.path.join(model_dir, 'final_model.pth')
    latent_path = os.path.join(model_dir, 'latent_features.h5')

    thetas, xs = generate_data(num_samples, num_events, device=torch.device('cpu'), problem=problem)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor_engineered = advanced_feature_engineering(xs)
    input_dim = xs_tensor_engineered.shape[-1]

    latent_dim = 1024
    model = InferenceNet(embedding_dim=latent_dim, output_dim=thetas.size(-1), nll_mode=nll_mode).to(device)
    pointnet_model = PointNetPMA(input_dim=input_dim, latent_dim=latent_dim)

    # Load PointNet model
    state_dict = torch.load(pointnet_model_path)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    pointnet_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    pointnet_model.eval().to(device)

    # Load inference model
    state_dict = torch.load(model_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.eval().to(device)
    # Precompute latents if necessary
    if not os.path.exists(latent_path):
        precompute_latents_to_disk(pointnet_model, xs_tensor_engineered, thetas, latent_path, chunk_size=64)

    dataset = H5Dataset(latent_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True,
                            collate_fn=EventDataset.collate_fn, num_workers=0, pin_memory=True, persistent_workers=False)

    return model, pointnet_model, dataloader, device


def detect_model_mode_mismatch(model, expected_nll_mode):
    """
    Detect if the loaded model was trained with a different loss mode than expected.
    
    Args:
        model: The loaded InferenceNet model
        expected_nll_mode (bool): The expected mode (True for NLL, False for MSE)
        
    Returns:
        tuple: (is_mismatch, detected_mode_str, expected_mode_str)
    """
    # Check if model has the NLL-specific heads
    has_mean_head = hasattr(model, 'mean_head')
    has_log_var_head = hasattr(model, 'log_var_head')
    has_output_head = hasattr(model, 'output_head')
    
    # Determine the model's actual mode
    model_is_nll = has_mean_head and has_log_var_head and not has_output_head
    model_is_mse = has_output_head and not has_mean_head and not has_log_var_head
    
    if not (model_is_nll or model_is_mse):
        # Unclear architecture, assume no mismatch to be safe
        return False, "Unknown", "Unknown"
    
    detected_mode_str = "NLL" if model_is_nll else "MSE"
    expected_mode_str = "NLL" if expected_nll_mode else "MSE"
    
    is_mismatch = model_is_nll != expected_nll_mode
    
    return is_mismatch, detected_mode_str, expected_mode_str


def main():
    """
    Main function for plotting driver with support for both MSE and NLL loss modes.
    
    Usage:
        python plotting_driver.py                    # Default MSE mode
        python plotting_driver.py --nll-loss        # NLL mode
        python plotting_driver.py --help            # Show help
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Plot training results and model predictions. '
                   'Supports both MSE and NLL loss modes with appropriate labeling.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plotting_driver.py                    # Plot with MSE loss labels (default)
  python plotting_driver.py --nll-loss        # Plot with NLL loss labels
        """
    )
    parser.add_argument('--nll-loss', action='store_true',
                       help='Use NLL loss mode for plot labels and model loading. '
                            'Should match the mode used during training.')
    
    args = parser.parse_args()
    
    # Log which mode is being used
    loss_mode = "NLL" if args.nll_loss else "MSE"
    print(f"Plotting mode: {loss_mode} loss")
    print(f"Using {'NLL (Gaussian negative log-likelihood)' if args.nll_loss else 'MSE (Mean Squared Error)'} loss labels")
    
    problem = 'simplified_dis'  # 'simplified_dis' or 'realistic_dis'
    latent_dim = 1024
    num_samples = 1000
    num_events = 100000
    model_dir = f"experiments/{problem}_latent{latent_dim}_ns_{num_samples}_ne_{num_events}"  # CHANGE per ablation
    plot_dir = os.path.join(model_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    n_mc = 100

    model, pointnet_model, dataloader, device = load_model_and_data(model_dir, problem=problem, nll_mode=args.nll_loss)
    
    # Check for model architecture mismatch
    is_mismatch, detected_mode, expected_mode = detect_model_mode_mismatch(model, args.nll_loss)
    if is_mismatch:
        print(f"⚠️  WARNING: Model architecture mismatch detected!")
        print(f"   Model was likely trained with {detected_mode} loss, but plotting in {expected_mode} mode.")
        print(f"   Plot labels may not accurately reflect the actual loss type used during training.")
        print(f"   Consider using {'--nll-loss' if detected_mode == 'NLL' else 'no --nll-loss flag'} instead.")
        print()
    true_params = torch.tensor([0.5, 0.5, 0.5, 0.5])
    if problem == 'realistic_dis':
        true_params = torch.tensor([1.0, 0.1, 0.7, 3.0, 0.0, 0.0])
        simulator = RealisticDIS(torch.device('cpu'))
    elif problem == 'simplified_dis':
        true_params = torch.tensor([0.5, 0.5, 0.5, 0.5])
        simulator = SimplifiedDIS(torch.device('cpu'))
    # true_params = torch.tensor([1.0, 0.1, 0.7, 3.0, 0.0, 0.0])
    """
    IF YOU ONLY CARE ABOUT THE SBI TOOLBOX POSTERIOR PLOTS, YOU JUST NEED THE FOLLOWING LINES AND CAN
    COMMENT OUT EVERYTHING AFTERWARDS.

    STARTING HERE:
    """
    samples_snpe = torch.tensor(np.loadtxt("samples_snpe.txt"), dtype=torch.float32)
    samples_wass = torch.tensor(np.loadtxt("samples_wasserstein.txt"), dtype=torch.float32)
    samples_mmd = torch.tensor(np.loadtxt("samples_mmd.txt"), dtype=torch.float32)

    # plot_params_distribution_single(
    #     model=model,
    #     pointnet_model=pointnet_model,
    #     true_params=true_params,
    #     device=device,
    #     n_mc=n_mc,
    #     compare_with_sbi=True,
    #     sbi_posteriors=[samples_snpe, samples_mmd, samples_wass],
    #     sbi_labels=["SNPE", "MCABC", "Wasserstein MCABC"],
    #     problem=problem,
    #     # save_dir=plot_dir if problem == 'simplified_dis' else None,
    #     save_path=os.path.join(plot_dir, "sbi_params_distribution.png")
    # )
    """
    ENDING HERE!
    """

    """
    If you want to run the full evaluation and plotting, uncomment the following lines.
    NOTE: you need to have already ran cl.py and PDF_learning.py already.
    """
    plot_PDF_distribution_single_same_plot(
        model=model,
        pointnet_model=pointnet_model,
        true_params=true_params,
        device=device,
        n_mc=n_mc,
        problem=problem,
        save_path=os.path.join(plot_dir, "pdf_overlay.png")
    )

    plot_params_distribution_single(
        model=model,
        pointnet_model=pointnet_model,
        true_params=true_params,
        device=device,
        n_mc=n_mc,
        compare_with_sbi=False,
        problem=problem,
        # save_dir=plot_dir if problem == 'simplified_dis' else None,
        save_path=os.path.join(plot_dir, "params_distribution.png")
    )
    plot_PDF_distribution_single(model, pointnet_model, true_params, device, n_mc=n_mc, problem=problem, Q2_slices=[0.5, 1.0, 1.5, 2.0, 10.0, 50.0, 200.0], save_dir=plot_dir)

    if problem == 'simplified_dis':
        plot_event_histogram_simplified_DIS(model, pointnet_model, true_params, device, n_mc=n_mc, num_events=num_events, save_path=os.path.join(plot_dir, "event_histogram_simplified.png"))
    # if problem == 'realistic_dis':
    #     plot_event_histogram_3d(model, pointnet_model, true_params, device, n_mc=n_mc, num_events=num_events, save_path=os.path.join(plot_dir, "event_histogram_3d.png"))

    # plot_loss_curves(save_path=os.path.join(plot_dir, "loss_curve.png"), nll_loss=args.nll_loss)

if __name__ == "__main__":
    main()