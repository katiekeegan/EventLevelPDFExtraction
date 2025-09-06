import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from simulator import advanced_feature_engineering, SimplifiedDIS, RealisticDIS
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from simulator import *
from PDF_learning import advanced_feature_engineering
from plotting_driver_UQ import reload_pointnet
from datasets import *
import pylab as py

def get_analytic_uncertainty(model, latent_embedding, laplace_model=None):
    """
    Return (mean_params, std_params) for the model outputs given a Laplace posterior.
    Works across older laplace-torch builds by trying several APIs.
    """
    device = latent_embedding.device
    model.eval()

    if laplace_model is not None:
        with torch.no_grad():
            # --- Path 1: predictive_distribution(x) -> distribution with .loc and .scale
            pred_dist_fn = getattr(laplace_model, "predictive_distribution", None)
            if callable(pred_dist_fn):
                dist = pred_dist_fn(latent_embedding)
                mean_params = dist.loc
                std_params  = dist.scale
                return mean_params.cpu(), std_params.cpu()

            # --- Path 2: calling the object sometimes returns (mean, var)
            try:
                out = laplace_model(latent_embedding, joint=False)
                if isinstance(out, tuple) and len(out) == 2:
                    pred_mean, pred_var = out
                    if pred_var.dim() == 3:
                        pred_std = torch.sqrt(torch.diagonal(pred_var, dim1=-2, dim2=-1))
                    else:
                        pred_std = torch.sqrt(pred_var.clamp_min(0))
                    return pred_mean.cpu(), pred_std.cpu()
            except Exception:
                pass

            # --- Path 3: predict(..., pred_type='glm', link_approx='mc')
            predict_fn = getattr(laplace_model, "predict", None)
            if callable(predict_fn):
                try:
                    pred = predict_fn(latent_embedding, pred_type='glm', link_approx='mc', n_samples=200)
                    if isinstance(pred, tuple) and len(pred) == 2:
                        mean, var = pred
                        std = torch.sqrt(var.clamp_min(0))
                        return mean.cpu(), std.cpu()
                    if hasattr(pred, "loc") and hasattr(pred, "scale"):
                        return pred.loc.cpu(), pred.scale.cpu()
                except Exception:
                    pass

    # --- Fallbacks (no Laplace available) ---
    with torch.no_grad():
        output = model(latent_embedding.to(device))
    if isinstance(output, tuple) and len(output) == 2:  # Gaussian head
        means, logvars = output
        stds = torch.exp(0.5 * logvars)
        return means.cpu(), stds.cpu()
    elif isinstance(output, tuple) and len(output) == 3:  # Multimodal head
        means, logvars, weights = output
        b = means.shape[0]
        idx = torch.argmax(weights, dim=-1)
        sel_means = means[torch.arange(b), idx]
        sel_stds  = torch.exp(0.5 * logvars[torch.arange(b), idx])
        return sel_means.cpu(), sel_stds.cpu()
    else:  # deterministic
        pred_mean = output
        pred_std  = torch.zeros_like(pred_mean)
        return pred_mean.cpu(), pred_std.cpu()


def get_gaussian_samples(model, latent_embedding, n_samples=100, laplace_model=None):
    """
    DEPRECATED: Use get_analytic_uncertainty for improved speed and accuracy.
    
    Legacy function for generating parameter samples from model uncertainty.
    When laplace_model is provided, converts analytic uncertainty to samples
    for backward compatibility. This function is maintained for compatibility
    but should be replaced with analytic methods in new code.
    
    Args:
        model: Neural network model (head)
        latent_embedding: Input latent embedding tensor
        n_samples: Number of samples to generate (for backward compatibility)
        laplace_model: Fitted Laplace approximation object
        
    Returns:
        torch.Tensor: Generated samples [n_samples, param_dim]
        
    Note: This function now uses analytic uncertainty internally and converts
    to samples, providing the same interface but with improved accuracy.
    """
    # For backward compatibility, convert analytic uncertainty to samples
    mean_params, std_params = get_analytic_uncertainty(model, latent_embedding, laplace_model)
    
    # Generate samples from analytic Gaussian distribution
    device = latent_embedding.device
    batch_size, param_dim = mean_params.shape
    
    # Generate n_samples for each batch element
    samples = []
    for i in range(n_samples):
        # Sample from Gaussian with analytic mean and std
        sample = torch.randn_like(mean_params) * std_params + mean_params
        samples.append(sample.squeeze(0) if batch_size == 1 else sample[0])
    
    return torch.stack(samples)

def plot_params_distribution_single(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,  # Kept for backward compatibility but ignored when using analytic uncertainty
    laplace_model=None,
    compare_with_sbi=False,
    sbi_posteriors=None,
    sbi_labels=None,
    save_path="Dist.png",
    problem='simplified_dis'
):
    """
    Plot parameter distributions using analytic Laplace uncertainty propagation.
    
    When laplace_model is provided, uses analytic uncertainty propagation via 
    delta method instead of Monte Carlo sampling for improved speed and accuracy.
    """
    model.eval()
    pointnet_model.eval()
    if problem == 'realistic_dis':
        simulator = RealisticDIS(torch.device('cpu'))
    else:
        simulator = SimplifiedDIS(torch.device('cpu'))

    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    if problem != 'mceg':
        xs_tensor = advanced_feature_engineering(xs_tensor)
    else:
        xs_tensor = xs_tensor  # No feature engineering for MCEG
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    if laplace_model is not None:
        # Use analytic uncertainty propagation (delta method)
        mean_params, std_params = get_analytic_uncertainty(model, latent_embedding, laplace_model)
        mean_params = mean_params.cpu().squeeze(0)
        std_params = std_params.cpu().squeeze(0)
        use_analytic = True
    else:
        # Fallback to MC sampling for backward compatibility
        samples = get_gaussian_samples(model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model).cpu()
        use_analytic = False

    n_params = true_params.size(0)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    colors = ['skyblue', 'orange', 'green', 'purple', 'gray']
    param_names = [r'$a_u$', r'$b_u$', r'$a_d$', r'$b_d$'] if problem == 'simplified_dis' else [r'$\log A_0$', r'$\delta$', r'$a$', r'$b$', r'$c$', r'$d$']

    # Prepare data for plotting
    all_samples = []
    if not use_analytic:
        all_samples = [samples]
    if compare_with_sbi and sbi_posteriors is not None:
        all_samples.extend([s.detach().cpu() for s in sbi_posteriors])

    for i in range(n_params):
        if use_analytic:
            # Plot analytic Gaussian distribution
            mu = mean_params[i].item()
            sigma = std_params[i].item()
            
            # Create x range around the mean
            x_range = 4 * sigma  # Show ±4 standard deviations
            x_vals = torch.linspace(mu - x_range, mu + x_range, 1000)
            
            # Compute Gaussian PDF
            gaussian_pdf = torch.exp(-0.5 * ((x_vals - mu) / sigma) ** 2) / (sigma * torch.sqrt(2 * torch.tensor(torch.pi)))
            
            # Plot the analytic Gaussian
            axes[i].plot(x_vals.numpy(), gaussian_pdf.numpy(), color=colors[0], linewidth=2, 
                        label='Analytic Posterior (Laplace)', alpha=0.8)
            axes[i].fill_between(x_vals.numpy(), 0, gaussian_pdf.numpy(), color=colors[0], alpha=0.3)
            
            # Set appropriate x limits
            axes[i].set_xlim(mu - x_range, mu + x_range)
        else:
            # Plot histogram from MC samples (legacy approach)
            param_vals = [s[:, i].numpy() for s in all_samples]
            xmin = min([v.min() for v in param_vals])
            xmax = max([v.max() for v in param_vals])
            padding = 0.05 * (xmax - xmin)
            xmin -= padding
            xmax += padding
            
            axes[i].hist(samples[:, i].numpy(), bins=20, alpha=0.6, density=True, 
                        color=colors[0], label='MC Posterior Samples')
            axes[i].set_xlim(xmin, xmax)

        # Add SBI comparison if requested
        if compare_with_sbi and sbi_posteriors is not None and sbi_labels is not None:
            for j, sbi_samples in enumerate(sbi_posteriors):
                label = sbi_labels[j] if j < len(sbi_labels) else f"SBI {j}"
                axes[i].hist(
                    sbi_samples[:, i].detach().cpu().numpy(),
                    bins=20, alpha=0.4, density=True,
                    color=colors[(j + 1) % len(colors)],
                    label=label
                )
        
        # Add true value line
        axes[i].axvline(true_params[i].item(), color='red', linestyle='dashed', linewidth=2, label='True Value')
        axes[i].set_title(f'{param_names[i]}')
        axes[i].set_ylabel('Density')
        if i == 0: 
            axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_PDF_distribution_single(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=200,  # bump a bit if you like smoother quantiles
    laplace_model=None,
    problem='simplified_dis',
    Q2_slices=None,
    save_dir=None,
    save_path="pdf_distribution.png"
):
    """
    Plot PDF distributions using parameter-posterior sampling.

    If laplace_model is provided, draw parameter samples from the Laplace
    Gaussian posterior and compute the pointwise median/IQR of the resulting
    function values (same aggregation as the legacy MC path).
    """
    model.eval()
    pointnet_model.eval()
    simulator = RealisticDIS(torch.device('cpu')) if problem == 'realistic_dis' else SimplifiedDIS(torch.device('cpu'))

    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    if problem != 'mceg':
        xs_tensor = advanced_feature_engineering(xs_tensor)
    else:
        xs_tensor = xs_tensor
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # --- Sampling strategy ---
    # If Laplace is available, sample parameters from its Gaussian posterior.
    # Otherwise, fall back to your legacy get_gaussian_samples behavior.
    if laplace_model is not None:
        samples = get_gaussian_samples(
            model,
            latent_embedding,
            n_samples=n_mc,
            laplace_model=laplace_model  # should draw from N(theta_hat, Sigma_laplace)
        ).cpu()
        label_curve = "Median (Laplace posterior MC)"
        label_band  = "IQR"
    else:
        samples = get_gaussian_samples(
            model,
            latent_embedding,
            n_samples=n_mc,
            laplace_model=None
        ).cpu()
        label_curve = "Median (MC)"
        label_band  = "IQR"

    if problem == 'simplified_dis':
        x_vals = torch.linspace(0, 1, 500).to(device)
        for fn_name, fn_label, color in [("up", "u", "royalblue"), ("down", "d", "darkorange")]:
            # Evaluate function for each sampled parameter vector
            fn_vals_all = []
            for i in range(samples.shape[0]):
                simulator.init(samples[i])
                fn = getattr(simulator, fn_name)
                fn_vals_all.append(fn(x_vals).unsqueeze(0))

            fn_stack = torch.cat(fn_vals_all, dim=0)  # [n_mc, 500]
            median_vals = fn_stack.median(dim=0).values.detach().cpu()
            lower_bounds = torch.quantile(fn_stack, 0.25, dim=0).detach().cpu()
            upper_bounds = torch.quantile(fn_stack, 0.75, dim=0).detach().cpu()

            # True curve
            simulator.init(true_params.squeeze())
            true_vals = getattr(simulator, fn_name)(x_vals).detach().cpu()

            # Plot
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(x_vals.detach().cpu(), true_vals, label=fr"True ${fn_label}(x|\theta^*)$", color=color, linewidth=2)
            ax.plot(
                x_vals.detach().cpu(),
                median_vals,
                linestyle='--',
                label=fr"{label_curve} ${fn_label}(x)$",
                color="crimson",
                linewidth=2
            )
            ax.fill_between(
                x_vals.detach().cpu(),
                lower_bounds,
                upper_bounds,
                color="crimson",
                alpha=0.3,
                label=label_band
            )

            ax.set_xlabel(r"$x$")
            ax.set_ylabel(fr"${fn_label}(x|\theta)$")
            ax.set_xlim(1e-3, 1)
            ax.set_xscale("log")
            ax.grid(True, which='both', linestyle=':', linewidth=0.5)
            ax.legend(frameon=False)
            plt.tight_layout()
            out_path = f"{save_dir}/{fn_name}.png" if save_dir else f"{fn_name}.png"
            plt.savefig(out_path, dpi=300)
            plt.close(fig)

    elif problem == 'realistic_dis':
        x_range = (1e-3, 0.9)
        x_vals = torch.linspace(x_range[0], x_range[1], 500).to(device)
        Q2_slices = Q2_slices or [2.0, 10.0, 50.0, 200.0]
        color_palette = plt.cm.viridis_r(np.linspace(0, 1, len(Q2_slices)))
        
        for i, Q2_fixed in enumerate(Q2_slices):
            Q2_vals = torch.full_like(x_vals, Q2_fixed).to(device)
            
            if use_analytic:
                # Compute PDF using analytic uncertainty
                simulator.init(mean_params)
                mean_q = simulator.q(x_vals, Q2_vals).detach().cpu()
                
                # Approximate uncertainty bounds
                param_std_norm = torch.norm(std_params).item()
                uncertainty_factor = 2.0 * param_std_norm
                
                lower_q = mean_q * (1 - uncertainty_factor)
                upper_q = mean_q * (1 + uncertainty_factor)
                lower_q = torch.clamp(lower_q, min=0.0)
                
            else:
                # MC sampling approach (legacy)
                q_vals_all = []
                for j in range(n_mc):
                    simulator.init(samples[j])
                    q_vals = simulator.q(x_vals, Q2_vals)
                    q_vals_all.append(q_vals.unsqueeze(0))
                q_stack = torch.cat(q_vals_all, dim=0)
                mean_q = torch.median(q_stack, dim=0).values.detach().cpu()
                lower_q = torch.quantile(q_stack, 0.25, dim=0).detach().cpu()
                upper_q = torch.quantile(q_stack, 0.75, dim=0).detach().cpu()

            # Compute true values
            simulator.init(true_params.squeeze())
            true_q = simulator.q(x_vals, Q2_vals).detach().cpu()

            # Create plot
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(x_vals.detach().cpu(), true_q, color=color_palette[i], linewidth=2.5,
                    label=fr"True $q(x,\ Q^2={Q2_fixed})$")
            
            if use_analytic:
                ax.plot(x_vals.detach().cpu(), mean_q, linestyle='--', color="crimson", linewidth=2,
                        label=fr"MAP $\hat{{q}}(x,\ Q^2={Q2_fixed})$ (Analytic)")
                ax.fill_between(x_vals.detach().cpu(), lower_q, upper_q, 
                               color="crimson", alpha=0.2, label="95% Analytic CI")
            else:
                ax.plot(x_vals.detach().cpu(), mean_q, linestyle='--', color="crimson", linewidth=2,
                        label=fr"Median $\hat{{q}}(x,\ Q^2={Q2_fixed})$ (MC)")
                ax.fill_between(x_vals.detach().cpu(), lower_q, upper_q, color="crimson", alpha=0.2)

            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$q(x,\ Q^2)$")
            ax.set_xlim(x_range)
            ax.set_xscale("log")
            ax.set_title(fr"$q(x)$ at $Q^2 = {Q2_fixed}\ \mathrm{{GeV}}^2$")
            ax.legend(frameon=False)
            ax.grid(True, which="both", linestyle=":", linewidth=0.5)
            plt.tight_layout()
            path = f"{save_dir}/q_Q2_{int(Q2_fixed)}.png" if save_dir else f"q_Q2_{int(Q2_fixed)}.png"
            plt.savefig(path, dpi=300)
            plt.close(fig)

def plot_PDF_distribution_single_same_plot(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,  # Kept for backward compatibility but ignored when using analytic uncertainty
    laplace_model=None,
    problem='simplified_dis',
    Q2_slices=None,
    plot_IQR=False,
    save_path="pdf_overlay.png"
):
    """
    Plot PDF distributions on the same plot using analytic Laplace uncertainty propagation.
    
    When laplace_model is provided, uses analytic uncertainty propagation to 
    compute error bands instead of Monte Carlo sampling for improved speed and accuracy.
    """
    model.eval()
    pointnet_model.eval()
    simulator = RealisticDIS(torch.device('cpu')) if problem == 'realistic_dis' else SimplifiedDIS(torch.device('cpu'))

    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    if problem != 'mceg':
        xs_tensor = advanced_feature_engineering(xs_tensor)
    else:
        xs_tensor = xs_tensor
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    if laplace_model is not None:
        # Use analytic uncertainty propagation (delta method)
        mean_params, std_params = get_analytic_uncertainty(model, latent_embedding, laplace_model)
        mean_params = mean_params.cpu().squeeze(0)
        std_params = std_params.cpu().squeeze(0)
        use_analytic = True
    else:
        # Fallback to MC sampling for backward compatibility
        samples = get_gaussian_samples(model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model).cpu()
        use_analytic = False

    if problem == 'realistic_dis':
        x_range = (1e-3, 0.9)
        x_vals = torch.linspace(*x_range, 500).to(device)
        Q2_slices = Q2_slices or [1.0, 1.5, 2.0, 10.0, 50.0]
        fig, ax = plt.subplots(figsize=(8, 6))
        color_palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(Q2_slices)))

        for i, Q2_fixed in enumerate(Q2_slices):
            Q2_vals = torch.full_like(x_vals, Q2_fixed).to(device)
            
            if use_analytic:
                # Compute PDF using analytic uncertainty
                simulator.init(mean_params)
                mean_q = simulator.q(x_vals, Q2_vals).detach().cpu()
                
                # Approximate uncertainty bounds
                param_std_norm = torch.norm(std_params).item()
                uncertainty_factor = 2.0 * param_std_norm
                
                lower_q = mean_q * (1 - uncertainty_factor)
                upper_q = mean_q * (1 + uncertainty_factor)
                lower_q = torch.clamp(lower_q, min=0.0)
                
            else:
                # MC sampling approach (legacy)
                q_vals_all = []
                for j in range(n_mc):
                    simulator.init(samples[j])
                    q_vals = simulator.q(x_vals, Q2_vals)
                    q_vals_all.append(q_vals.unsqueeze(0))
                q_stack = torch.cat(q_vals_all, dim=0)
                mean_q = q_stack.median(dim=0).values.detach().cpu()
                lower_q = torch.quantile(q_stack, 0.25, dim=0).detach().cpu()
                upper_q = torch.quantile(q_stack, 0.75, dim=0).detach().cpu()

            # Compute true values
            simulator.init(true_params.squeeze())
            true_q = simulator.q(x_vals, Q2_vals).detach().cpu()

            # Plot true and predicted values
            ax.plot(x_vals.detach().cpu(), true_q, color=color_palette[i], linewidth=2,
                    label=fr"True $q(x,\ Q^2={Q2_fixed})$")
            
            if use_analytic:
                ax.plot(x_vals.detach().cpu(), mean_q, linestyle='--', color=color_palette[i], linewidth=1.8,
                        label=fr"MAP $\hat{{q}}(x,\ Q^2={Q2_fixed})$ (Analytic)")
            else:
                ax.plot(x_vals.detach().cpu(), mean_q, linestyle='--', color=color_palette[i], linewidth=1.8,
                        label=fr"Median $\hat{{q}}(x,\ Q^2={Q2_fixed})$ (MC)")
            
            # Add uncertainty bands if requested
            if plot_IQR or use_analytic:
                label_suffix = "95% Analytic CI" if use_analytic else "IQR"
                ax.fill_between(x_vals.detach().cpu(), lower_q, upper_q,
                                color=color_palette[i], alpha=0.2)
                
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$q(x,\ Q^2)$")
        uncertainty_type = "Analytic Laplace" if use_analytic else "MC Sampling"
        ax.set_title(fr"Posterior over $q(x, Q^2)$ at Multiple $Q^2$ Slices ({uncertainty_type})")
        ax.set_xscale("log")
        ax.set_xlim(x_range)
        ax.grid(True, which='both', linestyle=':', linewidth=0.6)
        ax.legend(loc="best", frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

def plot_event_histogram_simplified_DIS(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,  # Kept for backward compatibility but ignored when using analytic uncertainty
    laplace_model=None,
    num_events=100000,
    save_path="event_histogram_simplified.png",
    problem='simplified_dis'
):
    """
    Plot event histograms using analytic Laplace uncertainty propagation.
    
    When laplace_model is provided, uses analytic MAP estimate instead of 
    Monte Carlo sampling for improved speed and accuracy.
    """
    model.eval()
    pointnet_model.eval()
    simulator = SimplifiedDIS(torch.device('cpu'))
    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), num_events).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    if problem != 'mceg':
        xs_tensor = advanced_feature_engineering(xs_tensor)
    else:
        xs_tensor = xs_tensor
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    if laplace_model is not None:
        # Use analytic uncertainty propagation (delta method)
        mean_params, std_params = get_analytic_uncertainty(model, latent_embedding, laplace_model)
        predicted_params = mean_params.cpu().squeeze(0)
        use_analytic = True
    else:
        # Fallback to MC sampling for backward compatibility
        samples = get_gaussian_samples(model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model).cpu()
        predicted_params = torch.median(samples, dim=0).values
        use_analytic = False

    # Generate events using the predicted parameters
    generated_events = simulator.sample(predicted_params.detach().cpu(), num_events).to(device)
    
    # Convert to numpy for plotting
    true_events_np = xs.detach().cpu().numpy()
    generated_events_np = generated_events.detach().cpu().numpy()
    
    # Create the plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    
    # True events
    axs[0].scatter(true_events_np[:, 0], true_events_np[:, 1], color='turquoise', alpha=0.2)
    axs[0].set_title(r"$\Xi_{\theta^{*}}$ (True Parameters)")
    axs[0].set_xlabel(r"$x_{u} \sim u(x|\theta^{*})$")
    axs[0].set_ylabel(r"$x_{d} \sim d(x|\theta^{*})$")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    
    # Generated events
    method_label = "MAP (Analytic)" if use_analytic else "Median (MC)"
    axs[1].scatter(generated_events_np[:, 0], generated_events_np[:, 1], color='darkorange', alpha=0.2)
    axs[1].set_title(fr"$\Xi_{{\hat{{\theta}}}}$ ({method_label})")
    axs[1].set_xlabel(fr"$x_{{u}} \sim u(x|\hat{{\theta}})$ ({method_label})")
    axs[1].set_ylabel(fr"$x_{{d}} \sim d(x|\hat{{\theta}})$ ({method_label})")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_loss_curves(loss_dir='.', save_path='loss_plot.png', show_plot=False, nll_loss=False):
    contrastive_path = os.path.join(loss_dir, 'loss_contrastive.npy')
    regression_path = os.path.join(loss_dir, 'loss_regression.npy')
    total_path = os.path.join(loss_dir, 'loss_total.npy')
    contrastive_loss = np.load(contrastive_path)
    regression_loss = np.load(regression_path)
    total_loss = np.load(total_path)
    epochs = np.arange(1, len(contrastive_loss) + 1)
    loss_type = "NLL" if nll_loss else "MSE"
    regression_label = f'Regression Loss ({loss_type}, scaled)'
    title = f'Training Loss Components Over Epochs ({loss_type} Regression)'
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, contrastive_loss, label='Contrastive Loss', linewidth=2)
    plt.plot(epochs, regression_loss, label=regression_label, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    if show_plot: plt.show()
    plt.close()
    epochs = np.arange(1, len(total_loss) + 1)
    total_title = f'Training Loss Over Epochs ({loss_type} Regression)'
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, total_loss, label='Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(total_title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_PDF_learning.png', dpi=300)
    if show_plot: plt.show()
    plt.close()
    plt.figure(figsize=(8, 6))
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
    if show_plot: plt.show()
    plt.close()


def plot_latents(latents, params, method='umap', param_idx=0, title=None, save_path=None):
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(latents)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(emb[:,0], emb[:,1], c=params[:,param_idx], cmap='viridis', s=30)
    plt.xlabel(f"{method.upper()} dim 1")
    plt.ylabel(f"{method.upper()} dim 2")
    plt.title(title or f"Latent space ({method.upper()}) colored by param {param_idx}")
    plt.colorbar(scatter, label=f"Parameter {param_idx}")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def extract_latents_from_data(pointnet_model, args, problem, device, num_samples=1000):
    """
    Generate data and extract latents using PointNet model.

    Returns:
        latents: [n_samples, latent_dim]
        thetas: [n_samples, param_dim]
    """
    # Generate parameters and simulated events
    thetas, xs = generate_data(1000, args.num_events, problem=problem, device=device)
    # Feature engineering
    if problem != 'mceg':
        feats = advanced_feature_engineering(xs)  # [n_samples * n_events, n_features]
    else:
        feats = xs
    # Reshape for PointNet batching
    n_samples = thetas.shape[0]
    n_events = xs.shape[1]
    feats = feats.view(n_samples, n_events, -1)  # [n_samples, n_events, n_features]

    # Extract latents
    latents = []
    with torch.no_grad():
        for i in range(n_samples):
            with torch.no_grad():
                latent = pointnet_model(feats[i].unsqueeze(0))  # [1, latent_dim]
            latents.append(latent.cpu().numpy().squeeze(0))
    latents = np.array(latents)
    return latents, thetas.cpu().numpy()

def plot_latents_umap(latents, params, color_mode='single', param_idx=0, method='umap', save_path=None, show=True):
    """
    Plot latent vectors (n_samples x latent_dim) reduced to 2D via UMAP or t-SNE,
    colored by parameters (n_samples x param_dim).
    """
    # Reduce latents to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(latents)

    # Determine coloring
    if color_mode == 'single':
        color = params[:, param_idx]
        label = f"Parameter {param_idx}"
    elif color_mode == 'mean':
        color = np.mean(params, axis=1)
        label = "Mean parameter"
    elif color_mode == 'pca':
        pca = PCA(n_components=1)
        color = pca.fit_transform(params).flatten()
        label = "First principal component of parameters"
    else:
        raise ValueError("color_mode must be 'single', 'mean', or 'pca'")

    # Plot
    plt.figure(figsize=(8,6))
    sc = plt.scatter(emb[:,0], emb[:,1], c=color, cmap='viridis', s=30)
    plt.xlabel(f"{method.upper()} dim 1")
    plt.ylabel(f"{method.upper()} dim 2")
    plt.title(f"Latent space ({method.upper()}), colored by {label}")
    plt.colorbar(sc, label=label)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

def plot_latents_all_params(latents, params, method='umap', save_path=None, show=True):
    """
    Plot latent vectors (n_samples x latent_dim) reduced to 2D via UMAP or t-SNE,
    with one subplot per parameter dimension.
    """
    # Reduce latents to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(latents)

    n_params = params.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))

    for i in range(n_params):
        ax = axes[i] if n_params > 1 else axes
        color = params[:, i]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=color, cmap='viridis', s=30)
        ax.set_xlabel(f"{method.upper()} dim 1")
        ax.set_ylabel(f"{method.upper()} dim 2")
        ax.set_title(f"Colored by Parameter {i}")
        plt.colorbar(sc, ax=ax, label=f"Parameter {i}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
def _bin_edges_log(evts_list, nx_bins=50, nQ2_bins=50, x_min=1e-4, x_max=1e-1, Q2_min=10.0, Q2_max=1e3):
    """
    Build common log-space edges across all provided event clouds.
    Fallbacks let you clamp the plotting window for stability/comparability.
    """
    # Allow autoscale if user doesn't want fixed ranges
    xs = []
    Q2s = []
    for E in evts_list:
        if E is None or len(E) == 0:
            continue
        xs.append(E[:,0])
        Q2s.append(E[:,1])
    if xs:
        x_min = max(x_min, np.nanmax([np.nanmin(a[a>0]) for a in xs]))
        x_max = min(x_max, np.nanmax([np.nanmax(a) for a in xs]))
    if Q2s:
        Q2_min = max(Q2_min, np.nanmax([np.nanmin(a[a>0]) for a in Q2s]))
        Q2_max = min(Q2_max, np.nanmax([np.nanmax(a) for a in Q2s]))
    logx_edges  = np.linspace(np.log(x_min),  np.log(x_max),  nx_bins+1)
    logQ2_edges = np.linspace(np.log(Q2_min), np.log(Q2_max), nQ2_bins+1)
    return logx_edges, logQ2_edges

def _hist2d_density_log(evts, logx_edges, logQ2_edges, total_xsec=None):
    """
    Histogram in (log x, log Q2), convert to differential rate by dividing by bin area (dx*dQ2).
    Optionally scale to match total cross section like your original code.
    """
    if evts is None or len(evts) == 0:
        H = np.zeros((len(logx_edges)-1, len(logQ2_edges)-1), dtype=float)
        return H, (logx_edges, logQ2_edges)

    H, xedges, q2edges = np.histogram2d(np.log(evts[:,0]), np.log(evts[:,1]),
                                        bins=(logx_edges, logQ2_edges))
    # Convert counts to density via (dx*dQ2)
    # Precompute dx, dQ2 on linear scale for each bin
    dx  = np.exp(xedges[1:]) - np.exp(xedges[:-1])          # (nx,)
    dQ2 = np.exp(q2edges[1:]) - np.exp(q2edges[:-1])        # (nQ2,)
    area = dx[:, None] * dQ2[None, :]
    density = np.divide(H, area, where=(area>0))
    if total_xsec is not None and H.sum() > 0:
        density *= total_xsec / H.sum()
    return density, (xedges, q2edges)

def _theory_grid(idis, xedges, q2edges, rs, tar, mode='xQ2'):
    """
    Evaluate theory on bin centers defined by (xedges, q2edges).
    """
    nx  = len(xedges)-1
    nQ2 = len(q2edges)-1
    out = np.zeros((nx, nQ2), dtype=float)
    # Bin centers in linear space
    x_centers  = np.exp(0.5*(xedges[:-1]  + xedges[1:]))
    q2_centers = np.exp(0.5*(q2edges[:-1] + q2edges[1:]))

    for i in tqdm(range(nx), desc="theory x bins", leave=False):
        x = x_centers[i]
        for j in range(nQ2):
            Q2 = q2_centers[j]
            out[i, j] = idis.get_diff_xsec(x, Q2, rs, tar, mode)
    return out

def _theory_grid_masked(idis, xedges, q2edges, rs, tar, mode, occupancy_counts):
    nx, nQ2 = len(xedges)-1, len(q2edges)-1
    out = np.zeros((nx, nQ2), dtype=float)
    x_centers  = np.exp(0.5*(xedges[:-1]  + xedges[1:]))
    q2_centers = np.exp(0.5*(q2edges[:-1] + q2edges[1:]))
    for i in range(nx):
        for j in range(nQ2):
            if occupancy_counts[i, j] > 0:      # <-- workbook’s guard
                val = idis.get_diff_xsec(x_centers[i], q2_centers[j], rs, tar, mode)
                out[i, j] = _to_scalar_xsec(val, mode_hint=mode)
    return out

def _to_scalar_xsec(v, mode_hint='xQ2'):
    import numpy as np
    try:
        import torch
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(v, dict):
        if mode_hint in v:
            return float(np.asarray(v[mode_hint]).squeeze().reshape(()))
        return float(np.asarray(next(iter(v.values()))).squeeze().reshape(()))
    if isinstance(v, (list, tuple)):
        v = v[0]
    v = np.asarray(v).squeeze()
    if v.size == 0:
        return 0.0
    if v.ndim > 0:
        v = v.flat[0]
    return float(v)

# 0) Filter bad/zero/non-finite events before logging
def _valid_evts(ev):
    if ev is None: return None
    ev = np.asarray(ev)
    m = np.isfinite(ev).all(axis=1) & (ev[:,0] > 0) & (ev[:,1] > 0)
    return ev[m]

def safe_log_levels(A, n=60, lo_pct=1.0, hi_pct=99.0, default=(1e-6, 1.0)):
    A = np.asarray(A, dtype=float)

    # keep only positive finite values
    A = np.where(np.isfinite(A) & (A > 0), A, np.nan)

    # pick percentiles to avoid extreme outliers
    vmin = np.nanpercentile(A, lo_pct)
    vmax = np.nanpercentile(A, hi_pct)

    # fallback if bad or empty
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0 or vmin >= vmax:
        vmin, vmax = default

    # log-spaced levels, same as your inline construction
    levels = 10**np.linspace(np.log10(vmin), np.log10(vmax), n)

    return levels

from typing import Optional, Tuple

@torch.no_grad()
def _simulate_pdf_curve_from_theta(
    simulator,
    theta: torch.Tensor,
    num_events: int,
    x_range: Tuple[float, float],
    bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate events under `theta` and return a density curve over x via a shared histogram.
    Returns (x_centers [B], pdf_values [B]).
    """
    xs = simulator.sample(theta.detach().cpu().float(), num_events)  # shape (N, 2) or (N, ...)
    x = np.asarray(xs)[:, 0]  # assume x is first column
    H, edges = np.histogram(x, bins=bins, range=x_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, H

def _to_SD(theta_samples: torch.Tensor) -> Tuple[int, int]:
    """Return (S, D). Accepts [S,D] or [B,S,D] -> flattens across B."""
    if theta_samples.dim() == 3:
        B, S, D = theta_samples.shape
        return B * S, D
    elif theta_samples.dim() == 2:
        S, D = theta_samples.shape
        return S, D
    else:
        raise ValueError(f"theta_samples must be [S,D] or [B,S,D], got {tuple(theta_samples.shape)}")

@torch.no_grad()
def plot_PDF_distribution_single_same_plot_from_theta_samples(
    simulator,
    theta_samples: torch.Tensor,   # [S,D] or [B,S,D] on any device
    true_params: Optional[torch.Tensor],
    device: torch.device,
    num_events_per_theta: int = 5000,
    x_range: Tuple[float, float] = (0.0, 1.0),
    bins: int = 100,
    quantiles = (5, 25, 50, 75, 95),
    overlay_point_estimate: bool = True,
    point_estimate: str = "mean",  # "mean" or "median"
    save_path: str = "pdf_overlay_flow.png",
    title: Optional[str] = None,
):
    """
    Builds posterior bands over the induced PDF(x) using simulator + theta_samples.
    - Posterior bands: 5–95% and 25–75% + median curve.
    - Optional overlays: truth curve (true_params) and a point-estimate curve (mean/median θ).
    """
    # Sanitize shapes
    if theta_samples.dim() == 3:
        B, S, D = theta_samples.shape
        thetas = theta_samples.reshape(B * S, D)
    else:
        thetas = theta_samples
        S, D = thetas.shape

    # Precompute a common x-grid (by simulating once with the first theta)
    x_centers_ref, _ = _simulate_pdf_curve_from_theta(
        simulator, thetas[0].to(device), max(2000, num_events_per_theta // 5), x_range, bins
    )
    # Simulate all θ-samples → stack PDFs
    pdf_mat = []
    for s in range(thetas.shape[0]):
        _, H = _simulate_pdf_curve_from_theta(
            simulator, thetas[s].to(device), num_events_per_theta, x_range, bins
        )
        pdf_mat.append(H)
    pdf_mat = np.stack(pdf_mat, axis=0)  # [S_total, BINS]

    # Quantile bands
    qdict = {q: np.quantile(pdf_mat, q/100.0, axis=0) for q in quantiles}

    # Optional truth curve
    truth_curve = None
    if true_params is not None:
        x_t, H_t = _simulate_pdf_curve_from_theta(
            simulator, true_params.to(device), num_events_per_theta * 5, x_range, bins
        )
        truth_curve = (x_t, H_t)

    # Optional point-estimate curve
    pe_curve = None
    if overlay_point_estimate:
        if point_estimate == "mean":
            theta_pe = thetas.mean(dim=0)
        elif point_estimate == "median":
            theta_pe = thetas.median(dim=0).values
        else:
            raise ValueError("point_estimate must be 'mean' or 'median'")
        x_pe, H_pe = _simulate_pdf_curve_from_theta(
            simulator, theta_pe.to(device), num_events_per_theta * 2, x_range, bins
        )
        pe_curve = (x_pe, H_pe)

    # Plot
    fig = plt.figure(figsize=(7, 4.5))
    ax = plt.gca()

    # Shaded bands
    if 95 in qdict and 5 in qdict:
        ax.fill_between(x_centers_ref, qdict[5], qdict[95], alpha=0.20, label="90% band")
    if 75 in qdict and 25 in qdict:
        ax.fill_between(x_centers_ref, qdict[25], qdict[75], alpha=0.35, label="50% band")

    # Median curve
    if 50 in qdict:
        ax.plot(x_centers_ref, qdict[50], linewidth=2.0, label="Posterior median")

    # Point estimate curve
    if pe_curve is not None:
        ax.plot(pe_curve[0], pe_curve[1], linestyle="--", linewidth=1.8,
                label=f"Posterior {point_estimate}")

    # Truth
    if truth_curve is not None:
        ax.plot(truth_curve[0], truth_curve[1], linewidth=2.0, label="Truth")

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    if title: ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_PDF_distribution_single_same_plot_mceg(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,                    # kept for backward compatibility
    laplace_model=None,
    problem='simplified_dis',
    Q2_slices=None,              # list of Q² values to show; if None we auto-pick
    plot_IQR=False,              # used only for MC mode (overlay disabled by default)
    save_dir=None,
    nx=100,
    nQ2=100,
    n_events=1000000,
    max_Q2_for_plot=100.0,
):
    """
    Reproduce the 'true vs reconstructed' plot style:
      - 2D histogram in (log x, log Q²) for reconstructed with error bars (Poisson)
      - 'true' curve from simulator.q at bin centers
      - OPTIONAL model overlay (MAP dashed) ONLY when laplace_model is provided
    """
    # -------- setup ----------
    model.eval()
    pointnet_model.eval()
    if problem == 'mceg':
        simulator = MCEGSimulator(torch.device('cpu'))
    elif problem == 'realistic_dis':
        simulator = RealisticDIS(torch.device('cpu'))
    elif problem == 'simplified_dis':
        simulator = SimplifiedDIS(torch.device('cpu'))
    # simulator = RealisticDIS(torch.device('cpu')) if problem == 'realistic_dis' else SimplifiedDIS(torch.device('cpu'))

    # Make sure params live on correct device
    true_params = true_params.to(device)

    # -------- initialize theory components ----------

    mellin = MELLIN(npts=8)
    alphaS = ALPHAS()
    eweak  = EWEAK()
    pdf    = PDF(mellin, alphaS)

    # -------- sample events for the reconstructed histogram ----------
    with torch.no_grad():
        events = simulator.sample(true_params.detach().cpu(), n_events)  # expected shape (N, 2) = [x, Q2]
    events = np.asarray(events)
    x_ev  = events[:, 0]
    Q2_ev = events[:, 1]

    xs_tensor = torch.tensor(events, dtype=torch.float32, device=device)
    if problem != 'mceg':
        xs_tensor = advanced_feature_engineering(xs_tensor)
    else:
        xs_tensor = xs_tensor
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))
    theta_pred = model(latent_embedding).cpu().squeeze(0).detach()

    new_cpar = pdf.get_current_par_array()[::]
    # Assume parameters are only corresponding to 'uv1' parameters
    if not isinstance(theta_pred, torch.Tensor):
        new_cpar[4:8] = theta_pred
    else:
        new_cpar[4:8] = theta_pred.cpu().numpy()  # Update uv1 parameters
    pdf.setup(new_cpar)
    idis = THEORY(mellin, pdf, alphaS, eweak)
    new_cpar_true = pdf.get_current_par_array()[::]
    new_cpar_true[4:8] = true_params.cpu().numpy() if isinstance(true_params, torch.Tensor) else true_params
    pdf_true = PDF(mellin, alphaS)
    pdf_true.setup(new_cpar_true)
    idis_true = THEORY(mellin, pdf_true, alphaS, eweak)
    mceg=MCEG(idis,rs=140,tar='p',W2min=10,nx=nx,nQ2=nQ2) 
    mceg_true = MCEG(idis_true,rs=140,tar='p',W2min=10,nx=nx,nQ2=nQ2)
    events_pred = mceg.gen_events(n_events,verb=False)

    events = mceg_true.gen_events(n_events,verb=False)
    evts = _valid_evts(events)
    evts_pred = _valid_evts(events_pred)
    if evts is None or len(evts) == 0:
        raise ValueError("No valid reco events with positive x and Q2.")

    hist=np.histogram2d(np.log(evts[:,0]),np.log(evts[:,1]),bins=(50,50))
    true=np.zeros(hist[0].shape)
    reco=np.zeros(hist[0].shape)
    gen=np.zeros(hist[0].shape)
    for i,j in tqdm((a,b) for a in range(hist[1].shape[0]-1) 
                        for b in range(hist[2].shape[0]-1)):
        if hist[0][i,j]>0: 
            x=np.exp(0.5*(hist[1][i]+hist[1][i+1]))
            Q2=np.exp(0.5*(hist[2][j]+hist[2][j+1]))
            true[i,j],_=idis_true.get_diff_xsec(x,Q2,mceg_true.rs,mceg_true.tar,'xQ2')
            
            dx=np.exp(hist[1][i+1])-np.exp(hist[1][i])
            dQ2=np.exp(hist[2][j+1])-np.exp(hist[2][j])
            reco[i,j]=hist[0][i,j]/dx/dQ2
            gen[i,j],_=idis.get_diff_xsec(x,Q2,mceg.rs,mceg.tar,'xQ2')

    reco*=mceg_true.total_xsec/np.sum(hist[0])
    gen*=mceg.total_xsec/np.sum(gen)


    nrows,ncols=1,3; AX=[]
    fig = py.figure(figsize=(ncols*6,nrows*5))
    ax=py.subplot(nrows,ncols,1);AX.append(ax)
    c=ax.pcolor(hist[1],hist[2],reco.T, norm=matplotlib.colors.LogNorm())
    ax=py.subplot(nrows,ncols,2);AX.append(ax)
    c=ax.pcolor(hist[1],hist[2],true.T, norm=matplotlib.colors.LogNorm())
    ax=py.subplot(nrows,ncols,3);AX.append(ax)
    c=ax.pcolor(hist[1],hist[2],gen.T, norm=matplotlib.colors.LogNorm())
    for ax in AX:
        ax.tick_params(axis='both', which='major', labelsize=20,direction='in')
        ax.set_ylabel(r'$Q^2$',size=30)
        ax.set_xlabel(r'$x$',size=30)
        ax.set_xticks(np.log([1e-4,1e-3,1e-2,1e-1]))
        ax.set_xticklabels([r'$0.0001$',r'$0.001$',r'$0.01$',r'$0.1$'])
        ax.set_yticks(np.log([10,100,1000]))
        ax.set_yticklabels([r'$10$',r'$100$',r'$1000$']);
    AX[0].text(0.1,0.8,r'$\rm Reco$',transform=AX[0].transAxes,size=30)
    AX[1].text(0.1,0.8,r'$\rm True$',transform=AX[1].transAxes,size=30)
    AX[2].text(0.1,0.8,r'$\rm Gen$',transform=AX[2].transAxes,size=30)
    py.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        py.savefig(os.path.join(save_dir, 'PDF_2D_distribution_mceg_oldstyle.png'), dpi=300)
    
    nrows,ncols=1,3; AX=[]

    fig = py.figure(figsize=(ncols*6,nrows*5))
    cmap='gist_rainbow'

    ax=py.subplot(nrows,ncols,1);AX.append(ax)
    levels=10**np.linspace( np.log10(np.amin(reco[reco>0])),np.log10(np.amax(reco)),60)
    cs = ax.contour(hist[1][:-1],hist[2][:-1],reco.T,levels=levels,cmap=cmap,norm=colors.LogNorm())
    ax=py.subplot(nrows,ncols,2);AX.append(ax)
    cs = ax.contour(hist[1][:-1],hist[2][:-1],true.T,levels=levels,cmap=cmap,norm=colors.LogNorm())
    ax=py.subplot(nrows,ncols,3);AX.append(ax)
    cs = ax.contour(hist[1][:-1],hist[2][:-1],gen.T,levels=levels,cmap=cmap,norm=colors.LogNorm())
    for ax in AX:
        ax.tick_params(axis='both', which='major', labelsize=20,direction='in')
        ax.set_ylabel(r'$Q^2$',size=30)
        ax.set_xlabel(r'$x$',size=30)
        ax.set_xticks(np.log([1e-4,1e-3,1e-2,1e-1]))
        ax.set_xticklabels([r'$0.0001$',r'$0.001$',r'$0.01$',r'$0.1$'])
        ax.set_yticks(np.log([10,100,1000]))
        ax.set_yticklabels([r'$10$',r'$100$',r'$1000$']);
    AX[0].text(0.1,0.8,r'$\rm Reco$',transform=AX[0].transAxes,size=30)
    AX[1].text(0.1,0.8,r'$\rm True$',transform=AX[1].transAxes,size=30)
    AX[2].text(0.1,0.8,r'$\rm Gen$',transform=AX[2].transAxes,size=30)
    py.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        py.savefig(os.path.join(save_dir, 'PDF_2D_distribution_mceg_contour.png'), dpi=300)


    #HERE


#     # 2) Compute histogram once, vectorize the density (no per-bin loop needed)
#     hist = np.histogram2d(np.log(evts_reco[:,0]), np.log(evts_reco[:,1]), bins=(50,50))
#     H = hist[0]                    # (nx, nQ2)
#     logx_edges = hist[1]
#     logQ2_edges = hist[2]
#     dx  = np.diff(np.exp(logx_edges))    # (nx,)
#     dQ2 = np.diff(np.exp(logQ2_edges))    # (nQ2,)
#     # avoid division by zero
#     dx  = np.where(dx  > 0, dx,  np.nan)
#     dQ2 = np.where(dQ2 > 0, dQ2, np.nan)
#     reco = H / (dx[:,None] * dQ2[None,:])

#     # scale (guard sum==0)
#     Hsum = H.sum()
#     if Hsum > 0:
#         reco *= (mceg_true.total_xsec / Hsum) 

#     gen_hist = np.histogram2d(np.log(evts_pred[:,0]), np.log(evts_pred[:,1]), bins=(logx_edges, logQ2_edges))
#     gen = gen_hist[0]                    # (nx, nQ2)
#     gen_Hsum = gen.sum()
#     gen = gen / (dx[:,None] * dQ2[None,:])
#     if gen_Hsum > 0:
#         gen *= (mceg.total_xsec / gen_Hsum)
    

#     # 3) Fill "true" and "gen" safely; define x,Q2 for every bin center
#     true = np.zeros_like(reco, dtype=float)
#     xc   = np.exp(0.5*(logx_edges[:-1] + logx_edges[1:]))   # (nx,)
#     Q2c  = np.exp(0.5*(logQ2_edges[:-1] + logQ2_edges[1:])) # (nQ2,)

#     for i in range(len(xc)):
#         for j in range(len(Q2c)):
#             x  = float(xc[i]); Q2 = float(Q2c[j])
#             # Evaluate theory everywhere, but guard exceptions/negatives
#             try:
#                 tval, _ = idis_true.get_diff_xsec(x, Q2, mceg_true.rs, mceg_true.tar, 'xQ2')
#             except Exception:
#                 tval = np.nan
#             true[i,j] = tval if np.isfinite(tval) and tval >= 0 else np.nan

#     # 4) Safe levels + consistent LogNorm bounds
#     levels = safe_log_levels(reco, n=60)

#     # --- Pseudocolor plots (with explicit vmin/vmax) ---
#     fig = py.figure(figsize=(18,5)); AX=[]
#     ax=py.subplot(1,3,1); AX.append(ax)
#     c=ax.pcolor(logx_edges, logQ2_edges, np.where(reco>0, reco, np.nan).T,
#                 norm=colors.LogNorm())
#     ax=py.subplot(1,3,2); AX.append(ax)
#     c=ax.pcolor(logx_edges, logQ2_edges, np.where(true>0, true, np.nan).T,
#                 norm=colors.LogNorm())
#     ax=py.subplot(1,3,3); AX.append(ax)
#     c=ax.pcolor(logx_edges, logQ2_edges, np.where(gen>0, gen, np.nan).T,
#                norm=colors.LogNorm())

    
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         py.savefig(os.path.join(save_dir, 'PDF_2D_distribution_mceg_pcolor.png'), dpi=300)
# # --- Panels: Gen / Reco / True, styled like your target snippet ---
#     nrows, ncols = 1, 3
#     fig = py.figure(figsize=(ncols*6, nrows*5))
#     AX = []
#     cmap = 'gist_rainbow'

#     # Use your log-binned edges (swap to hist[1], hist[2] if that's what you actually have)
#     x_edges = logx_edges     # or: hist[1]
#     y_edges = logQ2_edges    # or: hist[2]

#     # Robust log-spaced levels from reco (fallback if helper isn't defined)
#     try:
#         levels = safe_log_levels(reco, n=60, lo_pct=1.0, hi_pct=99.0, default=(1e-6, 1.0))
#     except NameError:
#         reco_pos = reco[np.isfinite(reco) & (reco > 0)]
#         if reco_pos.size == 0:
#             # harmless default if reco has no positives
#             levels = 10.0 ** np.linspace(-6, 0, 60)
#         else:
#             vmin = np.percentile(reco_pos, 1.0)
#             vmax = np.percentile(reco_pos, 99.0)
#             vmin = max(vmin, 1e-12)
#             vmax = max(vmax, vmin * 10)
#             levels = 10.0 ** np.linspace(np.log10(vmin), np.log10(vmax), 60)

#     def _contour_panel(ax, Z, title):
#         Zp = np.where(np.isfinite(Z) & (Z > 0), Z, np.nan)
#         if np.isfinite(Zp).any():
#             cs = ax.contour(x_edges[:-1], y_edges[:-1], Zp.T,
#                             levels=levels, cmap=cmap, norm=colors.LogNorm())
#             ax.set_title(title, fontsize=18)
#             return cs
#         else:
#             ax.text(0.5, 0.5, f'No positive data for {title}',
#                     ha='center', va='center', transform=ax.transAxes)
#             return None

#     # Create panels (keep "Generated (Ours)")
#     ax = py.subplot(nrows, ncols, 1); AX.append(ax); cs_gen  = _contour_panel(ax, gen,  'Generated (Ours)')
#     ax = py.subplot(nrows, ncols, 2); AX.append(ax); cs_reco = _contour_panel(ax, reco, 'Reco')
#     ax = py.subplot(nrows, ncols, 3); AX.append(ax); cs_true = _contour_panel(ax, true, 'True')

#     # Shared styling like your example
#     for ax in AX:
#         ax.tick_params(axis='both', which='major', labelsize=20, direction='in')
#         ax.set_xlabel(r'$x$',  size=30)
#         ax.set_ylabel(r'$Q^2$', size=30)
#         ax.set_xticks(np.log([1e-4, 1e-3, 1e-2, 1e-1]))
#         ax.set_xticklabels([r'$0.0001$', r'$0.001$', r'$0.01$', r'$0.1$'])
#         ax.set_yticks(np.log([10, 100, 1000]))
#         ax.set_yticklabels([r'$10$', r'$100$', r'$1000$'])

#     # One colorbar for whichever panel rendered last successfully
#     for cs in (cs_true, cs_reco, cs_gen):
#         if cs is not None:
#             cbar = fig.colorbar(cs, ax=AX, fraction=0.02, pad=0.02)
#             cbar.ax.tick_params(labelsize=16)
#             break

#     py.tight_layout()
#     # 6) Shared cosmetics
#     for ax in AX:
#         ax.tick_params(axis='both', which='major', labelsize=20, direction='in')
#         ax.set_ylabel(r'$Q^2$', size=30)
#         ax.set_xlabel(r'$x$', size=30)
#         ax.set_xticks(np.log([1e-4,1e-3,1e-2,1e-1]))
#         ax.set_xticklabels([r'$0.0001$',r'$0.001$',r'$0.01$',r'$0.1$'])
#         ax.set_yticks(np.log([10,100,1000]))
#         ax.set_yticklabels([r'$10$',r'$100$',r'$1000$'])
#     py.tight_layout()
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         py.savefig(os.path.join(save_dir, 'PDF_2D_distribution_mceg.png'), dpi=300)



    # HERE




    # H_reco, xedges, q2edges = np.histogram2d(np.log(evts_reco[:,0]), np.log(evts_reco[:,1]),
    #                                         bins=(logx_edges, logQ2_edges))
    # H_reco*=mceg.total_xsec/np.sum(H_reco)  # scale to total xsec
    # true_density = _theory_grid_masked(idis_true, xedges, q2edges, mceg_true.rs, mceg_true.tar,
    #                                 'xQ2', occupancy_counts=H_reco.astype(int))

    # # Predicted θ̂ events (right)
    # pred_density, _ = _hist2d_density_log(
    #     evts_pred, logx_edges, logQ2_edges,
    #     total_xsec=mceg_pred.total_xsec if 'mceg_pred' in globals() else None
    # )

    # # ---------- Plot: top row pcolor, bottom row contour ----------
    # # Order: True (left), Reco (middle), Pred (right)
    # panels = [
    #     ("True", true_density),
    #     ("Reco", reco_density),
    #     ("Pred", pred_density),
    # ]

    # # Compute shared contour levels (log-spaced) over all three, ignoring zeros
    # all_vals = np.concatenate([p[1].ravel() for p in panels])
    # all_vals = all_vals[all_vals > 0]
    # vmin = np.percentile(all_vals, 5) if all_vals.size else 1e-20
    # vmax = np.percentile(all_vals, 99.5) if all_vals.size else 1.0
    # # levels = np.geomspace(max(vmin, 1e-30), vmax, 12)
    # levels=10**np.linspace( np.log10(np.amin(H_reco[H_reco>0])),np.log10(np.amax(H_reco)),60)

    # fig = plt.figure(figsize=(18, 10))
    # AX = []
    # for col, (title, D) in enumerate(panels, start=1):
    #     # Top: heatmap
    #     ax = plt.subplot(2, 3, col); AX.append(ax)
    #     c = ax.pcolor(xedges, q2edges, D.T, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    #     ax.set_title(title, fontsize=18)
    #     # Bottom: contours
    #     ax2 = plt.subplot(2, 3, 3+col); AX.append(ax2)
    #     cs = ax2.contour(xedges[:-1], q2edges[:-1], D.T, levels=levels, norm=matplotlib.colors.LogNorm())
    #     ax2.clabel(cs, inline=True, fontsize=8)
    #     ax2.set_title(f"{title} (contours)", fontsize=16)

    # # Shared axis cosmetics
    # for ax in AX:
    #     ax.tick_params(axis='both', which='major', labelsize=12, direction='in')
    #     ax.set_xlabel(r'$x$', size=14)
    #     ax.set_ylabel(r'$Q^2$', size=14)
    #     ax.set_xticks(np.log([1e-4, 1e-3, 1e-2, 1e-1]))
    #     ax.set_xticklabels([r'$0.0001$', r'$0.001$', r'$0.01$', r'$0.1$'])
    #     ax.set_yticks(np.log([10, 100, 1000]))
    #     ax.set_yticklabels([r'$10$', r'$100$', r'$1000$'])

    # # Colorbar for the heatmaps (top row)
    # cbar_ax = fig.add_axes([0.92, 0.56, 0.015, 0.32])
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    #                                         cmap=plt.get_cmap()),
    #             cax=cbar_ax, label=r'd$\sigma$/d$x$d$Q^2$')
    # plt.tight_layout(rect=[0,0,0.9,1])
    # plt.show()
    # plt.savefig(save_path, dpi=300)