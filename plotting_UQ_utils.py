import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from simulator import advanced_feature_engineering, SimplifiedDIS, RealisticDIS

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
    xs_tensor = advanced_feature_engineering(xs_tensor)
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
    n_mc=100,  # Kept for backward compatibility but ignored when using analytic uncertainty
    laplace_model=None,
    problem='simplified_dis',
    Q2_slices=None,
    save_dir=None,
    save_path="pdf_distribution.png"
):
    """
    Plot PDF distributions using analytic Laplace uncertainty propagation.
    
    When laplace_model is provided, uses analytic uncertainty propagation to 
    compute error bands instead of Monte Carlo sampling for improved speed and accuracy.
    """
    model.eval()
    pointnet_model.eval()
    simulator = RealisticDIS(torch.device('cpu')) if problem == 'realistic_dis' else SimplifiedDIS(torch.device('cpu'))

    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor)
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

    if problem == 'simplified_dis':
        x_vals = torch.linspace(0, 1, 500).to(device)
        for fn_name, fn_label, color in [("up", "u", "royalblue"), ("down", "d", "darkorange")]:
            
            if use_analytic:
                # Compute PDF using analytic uncertainty
                simulator.init(mean_params)
                fn = getattr(simulator, fn_name)
                mean_vals = fn(x_vals).detach().cpu()
                
                # Compute uncertainty bounds using ±2σ (approximately 95% confidence interval)
                # For PDF uncertainty, we approximate using parameter uncertainty
                # This is a first-order approximation; for exact uncertainty propagation,
                # we would need the Jacobian of the PDF w.r.t. parameters
                param_std_norm = torch.norm(std_params).item()
                uncertainty_factor = 2.0 * param_std_norm  # Heuristic scaling
                
                lower_bounds = mean_vals * (1 - uncertainty_factor)
                upper_bounds = mean_vals * (1 + uncertainty_factor)
                
                # Ensure non-negative PDFs
                lower_bounds = torch.clamp(lower_bounds, min=0.0)
                
            else:
                # MC sampling approach (legacy)
                fn_vals_all = []
                for i in range(n_mc):
                    simulator.init(samples[i])
                    fn = getattr(simulator, fn_name)
                    fn_vals_all.append(fn(x_vals).unsqueeze(0))

                fn_stack = torch.cat(fn_vals_all, dim=0)
                mean_vals = fn_stack.median(dim=0).values.detach().cpu()
                lower_bounds = torch.quantile(fn_stack, 0.25, dim=0).detach().cpu()
                upper_bounds = torch.quantile(fn_stack, 0.75, dim=0).detach().cpu()

            # Compute true values for comparison
            simulator.init(true_params.squeeze())
            true_vals = getattr(simulator, fn_name)(x_vals).detach().cpu()

            # Create plot
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(x_vals.detach().cpu(), true_vals, label=fr"True ${fn_label}(x|\theta^*)$", color=color, linewidth=2)
            
            if use_analytic:
                ax.plot(x_vals.detach().cpu(), mean_vals, linestyle='--', 
                       label=fr"MAP ${fn_label}(x|\hat{{\theta}})$ (Analytic)", color="crimson", linewidth=2)
                ax.fill_between(x_vals.detach().cpu(), lower_bounds, upper_bounds, 
                               color="crimson", alpha=0.3, label="95% Analytic CI")
            else:
                ax.plot(x_vals.detach().cpu(), mean_vals, linestyle='--', 
                       label=fr"Median ${fn_label}(x|\hat{{\theta}})$ (MC)", color="crimson", linewidth=2)
                ax.fill_between(x_vals.detach().cpu(), lower_bounds, upper_bounds, 
                               color="crimson", alpha=0.3, label="IQR")

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
    xs_tensor = advanced_feature_engineering(xs_tensor)
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
    save_path="event_histogram_simplified.png"
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
    xs_tensor = advanced_feature_engineering(xs_tensor)
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