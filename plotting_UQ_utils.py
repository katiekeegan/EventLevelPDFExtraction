import torch
import matplotlib.pyplot as plt
import numpy as np
from simulator import advanced_feature_engineering, SimplifiedDIS, RealisticDIS

def get_gaussian_samples(model, latent_embedding, n_samples=100, laplace_model=None):
    """
    Samples output parameters from the model.
    If laplace_model is provided, samples weights from Laplace posterior before forward pass.
    Otherwise, just uses the model's output head (as before).
    Compatible with MLP, Transformer, Gaussian, Multimodal heads.
    """
    device = latent_embedding.device
    samples = []

    for _ in range(n_samples):
        if laplace_model is not None:
            # Laplace: sample weights for each iteration
            lap_model = laplace_model.sample()
            lap_model = lap_model.to(device)
            lap_model.eval()
            with torch.no_grad():
                output = lap_model(latent_embedding)
        else:
            # Standard model: just forward
            model.eval()
            with torch.no_grad():
                output = model(latent_embedding)

        if isinstance(output, tuple) and len(output) == 2:  # GaussianHead
            means, logvars = output
            stds = torch.exp(0.5 * logvars)
            param_sample = torch.randn_like(means) * stds + means
            samples.append(param_sample.squeeze(0))  # Squeeze batch dim
        elif isinstance(output, tuple) and len(output) == 3:  # MultimodalHead
            means, logvars, weights = output
            means = means.squeeze(0)
            logvars = logvars.squeeze(0)
            weights = weights.squeeze(0)
            nmodes, param_dim = means.shape
            stds = torch.exp(0.5 * logvars)
            mode = torch.multinomial(weights, 1).item()
            param_sample = torch.randn(param_dim, device=device) * stds[mode] + means[mode]
            samples.append(param_sample)
        else:
            # MLP/Transformer deterministic output
            samples.append(output.squeeze(0))
    return torch.stack(samples)

def plot_params_distribution_single(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,
    laplace_model=None,
    compare_with_sbi=False,
    sbi_posteriors=None,
    sbi_labels=None,
    save_path="Dist.png",
    problem='simplified_dis'
):
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

    samples = get_gaussian_samples(model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model).cpu()

    all_samples = [samples]
    if compare_with_sbi and sbi_posteriors is not None:
        all_samples.extend([s.detach().cpu() for s in sbi_posteriors])

    n_params = true_params.size(0)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    colors = ['skyblue', 'orange', 'green', 'purple', 'gray']
    param_names = [r'$a_u$', r'$b_u$', r'$a_d$', r'$b_d$'] if problem == 'simplified_dis' else [r'$\log A_0$', r'$\delta$', r'$a$', r'$b$', r'$c$', r'$d$']

    for i in range(n_params):
        param_vals = [s[:, i].numpy() for s in all_samples]
        xmin = min([v.min() for v in param_vals])
        xmax = max([v.max() for v in param_vals])
        padding = 0.05 * (xmax - xmin)
        xmin -= padding
        xmax += padding

        axes[i].hist(samples[:, i].numpy(), bins=20, alpha=0.6, density=True, color=colors[0], label='Posterior Samples')
        if compare_with_sbi and sbi_posteriors is not None and sbi_labels is not None:
            for j, sbi_samples in enumerate(sbi_posteriors):
                label = sbi_labels[j] if j < len(sbi_labels) else f"SBI {j}"
                axes[i].hist(
                    sbi_samples[:, i].detach().cpu().numpy(),
                    bins=20, alpha=0.4, density=True,
                    color=colors[(j + 1) % len(colors)],
                    label=label
                )
        axes[i].axvline(true_params[i].item(), color='red', linestyle='dashed', label='True Value')
        axes[i].set_xlim(xmin, xmax)
        axes[i].set_title(f'{param_names[i]}')
        if i == 0: 
            axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_PDF_distribution_single(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,
    laplace_model=None,
    problem='simplified_dis',
    Q2_slices=None,
    save_dir=None,
    save_path="pdf_distribution.png"
):
    model.eval()
    pointnet_model.eval()
    simulator = RealisticDIS(torch.device('cpu')) if problem == 'realistic_dis' else SimplifiedDIS(torch.device('cpu'))

    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    samples = get_gaussian_samples(model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model).cpu()

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
    laplace_model=None,
    problem='simplified_dis',
    Q2_slices=None,
    plot_IQR=False,
    save_path="pdf_overlay.png"
):
    model.eval()
    pointnet_model.eval()
    simulator = RealisticDIS(torch.device('cpu')) if problem == 'realistic_dis' else SimplifiedDIS(torch.device('cpu'))

    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    samples = get_gaussian_samples(model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model).cpu()

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

def plot_event_histogram_simplified_DIS(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,
    laplace_model=None,             # <-- ADD THIS ARGUMENT
    num_events=100000,
    save_path="event_histogram_simplified.png"
):
    model.eval()
    pointnet_model.eval()
    simulator = SimplifiedDIS(torch.device('cpu'))
    true_params = true_params.to(device)
    xs = simulator.sample(true_params.detach().cpu(), num_events).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # Use uncertainty-aware sampling
    samples = get_gaussian_samples(model, latent_embedding, n_samples=n_mc, laplace_model=laplace_model).cpu()
    mode_params = torch.median(samples, dim=0).values
    generated_events = simulator.sample(mode_params.detach().cpu(), num_events).to(device)
    true_events_np = xs.detach().cpu().numpy()
    generated_events_np = generated_events.detach().cpu().numpy()
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