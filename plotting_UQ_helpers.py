"""Small helper utilities for function-space UQ on the simplified_dis problem.

This module provides two helpers:
- collect_predicted_pdfs_simplified_dis(param_samples, device)
- save_function_UQ_metrics_table_simplified_dis(save_path, true_params, device, results_dict)

These are intentionally isolated from the larger plotting_UQ_utils module so they
can be imported without pulling in heavy plotting code or affected by editing churn
in that file.
"""

import numpy as np
import torch


def collect_predicted_pdfs_simplified_dis(param_samples, device):
    """
    For a batch or iterable of parameter samples, return (pdfs_up, pdfs_down).

    Inputs:
        param_samples: torch.Tensor of shape [N, param_dim] or iterable of parameter tensors
        device: torch.device or string

    Returns:
        pdfs_up: np.ndarray of shape [N, n_x]
        pdfs_down: np.ndarray of shape [N, n_x]
    """
    try:
        from simulator import SimplifiedDIS
    except Exception as e:
        raise ImportError(f"Could not import SimplifiedDIS from simulator: {e}")

    simulator = SimplifiedDIS(device=device)
    x_grid = torch.linspace(0.01, 0.99, 100).to(device)

    pdfs_up = []
    pdfs_down = []

    # Accept either a Tensor (N, D) or an iterable of parameter tensors
    if isinstance(param_samples, torch.Tensor):
        iterator = (param_samples[i] for i in range(param_samples.shape[0]))
    else:
        iterator = iter(param_samples)

    for theta in iterator:
        theta = theta.to(device)
        pdf = simulator.f(x_grid, theta)
        up_arr = pdf["up"].cpu().numpy()
        down_arr = pdf["down"].cpu().numpy()
        # If the simulator returned non-finite entries, try to salvage the sample by
        # filling NaNs using 1D interpolation. Only drop the sample if it is entirely non-finite.
        up_finite = np.isfinite(up_arr)
        down_finite = np.isfinite(down_arr)

        def fill_1d_nans(arr, finite_mask):
            if finite_mask.all():
                return arr
            if not finite_mask.any():
                return None
            # linear interpolation over indices
            inds = np.arange(arr.size)
            good = inds[finite_mask]
            vals = arr[finite_mask]
            try:
                filled = np.interp(inds, good, vals)
                return filled
            except Exception:
                # fallback: replace NaNs with nearest finite value
                filled = arr.copy()
                finite_vals = vals
                if finite_vals.size > 0:
                    filled[~finite_mask] = finite_vals.mean()
                    return filled
                return None

        up_filled = fill_1d_nans(up_arr, up_finite)
        down_filled = fill_1d_nans(down_arr, down_finite)

        if up_filled is None or down_filled is None:
            # Drop this sample only if either curve is entirely non-finite
            continue

        # Keep salvaged arrays
        pdfs_up.append(up_filled)
        pdfs_down.append(down_filled)

    pdfs_up = np.array(pdfs_up)
    pdfs_down = np.array(pdfs_down)
    # Convert to arrays
    pdfs_up = np.array(pdfs_up)
    pdfs_down = np.array(pdfs_down)
    # If we dropped all samples due to non-finite outputs, return empty arrays so callers can handle
    if pdfs_up.size == 0 or pdfs_down.size == 0:
        # Log helpful diagnostic
        try:
            print(
                f"[collect_predicted_pdfs] Warning: no valid predicted PDFs (all samples dropped) for device={device}"
            )
        except Exception:
            pass
        return np.array([]), np.array([])
    return pdfs_up, pdfs_down


def compute_function_lotv_for_simplified_dis(
    simulator,
    x_grid,
    per_boot_posterior_samples=None,
    posterior_sampler_callable=None,
    n_theta_per_boot=20,
    device="cpu",
):
    """
    Compute law-of-total-variance decomposition for function outputs f(x) for the simplified_dis simulator.

    Inputs (one of the following must be provided):
      - per_boot_posterior_samples: list of numpy arrays, each array shape (n_theta_b, n_params)
        representing posterior samples conditional on bootstrap b.
      - posterior_sampler_callable: a callable with signature sampler(bootstrap_index) -> np.ndarray
        returning an array shape (n_theta_per_boot, n_params) for the given bootstrap index.

    Args:
      simulator: an instance of SimplifiedDIS with method f(x_grid, params_tensor) -> dict('up','down')
      x_grid: 1D torch tensor on device of x evaluation points
      n_theta_per_boot: number of posterior samples to draw per bootstrap when using callable
      device: device string or torch.device

    Returns:
      dict with keys:
        'mean_up', 'mean_down' : mean function across bootstraps (E_b[ E_{θ|b}[ f(x) ] ])
        'avg_within_var_up', 'between_var_up', 'total_var_up' : arrays over x
        'avg_within_var_down', ... : same for down
        'unc_up', 'unc_down' : scalar summaries (mean_x sqrt(total_var(x)))
    """
    import numpy as np
    import torch

    # Collect per-bootstrap posterior samples
    per_boot_samples = []
    if per_boot_posterior_samples is not None:
        # assume list-like of arrays
        for arr in per_boot_posterior_samples:
            per_boot_samples.append(np.array(arr))
    elif posterior_sampler_callable is not None:
        # caller will supply how many bootstraps to generate via a special attribute
        # We attempt to draw until the callable raises or returns None; safer callers should pass list.
        b = 0
        while True:
            samples = posterior_sampler_callable(b)
            if samples is None:
                break
            samples = np.array(samples)
            if samples.size == 0:
                break
            per_boot_samples.append(samples)
            b += 1
    else:
        raise ValueError(
            "Either per_boot_posterior_samples or posterior_sampler_callable must be provided"
        )

    n_boot = len(per_boot_samples)
    if n_boot == 0:
        raise ValueError("No bootstrap posterior samples provided")

    x_grid_t = (
        x_grid.to(device)
        if isinstance(x_grid, torch.Tensor)
        else torch.tensor(x_grid, device=device)
    )
    n_x = x_grid_t.numel()

    # For each bootstrap b, evaluate f(x) across posterior samples
    boot_means_up = []
    boot_within_vars_up = []
    boot_means_down = []
    boot_within_vars_down = []

    for samples in per_boot_samples:
        # samples shape (n_theta_b, n_params)
        n_theta_b = samples.shape[0]
        # evaluate f for each theta
        f_up_ens = np.zeros((n_theta_b, n_x), dtype=float)
        f_down_ens = np.zeros((n_theta_b, n_x), dtype=float)
        for i in range(n_theta_b):
            theta = torch.tensor(samples[i], dtype=torch.float32, device=device)
            pdf = simulator.f(x_grid_t, theta)
            up = pdf["up"].detach().cpu().numpy().ravel()
            down = pdf["down"].detach().cpu().numpy().ravel()
            f_up_ens[i, :] = up
            f_down_ens[i, :] = down

        # per-boot mean and within-variance
        # Use nan-aware aggregations in case some theta evaluations returned NaNs
        mu_b_up = np.nanmean(f_up_ens, axis=0)
        var_b_up = np.nanvar(f_up_ens, axis=0, ddof=0)
        mu_b_down = np.nanmean(f_down_ens, axis=0)
        var_b_down = np.nanvar(f_down_ens, axis=0, ddof=0)

        boot_means_up.append(mu_b_up)
        boot_within_vars_up.append(var_b_up)
        boot_means_down.append(mu_b_down)
        boot_within_vars_down.append(var_b_down)

    # Stack into arrays; if any list is empty or contains uneven shapes, handle gracefully
    try:
        boot_means_up = np.stack(boot_means_up, axis=0)  # (B, n_x)
        boot_within_vars_up = np.stack(boot_within_vars_up, axis=0)
        boot_means_down = np.stack(boot_means_down, axis=0)
        boot_within_vars_down = np.stack(boot_within_vars_down, axis=0)
    except Exception:
        # Fall back to converting to arrays with dtype=float and allowing NaNs where lengths mismatch
        boot_means_up = np.asarray(boot_means_up, dtype=float)
        boot_within_vars_up = np.asarray(boot_within_vars_up, dtype=float)
        boot_means_down = np.asarray(boot_means_down, dtype=float)
        boot_within_vars_down = np.asarray(boot_within_vars_down, dtype=float)

    # Compute avg-within and between variance
    # Use nan-aware statistics for the LoTV decomposition
    avg_within_var_up = np.nanmean(boot_within_vars_up, axis=0)
    between_var_up = np.nanvar(boot_means_up, axis=0, ddof=0)
    total_var_up = avg_within_var_up + between_var_up

    avg_within_var_down = np.nanmean(boot_within_vars_down, axis=0)
    between_var_down = np.nanvar(boot_means_down, axis=0, ddof=0)
    total_var_down = avg_within_var_down + between_var_down

    mean_up = np.nanmean(boot_means_up, axis=0)
    mean_down = np.nanmean(boot_means_down, axis=0)

    # scalar summaries: mean over x of sqrt(total_var(x))
    unc_up = float(np.mean(np.sqrt(total_var_up)))
    unc_down = float(np.mean(np.sqrt(total_var_down)))

    return {
        "mean_up": mean_up,
        "avg_within_var_up": avg_within_var_up,
        "between_var_up": between_var_up,
        "total_var_up": total_var_up,
        "mean_down": mean_down,
        "avg_within_var_down": avg_within_var_down,
        "between_var_down": between_var_down,
        "total_var_down": total_var_down,
        "unc_up": unc_up,
        "unc_down": unc_down,
    }


def compute_function_lotv_for_mceg(
    simulator,
    per_boot_posterior_samples=None,
    posterior_sampler_callable=None,
    n_theta_per_boot=20,
    num_events=20000,
    nx=30,
    nQ2=20,
    device="cpu",
):
    """
    Compute law-of-total-variance decomposition for binned MCEG-style histograms.

    Inputs (one of the following must be provided):
      - per_boot_posterior_samples: list of numpy arrays, each array shape (n_theta_b, n_params)
      - posterior_sampler_callable: callable which when given a bootstrap index returns an
        array shape (n_theta_per_boot, n_params). If used, the callable should raise or
        return None to stop.

    The function will, for each posterior sample theta, simulate `num_events` from the
    provided `simulator` and build a 2D reco histogram (nx x nQ2) in log(x), log(Q2) space
    matching the logic used across the codebase. It then computes per-bootstrap within-\
    posterior variance and between-bootstrap variance and returns the decomposition arrays.

    Returns dict with keys:
      'mean': (nx, nQ2) mean histogram across bootstraps of posterior-conditional means
      'avg_within_var': (nx, nQ2) average within-bootstrap variance
      'between_var': (nx, nQ2) variance of per-boot means
      'total_var': (nx, nQ2) sum of avg_within_var + between_var
      'unc_scalar': scalar summary mean_x sqrt(total_var)
    """
    import numpy as np
    import torch
    from tqdm import tqdm

    # Collect per-bootstrap posterior samples
    per_boot = []
    if per_boot_posterior_samples is not None:
        for arr in per_boot_posterior_samples:
            per_boot.append(np.array(arr))
    elif posterior_sampler_callable is not None:
        b = 0
        while True:
            samples = posterior_sampler_callable(b)
            if samples is None:
                break
            samples = np.array(samples)
            if samples.size == 0:
                break
            per_boot.append(samples)
            b += 1
    else:
        raise ValueError(
            "Either per_boot_posterior_samples or posterior_sampler_callable must be provided"
        )

    if len(per_boot) == 0:
        raise ValueError("No bootstrap posterior samples provided")

    # Helper to build reco histogram from events (match logic from plotting_UQ_utils)
    def evts_to_np(evts):
        if isinstance(evts, torch.Tensor):
            return evts.detach().cpu().numpy()
        return np.asarray(evts)

    def get_reco(evts):
        evts_np = evts_to_np(evts)
        if evts_np.size == 0:
            hist = np.histogram2d(np.array([]), np.array([]), bins=(nx, nQ2))
            return np.zeros(hist[0].shape)
        # filter out non-finite or non-positive entries (avoid log of <=0)
        mask = (
            np.isfinite(evts_np).all(axis=1) & (evts_np[:, 0] > 0) & (evts_np[:, 1] > 0)
        )
        evts_np = evts_np[mask]
        if evts_np.size == 0:
            hist = np.histogram2d(np.array([]), np.array([]), bins=(nx, nQ2))
            return np.zeros(hist[0].shape)
        log_x = np.log(evts_np[:, 0])
        log_Q2 = np.log(evts_np[:, 1])
        hist = np.histogram2d(log_x, log_Q2, bins=(nx, nQ2))
        counts = hist[0].astype(float)
        reco = np.zeros_like(counts)
        x_edges = hist[1]
        Q2_edges = hist[2]
        for i in range(x_edges.shape[0] - 1):
            for j in range(Q2_edges.shape[0] - 1):
                c = counts[i, j]
                if c > 0:
                    xmin = np.exp(x_edges[i])
                    xmax = np.exp(x_edges[i + 1])
                    Q2min = np.exp(Q2_edges[j])
                    Q2max = np.exp(Q2_edges[j + 1])
                    dx = max(xmax - xmin, 1e-300)
                    dQ2 = max(Q2max - Q2min, 1e-300)
                    area = dx * dQ2
                    reco[i, j] = c / area
        if np.sum(counts) > 0:
            try:
                scale = float(simulator.mceg.total_xsec) / np.sum(counts)
            except Exception:
                scale = 1.0
            reco *= scale
        # ensure finite output
        reco = np.where(np.isfinite(reco), reco, 0.0)
        return reco

    boot_means = []
    boot_withins = []

    # For each bootstrap
    for samples in per_boot:
        n_theta = int(samples.shape[0])
        # evaluate reco hist for each theta
        ens = np.zeros((n_theta, nx, nQ2), dtype=float)
        for i in range(n_theta):
            theta = samples[i]
            # ensure simulator initialized with theta
            try:
                simulator.init(theta)
            except Exception:
                simulator.init(np.asarray(theta))
            evts = simulator.sample(theta, num_events)
            ens[i] = get_reco(evts.cpu() if hasattr(evts, "cpu") else evts)

        mu_b = np.mean(ens, axis=0)
        var_b = np.var(ens, axis=0, ddof=0)
        boot_means.append(mu_b)
        boot_withins.append(var_b)

    boot_means = np.stack(boot_means, axis=0)  # (B, nx, nQ2)
    boot_withins = np.stack(boot_withins, axis=0)

    avg_within = np.mean(boot_withins, axis=0)
    between = np.var(boot_means, axis=0, ddof=0)
    total = avg_within + between
    mean = np.mean(boot_means, axis=0)

    # scalar summary: mean over bins of sqrt(total_var)
    unc_scalar = float(np.mean(np.sqrt(total)))

    return {
        "mean": mean,
        "avg_within_var": avg_within,
        "between_var": between,
        "total_var": total,
        "unc_scalar": unc_scalar,
        # return per-bootstrap summaries so callers can compute bootstrap-level metrics
        "boot_means": boot_means,
        "boot_withins": boot_withins,
    }


def save_function_UQ_metrics_table_simplified_dis(
    save_path, true_params, device, results_dict, aggregation="mean"
):
    """
    Compute simple function-space metrics comparing predicted pdf ensembles to the
    ground truth and save a LaTeX-compatible tabular block to `save_path`.

    Arguments:
        save_path: str path to write the .tex/.txt file
        true_params: torch.Tensor (param_dim,) containing true theta
        device: torch.device or string
        results_dict: dict mapping approach name -> {"pdfs_up": array_like, "pdfs_down": array_like}
                      where pdf arrays have shape [N, n_x]
    """
    try:
        from simulator import SimplifiedDIS
    except Exception as e:
        raise ImportError(f"Could not import SimplifiedDIS from simulator: {e}")

    simulator = SimplifiedDIS(device=device)
    x_grid = torch.linspace(0.01, 0.99, 100).to(device)
    true_pdf = simulator.f(x_grid, true_params.to(device))
    true_up = true_pdf["up"].cpu().numpy()
    true_down = true_pdf["down"].cpu().numpy()

    rows = []
    # First compute rows for each provided approach (Laplace, Bootstrap, SBI etc.)
    for approach, vals in results_dict.items():
        pdfs_up = np.array(vals.get("pdfs_up", []))
        pdfs_down = np.array(vals.get("pdfs_down", []))
        # If the entry provides full ensembles (pdfs_up/pdfds_down), compute metrics from ensembles
        # Use the chosen aggregation (median|mean) to produce a central curve and report MSE/bias
        if pdfs_up.size > 0 and pdfs_down.size > 0:
            try:
                if aggregation == "median":
                    central_up = np.median(pdfs_up, axis=0)
                    central_down = np.median(pdfs_down, axis=0)
                else:
                    central_up = np.mean(pdfs_up, axis=0)
                    central_down = np.mean(pdfs_down, axis=0)

                up_mse = np.mean((central_up - true_up) ** 2)
                up_bias = np.mean(central_up - true_up)
                up_unc = np.mean(np.std(pdfs_up, axis=0))
                down_mse = np.mean((central_down - true_down) ** 2)
                down_bias = np.mean(central_down - true_down)
                down_unc = np.mean(np.std(pdfs_down, axis=0))
                rows.append(
                    [
                        approach,
                        f"{up_mse:.4g}",
                        f"{up_bias:.4g}",
                        f"{up_unc:.4g}",
                        f"{down_mse:.4g}",
                        f"{down_bias:.4g}",
                        f"{down_unc:.4g}",
                    ]
                )
                continue
            except Exception:
                # fallback to previous conservative computation if aggregation fails
                up_mse = np.mean((pdfs_up - true_up[None, :]) ** 2)
                up_bias = np.mean(pdfs_up - true_up[None, :])
                up_unc = np.mean(np.std(pdfs_up, axis=0))
                down_mse = np.mean((pdfs_down - true_down[None, :]) ** 2)
                down_bias = np.mean(pdfs_down - true_down[None, :])
                down_unc = np.mean(np.std(pdfs_down, axis=0))
                rows.append(
                    [
                        approach,
                        f"{up_mse:.4g}",
                        f"{up_bias:.4g}",
                        f"{up_unc:.4g}",
                        f"{down_mse:.4g}",
                        f"{down_bias:.4g}",
                        f"{down_unc:.4g}",
                    ]
                )
                continue

        # Otherwise, allow entries that provide mean+unc directly (decomposition output)
        mean_up = vals.get("mean_up", None)
        mean_down = vals.get("mean_down", None)
        unc_up = vals.get("unc_up", None)
        unc_down = vals.get("unc_down", None)
        if (
            mean_up is not None
            and mean_down is not None
            and unc_up is not None
            and unc_down is not None
        ):
            # mean_* expected as 1D arrays over x; unc_* as scalar summary (e.g., mean_x sqrt(total_var(x)))
            mean_up = np.array(mean_up)
            mean_down = np.array(mean_down)
            up_mse = np.mean((mean_up - true_up) ** 2)
            up_bias = np.mean(mean_up - true_up)
            up_unc = float(unc_up)
            down_mse = np.mean((mean_down - true_down) ** 2)
            down_bias = np.mean(mean_down - true_down)
            down_unc = float(unc_down)
            rows.append(
                [
                    approach,
                    f"{up_mse:.4g}",
                    f"{up_bias:.4g}",
                    f"{up_unc:.4g}",
                    f"{down_mse:.4g}",
                    f"{down_bias:.4g}",
                    f"{down_unc:.4g}",
                ]
            )
            continue

        # If neither ensembles nor mean+unc provided, write placeholder
        rows.append([approach, "--", "--", "--", "--", "--", "--"])

    # Now add a Combined row that matches `plot_function_uncertainty(..., mode='combined')`
    # i.e. pool (concatenate) the posterior/Laplace ensemble and the bootstrap ensemble
    # and compute metrics on the pooled set. Avoid duplicating the Combined row if already present.
    existing_approaches = [r[0] for r in rows]
    if "Combined" not in existing_approaches:
        laplace = results_dict.get("Laplace")
        bootstrap = results_dict.get("Bootstrap")
        if laplace is not None and bootstrap is not None:
            lap_up = np.array(laplace.get("pdfs_up", []))
            lap_down = np.array(laplace.get("pdfs_down", []))
            boot_up = np.array(bootstrap.get("pdfs_up", []))
            boot_down = np.array(bootstrap.get("pdfs_down", []))
            if (
                lap_up.size > 0
                and lap_down.size > 0
                and boot_up.size > 0
                and boot_down.size > 0
            ):
                pooled_up = np.concatenate([lap_up, boot_up], axis=0)
                pooled_down = np.concatenate([lap_down, boot_down], axis=0)

                # central pooled curve follows aggregation choice
                if aggregation == "median":
                    pooled_up_mean = np.median(pooled_up, axis=0)
                    pooled_down_mean = np.median(pooled_down, axis=0)
                else:
                    pooled_up_mean = np.mean(pooled_up, axis=0)
                    pooled_down_mean = np.mean(pooled_down, axis=0)

                pooled_up_unc = np.mean(np.std(pooled_up, axis=0))
                pooled_down_unc = np.mean(np.std(pooled_down, axis=0))

                pooled_up_mse = np.mean((pooled_up_mean - true_up) ** 2)
                pooled_up_bias = np.mean(pooled_up_mean - true_up)
                pooled_down_mse = np.mean((pooled_down_mean - true_down) ** 2)
                pooled_down_bias = np.mean(pooled_down_mean - true_down)

                rows.append(
                    [
                        "Combined",
                        f"{pooled_up_mse:.4g}",
                        f"{pooled_up_bias:.4g}",
                        f"{pooled_up_unc:.4g}",
                        f"{pooled_down_mse:.4g}",
                        f"{pooled_down_bias:.4g}",
                        f"{pooled_down_unc:.4g}",
                    ]
                )
            else:
                rows.append(["Combined", "--", "--", "--", "--", "--", "--"])
        else:
            rows.append(["Combined", "--", "--", "--", "--", "--", "--"])

    header = [
        "Approach",
        "Up MSE",
        "Up Bias",
        "Up Unc.",
        "Down MSE",
        "Down Bias",
        "Down Unc.",
    ]

    latex_lines = []
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\toprule")
    header_line = " & ".join(header) + " \\\\"
    latex_lines.append(header_line)
    latex_lines.append("\\midrule")
    for row in rows:
        latex_lines.append(" & ".join(row) + " \\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")

    with open(save_path, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"\u2713 Saved function UQ metrics table to {save_path}")
