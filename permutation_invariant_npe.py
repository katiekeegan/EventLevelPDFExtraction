"""Permutation-invariant NPE pipelines for DIS simulators (GPU-friendly, chunked).

Key changes vs your version:
- Separate simulation_device (where simulator runs) and training_device (where NN trains).
- Stream simulations into SBI in chunks to avoid giant (num_thetas, num_events, x_dim) tensors.
- Keep stored simulations on CPU by default to avoid VRAM blowups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from sbi.utils.torchutils import BoxUniform

from simulator import MCEGSimulator, SimplifiedDIS


@dataclass
class PermutationInvariantConfig:
    posterior: object
    prior: BoxUniform
    simulator: Callable[[torch.Tensor, int], torch.Tensor]
    num_events: int
    x_dim: int
    training_device: torch.device
    simulation_device: torch.device


def ensure_num_events(x: torch.Tensor, num_events: int) -> torch.Tensor:
    """Ensure x has exactly [1, num_events, F] (no padding)."""
    if x.dim() != 2:
        raise ValueError(f"Expected x to have shape [N, F], got {tuple(x.shape)}")
    if x.shape[0] != num_events:
        raise ValueError(f"Expected exactly {num_events} events, got {x.shape[0]}.")
    return x.unsqueeze(0)


def _get_simulator_and_prior(
    problem: str,
    simulation_device: torch.device,
    prior_device: torch.device,
) -> Tuple[Callable[[torch.Tensor, int], torch.Tensor], BoxUniform, int]:
    def _log_transform(x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(x.clamp_min(1e-8))

    if problem == "simplified_dis":
        sim = SimplifiedDIS(device=simulation_device)

        prior = BoxUniform(
            low=torch.tensor([-1.0, 0.0, -1.0, 0.0], device=prior_device),
            high=torch.tensor([0.0, 5.0, 0.0, 5.0], device=prior_device),
        )
        x_dim = 2

        def _simulate(theta: torch.Tensor, n_events: int) -> torch.Tensor:
            raw = sim.sample(theta, n_events=n_events)
            return _log_transform(raw)

        return _simulate, prior, x_dim

    if problem in {"mceg", "mceg4dis"}:
        sim = MCEGSimulator(device=simulation_device)

        prior = BoxUniform(
            low=torch.tensor([-1.0, 0.0, -10.0, -10.0], device=prior_device),
            high=torch.tensor([10.0, 10.0, 10.0, 10.0], device=prior_device),
        )
        x_dim = 2

        def _simulate(theta: torch.Tensor, n_events: int) -> torch.Tensor:
            raw = sim.sample(theta, n_events=n_events)
            return _log_transform(raw)

        return _simulate, prior, x_dim

    raise ValueError(f"Unsupported problem '{problem}'. Choose 'simplified_dis' or 'mceg4dis'.")


def _build_embedding(x_dim: int, latent_dim: int = 32) -> PermutationInvariantEmbedding:
    trial_net = FCEmbedding(
        input_dim=x_dim,
        num_hiddens=64,
        num_layers=2,
        output_dim=latent_dim,
    )
    return PermutationInvariantEmbedding(
        trial_net=trial_net,
        trial_net_output_dim=latent_dim,
        aggregation_fn="sum",   # "mean" is also common
        num_layers=1,
        num_hiddens=64,
        output_dim=latent_dim,
        aggregation_dim=1,      # pool over events
    )


@torch.no_grad()
def _append_simulations_fixed_num_events_chunked(
    inference: NPE,
    simulator: Callable[[torch.Tensor, int], torch.Tensor],
    prior: BoxUniform,
    *,
    num_thetas: int,
    num_events: int,
    x_dim: int,
    simulation_device: torch.device,
    storage_device: torch.device,
    chunk_size: int,
    show_progress_bars: bool,
) -> NPE:
    """Generate and append simulations in chunks to avoid huge memory spikes.

    - simulator runs on simulation_device
    - appended tensors are moved to storage_device (typically CPU)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    # simple local progress (avoid extra deps)
    n_done = 0
    while n_done < num_thetas:
        n_cur = min(chunk_size, num_thetas - n_done)

        theta = prior.sample((n_cur,))  # CPU
        theta_sim = theta.to(simulation_device, non_blocking=True)

        xs = torch.empty((n_cur, num_events, x_dim), device=simulation_device)
        for i in range(n_cur):
            xs[i] = simulator(theta_sim[i], n_events=num_events)

        # Store/append on CPU unless you *really* want GPU-resident datasets
        inference.append_simulations(
            theta.to(storage_device),
            xs.to(storage_device),
            exclude_invalid_x=True,
        )

        n_done += n_cur
        if show_progress_bars:
            print(f"Appended simulations: {n_done}/{num_thetas}", end="\r", flush=True)

    if show_progress_bars:
        print()
    return inference


def build_permutation_invariant_posterior(
    *,
    problem: str,
    num_thetas: int = 1000,
    num_events: int = 256,
    latent_dim: int = 32,
    training_device: Optional[torch.device] = None,
    simulation_device: Optional[torch.device] = None,
    storage_device: Optional[torch.device] = None,
    training_batch_size: int = 256,
    chunk_size: int = 1024,
    show_progress_bars: bool = False,
) -> Tuple[object, PermutationInvariantConfig]:
    """
    GPU-friendly build:
    - training_device: where the neural net trains (cuda recommended if available)
    - simulation_device: where the simulator runs (cuda if simulator supports it, else cpu)
    - storage_device: where appended simulations live (cpu recommended)
    """

    if training_device is None:
        training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if simulation_device is None:
        # If your simulator supports CUDA, set this to cuda too; otherwise keep cpu.
        simulation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if storage_device is None:
        storage_device = torch.device("cpu")

    # Optional: speed on Ampere+ (harmless on CPU)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    simulator_fn, prior, x_dim = _get_simulator_and_prior(
        problem,
        simulation_device=simulation_device,
        prior_device=training_device,   # <- key line
    )

    embedding_net = _build_embedding(x_dim=x_dim, latent_dim=latent_dim)

    density_estimator = posterior_nn(
        "mdn",
        embedding_net=embedding_net,
        z_score_x="none",
        z_score_theta="independent",
    )

    inference = NPE(
        prior=prior,
        density_estimator=density_estimator,
        show_progress_bars=show_progress_bars,
        device=training_device,  # <- important: train the neural net on GPU
    )

    # Stream simulations into SBI
    inference = _append_simulations_fixed_num_events_chunked(
        inference,
        simulator_fn,
        prior,
        num_thetas=num_thetas,
        num_events=num_events,
        x_dim=x_dim,
        simulation_device=simulation_device,
        storage_device=storage_device,
        chunk_size=chunk_size,
        show_progress_bars=show_progress_bars,
    )

    inference.train(training_batch_size=training_batch_size)
    posterior = inference.build_posterior()

    cfg = PermutationInvariantConfig(
        posterior=posterior,
        prior=prior,
        simulator=simulator_fn,
        num_events=num_events,
        x_dim=x_dim,
        training_device=training_device,
        simulation_device=simulation_device,
    )
    return posterior, cfg


def demo(problem: str = "simplified_dis") -> None:
    posterior, cfg = build_permutation_invariant_posterior(
        problem=problem,
        num_thetas=512,
        num_events=128,
        latent_dim=16,
        training_batch_size=128,
        chunk_size=256,
        show_progress_bars=True,
    )

    with torch.no_grad():
        theta_o = cfg.prior.sample((1,))
        x_o_raw = cfg.simulator(theta_o[0].to(cfg.simulation_device), n_events=cfg.num_events)
        x_o = ensure_num_events(x_o_raw, num_events=cfg.num_events)

        # Sample on the device SBI expects (usually training_device)
        x_o = x_o.to(cfg.training_device)
        samples = posterior.sample((128,), x=x_o)

    print(f"Posterior mean for {problem}:", samples.mean(dim=0), "True parameters:", theta_o[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Permutation-invariant NPE for DIS simulators (GPU-friendly)")
    parser.add_argument("--problem", choices=["simplified_dis", "mceg4dis"], default="simplified_dis")
    parser.add_argument("--num_thetas", type=int, default=10000)
    parser.add_argument("--num_events", type=int, default=10000)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--training_batch_size", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--cpu_sim", action="store_true", help="Force simulator to run on CPU.")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress prints/bars")
    args = parser.parse_args()

    training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulation_device = torch.device("cpu") if args.cpu_sim else training_device

    posterior, cfg = build_permutation_invariant_posterior(
        problem=args.problem,
        num_thetas=args.num_thetas,
        num_events=args.num_events,
        latent_dim=args.latent_dim,
        training_batch_size=args.training_batch_size,
        chunk_size=args.chunk_size,
        training_device=training_device,
        simulation_device=simulation_device,
        storage_device=torch.device("cpu"),
        show_progress_bars=not args.no_progress,
    )

    with torch.no_grad():
        theta_o = cfg.prior.sample((1,))
        x_o_raw = cfg.simulator(theta_o[0].to(cfg.simulation_device), n_events=cfg.num_events)
        x_o = ensure_num_events(x_o_raw, num_events=cfg.num_events).to(cfg.training_device)
        samples = posterior.sample((64,), x=x_o)

    print("Inference finished.")
    print("Example posterior mean:", samples.mean(dim=0), "True parameters:", theta_o[0])