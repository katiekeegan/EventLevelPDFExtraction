"""Quick SNPE baseline for SimplifiedDIS using the local task adapter.
This script purposely avoids importing the heavy simformer package; it loads
only the local task adapter file directly and runs a small SNPE run.

Usage (example):
    python run_sbi_simplified_baseline.py --device cpu --nevents 2000 --nsim 1000

"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
from sbi.inference import SNPE, simulate_for_sbi


# Helper to load the adapter module by path (avoids importing full simformer package)
def load_local_tasks_module(repo_root):
    path = os.path.join(
        repo_root, "simformer", "src", "scoresbibm", "tasks", "local_tasks.py"
    )
    spec = importlib.util.spec_from_file_location("local_tasks_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(args):
    repo_root = os.path.abspath(os.path.dirname(__file__))
    mod = load_local_tasks_module(repo_root)
    TaskCls = mod.SimplifiedDISTask

    device = args.device
    task = TaskCls(
        backend="torch", nevents=args.nevents, nbins=args.nbins, device=device
    )

    # Build simulator function: maps theta_batch->summaries (torch.Tensor)
    sim_fn = task.get_simulator()

    prior = task.get_prior()

    print("Running small SNPE baseline with settings:")
    print(
        f"  device={device}, nevents={args.nevents}, nbins={args.nbins}, nsim={args.nsim}"
    )

    # Use simulate_for_sbi to simulate a dataset then train SNPE (small nsim for smoke)
    theta, x = simulate_for_sbi(sim_fn, prior, num_simulations=args.nsim)
    print("Simulated theta shape:", theta.shape, "x shape:", x.shape)

    inference = SNPE(prior)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)

    # Create an example observation (true theta + observed summary)
    true_theta = prior.sample(()) if False else torch.tensor([1.0, 1.2, 1.1, 0.5])
    x_o = sim_fn(true_theta.unsqueeze(0))[0]

    samples = posterior.sample((1000,), x=x_o)
    print("Posterior samples shape:", samples.shape)
    print("Posterior mean:", samples.mean(0))

    out_file = "samples_snpe_simplified_baseline.txt"
    np.savetxt(out_file, samples.cpu().numpy())
    print("Saved samples to", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--nevents", type=int, default=2000)
    parser.add_argument("--nbins", type=int, default=32)
    parser.add_argument("--nsim", type=int, default=1000)
    args = parser.parse_args()
    main(args)
