"""Small smoke-run utility to run a tiny SNPE baseline on local tasks.

This script avoids importing the full simformer package and uses the local
task registry in simformer/src/scoresbibm/tasks.
"""

import os
import sys

sys.path.insert(0, "simformer/src")
import numpy as np
import torch
from scoresbibm.tasks import get_task

try:
    from sbi.inference import SNPE, Posterior

    SBI_AVAILABLE = True
except Exception:
    SBI_AVAILABLE = False


def run_smoke(task_name="simplified_dis", nsim=200, nposterior_samples=500):
    t = get_task(task_name)
    sim = t.get_simulator()
    prior = t.get_prior()

    # simulate
    thetas = torch.from_numpy(
        np.random.uniform(low=0.0, high=1.0, size=(nsim, t.get_theta_dim())).astype(
            np.float32
        )
    )
    X = sim(thetas)
    print("Simulated theta shape:", thetas.shape, "x shape:", X.shape)

    if not SBI_AVAILABLE:
        print("sbi not available in this environment; skipping SNPE run.")
        return

    # build a trivial prior wrapper for sbi
    from sbi.utils import BoxUniform

    sbi_prior = BoxUniform(
        low=torch.zeros(t.get_theta_dim()), high=5.0 * torch.ones(t.get_theta_dim())
    )

    inference = SNPE(prior=sbi_prior)
    density_estimator = inference.append_simulations(thetas, X).train()
    posterior = inference.build_posterior(density_estimator)
    samples = posterior.sample((nposterior_samples,), x=X[0])
    print("Posterior samples shape:", samples.shape)


if __name__ == "__main__":
    # choose mceg4dis if available otherwise simplified_dis
    try:
        t = get_task("mceg4dis")
        # ensure MCEG is available
        sim = t.get_simulator()
        chosen = "mceg4dis"
    except Exception:
        chosen = "simplified_dis"
    print("Running smoke on task:", chosen)
    run_smoke(task_name=chosen, nsim=200)
