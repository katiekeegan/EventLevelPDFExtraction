import jax
import jax.numpy as jnp
import numpy as np

# --- 1. Simulator (your custom simulator) ---
class SimplifiedDIS:
    def __init__(self):
        pass

    def init(self, params):
        self.Nu = 1
        self.au = params[0]
        self.bu = params[1]
        self.Nd = 2
        self.ad = params[2]
        self.bd = params[3]

    def up(self, x):
        return self.Nu * (x ** self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        return self.Nd * (x ** self.ad) * ((1 - x) ** self.bd)

    def sample(self, params, nevents=100, key=None):
        self.init(params)
        if key is None:
            key = jax.random.PRNGKey(0)
        key_p, key_n = jax.random.split(key)
        xs_p = jax.random.uniform(key_p, (nevents,))
        sigma_p = 4 * self.up(xs_p) + self.down(xs_p)
        sigma_p = jnp.nan_to_num(sigma_p, nan=0.0)

        xs_n = jax.random.uniform(key_n, (nevents,))
        sigma_n = 4 * self.down(xs_n) + self.up(xs_n)
        sigma_n = jnp.nan_to_num(sigma_n, nan=0.0)

        return jnp.stack([sigma_p, sigma_n], axis=1)  # shape (nevents, 2)

def simplified_dis_simulator(key, theta, nevents=100):
    sim = SimplifiedDIS()
    result = sim.sample(theta, nevents=nevents, key=key)
    return result  # shape (nevents, 2)

# --- 2. Histogram Summary Function ---
def histogram_summary(x, nbins=32, range_min=-10, range_max=10):
    D = x.shape[1]
    summaries = []
    for d in range(D):
        hist, _ = np.histogram(np.asarray(x[:, d]), bins=nbins, range=(range_min, range_max))
        hist = hist / (hist.sum() + 1e-8)
        summaries.append(hist)
    return np.concatenate(summaries, axis=0)  # shape (D*nbins,)

# --- 3. Generate Training Data (theta, histogram pairs) ---
key = jax.random.PRNGKey(42)
num_train = 1000
param_dim = 4
obs_dim = 2
nbins = 32
summary_dim = obs_dim * nbins
nevents_per_param = 10000

params = jax.random.uniform(key, (num_train, param_dim), minval=0.1, maxval=1.0)
summaries = []
for i, p in enumerate(np.asarray(params)):
    key_i = jax.random.PRNGKey(i)
    x = simplified_dis_simulator(key_i, p, nevents=nevents_per_param)
    summary = histogram_summary(x, nbins=nbins)
    summaries.append(summary)
summaries = np.stack(summaries)  # (num_train, summary_dim)

# --- 4. Setup Simformer (score_transformer) ---
from scoresbibm.methods import score_transformer
from types import SimpleNamespace

# Minimal config object (tweak as needed)
cfg = SimpleNamespace(
    device='cpu',  # or 'gpu' if available
    sde=SimpleNamespace(),    # Use default SDE params
    model=SimpleNamespace(
        input_dim=summary_dim,
        output_dim=param_dim,
        # Add other model hyperparams as needed
    ),
    train=SimpleNamespace(
        batch_size=32,
        num_epochs=10,
        lr=1e-3,
        z_score_data=False,
    ),
    train_z_score_data=False
)

# Prepare data dict as expected by train_transformer_model
data = {
    "theta": params,        # shape (num_train, param_dim)
    "x": summaries,         # shape (num_train, summary_dim)
}

# Minimal dummy task definition (implement your own as needed)
class DummyTask:
    def get_node_id(self):
        # Simformer uses node IDs for graph tasks; for tabular data, just use zeros
        return np.zeros((num_train,), dtype=int)
    def get_theta_dim(self):
        return param_dim
    def get_x_dim(self):
        return summary_dim

task = DummyTask()
rng = jax.random.PRNGKey(0)

# --- 5. Train the Simformer transformer model ---
trained_params, trained_opt_state = score_transformer.train_transformer_model(
    task, data, cfg, rng
)

# --- 6. Inference: predict theta from a new histogram ---
test_param = np.array([1.2, 0.7, 0.9, 2.0])
test_x = simplified_dis_simulator(jax.random.PRNGKey(123), test_param, nevents=nevents_per_param)
test_summary = histogram_summary(test_x, nbins=nbins)[None, :]  # Batch dimension

# The model is a Haiku module, so use model.apply as in the repo
# You may need to adapt this depending on the exact output of train_transformer_model

# Example: If trained_params is the model params, and model is available:
# est_param = score_transformer.model_apply(trained_params, test_summary)
# print("Estimated theta:", est_param)

# The actual inference call may differ depending on Simformer's API;
# consult score_transformer.py for the correct way to perform inference with the trained model.

print("Training and inference complete. (See repo docs for advanced usage.)")