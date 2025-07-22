import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import flax.linen as nn
import optax
from flax.training import train_state

# --- 1. Simulator Definition ---
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

# --- 2. Histogram Summary Function (JAX) ---
def histogram_summary(x, nbins=32, range_min=-10, range_max=10):
    D = x.shape[1]
    summaries = []
    for d in range(D):
        hist, _ = jnp.histogram(x[:, d], bins=nbins, range=(range_min, range_max))
        hist = hist / (hist.sum() + 1e-8)
        summaries.append(hist)
    return jnp.concatenate(summaries, axis=0)  # shape (D*nbins,)

# --- 3. Generate Training Data ---
key = jax.random.PRNGKey(42)
num_train = 1000
param_dim = 4
obs_dim = 2
nbins = 32
summary_dim = obs_dim * nbins
nevents_per_param = 100000

params = jax.random.uniform(key, (num_train, param_dim), minval=0.1, maxval=1.0)
summaries = []
for i, p in enumerate(params):
    key_i = jax.random.PRNGKey(i)
    x = simplified_dis_simulator(key_i, p, nevents=nevents_per_param)  # shape (100, 2)
    summary = histogram_summary(x, nbins=nbins)
    summaries.append(summary)
summaries = jnp.stack(summaries)  # (num_train, summary_dim)

# --- 4. Dummy Transformer Model (Flax, simple for demonstration) ---
class DummyScoreTransformer(nn.Module):
    param_dim: int
    summary_dim: int
    hidden_dim: int = 32
    num_layers: int = 2

    @nn.compact
    def __call__(self, params, summaries):
        x = jnp.concatenate([params, summaries], axis=-1)
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.param_dim)(x)
        return x

# --- 5. Training Setup ---
class TrainState(train_state.TrainState):
    pass

def loss_fn(params, batch_params, batch_summaries, model):
    preds = model.apply({'params': params}, batch_params, batch_summaries)
    # MSE for demonstration; replace with score-matching loss if needed
    loss = jnp.mean((preds - batch_params) ** 2)
    return loss

model = DummyScoreTransformer(param_dim=param_dim, summary_dim=summary_dim)
variables = model.init(key, params[:1], summaries[:1])
state = TrainState.create(apply_fn=model.apply, params=variables['params'],
                         tx=optax.adam(learning_rate=1e-3))

batch_size = 32
num_epochs = 10

for epoch in range(num_epochs):
    for i in range(0, num_train, batch_size):
        batch_params = params[i:i+batch_size]
        batch_summaries = summaries[i:i+batch_size]
        def step_fn(state, batch):
            batch_params, batch_summaries = batch
            grads = jax.grad(loss_fn)(state.params, batch_params, batch_summaries, model)
            return state.apply_gradients(grads=grads)
        state = step_fn(state, (batch_params, batch_summaries))
    print(f"Epoch {epoch+1} completed.")

# --- 6. Example Inference ---
test_param = jnp.array([1.2, 0.7, 0.9, 2.0])
test_x = simplified_dis_simulator(jax.random.PRNGKey(123), test_param, nevents=nevents_per_param)
test_summary = histogram_summary(test_x, nbins=nbins)
est_param = model.apply({'params': state.params}, test_param[None, :], test_summary[None, :])
print("Estimated parameter:", est_param)