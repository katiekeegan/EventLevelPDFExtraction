import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from params_learning import LatentToParamsNN, PointNetEmbedding, EventDataset, generate_data  # Adjust the import as needed
from simulator import *
from models import *
from torch.distributions import *

def evaluate_over_n_samples(model, pointnet_model, n=100, num_events=100000, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simulator = SimplifiedDIS(torch.device('cpu'))

    all_errors = []
    chi2_up = []
    chi2_down = []

    for i in range(n):
        # === 1. Sample true parameters from a known range (e.g., Uniform[1, 10]) ===
        true_params = torch.FloatTensor(4).uniform_(1.0, 10.0).to(device)

        # === 2. Generate data from simulator ===
        xs = simulator.sample(true_params.cpu(), num_events).to(device)
        xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
        xs_feat = advanced_feature_engineering(xs_tensor)
        latent = pointnet_model(xs_feat.unsqueeze(0))

        # === 3. Sample predicted parameters ===
        samples = model(latent, 1000).squeeze()  # (1000, 4)
        predicted = torch.mean(samples, dim=0)

        # === 4. Compute relative error ===
        error = torch.abs(predicted - true_params) / (true_params + 1e-8)
        all_errors.append(error.cpu())

        # === 5. Evaluate Chi-squared statistics ===
        x_grid = torch.linspace(0.01, 0.99, 1000).to(device)  # Avoid 0/1
        pred_up = torch.mean(torch.stack([up(x_grid.cpu(), p.cpu()) for p in samples]), dim=0)
        pred_down = torch.mean(torch.stack([down(x_grid.cpu(), p.cpu()) for p in samples]), dim=0)
        true_up = up(x_grid, true_params)
        true_down = down(x_grid, true_params)

        chi2_up.append(compute_chisq_statistic(true_up.cpu().numpy(), pred_up.cpu().numpy()))
        chi2_down.append(compute_chisq_statistic(true_down.cpu().numpy(), pred_down.cpu().numpy()))

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
    print(f"Mean Chi² up: {chi2_up.mean():.4f} ± {chi2_up.std():.4f}")
    print(f"Mean Chi² down: {chi2_down.mean():.4f} ± {chi2_down.std():.4f}")

    plt.figure(figsize=(10, 5))
    plt.hist(chi2_up, bins=50, alpha=0.6, label='Chi² Up')
    plt.hist(chi2_down, bins=50, alpha=0.6, label='Chi² Down')
    plt.legend()
    plt.title("Chi² Statistic Distribution")
    plt.xlabel("Chi²")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("chisq_distributions.png")
    plt.show()

def plot_2d_histograms(true_params, generated_params, simulator, num_events=100000, bins=50, device=None):
    """
    Plots 2-D histograms for events generated using true parameters and generated parameters.
    
    Args:
        true_params: Tensor of shape (4,) representing the true parameters.
        generated_params: Tensor of shape (4,) representing the generated parameters.
        simulator: Instance of the SimplifiedDIS simulator.
        num_events: Number of events to generate (default is 100000).
        bins: Number of bins for the histogram (default is 50).
        device: Device to run computations on (default is None).
    """
    # Generate events using true parameters
    true_events = simulator.sample(true_params, num_events).cpu().numpy()

    # Generate events using generated parameters
    generated_events = simulator.sample(generated_params, num_events).cpu().numpy()

    # Create figure and axis
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Plot 2-D histogram for true parameters
    axes[0].hist2d(true_events[:, 0], true_events[:, 1], bins=bins, cmap='Blues')
    axes[0].set_title("2D Histogram - True Parameters")
    axes[0].set_xlabel("Sigma_p")
    axes[0].set_ylabel("Sigma_n")

    # Plot 2-D histogram for generated parameters
    axes[1].hist2d(generated_events[:, 0], generated_events[:, 1], bins=bins, cmap='Oranges')
    axes[1].set_title("2D Histogram - Generated Parameters")
    axes[1].set_xlabel("Sigma_p")
    axes[1].set_ylabel("Sigma_n")

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    plt.savefig("histograms.png")


def up(x, params):
    return (x ** params[0]) * ((1 - x) ** params[1])
    
def down(x, params):
    return (x ** params[2]) * ((1 - x) ** params[3])


# SimplifiedDIS class (Simulator)
class SimplifiedDIS:
    def __init__(self, device=None):
        self.Nu = 1
        self.au = 1  # params[0]
        self.bu = 1  # params[1]
        self.Nd = 2
        self.ad = 1  # params[2]
        self.bd = 1  # params[3]
        self.device = device

    def __call__(self, depth_profiles: np.ndarray):
        return self.sample(depth_profiles)

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

    def sample(self, params, nevents=1):
        self.init(torch.tensor(params, device=self.device))

        xs_p = torch.rand(nevents, device=self.device)
        sigma_p = 4 * self.up(xs_p) + self.down(xs_p)
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0)  # Replace NaNs with 0

        xs_n = torch.rand(nevents, device=self.device)
        sigma_n = 4 * self.down(xs_n) + self.up(xs_n)
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0)

        return torch.cat([sigma_p.unsqueeze(0), sigma_n.unsqueeze(0)], dim=0).t()


def advanced_feature_engineering(xs_tensor):
    log_features = torch.log1p(xs_tensor)
    symlog_features = torch.sign(xs_tensor) * torch.log1p(xs_tensor.abs())

    ratio_features = []
    diff_features = []
    for i in range(xs_tensor.shape[-1]):
        for j in range(i + 1, xs_tensor.shape[-1]):
            ratio = xs_tensor[..., i] / (xs_tensor[..., j] + 1e-8)
            ratio_features.append(torch.log1p(ratio.abs()).unsqueeze(-1))

            diff = torch.log1p(xs_tensor[..., i]) - torch.log1p(xs_tensor[..., j])
            diff_features.append(diff.unsqueeze(-1))

    ratio_features = torch.cat(ratio_features, dim=-1)
    diff_features = torch.cat(diff_features, dim=-1)

    return torch.cat([log_features, symlog_features, ratio_features, diff_features], dim=-1)

def plot_event_histogram(model, pointnet_model, true_params, device, n_mc=100, num_events=100000):
    """
    Plots the histograms of true and generated events side by side for comparison.
    
    Args:
        model: MDN-based model.
        pointnet_model: Model to get the latent embedding.
        true_params: Tensor of shape (1, param_dim) representing true parameters.
        device: Device to run computations on.
        n_mc: Number of Monte Carlo samples to draw.
        num_events: Number of events to generate for histogram.
    """
    model.eval()
    pointnet_model.eval()

    # Prepare latent embedding
    true_params = true_params.to(device)
    simulator = SimplifiedDIS(torch.device('cpu'))

    # Generate true events based on the true parameters
    xs = simulator.sample(true_params.cpu(), num_events).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor).to(device)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # Perform Monte Carlo sampling
    samples = model(latent_embedding, n_mc).squeeze()

    # Check for NaNs in samples
    if torch.any(torch.isnan(samples)):
        print("NaNs detected in samples!")
        return

    # Get the mode of the generated parameters
    mode_params = torch.mode(samples, dim=0).values

    # Check for NaNs in the mode parameters
    if torch.any(torch.isnan(mode_params)):
        print("NaNs detected in mode parameters!")
        return

    # Generate generated events based on the mode parameters
    generated_events = simulator.sample(mode_params.cpu(), num_events).to(device)

    # Convert generated events to numpy for plotting
    true_events_np = xs.cpu().numpy()
    generated_events_np = generated_events.cpu().numpy()

    # Ensure there are no NaNs in the generated events
    if np.any(np.isnan(generated_events_np)) or np.any(np.isnan(true_events_np)):
        print("NaNs detected in the events!")
        return

    # Check the shape of the data
    print(f"Shape of true_events_np: {true_events_np.shape}")
    print(f"Shape of generated_events_np: {generated_events_np.shape}")

    # Plot the two 2D histograms side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    # True events histogram
    axs[0].hist2d(true_events_np[:, 0], true_events_np[:, 1], bins=100, cmap='viridis')
    axs[0].set_title('True Events Histogram')
    axs[0].set_xlabel('Event X')
    axs[0].set_ylabel('Event Y')
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")

    # Generated events histogram
    axs[1].hist2d(generated_events_np[:, 0], generated_events_np[:, 1], bins=100, cmap='viridis')
    axs[1].set_title('Generated Events Histogram')
    axs[1].set_xlabel('Event X')
    axs[1].set_ylabel('Event Y')
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    plt.tight_layout()
    plt.show()
    plt.savefig("histograms.png")

def plot_params_distribution_single(model, pointnet_model, true_params, device, n_mc=100):
    """
    Plots the predicted parameters distribution for a single set of true parameters.
    
    Args:
        model: Your MDN-based model.
        pointnet_model: The model to get the latent embedding (if needed).
        true_params: A tensor of shape (1, param_dim) representing the fixed true parameters.
        device: Device to run the computations on.
        n_mc: Number of Monte Carlo samples to draw.
    """
    model.eval()
    pointnet_model.eval()
    
    # For a single set of true parameters, you need a corresponding latent embedding.
    # This could come from an input that you know corresponds to true_params.
    # For illustration, we assume you have a representative input 'xs_tensor'
    # that generates a latent embedding via your PointNet. If you don't have that,
    # you could simply use a random latent vector as a proxy.
    #
    # Here, we'll assume a random latent embedding as a placeholder:
    true_params = true_params.to(device)
    simulator = SimplifiedDIS(torch.device('cpu'))
    num_events = 100000
    xs = simulator.sample(true_params.cpu(), num_events).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor).to(device)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # latent_embedding = pointnet_model(xstorch.randn(1, 128).to(device)
    
    # Perform Monte Carlo sampling from the MDN for this latent embedding.
    samples = []
    with torch.no_grad():
        # for _ in range(n_mc):
        #     pi, mu, sigma = model(latent_embedding)
        #     sampled = sample_from_mdn(pi, mu, sigma)  # shape: (1, param_dim)
        #     samples.append(sampled.squeeze(0))
        samples = model(latent_embedding, n_mc).squeeze()
    
    # samples = torch.stack(samples)  # shape: (n_mc, param_dim)
    # Plot histograms for each parameter.
    fig, axes = plt.subplots(1, true_params.size(0), figsize=(20, 5))
    true_params_np = true_params.cpu().numpy()
    for i in range(true_params.size(0)):
        ax = axes[i]
        ax.hist(samples[:, i].cpu().squeeze().numpy(), bins=50, alpha=0.7)
        ax.axvline(true_params_np[i], color='r', linestyle='dashed', linewidth=2, label='True Value')
        ax.set_title(f'Parameter {i+1}')
        ax.set_xlabel('Predicted value')
        ax.set_ylabel('Frequency')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('Single_Params_Distribution_Samples.png')
    plt.show()

def plot_PDF_distribution_single(model, pointnet_model, true_params, device, n_mc=100, compute_chisq=True, plot_ratio=True):
    """
    Plots the predicted PDF distribution for a single set of true parameters for both 'up' and 'down' functions.
    
    Args:
        model: MDN-based model.
        pointnet_model: Model to get the latent embedding.
        true_params: Tensor of shape (1, param_dim) representing true parameters.
        device: Device to run computations on.
        n_mc: Number of Monte Carlo samples to draw.
        compute_chisq: Whether to compute and display the Chi-square statistic.
    """
    model.eval()
    pointnet_model.eval()

    # Prepare latent embedding
    true_params = true_params.to(device)
    simulator = SimplifiedDIS(torch.device('cpu'))
    num_events = 1000000
    xs = simulator.sample(true_params.cpu(), num_events).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor).to(device)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # Perform Monte Carlo sampling
    samples = model.sample(latent_embedding, n_mc).squeeze()

    # Set up grid for evaluation
    x_grid = torch.linspace(0, 1, 1000).to(device)

    # Plot the function for both 'down' and 'up'
    for func_name in ['down', 'up']:
        plot_function(x_grid, samples, true_params, func_name, device, compute_chisq, plot_ratio)
        # Plot ratio of mean estimated function to true function

def plot_function(x_grid, samples, true_params, func_name, device, compute_chisq, plot_ratio):
    """
    Helper function to plot the mean and standard deviation for a given function ('down' or 'up').
    
    Args:
        x_grid: The grid of x values.
        samples: The parameter samples to evaluate.
        true_params: The true parameters for the function.
        func_name: The name of the function to plot ('down' or 'up').
        device: Device to run the computations on.
        compute_chisq: Whether to compute and display the Chi-square statistic.
    """
    # Evaluate function for all samples
    num_samples = samples.shape[0]
    function_values = torch.zeros((num_samples, len(x_grid))).to(device)

    for i in range(num_samples):
        params = samples[i]
        function_values[i] = globals()[func_name](x_grid.cpu(), params.cpu()).to(device)

    # Compute statistics
    mean_f = torch.mean(function_values, dim=0)
    std_f = torch.std(function_values, dim=0)
    true_function = globals()[func_name](x_grid, true_params)

    # Compute quantiles (5% and 95%)
    lower = torch.quantile(function_values, 0.05, dim=0)
    upper = torch.quantile(function_values, 0.95, dim=0)

    # Convert to numpy for plotting
    x_grid_np = x_grid.cpu().numpy()
    mean_f_np = mean_f.detach().cpu().numpy()
    std_f_np = std_f.detach().cpu().numpy()
    lower_np = lower.detach().cpu().numpy()
    upper_np = upper.detach().cpu().numpy()
    true_function_np = true_function.detach().cpu().numpy()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid_np, true_function_np, label="True function", color="red", linestyle="--", linewidth=2)
    plt.plot(x_grid_np, mean_f_np, label="Mean function", color="blue")
    plt.fill_between(
        x_grid_np,
        mean_f_np - std_f_np,
        mean_f_np + std_f_np,
        alpha=0.3,
        color="blue",
        label="±1 std dev",
    )
    # Optionally plot quantiles
    # plt.fill_between(x_grid_np, lower_np, upper_np, alpha=0.3, color="blue", label="90% CI")

    plt.xlabel("x")
    plt.xscale("log")
    plt.ylabel(f"{func_name}(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'{func_name}.png')

    if compute_chisq:
        chi_square_statistic = compute_chisq_statistic(true_function_np, mean_f_np)
        print(f"Chi-square statistic for {func_name}: {chi_square_statistic}")
    
    # Plot ratio of mean estimated function to true function
    if plot_ratio:
        plot_function_ratio(x_grid, mean_f, true_function, func_name)

def plot_function_ratio(x_grid, mean_f, true_function, func_name):
    """
    Plots the ratio of the mean estimated function to the true function.
    
    Args:
        x_grid: The grid of x values.
        mean_f: The mean of the estimated function values.
        true_function: The true function values.
    """
    # Compute the ratio (mean function / true function)
    ratio = torch.abs(mean_f-true_function) / torch.abs((true_function + 1e-10))  # Avoid division by zero by adding a small constant

    # Plot the ratio
    x_grid_np = x_grid.cpu().numpy()
    ratio_np = ratio.detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(x_grid_np, ratio_np, label="Mean Estimated / True Function", color="green")
    plt.xlabel("x")
    plt.xscale("log")
    plt.ylabel(f"{func_name}(x) Relative Error")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'{func_name}_ratio.png')


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


def plot_params_distribution(model, pointnet_model, true_params, dataloader, device):
    model.train()  # Keep dropout active

    num_events = 1000000

    # Ensure the model is on the correct device
    model.to(device)
    pointnet_model.to(device)
    true_params = true_params.to(device)
    simulator = SimplifiedDIS(torch.device('cpu'))
    xs = simulator.sample(true_params.cpu(), num_events).to(device)

    multi_log=True
    log=False

    # Convert xs to torch tensor
    xs_tensor = torch.tensor(xs, dtype=torch.float32)
    num_samples = 50
    # Generate test data
    thetas, xs = generate_data(num_samples, num_events, device=torch.device('cpu')).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor).to(device)
    
    # Iterate over the dataset
    with torch.no_grad():  # No need to compute gradients for evaluation
        latent_embedding = pointnet_model(xs_tensor)
        
        # Forward pass through LatentToParamsNN
        predicted_params = model.sample(latent_embedding)
        latent_embedding = latent_embedding.to(device)
        preds = torch.stack([model(latent_embedding[i, :].unsqueeze(0)) for i in range(num_samples)], dim=0).squeeze()
    # Convert true parameters to CPU numpy array
    true_params_np = true_params.cpu().numpy()
    # Plotting the error distribution for each parameter
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 4 subplots for the 4 parameters
    for i in range(4):
        ax = axes[i]
        ax.hist(preds[:, i].squeeze().cpu().numpy(), bins=50, alpha=0.7)
        ax.axvline(true_params_np[i], color='r', linestyle='dashed', linewidth=2, label='True Value')
        ax.set_title(f'Parameter {i+1}')
        ax.set_xlabel('Error (|theta - theta*|/|theta*|)')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.savefig('Params_Distribution.png')
    plt.show()


# Load the model and data
def load_model_and_data(model_path, pointnet_model_path, num_samples=100, num_events=10000, device=None):
    # Ensure the device is set
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate test data
    thetas, xs = generate_data(num_samples, num_events, device=torch.device('cpu'))
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor_engineered = advanced_feature_engineering(xs)
    input_dim = xs_tensor_engineered.shape[-1]

    # Instantiate the models
    latent_dim = 1024  # Example latent dimension
    # model = ImprovedMDN(latent_dim, 4)  # Parameter dimension is 4
    model = ConditionalRealNVP(latent_dim=latent_dim, param_dim=4, hidden_dim=1024, num_flows=6)
    pointnet_model = PointNetCrossAttention(input_dim=input_dim, latent_dim=latent_dim)
    state_dict = torch.load(pointnet_model_path)
    # Remove the 'module.' prefix from the state_dict keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    # Load the modified state_dict into the model
    pointnet_model.load_state_dict(new_state_dict)
    pointnet_model.eval()
    # Load the model weights
    state_dict = torch.load(model_path)
    # Remove the 'module.' prefix from the state_dict keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    # Load the modified state_dict into the model
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    pointnet_model = pointnet_model.to(device)
    # Set the models to evaluation mode
    model.eval()
    pointnet_model.eval()
    # Prepare the dataset and dataloader
    dataset = EventDataset(xs_tensor_engineered.to(device), thetas.to(device), pointnet_model.to(device), device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return model, pointnet_model, dataloader, device

# Main function to load, evaluate and plot the results
def main():
    model_path = 'simplified_dis_latent1024_ns_1000_ne_100000_inference/final_inference_net.pth'  # Path to the trained model
    pointnet_model_path = 'simplified_dis_latent1024_ns_1000_ne_100000/model_epoch_200.pth'  # Path to the pretrained PointNet model
    multi_log=True
    log=False
    # Load the model and data
    model, pointnet_model, dataloader, device = load_model_and_data(model_path, pointnet_model_path)
    true_params = torch.tensor([0.5, 0.5, 0.5, 0.5])
    # Plot loss values
    # Load and plot
    # loss_history = load_and_plot('loss_history.npy')
    # Evaluate the model and plot the error distribution
    # evaluate_and_plot(model, pointnet_model, dataloader, device)
    # Plot distribution of estimated parameters given a specific set of true parameters
    # true_params = torch.tensor([1.0, 4.0, 1.0, 4.0])
    # plot_params_distribution(model, pointnet_model, true_params, dataloader, device)
    plot_params_distribution_single(model, pointnet_model, true_params, device, n_mc=10000)
    plot_PDF_distribution_single(model, pointnet_model, true_params, device, n_mc=10000)
    plot_event_histogram(model, pointnet_model, true_params, device, n_mc=10000, num_events=1000000)
    evaluate_over_n_samples(model, pointnet_model, n=100, num_events=100000, device=device)
if __name__ == "__main__":
    main()