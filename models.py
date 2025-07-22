import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm as sn
from torch.nn import MultiheadAttention
# from nflows.transforms import (
#     CompositeTransform,
#     ReversePermutation,
#     MaskedAffineAutoregressiveTransform
# )
# from nflows.distributions import StandardNormal
# from nflows.flows import Flow
class InferenceNet(nn.Module):
    def __init__(self, embedding_dim, output_dim = 6, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)  # Output raw (unconstrained) parameters
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # Get raw network output
        params = self.net(z)
        
        return params

# class ConditionalRealNVP(nn.Module):
#     def __init__(self, latent_dim, param_dim, hidden_dim=256, num_flows=5):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.param_dim = param_dim
        
#         def create_transform():
#             return MaskedAffineAutoregressiveTransform(
#                 features=param_dim,
#                 hidden_features=hidden_dim,
#                 context_features=latent_dim,  # <- this makes it conditional
#                 num_blocks=2,
#                 use_residual_blocks=True,
#                 activation=nn.ReLU()
#             )
        
#         transforms = []
#         for _ in range(num_flows):
#             transforms.append(ReversePermutation(features=param_dim))
#             transforms.append(create_transform())
        
#         transform = CompositeTransform(transforms)
#         base_distribution = StandardNormal(shape=[param_dim])
        
#         self.flow = Flow(transform=transform, distribution=base_distribution)

#     def forward(self, latent_embedding, true_params):
#         # true_params: [batch_size, param_dim]
#         # latent_embedding: [batch_size, latent_dim]
#         return self.flow.log_prob(inputs=true_params, context=latent_embedding)

#     def sample(self, latent_embedding, num_samples=1):
#         # latent_embedding: [batch_size, latent_dim]
#         return self.flow.sample(num_samples=num_samples, context=latent_embedding)

class HierarchicalAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, chunk_size=4096):
        super().__init__()
        self.chunk_size = chunk_size
        # Local attention within chunks
        self.local_attn = nn.Linear(hidden_dim, 1)
        # Global attention over chunk summaries
        self.global_attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        B, N, D = x.shape
        # Step 1: Split into manageable chunks
        x = x.view(B, -1, self.chunk_size, D)  # [B, num_chunks, chunk_size, D]
        
        # Step 2: Local attention within each chunk
        local_weights = torch.softmax(self.local_attn(x), dim=2)  # [B, num_chunks, chunk_size, 1]
        chunk_summaries = (x * local_weights).sum(dim=2)  # [B, num_chunks, D]
        
        # Step 3: Global attention across chunks
        global_weights = torch.softmax(self.global_attn(chunk_summaries), dim=1)  # [B, num_chunks, 1]
        pooled = (chunk_summaries * global_weights).sum(dim=1)  # [B, D]
        
        return pooled

class PointNetEmbedding(nn.Module):
    def __init__(self, input_dim=2, latent_dim=64, hidden_dim=256, predict_theta=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.predict_theta = predict_theta  # control whether to return regression output

        # Initial MLP for point-wise feature extraction
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Hierarchical pooling layer
        self.pool = HierarchicalAttentionPooling(hidden_dim, chunk_size=1000)

        # Final MLP to get latent embedding
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Optional theta regressor (only used if predict_theta=True)
        if predict_theta:
            self.theta_regressor = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 4)  # Output dimension is number of θ parameters
            )

    def forward(self, x):
        batch_size, num_events, _ = x.shape

        # Step 1: Point-wise feature extraction
        x = self.mlp1(x)  # (B, N, hidden_dim)

        # Step 2: Hierarchical attention pooling → (B, hidden_dim)
        x = self.pool(x)

        # Step 3: Latent projection → (B, latent_dim)
        latent = self.mlp2(x)

        # Step 4: (optional) Predict theta from latent
        if self.predict_theta:
            theta_hat = self.theta_regressor(latent)
            return latent, theta_hat
        else:
            return latent

class PointNetWithAttention(nn.Module):
    def __init__(self, input_dim=2, latent_dim=64, hidden_dim=256, num_heads=4, predict_theta=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.predict_theta = predict_theta

        # Initial point-wise MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Self-attention layer: expects (N, B, D)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Optional: LayerNorm + residual
        self.norm = nn.LayerNorm(hidden_dim)

        # Pooling layer
        self.pool = HierarchicalAttentionPooling(hidden_dim, chunk_size=1000)

        # Latent projection
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        if predict_theta:
            self.theta_regressor = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 4)
            )

    def forward(self, x):
        # x: (B, N, input_dim)
        x = self.mlp1(x)  # (B, N, hidden_dim)

        # Self-attention needs input as (B, N, D)
        attn_out, _ = self.self_attn(x, x, x)  # (B, N, hidden_dim)
        x = self.norm(x + attn_out)  # Residual + normalization

        x = self.pool(x)  # (B, hidden_dim)
        latent = self.mlp2(x)  # (B, latent_dim)

        if self.predict_theta:
            theta_hat = self.theta_regressor(latent)
            return latent, theta_hat
        else:
            return latent

class PointNetCrossAttention(nn.Module):
    def __init__(self, input_dim=2, latent_dim=64, hidden_dim=16, num_heads=2, predict_theta=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.predict_theta = predict_theta

        # Point-wise feature extraction
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Learnable global token (same per batch)
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # (1, 1, D)

        # Multi-head attention: Q from global token, K/V from points
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Final latent projection
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        if predict_theta:
            self.theta_regressor = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 4)
            )

    def forward(self, x):
        B, N, _ = x.shape  # (B, N, input_dim)

        x = self.mlp1(x)  # (B, N, hidden_dim)

        # Expand global token for batch
        global_token = self.global_token.expand(B, -1, -1)  # (B, 1, hidden_dim)

        # Apply cross-attention: Q=global_token, K/V=point features
        attended, _ = self.cross_attn(query=global_token, key=x, value=x)  # (B, 1, hidden_dim)
        attended = attended.squeeze(1)  # (B, hidden_dim)

        latent = self.mlp2(attended)  # (B, latent_dim)

        if self.predict_theta:
            theta_hat = self.theta_regressor(latent)
            return latent, theta_hat
        else:
            return latent

class DISPointCloudRegressor(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, latent_dim=128, predict_theta=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.predict_theta = predict_theta

        # Local encoding (shared MLP over all points)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Set-based pooling (learnable or stateless)
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        # Regression head
        # if predict_theta:
        #     self.regressor = nn.Sequential(
        #         nn.Linear(latent_dim, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 4)  # predict e.g. q, qbar, g parameters
        #     )

    def forward(self, x):
        B, N, D = x.shape
        # encoded_chunks = []

        # chunk_size = 4096
        # for i in range(0, N, chunk_size):
        #     chunk = x[:, i:i+chunk_size, :]
        #     enc = self.encoder(chunk)  # (B, chunk_size, hidden_dim)
        #     encoded_chunks.append(enc)

        # x_encoded = torch.cat(encoded_chunks, dim=1)  # (B, N, hidden_dim)
        x_encoded = self.encoder(x)
        z = torch.mean(x_encoded, dim=1)  # (B, hidden_dim)
        z = self.pool(z)  # (B, latent_dim)

        # if self.predict_theta:
        #     theta_hat = self.regressor(z)
        #     return z, theta_hat
        return z


class DiffusionModel(nn.Module):
    def __init__(self, sample_dim, param_dim, hidden_dim=128, time_embedding_dim=32, sample_encode_dim=32, timesteps=1000, n_events=1000):
        super(DiffusionModel, self).__init__()
        self.timesteps = timesteps
        
        self.sample_encoder = PointNetCompressor(input_dim=sample_dim, latent_dim=sample_encode_dim)
        
        self.time_embedding = nn.Linear(1, time_embedding_dim)
        
        self.diffusion_process = nn.Sequential(
            nn.Linear(sample_encode_dim + param_dim + time_embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, param_dim)
        )
    
    def forward(self, samples, t, params):
        t_embedding = self.time_embedding(t.squeeze().unsqueeze(-1).float())  # Simple sinusoidal encoding
        samples_emb = self.sample_encoder(samples).squeeze()
        combined_input = torch.cat([samples_emb, t_embedding, params], dim=-1)
        pred_noise = self.diffusion_process(combined_input)
        return pred_noise

class PointNetPDFRegressor(nn.Module):
    def __init__(self, input_dim=6, latent_dim=64, hidden_dim=256, num_heads=4, num_seeds=1, num_points_sampled=4096):
        super().__init__()
        self.num_points_sampled = num_points_sampled
        self.hidden_dim = hidden_dim

        # Per-point MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Learnable seed vector(s)
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, hidden_dim))

        # Pooling by Multihead Attention (PMA)
        self.pma = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Latent MLP → latent_dim → predict PDF parameters
        self.mlp2 = nn.Sequential(
            nn.Linear(num_seeds * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)  # predict 4 parameters: au, bu, ad, bd
        )

    def subsample(self, x):
        # Random subsampling of points to avoid memory blowup
        B, N, D = x.shape
        idx = torch.randint(0, N, (B, self.num_points_sampled), device=x.device)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(x, dim=1, index=idx_exp)

    def forward(self, x):
        # x: (B, N=1M, 6)
        x = self.mlp1(x)  # (B, N, hidden_dim) -- efficient point-wise transformation

        # Subsample transformed features
        x = self.subsample(x)  # (B, num_points_sampled, hidden_dim)

        # Expand learnable seeds
        seed = self.seed_vectors.expand(x.size(0), -1, -1)  # (B, num_seeds, hidden_dim)

        # Attention: PMA (Pooling by Multihead Attention)
        attended, _ = self.pma(query=seed, key=x, value=x)  # (B, num_seeds, hidden_dim)

        # Flatten and regress
        latent = attended.view(x.size(0), -1)
        theta_hat = self.mlp2(latent)  # (B, 4)
        return theta_hat

# class PointNetPMA(nn.Module):
#     def __init__(self, input_dim=2, latent_dim=64, hidden_dim=32, num_heads=2, num_seeds=4, predict_theta=True):
#         super().__init__()
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.hidden_dim = hidden_dim
#         self.predict_theta = predict_theta
#         self.num_seeds = num_seeds

#         # Point-wise MLP encoder with residual
#         self.mlp1 = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )
#         self.mlp1_bn = nn.BatchNorm1d(hidden_dim)

#         # Optional deeper point-wise processing
#         self.mlp2 = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )
#         self.mlp2_bn = nn.BatchNorm1d(hidden_dim)

#         # Learnable seed vectors
#         self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, hidden_dim))

#         # Multihead attention pooling
#         self.pma = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

#         # Latent projection
#         self.latent_proj = nn.Sequential(
#             nn.Linear(hidden_dim * num_seeds, latent_dim),
#             nn.ReLU(),
#             nn.Linear(latent_dim, latent_dim)
#         )

#         if predict_theta:
#             self.theta_regressor = nn.Sequential(
#                 nn.Linear(latent_dim, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 4)
#             )

#     def forward(self, x):
#         B, N, D = x.shape

#         # Point-wise feature extraction
#         x = self.mlp1(x)
#         x = self.mlp1_bn(x.transpose(1, 2)).transpose(1, 2)
#         x = x + self.mlp2_bn(self.mlp2(x).transpose(1, 2)).transpose(1, 2)  # Residual

#         # Seed vectors broadcast
#         seed = self.seed_vectors.expand(B, -1, -1)

#         # PMA: Query from seed, Key/Value from points
#         attended, _ = self.pma(query=seed, key=x, value=x)  # (B, num_seeds, hidden_dim)

#         # Flatten pooled output
#         latent = self.latent_proj(attended.reshape(B, -1))

#         if self.predict_theta:
#             theta_hat = self.theta_regressor(latent)
#             return latent, theta_hat
#         return latent

class PointNetPMA(nn.Module):
    def __init__(self, input_dim=2, latent_dim=64, hidden_dim=16, num_heads=2, num_seeds=1, predict_theta=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.predict_theta = predict_theta
        self.num_seeds = num_seeds

        # Point-wise MLP (same as before)
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Learnable seed vectors (shared across batch)
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, hidden_dim))

        # Pooling via multihead attention: Q from seeds, K/V from point features
        self.pma = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Latent projection
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim * num_seeds, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # if predict_theta:
        #     self.theta_regressor = nn.Sequential(
        #         nn.Linear(latent_dim, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 4)
        #     )

    def forward(self, x):
        B, N, _ = x.shape  # (B, N, input_dim)

        x = self.mlp1(x)  # (B, N, hidden_dim)

        # Expand seed vectors for batch
        seed = self.seed_vectors.expand(B, -1, -1)  # (B, num_seeds, hidden_dim)

        # Apply attention from seeds to point features (PMA)
        attended, _ = self.pma(query=seed, key=x, value=x)  # (B, num_seeds, hidden_dim)

        # Flatten seed outputs
        attended_flat = attended.reshape(B, -1)  # (B, num_seeds * hidden_dim)

        latent = self.mlp2(attended_flat)  # (B, latent_dim)
        return latent

class LatentToParamsNN(nn.Module):
    def __init__(self, latent_dim, param_dim, dropout_prob=0.2):
        super(LatentToParamsNN, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        # Separate layers for mean and variance
        self.fc_mean = nn.Linear(64, param_dim)
        self.fc_log_var = nn.Linear(64, param_dim)  # Output log variance

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        
        mean = 10 * torch.sigmoid(self.fc_mean(x))  # Scale to parameter range
        # log_var = self.fc_log_var(x)  # Unconstrained log variance
        log_var = torch.tanh(self.fc_log_var(x), min=-10, max=10)  # Prevent extreme values
        variance = torch.exp(log_var)
        # log_var = torch.tanh(self.fc_logvar(x)) * 5  # Keep within a reasonable range
        return mean, variance