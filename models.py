import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm as sn
from torch.nn import MultiheadAttention
import math
from typing import Optional, Tuple
# from nflows.transforms import (
#     CompositeTransform,
#     ReversePermutation,
#     MaskedAffineAutoregressiveTransform
# )
# from nflows.distributions import StandardNormal
# from nflows.flows import Flow
class InferenceNet(nn.Module):
    def __init__(self, embedding_dim, output_dim = 6, hidden_dim=512, nll_mode=False):
        super().__init__()
        self.nll_mode = nll_mode
        self.output_dim = output_dim
        
        # Shared network layers
        self.shared_net = nn.Sequential(
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
        )
        
        if nll_mode:
            # Separate heads for mean and log-variance
            self.mean_head = nn.Linear(hidden_dim//2, output_dim)
            self.log_var_head = nn.Linear(hidden_dim//2, output_dim)
        else:
            # Original single output head
            self.output_head = nn.Linear(hidden_dim//2, output_dim)
            
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # Get shared features
        features = self.shared_net(z)
        
        if self.nll_mode:
            # Return both means and log-variances
            means = self.mean_head(features)
            log_vars = self.log_var_head(features)
            # Clamp log-variances for numerical stability
            log_vars = torch.clamp(log_vars, min=-10, max=10)
            return means, log_vars
        else:
            # Original behavior - return raw parameters
            params = self.output_head(features)
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

class SmallPointEncoder(nn.Module):
    """
    Tiny point-wise encoder applied to points in a chunk.
    Input: (B*C, K, D)
    Output: (B*C, K, chunk_latent)
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, chunk_latent: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_latent),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(chunk_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*C, K, D)
        Bc, K, D = x.shape
        x_flat = x.view(Bc * K, D)
        z = self.net(x_flat)                  # (Bc*K, chunk_latent)
        z = z.view(Bc, K, -1)                 # (Bc, K, chunk_latent)
        z = self.dropout(z)
        # per-chunk pooling (mean + LayerNorm)
        z_pool = z.mean(dim=1)                # (Bc, chunk_latent)
        z_out = self.ln(z_pool)
        return z_out                          # (Bc, chunk_latent)


class ChunkedPointNetPMA(nn.Module):
    """
    Combined encoder for very large point clouds (N ~ 100k).
    Strategy:
      - Pad and view input in chunks of `chunk_size`
      - Encode each chunk with a small PointEncoder -> chunk summaries
      - Aggregate chunk summaries with PMA using small number of seeds
      - Flatten seed outputs and project to latent_dim
    Input: x (B, N, D)
    Output: latent (B, latent_dim)
    """
    def __init__(
        self,
        input_dim: int = 2,
        chunk_size: int = 4096,
        chunk_latent: int = 64,
        hidden_dim: int = 64,
        num_seeds: int = 12,
        latent_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.chunk_size = int(chunk_size)
        self.chunk_latent = chunk_latent
        self.num_seeds = num_seeds
        self.latent_dim = latent_dim

        # per-chunk encoder (applied to K points at a time)
        self.chunk_encoder = SmallPointEncoder(input_dim=input_dim,
                                               hidden_dim=hidden_dim,
                                               chunk_latent=chunk_latent,
                                               dropout=dropout)

        # PMA: queries = learnable seed vectors, keys/vals = chunk summaries
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, chunk_latent) * 0.05)
        self.ln_seed = nn.LayerNorm(chunk_latent)
        self.pma = nn.MultiheadAttention(embed_dim=chunk_latent, num_heads=num_heads, batch_first=True)

        # final projection from flattened seeds -> latent
        self.proj = nn.Sequential(
            nn.Linear(num_seeds * chunk_latent, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim)
        )

        # init weights
        self._init_weights()

    def _init_weights(self):
        # Xavier init for Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # reinit MHA subparams safe
        for name, p in self.pma.named_parameters():
            if name.endswith('in_proj_weight') or name.endswith('out_proj.weight'):
                nn.init.xavier_uniform_(p)
            if name.endswith('in_proj_bias') or name.endswith('out_proj.bias'):
                nn.init.zeros_(p)

    def _pad_and_chunk(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Pad x so that N is multiple of chunk_size and return reshaped tensor for chunk encoding.
        Input x: (B, N, D)
        Returns:
           x_chunked: (B * n_chunks, chunk_size, D)
           n_chunks: int (per-batch, same for every batch because of padding)
        """
        B, N, D = x.shape
        K = self.chunk_size
        n_chunks = (N + K - 1) // K
        pad = n_chunks * K - N
        if pad > 0:
            pad_t = x.new_zeros(B, pad, D)
            x = torch.cat([x, pad_t], dim=1)   # (B, n_chunks*K, D)
        x = x.view(B, n_chunks, K, D)          # (B, C, K, D)
        x = x.view(B * n_chunks, K, D)         # (B*C, K, D)
        return x, n_chunks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        returns: latent (B, latent_dim)
        """
        if x.dim() != 3:
            raise ValueError("Expected input x to be (B, N, D)")
        B, N, D = x.shape
        # 1) chunk and encode
        x_chunks, n_chunks = self._pad_and_chunk(x)          # (B*C, K, D)
        chunk_summaries = self.chunk_encoder(x_chunks)       # (B*C, chunk_latent)
        chunk_summaries = chunk_summaries.view(B, n_chunks, -1)  # (B, C, chunk_latent)

        # 2) PMA aggregation: queries = seeds, keys/vals = chunk_summaries
        seed = self.seed_vectors.expand(B, -1, -1)           # (B, S, E)
        seed_norm = self.ln_seed(seed)
        attended, _ = self.pma(query=seed_norm, key=chunk_summaries, value=chunk_summaries)
        # residual + norm on seeds
        attended = self.ln_seed(attended + seed)             # (B, S, E)

        # 3) flatten seeds and project
        attended_flat = attended.reshape(B, self.num_seeds * self.chunk_latent)
        latent = self.proj(attended_flat)                    # (B, latent_dim)
        return latent


class PointNetPMA(nn.Module):
    """
    PointNet with a Pooling by Multihead Attention (PMA) head.
    - input x: (B, N, input_dim)
    - returns: (B, latent_dim)
    Stable defaults: LayerNorm, small seed init, residual, Xavier init, optional dropout.
    """
    def __init__(self,
                 input_dim=2,
                 latent_dim=64,
                 hidden_dim=64,
                 num_heads=4,
                 num_seeds=8,
                 dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_seeds = num_seeds

        # point-wise MLP (applied to every point)
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # normalization for point features
        self.ln_point = nn.LayerNorm(hidden_dim)

        # learnable seed vectors for PMA (small init)
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, hidden_dim) * 0.05)
        self.ln_seed = nn.LayerNorm(hidden_dim)

        # PMA via MultiheadAttention: queries = seeds, keys/vals = point features
        # batch_first=True keeps shapes (B, S, E) and (B, N, E)
        self.pma = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # optional projection after attended seeds (residual + norm already added in forward)
        # final latent projection: flatten seeds -> latent
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim * num_seeds, latent_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(latent_dim, latent_dim),
        )

        # initialize weights cleanly
        self._init_weights()

    def _init_weights(self):
        # Xavier for linear layers, smaller init for seed vectors already applied on creation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # MultiheadAttention has in_proj_weight and out_proj.weight which are Linear-like
        # Torch already initializes these, but we will do a safe re-init if present:
        for name, p in self.pma.named_parameters():
            if name.endswith('in_proj_weight') or name.endswith('out_proj.weight'):
                nn.init.xavier_uniform_(p)
            if name.endswith('in_proj_bias') or name.endswith('out_proj.bias'):
                nn.init.zeros_(p)

    def forward(self, x):
        """
        x: (B, N, input_dim)
        returns: (B, latent_dim)
        """
        if x.dim() != 3:
            raise ValueError("Expected input x to be (B, N, input_dim)")

        B, N, D = x.shape
        if D != self.input_dim:
            # allow flexible input_dim but warn
            # raise ValueError(f"input_dim mismatch: expected {self.input_dim}, got {D}")
            pass

        # point-wise encoding
        x_flat = x.view(B * N, self.input_dim)           # (B*N, input_dim)
        h = self.mlp1(x_flat)                            # (B*N, hidden_dim)
        h = h.view(B, N, self.hidden_dim)                # (B, N, hidden_dim)
        h = self.ln_point(h)

        # seeds
        seed = self.seed_vectors.expand(B, -1, -1)       # (B, num_seeds, hidden_dim)
        seed_norm = self.ln_seed(seed)

        # PMA: queries = seed, keys/vals = h
        # MultiheadAttention returns (output, attn_weights)
        attended, _ = self.pma(query=seed_norm, key=h, value=h)  # (B, num_seeds, hidden_dim)

        # residual + norm (helps stability)
        attended = self.ln_seed(attended + seed)

        # flatten seeds and produce latent
        attended_flat = attended.reshape(B, self.num_seeds * self.hidden_dim)
        latent = self.mlp2(attended_flat)                 # (B, latent_dim)

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

# class TransformerHead(nn.Module):
#     def __init__(self, embedding_dim, out_dim, nhead=4, num_layers=2, dropout=0.1):
#         super().__init__()
#         self.embedding = nn.Linear(embedding_dim, 128)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=nhead, dropout=dropout)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
#         self.fc = nn.Linear(128, out_dim)
#     def forward(self, x):
#         x = self.embedding(x).unsqueeze(0)
#         x = self.transformer(x).squeeze(0)
#         return self.fc(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerHead(nn.Module):
    def __init__(self, embedding_dim, out_dim, ranges=None, nhead=4, num_layers=2,
                 dropout=0.1, eps=1e-8, pool='mean'):
        super().__init__()
        self.out_dim = out_dim
        self.use_scaling = ranges is not None
        self.eps = float(eps)
        self.pool = pool

        if self.use_scaling:
            ranges_t = torch.as_tensor(ranges, dtype=torch.float32)
            if ranges_t.shape != (out_dim, 2):
                raise ValueError(f"ranges must be (out_dim,2); got {ranges_t.shape}")
            low, high = ranges_t[:, 0], ranges_t[:, 1]
            width = F.softplus(high - low) + self.eps
            self.register_buffer("low", low)
            self.register_buffer("width", width)
        else:
            self.low = None
            self.width = None

        self.embedding = nn.Linear(embedding_dim, 128)
        enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=nhead,
                                               dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        # Accept (B, E), (B, S, E)
        if x.dim() == 2:
            x = x.unsqueeze(1)   # (B, 1, E)
        elif x.dim() == 3:
            pass                 # (B, S, E)
        else:
            raise ValueError(f"Unexpected input dim {x.dim()}")

        x = self.embedding(x)    # (B, S, 128)
        x = self.transformer(x)  # (B, S, 128)
        z = self.fc(x)           # (B, S, out_dim)

        # default pooling -> (B, out_dim)
        if self.pool == 'mean':
            z = z.mean(dim=1)
        elif self.pool == 'first':
            z = z[:, 0, :]
        elif self.pool in (None, 'none'):
            pass
        else:
            raise ValueError("pool must be 'mean', 'first', or None")

        # if not self.use_scaling:
        #     return z
        return z

        # apply per-dim scaling
        # if z.dim() == 3:
        #     return self.low.view(1, 1, -1) + self.width.view(1, 1, -1) * torch.sigmoid(z)
        # else:
        #     return self.low.view(1, -1) + self.width.view(1, -1) * torch.sigmoid(z)

class MLPHead(nn.Module):
    """Simple, stable MLP head for mapping (B, E) -> (B, out_dim)."""
    def __init__(self, embedding_dim, out_dim, hidden=256, dropout=0.0, use_layernorm=True):
        super().__init__()
        self.use_ln = use_layernorm
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden) if use_layernorm else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.LayerNorm(hidden//2) if use_layernorm else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, out_dim)
        )
        # safe init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        # Accept (B, E) or (B, 1, E)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        return self.net(x)    # (B, out_dim)