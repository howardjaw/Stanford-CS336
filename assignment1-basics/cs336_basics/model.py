import math
import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    """
    Linear layer without bias: y = x Wᵀ.
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # Initialize weights
        W = torch.empty(
            out_features,
            in_features,
            device = device,
            dtype = dtype
        )

        self.weight = nn.Parameter(W) # Initialize parameters
       
        sigma = math.sqrt(2/(out_features + in_features)) # Define std.dev as the square root of 2 over the sum of in and out feature
        
        # Randomnize weights using [-3sigma, 3sigma]
        nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = sigma,
            a = -3*sigma,
            b = 3*sigma
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear transformation: XW^T = Y.
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    """
    Embedding class for token lookups.
    """       
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # Initialize embeddings
        E = torch.empty(
            num_embeddings,
            embedding_dim,
            device = device,
            dtype = dtype
        )
        self.weight = nn.Parameter(E) # Initialize parameters
        
        # Randomnize weights using [-3sigma, 3sigma]
        nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = 1,
            a = -3,
            b = 3
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    """
    RMSNorm class for normalization.
    """
    def __init__(self, d_model, eps = 1e-5, device = None, dtype = None):
        super().__init__()
        self.eps = eps
        W = torch.ones(
            d_model,
            device = device,
            dtype = dtype
        )
        self.weight = nn.Parameter(W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        root_mean_square = torch.sqrt(mean_square+self.eps)
        normalized = x /root_mean_square
        result = normalized * self.weight
        return result.to(in_dtype)

def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    def forward(self, x):
        a = self.w1(x)
        b = self.w3(x)
        gated = silu(a)*b
        out = self.w2(gated)
        return out

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE: rotates each pair of numbers (0, 1), (2, 3), ... in a query or key vector
    by an angle that grows with the token's position.
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        # One speed per pair: theta^(-2k/d_k) for k = 0, 1, ..., d_k/2 - 1.
        two_k = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        speeds = theta ** (-two_k / d_k)

        # Positions 0, 1, ..., max_seq_len - 1.
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # Angle for every (position, pair) combination: shape (max_seq_len, d_k/2).
        angles = einsum(positions, speeds, "pos, pair -> pos pair")

        # Fixed tables: not trained, move with .to(device), not saved in checkpoints.
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Look up the angles for these positions: shape (..., seq_len, d_k/2).
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # Split each pair into its first and second number: (..., seq_len, d_k/2) each.
        first = x[..., 0::2]
        second = x[..., 1::2]

        # Rotate each pair: (a, b) -> (a cos - b sin, a sin + b cos).
        rotated_first = first * cos - second * sin
        rotated_second = first * sin + second * cos

        # Put the numbers back in their original order: first0, second0, first1, second1, ...
        # cos/sin are float32, so convert back in case x was lower precision (e.g. bf16).
        rotated = torch.stack([rotated_first, rotated_second], dim=-1).flatten(-2)
        return rotated.to(x.dtype)
    
        


