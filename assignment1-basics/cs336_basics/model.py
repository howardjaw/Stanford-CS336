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
    


