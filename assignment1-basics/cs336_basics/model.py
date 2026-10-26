import math
import torch
import torch.nn as nn
from einops import einsum, rearrange

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

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Turn scores into probabilities along `dim` (all positive, summing to 1).
    """
    # Subtract the largest score first so exp() never overflows; the result is unchanged.
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)

    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V.
    Q: (..., queries, d_k), K: (..., keys, d_k), V: (..., keys, d_v) -> (..., queries, d_v).
    mask: True where a query may look at a key, False where it may not.
    """
    d_k = Q.shape[-1]

    # Score every (query, key) pair with a dot product, scaled down by √d_k: (..., queries, keys).
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)

    # Blocked pairs get -inf, so softmax gives them exactly zero weight.
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Each query's weights over the keys sum to 1.
    weights = softmax(scores, dim=-1)

    # Output = weighted average of the values: (..., queries, d_v).
    return einsum(weights, V, "... q k, ... k d_v -> ... q d_v")

class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention, with optional RoPE on the queries and keys.
    Pass `theta` and `max_seq_len` to turn RoPE on.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # each head works on its own slice of size d_k

        # Names match the reference weights: attn.q_proj.weight, ..., attn.output_proj.weight.
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE rotates each head's slice, so it works on d_k, not d_model.
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]

        # 1. Project the input into queries, keys and values: (..., seq, d_model) each.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Split d_model into heads: (..., seq, d_model) -> (..., heads, seq, d_k).
        q = rearrange(q, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        k = rearrange(k, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        v = rearrange(v, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)

        # 3. RoPE on queries and keys only (never values), the same rotation for every head.
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            # Add a heads axis so (..., seq) positions line up with (..., heads, seq, d_k).
            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 4. Causal mask: token i may look at tokens 0..i (True on and below the diagonal).
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # 5. Attention for all heads at once: (..., heads, seq, d_k).
        out = scaled_dot_product_attention(q, k, v, mask)

        # 6. Merge the heads back together: (..., heads, seq, d_k) -> (..., seq, d_model).
        out = rearrange(out, "... heads seq d_k -> ... seq (heads d_k)")

        # 7. Mix the heads' outputs with the output projection.
        return self.output_proj(out)

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block:
        y = x + attn(ln1(x))
        z = y + ffn(ln2(y))
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        # Names match the reference weights: ln1.weight, attn.q_proj.weight, ffn.w1.weight, ...
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # Each sub-layer: normalize, run it, then add the result back onto the input (residual).
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    """
    Token IDs -> embeddings -> num_layers Transformer blocks -> final RMSNorm -> logits over the vocabulary.
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # nn.ModuleList registers every block, so their weights are trained and saved as layers.0..., layers.1...
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )

        # The residual stream is never normalized inside the blocks, so normalize once before the output layer.
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # (batch, seq) -> (batch, seq, d_model)
        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x)

        # (batch, seq, d_model) -> (batch, seq, vocab_size): raw scores (logits), no softmax.
        return self.lm_head(self.ln_final(x))
    
        


