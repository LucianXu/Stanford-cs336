import torch
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        sigma = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return output
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        sigma = 1
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]
    
class RootMeanSquareLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        output = (x / norm * self.weight).to(in_dtype)
        return output

class SwiGLUFeedFowardNeuralNerwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.linear1 = Linear(d_model, d_ff, **factory_kwargs)
        self.linear2 = Linear(d_ff, d_model, **factory_kwargs)
        self.linear3 = Linear(d_model, d_ff, **factory_kwargs)

    def SiLU(self, x: torch.Tensor) -> torch.Tensor: 
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear2(self.SiLU(self.linear1(x)) * self.linear3(x))
        return output
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, 
                 device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        factory_kwargs = {'device': device}
        temp = torch.outer(
            torch.arange(max_seq_len, **factory_kwargs, dtype=torch.float64),
            1.0 / (theta ** (torch.arange(0, d_k, 2, **factory_kwargs, dtype=torch.float64) / d_k))
        )
        """
        The formula for \\theta_{i,k} in Section 2.3.5 of the document is incorrect; it should be \\theta_{i,k} = frac{i}{\\Theta^{2(k-1)d}}
        """
        self.register_buffer('sin', torch.sin(temp).to(torch.float32), persistent=False)
        self.register_buffer('cos', torch.cos(temp).to(torch.float32), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos[token_positions], self.sin[token_positions]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2], x_rotated[..., 1::2] = x_rotated_even, x_rotated_odd
        return x_rotated

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_exp = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    output = x_exp / x_exp.sum(dim=dim, keepdim=True) 
    return output

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.scale = 1.0 / (d_k ** 0.5)

    def forward(self, Q, K, V, mask=None):
        attn_scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == False, float('-inf'))
        attn_weights = softmax(attn_scores, dim=-1)
        output = einsum(attn_weights, V, "... q k, ... k v -> ... q v")
        return output

"""
class CausalSingleHeadSelfAttention_V1(nn.Module):
    def __init__(self, d_model: int, d_k: int, max_seq_len: int, rope_type: bool = True, theta: float = 10000.0,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.rope_type = rope_type
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.Q = Linear(d_model, d_k, **factory_kwargs)
        self.K_trans = Linear(d_model, d_k, **factory_kwargs)
        self.V = Linear(d_model, d_k, **factory_kwargs)
        self.attention = ScaledDotProductAttention(d_k=d_k)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)
        if rope_type:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self.Q(x), self.K_trans(x), self.V(x)
        if self.rope:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        output = self.attention(q, k, v, mask=self.causal_mask)
        return output

class CausalMultiHeadSelfAttention_V1(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, rope_type: bool = True, theta: float = 10000.0,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        assert(d_model % num_heads == 0)
        d_k = d_model // num_heads
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.heads = nn.ModuleList([
            CausalSingleHeadSelfAttention(
                d_model=d_model, d_k=d_k, max_seq_len=max_seq_len, rope_type=rope_type, theta=theta, **factory_kwargs)
            for _ in range(num_heads)
        ])
        self.O = nn.Linear(d_model, d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        head_output = [attention(x, token_positions) for attention in self.heads]
        output = self.O(torch.cat(head_output, dim=-1))
        return output
"""

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, rope_type: bool = True, theta: float = 10000.0,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        assert(d_model % num_heads == 0)
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope_type = rope_type
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.Q = Linear(d_model, d_model, **factory_kwargs)
        self.K_trans = Linear(d_model, d_model, **factory_kwargs)
        self.V = Linear(d_model, d_model, **factory_kwargs)
        self.O = Linear(d_model, d_model, **factory_kwargs)
        self.attention = ScaledDotProductAttention(d_k=self.d_k)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)
        if rope_type:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q, K, V = self.Q(x), self.K_trans(x), self.V(x)
        Q = rearrange(Q, "batch_size seq_length (num_heads d_k) -> batch_size num_heads seq_length d_k", 
                      num_heads=self.num_heads, d_k=self.d_k)
        K = rearrange(K, "batch_size seq_length (num_heads d_k) -> batch_size num_heads seq_length d_k", 
                      num_heads=self.num_heads, d_k=self.d_k)
        V = rearrange(V, "batch_size seq_length (num_heads d_k) -> batch_size num_heads seq_length d_k", 
                      num_heads=self.num_heads, d_k=self.d_k)
        if self.rope_type:
            Q, K = self.rope(Q, token_positions), self.rope(K, token_positions)
        output = self.attention(Q, K, V, mask=self.causal_mask[..., :x.size(1), :x.size(1)])
        output = rearrange(output, "batch_size num_heads seq_length d_k -> batch_size seq_length (num_heads d_k)", 
                           num_heads=self.num_heads, d_k=self.d_k)
        output = self.O(output)
        return output
