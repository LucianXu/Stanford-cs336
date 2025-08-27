import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.Model.Modules import softmax, Linear, Embedding, RootMeanSquareLayerNorm, SwiGLUFeedFowardNeuralNerwork, CausalMultiHeadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float = 10000.0,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.RSMNorm_1 = RootMeanSquareLayerNorm(d_model=d_model, eps=1e-5, **factory_kwargs)
        self.RSMNorm_2 = RootMeanSquareLayerNorm(d_model=d_model, eps=1e-5, **factory_kwargs)
        self.AttentionLayer = CausalMultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, rope_type=True, theta=theta,**factory_kwargs
        )
        self.FeedForwardLayer = SwiGLUFeedFowardNeuralNerwork(
            d_model=d_model, d_ff=d_ff, **factory_kwargs
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.AttentionLayer(self.RSMNorm_1(x), token_positions)
        output = x + self.FeedForwardLayer(self.RSMNorm_2(x))
        return output

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, 
                 num_heads: int, d_ff: int, max_seq_len: int, theta: float = 10000.0,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.EmbaddingLayer = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, **factory_kwargs)
        self.TranformerBlocks = nn.ModuleList([TransformerBlock(
            d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len, theta=theta, **factory_kwargs
            ) for _ in range(num_layers)
        ])
        self.RSMNorm = RootMeanSquareLayerNorm(d_model=d_model, eps=1e-5, **factory_kwargs)
        self.OutputLayer = Linear(in_features=d_model, out_features=vocab_size, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = x.shape
        token_positions = torch.arange(seq_length, device=x.device).expand(batch_size, seq_length)
        x = self.EmbaddingLayer(x)
        for block in self.TranformerBlocks:
            x = block(x, token_positions)
        x = self.RSMNorm(x)
        output = self.OutputLayer(x)
        # output = softmax(output, dim=-1)
        return output