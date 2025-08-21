import torch
from torch import nn
from .attention import MultiHeadAttention
from .layers import GELU, FeedForward, LayerNorm

class Transformer(nn.Module):
    def __init__(self, emb_dim: int, ctx_length: int, n_heads: int, dropout_rate: float, qkv_bias: bool):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in = emb_dim,
            d_out = emb_dim,
            ctx_length=ctx_length,
            n_heads=n_heads,
            dropout=dropout_rate,
            qkv_bias=qkv_bias
        )
        self.ff = FeedForward(emb_dim)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor):
        # Attention block with residual connection
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + residual

        # Feedforward block with residual connection
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + residual
        return x

