import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, ctx_length: int, dropout: float, n_heads: int, qkv_bias=False):
        super().__init__()
        assert (d_out % n_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.V = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",
                             torch.triu(torch.ones(ctx_length, ctx_length), diagonal = 1))
        
    def forward(self, x: torch.Tensor):
        batch, n_tokens, d_in = x.shape
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        # Split K, Q, and V into multiple heads.
        # The final shape is: (batch, n_heads, n_tokens, head_dim)
        k = k.view(batch, n_tokens, self.n_heads, self.head_dim)
        k = k.transpose(1,2)
        q = q.view(batch, n_tokens, self.n_heads, self.head_dim)
        q = q.transpose(1,2)
        v = v.view(batch, n_tokens, self.n_heads, self.head_dim)
        v = v.transpose(1,2)
        # Compute attention scores, thanks to the previous transformations
        # we can compute all heads in parallel
        attn_scores = q @ k.transpose(2,3)
        # Causal masking with -inf trick to easily feed into softmax
        bool_mask = self.mask.bool()[:n_tokens, :n_tokens]
        attn_scores.masked_fill_(bool_mask, -torch.inf)
        # Compute attention weights and scale them according to Scaled Dot Product Attention recipe
        attn_weights = torch.softmax(attn_scores / k.shape[-1]**0.5, dim=-1)
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        # Compute context vectors and combine heads
        context_vec = (attn_weights @ v).transpose(1,2)
        context_vec = context_vec.contiguous().view(batch, n_tokens, self.d_out)
        # Final linear layer
        context_vec = self.out_proj(context_vec)
        return context_vec



        