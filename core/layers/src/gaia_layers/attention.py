import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
import einops
from rotary_embedding_torch import RotaryEmbedding

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

class GQA(nn.Module):
    """
    Grouped Query Attention with RoPE following the authors implementation
    except using normal Linear layers and an open-source RoPE module.
    """

    def __init__(self, hidden_dim: int, head_dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads

        # The author implements RoPE where half the hidden dims are rotated.
        # This is a popular open-source implementation to avoid complexity in this codebase.
        self.rope = RotaryEmbedding(dim=head_dim // 2)

        # The output projection flattens all of the heads
        self.output_dim = self.head_dim * self.n_heads

        # Instead of computing three seperate layers for each of Q, K, and V we compute a single large layer
        # and then slice it up to get Q, K, and V with the expected dimensions
        self.attention_output_dim = (self.n_heads + 2 * n_kv_heads) * self.head_dim
        self.QKV = nn.Linear(hidden_dim, self.attention_output_dim, bias=False)

        self.output_projection = nn.Linear(self.output_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        batch_size, sequence_length, _ = x.shape

        # Compute attention heads all at once
        QKV = self.QKV(x)

        # Reshape to seperate attention heads
        QKV = QKV.view(
            batch_size,
            sequence_length,
            self.n_heads + 2 * self.n_kv_heads,
            self.head_dim,
        )

        # Extract Q, K, and V heads from the last dimension
        # [Q heads... | K heads... | V heads...]
        Q = QKV[:, :, : self.n_heads]
        K = QKV[:, :, self.n_heads : self.n_heads + self.n_kv_heads]
        V = QKV[:, :, self.n_heads + self.n_kv_heads :]

        # Apply RoPE
        Q = self.rope.rotate_queries_or_keys(Q)
        K = self.rope.rotate_queries_or_keys(K)

        # Swap Head and Sequence dimensions to prepare for attention computations
        Q = einops.rearrange(Q, "B S H D -> B H S D")
        K = einops.rearrange(K, "B S H D -> B H S D")
        V = einops.rearrange(V, "B S H D -> B H S D")

        # Compute attention scores using efficient PyTorch implementation
        attention_scores = scaled_dot_product_attention(Q, K, V)

        # Reshape output so we can feed it to output projection layer
        attention_scores = einops.rearrange(attention_scores, "B H S D -> B S H D")
        attention_scores = attention_scores.reshape(
            batch_size, sequence_length, self.output_dim
        )

        return self.output_projection(attention_scores)

class GQAKV(nn.Module):
    """
    This is identical to GQA above but with the addition of a naive KV-cache.
    """

        