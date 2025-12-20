import torch
import einops
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn import RMSNorm
from rotary_embedding_torch import RotaryEmbedding
from dataclasses import dataclass


class SwiGLU(nn.Module):
    """
    SwiGLU activation function of the form:

        SwiGLU(x) = Swish(xW1) * (xW2)

    Where * is element-wise multiplication

    A more efficient implementation would be to use a single linear layer and chunk it into two
    after the forward pass on it but I think this is easier to understand since it more closely resembles the
    mathematical formulation.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        # Note the TRM paper uses no bias terms
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        x1 = self.W1(x)
        x2 = self.W1(x)
        return F.silu(x1) * x2


class TRMMLP(nn.Module):
    """
    This is a simple two-layer MLP with the SwiGLU activation function.
    The author notes that two layers had the best performance across tasks.
    The author opts for their own CastedLinear layers with automatic type-casting for mixed-precision
    training and LeCun initialization. I opt for standard nn.Linear layers for simplicity.
    If there is training instability this is a place worth checking for improvements.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim), SwiGLU(), nn.Linear(hidden_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Attention(nn.Module):
    """
    Grouped Query Attention with RoPE following the authors implementation
    except using normal Linear layers and an open-source RoPE module.
    """

    def __init__(self, hidden_dim: int, head_dim: int, n_heads: int, n_kv_heads: int):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads

        # The author implements RoPE where half the hidden dims are rotated.
        # This is a popular open-source implementation to avoid complexity in this codebase.
        self.rope = RotaryEmbedding(hidden_dim // 2)

        # Instead of computing three seperate layers for each of Q, K, and V we compute a single large layer
        # and then slice it up to get Q, K, and V with the expected dimensions
        self.attention_output_dim = (self.n_heads + 2 * n_kv_heads) * self.head_dim
        self.QKV = nn.Linear(hidden_dim, self.output_dim, bias=False)

        # The output projection flattens all of the heads
        self.output_dim = self.head_dim * self.n_heads
        self.output_projection = nn.Linear(self.output_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        batch_size, sequence_length, _ = x.shape

        # Compute attention heads all at once
        QKV = self.QKV(x)

        # Reshape to seperate attention heads
        QKV = QKV.view(
            batch_size,
            sequence_length,
            self.num_heads + 2 * self.n_kv_heads,
            self.head_dim,
        )

        # Extract Q, K, and V heads from the last dimension
        # [Q heads... | K heads... | V heads...]
        Q = QKV[:, :, : self.n_heads]
        K = QKV[:, :, self.n_kv_heads : self.n_heads + self.n_kv_heads]
        V = QKV[:, :, self.n_heads + self.n_kv_heads :]

        # Apply RoPE
        Q = self.rope(Q)
        K = self.rope(K)

        # Swap Head and Sequence dimensions to prepare for attention computations
        Q = einops.rearrange(Q, "B S H D -> B H S D")
        K = einops.rearrange(K, "B S H D -> B H S D")
        V = einops.rearrange(V, "B S H D -> B H S D")

        # Compute attention scores using efficient PyTorch implementation
        attention_scores = scaled_dot_product_attention(Q, K, V)

        # Reshape output so we can feed it to output projection layer
        attention_scores = einops.rearrange(attention_scores, "B H S D -> B S H D")
        attention_scores = attention_scores.view(
            batch_size, sequence_length, self.output_dim
        )

        return self.output_projection(attention_scores)


class TRMNet(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int):
        self.attention = Attention(
            hidden_dim,
        )
        self.mlp = TRMMLP(hidden_dim)
        self.rms = RMSNorm()

    def forward(self, x: torch.Tensor):
        # Attention
        attn = self.attention(x)
        # Add the residual and Norm
        y = self.rms(x + attn)
        # TRMLMLP
        z = self.mlp(y)
        # Add the residual and Norm
        return self.rms(x + z)


@dataclass
class TRMConfig:
    hidden_dim: int
    # TODO: Add the rest of the model hyperparams


class TRM(nn.Module):
    """
    This is a Tiny Recursive Model implementation using attention, RoPE,
    a two-layer MLP with SwiGLU activation, and no ACT.
    """

    def __init__(self, T: int, n: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.T = T
        self.n = n
        self.embedding = nn.Embedding(
            num_embeddings=11, embedding_dim=hidden_dim
        )  # vocab_size=11 for sudoku (PAD + 0-9)
        self.net = TRMNet(hidden_dim, n_layers)
        # Non-learnable initial params to match original implementation
        self.y_init = nn.Buffer(torch.randn(hidden_dim), persistent=True)
        self.z_init = nn.Buffer(torch.randn(hidden_dim), persistent=True)

    def _latent_recursion(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, n: int
    ):
        # Update the latent n times
        for i in range(n):
            z = self.net(x + y + z)
        # Use the most recent answer and the latent to output the final answer
        y = self.net(y + z)
        return y, z

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        y = self.y_init
        z = self.z_init
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self._latent_recursion(x, y, z, self.n)
        y, _ = self._latent_recursion(x, y, z, self.n)
        return y
