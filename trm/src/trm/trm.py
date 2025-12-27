import torch
from torch import nn
from torch.nn import RMSNorm
from dataclasses import dataclass
from core import GQA, SwiGLU


class TRMMLP(nn.Module):
    """
    This is a simple two-layer MLP with the SwiGLU activation function.
    The author notes that two layers had the best performance across tasks.
    The author opts for their own CastedLinear layers with automatic type-casting for mixed-precision
    training and LeCun initialization. I opt for standard nn.Linear layers for simplicity.
    If there is training instability this is a place worth checking for improvements.
    """

    def __init__(self, hidden_dim: int, mlp_hidden_dim: int | None = None):
        super().__init__()
        # Default to 4x hidden_dim for MLP intermediate size (common practice)
        if mlp_hidden_dim is None:
            mlp_hidden_dim = hidden_dim * 4
        self.fc_in = nn.Linear(hidden_dim, mlp_hidden_dim)
        self.swiglu = SwiGLU(mlp_hidden_dim, mlp_hidden_dim)
        self.fc_out = nn.Linear(mlp_hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor):
        x = self.fc_in(x)
        x = self.swiglu(x)
        x = self.fc_out(x)
        return x


class TRMNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        head_dim: int = 64,
        n_heads: int = 4,
        n_kv_heads: int = 4,
    ):
        super().__init__()
        self.attention = GQA(
            hidden_dim=hidden_dim,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        )
        self.mlp = TRMMLP(hidden_dim)
        self.rms = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        # Attention
        attn = self.attention(x)
        # Add the residual and Norm
        y = self.rms(x + attn)
        # TRMLMLP
        z = self.mlp(y)
        # Add the residual and Norm
        return self.rms(y + z)


@dataclass
class TRMConfig:
    hidden_dim: int
    # TODO: Add the rest of the model hyperparams


class TRM(nn.Module):
    """
    This is a Tiny Recursive Model implementation using attention, RoPE,
    a two-layer MLP with SwiGLU activation, and no ACT.
    """

    def __init__(self, T: int, n: int, hidden_dim: int, vocab_size: int = 11):
        super().__init__()
        self.T = T
        self.n = n
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=hidden_dim
        )  # vocab_size=11 for sudoku (PAD + 0-9)
        self.net = TRMNet(hidden_dim)
        # Output projection to vocabulary logits
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
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
        # Project to vocabulary logits
        return self.lm_head(y)
