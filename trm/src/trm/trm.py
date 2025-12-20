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

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim), SwiGLU(), nn.Linear(hidden_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)

class TRMNet(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int):
        self.attention = GQA(
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
