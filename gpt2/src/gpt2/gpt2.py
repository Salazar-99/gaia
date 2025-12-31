import torch
from torch import nn
from gaia_layers import Transformer, LayerNorm
from dataclasses import dataclass


@dataclass(frozen=True)
class GPT2Config:
    """Configuration for GPT2 model."""

    vocab_size: int = 50257  # Vocabulary size
    context_length: int = 1024  # Context length
    emb_dim: int = 768  # Embedding dimension
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of layers
    drop_rate: float = 0.1  # Dropout rate
    qkv_bias: bool = False  # Query-Key-Value bias

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.emb_dim % self.n_heads != 0:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {self.context_length}"
            )


class GPT2(nn.Module):
    """
    Configurable GPT2 Model
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.emb_dim)
        self.positional_embeddings = nn.Embedding(config.context_length, config.emb_dim)
        self.embeddings_dropout = nn.Dropout(config.drop_rate)
        self.transformer_blocks = nn.Sequential(
            *[
                Transformer(
                    config.emb_dim,
                    config.context_length,
                    config.n_heads,
                    dropout_rate=0,
                    qkv_bias=config.qkv_bias,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = LayerNorm(config.emb_dim)
        self.output_layer = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        batch_size, sequence_length = x.shape
        token_embeddings = self.token_embeddings(x)
        positional_embeddings = self.positional_embeddings(
            torch.arange(sequence_length, device=x.device)
        )
        x = token_embeddings + positional_embeddings
        x = self.embeddings_dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits
