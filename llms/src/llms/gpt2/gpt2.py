import torch
from torch import nn
from ..transformer import Transformer
from ..layers import LayerNorm

class GPT2(nn.Module):
    """
    GPT2 Model
    """
    def __init__(self, vocab_size: int, emb_dim: int, ctx_length: int, dropout_rate: float, n_layers: int, n_heads: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.positional_embeddings = nn.Embedding(ctx_length, emb_dim)
        self.embeddings_dropout = nn.Dropout(dropout_rate)
        self.transformer_blocks = nn.Sequential(
            *[Transformer(emb_dim, ctx_length, n_heads, dropout_rate=0, qkv_bias=False) for _ in range(n_layers)]
        )
        self.final_norm = LayerNorm(emb_dim)
        self.output_layer = nn.Linear(emb_dim, vocab_size, bias=False)
    
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
        
