from curses import flash
import tiktoken
from typing import Sequence
import torch

# A tokenizer takes a string, parses it into tokens based on a precomputed vocabulary,
# then converts those tokens into integer token IDs.


# For now, I simply wrap tiktoken which contains pre-computed vocabularies for GPT2.
# In the future I'd like to implement my own BPE or other tokenizers
class Tokenizer:
    def __init__(self, model: str):
        self.tokenizer = tiktoken.get_encoding(model)

    def encode(self, x: str, allowed_special: set) -> list[int]:
        return self.tokenizer.encode(x, allowed_special=allowed_special)

    def decode(self, x: Sequence[int]) -> str:
        return self.tokenizer.decode(x)

    def text_to_token_ids(self, x: str) -> list[int]:
        encoded = self.encode(x, allowed_special={"<|endoftext|>"})
        # Add a batch dimension
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor

    def token_ids_to_text(self, token_ids: torch.Tensor):
        # remove batch dimension
        flat_ids = token_ids.squeeze(0)
        return self.decode(flat_ids)
