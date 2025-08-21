import tiktoken
from typing import Sequence

# A tokenizer takes a string, parses it into tokens based on a precomputed vocabulary,
# then converts those tokens into integer token IDs.

# For now, I simply wrap tiktoken which contains pre-computed vocabularies for GPT2.
# In the future I'd like to implement my own BPE or other tokenizers
class Tokenizer:
    def __init__(self, model: str):
        self.tokenizer = tiktoken.get_encoding(model)

    def encode(self, x: str) -> list[int]:
        return self.tokenizer.encode(x)
    
    def decode(self, x: Sequence[int]) -> str:
        return self.tokenizer.decode(x)