import tiktoken
from typing import Sequence

# For now, I simply wrap tiktoken
# In the future I'd like to implement my own BPE or other tokenizers
class Tokenizer:
    def __init__(self, model: str):
        self.tokenizer = tiktoken.get_encoding(model)

    def encode(self, x: str) -> list[int]:
        return self.tokenizer.encode(x)
    
    def decode(self, x: Sequence[int]) -> str:
        return self.tokenizer.decode(x)