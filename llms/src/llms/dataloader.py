import torch
from torch.utils.data import Dataset, DataLoader
from .tokenizer import Tokenizer


class PretrainingDataset(Dataset):
    """
    A PretrainingDataset takes raw text in a single string and converts
    it into a dataset for pre-training an LLM via next-token prediction.
    It first converts the text to tokens, then the tokens are grouped into
    input and target chunks, finally the chunks are converted to PyTorch Tensors.
    This Dataset can then be used to instantiate a PyTorch DataLoader for training.
    """

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Loop over dataset
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            # Target is input shifted to the right by 1
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    # Return an (input, target) pair
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text: str,
    tokenizer: Tokenizer,
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle=True,
    drop_last=True,
    num_workers=0,
) -> DataLoader:
    dataset = PretrainingDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
