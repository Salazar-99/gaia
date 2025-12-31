from .gpt2 import GPT2, GPT2Config
from .tokenizer import Tokenizer
from .dataloader import create_dataloader
from gaia_core import train, plot_losses
from torch.utils.data import DataLoader
from dataclasses import dataclass
import torch
import argparse


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 5e-4
    n_epochs: int = 10
    batch_size: int = 2
    weight_decay: float = 0.1
    max_length: int = 256
    stride: int = 256


def train_gpt2(train_loader: DataLoader, val_loader: DataLoader, config: GPT2Config):
    # Create model and optimizer
    gpt2 = GPT2(config)
    optimizer = torch.optim.AdamW(
        gpt2.parameters(),
        lr=TrainingConfig.learning_rate,
        weight_decay=TrainingConfig.weight_decay,
    )
    # Run training
    train_losses, val_losses, tokens_seen = train(
        gpt2,
        train_loader,
        val_loader,
        optimizer,
        TrainingConfig.n_epochs,
        eval_freq=5,
        eval_iter=1,
    )
    # Report results
    return train_losses, val_losses, tokens_seen


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train GPT-2 model on text file")
    parser.add_argument("text_file", help="Path to the text file for training")
    args = parser.parse_args()

    # Read the text file
    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Split text into train/validation (90/10)
    split_idx = int(0.9 * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # Create tokenizer and dataloaders
    tokenizer = Tokenizer("gpt2")

    train_loader = create_dataloader(
        text=train_text,
        tokenizer=tokenizer,
        batch_size=TrainingConfig.batch_size,
        max_length=TrainingConfig.max_length,
        stride=TrainingConfig.stride,
        shuffle=True,
        drop_last=True,
    )

    val_loader = create_dataloader(
        text=val_text,
        tokenizer=tokenizer,
        batch_size=TrainingConfig.batch_size,
        max_length=TrainingConfig.max_length,
        stride=TrainingConfig.stride,
        shuffle=False,  # Don't shuffle validation data
        drop_last=False,  # Keep all validation data
    )

    # Create model config and train
    config = GPT2Config()
    train_losses, val_losses, tokens_seen_history = train_gpt2(
        train_loader, val_loader, config
    )

    # Create epochs_seen list for plotting
    epochs_seen = list(range(len(train_losses)))

    # Print results
    print("Training completed!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Total tokens seen: {tokens_seen_history[-1]}")

    # Generate and save loss plot
    plot_losses(epochs_seen, tokens_seen_history, train_losses, val_losses)
    print("Loss plot saved as loss.png")
