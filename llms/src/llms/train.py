import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .device import get_device
from .tokenizer import Tokenizer
import matplotlib.pyplot as plt


def calc_loss_batch(
    input: torch.Tensor, output: torch.Tensor, model: nn.Module, device
):
    input = input.to(device)
    output = output.to(device)
    logits = model(input)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), output.flatten())
    return loss


def calc_loss_loader(
    data_loader: DataLoader, model: nn.Module, device, num_batches=None
):
    """
    calc_loss_loader computes the average loss of a model over a given DataLoader
    for the first num_batches batches in the DataLoader.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    n_epochs: int,
    tokenizer: Tokenizer,
    eval_freq: int,
    eval_iter: int,
):
    # Track training metrics
    train_losses, val_losses, tokens_seen_history = [], [], []
    tokens_seen = 0
    # Initialize at -1 to always trigger an eval after the first epoch
    global_step = -1
    device = get_device()
    model.to(device)

    # Core training loop
    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            # Eval and reporting
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen_history.append(tokens_seen)
                print(
                    f"Ep {epoch + 1}/{n_epochs} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

    return train_losses, val_losses, tokens_seen_history


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss.png")
