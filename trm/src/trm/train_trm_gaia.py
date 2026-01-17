import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Any, Dict, Tuple

from trm import TRM
from trm.sudoku_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata

# Import gaia core libraries
from gaia_core import (
    MetricsConfig,
    CheckpointConfig,
    get_device,
    print_device_info,
    TrainingContext,
)


def _create_dataset_config(
    cfg: DictConfig, data_path: str, split: str
) -> PuzzleDatasetConfig:
    """Create PuzzleDatasetConfig from training config."""
    return PuzzleDatasetConfig(
        seed=cfg.training.seed,
        dataset_paths=[data_path],
        global_batch_size=cfg.training.global_batch_size,
        test_set_mode=(split == "test"),
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )


def create_dataloader(
    cfg: DictConfig, data_path: str, device: str, split: str = "train"
) -> Tuple[DataLoader, PuzzleDatasetMetadata]:
    """Create dataloader for sudoku dataset."""
    dataset_config = _create_dataset_config(cfg, data_path, split)
    dataset = PuzzleDataset(dataset_config, split=split)

    # pin_memory is only beneficial for CUDA devices
    pin_memory = device == "cuda"
    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=1, pin_memory=pin_memory
    )
    return dataloader, dataset.metadata


def create_dataloaders(
    cfg: DictConfig, device: str
) -> Tuple[DataLoader, PuzzleDatasetMetadata, DataLoader]:
    """Create training and evaluation dataloaders.

    Returns:
        Tuple of (train_loader, train_metadata, eval_loader)
    """
    # Create training dataloader
    train_loader, train_metadata = create_dataloader(
        cfg, cfg.data.path, device, "train"
    )

    # Create evaluation dataloader
    eval_loader, _ = create_dataloader(cfg, cfg.data.test_path, device, "test")

    return train_loader, train_metadata, eval_loader


def create_model(
    cfg: DictConfig, metadata: PuzzleDatasetMetadata, total_steps: int, device: str
):
    """Create TRM model, optimizer, and scheduler."""
    model = TRM(
        T=cfg.model.T,
        n=cfg.model.n,
        hidden_dim=cfg.model.hidden_dim,
        vocab_size=cfg.model.vocab_size,
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.beta1, cfg.training.beta2),
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg.training.min_lr
    )

    return model, optimizer, scheduler


def train_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    batch: Any,
    device: str,
):
    """Train on a single batch."""
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(batch["inputs"])

    loss = nn.CrossEntropyLoss()(
        outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1)
    )

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    lr = scheduler.get_last_lr()[0]

    return {"loss": loss.item(), "lr": lr}


def evaluate_model(
    cfg: DictConfig,
    model: nn.Module,
    eval_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on evaluation dataset.

    Computes three main metrics:
    - loss: Average cross-entropy loss over all tokens
    - accuracy: Mean per-sequence token accuracy (fraction of correct tokens per sequence)
    - exact_accuracy: Fraction of sequences where ALL tokens are predicted correctly

    Labels with value -100 are ignored in all metric computations (standard PyTorch convention).
    """
    IGNORE_LABEL_ID = -100

    model.eval()

    # Accumulators for loss computation
    total_loss = 0.0
    total_samples = 0

    # Accumulators for accuracy metrics
    # count: number of sequences with at least one valid (non-ignored) token
    # accuracy_sum: sum of per-sequence accuracies (each in [0, 1])
    # exact_accuracy_count: number of sequences with 100% token accuracy
    count = 0
    accuracy_sum = 0.0
    exact_accuracy_count = 0

    with torch.inference_mode():
        for _, batch, global_batch_size in eval_loader:
            # Optionally limit the number of samples evaluated (for faster eval during training)
            batch_size_to_use = global_batch_size
            if cfg.eval.max_samples is not None:
                remaining_samples = cfg.eval.max_samples - total_samples
                if remaining_samples <= 0:
                    break
                if remaining_samples < global_batch_size:
                    batch_size_to_use = remaining_samples

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass: outputs shape is (batch_size, seq_len, vocab_size)
            outputs = model(batch["inputs"])

            labels = batch["labels"]  # shape: (batch_size, seq_len)

            # Compute cross-entropy loss (averaged over all non-ignored tokens)
            loss = nn.CrossEntropyLoss()(
                outputs.view(-1, outputs.size(-1)), labels.view(-1)
            )
            total_loss += loss.item() * batch_size_to_use

            # Get predicted token IDs by taking argmax over vocabulary dimension
            predictions = torch.argmax(outputs, dim=-1)  # shape: (batch_size, seq_len)

            # Create mask for valid (non-ignored) positions
            # mask[i, j] = True if labels[i, j] should be included in accuracy
            mask = labels != IGNORE_LABEL_ID  # shape: (batch_size, seq_len)

            # Count valid tokens per sequence
            loss_counts = mask.sum(-1)  # shape: (batch_size,)

            # Divisor for per-sequence accuracy (clamp to 1 to avoid division by zero)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(
                -1
            )  # shape: (batch_size, 1)

            # Boolean tensor: True where prediction matches label AND position is valid
            is_correct = mask & (predictions == labels)  # shape: (batch_size, seq_len)

            # A sequence is "exactly correct" if all its valid tokens are correctly predicted
            # i.e., the count of correct tokens equals the count of valid tokens
            seq_is_correct = is_correct.sum(-1) == loss_counts  # shape: (batch_size,)

            # Only compute metrics for sequences that have at least one valid token
            valid_metrics = loss_counts > 0  # shape: (batch_size,)

            # Per-sequence accuracy: (# correct tokens) / (# valid tokens)
            # Result is a value in [0, 1] for each sequence
            per_seq_accuracy = (is_correct.to(torch.float32) / loss_divisor).sum(-1)

            # Count sequences with valid metrics
            valid_count = valid_metrics.sum().item()
            count += valid_count

            # Sum up per-sequence accuracies (only for valid sequences)
            accuracy_sum += (
                torch.where(
                    valid_metrics,
                    per_seq_accuracy,
                    torch.zeros_like(per_seq_accuracy),
                )
                .sum()
                .item()
            )

            # Count sequences that are both valid AND exactly correct
            exact_accuracy_count += (valid_metrics & seq_is_correct).sum().item()

            total_samples += batch_size_to_use

    # Compute final metrics as averages
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = (
        accuracy_sum / count if count > 0 else 0.0
    )  # mean of per-sequence accuracies
    exact_accuracy = (
        exact_accuracy_count / count if count > 0 else 0.0
    )  # fraction of perfect sequences

    return {
        "eval/loss": avg_loss,
        "eval/accuracy": accuracy,
        "eval/exact_accuracy": exact_accuracy,
        "eval/samples": total_samples,
        "eval/count": count,
    }


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # Validate core configs with Pydantic
    metrics_config = MetricsConfig(**cfg.metrics)
    checkpoint_config = CheckpointConfig(**cfg.checkpoint)

    # Initialize training context (dashboard, metrics, checkpointing)
    ctx = TrainingContext(metrics_config, checkpoint_config)
    print(f"Metrics initialized with run_id={ctx.run_id}")
    if ctx.checkpoint_saver:
        print(f"Checkpoint saver initialized: {checkpoint_config.path}")

    # Get device
    device = get_device(cfg.device)
    print_device_info(device)

    # Set random seed
    torch.manual_seed(cfg.training.seed)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, train_metadata, eval_loader = create_dataloaders(cfg, device)

    # Calculate total steps
    total_steps = (
        cfg.training.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        // cfg.training.global_batch_size
    )

    # Create model
    model, optimizer, scheduler = create_model(cfg, train_metadata, total_steps, device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Training loop
    model.train()
    step = 0

    for epoch in range(cfg.training.epochs):
        for _, batch, _ in train_loader:
            if step >= total_steps:
                break

            step += 1
            metrics = train_batch(model, optimizer, scheduler, batch, device)

            ctx.dashboard.update(
                training_loss=metrics["loss"],
                epoch=epoch + 1,
            )

            ctx.gauges["training_loss"].set(metrics["loss"])
            ctx.gauges["lr"].set(metrics["lr"])

        # Evaluation phase (every eval.interval epochs)
        if (epoch + 1) % cfg.eval.interval == 0:
            eval_metrics = evaluate_model(cfg, model, eval_loader, device)

            ctx.dashboard.update(validation_loss=eval_metrics["eval/loss"])

            ctx.gauges["validation_loss"].set(eval_metrics["eval/loss"])
            ctx.gauges["accuracy"].set(eval_metrics["eval/accuracy"])
            ctx.gauges["exact_accuracy"].set(eval_metrics["eval/exact_accuracy"])

        # Save checkpoint
        if (epoch + 1) % checkpoint_config.interval == 0:
            if ctx.checkpoint_saver is not None:
                filename = f"checkpoint_step_{step}.pt"
                ctx.checkpoint_saver.save(
                    global_step=step,
                    model=model,
                    optimizer=optimizer,
                    filename=filename,
                    scheduler=scheduler,
                )

    ctx.close()


if __name__ == "__main__":
    main()
