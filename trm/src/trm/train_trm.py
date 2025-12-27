import os
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Any, Dict
import tqdm

from trm import TRM, TRMConfig
from trm.sudoku_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata


@dataclass
class TrainConfig:
    # Data
    data_path: str
    global_batch_size: int
    epochs: int

    # Model
    hidden_dim: int = 256
    T: int = 3  # Number of outer iterations
    n: int = 6  # Number of inner iterations
    vocab_size: int = 11  # PAD + 0-9 for sudoku

    # Training
    lr: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    min_lr: float = 1e-5  # Minimum LR for cosine annealing

    # Evaluation
    eval_interval: Optional[int] = None  # Evaluate every N epochs
    min_eval_interval: int = 0  # When to start evaluation
    test_data_path: Optional[str] = None  # Separate test data path
    max_eval_samples: Optional[int] = None  # Maximum number of samples to evaluate on (None = all)

    # Other
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: Optional[str] = None


@dataclass
class TrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: CosineAnnealingLR
    step: int
    total_steps: int


def create_dataloader(config: TrainConfig, split: str = "train"):
    """Create dataloader for sudoku dataset"""
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=[config.data_path],
            global_batch_size=config.global_batch_size,
            test_set_mode=(split == "test"),
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split=split,
    )

    # pin_memory is only beneficial for CUDA devices
    pin_memory = config.device == "cuda"
    dataloader = DataLoader(dataset, batch_size=None, num_workers=1, pin_memory=pin_memory)
    return dataloader, dataset.metadata


def create_model(config: TrainConfig, metadata: PuzzleDatasetMetadata, total_steps: int):
    """Create TRM model, optimizer, and scheduler"""
    model = TRM(
        T=config.T,
        n=config.n,
        hidden_dim=config.hidden_dim,
        vocab_size=config.vocab_size,
    )

    # Move to device
    model = model.to(config.device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )

    # Create cosine annealing LR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.min_lr)

    return model, optimizer, scheduler


def train_batch(config: TrainConfig, train_state: TrainState, batch: Any):
    """Train on a single batch"""
    train_state.step += 1

    # Move batch to device
    batch = {k: v.to(config.device) for k, v in batch.items()}

    # Forward pass
    outputs = train_state.model(batch["inputs"])

    # Compute loss (assuming we have targets in batch)
    if "labels" in batch:
        # Simple cross-entropy loss for sudoku
        loss = nn.CrossEntropyLoss()(
            outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1)
        )
    else:
        # If no labels, use a dummy loss (this would need to be adapted based on your specific setup)
        loss = torch.tensor(0.0, device=config.device, requires_grad=True)

    # Backward pass
    loss.backward()

    # Update parameters
    train_state.optimizer.step()
    train_state.optimizer.zero_grad()

    # Step the scheduler
    train_state.scheduler.step()

    # Get current LR
    lr = train_state.scheduler.get_last_lr()[0]

    return {"loss": loss.item(), "lr": lr, "step": train_state.step}


def save_checkpoint(config: TrainConfig, train_state: TrainState):
    """Save model checkpoint"""
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(
        {
            "model_state_dict": train_state.model.state_dict(),
            "optimizer_state_dict": train_state.optimizer.state_dict(),
            "scheduler_state_dict": train_state.scheduler.state_dict(),
            "step": train_state.step,
        },
        os.path.join(config.checkpoint_path, f"checkpoint_step_{train_state.step}.pt"),
    )


def load_checkpoint(config: TrainConfig, train_state: TrainState, checkpoint_path: str):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    train_state.model.load_state_dict(checkpoint["model_state_dict"])
    train_state.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    train_state.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    train_state.step = checkpoint["step"]


def evaluate_model(
    config: TrainConfig,
    model: nn.Module,
    eval_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on evaluation dataset
    
    Matches the paper implementation:
    - accuracy: average per-sequence accuracy (normalized per sequence, then averaged)
    - exact_accuracy: count of fully correct sequences
    """
    IGNORE_LABEL_ID = -100
    
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    # Metrics matching paper implementation
    count = 0  # Number of valid sequences
    accuracy_sum = 0.0  # Sum of per-sequence accuracies
    exact_accuracy_count = 0  # Count of fully correct sequences

    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            # Check if we've reached the maximum number of evaluation samples
            if config.max_eval_samples is not None and total_samples >= config.max_eval_samples:
                break
            
            # Adjust batch size if we're near the limit
            batch_size_to_use = global_batch_size
            if config.max_eval_samples is not None:
                remaining_samples = config.max_eval_samples - total_samples
                if remaining_samples < global_batch_size:
                    batch_size_to_use = remaining_samples
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(batch["inputs"])

            # Compute loss and accuracy
            if "labels" in batch:
                labels = batch["labels"]
                
                # Compute loss
                loss = nn.CrossEntropyLoss()(
                    outputs.view(-1, outputs.size(-1)), labels.view(-1)
                )
                total_loss += loss.item() * batch_size_to_use

                # Compute accuracy metrics matching paper implementation
                predictions = torch.argmax(outputs, dim=-1)
                
                # Create mask for valid tokens (excluding padding)
                mask = (labels != IGNORE_LABEL_ID)
                
                # Number of valid tokens per sequence: [batch_size]
                loss_counts = mask.sum(-1)
                
                # Avoid division by zero: [batch_size, 1] for broadcasting
                loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
                
                # Boolean tensor of correct tokens: [batch_size, seq_len]
                is_correct = mask & (predictions == labels)
                
                # Boolean tensor indicating if entire sequence is correct: [batch_size]
                seq_is_correct = is_correct.sum(-1) == loss_counts
                
                # Valid sequences (non-empty): [batch_size]
                valid_metrics = loss_counts > 0
                
                # Per-sequence accuracy: divide correct tokens by sequence length, then sum per sequence
                # Shape: [batch_size] - each element is the accuracy for that sequence
                per_seq_accuracy = (is_correct.to(torch.float32) / loss_divisor).sum(-1)
                
                # Accumulate metrics (only for valid sequences)
                valid_count = valid_metrics.sum().item()
                count += valid_count
                
                # Sum of per-sequence accuracies (only for valid sequences)
                accuracy_sum += torch.where(valid_metrics, per_seq_accuracy, torch.zeros_like(per_seq_accuracy)).sum().item()
                
                # Count of fully correct sequences
                exact_accuracy_count += (valid_metrics & seq_is_correct).sum().item()

            total_samples += batch_size_to_use
            
            # Break if we've reached the limit exactly
            if config.max_eval_samples is not None and total_samples >= config.max_eval_samples:
                break

    # Calculate metrics (matching paper normalization)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = accuracy_sum / count if count > 0 else 0.0
    exact_accuracy = exact_accuracy_count / count if count > 0 else 0.0

    return {
        "eval/loss": avg_loss,
        "eval/accuracy": accuracy,
        "eval/exact_accuracy": exact_accuracy,
        "eval/samples": total_samples,
        "eval/count": count,
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train TRM model on sudoku dataset")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda, mps, cpu). Default: auto-detect"
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate on (None = all). Default: None"
    )
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device.lower()
        # Validate device availability
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use a different device.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available. Please use a different device.")
    else:
        # Auto-detect: prefer CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Print device information
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print("  Using Metal Performance Shaders (MPS)")

    # Configuration
    config = TrainConfig(
        data_path="data/sudoku-extreme-1k-aug-1000",
        global_batch_size=768,
        epochs=60000,
        hidden_dim=512,
        T=3,
        n=6,
        lr=1e-4,
        checkpoint_path="checkpoints/trm_sudoku",
        eval_interval=100,  # Evaluate every 2 epochs
        min_eval_interval=100,  # Start evaluation after 1 epoch
        test_data_path="data/sudoku-extreme-1k-aug-1000",  # Use same data for test
        max_eval_samples=args.max_eval_samples,  # Configurable number of evaluation samples
        device=device,
    )

    # Set random seed
    torch.manual_seed(config.seed)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, train_metadata = create_dataloader(config, "train")

    # Calculate total steps
    total_steps = (
        config.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        // config.global_batch_size
    )

    # Create evaluation dataloader if test data path is provided
    eval_loader, eval_metadata = None, None
    if config.test_data_path:
        try:
            eval_config = TrainConfig(
                data_path=config.test_data_path,
                global_batch_size=config.global_batch_size,
                epochs=1,
                hidden_dim=config.hidden_dim,
                T=config.T,
                n=config.n,
                lr=config.lr,
                weight_decay=config.weight_decay,
                beta1=config.beta1,
                beta2=config.beta2,
                eval_interval=config.eval_interval,
                min_eval_interval=config.min_eval_interval,
                test_data_path=config.test_data_path,
                seed=config.seed,
                device=config.device,
                checkpoint_path=config.checkpoint_path,
            )
            eval_loader, eval_metadata = create_dataloader(eval_config, "test")
            print("Evaluation dataloader created successfully")
        except Exception as e:
            print(f"Warning: Could not create evaluation dataloader: {e}")
            eval_loader, eval_metadata = None, None

    # Create model
    print("Creating model...")
    model, optimizer, scheduler = create_model(config, train_metadata, total_steps)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Create training state
    train_state = TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=0,
        total_steps=total_steps,
    )

    # Training loop
    print(f"Starting training for {config.epochs} epochs ({total_steps} steps)...")
    model.train()

    progress_bar = tqdm.tqdm(total=total_steps, desc="Training")

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Training phase
        for batch_idx, (set_name, batch, global_batch_size) in enumerate(train_loader):
            if train_state.step >= total_steps:
                break

            # Train on batch
            metrics = train_batch(config, train_state, batch)

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(
                {"loss": f"{metrics['loss']:.4f}", "lr": f"{metrics['lr']:.2e}"}
            )

            # Log every 100 steps
            if train_state.step % 100 == 0:
                print(
                    f"Step {train_state.step}: loss={metrics['loss']:.4f}, lr={metrics['lr']:.2e}"
                )

        # Evaluation phase, only on the last epoch
        if (epoch == config.epochs -1):
            print(f"\nEvaluating at epoch {epoch + 1}...")

            # Run evaluation
            eval_metrics = evaluate_model(config, model, eval_loader, config.device)

            # Print evaluation results
            print(f"Evaluation Results:")
            for key, value in eval_metrics.items():
                print(f"  {key}: {value:.4f}")

        # Save checkpoint at end of epoch
        save_checkpoint(config, train_state)
        print(f"Saved checkpoint at step {train_state.step}")

    print("Training completed!")
    progress_bar.close()


if __name__ == "__main__":
    main()
