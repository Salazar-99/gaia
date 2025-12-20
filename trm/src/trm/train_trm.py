import os
import math
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Any, Dict
import tqdm

from trm import TRM, TRMConfig
from sudoku_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata


class EMAHelper:
    """Exponential Moving Average helper for model parameters"""

    def __init__(self, mu: float = 0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module: nn.Module):
        """Register model parameters for EMA tracking"""
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module: nn.Module):
        """Update EMA parameters"""
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module: nn.Module):
        """Apply EMA parameters to module"""
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module: nn.Module) -> nn.Module:
        """Create a copy of module with EMA parameters"""
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


@dataclass
class TrainConfig:
    # Data
    data_path: str
    global_batch_size: int
    epochs: int

    # Model
    hidden_dim: int = 256
    n_layers: int = 4
    T: int = 5  # Number of outer iterations
    n: int = 3  # Number of inner iterations

    # Training
    lr: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95

    # EMA
    ema: bool = False
    ema_rate: float = 0.999

    # Evaluation
    eval_interval: Optional[int] = None  # Evaluate every N epochs
    min_eval_interval: int = 0  # When to start evaluation
    test_data_path: Optional[str] = None  # Separate test data path

    # Other
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: Optional[str] = None


@dataclass
class TrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
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

    dataloader = DataLoader(dataset, batch_size=None, num_workers=1, pin_memory=True)
    return dataloader, dataset.metadata


def create_model(config: TrainConfig, metadata: PuzzleDatasetMetadata):
    """Create TRM model"""
    model = TRM(
        T=config.T, n=config.n, hidden_dim=config.hidden_dim, n_layers=config.n_layers
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

    return model, optimizer


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
):
    """Cosine learning rate schedule with warmup"""
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return base_lr * (
        min_ratio
        + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    )


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

    # Update learning rate
    lr = cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=config.lr,
        num_warmup_steps=min(100, train_state.total_steps // 10),
        num_training_steps=train_state.total_steps,
        min_ratio=0.1,
    )

    for param_group in train_state.optimizer.param_groups:
        param_group["lr"] = lr

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
            "step": train_state.step,
        },
        os.path.join(config.checkpoint_path, f"checkpoint_step_{train_state.step}.pt"),
    )


def load_checkpoint(config: TrainConfig, train_state: TrainState, checkpoint_path: str):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    train_state.model.load_state_dict(checkpoint["model_state_dict"])
    train_state.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    train_state.step = checkpoint["step"]


def evaluate_model(
    config: TrainConfig,
    model: nn.Module,
    eval_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on evaluation dataset"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(batch["inputs"])

            # Compute loss
            if "labels" in batch:
                loss = nn.CrossEntropyLoss()(
                    outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1)
                )
                total_loss += loss.item() * global_batch_size

                # Compute accuracy for sudoku
                predictions = torch.argmax(outputs, dim=-1)
                labels = batch["labels"]

                # Count correct predictions (excluding padding)
                mask = labels != -100  # Assuming -100 is padding token
                correct = (predictions == labels) & mask
                correct_predictions += correct.sum().item()
                total_predictions += mask.sum().item()

            total_samples += global_batch_size

    # Calculate metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return {
        "eval/loss": avg_loss,
        "eval/accuracy": accuracy,
        "eval/samples": total_samples,
    }


def main():
    # Configuration
    config = TrainConfig(
        data_path="data/sudoku-extreme-1k-aug-1000",
        global_batch_size=32,
        epochs=10,
        hidden_dim=256,
        n_layers=4,
        T=5,
        n=3,
        lr=1e-4,
        checkpoint_path="checkpoints/trm_sudoku",
        ema=True,
        ema_rate=0.999,
        eval_interval=2,  # Evaluate every 2 epochs
        min_eval_interval=1,  # Start evaluation after 1 epoch
        test_data_path="data/sudoku-extreme-1k-aug-1000",  # Use same data for test
    )

    # Set random seed
    torch.manual_seed(config.seed)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, train_metadata = create_dataloader(config, "train")

    # Create evaluation dataloader if test data path is provided
    eval_loader, eval_metadata = None, None
    if config.test_data_path:
        try:
            eval_config = TrainConfig(
                data_path=config.test_data_path,
                global_batch_size=config.global_batch_size,
                epochs=1,
                hidden_dim=config.hidden_dim,
                n_layers=config.n_layers,
                T=config.T,
                n=config.n,
                lr=config.lr,
                weight_decay=config.weight_decay,
                beta1=config.beta1,
                beta2=config.beta2,
                ema=config.ema,
                ema_rate=config.ema_rate,
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
    model, optimizer = create_model(config, train_metadata)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Setup EMA if enabled
    ema_helper = None
    if config.ema:
        print("Setting up EMA...")
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(model)

    # Calculate total steps
    total_steps = (
        config.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        // config.global_batch_size
    )

    # Create training state
    train_state = TrainState(
        model=model, optimizer=optimizer, step=0, total_steps=total_steps
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

            # Update EMA if enabled
            if config.ema and ema_helper is not None:
                ema_helper.update(model)

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

        # Evaluation phase
        if (
            eval_loader is not None
            and config.eval_interval is not None
            and epoch >= config.min_eval_interval
            and (epoch + 1) % config.eval_interval == 0
        ):
            print(f"\nEvaluating at epoch {epoch + 1}...")

            # Use EMA model for evaluation if available
            eval_model = model
            if config.ema and ema_helper is not None:
                print("Using EMA model for evaluation...")
                eval_model = ema_helper.ema_copy(model)

            # Run evaluation
            eval_metrics = evaluate_model(
                config, eval_model, eval_loader, config.device
            )

            # Print evaluation results
            print(f"Evaluation Results:")
            for key, value in eval_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Clean up EMA copy if created
            if config.ema and ema_helper is not None and eval_model != model:
                del eval_model

        # Save checkpoint at end of epoch
        save_checkpoint(config, train_state)
        print(f"Saved checkpoint at step {train_state.step}")

    print("Training completed!")
    progress_bar.close()


if __name__ == "__main__":
    main()
