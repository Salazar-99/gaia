import hydra
import os
from omegaconf import DictConfig
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity

from .train import create_dataloaders, create_model, train_step
from gaia_core import print_device_info

# Use absolute path for profile logs (relative to this file's location)
PROFILE_LOGS_DIR = Path(__file__).parent.parent.parent / "profile_logs"


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """Minimal training script for PyTorch profiling."""
    # Require CUDA device
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This profiling script requires CUDA.")
        exit(1)

    device = "cuda"
    print_device_info(device)

    # Set random seed
    torch.manual_seed(cfg.training.seed)

    # Create dataloaders (only need train loader for profiling)
    print("Creating dataloaders...")
    train_loader, train_metadata, _ = create_dataloaders(cfg, device)

    # Calculate total steps (needed for scheduler, but we'll only run a few steps)
    total_steps = (
        cfg.training.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        // cfg.training.global_batch_size
    )

    # Create model
    model, optimizer, scheduler = create_model(cfg, train_metadata, total_steps, device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    print("Starting profiling...")
    model.train()

    # Create profile logs directory
    os.makedirs(PROFILE_LOGS_DIR, exist_ok=True)
    print(f"Profile logs will be saved to: {PROFILE_LOGS_DIR}")

    # Warmup steps (not profiled)
    print("Running warmup steps...")
    warmup_steps = 3
    step = 0
    for _, batch, _ in train_loader:
        if step >= warmup_steps:
            break
        _ = train_step(model, optimizer, scheduler, batch, device)
        step += 1

    # Profiled steps
    print("Running profiled steps...")
    profile_steps = 5
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        step = 0
        for _, batch, _ in train_loader:
            if step >= profile_steps:
                break
            _ = train_step(model, optimizer, scheduler, batch, device)
            step += 1

    # Export trace
    trace_file = PROFILE_LOGS_DIR / "trace.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"Profiling complete. Trace saved to {trace_file}")
    print(f"Open in Chrome at chrome://tracing or use Perfetto UI")


if __name__ == "__main__":
    main()
