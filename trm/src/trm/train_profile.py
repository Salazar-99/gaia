import hydra
from omegaconf import DictConfig
import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)

from trm.train_trm_gaia import create_dataloaders, create_model, train_step
from gaia_core import print_device_info


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

    # Setup profiler with CPU and CUDA activities
    # Schedule: wait=1 (skip first step), warmup=3 (warmup 3 steps), active=10 (profile 10 steps), repeat=1
    profiler_schedule = schedule(wait=1, warmup=3, active=10, repeat=1)

    print("Starting profiling...")
    model.train()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=profiler_schedule,
        on_trace_ready=tensorboard_trace_handler("./profile_logs"),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        step = 0
        max_steps = 10  # Profile a small number of steps

        for _, batch, _ in train_loader:
            if step >= max_steps:
                break

            _ = train_step(model, optimizer, scheduler, batch, device)
            prof.step()

            step += 1

    print("Profiling complete. Traces saved to ./profile_logs")
    print("View in TensorBoard with: tensorboard --logdir=./profile_logs")


if __name__ == "__main__":
    main()
