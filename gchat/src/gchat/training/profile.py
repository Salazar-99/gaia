from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from gchat.training.config import ProfilingConfig
from gchat.training.model import GChat
from gchat.training.train import train_step


def run_profiling_loop(
    batch_iter: Iterator[dict[str, Any]],
    model: GChat,
    optimizer: nnx.Optimizer,
    profiling: ProfilingConfig,
) -> None:
    """Warm up, trace a short training window, and write a TensorBoard profile."""
    assert profiling.log_dir is not None

    print(f"Warming up profiler ({profiling.warmup_steps} steps)...")
    for _ in range(profiling.warmup_steps):
        batch = next(batch_iter)
        tokens = jnp.asarray(batch["tokens"])
        loss = train_step(model, optimizer, tokens)
        jax.block_until_ready(loss)

    # Give JAX/TPU background work a chance to finish before starting the trace.
    jax.effects_barrier()

    profiling.log_dir.mkdir(parents=True, exist_ok=True)
    log_dir = str(profiling.log_dir)
    print(f"Starting profile trace. Saving to {log_dir}")

    jax.profiler.start_trace(log_dir)
    try:
        for step in range(profiling.steps):
            start_time = time.time()
            batch = next(batch_iter)
            tokens = jnp.asarray(batch["tokens"])
            loss = train_step(model, optimizer, tokens)
            jax.block_until_ready(loss)
            print(f"Profile step {step} took {time.time() - start_time:.4f}s")
    finally:
        jax.profiler.stop_trace()

    print("Tracing finished!")
