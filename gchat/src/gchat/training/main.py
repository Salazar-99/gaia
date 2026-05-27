from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from gchat.training.config import GChatConfig
from gchat.training.data import build_training_dataset
from gchat.training.model import GChat
from gchat.training.profile import run_profiling_loop
from gchat.training.train import run_training_loop


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain GChat on token batches (single-device JAX; no sharding)."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to a YAML training config (see gchat/conf/speedrun.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = GChatConfig.from_yaml(args.config).resolve()
    config.apply_runtime_env()

    training = config.training
    metrics_cfg = config.metrics

    print("Starting training")
    print(f"  run id:  {metrics_cfg.run_id}")
    print(f"  data:    {config.data.data_dir}")

    batches = build_training_dataset(config.data)
    batch_iter = iter(batches)

    rngs = nnx.Rngs(jax.random.key(training.seed))
    model = GChat(config.model, rngs)
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(training.learning_rate, mu_dtype=jnp.float32),
        wrt=nnx.Param,
    )

    if config.profiling.enabled:
        run_profiling_loop(batch_iter, model, optimizer, config.profiling)
    else:
        run_training_loop(config, batch_iter, model, optimizer)


if __name__ == "__main__":
    main()
