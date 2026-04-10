from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from nanojaxpt.training.data import DatasetConfig, build_training_dataset
from nanojaxpt.training.model import GPTConfig, GPT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain GPT on token batches (single-device JAX; no sharding)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("climbmix_tokens"),
        help="Directory containing tokens-*.arrayrecord files.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable dataset shuffle.",
    )
    parser.add_argument(
        "--no-repeat",
        action="store_true",
        help="Do not repeat the dataset; stop when the input data is exhausted.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    data_config = DatasetConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        shuffle=not args.no_shuffle,
        repeat=not args.no_repeat,
    )
    batches = build_training_dataset(data_config)

    gpt_config = GPTConfig(sequence_len=args.sequence_length)
    rngs = nnx.Rngs(jax.random.key(args.seed))
    model = GPT(gpt_config, rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(args.learning_rate), wrt=nnx.Param)

    @nnx.jit
    def train_step(
        model: GPT,
        optimizer: nnx.Optimizer,
        inputs: jax.Array,
        targets: jax.Array,
    ) -> jax.Array:
        def loss_fn(m: GPT) -> jax.Array:
            logits = m(inputs)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, targets
            ).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    print("Starting training")
    for step, batch in enumerate(batches):
        inputs = jnp.asarray(batch["inputs"])
        targets = jnp.asarray(batch["targets"])
        loss = train_step(model, optimizer, inputs, targets)

        if step % args.log_every == 0:
            loss_val = float(jax.block_until_ready(loss))
            print(f"step {step:6d}  loss {loss_val:.4f}")


if __name__ == "__main__":
    main()
