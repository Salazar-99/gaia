from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nanojaxpt.training.data import DatasetConfig, build_training_dataset
from nanojaxpt.training.evals import evaluate_bpb
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

    eval_group = parser.add_argument_group("evaluation")
    eval_group.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run BPB evaluation every N steps (0 = disabled).",
    )
    eval_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Batch size for BPB evaluation.",
    )
    eval_group.add_argument(
        "--eval-split-tokens",
        type=int,
        default=40 * 524_288,
        help="Total tokens to evaluate per BPB run (steps derived automatically).",
    )
    eval_group.add_argument(
        "--token-bytes-path",
        type=Path,
        default=None,
        help="Path to a .npy file of shape (vocab_size,) with byte lengths per token.",
    )
    return parser.parse_args()


def _load_token_bytes(path: Path) -> jax.Array:
    return jnp.asarray(np.load(path))


TEST_SHARD_GLOB = "test-tokens*.arrayrecord"


def _build_eval_state(args: argparse.Namespace) -> dict | None:
    """Prepare everything needed for periodic BPB evaluation, or return None."""
    if args.eval_every <= 0:
        return None
    if args.token_bytes_path is None:
        raise ValueError("--token-bytes-path is required when --eval-every > 0")

    test_files = sorted(args.data_dir.resolve().glob(TEST_SHARD_GLOB))
    if not test_files:
        raise FileNotFoundError(
            f"No files matching {TEST_SHARD_GLOB!r} in {args.data_dir.resolve()}. "
            "Run nanojaxpt-download to generate test-tokens.arrayrecord."
        )

    token_bytes = _load_token_bytes(args.token_bytes_path)
    tokens_per_step = args.eval_batch_size * args.sequence_length
    split_tokens = (args.eval_split_tokens // tokens_per_step) * tokens_per_step
    eval_steps = split_tokens // tokens_per_step

    eval_config = DatasetConfig(
        data_dir=args.data_dir,
        token_file_glob=TEST_SHARD_GLOB,
        batch_size=args.eval_batch_size,
        sequence_length=args.sequence_length,
        shuffle=False,
        repeat=True,
        seed=None,
    )
    print(
        f"BPB eval: every {args.eval_every} steps, "
        f"{eval_steps} steps/run ({split_tokens:,} tokens), "
        f"batch_size={args.eval_batch_size}, "
        f"test files: {[f.name for f in test_files]}"
    )
    return {
        "config": eval_config,
        "steps": eval_steps,
        "token_bytes": token_bytes,
    }


def _run_eval(model: GPT, eval_state: dict) -> float:
    eval_batches = build_training_dataset(eval_state["config"])
    bpb = evaluate_bpb(
        model, eval_batches, eval_state["steps"], eval_state["token_bytes"]
    )
    return bpb


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

    eval_state = _build_eval_state(args)

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

        if eval_state and step > 0 and step % args.eval_every == 0:
            bpb = _run_eval(model, eval_state)
            print(f"step {step:6d}  bpb  {bpb:.6f}")


if __name__ == "__main__":
    main()
