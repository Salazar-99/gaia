"""
BPB evaluation for the speedrun test shard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from gchat.training.config import GChatConfig
from gchat.training.data import build_token_batches, test_token_file
from gchat.training.model import GChat


@dataclass
class EvalState:
    data_dir: str
    batch_size: int
    sequence_length: int
    steps: int
    token_bytes: jax.Array


@nnx.jit
def _eval_step(
    model: GChat,
    tokens: jax.Array,
    token_bytes: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    logits = model(x)
    loss_flat = optax.softmax_cross_entropy_with_integer_labels(logits, y).astype(
        jnp.float32
    ).reshape(-1)
    y_flat = y.reshape(-1)

    valid = y_flat >= 0
    y_safe = jnp.where(valid, y_flat, 0)
    num_bytes_flat = jnp.where(valid, token_bytes[y_safe], 0)

    countable = num_bytes_flat > 0
    batch_nats = jnp.sum(jnp.where(countable, loss_flat, 0.0))
    batch_bytes = jnp.sum(num_bytes_flat)
    return batch_nats, batch_bytes


# Copied from nanochat
def evaluate_bpb(
    model: GChat,
    batches: Iterator[dict[str, jax.Array]],
    steps: int,
    token_bytes: jax.Array,
) -> float:
    """Accumulate loss over byte-weighted target tokens and return bits/byte."""
    all_nats: list[jax.Array] = []
    all_bytes: list[jax.Array] = []
    batch_iter = iter(batches)

    for _ in range(steps):
        batch = next(batch_iter)
        tokens = jnp.asarray(batch["tokens"])
        batch_nats, batch_bytes = _eval_step(model, tokens, token_bytes)
        all_nats.append(batch_nats)
        all_bytes.append(batch_bytes)

    total_nats_val = float(jnp.stack(all_nats).sum())
    total_bytes_val = int(jnp.stack(all_bytes).sum())

    if total_bytes_val == 0:
        return float("inf")
    return total_nats_val / (math.log(2) * total_bytes_val)


def _load_token_bytes(path: str) -> jax.Array:
    with epath.Path(path).open("rb") as f:
        return jnp.asarray(np.load(f))


def build_eval_state(config: GChatConfig) -> EvalState:
    data = config.data
    eval_cfg = config.eval
    token_bytes_path = data.token_bytes_path
    assert token_bytes_path is not None

    token_bytes = _load_token_bytes(token_bytes_path)
    tokens_per_step = eval_cfg.batch_size * data.sequence_length
    split_tokens = (eval_cfg.split_tokens // tokens_per_step) * tokens_per_step
    eval_steps = split_tokens // tokens_per_step

    print(
        f"BPB eval: every {eval_cfg.every} steps, "
        f"{eval_steps} steps/run ({split_tokens:,} tokens), "
        f"batch_size={eval_cfg.batch_size}, "
        f"test file: {test_token_file(data.data_dir)}"
    )
    return EvalState(
        data_dir=data.data_dir,
        batch_size=eval_cfg.batch_size,
        sequence_length=data.sequence_length,
        steps=eval_steps,
        token_bytes=token_bytes,
    )


def run_eval(model: GChat, eval_state: EvalState) -> float:
    eval_batches = build_token_batches(
        files=[test_token_file(eval_state.data_dir)],
        batch_size=eval_state.batch_size,
        sequence_length=eval_state.sequence_length,
        shuffle=False,
    )
    return evaluate_bpb(model, eval_batches, eval_state.steps, eval_state.token_bytes)
