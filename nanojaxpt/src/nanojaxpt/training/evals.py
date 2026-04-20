"""
Functions for evaluating a base model.
"""

from __future__ import annotations

import math
from typing import Iterator

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from nanojaxpt.training.model import GPT


@nnx.jit
def _eval_step(
    model: GPT,
    x: jax.Array,
    y: jax.Array,
    token_bytes: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    logits = model(x)
    loss_flat = optax.softmax_cross_entropy_with_integer_labels(logits, y).reshape(-1)
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
    model: GPT,
    batches: Iterator[dict[str, jax.Array]],
    steps: int,
    token_bytes: jax.Array,
) -> float:
    """
    Returns bits per byte (bpb) — a vocab-size-independent metric.

    Instead of averaging the per-token loss, we accumulate the total nats and
    total target-token bytes across *steps* batches, then convert to bits/byte.

    Masking rules:
      1) Normal tokens are weighted by their byte length in *token_bytes*.
      2) Special tokens (byte length 0 in *token_bytes*) are excluded.
      3) Ignored targets (negative token ids) are excluded.

    Args:
        model:       GPT model (returns logits only).
        batches:     Iterator yielding {"inputs": ..., "targets": ...} dicts.
        steps:       Number of batches to consume from *batches*.
        token_bytes: 1-D int array of shape (vocab_size,) giving the byte
                     length of each token, or 0 for tokens to exclude.
    """
    total_nats = jnp.float32(0.0)
    total_bytes = jnp.int32(0)
    batch_iter = iter(batches)

    for _ in range(steps):
        batch = next(batch_iter)
        x = jnp.asarray(batch["inputs"])
        y = jnp.asarray(batch["targets"])
        batch_nats, batch_bytes = _eval_step(model, x, y, token_bytes)
        total_nats = total_nats + batch_nats
        total_bytes = total_bytes + batch_bytes

    total_nats_val = float(jax.block_until_ready(total_nats))
    total_bytes_val = int(jax.block_until_ready(total_bytes))

    if total_bytes_val == 0:
        return float("inf")
    return total_nats_val / (math.log(2) * total_bytes_val)
