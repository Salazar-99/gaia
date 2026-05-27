from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from gaia_metrics import initialize_metrics

from gchat.training.checkpoint import save_model
from gchat.training.config import GChatConfig
from gchat.training.evals import build_eval_state, run_eval
from gchat.training.model import GChat


class MetricsWriter:
    """Background-thread metric writer — training loop enqueues, IO thread writes.

    queue.put() is non-blocking so JAX async dispatch is never stalled.
    float() materialization and force_flush() network IO both happen off the hot path.
    """

    def __init__(
        self,
        log_every: int,
        training_loss_gauge: Any,
        validation_loss_gauge: Any,
    ):
        self.log_every = log_every
        self.training_loss_gauge = training_loss_gauge
        self.validation_loss_gauge = validation_loss_gauge
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def log_train(self, step: int, loss: Any) -> None:
        self._queue.put({"type": "train", "step": step, "loss": loss})

    def log_eval(self, step: int, bpb: float) -> None:
        self._queue.put({"type": "eval", "step": step, "bpb": bpb})

    def flush(self) -> None:
        self._queue.put(None)
        self._thread.join()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            if item["type"] == "train":
                self._write_train(item)
            else:
                self._write_eval(item)

    def _write_train(self, item: dict[str, Any]) -> None:
        if self.log_every <= 0:
            return
        step = int(item["step"])
        if step % self.log_every != 0:
            return
        loss_val = float(item["loss"])
        print(f"step {step:6d}  loss {loss_val:.4f}")
        self.training_loss_gauge.set(loss_val)

    def _write_eval(self, item: dict[str, Any]) -> None:
        step = int(item["step"])
        bpb = item["bpb"]
        print(f"step {step:6d}  bpb  {bpb:.6f}")
        self.validation_loss_gauge.set(item["bpb"])


@nnx.jit
def train_step(
    model: GChat,
    optimizer: nnx.Optimizer,
    tokens: jax.Array,
) -> jax.Array:
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    def loss_fn(m: GChat) -> jax.Array:
        logits = m(inputs)
        return (
            optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            .astype(jnp.float32)
            .mean()
        )

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def run_training_loop(
    config: GChatConfig,
    batch_iter: Iterator[dict[str, Any]],
    model: GChat,
    optimizer: nnx.Optimizer,
) -> None:
    training = config.training
    eval_cfg = config.eval
    metrics_cfg = config.metrics

    eval_state = build_eval_state(config)
    training_loss_gauge, validation_loss_gauge, _ = initialize_metrics(
        run_id=metrics_cfg.run_id,
        config=metrics_cfg.exporter_config(),
    )
    print(f"Metrics initialized with run_id={metrics_cfg.run_id}")
    metrics_writer = MetricsWriter(
        log_every=training.log_every,
        training_loss_gauge=training_loss_gauge,
        validation_loss_gauge=validation_loss_gauge,
    )

    for step, batch in enumerate(batch_iter):
        tokens = jnp.asarray(batch["tokens"])
        loss = train_step(model, optimizer, tokens)
        metrics_writer.log_train(step, loss)

        if step > 0 and step % eval_cfg.every == 0:
            bpb = run_eval(model, eval_state)
            metrics_writer.log_eval(step, bpb)

    metrics_writer.flush()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = config.checkpoint_uri(timestamp)
    print(f"Saving final model checkpoint to {checkpoint_dir}")
    save_model(model, checkpoint_dir)
