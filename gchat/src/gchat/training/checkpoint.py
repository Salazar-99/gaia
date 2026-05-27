from __future__ import annotations

import jax
import orbax.checkpoint as ocp
from etils import epath
from flax import nnx

from gchat.training.model import GChat


def save_model(model: GChat, checkpoint_uri: str) -> None:
    """Save the model state to the fixed `<gs checkpoint uri>/model` path."""
    if not checkpoint_uri.startswith("gs://"):
        raise ValueError(f"checkpoint_uri must be a gs:// URI, got {checkpoint_uri!r}")
    root = epath.Path(checkpoint_uri.rstrip("/")) / "model"
    jax.block_until_ready(jax.tree.leaves(nnx.state(model)))
    _, model_state = nnx.split(model)
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(root, model_state)
