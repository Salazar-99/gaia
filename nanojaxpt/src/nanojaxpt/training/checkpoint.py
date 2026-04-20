from __future__ import annotations

from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx

from nanojaxpt.training.model import GPT, GPTConfig


def save_model(model: GPT, out_dir: Path) -> None:
    """Save the model's pure state to `out_dir/model` via Orbax."""
    out_dir = out_dir.resolve()
    _, model_state = nnx.split(model)
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(out_dir / "model", model_state)


def load_model(config: GPTConfig, out_dir: Path) -> GPT:
    """Restore a GPT model previously saved with `save_model`."""
    abstract_model = nnx.eval_shape(lambda: GPT(config, nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract_model)
    with ocp.StandardCheckpointer() as ckptr:
        restored_state = ckptr.restore(out_dir.resolve() / "model", abstract_state)
    return nnx.merge(graphdef, restored_state)
