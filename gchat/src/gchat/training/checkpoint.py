from __future__ import annotations

from pathlib import Path

import orbax.checkpoint as ocp
from etils import epath
from flax import nnx

from gchat.training.model import GPT, GPTConfig


def _as_epath(out_dir: str | Path | epath.Path) -> epath.Path:
    """Coerce local paths or gs:// URIs to an epath.Path Orbax understands."""
    if isinstance(out_dir, epath.Path):
        return out_dir
    s = str(out_dir)
    if s.startswith("gs://"):
        return epath.Path(s.rstrip("/"))
    return epath.Path(Path(s).expanduser())


def save_model(model: GPT, out_dir: str | Path | epath.Path) -> None:
    """Save the model's pure state to `out_dir/model` via Orbax.

    `out_dir` may be a local filesystem path or a `gs://bucket/prefix` URI;
    Orbax handles both transparently through etils.epath / tensorstore.
    """
    root = _as_epath(out_dir)
    _, model_state = nnx.split(model)
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(root / "model", model_state)


def load_model(config: GPTConfig, out_dir: str | Path | epath.Path) -> GPT:
    """Restore a GPT model previously saved with `save_model`."""
    root = _as_epath(out_dir)
    abstract_model = nnx.eval_shape(lambda: GPT(config, nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract_model)
    with ocp.StandardCheckpointer() as ckptr:
        restored_state = ckptr.restore(root / "model", abstract_state)
    return nnx.merge(graphdef, restored_state)
