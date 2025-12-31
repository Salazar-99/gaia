# gaia-core: Umbrella package that re-exports all core sub-packages

from gaia_layers import *
from gaia_metrics import *
from gaia_dashboard import *
from gaia_checkpoints import *

from .device import get_device
from .train import train, calc_loss_batch, calc_loss_loader, evaluate_model, plot_losses
from .generate import generate

__all__ = [
    # Re-exported from sub-packages
    "MultiHeadAttention",
    "GQA",
    "GQAKV",
    "GELU",
    "SwiGLU",
    "FeedForward",
    "LayerNorm",
    "Transformer",
    # Local modules
    "get_device",
    "train",
    "calc_loss_batch",
    "calc_loss_loader",
    "evaluate_model",
    "plot_losses",
    "generate",
]
