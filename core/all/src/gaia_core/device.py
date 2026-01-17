import torch


def get_device(device_cfg: str | None = None) -> str:
    """Determine device from config or auto-detect.

    Args:
        device_cfg: Optional device name ("cuda", "mps", "cpu"). If None, auto-detects.

    Returns:
        Device name as string ("cuda", "mps", or "cpu").

    Raises:
        RuntimeError: If requested device is not available.
    """
    if device_cfg:
        device = device_cfg.lower()
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available.")
        return device

    # Auto-detect
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_device_info(device: str) -> None:
    """Print device information to console.

    Args:
        device: Device name ("cuda", "mps", or "cpu")
    """
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print("  Using Metal Performance Shaders (MPS)")
