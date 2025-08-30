import torch


def get_device():
    # Check if MPS is available and use it, otherwise fallback to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) accelerator")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    return device
