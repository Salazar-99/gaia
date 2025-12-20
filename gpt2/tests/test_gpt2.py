from gpt2 import GPT2, GPT2Config, Tokenizer
from core import get_device
import torch


# Predefined configurations
GPT2_CONFIG_124M = GPT2Config()


def create_model():
    # Use a small model for testing
    gpt2 = GPT2(GPT2_CONFIG_124M)
    device = get_device()
    # Move model to device (MPS or CPU)
    gpt2 = gpt2.to(device)
    return gpt2


def test_GPT2_forward():
    gpt2 = create_model()
    tokenizer = Tokenizer("gpt2")
    # Disable dropout
    gpt2.eval()
    test_input = "Testing GPT2 forward pass works"
    test_tokens = tokenizer.encode(test_input, allowed_special={"<|endoftext|>"})

    # Move input tensor to the same device as the model
    device = get_device()
    test_tensor = torch.tensor(test_tokens, device=device).unsqueeze(0)

    # Don't compute gradients
    with torch.no_grad():
        logits = gpt2(test_tensor)

    assert logits is not None


def test_GPT2_pretrain():
    gpt2 = create_model()
    # Test that model was created successfully
    assert gpt2 is not None
    assert hasattr(gpt2, "config")
    assert isinstance(gpt2.config, GPT2Config)
