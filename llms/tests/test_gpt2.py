from llms import GPT2, Tokenizer
import torch

GPT2_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }


def test_GPT2_forward():
    # Check if MPS is available and use it, otherwise fallback to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) accelerator")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    
    gpt2 = GPT2(
        GPT2_CONFIG_124M["vocab_size"],
        GPT2_CONFIG_124M["emb_dim"],
        GPT2_CONFIG_124M["context_length"],
        GPT2_CONFIG_124M["drop_rate"],
        GPT2_CONFIG_124M["n_layers"],
        GPT2_CONFIG_124M["n_heads"]
    )
    
    # Move model to device (MPS or CPU)
    gpt2 = gpt2.to(device)
    
    tokenizer = Tokenizer("gpt2")
    # Disable dropout
    gpt2.eval() 
    test_input = "Testing GPT2 forward pass works"
    test_tokens = tokenizer.encode(test_input)
    
    # Move input tensor to the same device as the model
    test_tensor = torch.tensor(test_tokens, device=device).unsqueeze(0)
    
    # Don't compute gradients
    with torch.no_grad():
        logits = gpt2(test_tensor)
    
    assert logits is not None

