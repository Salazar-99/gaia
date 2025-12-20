import torch
from torch import nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    # Approximation formula
    def forward(self, x: torch.Tensor):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.44715 * torch.pow(x, 3))
        ))

class SwiGLU(nn.Module):
    """
    SwiGLU activation function of the form:

        SwiGLU(x) = Swish(xW1) * (xW2)

    Where * is element-wise multiplication

    A more efficient implementation would be to use a single linear layer and chunk it into two
    after the forward pass on it but I think this is easier to understand since it more closely resembles the
    mathematical formulation.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        # Note the TRM paper uses no bias terms
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        x1 = self.W1(x)
        x2 = self.W1(x)
        return F.silu(x1) * x2

class FeedForward(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )
    
    def forward(self, x: torch.Tensor):
        return self.layers(x)
    
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * norm_x + self.shift