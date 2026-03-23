# gaia
gaia is a monorepo of my AI research code. It is structured to enable simple reuse of code/context across different projects and experiments.

## Project Structure

```text
gaia/
├── core/                # Reusable packages
│   ├── all/             # Catch-all package for everything in core
│   ├── checkpoints/     # Utilities for saving and writing model checkpoints
│   ├── dashboard/       # Reusable Rich dashboard for model training
│   ├── layers/          # Reusable neural network layers (Attention, GELU, ...)
│   └── metrics/         # Utilities for creating and writing OTel metrics
├── gpt2/                # GPT2 implementation in PyTorch
├── nanojaxpt/           # nanochat implementation in Jax trained on TPUs
├── trm/                 # Simplified Tiny Recursive Model implementation
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
└── uv.lock
```

[Name inspiration](https://horizon.fandom.com/wiki/GAIA_(original))