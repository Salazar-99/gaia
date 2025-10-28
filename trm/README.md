# Tiny Recursive Models
My personal implementation of TRM from: https://arxiv.org/pdf/2510.04871v1

Link to original repo: https://github.com/SamsungSAILMontreal/TinyRecursiveModels/tree/main#

## Model
In `trm.py` I build up a TRM as a `nn.Module` starting from the individual layers up to the end-to-end model. It's not a fully general implementation. My goal was
to reproduce the model with the best hyperparameters described in the paper as simply as possible. I opt for off-the-shelf PyTorch modules wherever possible. The logic is as follows:
- Start with implementing `SwiGLU` since it's not in PyTorch yet (see open PR [here](https://github.com/pytorch/pytorch/pull/144465))
- Implement the two-layer MLP with SwiGLU activation I call `TRMMLP`
- Implementing `Attention` the same way the author did
- Stitching together `TRMLMLP`, `Attention`, and `RMSNorm` to form the "net" which I call `TRMNet`
- Finally, using the `TRMNet` to create `TRM` by applying the "recursive reasoning" process in an outer loop

## Data
I copy and pasted the code from the original repo and slightly modified it to work as a single file. I then generated the dataset by running the same command as the original repo:
```bash
uv run sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000
```

## Training

## Using this Code