# gaia
gaia is my personal collection of AI model training code. The purpose of Gaia is to learn SOTA AI model architectures and training schemes by implementing them myself and running my own training experiments on GPUs.

## Current Capabilities
- Build and train GPT2 on a small corpus with: `python -m llms.gpt2.train_gpt2 <path-to-corpus-txt-file>`. Model and training parameters can be tweaked
by modifying the `GPT2Config` and `TrainingConfig` dataclasses respectively.

## Future goals
### Model Architectures
- Implement more modern LLM architecture patterns
    - Grouped Query Attention (GQA)
    - Multi-Head Latent Attention (MLA)
    - Mixture of Experts (MoE)
- Implement a LLama model
- Implement a Multi-Modal model
- Implement a Vision Language Action Model (VLA) as used in Robotics applications
- Implement a State Space Model
- Implement a Joint Embedding Predictive Architecture (JEPA) model
### Systems
- Add code for tracking training runs (WandB or TensorBoard)
- Add Distributed Data Parralel training
- Add Tensor Parallel training
- Implement GPU kernels in Triton and Cuda to accelerate training
- Implement KV Caching

## Learning References
- [Build a Large Language Model From Scratch Book](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [Stanford CS336 Lectures](https://youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_&si=PJJPnM6kjDunXt_R)
- [Sebastian Raschka's Blog](https://sebastianraschka.com/blog/)


[Name inspiration](https://horizon.fandom.com/wiki/GAIA_(original))