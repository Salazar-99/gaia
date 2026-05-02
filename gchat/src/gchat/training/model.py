from __future__ import annotations

from dataclasses import dataclass

from flax import nnx
import jax
import jax.numpy as jnp


# Copied from nanochat
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 50257
    n_layer: int = 24
    n_head: int = 12  # number of query heads
    n_kv_head: int = 12  # number of key/value heads (GQA)
    n_embd: int = 1536
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


def compute_window_sizes(config: GPTConfig) -> list[tuple[int, int]]:
    pattern = config.window_pattern.upper()
    if not pattern or any(char not in "SL" for char in pattern):
        raise ValueError(
            f"Invalid window_pattern: {pattern!r}. Use a non-empty string with only S and L."
        )

    long_window = config.sequence_len
    short_window = long_window // 2
    char_to_window = {
        "L": (long_window, 0),
        "S": (short_window, 0),
    }

    window_sizes = []
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])

    window_sizes[-1] = (long_window, 0)
    return window_sizes


class MLP(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        super().__init__()
        self.w1 = nnx.Linear(
            in_features=config.n_embd,
            out_features=4 * config.n_embd,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.w2 = nnx.Linear(
            in_features=4 * config.n_embd,
            out_features=config.n_embd,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array):
        x = self.w1(x)
        x = jnp.square(nnx.relu(x))
        x = self.w2(x)
        return x


class RoPE(nnx.Module):
    """RoPE with MaxText-style frequency schedule.

    If `embedding_dims` is smaller than the input head dimension, only that leading
    slice is rotated and the remaining features pass through unchanged.
    """

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10_000,
    ):
        super().__init__()
        if embedding_dims % 2:
            raise ValueError("embedding_dims for RoPE must be a multiple of 2.")
        self.embedding_dims = embedding_dims
        half_dim = embedding_dims // 2
        fraction = 2 * jnp.arange(0, half_dim) / embedding_dims
        self._timescale = min_timescale * (max_timescale / min_timescale) ** fraction

    def __call__(
        self, inputs: jax.Array, position: jax.Array | None = None
    ) -> jax.Array:
        """Apply RoPE to Q or K. Shape [B, S, N, H]. If `position` is None, uses 0, 1, …, S-1 per row."""
        if inputs.ndim != 4:
            raise ValueError("inputs must be rank 4 [batch, sequence, heads, dims].")
        if inputs.shape[-1] < self.embedding_dims:
            raise ValueError("inputs last dim must be at least embedding_dims.")
        b, s, _, _ = inputs.shape
        dt = inputs.dtype
        if position is None:
            pos = jnp.arange(s, dtype=dt)[None, :, None]
        else:
            if position.shape != (b, s):
                raise ValueError(
                    f"position must be [B, S] = ({b}, {s}), got {position.shape}."
                )
            pos = position.astype(dt)[:, :, None]
        ts = self._timescale.astype(dt)
        sinusoid_inp = pos / ts[None, None, :]
        sin_half = jnp.sin(sinusoid_inp).astype(dt)
        cos_half = jnp.cos(sinusoid_inp).astype(dt)
        sin = jnp.concatenate([sin_half, sin_half], axis=-1)
        cos = jnp.concatenate([cos_half, cos_half], axis=-1)
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
        rotary_inputs = inputs[..., : self.embedding_dims]
        passthrough_inputs = inputs[..., self.embedding_dims :]
        x1, x2 = jnp.split(rotary_inputs, 2, axis=-1)
        rotated = jnp.concatenate((-x2, x1), axis=-1)
        rotary_outputs = rotary_inputs * cos + rotated * sin
        return jnp.concatenate((rotary_outputs, passthrough_inputs), axis=-1)


class Attention(nnx.Module):
    def __init__(self, config: GPTConfig, layer_idx: int, rngs: nnx.Rngs):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_head
        self.n_kv_heads = config.n_kv_head

        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if config.n_head % config.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head for GQA.")

        self.head_dim = config.n_embd // config.n_head
        self.output_dim = self.head_dim * self.n_heads
        self.attention_output_dim = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim

        # Match the PyTorch reference by rotating only half of each head.
        self.rope = RoPE(embedding_dims=self.head_dim // 2)
        self.qk_norm = nnx.RMSNorm(
            self.head_dim,
            use_scale=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.qkv = nnx.Linear(
            in_features=config.n_embd,
            out_features=self.attention_output_dim,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.output_projection = nnx.Linear(
            in_features=self.output_dim,
            out_features=config.n_embd,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        window_size: tuple[int, int] | None = None,
    ) -> jax.Array:
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            self.n_heads + 2 * self.n_kv_heads,
            self.head_dim,
        )

        q = qkv[:, :, : self.n_heads]
        k = qkv[:, :, self.n_heads : self.n_heads + self.n_kv_heads]
        v = qkv[:, :, self.n_heads + self.n_kv_heads :]

        q = self.rope(q)
        k = self.rope(k)
        q = self.qk_norm(q)
        k = self.qk_norm(k)

        attention_outputs = jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            local_window_size=window_size,
        )
        attention_outputs = attention_outputs.reshape(
            batch_size,
            sequence_length,
            self.output_dim,
        )
        return self.output_projection(attention_outputs)


class TransformerBlock(nnx.Module):
    def __init__(self, config: GPTConfig, layer_idx: int, rngs: nnx.Rngs):
        super().__init__()
        self.norm = nnx.RMSNorm(
            config.n_embd,
            use_scale=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.attn = Attention(config=config, layer_idx=layer_idx, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        window_size: tuple[int, int] | None = None,
    ) -> jax.Array:
        x = x + self.attn(self.norm(x), window_size=window_size)
        x = x + self.mlp(self.norm(x))
        return x


class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.window_sizes = compute_window_sizes(config)
        self.embedding = nnx.Embed(
            config.vocab_size,
            config.n_embd,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.blocks = nnx.List(
            [TransformerBlock(config, i, rngs) for i in range(config.n_layer)]
        )
        self.lm_head = nnx.Linear(
            config.n_embd,
            config.vocab_size,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.prenorm = nnx.RMSNorm(
            config.n_embd,
            use_scale=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.postnorm = nnx.RMSNorm(
            config.n_embd,
            use_scale=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.embedding(x)
        x = self.prenorm(x)
        for block in self.blocks:
            x = block(x)
        x = self.postnorm(x)
        logits = self.lm_head(x)
        return logits
