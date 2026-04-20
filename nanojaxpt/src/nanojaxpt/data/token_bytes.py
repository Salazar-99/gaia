"""
Build a `token_bytes` vector for bits-per-byte (BPB) evaluation.

This mirrors nanochat's `scripts/tok_train.py` (after tokenizer save): for each
token id, record the UTF-8 byte length of that token's decoded string, or 0 for
special tokens so they are excluded from the BPB denominator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def special_token_strings(tok: Tokenizer) -> set[str]:
    """Content strings for added/special tokens (same idea as nanochat HuggingFaceTokenizer.get_special_tokens)."""
    return {w.content for w in tok.get_added_tokens_decoder().values()}


def build_token_bytes_array(tok: Tokenizer) -> np.ndarray:
    """
    For each id in [0, vocab_size): byte length of decode([id]) in UTF-8, or 0 if
    the id is a special token (excluded from BPB), matching nanochat's tok_train
    intent.

    GPT-2-style models often decode a special id to the empty string while
    `id_to_token` still shows ``<|endoftext|>``; we treat an id as special if
    either the decoded piece or `id_to_token` is listed as a special token.
    """
    vocab_size = tok.get_vocab_size()
    specials = special_token_strings(tok)
    out = np.zeros(vocab_size, dtype=np.int32)
    for token_id in range(vocab_size):
        piece = tok.decode([token_id])
        symbol = tok.id_to_token(token_id)
        if piece in specials or symbol in specials:
            out[token_id] = 0
        else:
            out[token_id] = len(piece.encode("utf-8"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write token_bytes.npy: per-token UTF-8 byte lengths for BPB eval "
            "(nanochat-compatible semantics; vocab-only, no dataset read)."
        )
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help='Hugging Face tokenizer id or path to a tokenizer.json directory (default: "gpt2").',
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("token_bytes.npy"),
        help="Output path for an int32 .npy array of shape (vocab_size,) (default: token_bytes.npy).",
    )
    args = parser.parse_args()

    path = Path(args.tokenizer)
    if path.is_dir() and (path / "tokenizer.json").is_file():
        tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    elif path.is_file() and path.name == "tokenizer.json":
        tok = Tokenizer.from_file(str(path))
    else:
        tok = Tokenizer.from_pretrained(str(args.tokenizer))

    arr = build_token_bytes_array(tok)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, arr)
    nonzero = arr[arr > 0]
    print(f"Wrote {args.output} shape={arr.shape} dtype={arr.dtype}")
    print(
        f"nonzero byte lengths: min={int(nonzero.min())} max={int(nonzero.max())} "
        f"mean={float(nonzero.mean()):.4f}"
        if len(nonzero)
        else "all zeros (unexpected)"
    )


if __name__ == "__main__":
    main()
