from __future__ import annotations

import argparse
from pathlib import Path

from nanojaxpt.training.data import DatasetConfig, build_training_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training dataset and print first token batch."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("climbmix_tokens"),
        help="Directory containing tokens-*.arrayrecord files.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling for deterministic inspection.",
    )
    parser.add_argument(
        "--no-repeat",
        action="store_true",
        help="Disable repeat mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config = DatasetConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        shuffle=not args.no_shuffle,
        repeat=not args.no_repeat,
    )
    dataset = build_training_dataset(config)
    first_batch = next(dataset)

    print("First batch:")
    print("inputs:")
    print(first_batch["inputs"])
    print("targets:")
    print(first_batch["targets"])


if __name__ == "__main__":
    main()
