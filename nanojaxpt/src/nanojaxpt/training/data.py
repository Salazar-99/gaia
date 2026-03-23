from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import struct
from typing import Any, Iterator, Mapping

import grain  # type: ignore[import-not-found]
import numpy as np

TOKEN_FILE_GLOB = "tokens-*.arrayrecord"
Batch = dict[str, np.ndarray]


@dataclass(frozen=True)
class DatasetConfig:
    """Config for reading tokenized ArrayRecord shards with Grain."""

    data_dir: Path | str = Path("climbmix_tokens")
    batch_size: int = 8
    sequence_length: int = 1024
    shuffle: bool = True
    seed: int | None = 0
    repeat: bool = True
    drop_remainder: bool = True
    reader_options: Mapping[str, str] = field(default_factory=dict)

    def normalized_data_dir(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()


def list_arrayrecord_files(data_dir: Path | str) -> list[Path]:
    """Returns sorted ArrayRecord token files in a directory."""
    root = Path(data_dir).expanduser().resolve()
    files = sorted(root.glob(TOKEN_FILE_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No files matching {TOKEN_FILE_GLOB!r} found in {root}."
        )
    return files


def decode_token_record(record: bytes) -> np.ndarray:
    """
    Decodes one ArrayRecord payload from nanojaxpt.data.download.

    Record format:
    - uint32 little-endian token count
    - int32 token IDs (count entries)
    """
    if len(record) < 4:
        raise ValueError("Invalid token record: payload is too short.")

    (num_tokens,) = struct.unpack_from("<I", record, 0)
    payload = memoryview(record)[4:]
    expected_payload_bytes = num_tokens * np.dtype(np.int32).itemsize
    if len(payload) != expected_payload_bytes:
        raise ValueError(
            "Invalid token record size: expected "
            f"{expected_payload_bytes} bytes, got {len(payload)}."
        )

    return np.frombuffer(payload, dtype=np.int32, count=num_tokens)


def build_grain_token_dataset(config: DatasetConfig) -> Any:
    """
    Builds a Grain MapDataset that yields token arrays per ArrayRecord entry.
    """
    files = list_arrayrecord_files(config.normalized_data_dir())

    source = grain.sources.ArrayRecordDataSource(
        paths=[str(path) for path in files],
        reader_options=dict(config.reader_options) or None,
    )
    ds = grain.MapDataset.source(source).map(decode_token_record)
    if config.shuffle:
        ds = ds.shuffle() if config.seed is None else ds.shuffle(seed=config.seed)
    if config.repeat:
        ds = ds.repeat()
    return ds


def _iter_token_windows(
    token_records: Iterator[np.ndarray],
    sequence_length: int,
) -> Iterator[np.ndarray]:
    """Converts variable-length token records into fixed-size contiguous windows."""
    window_size = sequence_length + 1
    carry = np.empty((0,), dtype=np.int32)

    for record in token_records:
        if record.size == 0:
            continue

        record_i32 = record.astype(np.int32, copy=False)
        if carry.size == 0:
            stream = record_i32
        else:
            stream = np.concatenate((carry, record_i32))

        while stream.size >= window_size:
            yield stream[:window_size]
            # Advance by sequence_length so adjacent windows are contiguous.
            stream = stream[sequence_length:]
        carry = stream


def _batch_windows(
    windows: Iterator[np.ndarray],
    batch_size: int,
    drop_remainder: bool,
) -> Iterator[Batch]:
    """Batches token windows into `{inputs, targets}` arrays."""
    first_window = next(windows, None)
    if first_window is None:
        return

    seq_len = first_window.size - 1
    inputs = np.empty((batch_size, seq_len), dtype=np.int32)
    targets = np.empty((batch_size, seq_len), dtype=np.int32)
    batch_pos = 0

    def add_window(window: np.ndarray) -> None:
        nonlocal batch_pos
        inputs[batch_pos] = window[:-1]
        targets[batch_pos] = window[1:]
        batch_pos += 1

    add_window(first_window)
    if batch_pos == batch_size:
        yield {"inputs": inputs, "targets": targets}
        inputs = np.empty((batch_size, seq_len), dtype=np.int32)
        targets = np.empty((batch_size, seq_len), dtype=np.int32)
        batch_pos = 0

    for window in windows:
        add_window(window)
        if batch_pos == batch_size:
            yield {"inputs": inputs, "targets": targets}
            inputs = np.empty((batch_size, seq_len), dtype=np.int32)
            targets = np.empty((batch_size, seq_len), dtype=np.int32)
            batch_pos = 0

    if batch_pos and not drop_remainder:
        yield {
            "inputs": inputs[:batch_pos].copy(),
            "targets": targets[:batch_pos].copy(),
        }


def _validate_training_config(config: DatasetConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")


def build_training_dataset(
    config: DatasetConfig,
) -> Iterator[Batch]:
    """
    Builds a training-ready batch iterator from ArrayRecord token shards.

    Output batch structure:
      - `inputs`:  int32 [batch_size, sequence_length]
      - `targets`: int32 [batch_size, sequence_length]
    """
    _validate_training_config(config)

    grain_token_dataset = build_grain_token_dataset(config)
    token_records = iter(grain_token_dataset.to_iter_dataset())
    windows = _iter_token_windows(
        token_records=token_records,
        sequence_length=config.sequence_length,
    )
    return _batch_windows(
        windows=windows,
        batch_size=config.batch_size,
        drop_remainder=config.drop_remainder,
    )


__all__ = [
    "DatasetConfig",
    "build_training_dataset",
]
