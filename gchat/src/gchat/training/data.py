from __future__ import annotations

import struct
from collections.abc import Iterator

import grain  # type: ignore[import-not-found]
import jax
import numpy as np

from gchat.training.config import DataConfig

TOKEN_FILE_TEMPLATE = "tokens-{index:05d}.arrayrecord"
TEST_TOKEN_FILE = "test-tokens.arrayrecord"
SHUFFLE_SEED = 0
NUM_THREADS = 16
PREFETCH_BUFFER_SIZE = 128
Batch = dict[str, np.ndarray]


def training_token_files(data_dir: str, count: int) -> list[str]:
    if count <= 0:
        raise ValueError(f"gcs_token_shard_count must be > 0, got {count}")
    base = data_dir.rstrip("/")
    return [
        f"{base}/{TOKEN_FILE_TEMPLATE.format(index=index)}"
        for index in range(count)
    ]


def test_token_file(data_dir: str) -> str:
    return f"{data_dir.rstrip('/')}/{TEST_TOKEN_FILE}"


def decode_token_record(record: bytes) -> np.ndarray:
    """
    Decodes one ArrayRecord payload from gchat.data.download.

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


def _build_grain_token_dataset(files: list[str], shuffle: bool):
    source = grain.sources.ArrayRecordDataSource(
        paths=files,
    )
    ds = grain.MapDataset.source(source).map(decode_token_record)

    if jax.process_count() > 1:
        ds = ds.slice(slice(jax.process_index(), None, jax.process_count()))

    if shuffle:
        ds = ds.shuffle(seed=SHUFFLE_SEED)
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
) -> Iterator[Batch]:
    batch: list[np.ndarray] = []
    for window in windows:
        batch.append(window)
        if len(batch) == batch_size:
            tokens = np.stack(batch)
            # Keep [batch, seq + 1] intact so training does one host-to-device
            # transfer, then splits inputs/targets inside the jitted step.
            yield {"tokens": tokens}
            batch = []


def _validate_batch_config(batch_size: int, sequence_length: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")


def build_token_batches(
    files: list[str],
    batch_size: int,
    sequence_length: int,
    shuffle: bool,
) -> Iterator[Batch]:
    _validate_batch_config(batch_size, sequence_length)
    grain_token_dataset = _build_grain_token_dataset(files, shuffle=shuffle)
    read_options = grain.ReadOptions(
        num_threads=NUM_THREADS,
        prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    )
    token_records = iter(grain_token_dataset.to_iter_dataset(read_options))
    windows = _iter_token_windows(
        token_records=token_records,
        sequence_length=sequence_length,
    )
    return _batch_windows(
        windows=windows,
        batch_size=batch_size,
    )


def build_training_dataset(config: DataConfig) -> Iterator[Batch]:
    files = training_token_files(config.data_dir, config.gcs_token_shard_count)
    return build_token_batches(
        files=files,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        shuffle=config.shuffle,
    )
