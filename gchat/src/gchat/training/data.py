from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import fnmatch
import shutil
import struct
import subprocess
from typing import Any, Iterator, Mapping

import grain  # type: ignore[import-not-found]
import jax
import numpy as np

TOKEN_FILE_GLOB = "tokens-*.arrayrecord"
TOKEN_FILE_TEMPLATE = "tokens-{index:05d}.arrayrecord"
GCS_SCHEME = "gs://"
Batch = dict[str, np.ndarray]


@dataclass(frozen=True)
class DatasetConfig:
    """Config for reading tokenized ArrayRecord shards with Grain.

    `data_dir` accepts either a local path or a `gs://bucket/prefix` URI; in
    the latter case shards are streamed directly from GCS via ArrayRecord's
    GCS backend (no local copy). See `gchat/data/upload_dataset.sh` for
    the bucket layout this expects.
    """

    data_dir: str = "climbmix_tokens"
    token_file_glob: str = TOKEN_FILE_GLOB
    batch_size: int = 8
    sequence_length: int = 1024
    shuffle: bool = True
    seed: int | None = 0
    repeat: bool = True
    drop_remainder: bool = True
    # Grain IterDataset read tuning. For streaming from GCS, bumping
    # `num_threads` is what turns a latency-bound pipeline into a
    # bandwidth-bound one; 16-64 is a reasonable range per host.
    num_threads: int = 16
    prefetch_buffer_size: int = 128
    # Per-host sharding. None means "use jax.process_{index,count}()" so the
    # same config works on a single host (no-op) and on a TPU pod.
    process_index: int | None = None
    process_count: int | None = None
    reader_options: Mapping[str, str] = field(default_factory=dict)
    # For GCS, avoid `gcloud storage ls` by constructing the expected shard
    # names directly. This is useful on TPU VMs where gcloud credentials may
    # not be configured even though ArrayRecord can read the objects.
    expected_gcs_token_file_count: int | None = None
    expected_gcs_file_names: tuple[str, ...] | None = None

    def normalized_data_dir(self) -> str:
        """Normalize `data_dir` while preserving a `gs://` scheme."""
        s = str(self.data_dir)
        if s.startswith(GCS_SCHEME):
            return s.rstrip("/")
        return str(Path(s).expanduser().resolve())


def _list_local_arrayrecord_files(root: Path, pattern: str) -> list[str]:
    files = sorted(str(p) for p in root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {root}.")
    return files


def _list_gcs_arrayrecord_files(uri: str, pattern: str) -> list[str]:
    if shutil.which("gcloud") is None:
        raise RuntimeError(
            "gcloud is required to list gs:// paths. Install the Google "
            "Cloud SDK (or run on a GCP VM where it is preinstalled)."
        )
    listing = subprocess.run(
        ["gcloud", "storage", "ls", f"{uri.rstrip('/')}/"],
        check=False,
        capture_output=True,
        text=True,
    )
    if listing.returncode != 0:
        detail = (listing.stderr or listing.stdout).strip()
        message = f"Failed to list {uri} with gcloud storage ls"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message)
    candidates = [ln.strip() for ln in listing.stdout.splitlines()
                  if ln.strip().startswith(GCS_SCHEME) and not ln.strip().endswith("/")]
    files = sorted(u for u in candidates if fnmatch.fnmatch(u.rsplit("/", 1)[-1], pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} under {uri}.")
    return files


def _join_gcs_file_names(uri: str, file_names: tuple[str, ...]) -> list[str]:
    if not file_names:
        raise ValueError("expected_gcs_file_names must not be empty")
    base = uri.rstrip("/")
    return [f"{base}/{name}" for name in file_names]


def _numbered_gcs_token_files(uri: str, count: int) -> list[str]:
    if count <= 0:
        raise ValueError(f"expected_gcs_token_file_count must be > 0, got {count}")
    base = uri.rstrip("/")
    return [
        f"{base}/{TOKEN_FILE_TEMPLATE.format(index=index)}"
        for index in range(count)
    ]


def list_arrayrecord_files(
    data_dir: str | Path,
    token_file_glob: str = TOKEN_FILE_GLOB,
    expected_gcs_token_file_count: int | None = None,
    expected_gcs_file_names: tuple[str, ...] | None = None,
) -> list[str]:
    """Return a sorted list of ArrayRecord shard URIs (local paths or gs://)."""
    s = str(data_dir)
    if s.startswith(GCS_SCHEME):
        if expected_gcs_file_names is not None:
            return _join_gcs_file_names(s, expected_gcs_file_names)
        if expected_gcs_token_file_count is not None:
            if token_file_glob != TOKEN_FILE_GLOB:
                raise ValueError(
                    "expected_gcs_token_file_count only supports "
                    f"{TOKEN_FILE_GLOB!r}; got {token_file_glob!r}"
                )
            return _numbered_gcs_token_files(s, expected_gcs_token_file_count)
        return _list_gcs_arrayrecord_files(s, token_file_glob)
    root = Path(s).expanduser().resolve()
    return _list_local_arrayrecord_files(root, token_file_glob)


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


def _resolve_process_sharding(config: DatasetConfig) -> tuple[int, int]:
    """Resolve (process_index, process_count), falling back to JAX defaults."""
    pc = config.process_count if config.process_count is not None else jax.process_count()
    pi = config.process_index if config.process_index is not None else jax.process_index()
    if pc <= 0:
        raise ValueError(f"process_count must be > 0, got {pc}")
    if not 0 <= pi < pc:
        raise ValueError(f"process_index {pi} out of range for process_count {pc}")
    return pi, pc


def build_grain_token_dataset(config: DatasetConfig) -> Any:
    """
    Builds a Grain MapDataset that yields token arrays per ArrayRecord entry.

    Each host sees a disjoint stride of the record index space
    (`ds[process_index::process_count]`), so N TPU hosts stream N/1 of the
    data in parallel with no cross-host coordination.
    """
    files = list_arrayrecord_files(
        config.normalized_data_dir(),
        config.token_file_glob,
        expected_gcs_token_file_count=config.expected_gcs_token_file_count,
        expected_gcs_file_names=config.expected_gcs_file_names,
    )

    source = grain.sources.ArrayRecordDataSource(
        paths=files,
        reader_options=dict(config.reader_options) or None,
    )
    ds = grain.MapDataset.source(source).map(decode_token_record)

    process_index, process_count = _resolve_process_sharding(config)
    if process_count > 1:
        ds = ds.slice(slice(process_index, None, process_count))

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
    if config.num_threads <= 0:
        raise ValueError("num_threads must be > 0")
    if config.prefetch_buffer_size <= 0:
        raise ValueError("prefetch_buffer_size must be > 0")


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
    read_options = grain.ReadOptions(
        num_threads=config.num_threads,
        prefetch_buffer_size=config.prefetch_buffer_size,
    )
    token_records = iter(grain_token_dataset.to_iter_dataset(read_options))
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
    "list_arrayrecord_files",
]
