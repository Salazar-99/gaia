"""
Download ClimbMix train shards (nanochat-style indexing), tokenize with GPT-2
(Hugging Face `tokenizers`), and write records as ArrayRecord.

Shard layout matches nanochat: train files shard_00000.parquet … shard_06541.parquet
(6542 files), validation is shard_06542.parquet (not used here).
"""

from __future__ import annotations

import argparse
import array
from collections import deque
import multiprocessing as mp
import os
import queue
import re
import struct
import sys
import threading
import time
from pathlib import Path

import pyarrow.parquet as pq
import rich
from rich.console import Console, Group
from rich.table import Table

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# -----------------------------------------------------------------------------
# Same dataset / naming as nanochat/nanochat/dataset.py
CLIMBMIX_REPO_ID = "karpathy/climbmix-400b-shuffle"
# Index of the validation shard file (train indices are 0 .. MAX_SHARD_INDEX - 1)
MAX_SHARD_INDEX = 6542
NUM_TRAIN_SHARDS = MAX_SHARD_INDEX  # 6542 files: indices 0 .. 6541
DEFAULT_ARRAYRECORD_FILE_SIZE_MB = 500
DEFAULT_ARRAYRECORD_GROUP_SIZE = 1
DEFAULT_TOKENIZER_BATCH_SIZE = 128
DEFAULT_RECORD_QUEUE_BATCH_MB = 1
DEFAULT_RECORD_QUEUE_MAXSIZE = 32
DASHBOARD_REFRESH_SECONDS = 1.0
RATE_WINDOW_SECONDS = 60.0
RECENT_OUTPUT_PATHS = 4


def _shard_filename(index: int) -> str:
    return f"shard_{index:05d}.parquet"


def _parse_tokens_b(s: str) -> int:
    """Parse values like 9B → 9_000_000_000. Only integer + B suffix."""
    m = re.fullmatch(r"(\d+)B", s.strip().upper())
    if not m:
        raise argparse.ArgumentTypeError(
            "Expected --tokens in the form <integer>B (e.g. 9B, 1B), case-insensitive."
        )
    n = int(m.group(1))
    return n * 1_000_000_000


def _pack_token_record(ids: list[int]) -> bytes:
    """Length-prefixed int32 token payload for one ArrayRecord entry."""
    n = len(ids)
    buf = array.array("i", ids)
    return struct.pack("<I", n) + buf.tobytes()


def _format_int(n: int) -> str:
    return f"{n:,}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_rate(tokens_per_second: float) -> str:
    if tokens_per_second <= 0:
        return "--"
    return f"{tokens_per_second:,.0f} tok/s"


def _format_gb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 ** 3):.2f} GB"


class WriterStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.file_count = 0
        self.current_path: str | None = None
        self.current_file_bytes = 0
        self.total_bytes = 0
        self.completed_paths: deque[str] = deque(maxlen=RECENT_OUTPUT_PATHS)

    def open_file(self, path: Path) -> None:
        with self._lock:
            self.file_count += 1
            self.current_path = str(path)
            self.current_file_bytes = 0

    def note_write(self, payload_bytes: int) -> None:
        with self._lock:
            self.current_file_bytes += payload_bytes
            self.total_bytes += payload_bytes

    def close_current_file(self) -> None:
        with self._lock:
            if self.current_path is not None:
                self.completed_paths.append(self.current_path)
                self.current_path = None
                self.current_file_bytes = 0

    def snapshot(self) -> dict[str, int | str | list[str] | None]:
        with self._lock:
            return {
                "file_count": self.file_count,
                "current_path": self.current_path,
                "current_file_bytes": self.current_file_bytes,
                "total_bytes": self.total_bytes,
                "completed_paths": list(self.completed_paths),
            }


class DownloadDashboard:
    def __init__(self, token_target: int, output_dir: Path) -> None:
        self.console = Console()
        self.start_time = time.time()
        self.token_target = token_target
        self.output_dir = output_dir
        self.history: deque[tuple[float, int]] = deque()

    def render(self, token_count: int, parquet_files_read: int, writer_snapshot: dict) -> Group:
        now = time.time()
        self.history.append((now, token_count))
        while self.history and now - self.history[0][0] > RATE_WINDOW_SECONDS:
            self.history.popleft()

        tokens_per_second = 0.0
        if len(self.history) >= 2:
            start_time, start_tokens = self.history[0]
            delta_t = now - start_time
            delta_tokens = token_count - start_tokens
            if delta_t > 0 and delta_tokens > 0:
                tokens_per_second = delta_tokens / delta_t

        remaining_tokens = max(0, self.token_target - token_count)
        eta_seconds = None if tokens_per_second <= 0 else remaining_tokens / tokens_per_second
        progress_pct = 100.0 * token_count / self.token_target if self.token_target else 0.0
        uptime = now - self.start_time

        title = Table(
            box=None,
            expand=True,
            show_header=False,
        )
        title.add_column(justify="left")
        title.add_row("[bright_white]nanojaxpt Data Downloader[/bright_white]")

        summary = Table(
            box=rich.box.ROUNDED,
            expand=True,
            show_header=False,
            border_style="bright_white",
        )
        summary.add_column(style="bright_white", width=24)
        summary.add_column(style="default", ratio=1)
        summary.add_column(style="bright_white", width=24)
        summary.add_column(style="default", ratio=1)

        summary.add_row("Output directory", str(self.output_dir.resolve()), "Parquet files read", _format_int(parquet_files_read))
        summary.add_row(
            "Current tokens",
            f"{_format_int(token_count)} / {_format_int(self.token_target)} ({progress_pct:.2f}%)",
            "Token rate",
            _format_rate(tokens_per_second),
        )
        summary.add_row(
            "ETA",
            _format_duration(eta_seconds),
            "Uptime",
            _format_duration(uptime),
        )
        summary.add_row(
            "ArrayRecord files",
            _format_int(int(writer_snapshot["file_count"])),
            "GB written",
            _format_gb(int(writer_snapshot["total_bytes"])),
        )
        current_path = writer_snapshot["current_path"] or "--"
        current_file_bytes = int(writer_snapshot["current_file_bytes"])
        summary.add_row(
            "Current file",
            current_path,
            "Current file size",
            _format_gb(current_file_bytes),
        )
        return Group(title, summary)

    def print_dashboard(self, token_count: int, parquet_files_read: int, writer_snapshot: dict) -> None:
        renderable = self.render(
            token_count=token_count,
            parquet_files_read=parquet_files_read,
            writer_snapshot=writer_snapshot,
        )
        with self.console.capture() as capture:
            self.console.print(renderable)
        print("\033[2J\033[H" + capture.get(), end="")


def _flush_record_batch(
    record_queue: mp.Queue,
    pending_payloads: list[bytes],
) -> list[bytes]:
    if pending_payloads:
        record_queue.put(pending_payloads)
    return []


def _producer(
    worker_id: int,
    shard_queue: mp.Queue,
    stop_event: mp.Event,
    shard_counter: mp.Value,
    cache_dir: Path,
    revision: str | None,
) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError
    from huggingface_hub.utils import HfHubHTTPError

    cache_dir.mkdir(parents=True, exist_ok=True)
    while not stop_event.is_set():
        with shard_counter.get_lock():
            seq = shard_counter.value
            shard_counter.value = seq + 1
        idx = seq % NUM_TRAIN_SHARDS
        fname = _shard_filename(idx)
        if stop_event.is_set():
            return
        try:
            path = hf_hub_download(
                repo_id=CLIMBMIX_REPO_ID,
                filename=fname,
                repo_type="dataset",
                local_dir=str(cache_dir),
                revision=revision,
            )
            shard_queue.put(path)
        except (RepositoryNotFoundError, HfHubHTTPError, OSError) as e:
            print(f"[producer {worker_id}] failed {fname}: {e}", file=sys.stderr)
            time.sleep(2.0)
            continue


def _consumer(
    worker_id: int,
    shard_queue: mp.Queue,
    record_queue: mp.Queue,
    stop_event: mp.Event,
    token_count: mp.Value,
    token_target: int,
    token_lock: mp.Lock,
    tokenizer_batch_size: int,
    parquet_files_read: mp.Value,
    record_queue_batch_bytes: int,
) -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained("gpt2")

    while True:
        try:
            path = shard_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if path is None:
            break

        with parquet_files_read.get_lock():
            parquet_files_read.value += 1

        _process_parquet_path(
            path,
            tok,
            record_queue,
            stop_event,
            token_count,
            token_target,
            token_lock,
            worker_id,
            max(1, tokenizer_batch_size),
            record_queue_batch_bytes,
        )


def _process_parquet_path(
    path: str,
    tok,
    record_queue: mp.Queue,
    stop_event: mp.Event,
    token_count: mp.Value,
    token_target: int,
    token_lock: mp.Lock,
    worker_id: int,
    tokenizer_batch_size: int,
    record_queue_batch_bytes: int,
) -> bool:
    try:
        pf = pq.ParquetFile(path)
    except OSError as e:
        print(f"[consumer {worker_id}] cannot open {path}: {e}", file=sys.stderr)
        return False

    pending_payloads: list[bytes] = []
    pending_payload_bytes = 0

    def flush_pending_payloads() -> None:
        nonlocal pending_payloads, pending_payload_bytes
        pending_payloads = _flush_record_batch(record_queue, pending_payloads)
        pending_payload_bytes = 0

    def claim_token_budget(id_batches: list[list[int]]) -> tuple[list[int], bool]:
        """Reserve token budget for this batch and return truncation sizes."""
        lengths = [len(ids) for ids in id_batches]
        take_sizes: list[int] = []
        target_reached = False

        with token_lock:
            remaining = token_target - token_count.value
            if remaining <= 0:
                stop_event.set()
                return take_sizes, True

            taken = 0
            for length in lengths:
                if taken >= remaining:
                    break
                take = min(length, remaining - taken)
                if take <= 0:
                    break
                take_sizes.append(take)
                taken += take

            token_count.value += taken
            if token_count.value >= token_target:
                stop_event.set()
                target_reached = True

        return take_sizes, target_reached

    for rg_idx in range(pf.num_row_groups):
        if stop_event.is_set():
            with token_lock:
                if token_count.value >= token_target:
                    flush_pending_payloads()
                    return False
        rg = pf.read_row_group(rg_idx)
        texts = rg.column("text").to_pylist()
        for start in range(0, len(texts), tokenizer_batch_size):
            batch_texts = [
                text
                for text in texts[start : start + tokenizer_batch_size]
                if isinstance(text, str) and text
            ]
            if not batch_texts:
                continue

            id_batches = [encoding.ids for encoding in tok.encode_batch(batch_texts)]
            id_batches = [ids for ids in id_batches if ids]
            if not id_batches:
                continue

            take_sizes, target_reached = claim_token_budget(id_batches)

            if not take_sizes:
                flush_pending_payloads()
                return False

            for ids, take in zip(id_batches, take_sizes):
                payload = _pack_token_record(ids[:take])
                pending_payloads.append(payload)
                pending_payload_bytes += len(payload)
                if pending_payload_bytes >= record_queue_batch_bytes:
                    flush_pending_payloads()

            if target_reached or len(take_sizes) < len(id_batches):
                flush_pending_payloads()
                return False

    flush_pending_payloads()
    return True


def _writer_thread(
    record_queue: mp.Queue,
    output_dir: Path,
    file_size_bytes: int,
    done_event: threading.Event,
    writer_stats: WriterStats,
) -> None:
    import array_record.python.array_record_module as ar_module

    writer = None
    file_index = 0
    current_file_bytes = 0

    def open_writer(index: int):
        path = output_dir / f"tokens-{index:05d}.arrayrecord"
        writer_stats.open_file(path)
        return ar_module.ArrayRecordWriter(
            str(path), f"group_size:{DEFAULT_ARRAYRECORD_GROUP_SIZE}"
        )

    try:
        while True:
            payloads = record_queue.get()
            if payloads is None:
                break
            for payload in payloads:
                if writer is None:
                    writer = open_writer(file_index)
                elif current_file_bytes > 0 and (
                    current_file_bytes + len(payload) > file_size_bytes
                ):
                    writer.close()
                    writer_stats.close_current_file()
                    file_index += 1
                    writer = open_writer(file_index)
                    current_file_bytes = 0
                writer.write(payload)
                current_file_bytes += len(payload)
                writer_stats.note_write(len(payload))
    finally:
        if writer is not None:
            writer.close()
            writer_stats.close_current_file()
        done_event.set()


def _join_with_dashboard(
    proc: mp.Process,
    timeout_seconds: float,
    refresh_dashboard,
) -> None:
    deadline = time.time() + timeout_seconds
    while proc.is_alive() and time.time() < deadline:
        proc.join(timeout=DASHBOARD_REFRESH_SECONDS)
        refresh_dashboard()
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        refresh_dashboard()


def run(
    token_target: int,
    output_dir: Path,
    num_producers: int,
    num_consumers: int,
    revision: str | None,
    shard_cache: Path,
    tokenizer_batch_size: int,
    arrayrecord_file_size_mb: int,
    record_queue_batch_mb: int,
    record_queue_maxsize: int,
) -> None:
    if token_target <= 0:
        print("Nothing to do (token target is 0).", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_cache.mkdir(parents=True, exist_ok=True)
    arrayrecord_file_size_bytes = max(1, arrayrecord_file_size_mb) * 1024 * 1024

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    shard_counter = ctx.Value("Q", 0)
    token_count = ctx.Value("Q", 0)
    parquet_files_read = ctx.Value("Q", 0)
    token_lock = ctx.Lock()
    # Bounded backpressure: producers block when consumers fall behind
    shard_queue: mp.Queue = ctx.Queue(maxsize=max(4, num_producers * 2))
    # Bounded backpressure: cap writer backlog to avoid large RAM/SWAP spikes.
    record_queue: mp.Queue = ctx.Queue(maxsize=max(1, record_queue_maxsize))
    record_queue_batch_bytes = max(1, record_queue_batch_mb) * 1024 * 1024

    done_writing = threading.Event()
    writer_stats = WriterStats()
    dashboard = DownloadDashboard(token_target=token_target, output_dir=output_dir)

    writer = threading.Thread(
        target=_writer_thread,
        args=(
            record_queue,
            output_dir,
            arrayrecord_file_size_bytes,
            done_writing,
            writer_stats,
        ),
        daemon=False,
    )

    producers = [
        ctx.Process(
            target=_producer,
            args=(i, shard_queue, stop_event, shard_counter, shard_cache, revision),
        )
        for i in range(num_producers)
    ]
    consumers = [
        ctx.Process(
            target=_consumer,
            args=(
                i,
                shard_queue,
                record_queue,
                stop_event,
                token_count,
                token_target,
                token_lock,
                tokenizer_batch_size,
                parquet_files_read,
                record_queue_batch_bytes,
            ),
        )
        for i in range(num_consumers)
    ]

    def refresh_dashboard() -> None:
        dashboard.print_dashboard(
            token_count=token_count.value,
            parquet_files_read=parquet_files_read.value,
            writer_snapshot=writer_stats.snapshot(),
        )
    writer.start()
    for p in producers:
        p.start()
    for c in consumers:
        c.start()

    try:
        refresh_dashboard()
        while token_count.value < token_target and not stop_event.is_set():
            time.sleep(DASHBOARD_REFRESH_SECONDS)
            refresh_dashboard()
        stop_event.set()
        refresh_dashboard()

        for p in producers:
            _join_with_dashboard(p, timeout_seconds=120, refresh_dashboard=refresh_dashboard)
        for _ in range(num_consumers):
            shard_queue.put(None)
        for c in consumers:
            _join_with_dashboard(c, timeout_seconds=600, refresh_dashboard=refresh_dashboard)
    finally:
        record_queue.put(None)
        while not done_writing.wait(timeout=DASHBOARD_REFRESH_SECONDS):
            refresh_dashboard()
        while writer.is_alive():
            writer.join(timeout=DASHBOARD_REFRESH_SECONDS)
            refresh_dashboard()
        refresh_dashboard()

    print(
        "Wrote ArrayRecord shards to "
        f"{output_dir.resolve()} (tokens-00000.arrayrecord, tokens-00001.arrayrecord, ...)"
    )
    print(f"Total tokens (approx): {token_count.value:,} / target {token_target:,}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download ClimbMix shards (nanochat-style indices), tokenize with GPT-2, "
            "and write ArrayRecord. Requires network and `huggingface-cli login` if needed."
        )
    )
    p.add_argument(
        "--tokens",
        type=_parse_tokens_b,
        required=True,
        help='Target number of GPT-2 tokens to emit (e.g. 9B = 9 billion). Format: <integer>B only.',
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("climbmix_tokens"),
        help=(
            "Output directory (default: ./climbmix_tokens). "
            "Creates tokens-00000.arrayrecord, tokens-00001.arrayrecord, ... here."
        ),
    )
    p.add_argument(
        "--num-producers",
        type=int,
        default=4,
        help="Parallel shard download workers (default: 4).",
    )
    p.add_argument(
        "--num-consumers",
        type=int,
        default=4,
        help="Parallel tokenizer workers (default: 4).",
    )
    p.add_argument(
        "--shard-cache",
        type=Path,
        default=None,
        help="Directory for downloaded Parquet shards (default: <output>/shard_cache).",
    )
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Hub revision for ClimbMix (branch/tag/commit). Default: Hub default branch.",
    )
    p.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=DEFAULT_TOKENIZER_BATCH_SIZE,
        help=f"Documents per tokenizer batch (default: {DEFAULT_TOKENIZER_BATCH_SIZE}).",
    )
    p.add_argument(
        "--arrayrecord-file-size-mb",
        type=int,
        default=DEFAULT_ARRAYRECORD_FILE_SIZE_MB,
        help=(
            "Approximate target size for each ArrayRecord output file in MB "
            f"(default: {DEFAULT_ARRAYRECORD_FILE_SIZE_MB})."
        ),
    )
    p.add_argument(
        "--record-queue-batch-mb",
        type=int,
        default=DEFAULT_RECORD_QUEUE_BATCH_MB,
        help=(
            "Approximate payload size per queue item written by tokenizer workers "
            f"(default: {DEFAULT_RECORD_QUEUE_BATCH_MB} MB)."
        ),
    )
    p.add_argument(
        "--record-queue-maxsize",
        type=int,
        default=DEFAULT_RECORD_QUEUE_MAXSIZE,
        help=(
            "Max in-flight payload batches waiting for writer thread "
            f"(default: {DEFAULT_RECORD_QUEUE_MAXSIZE})."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    shard_cache = args.shard_cache if args.shard_cache is not None else args.output / "shard_cache"
    run(
        token_target=args.tokens,
        output_dir=args.output,
        num_producers=max(1, args.num_producers),
        num_consumers=max(1, args.num_consumers),
        revision=args.revision,
        shard_cache=shard_cache,
        tokenizer_batch_size=max(1, args.tokenizer_batch_size),
        arrayrecord_file_size_mb=max(1, args.arrayrecord_file_size_mb),
        record_queue_batch_mb=max(1, args.record_queue_batch_mb),
        record_queue_maxsize=max(1, args.record_queue_maxsize),
    )


if __name__ == "__main__":
    main()
