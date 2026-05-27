from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import yaml

from gchat.training.model import ModelConfig

GCS_SCHEME = "gs://"


@dataclass
class RuntimeConfig:
    omp_num_threads: int = 1
    base_dir: Path = Path("~/.cache/gchat")


@dataclass
class MetricsConfig:
    enabled: bool = True
    run_id: str = "gchat-test"
    endpoint: str | None = "https://otel.gerardosalazar.com/v1/metrics"
    username: str | None = "otel"
    password: str | None = None
    use_console: bool = False

    def exporter_config(self) -> SimpleNamespace:
        headers: dict[str, str] | None = None
        if self.username and self.password:
            token = base64.b64encode(
                f"{self.username}:{self.password}".encode()
            ).decode()
            headers = {"Authorization": f"Basic {token}"}

        use_console = self.use_console or (self.enabled and self.endpoint is None)
        return SimpleNamespace(
            enabled=self.enabled,
            endpoint=self.endpoint,
            headers=headers,
            timeout=None,
            use_console=use_console,
        )


@dataclass
class DataConfig:
    data_dir: str = "gs://gchat-climbmix-7b/data"
    token_bytes_path: str | None = None
    batch_size: int = 4
    sequence_length: int = 1024
    shuffle: bool = True
    gcs_token_shard_count: int = 54


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    seed: int = 0
    log_every: int = 1


@dataclass
class EvalConfig:
    every: int = 100
    batch_size: int = 32
    split_tokens: int = 1_310_720


@dataclass
class CheckpointConfig:
    gcs_bucket: str | None = None
    gcs_prefix: str | None = None


@dataclass
class ProfilingConfig:
    enabled: bool = False
    log_dir: Path | None = None
    warmup_steps: int = 5
    steps: int = 3


@dataclass
class GChatConfig:
    """Top-level training configuration loaded from a single YAML file."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GChatConfig:
        config_path = Path(path).expanduser().resolve()
        with config_path.open() as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"Config root must be a mapping, got {type(raw).__name__}."
            )
        return cls.from_dict(raw)

    def resolve(self) -> GChatConfig:
        """Fill the few paths derived from the speedrun layout."""
        base_dir = Path(self.runtime.base_dir).expanduser().resolve()
        self.runtime.base_dir = base_dir

        if self.data.token_bytes_path is None:
            data_dir = self.data.data_dir.rstrip("/")
            self.data.token_bytes_path = f"{data_dir}/token_bytes.npy"

        data_dir = self.data.data_dir.rstrip("/")
        if not data_dir.startswith(GCS_SCHEME):
            raise ValueError("gchat training expects data.data_dir to be a gs:// URI.")
        gcs_path = data_dir[len(GCS_SCHEME) :]
        bucket, _, prefix = gcs_path.partition("/")
        if self.checkpoint.gcs_bucket is None:
            self.checkpoint.gcs_bucket = bucket
        if self.checkpoint.gcs_prefix is None:
            self.checkpoint.gcs_prefix = (
                f"{prefix}/checkpoint" if prefix else "checkpoint"
            )

        if self.profiling.log_dir is None:
            self.profiling.log_dir = base_dir / "profiles"
        else:
            self.profiling.log_dir = Path(self.profiling.log_dir).expanduser().resolve()

        self.model.sequence_len = self.data.sequence_length
        if self.eval.every <= 0:
            raise ValueError("eval.every must be a positive integer.")
        return self

    def apply_runtime_env(self) -> None:
        os.environ["OMP_NUM_THREADS"] = str(self.runtime.omp_num_threads)

    def checkpoint_uri(self, timestamp: str) -> str:
        checkpoint = self.checkpoint
        if checkpoint.gcs_bucket is None:
            raise ValueError(
                "checkpoint.gcs_bucket must be resolved before checkpointing."
            )
        prefix = checkpoint.gcs_prefix.strip("/") if checkpoint.gcs_prefix else ""
        if prefix:
            return f"gs://{checkpoint.gcs_bucket}/{prefix}/{timestamp}"
        return f"gs://{checkpoint.gcs_bucket}/{timestamp}"
