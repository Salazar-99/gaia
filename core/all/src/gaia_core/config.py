"""Pydantic schemas for Gaia core configuration.

These schemas define validated configuration for core services like
metrics export, checkpointing, and run metadata. Used with Hydra for
config management in individual experiment projects.
"""

from pydantic import BaseModel, Field
from typing import Optional, Mapping
import uuid


class MetricsConfig(BaseModel):
    """Configuration for OpenTelemetry metrics export."""

    enabled: bool = True
    """Enable metrics tracking. Set to False to disable all metric collection."""

    endpoint: Optional[str] = None
    """OTLP endpoint for metrics export (e.g., http://localhost:4318/v1/metrics)"""

    run_id: Optional[str] = None
    """Unique run identifier. Auto-generated if not provided."""

    headers: Optional[Mapping[str, str]] = None
    """Optional headers for OTLP exporter."""

    timeout: Optional[float] = None
    """Optional timeout for OTLP exporter."""

    use_console: bool = False
    """Use console/stdout exporter instead of OTLP. Auto-enabled if no endpoint."""

    def get_run_id(self) -> str:
        """Get run_id, generating one if not set."""
        return self.run_id or str(uuid.uuid4())[:8]


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint saving."""

    path: Optional[str] = None
    """Path/prefix for checkpoint files."""

    storage: str = "local"
    """Storage backend: 'local' or 'azure'."""

    interval: int = 1
    """Save checkpoint every N epochs."""

    max_queue_size: int = 3
    """Maximum checkpoints to queue for async saving."""


class CoreConfig(BaseModel):
    """Umbrella config for all Gaia core services.

    Example usage with Hydra:

        @hydra.main(version_base=None, config_path="../conf", config_name="config")
        def main(cfg: DictConfig):
            core = CoreConfig(
                metrics=cfg.metrics,
                checkpoint=cfg.checkpoint,
            )
            # Use core.metrics, core.checkpoint...
    """

    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
