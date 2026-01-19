"""TrainingContext: Unified context manager for training services.

Provides a single interface to initialize and manage Dashboard, Metrics,
and CheckpointSaver services used during training.
"""

from typing import Optional, Dict

from .config import MetricsConfig, CheckpointConfig
from gaia_dashboard import Dashboard
from gaia_metrics import initialize_metrics, create_gauge, Gauge
from gaia_checkpoints import CheckpointSaver


class TrainingContext:
    """Unified context for training services.

    Initializes and manages Dashboard, Metrics (with gauges), and CheckpointSaver.
    Provides convenient access to all training infrastructure in one place.

    Example:
        ctx = TrainingContext(
            MetricsConfig(**cfg.metrics),
            CheckpointConfig(**cfg.checkpoint),
        )
        # Use ctx.dashboard, ctx.gauges["training_loss"], ctx.checkpoint_saver
        ctx.close()
    """

    def __init__(
        self,
        metrics_config: MetricsConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """Initialize training context with metrics and checkpoint configuration.

        Args:
            metrics_config: Configuration for metrics export
            checkpoint_config: Configuration for checkpoint saving
        """
        # Initialize dashboard
        self.dashboard = Dashboard()

        # Initialize metrics
        run_id = metrics_config.get_run_id()
        training_loss_gauge, validation_loss_gauge, meter = initialize_metrics(
            run_id=run_id, config=metrics_config
        )

        # Create standard gauges dictionary for easy access
        self.gauges: Dict[str, Gauge] = {
            "training_loss": training_loss_gauge,
            "validation_loss": validation_loss_gauge,
        }

        # Create additional common gauges
        self.gauges["lr"] = create_gauge(
            meter, "learning_rate", "Current learning rate"
        )
        self.gauges["accuracy"] = create_gauge(meter, "accuracy", "Evaluation accuracy")
        self.gauges["exact_accuracy"] = create_gauge(
            meter, "exact_accuracy", "Exact sequence accuracy"
        )

        self.meter = meter
        self.run_id = run_id

        # Initialize checkpoint saver if path is configured
        self.checkpoint_saver: Optional[CheckpointSaver] = None
        if checkpoint_config.path:
            self.checkpoint_saver = CheckpointSaver(
                prefix=checkpoint_config.path,
                storage=checkpoint_config.storage,
                base_path=".",
                max_queue_size=checkpoint_config.max_queue_size,
            )

        self.checkpoint_config = checkpoint_config

    def close(self):
        """Close all resources and cleanup."""
        self.dashboard.close()
