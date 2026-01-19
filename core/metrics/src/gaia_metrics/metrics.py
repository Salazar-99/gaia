from typing import Optional, Tuple, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gaia_core.config import MetricsConfig

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
    MetricExporter,
)
from opentelemetry.sdk.resources import Resource

# Type aliases for OpenTelemetry metrics
# In OpenTelemetry Python SDK, Gauge and Meter are not directly importable
# They are returned from meter methods, so we use Any for runtime type hints
Meter = Any
Gauge = Any


class NoOpGauge:
    """A no-op gauge that does nothing. Used when metrics are disabled."""

    def set(self, value: float, attributes: dict | None = None) -> None:
        """No-op set method."""
        pass


class NoOpMeter:
    """A no-op meter that creates no-op gauges. Used when metrics are disabled."""

    def create_gauge(
        self, name: str, description: str = "", unit: str = "1"
    ) -> NoOpGauge:
        """Return a no-op gauge."""
        return NoOpGauge()


def create_gauge(meter: Meter, name: str, description: str, unit: str = "1") -> Gauge:
    """Create a gauge metric.

    Args:
        meter: OpenTelemetry meter instance
        name: Name of the gauge metric
        description: Description of what the metric measures
        unit: Unit of measurement (default: "1" for unitless)

    Returns:
        Gauge metric instance
    """
    return meter.create_gauge(
        name=name,
        description=description,
        unit=unit,
    )


def initialize_metrics(
    run_id: str,
    config: Optional["MetricsConfig"] = None,
) -> Tuple[Gauge, Gauge, Meter]:
    """Initialize the metrics system and create default gauges.

    Args:
        run_id: Unique identifier for this training run
        config: Optional MetricsConfig for the OTLP exporter

    Returns:
        Tuple of (training_loss gauge, validation_loss gauge, meter)
        The meter can be used to create additional gauges via create_gauge()
    """
    # Return no-op objects if metrics are disabled
    if config is not None and not config.enabled:
        noop_meter = NoOpMeter()
        return NoOpGauge(), NoOpGauge(), noop_meter

    # Create a resource for the meter provider
    # These attributes make it easy to filter for metrics downstream
    resource = Resource.create({"service.name": "gaia", "run.id": run_id})

    # Determine which exporter to use
    # Use console exporter if explicitly requested or if no endpoint is configured
    use_console = config is None or config.use_console or config.endpoint is None

    exporter: MetricExporter
    if use_console:
        exporter = ConsoleMetricExporter()
    else:
        # Build OTLP exporter kwargs from config, filtering out None values
        exporter_kwargs: dict = {}
        if config is not None:
            if config.endpoint is not None:
                exporter_kwargs["endpoint"] = config.endpoint
            if config.headers is not None:
                exporter_kwargs["headers"] = config.headers
            if config.timeout is not None:
                exporter_kwargs["timeout"] = config.timeout
        exporter = OTLPMetricExporter(**exporter_kwargs)

    metric_reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(metric_readers=[metric_reader], resource=resource)
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter("gaia.meter")

    training_loss = create_gauge(
        meter,
        name="training_loss",
        description="A gauge metric for training loss",
    )

    validation_loss = create_gauge(
        meter,
        name="validation_loss",
        description="A gauge metric for validation loss",
    )

    return training_loss, validation_loss, meter
