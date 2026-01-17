from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from gaia_core.config import MetricsConfig

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.metrics import Gauge, Meter


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
    # Create a resource for the meter provider
    # These attributes make it easy to filter for metrics downstream
    resource = Resource.create({"service.name": "gaia", "run.id": run_id})

    # Build exporter kwargs from config, filtering out None values
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
