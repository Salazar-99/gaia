import base64
import os
from types import SimpleNamespace
from typing import Optional, Tuple, TYPE_CHECKING, Any, Mapping

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

OTEL_COLLECTOR_USERNAME_ENV = "OTEL_COLLECTOR_USERNAME"
OTEL_COLLECTOR_PASSWORD_ENV = "OTEL_COLLECTOR_PASSWORD"
OTEL_EXPORTER_OTLP_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
OTEL_EXPORTER_OTLP_METRICS_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"


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


class ForceFlushingGauge:
    """Gauge wrapper that exports each set value immediately."""

    def __init__(self, gauge: Gauge, provider: MeterProvider):
        self._gauge = gauge
        self._provider = provider

    def set(self, value: float, attributes: dict | None = None) -> None:
        """Set the gauge value and force an immediate metric export."""
        self._gauge.set(value, attributes)
        self._provider.force_flush()


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


def create_force_flushing_gauge(
    meter: Meter,
    provider: MeterProvider,
    name: str,
    description: str,
    unit: str = "1",
) -> ForceFlushingGauge:
    """Create a gauge whose .set() exports immediately.

    OpenTelemetry gauges keep only the latest value between periodic export
    intervals. This wrapper preserves the familiar .set() call site while
    forcing a collection after each value is recorded.
    """
    gauge = create_gauge(meter, name=name, description=description, unit=unit)
    return ForceFlushingGauge(gauge, provider)


def force_flush_on_set(gauge: Gauge, provider: MeterProvider) -> ForceFlushingGauge:
    """Wrap an existing gauge so every .set() call is exported immediately."""
    return ForceFlushingGauge(gauge, provider)


def _headers_with_env_auth(headers: Optional[Mapping[str, str]]) -> dict[str, str] | None:
    """Add OTEL collector Basic Auth from env unless the caller already supplied auth."""
    merged_headers = dict(headers or {})
    has_auth_header = any(key.lower() == "authorization" for key in merged_headers)

    if not has_auth_header:
        username = os.getenv(OTEL_COLLECTOR_USERNAME_ENV)
        password = os.getenv(OTEL_COLLECTOR_PASSWORD_ENV)

        if username and password:
            token = base64.b64encode(f"{username}:{password}".encode()).decode()
            merged_headers["Authorization"] = f"Basic {token}"

    return merged_headers or None


def _metrics_config_from_env() -> SimpleNamespace:
    endpoint = os.getenv(OTEL_EXPORTER_OTLP_METRICS_ENDPOINT_ENV)
    if endpoint is None:
        otlp_endpoint = os.getenv(OTEL_EXPORTER_OTLP_ENDPOINT_ENV)
        if otlp_endpoint is not None:
            endpoint = otlp_endpoint.rstrip("/") + "/v1/metrics"

    return SimpleNamespace(
        enabled=True,
        endpoint=endpoint,
        headers=None,
        timeout=None,
        use_console=endpoint is None,
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
            headers = _headers_with_env_auth(config.headers)
            if headers is not None:
                exporter_kwargs["headers"] = headers
            if config.timeout is not None:
                exporter_kwargs["timeout"] = config.timeout
        exporter = OTLPMetricExporter(**exporter_kwargs)

    metric_reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(metric_readers=[metric_reader], resource=resource)
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter("gaia.meter")

    training_loss = create_force_flushing_gauge(
        meter,
        provider,
        name="training_loss",
        description="A gauge metric for training loss",
    )

    validation_loss = create_force_flushing_gauge(
        meter,
        provider,
        name="validation_loss",
        description="A gauge metric for validation loss",
    )

    return training_loss, validation_loss, meter


def initialize_metrics_from_env(run_id: str) -> Tuple[Gauge, Gauge, Meter]:
    """Initialize default gauges using OTEL endpoint and auth environment variables."""
    return initialize_metrics(run_id=run_id, config=_metrics_config_from_env())
