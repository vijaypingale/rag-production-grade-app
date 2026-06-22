"""
OpenTelemetry tracing setup — Section 12

What this provides:
-------------------
A single configured OpenTelemetry tracer the rest of the app uses to create
spans (the "journey" of a request through each pipeline stage).

Vendor-neutral by design:
-------------------------
The instrumentation (start_span, set attributes) NEVER mentions Datadog or
any vendor. Only the EXPORTER here decides where spans go:
  - OTEL_EXPORTER=console -> printed to stdout (local dev, no account/cost)
  - OTEL_EXPORTER=otlp    -> sent to an OTLP endpoint (Datadog Agent, Grafana,
                             SigNoz, etc.) — wired up in a later step
Switching backends = changing one env var, not the code. That's the whole
point of OpenTelemetry.

Safe-by-default:
----------------
- If OTEL_ENABLED is false, get_tracer() returns a no-op tracer (zero overhead)
  so unit tests and offline scripts are unaffected.
- Setup is idempotent — calling it more than once won't double-register.
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

from app.config.settings import (
    OTEL_ENABLED,
    OTEL_EXPORTER,
    OTEL_SERVICE_NAME,
)
from app.utils.logger import logger


_INITIALIZED = False


def _build_exporter():
    """
    Pick the span exporter based on OTEL_EXPORTER.

    console -> ConsoleSpanExporter (prints spans as JSON to stdout)
    otlp    -> OTLP exporter (added in the Datadog step; falls back to
               console with a warning until then so nothing breaks)
    """
    if OTEL_EXPORTER == "console":
        return ConsoleSpanExporter()

    if OTEL_EXPORTER == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            return OTLPSpanExporter()  # endpoint configured via env vars
        except ImportError:
            logger.warning(
                "otel_otlp_exporter_unavailable",
                detail="opentelemetry-exporter-otlp not installed; using console",
            )
            return ConsoleSpanExporter()

    logger.warning("otel_unknown_exporter", exporter=OTEL_EXPORTER, fallback="console")
    return ConsoleSpanExporter()


def setup_tracing() -> None:
    """
    Initialize the global OpenTelemetry TracerProvider once.

    Idempotent and gated by OTEL_ENABLED. Called lazily by get_tracer().
    """
    global _INITIALIZED

    if _INITIALIZED or not OTEL_ENABLED:
        return

    resource = Resource.create({"service.name": OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(_build_exporter()))
    trace.set_tracer_provider(provider)

    _INITIALIZED = True
    logger.info(
        "otel_tracing_initialized",
        exporter=OTEL_EXPORTER,
        service_name=OTEL_SERVICE_NAME,
    )


def get_tracer(name: str = "rag.pipeline"):
    """
    Return a tracer for creating spans.

    If OTEL_ENABLED is false, this returns OpenTelemetry's built-in no-op
    tracer (spans become cheap no-ops), so callers can always wrap code in
    `with tracer.start_as_current_span(...)` without checking flags.
    """
    if OTEL_ENABLED:
        setup_tracing()
    return trace.get_tracer(name)
