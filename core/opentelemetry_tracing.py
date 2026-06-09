"""
core/opentelemetry_tracing.py

OpenTelemetry tracing integration for the Brawl Stars bot.

Wraps the existing custom distributed_tracing.py with proper OTel spans,
providing export to OTLP, Jaeger, Zipkin, or console.

Design:
    - Lazy initialization (no cost when disabled)
    - Falls back to no-op if opentelemetry SDK is unavailable
    - Integrates with orchestrator tick loop for per-tick spans
    - Context propagation between vision → decision → action

Usage:
    from core.opentelemetry_tracing import get_tracer, span_context
    with get_tracer().start_as_current_span("vision.capture") as span:
        span.set_attribute("resolution", "1920x1080")
        screenshot = capture()
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# Lazy imports so the system works even without opentelemetry installed
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    trace = None  # type: ignore[assignment]


class NoOpSpan:
    """Fallback span when OpenTelemetry is unavailable."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key: str, value: Any):
        pass

    def set_status(self, status: Any):
        pass

    def record_exception(self, exception: Exception):
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        pass


class NoOpTracer:
    """Fallback tracer when OpenTelemetry is unavailable."""

    def start_as_current_span(self, name: str, **kwargs):
        return NoOpSpan(name)

    def start_span(self, name: str, **kwargs):
        return NoOpSpan(name)


# ------------------------------------------------------------------
# Global tracer state (lazy)
# ------------------------------------------------------------------
_tracer_instance: Optional[Any] = None
_provider: Optional[Any] = None


def _get_tracer() -> Any:
    """Return the global tracer (lazy init)."""
    global _tracer_instance
    if _tracer_instance is None:
        if HAS_OTEL and trace is not None:
            _tracer_instance = trace.get_tracer("soberana-omega-bot", "1.0.0")
        else:
            _tracer_instance = NoOpTracer()
    return _tracer_instance


def initialize_tracing(
    service_name: str = "soberana-omega-bot",
    endpoint: Optional[str] = None,
    exporter_type: str = "otlp",
    enabled: bool = True,
) -> bool:
    """
    Initialize OpenTelemetry tracing.

    Args:
        service_name: Name shown in trace backends
        endpoint: OTLP/gRPC/HTTP endpoint URL (e.g., http://localhost:4317)
        exporter_type: 'otlp' | 'jaeger' | 'zipkin' | 'console'
        enabled: If False, use no-op tracer

    Returns:
        True if tracing was successfully initialized
    """
    global _provider, _tracer_instance

    if not enabled or not HAS_OTEL:
        logger.info("[OTEL] Tracing disabled or SDK unavailable")
        return False

    try:
        _provider = TracerProvider()
        trace.set_tracer_provider(_provider)

        if exporter_type == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporter = ConsoleSpanExporter()
        elif exporter_type == "jaeger":
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
        elif exporter_type == "zipkin":
            from opentelemetry.exporter.zipkin.json import ZipkinExporter
            exporter = ZipkinExporter(endpoint=endpoint or "http://localhost:9411/api/v2/spans")
        else:  # otlp (default)
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=endpoint or "http://localhost:4317")

        processor = BatchSpanProcessor(exporter)
        _provider.add_span_processor(processor)
        _tracer_instance = trace.get_tracer(service_name, "1.0.0")
        logger.info("[OTEL] Tracing initialized: %s -> %s", service_name, exporter_type)
        return True
    except (ImportError, ModuleNotFoundError, TypeError) as e:
        logger.error("[OTEL] Tracing init failed: %s", e)
        _tracer_instance = NoOpTracer()
        return False


def shutdown_tracing() -> None:
    """Flush and shutdown the tracer provider."""
    global _provider
    if _provider is not None:
        try:
            _provider.shutdown()
            logger.info("[OTEL] Tracing shutdown complete")
        except (RuntimeError, AttributeError, OSError) as e:
            logger.warning("[OTEL] Tracing shutdown error: %s", e)
        _provider = None


@contextmanager
def span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    parent: Optional[Any] = None,
) -> Generator[Any, None, None]:
    """
    Context manager for creating a span.

    Args:
        name: Span name (e.g., 'vision.capture', 'decision.choose_action')
        attributes: Dict of span attributes
        parent: Optional parent span context
    """
    tracer = _get_tracer()
    kwargs: Dict[str, Any] = {}
    if parent is not None:
        kwargs["context"] = parent

    with tracer.start_as_current_span(name, **kwargs) as sp:
        if attributes:
            for k, v in attributes.items():
                sp.set_attribute(k, v)
        yield sp


def record_error(span_obj: Any, exception: Exception, message: Optional[str] = None) -> None:
    """Record an exception on a span."""
    if span_obj is None or isinstance(span_obj, NoOpSpan):
        return
    try:
        span_obj.record_exception(exception)
        if message:
            span_obj.set_status(Status(StatusCode.ERROR, message))
    except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
        pass


def get_trace_id() -> Optional[str]:
    """Return current trace ID for correlation with logs."""
    if not HAS_OTEL:
        return None
    try:
        ctx = trace.get_current_span().get_span_context()
        return format(ctx.trace_id, "032x") if ctx.trace_id else None
    except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
        return None
