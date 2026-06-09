"""
core/adapters/telemetry_adapter.py

Adapter: ObservabilityCollector -> TelemetryPort
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from core.ports.telemetry_port import MetricEvent, TelemetryPort

logger = logging.getLogger(__name__)

try:
    from core.opentelemetry_metrics import OTelMetrics
    HAS_OTEL_METRICS = True
except ImportError:
    HAS_OTEL_METRICS = False


class TelemetryAdapter(TelemetryPort):
    """Wraps ObservabilityCollector and optionally OpenTelemetry metrics."""

    def __init__(self, observability=None, otel_metrics: Any = None):
        self._obs = observability
        self._buffer: list = []
        self._buffer_size = 100
        self._otel = otel_metrics
        if self._otel is None and HAS_OTEL_METRICS:
            try:
                self._otel = OTelMetrics(enabled=False)
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass

    def record_metric(self, event: MetricEvent) -> None:
        if self._obs is not None and hasattr(self._obs, "record_metric"):
            try:
                self._obs.record_metric(event.name, event.value, event.tags)
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass
        else:
            self._buffer.append(event)
            if len(self._buffer) > self._buffer_size:
                self._buffer.pop(0)

        # Mirror key metrics to OpenTelemetry
        if self._otel is not None and self._otel.enabled:
            try:
                if "vision_latency" in event.name:
                    self._otel.record_vision_latency(event.value)
                elif "decision" in event.name and "latency" in event.name:
                    self._otel.record_decision_latency(event.value)
                elif "cycle_latency" in event.name:
                    self._otel.record_cycle_latency(event.value)
                elif "error" in event.name:
                    self._otel.record_error("orchestrator", event.tags.get("type", "unknown") if event.tags else "unknown")
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass

    def record_event(self, event_name: str, details: Dict[str, Any]) -> None:
        if self._obs is not None and hasattr(self._obs, "record_event"):
            try:
                self._obs.record_event(event_name, details)
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass

    def flush(self) -> None:
        if self._obs is not None and hasattr(self._obs, "flush"):
            try:
                self._obs.flush()
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass

    def health_check(self) -> Dict[str, Any]:
        return {
            "observability_available": self._obs is not None,
            "otel_metrics_available": self._otel is not None and getattr(self._otel, "enabled", False),
            "buffered_metrics": len(self._buffer),
        }
