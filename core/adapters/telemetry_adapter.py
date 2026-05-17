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


class TelemetryAdapter(TelemetryPort):
    """Wraps ObservabilityCollector to satisfy TelemetryPort."""

    def __init__(self, observability=None):
        self._obs = observability
        self._buffer: list = []
        self._buffer_size = 100

    def record_metric(self, event: MetricEvent) -> None:
        if self._obs is not None and hasattr(self._obs, "record_metric"):
            try:
                self._obs.record_metric(event.name, event.value, event.tags)
            except Exception:
                pass
        else:
            self._buffer.append(event)
            if len(self._buffer) > self._buffer_size:
                self._buffer.pop(0)

    def record_event(self, event_name: str, details: Dict[str, Any]) -> None:
        if self._obs is not None and hasattr(self._obs, "record_event"):
            try:
                self._obs.record_event(event_name, details)
            except Exception:
                pass

    def flush(self) -> None:
        if self._obs is not None and hasattr(self._obs, "flush"):
            try:
                self._obs.flush()
            except Exception:
                pass

    def health_check(self) -> Dict[str, Any]:
        return {
            "observability_available": self._obs is not None,
            "buffered_metrics": len(self._buffer),
        }
