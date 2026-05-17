"""
core/ports/telemetry_port.py

Telemetry Port — abstract interface for metrics, logging, and observability.

Adapters:
    - LocalTelemetryAdapter (core/observability.py)
    - OpenTelemetryAdapter (future: OpenTelemetry, Prometheus, Grafana)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MetricEvent:
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = 0.0


class TelemetryPort(abc.ABC):
    """Abstract telemetry/metrics interface."""

    @abc.abstractmethod
    def record_metric(self, event: MetricEvent) -> None:
        """Record a single metric point."""
        ...

    @abc.abstractmethod
    def record_event(self, event_name: str, details: Dict[str, Any]) -> None:
        """Record a structured event (match start, state change, error, etc.)."""
        ...

    @abc.abstractmethod
    def flush(self) -> None:
        """Flush any buffered metrics to backend."""
        ...

    @abc.abstractmethod
    def health_check(self) -> Dict[str, Any]:
        ...
