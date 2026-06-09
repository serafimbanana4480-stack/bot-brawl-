"""
core/opentelemetry_metrics.py

OpenTelemetry metrics integration for the Brawl Stars bot.

Instruments:
    - Counter:   bot.matches_played, bot.actions_executed
    - Histogram: bot.cycle_latency_ms, bot.decision_latency_ms
    - UpDownCounter: bot.active_matches
    - ObservableGauge: bot.apm, bot.win_rate

All instruments are lazily created and no-op when opentelemetry is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

    HAS_OTEL_METRICS = True
except ImportError:
    HAS_OTEL_METRICS = False
    metrics = None  # type: ignore[assignment]


class NoOpCounter:
    def add(self, amount: float, attributes: Optional[Dict[str, Any]] = None):
        pass


class NoOpHistogram:
    def record(self, amount: float, attributes: Optional[Dict[str, Any]] = None):
        pass


class NoOpObservableGauge:
    pass


class OTelMetrics:
    """
    High-level metrics facade that wraps OpenTelemetry instruments.
    """

    _instance: Optional["OTelMetrics"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        enabled: bool = True,
        endpoint: Optional[str] = None,
        export_interval_ms: float = 60000.0,
    ):
        if self._initialized:
            return
        self._initialized = True

        self.enabled = enabled and HAS_OTEL_METRICS
        self._instruments: Dict[str, Any] = {}
        self._callbacks: Dict[str, Callable[[], float]] = {}

        if self.enabled:
            self._init_provider(endpoint, export_interval_ms)
        else:
            logger.info("[OTEL_METRICS] Metrics disabled or SDK unavailable")

    def _init_provider(self, endpoint: Optional[str], interval_ms: float) -> None:
        try:
            exporter = OTLPMetricExporter(endpoint=endpoint or "http://localhost:4317")
            reader = PeriodicExportingMetricReader(exporter, export_interval_millis=int(interval_ms))
            provider = MeterProvider(metric_readers=[reader])
            metrics.set_meter_provider(provider)
            self._meter = metrics.get_meter("soberana-omega-bot", "1.0.0")
            logger.info("[OTEL_METRICS] Provider initialized")
        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            logger.error("[OTEL_METRICS] Init failed: %s", e)
            self.enabled = False

    # ------------------------------------------------------------------
    # Instrument getters (lazy creation)
    # ------------------------------------------------------------------

    def _counter(self, name: str, description: str, unit: str = "1") -> Any:
        if not self.enabled:
            return NoOpCounter()
        key = f"counter:{name}"
        if key not in self._instruments:
            self._instruments[key] = self._meter.create_counter(name, unit=unit, description=description)
        return self._instruments[key]

    def _histogram(self, name: str, description: str, unit: str = "1") -> Any:
        if not self.enabled:
            return NoOpHistogram()
        key = f"hist:{name}"
        if key not in self._instruments:
            self._instruments[key] = self._meter.create_histogram(name, unit=unit, description=description)
        return self._instruments[key]

    def _gauge(self, name: str, description: str, callback: Callable[[], float], unit: str = "1") -> Any:
        if not self.enabled:
            return NoOpObservableGauge()
        key = f"gauge:{name}"
        if key not in self._instruments:
            self._instruments[key] = self._meter.create_observable_gauge(
                name, unit=unit, description=description,
                callbacks=[lambda options: callback()]
            )
        return self._instruments[key]

    # ------------------------------------------------------------------
    # Business metric helpers
    # ------------------------------------------------------------------

    def record_match_start(self, map_name: Optional[str] = None, brawler: Optional[str] = None) -> None:
        attrs: Dict[str, Any] = {}
        if map_name:
            attrs["map"] = map_name
        if brawler:
            attrs["brawler"] = brawler
        self._counter("bot.matches.started", "Number of matches started").add(1, attrs)

    def record_match_end(self, result: str, map_name: Optional[str] = None) -> None:
        attrs = {"result": result}
        if map_name:
            attrs["map"] = map_name
        self._counter("bot.matches.ended", "Number of matches ended").add(1, attrs)

    def record_action(self, action_type: str) -> None:
        self._counter("bot.actions", "Total actions executed").add(1, {"type": action_type})

    def record_cycle_latency(self, latency_ms: float) -> None:
        self._histogram("bot.cycle_latency", "Main loop cycle latency", unit="ms").record(latency_ms)

    def record_decision_latency(self, latency_ms: float) -> None:
        self._histogram("bot.decision_latency", "Decision engine latency", unit="ms").record(latency_ms)

    def record_vision_latency(self, latency_ms: float) -> None:
        self._histogram("bot.vision_latency", "Vision pipeline latency", unit="ms").record(latency_ms)

    def register_apm_callback(self, getter: Callable[[], float]) -> None:
        """Register a callback that returns current APM."""
        self._gauge("bot.apm", "Actions per minute", getter, unit="1")

    def register_win_rate_callback(self, getter: Callable[[], float]) -> None:
        """Register a callback that returns current win rate (0-1)."""
        self._gauge("bot.win_rate", "Win rate percentage", getter, unit="1")

    def record_error(self, component: str, error_type: str) -> None:
        self._counter("bot.errors", "Total errors by component").add(1, {
            "component": component,
            "error_type": error_type,
        })

    def record_safety_veto(self, reason: str) -> None:
        self._counter("bot.safety_vetoes", "Safety system vetoes").add(1, {"reason": reason})

    def record_vlm_fallback(self, model: str, cached: bool) -> None:
        self._counter("bot.vlm_fallback", "VLM fallback invocations").add(1, {
            "model": model,
            "cached": str(cached),
        })
