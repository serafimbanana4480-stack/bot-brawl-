"""
core/metrics.py

Prometheus metrics helpers for the Brawl Stars Bot.
"""

from prometheus_client import Counter, Gauge, Histogram

state_gauge = Gauge(
    "bot_state_current",
    "Current bot state (1 for active state, 0 otherwise)",
    ["state"],
)

matches_counter = Counter(
    "bot_matches_completed_total",
    "Total number of matches completed",
)

errors_counter = Counter(
    "bot_errors_total",
    "Total number of errors by type",
    ["type"],
)

confidence_gauge = Gauge(
    "bot_detection_confidence",
    "Detection confidence score by method",
    ["method"],
)

cycle_duration = Histogram(
    "bot_cycle_duration_seconds",
    "Duration of a bot cycle in seconds",
)

_last_state: str = ""


def set_bot_state(state: str) -> None:
    """Set the current bot state to 1 and clear previous state."""
    global _last_state
    if _last_state and _last_state != state:
        state_gauge.labels(state=_last_state).set(0)
    state_gauge.labels(state=state).set(1)
    _last_state = state


def inc_matches_completed() -> None:
    matches_counter.inc()


def inc_errors(error_type: str) -> None:
    errors_counter.labels(type=error_type).inc()


def set_detection_confidence(method: str, value: float) -> None:
    confidence_gauge.labels(method=method).set(value)


def observe_cycle_duration(seconds: float) -> None:
    cycle_duration.observe(seconds)
