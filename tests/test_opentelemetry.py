"""
tests/test_opentelemetry.py

Tests for OpenTelemetry tracing and metrics integration (Phase 5).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from core.opentelemetry_tracing import (
    initialize_tracing,
    shutdown_tracing,
    span,
    record_error,
    get_trace_id,
    NoOpSpan,
    NoOpTracer,
)
from core.opentelemetry_metrics import OTelMetrics, NoOpCounter, NoOpHistogram


# ------------------------------------------------------------------
# Tracing tests
# ------------------------------------------------------------------

class TestInitializeTracing:
    def test_disabled_returns_false(self):
        assert initialize_tracing(enabled=False) is False

    def test_no_opentelemetry_returns_false(self):
        with patch("core.opentelemetry_tracing.HAS_OTEL", False):
            assert initialize_tracing(enabled=True) is False

    def test_console_exporter_initializes(self):
        with patch("core.opentelemetry_tracing.HAS_OTEL", True):
            with patch("core.opentelemetry_tracing.trace") as mock_trace:
                mock_provider = MagicMock()
                mock_trace.TracerProvider.return_value = mock_provider
                mock_trace.get_tracer_provider.return_value = mock_provider
                mock_trace.set_tracer_provider.return_value = None

                ok = initialize_tracing(enabled=True, exporter_type="console")
                assert ok is True
                mock_trace.set_tracer_provider.assert_called_once()

    def test_shutdown_does_not_crash(self):
        shutdown_tracing()  # should not raise even if never initialized


class TestSpanContextManager:
    def test_no_op_when_otel_unavailable(self):
        with patch("core.opentelemetry_tracing.HAS_OTEL", False):
            with patch("core.opentelemetry_tracing._tracer_instance", None):
                with span("test.span") as sp:
                    assert isinstance(sp, NoOpSpan)
                    sp.set_attribute("key", "value")

    def test_attributes_set_on_real_span(self):
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = lambda *a: mock_span
        mock_tracer.start_as_current_span.return_value.__exit__ = lambda *a: None

        with patch("core.opentelemetry_tracing._tracer_instance", mock_tracer):
            with patch("core.opentelemetry_tracing.HAS_OTEL", True):
                with span("test.span", attributes={"foo": "bar"}) as sp:
                    sp.set_attribute("baz", 42)

        mock_span.set_attribute.assert_any_call("foo", "bar")
        mock_span.set_attribute.assert_any_call("baz", 42)

    def test_record_error_on_noop_is_safe(self):
        noop = NoOpSpan("noop")
        record_error(noop, ValueError("test"), message="oops")
        # Should not raise

    def test_get_trace_id_when_disabled(self):
        with patch("core.opentelemetry_tracing.HAS_OTEL", False):
            assert get_trace_id() is None


# ------------------------------------------------------------------
# Metrics tests
# ------------------------------------------------------------------

class TestOTelMetrics:
    def test_disabled_instance_uses_noop(self):
        metrics = OTelMetrics(enabled=False)
        counter = metrics._counter("test.counter", "A test counter")
        assert isinstance(counter, NoOpCounter)
        counter.add(1)  # should not raise

    def test_histogram_noop(self):
        metrics = OTelMetrics(enabled=False)
        hist = metrics._histogram("test.hist", "A test histogram", "ms")
        assert isinstance(hist, NoOpHistogram)
        hist.record(42.0)  # should not raise

    def test_record_match_start_noop(self):
        metrics = OTelMetrics(enabled=False)
        metrics.record_match_start(map_name="Gem Grab", brawler="Colt")

    def test_record_match_end_noop(self):
        metrics = OTelMetrics(enabled=False)
        metrics.record_match_end(result="win", map_name="Gem Grab")

    def test_record_action_noop(self):
        metrics = OTelMetrics(enabled=False)
        metrics.record_action("tap")

    def test_record_latencies_noop(self):
        metrics = OTelMetrics(enabled=False)
        metrics.record_cycle_latency(45.0)
        metrics.record_decision_latency(12.0)
        metrics.record_vision_latency(30.0)

    def test_record_error_noop(self):
        metrics = OTelMetrics(enabled=False)
        metrics.record_error("vision", "timeout")

    def test_record_safety_veto_noop(self):
        metrics = OTelMetrics(enabled=False)
        metrics.record_safety_veto("apm_limit")

    def test_record_vlm_fallback_noop(self):
        metrics = OTelMetrics(enabled=False)
        metrics.record_vlm_fallback("openai:gpt-4o", cached=True)

    def test_singleton_behavior(self):
        m1 = OTelMetrics(enabled=False)
        m2 = OTelMetrics(enabled=False)
        assert m1 is m2

    def test_enabled_but_init_falls_back(self):
        # If OTEL SDK is unavailable, enabled should become False
        with patch("core.opentelemetry_metrics.HAS_OTEL_METRICS", False):
            metrics = OTelMetrics(enabled=True)
            assert metrics.enabled is False
