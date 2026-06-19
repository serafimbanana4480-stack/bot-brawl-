"""
core/distributed_tracing.py

Observability Distribuído com tracing end-to-end.

Implementa conceitos do OpenTelemetry sem dependências externas pesadas:
- Spans aninhados (perception → decision → action)
- Context propagation entre subsistemas
- Exportação para JSON (compatível com Jaeger/Zipkin futuro)
- Métricas de latência por componente
- Correlação entre falhas e bottlenecks

Uso:
    with Tracer().start_span("game_cycle") as cycle_span:
        with Tracer().start_span("perception", parent=cycle_span):
            screenshot = capture()
        with Tracer().start_span("decision", parent=cycle_span):
            action = decide(screenshot)
"""

import functools
import json
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Um span de tracing — representa uma operação."""
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    status: str = "ok"  # ok | error | cancelled
    tags: dict[str, str] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    children: list["Span"] = field(default_factory=list)

    def finish(self, status: str = "ok", tags: dict[str, str] = None):
        """Finaliza o span."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        if tags:
            self.tags.update(tags)

    def add_log(self, message: str, **fields):
        """Adiciona log estruturado ao span."""
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            "fields": fields,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms else None,
            "status": self.status,
            "tags": self.tags,
            "logs": self.logs,
        }


class SpanContext:
    """Contexto de tracing que propaga entre funções."""

    def __init__(self, trace_id: str | None = None, span_id: str | None = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = span_id or str(uuid.uuid4())

    def child(self) -> "SpanContext":
        """Cria contexto filho."""
        return SpanContext(trace_id=self.trace_id, span_id=str(uuid.uuid4()))


class Tracer:
    """
    Tracer simples mas completo para Soberana Omega.

    Não usa OpenTelemetry para evitar overhead e dependências,
    mas exporta formato compatível.
    """

    def __init__(
        self,
        service_name: str = "soberana_omega",
        export_dir: Path | None = None,
        max_spans_in_memory: int = 10000,
    ):
        self.service_name = service_name
        self.export_dir = Path(export_dir) if export_dir else Path("logs/traces")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.max_spans = max_spans_in_memory

        self._active_spans: dict[str, Span] = {}
        self._finished_spans: deque = deque(maxlen=max_spans_in_memory)
        self._lock = threading.RLock()

        # Thread-local para contexto atual
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Span lifecycle
    # ------------------------------------------------------------------

    def start_span(
        self,
        name: str,
        parent: Span | None = None,
        tags: dict[str, str] = None,
    ) -> Span:
        """Inicia um novo span."""
        trace_id = parent.trace_id if parent else str(uuid.uuid4())
        parent_id = parent.span_id if parent else None

        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_id=parent_id,
            name=name,
            start_time=time.time(),
            tags=tags or {},
        )

        with self._lock:
            self._active_spans[span.span_id] = span
            if parent:
                parent.children.append(span)

        return span

    def finish_span(self, span: Span, status: str = "ok", tags: dict[str, str] = None):
        """Finaliza um span e o arquiva."""
        span.finish(status, tags)

        with self._lock:
            self._active_spans.pop(span.span_id, None)
            self._finished_spans.append(span)

    def current_span(self) -> Span | None:
        """Retorna span ativo do contexto atual."""
        return getattr(self._local, "current_span", None)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def start_as_current_span(self, name: str, tags: dict[str, str] = None):
        """Context manager para span com propagação automática."""
        return _SpanContextManager(self, name, tags)

    # ------------------------------------------------------------------
    # Exportação
    # ------------------------------------------------------------------

    def export_spans(self, force: bool = False) -> Path:
        """Exporta spans acumulados para arquivo JSON."""
        with self._lock:
            if not self._finished_spans and not force:
                return None

            spans_to_export = list(self._finished_spans)
            self._finished_spans.clear()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        export_path = self.export_dir / f"trace_{timestamp}.json"

        data = {
            "service": self.service_name,
            "exported_at": time.time(),
            "span_count": len(spans_to_export),
            "spans": [s.to_dict() for s in spans_to_export],
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug("[TRACER] %d spans exportados para %s", len(spans_to_export), export_path.name)
        return export_path

    def export_jaeger_format(self) -> Path:
        """Exporta no formato JSON aceito por Jaeger UI."""
        with self._lock:
            spans_to_export = list(self._finished_spans)
            self._finished_spans.clear()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        export_path = self.export_dir / f"jaeger_trace_{timestamp}.json"

        # Formato Jaeger-like
        jaeger_data = {
            "data": [
                {
                    "traceID": s.trace_id,
                    "spanID": s.span_id,
                    "parentSpanID": s.parent_id or "",
                    "operationName": s.name,
                    "startTime": int(s.start_time * 1_000_000),  # micros
                    "duration": int((s.duration_ms or 0) * 1000),  # micros
                    "tags": [{"key": k, "value": v} for k, v in s.tags.items()],
                    "logs": [
                        {
                            "timestamp": int(log_entry["timestamp"] * 1_000_000),
                            "fields": [{"key": k, "value": str(v)} for k, v in log_entry["fields"].items()] + [{"key": "message", "value": log_entry["message"]}],
                        }
                        for log_entry in s.logs
                    ],
                    "status": {"code": s.status},
                }
                for s in spans_to_export
            ],
            "total": len(spans_to_export),
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(jaeger_data, f, indent=2, default=str)

        return export_path

    # ------------------------------------------------------------------
    # Análise
    # ------------------------------------------------------------------

    def get_slow_spans(self, threshold_ms: float = 100.0, limit: int = 20) -> list[dict]:
        """Retorna spans mais lentos que threshold."""
        with self._lock:
            slow = [s for s in self._finished_spans if s.duration_ms and s.duration_ms > threshold_ms]
            slow_sorted = sorted(slow, key=lambda x: x.duration_ms or 0, reverse=True)

        return [
            {
                "name": s.name,
                "duration_ms": round(s.duration_ms, 2),
                "trace_id": s.trace_id,
                "tags": s.tags,
            }
            for s in slow_sorted[:limit]
        ]

    def get_error_spans(self, limit: int = 20) -> list[dict]:
        """Retorna spans com erro."""
        with self._lock:
            errors = [s for s in self._finished_spans if s.status == "error"]
            errors_sorted = sorted(errors, key=lambda x: x.start_time, reverse=True)

        return [
            {
                "name": s.name,
                "duration_ms": round(s.duration_ms, 2) if s.duration_ms else None,
                "trace_id": s.trace_id,
                "tags": s.tags,
            }
            for s in errors_sorted[:limit]
        ]

    def get_latency_summary(self) -> dict[str, Any]:
        """Resumo de latência por tipo de operação."""
        with self._lock:
            spans = list(self._finished_spans)

        from collections import defaultdict
        by_name = defaultdict(list)
        for s in spans:
            if s.duration_ms is not None:
                by_name[s.name].append(s.duration_ms)

        summary = {}
        for name, durations in by_name.items():
            durations_sorted = sorted(durations)
            n = len(durations_sorted)
            summary[name] = {
                "count": n,
                "p50": durations_sorted[int(n * 0.5)],
                "p95": durations_sorted[min(int(n * 0.95), n - 1)],
                "p99": durations_sorted[min(int(n * 0.99), n - 1)],
                "avg": sum(durations) / n,
                "max": durations_sorted[-1],
            }

        return summary

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def trace(self, name: str | None = None, tags: dict[str, str] = None):
        """Decorator para tracing automático de funções."""
        def decorator(func: Callable) -> Callable:
            span_name = name or func.__name__
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                span = self.start_span(span_name, tags=tags)
                try:
                    result = func(*args, **kwargs)
                    span.add_log("success")
                    self.finish_span(span, status="ok")
                    return result
                except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                    span.add_log("error", error=str(e))
                    self.finish_span(span, status="error", tags={"error": str(e)})
                    raise
            return wrapper
        return decorator


class _SpanContextManager:
    """Context manager para spans."""

    def __init__(self, tracer: Tracer, name: str, tags: dict[str, str] = None):
        self.tracer = tracer
        self.name = name
        self.tags = tags or {}
        self.span: Span | None = None

    def __enter__(self) -> Span:
        parent = self.tracer.current_span()
        self.span = self.tracer.start_span(self.name, parent=parent, tags=self.tags)
        self.tracer._local.current_span = self.span
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_val:
                self.span.add_log("exception", error=str(exc_val), type=exc_type.__name__ if exc_type else None)
                self.tracer.finish_span(self.span, status="error")
            else:
                self.tracer.finish_span(self.span, status="ok")
        # Restaurar parent
        if self.span and self.span.parent_id:
            for s in self.tracer._active_spans.values():
                if s.span_id == self.span.parent_id:
                    self.tracer._local.current_span = s
                    break
            else:
                self.tracer._local.current_span = None
        else:
            self.tracer._local.current_span = None
