"""Distributed Tracing Service - End-to-end request tracing"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class SpanStatus(Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class SpanKind(Enum):
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    kind: SpanKind
    status: SpanStatus = SpanStatus.OK
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": (self.end_time - self.start_time) * 1000 if self.end_time else None,
            "attributes": self.attributes,
            "events": self.events,
            "errors": self.errors,
        }


class TracingService:
    def __init__(self, service_name: str = "enterprise-ai"):
        self.service_name = service_name
        self.spans: Dict[str, Span] = {}
        self.trace_history: List[Dict[str, Any]] = []
        self.max_history = 10000
        
        self._active_spans: Dict[str, Span] = {}
        self._lock = asyncio.Lock()
        
    async def start_span(self, name: str, parent_id: Optional[str] = None,
                        kind: SpanKind = SpanKind.INTERNAL,
                        attributes: Dict[str, Any] = None) -> str:
        trace_id = parent_id or str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            kind=kind,
            attributes=attributes or {},
        )
        
        async with self._lock:
            self.spans[span_id] = span
            self._active_spans[span_id] = span
        
        return span_id
    
    async def end_span(self, span_id: str, status: SpanStatus = SpanStatus.OK,
                      attributes: Dict[str, Any] = None):
        async with self._lock:
            if span_id not in self._active_spans:
                return
            
            span = self._active_spans[span_id]
            span.end_time = time.time()
            span.status = status
            
            if attributes:
                span.attributes.update(attributes)
            
            del self._active_spans[span_id]
            
            self.trace_history.append(span.to_dict())
            if len(self.trace_history) > self.max_history:
                self.trace_history = self.trace_history[-self.max_history:]
    
    async def add_span_event(self, span_id: str, name: str, 
                            attributes: Dict[str, Any] = None):
        async with self._lock:
            if span_id not in self.spans:
                return
            
            span = self.spans[span_id]
            span.events.append({
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            })
    
    async def add_span_error(self, span_id: str, error: Exception,
                            attributes: Dict[str, Any] = None):
        async with self._lock:
            if span_id not in self.spans:
                return
            
            span = self.spans[span_id]
            span.status = SpanStatus.ERROR
            span.errors.append({
                "type": type(error).__name__,
                "message": str(error),
                "timestamp": time.time(),
                "attributes": attributes or {},
            })
    
    async def trace(self, name: str, kind: SpanKind = SpanKind.INTERNAL,
                   attributes: Dict[str, Any] = None):
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                span_id = await self.start_span(
                    name, kind=kind, attributes=attributes
                )
                try:
                    result = await func(*args, **kwargs)
                    await self.end_span(span_id, SpanStatus.OK)
                    return result
                except Exception as e:
                    await self.add_span_error(span_id, e)
                    await self.end_span(span_id, SpanStatus.ERROR)
                    raise
            return wrapper
        return decorator
    
    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        return [s for s in self.trace_history if s["trace_id"] == trace_id]
    
    def get_span(self, span_id: str) -> Optional[Dict[str, Any]]:
        for span in self.trace_history:
            if span["span_id"] == span_id:
                return span
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        total_spans = len(self.trace_history)
        error_spans = sum(1 for s in self.trace_history if s["status"] == "error")
        
        durations = [
            s["duration_ms"] for s in self.trace_history 
            if s.get("duration_ms") is not None
        ]
        
        return {
            "total_traces": total_spans,
            "error_count": error_spans,
            "error_rate": error_spans / total_spans if total_spans > 0 else 0,
            "average_duration_ms": sum(durations) / len(durations) if durations else 0,
            "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            "p99_duration_ms": sorted(durations)[int(len(durations) * 0.99)] if durations else 0,
        }
