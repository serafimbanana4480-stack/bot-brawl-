"""Metrics Collection Service - System and application metrics"""

import asyncio
import time
import psutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import Enum


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


class MetricsCollector:
    def __init__(self, retention_period: int = 3600):
        self.retention_period = retention_period
        self.metrics: Dict[str, deque] = {}
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        
    async def increment(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        async with self._lock:
            if name not in self.counters:
                self.counters[name] = 0.0
            self.counters[name] += value
            
            point = MetricPoint(
                name=name,
                value=self.counters[name],
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.COUNTER,
            )
            
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=10000)
            self.metrics[name].append(point)
    
    async def gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        async with self._lock:
            self.gauges[name] = value
            
            point = MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.GAUGE,
            )
            
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=10000)
            self.metrics[name].append(point)
    
    async def histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        async with self._lock:
            point = MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.HISTOGRAM,
            )
            
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=10000)
            self.metrics[name].append(point)
    
    async def get_metric(self, name: str, duration: float = None) -> List[MetricPoint]:
        if name not in self.metrics:
            return []
        
        cutoff = time.time() - duration if duration else 0
        return [p for p in self.metrics[name] if p.timestamp >= cutoff]
    
    async def get_current_value(self, name: str) -> float:
        if name in self.counters:
            return self.counters[name]
        if name in self.gauges:
            return self.gauges[name]
        return 0.0
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available": psutil.virtual_memory().available,
            "memory_total": psutil.virtual_memory().total,
            "disk_percent": psutil.disk_usage('/').percent,
            "network_connections": len(psutil.net_connections()),
        }
    
    async def get_all_metrics_summary(self) -> Dict[str, Any]:
        summary = {
            "counters": self.counters.copy(),
            "gauges": self.gauges.copy(),
        }
        
        for name, points in self.metrics.items():
            if points:
                recent = list(points)[-100:]
                values = [p.value for p in recent]
                
                summary[f"{name}_stats"] = {
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "avg": sum(values) / len(values) if values else 0,
                    "last": values[-1] if values else 0,
                }
        
        return summary
