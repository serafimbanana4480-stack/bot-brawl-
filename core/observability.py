"""
core/observability.py

Sistema de observabilidade para o bot Brawl Stars.

Funcionalidades:
- Métricas de runtime (ciclos, latências, throughput)
- Eventos estruturados (match lifecycle, erros, transições de estado)
- Exportação para JSON e/ou endpoint HTTP simples
- Health checks do sistema
- FPS counter e latency histogram
- Prometheus-compatible metrics export
"""

import json
import logging
import time

__all__ = ["ObservabilityCollector", "HealthChecker", "MatchEvent", "MetricSnapshot"]
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class MatchEvent:
    event_type: str  # start, end, kill, death, state_change, error
    timestamp: float
    state: Optional[str] = None
    brawler: Optional[str] = None
    map_name: Optional[str] = None
    result: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MetricSnapshot:
    timestamp: str
    cycle_time_ms: float
    matches_total: int
    wins: int
    losses: int
    current_state: str
    brawler: Optional[str]
    map_name: Optional[str]
    avg_reward: float
    data_collector_samples: int
    last_error: Optional[str]
    fps: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class LatencyHistogram:
    """Thread-safe latency histogram for tracking operation durations."""

    def __init__(self, max_samples: int = 1000):
        self._lock = Lock()
        self._samples: deque = deque(maxlen=max_samples)

    def record(self, duration_ms: float):
        """Record a latency sample in milliseconds."""
        with self._lock:
            self._samples.append(duration_ms)

    def get_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles (p50, p95, p99)."""
        with self._lock:
            if not self._samples:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
            sorted_samples = sorted(self._samples)
            n = len(sorted_samples)
            return {
                "p50": sorted_samples[int(n * 0.50)],
                "p95": sorted_samples[min(int(n * 0.95), n - 1)],
                "p99": sorted_samples[min(int(n * 0.99), n - 1)],
                "avg": sum(sorted_samples) / n,
                "min": sorted_samples[0],
                "max": sorted_samples[-1],
            }


class FPSCounter:
    """Thread-safe FPS counter based on frame timestamps."""

    def __init__(self, window_seconds: float = 5.0):
        self._lock = Lock()
        self._frame_times: deque = deque()
        self._window_seconds = window_seconds

    def tick(self):
        """Record a frame tick."""
        now = time.time()
        with self._lock:
            self._frame_times.append(now)
            # Prune old entries
            cutoff = now - self._window_seconds
            while self._frame_times and self._frame_times[0] < cutoff:
                self._frame_times.popleft()

    def get_fps(self) -> float:
        """Calculate current FPS."""
        with self._lock:
            if len(self._frame_times) < 2:
                return 0.0
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed <= 0:
                return 0.0
            return (len(self._frame_times) - 1) / elapsed


class ObservabilityCollector:
    """Coleta métricas e eventos do bot em tempo real."""

    def __init__(self, max_events: int = 1000, metrics_dir: Optional[Path] = None):
        self._lock = Lock()
        self.events: deque = deque(maxlen=max_events)
        self.metrics_dir = Path(metrics_dir) if metrics_dir else None
        if self.metrics_dir:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Counters
        self.matches_total = 0
        self.wins = 0
        self.losses = 0
        self.cycle_times: deque = deque(maxlen=100)
        self.last_error: Optional[str] = None
        self.current_state = "unknown"
        self.current_brawler: Optional[str] = None
        self.current_map: Optional[str] = None
        self.total_reward = 0.0
        self.reward_count = 0
        self.data_collector_samples = 0

        # FPS and latency tracking
        self.fps_counter = FPSCounter(window_seconds=5.0)
        self.cycle_latency = LatencyHistogram(max_samples=1000)
        self.inference_latency = LatencyHistogram(max_samples=1000)
        self.screenshot_latency = LatencyHistogram(max_samples=500)

    def record_event(self, event_type: str, **kwargs):
        """Registra um evento estruturado."""
        event = MatchEvent(
            event_type=event_type,
            timestamp=time.time(),
            state=self.current_state,
            brawler=self.current_brawler,
            map_name=self.current_map,
            details=kwargs,
        )
        with self._lock:
            self.events.append(event)
        logger.debug(f"[OBS] Evento: {event_type} | {kwargs}")

    def record_cycle_time(self, duration_sec: float):
        """Registra duração de um ciclo de processamento."""
        with self._lock:
            self.cycle_times.append(duration_sec)
        self.cycle_latency.record(duration_sec * 1000)  # Convert to ms
        self.fps_counter.tick()

    def record_inference_time(self, duration_sec: float):
        """Registra duração de inferência YOLO."""
        self.inference_latency.record(duration_sec * 1000)

    def record_screenshot_time(self, duration_sec: float):
        """Registra duração de captura de screenshot."""
        self.screenshot_latency.record(duration_sec * 1000)

    def record_match_result(self, result: str, brawler: Optional[str] = None, map_name: Optional[str] = None):
        """Registra resultado de uma partida."""
        with self._lock:
            self.matches_total += 1
            if result == "win":
                self.wins += 1
            elif result == "loss":
                self.losses += 1
            if brawler:
                self.current_brawler = brawler
            if map_name:
                self.current_map = map_name
        self.record_event("match_end", result=result, brawler=brawler, map_name=map_name)

    def record_reward(self, value: float):
        """Registra um valor de reward."""
        with self._lock:
            self.total_reward += value
            self.reward_count += 1

    def record_error(self, error: str):
        """Registra um erro."""
        with self._lock:
            self.last_error = error
        self.record_event("error", error=error)

    def update_state(self, state: str):
        """Atualiza o estado atual do bot."""
        with self._lock:
            self.current_state = state
        self.record_event("state_change", new_state=state)

    def update_data_collector_samples(self, count: int):
        """Atualiza contagem de amostras do data collector."""
        with self._lock:
            self.data_collector_samples = count

    def get_snapshot(self) -> MetricSnapshot:
        """Retorna snapshot atual das métricas."""
        with self._lock:
            avg_cycle = sum(self.cycle_times) / len(self.cycle_times) * 1000 if self.cycle_times else 0.0
            avg_reward = self.total_reward / max(1, self.reward_count)
            fps = self.fps_counter.get_fps()
            latency = self.cycle_latency.get_percentiles()
            return MetricSnapshot(
                timestamp=datetime.now().isoformat(),
                cycle_time_ms=avg_cycle,
                matches_total=self.matches_total,
                wins=self.wins,
                losses=self.losses,
                current_state=self.current_state,
                brawler=self.current_brawler,
                map_name=self.current_map,
                avg_reward=avg_reward,
                data_collector_samples=self.data_collector_samples,
                last_error=self.last_error,
                fps=fps,
                p50_latency_ms=latency["p50"],
                p95_latency_ms=latency["p95"],
                p99_latency_ms=latency["p99"],
            )

    def get_recent_events(self, n: int = 50) -> List[Dict]:
        """Retorna os N eventos mais recentes."""
        with self._lock:
            return [e.to_dict() for e in list(self.events)[-n:]]

    def export(self) -> Dict:
        """Exporta todas as métricas e eventos para dict."""
        snapshot = self.get_snapshot()
        return {
            "snapshot": snapshot.to_dict(),
            "recent_events": self.get_recent_events(100),
            "win_rate": snapshot.wins / max(1, snapshot.matches_total),
            "latency": {
                "cycle": self.cycle_latency.get_percentiles(),
                "inference": self.inference_latency.get_percentiles(),
                "screenshot": self.screenshot_latency.get_percentiles(),
            },
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format.
        
        Returns:
            String in Prometheus format suitable for /metrics endpoint.
        """
        snapshot = self.get_snapshot()
        lines = []
        
        # Helper to format Prometheus metric
        def metric(name, value, metric_type="gauge", labels=None):
            label_str = ""
            if labels:
                label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
            lines.append(f"# TYPE {name} {metric_type}")
            lines.append(f"{name}{label_str} {value}")
        
        # Match metrics
        metric("brawl_bot_matches_total", snapshot.matches_total, "counter")
        metric("brawl_bot_wins_total", snapshot.wins, "counter")
        metric("brawl_bot_losses_total", snapshot.losses, "counter")
        win_rate = snapshot.wins / max(1, snapshot.matches_total)
        metric("brawl_bot_win_rate", f"{win_rate:.4f}")
        
        # Performance metrics
        metric("brawl_bot_fps", f"{snapshot.fps:.2f}")
        metric("brawl_bot_cycle_time_ms", f"{snapshot.cycle_time_ms:.2f}")
        metric("brawl_bot_latency_p50_ms", f"{snapshot.p50_latency_ms:.2f}")
        metric("brawl_bot_latency_p95_ms", f"{snapshot.p95_latency_ms:.2f}")
        metric("brawl_bot_latency_p99_ms", f"{snapshot.p99_latency_ms:.2f}")
        
        # Latency histograms
        cycle_lat = self.cycle_latency.get_percentiles()
        metric("brawl_bot_cycle_latency_avg_ms", f"{cycle_lat['avg']:.2f}")
        metric("brawl_bot_cycle_latency_min_ms", f"{cycle_lat['min']:.2f}")
        metric("brawl_bot_cycle_latency_max_ms", f"{cycle_lat['max']:.2f}")
        
        inf_lat = self.inference_latency.get_percentiles()
        metric("brawl_bot_inference_latency_avg_ms", f"{inf_lat['avg']:.2f}")
        metric("brawl_bot_inference_latency_p95_ms", f"{inf_lat['p95']:.2f}")
        
        ss_lat = self.screenshot_latency.get_percentiles()
        metric("brawl_bot_screenshot_latency_avg_ms", f"{ss_lat['avg']:.2f}")
        
        # Reward metrics
        metric("brawl_bot_avg_reward", f"{snapshot.avg_reward:.4f}")
        metric("brawl_bot_data_collector_samples", snapshot.data_collector_samples, "counter")
        
        # State info
        if snapshot.brawler:
            metric("brawl_bot_current_brawler", 1, "gauge", {"brawler": snapshot.brawler})
        if snapshot.map_name:
            metric("brawl_bot_current_map", 1, "gauge", {"map": snapshot.map_name})
        
        return "\n".join(lines) + "\n"

    def save_to_disk(self, filename: Optional[str] = None):
        """Persiste métricas em disco."""
        if not self.metrics_dir:
            return
        filename = filename or f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.metrics_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export(), f, indent=2, ensure_ascii=False)
        logger.info(f"[OBS] Métricas salvas em {path}")


class HealthChecker:
    """Verifica saúde dos componentes do bot."""

    def __init__(self):
        self.checks: Dict[str, Any] = {}

    def register(self, name: str, check_func):
        """Registra uma função de health check."""
        self.checks[name] = check_func

    def run(self) -> Dict[str, Any]:
        """Executa todos os health checks."""
        results = {}
        for name, func in self.checks.items():
            try:
                results[name] = {"status": "ok", "details": func()}
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        return results
