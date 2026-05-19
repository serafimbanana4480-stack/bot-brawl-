"""
wrapper_monitoring.py

Loop de monitoring e health checks para PylaAIEnhanced.
Extraído do wrapper.py para reduzir complexidade.

Autor: Sobberana Omega
"""

import logging
import threading
import time
from typing import Optional, Any, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Status de saúde de um subsistema."""
    name: str
    healthy: bool
    last_update: float
    error: Optional[str] = None


class WrapperMonitor:
    """Monitor de saúde e estatísticas do wrapper."""

    def __init__(self, heartbeat_timeout: float = 30.0, max_unknown_duration: float = 60.0):
        self.heartbeat_timeout = heartbeat_timeout
        self.max_unknown_duration = max_unknown_duration
        self._last_action_time = time.time()
        self._unknown_state_start: Optional[float] = None
        self._health_lock = threading.Lock()
        self._subsystem_health: Dict[str, HealthStatus] = {}

    def record_action(self):
        """Regista que uma ação foi executada."""
        with self._health_lock:
            self._last_action_time = time.time()

    def check_heartbeat(self) -> bool:
        """Verifica se o loop principal está activo."""
        with self._health_lock:
            elapsed = time.time() - self._last_action_time
            if elapsed > self.heartbeat_timeout:
                logger.warning(f"[HEALTH] Heartbeat timeout: {elapsed:.1f}s sem ação")
                return False
            return True

    def check_unknown_state(self, current_state: str) -> bool:
        """Verifica se está preso em estado desconhecido."""
        if current_state == "unknown":
            if self._unknown_state_start is None:
                self._unknown_state_start = time.time()
            else:
                elapsed = time.time() - self._unknown_state_start
                if elapsed > self.max_unknown_duration:
                    logger.warning(f"[HEALTH] Preso em estado unknown há {elapsed:.1f}s")
                    return False
        else:
            self._unknown_state_start = None
        return True

    def update_subsystem_health(self, name: str, healthy: bool, error: Optional[str] = None):
        """Atualiza status de um subsistema."""
        with self._health_lock:
            self._subsystem_health[name] = HealthStatus(
                name=name,
                healthy=healthy,
                last_update=time.time(),
                error=error
            )

    def get_all_health(self) -> Dict[str, HealthStatus]:
        """Retorna status de todos os subsistemas."""
        with self._health_lock:
            return dict(self._subsystem_health)

    def is_all_healthy(self) -> bool:
        """Verifica se todos os subsistemas estão saudáveis."""
        with self._health_lock:
            if not self._subsystem_health:
                return True
            return all(h.healthy for h in self._subsystem_health.values())


class MonitoringCollector:
    """Coleta estatísticas para o dashboard."""

    def __init__(self):
        self._stats_lock = threading.Lock()
        self._stats = {
            "cycle_times": [],
            "fps_samples": [],
            "errors": [],
            "last_error": None,
        }
        self._last_cycle_time = time.time()

    def record_cycle(self, duration_ms: float):
        """Regista tempo de um ciclo."""
        with self._stats_lock:
            self._stats["cycle_times"].append(duration_ms)
            if len(self._stats["cycle_times"]) > 100:
                self._stats["cycle_times"] = self._stats["cycle_times"][-100:]

    def record_fps(self, fps: float):
        """Regista sample de FPS."""
        with self._stats_lock:
            self._stats["fps_samples"].append(fps)
            if len(self._stats["fps_samples"]) > 100:
                self._stats["fps_samples"] = self._stats["fps_samples"][-100:]

    def record_error(self, error: str):
        """Regista um erro."""
        with self._stats_lock:
            self._stats["errors"].append({"time": time.time(), "error": error})
            if len(self._stats["errors"]) > 50:
                self._stats["errors"] = self._stats["errors"][-50:]
            self._stats["last_error"] = error

    def get_avg_cycle_time(self) -> float:
        """Retorna tempo médio de ciclo em ms."""
        with self._stats_lock:
            if not self._stats["cycle_times"]:
                return 0.0
            return sum(self._stats["cycle_times"]) / len(self._stats["cycle_times"])

    def get_avg_fps(self) -> float:
        """Retorna FPS médio."""
        with self._stats_lock:
            if not self._stats["fps_samples"]:
                return 0.0
            return sum(self._stats["fps_samples"]) / len(self._stats["fps_samples"])

    def get_stats(self) -> dict:
        """Retorna todas as estatísticas."""
        with self._stats_lock:
            return dict(self._stats)