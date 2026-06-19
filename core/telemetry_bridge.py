"""
core/telemetry_bridge.py

Bridge de telemetria em tempo real entre V2Integrator e Dashboard.

Conecta os dados dos módulos v2.1 ao DashboardDataBridge para
exibição no dashboard web. Atualiza automaticamente a cada ciclo.

Endpoints expostos:
- /api/v2/status        → dados agregados de todos os módulos
- /api/v2/degradation   → modo e config de degradação
- /api/v2/alerts        → alertas ativos
- /api/v2/rate-limiter  → status da conta
- /api/v2/checkpoints   → estatísticas de checkpoints
- /api/v2/telemetry     → stream de dados em tempo real
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TelemetrySnapshot:
    """Snapshot completo de telemetria para o dashboard."""
    timestamp: float = 0.0
    v2_enabled: bool = False
    cycle_count: int = 0
    # Degradation
    degradation_mode: str = "unknown"
    degradation_fps: float = 0.0
    degradation_max_apm: int = 0
    # Rate Limiter
    can_play: bool = True
    account_session_minutes: float = 0.0
    win_streak: int = 0
    loss_streak: int = 0
    # Brawler
    current_brawler: str | None = None
    current_playstyle: str | None = None
    # Checkpoints
    checkpoint_count: int = 0
    checkpoint_age_seconds: float | None = None
    # Alerts
    active_alerts: int = 0
    alerts: list = field(default_factory=list)
    # Combat
    recent_actions: list = field(default_factory=list)
    action_diversity: int = 0
    # Frame Skip
    processed_ratio: float = 1.0
    # Performance
    cycle_duration_ms: float = 0.0
    # Tracing
    slow_spans: list = field(default_factory=list)


class TelemetryBridge:
    """
    Coleta dados de todos os módulos v2.1 e os disponibiliza para o dashboard.

    Design:
    - Atualizado a cada ciclo pelo V2Integrator.on_cycle_end()
    - Thread-safe para acesso pelo dashboard
    - Cache dos últimos 100 snapshots para histórico
    """

    def __init__(self, max_history: int = 100):
        self._current: TelemetrySnapshot = TelemetrySnapshot()
        self._history: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._last_update = 0.0
        self._update_interval = 1.0  # mínimo 1s entre updates

    def update(self, v2_integrator: Any, cycle_duration: float = 0.0):
        """
        Atualiza snapshot a partir do V2Integrator.
        Chamado pelo V2Integrator a cada ciclo.
        """
        now = time.time()
        if now - self._last_update < self._update_interval:
            return  # Throttle
        self._last_update = now

        snapshot = TelemetrySnapshot(timestamp=now)

        if not v2_integrator or not getattr(v2_integrator, "config", None):
            return

        cfg = v2_integrator.config
        snapshot.v2_enabled = cfg.enabled if cfg else False
        snapshot.cycle_count = getattr(v2_integrator, "_cycle_count", 0)

        # Degradation
        if v2_integrator._degradation_mgr:
            deg = v2_integrator._degradation_mgr.get_status()
            snapshot.degradation_mode = deg.get("mode", "unknown")
            snapshot.degradation_fps = deg.get("target_fps", 0.0)
            snapshot.degradation_max_apm = deg.get("max_apm", 0)

        # Rate Limiter
        if v2_integrator._rate_limiter:
            acc = v2_integrator.config.account_id
            rl = v2_integrator._rate_limiter.get_account_status(acc)
            snapshot.can_play = rl.get("should_play_now", True)
            snapshot.account_session_minutes = rl.get("current_session_minutes", 0.0)
            snapshot.win_streak = rl.get("win_streak", 0)
            snapshot.loss_streak = rl.get("loss_streak", 0)

        # Brawler
        if v2_integrator._brawler_ctrl:
            bstat = v2_integrator._brawler_ctrl.get_status()
            snapshot.current_brawler = bstat.get("current_brawler")
            snapshot.current_playstyle = bstat.get("current_playstyle")

        # Checkpoints
        if v2_integrator._checkpointer:
            cp = v2_integrator._checkpointer.get_stats()
            snapshot.checkpoint_count = cp.get("total_checkpoints", 0)
            snapshot.checkpoint_age_seconds = cp.get("last_checkpoint_age_seconds")

        # Alerts
        if v2_integrator._alert_system:
            alerts = v2_integrator._alert_system.get_active_alerts()
            snapshot.active_alerts = len(alerts)
            snapshot.alerts = alerts[:5]  # top 5

        # Frame Skip
        if v2_integrator._frame_skipper:
            fs = v2_integrator._frame_skipper.get_stats()
            snapshot.processed_ratio = fs.get("processed_ratio", 1.0)

        # Performance
        snapshot.cycle_duration_ms = round(cycle_duration * 1000, 2)

        # Tracing
        if v2_integrator._tracer:
            snapshot.slow_spans = v2_integrator._tracer.get_slow_spans(threshold_ms=100, limit=5)

        with self._lock:
            self._current = snapshot
            self._history.append(snapshot)

    def get_current(self) -> TelemetrySnapshot:
        """Retorna snapshot mais recente (thread-safe)."""
        with self._lock:
            return self._current

    def get_history(self, limit: int = 100) -> list:
        """Retorna histórico de snapshots."""
        with self._lock:
            return list(self._history)[-limit:]

    def to_dict(self) -> dict[str, Any]:
        """Converte snapshot atual para dict JSON-serializável."""
        snap = self.get_current()
        return {
            "timestamp": snap.timestamp,
            "v2_enabled": snap.v2_enabled,
            "cycle_count": snap.cycle_count,
            "degradation": {
                "mode": snap.degradation_mode,
                "target_fps": snap.degradation_fps,
                "max_apm": snap.degradation_max_apm,
            },
            "rate_limiter": {
                "can_play": snap.can_play,
                "session_minutes": snap.account_session_minutes,
                "win_streak": snap.win_streak,
                "loss_streak": snap.loss_streak,
            },
            "brawler": {
                "name": snap.current_brawler,
                "playstyle": snap.current_playstyle,
            },
            "checkpoints": {
                "count": snap.checkpoint_count,
                "age_seconds": snap.checkpoint_age_seconds,
            },
            "alerts": {
                "active_count": snap.active_alerts,
                "items": snap.alerts,
            },
            "performance": {
                "processed_ratio": snap.processed_ratio,
                "cycle_duration_ms": snap.cycle_duration_ms,
            },
            "slow_spans": snap.slow_spans,
        }
