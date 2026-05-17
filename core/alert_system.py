"""
core/alert_system.py

Sistema de Alertas Inteligentes para Soberana Omega.

Detecta condições anômalas e emite alertas com severidade e recomendações.
Integra com EventStore para auditoria e pode notificar via dashboard.

Tipos de alerta:
- PERFORMANCE: ciclo muito lento, FPS baixo
- SAFETY: APM muito alto, padrões suspeitos
- HEALTH: muitos erros, circuit breaker aberto
- DETECTION: risco de ban aumentando
- SYSTEM: crash iminente, memória alta
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCategory(Enum):
    PERFORMANCE = "performance"
    SAFETY = "safety"
    HEALTH = "health"
    DETECTION = "detection"
    SYSTEM = "system"


@dataclass
class Alert:
    """Um alerta emitido pelo sistema."""
    id: str
    category: AlertCategory
    severity: AlertSeverity
    message: str
    recommendation: str
    timestamp: float
    acknowledged: bool = False
    auto_resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertSystem:
    """
    Sistema de alertas com thresholds configuráveis.
    """

    # Thresholds padrão
    DEFAULT_THRESHOLDS = {
        "cycle_time_max_ms": 2000.0,
        "fps_min": 5.0,
        "apm_max": 50,
        "error_rate_max": 0.3,
        "degradation_emergency": True,
        "consecutive_losses": 5,
        "memory_usage_max_mb": 2048,
    }

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._alerts: deque = deque(maxlen=100)
        self._alert_counter = 0
        self._last_check = 0.0
        self._check_interval = 5.0  # segundos

    def check_alerts(
        self,
        cycle_duration: float = 0.0,
        wrapper_state: str = "unknown",
        degradation_status: Optional[Dict] = None,
        fps: float = 0.0,
        apm: int = 0,
        error_count: int = 0,
    ):
        """
        Verifica condições e emite alertas se necessário.
        Chamada a cada ciclo do monitor loop.
        """
        now = time.time()
        if now - self._last_check < self._check_interval:
            return
        self._last_check = now

        # Performance alerts
        if cycle_duration > self.thresholds["cycle_time_max_ms"] / 1000.0:
            self._emit(
                AlertCategory.PERFORMANCE,
                AlertSeverity.WARNING,
                f"Ciclo muito lento: {cycle_duration*1000:.0f}ms",
                "Reduzir inferência ou ativar degradação",
                {"cycle_duration": cycle_duration},
            )

        if fps > 0 and fps < self.thresholds["fps_min"]:
            self._emit(
                AlertCategory.PERFORMANCE,
                AlertSeverity.WARNING,
                f"FPS muito baixo: {fps:.1f}",
                "Verificar GPU/CPU e reduzir resolução YOLO",
                {"fps": fps},
            )

        # Safety alerts
        if apm > self.thresholds["apm_max"]:
            self._emit(
                AlertCategory.SAFETY,
                AlertSeverity.CRITICAL,
                f"APM muito alto: {apm} (limite: {self.thresholds['apm_max']})",
                "Aumentar delays de humanização imediatamente",
                {"apm": apm},
            )

        # Degradation alert
        if degradation_status:
            mode = degradation_status.get("mode", "full_quality")
            if mode == "emergency" and self.thresholds.get("degradation_emergency", True):
                self._emit(
                    AlertCategory.HEALTH,
                    AlertSeverity.CRITICAL,
                    "Modo EMERGENCY ativado — sistema crítico",
                    "Verificar logs e reiniciar componentes",
                    degradation_status,
                )
            elif mode == "minimal":
                self._emit(
                    AlertCategory.HEALTH,
                    AlertSeverity.WARNING,
                    "Modo MINIMAL — funcionalidade reduzida",
                    "Verificar erros e recuperar gradualmente",
                    degradation_status,
                )

        # Error rate
        if error_count > 10:
            self._emit(
                AlertCategory.HEALTH,
                AlertSeverity.WARNING,
                f"Muitos erros acumulados: {error_count}",
                "Verificar error_recovery.log e circuit breakers",
                {"error_count": error_count},
            )

    def _emit(
        self,
        category: AlertCategory,
        severity: AlertSeverity,
        message: str,
        recommendation: str,
        metadata: Optional[Dict] = None,
    ) -> Alert:
        """Emite um novo alerta."""
        self._alert_counter += 1
        alert = Alert(
            id=f"alert_{self._alert_counter}",
            category=category,
            severity=severity,
            message=message,
            recommendation=recommendation,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._alerts.append(alert)
        logger.warning(
            "[ALERT] %s | %s | %s | Recomendação: %s",
            severity.value.upper(), category.value, message, recommendation
        )
        return alert

    def get_active_alerts(self, min_severity: Optional[AlertSeverity] = None) -> List[Dict]:
        """Retorna alertas ativos (não resolvidos)."""
        result = []
        for alert in self._alerts:
            if alert.acknowledged or alert.auto_resolved:
                continue
            if min_severity and severity_rank(alert.severity) < severity_rank(min_severity):
                continue
            result.append({
                "id": alert.id,
                "category": alert.category.value,
                "severity": alert.severity.value,
                "message": alert.message,
                "recommendation": alert.recommendation,
                "timestamp": alert.timestamp,
                "age_seconds": round(time.time() - alert.timestamp, 1),
            })
        return result

    def acknowledge(self, alert_id: str) -> bool:
        """Marca alerta como reconhecido."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def resolve(self, alert_id: str) -> bool:
        """Marca alerta como resolvido."""
        for alert in alert_id in self._alerts:
            if alert.id == alert_id:
                alert.auto_resolved = True
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Estatísticas de alertas."""
        total = len(self._alerts)
        by_severity = {}
        by_category = {}
        active = 0
        for alert in self._alerts:
            by_severity[alert.severity.value] = by_severity.get(alert.severity.value, 0) + 1
            by_category[alert.category.value] = by_category.get(alert.category.value, 0) + 1
            if not alert.acknowledged and not alert.auto_resolved:
                active += 1
        return {
            "total_alerts": total,
            "active_alerts": active,
            "by_severity": by_severity,
            "by_category": by_category,
        }


def severity_rank(sev: AlertSeverity) -> int:
    """Rank numérico para comparação de severidade."""
    return {"info": 0, "warning": 1, "critical": 2}.get(sev.value, 0)
