"""
core/behavioral_anomaly_detector.py

Behavioral Anomaly Detection — Sistema preventivo anti-ban.

Analisa métricas comportamentais do bot em tempo real e detecta
quando o comportamento está se tornando "não-humano" ou detectável.

Features monitoradas:
- APM (actions per minute) e sua variância
- Padrões de tempo entre ações (regularidade = suspeito)
- Posicionamento previsível (entropia espacial)
- Reação instantânea (< 100ms consistentemente)
- Padrões de swipe/perfeição mecânica
- Duração de sessão excessiva
- Win rate anômalo (> 95% ou < 5%)

Uso:
    detector = BehavioralAnomalyDetector()
    detector.record_action(action_type, timestamp)
    detector.record_position(x, y)
    risk = detector.get_ban_risk_score()
    if risk > 0.8:
        detector.trigger_mitigation()
"""

import logging
import time
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BehavioralAnomalyDetector:
    """
    Detecta anomalias comportamentais que podem levar a ban.
    """

    def __init__(
        self,
        action_history_size: int = 500,
        position_history_size: int = 1000,
        session_window_minutes: float = 10.0,
    ):
        self._actions: deque = deque(maxlen=action_history_size)
        self._positions: deque = deque(maxlen=position_history_size)
        self._session_window = session_window_minutes * 60
        self._session_start = time.time()

        # Thresholds
        self._max_apm = 45
        self._min_reaction_ms = 80
        self._max_perfection_score = 0.95  # swipes perfeitos demais
        self._min_entropy = 2.0  # entropia espacial mínima
        self._max_session_hours = 4.0
        self._max_win_rate = 0.95
        self._min_win_rate = 0.05

        # Estado
        self._ban_risk_score = 0.0
        self._anomalies_detected: list[dict] = []
        self._mitigation_triggered = False

    # ------------------------------------------------------------------
    # Registro de dados
    # ------------------------------------------------------------------

    def record_action(self, action_type: str, timestamp: float | None = None):
        """Registra uma ação do bot."""
        self._actions.append({
            "type": action_type,
            "timestamp": timestamp or time.time(),
        })

    def record_position(self, x: int, y: int):
        """Registra posição do player."""
        self._positions.append((x, y, time.time()))

    def record_match_result(self, result: str):
        """Registra resultado de partida."""
        self._actions.append({
            "type": "match_result",
            "result": result,
            "timestamp": time.time(),
        })

    def record_reaction_time(self, reaction_ms: float):
        """Registra tempo de reação (ms)."""
        self._actions.append({
            "type": "reaction",
            "ms": reaction_ms,
            "timestamp": time.time(),
        })

    # ------------------------------------------------------------------
    # Detecção de anomalias
    # ------------------------------------------------------------------

    def analyze(self) -> dict[str, Any]:
        """Roda análise completa e retorna score de risco."""
        scores = {}

        # 1. APM
        scores["apm"] = self._check_apm()

        # 2. Regularity (padrões temporais regulares = suspeito)
        scores["regularity"] = self._check_regularity()

        # 3. Spatial entropy
        scores["entropy"] = self._check_spatial_entropy()

        # 4. Reaction times
        scores["reaction"] = self._check_reaction_times()

        # 5. Session duration
        scores["session"] = self._check_session_duration()

        # 6. Win rate
        scores["winrate"] = self._check_win_rate()

        # Score combinado (média ponderada)
        weights = {"apm": 0.25, "regularity": 0.20, "entropy": 0.15,
                   "reaction": 0.20, "session": 0.10, "winrate": 0.10}

        total_risk = sum(scores[k] * weights[k] for k in weights)
        self._ban_risk_score = min(total_risk, 1.0)

        # Registrar anomalias
        for category, score in scores.items():
            if score > 0.7:
                self._anomalies_detected.append({
                    "timestamp": time.time(),
                    "category": category,
                    "score": round(score, 3),
                    "severity": "critical" if score > 0.9 else "warning",
                })

        return {
            "ban_risk_score": round(self._ban_risk_score, 3),
            "category_scores": {k: round(v, 3) for k, v in scores.items()},
            "anomalies_count": len(self._anomalies_detected),
            "mitigation_triggered": self._mitigation_triggered,
        }

    def _check_apm(self) -> float:
        """Retorna risco baseado em APM."""
        recent = [a for a in self._actions
                  if a.get("timestamp", 0) > time.time() - 60]
        apm = len(recent)
        if apm > self._max_apm:
            return min((apm - self._max_apm) / 20, 1.0)
        return 0.0

    def _check_regularity(self) -> float:
        """Detecta padrões temporais regulares (bot-like)."""
        timestamps = [a["timestamp"] for a in self._actions
                      if "timestamp" in a]
        if len(timestamps) < 10:
            return 0.0

        # Calcular diferenças entre ações consecutivas
        diffs = np.diff(timestamps)
        if len(diffs) < 5:
            return 0.0

        # Coeficiente de variação: baixo = regular = suspeito
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        if mean_diff == 0:
            return 1.0

        cv = std_diff / mean_diff
        # CV < 0.1 é muito regular
        if cv < 0.1:
            return min((0.1 - cv) / 0.1, 1.0)
        return 0.0

    def _check_spatial_entropy(self) -> float:
        """Verifica se posicionamento é previsível (baixa entropia)."""
        if len(self._positions) < 50:
            return 0.0

        # Grid 10x10
        grid = np.zeros((10, 10), dtype=np.float32)
        for x, y, _ in self._positions:
            gx = min(int(x / 192), 9)
            gy = min(int(y / 108), 9)
            grid[gy, gx] += 1

        probs = grid / grid.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(100)  # 10x10

        normalized = entropy / max_entropy
        if normalized < self._min_entropy / max_entropy:
            return min(1.0 - normalized, 1.0)
        return 0.0

    def _check_reaction_times(self) -> float:
        """Detecta reações instantâneas (não-humanas)."""
        reactions = [a["ms"] for a in self._actions
                     if a.get("type") == "reaction"]
        if len(reactions) < 10:
            return 0.0

        fast_count = sum(1 for r in reactions if r < self._min_reaction_ms)
        ratio = fast_count / len(reactions)
        # Mais de 30% de reações < 80ms é suspeito
        return max((ratio - 0.1) / 0.3, 0.0) if ratio > 0.1 else 0.0

    def _check_session_duration(self) -> float:
        """Detecta sessões excessivamente longas."""
        hours = (time.time() - self._session_start) / 3600
        if hours > self._max_session_hours:
            return min((hours - self._max_session_hours) / 2, 1.0)
        return 0.0

    def _check_win_rate(self) -> float:
        """Detecta win rate anômalo."""
        matches = [a for a in self._actions
                   if a.get("type") == "match_result"]
        if len(matches) < 20:
            return 0.0

        wins = sum(1 for m in matches if m.get("result") in ("win", "victory"))
        wr = wins / len(matches)

        if wr > self._max_win_rate:
            return min((wr - self._max_win_rate) / 0.05, 1.0)
        if wr < self._min_win_rate:
            return min((self._min_win_rate - wr) / 0.05, 1.0)
        return 0.0

    # ------------------------------------------------------------------
    # Mitigação
    # ------------------------------------------------------------------

    def trigger_mitigation(self) -> list[str]:
        """Ativa mitigações baseadas nas anomalias detectadas."""
        if self._mitigation_triggered:
            return []

        mitigations = []
        analysis = self.analyze()
        scores = analysis["category_scores"]

        if scores.get("apm", 0) > 0.5:
            mitigations.append("reduce_apm")
        if scores.get("regularity", 0) > 0.5:
            mitigations.append("increase_jitter")
        if scores.get("entropy", 0) > 0.5:
            mitigations.append("randomize_positioning")
        if scores.get("reaction", 0) > 0.5:
            mitigations.append("add_reaction_delay")
        if scores.get("session", 0) > 0.5:
            mitigations.append("force_break")
        if scores.get("winrate", 0) > 0.5:
            mitigations.append("adjust_difficulty")

        self._mitigation_triggered = True
        logger.warning("[ANOMALY] Mitigações ativadas: %s", mitigations)
        return mitigations

    def get_ban_risk_score(self) -> float:
        """Retorna score atual de risco (0-1)."""
        return self._ban_risk_score

    def reset_session(self):
        """Reseta contadores para nova sessão."""
        self._session_start = time.time()
        self._actions.clear()
        self._positions.clear()
        self._anomalies_detected.clear()
        self._mitigation_triggered = False
        self._ban_risk_score = 0.0

    def get_status(self) -> dict[str, Any]:
        return {
            "ban_risk_score": round(self._ban_risk_score, 3),
            "actions_recorded": len(self._actions),
            "positions_recorded": len(self._positions),
            "session_minutes": round((time.time() - self._session_start) / 60, 1),
            "anomalies_count": len(self._anomalies_detected),
            "mitigation_triggered": self._mitigation_triggered,
        }
