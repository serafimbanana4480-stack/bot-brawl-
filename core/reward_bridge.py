"""
core/reward_bridge.py

Ponte de integracao entre o sistema de rewards e o gameplay.
Conecta StateManager, PlayLogic, MatchController e o RealRewardCalculator.

Funcionalidades:
- Calcula rewards em tempo real durante o gameplay
- Acumula metricas por match
- Emite rewards por frame e por match completo
- Integra com DataCollector para logging
"""

import time
import logging
from datetime import datetime

__all__ = ["RewardBridge"]
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy import para evitar circular imports
try:
    from training.real_reward_system import RealRewardCalculator, GameMetrics
except ImportError:
    RealRewardCalculator = None
    GameMetrics = None


class RewardBridge:
    """Ponte entre gameplay e sistema de rewards."""

    def __init__(self, data_collector=None):
        self.data_collector = data_collector
        self.calculator = None
        if RealRewardCalculator is not None:
            self.calculator = RealRewardCalculator()

        # Metricas acumuladas por match
        self.reset_match_metrics()
        self.match_start_time: Optional[float] = None
        self.frame_count = 0
        self.last_reward: float = 0.0

        logger.info("[REWARD_BRIDGE] Inicializado")

    def reset_match_metrics(self):
        """Reseta metricas para novo match."""
        self.current_metrics = {
            "kills": 0,
            "deaths": 0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "power_cubes_collected": 0,
            "power_cubes_denied": 0,
            "objectives_completed": 0,
            "enemies_detected": 0,
            "good_decisions": 0,
            "bad_decisions": 0,
            "detection_accuracy": 0.0,
            "decision_accuracy": 0.0,
        }
        self.frame_count = 0

    def start_match(self) -> None:
        """Chamado quando uma partida comeca."""
        self.reset_match_metrics()
        self.match_start_time = time.time()
        logger.info("[REWARD_BRIDGE] Match iniciado, metricas resetadas")

    def log_combat_frame(
        self,
        enemies_detected: int = 0,
        damage_dealt: float = 0.0,
        damage_taken: float = 0.0,
        power_cubes_collected: int = 0,
        action_taken: Optional[str] = None,
        action_was_good: Optional[bool] = None,
    ) -> None:
        """Loga metricas de um frame de combate."""
        self.frame_count += 1
        self.current_metrics["enemies_detected"] = max(
            self.current_metrics["enemies_detected"], enemies_detected
        )
        self.current_metrics["damage_dealt"] += damage_dealt
        self.current_metrics["damage_taken"] += damage_taken
        self.current_metrics["power_cubes_collected"] += power_cubes_collected

        if action_was_good is not None:
            if action_was_good:
                self.current_metrics["good_decisions"] += 1
            else:
                self.current_metrics["bad_decisions"] += 1

    def update_from_gameplay(
        self,
        match_active: bool = True,
        elapsed_seconds: float = 0.0,
        enemies_detected: int = 0,
        action_taken: Optional[str] = None,
        win: bool = False,
        draw: bool = False,
        survival_time: float = 0.0,
    ) -> Optional[Dict]:
        """Atualiza métricas a partir do gameplay em tempo real.

        Chamado pelo StateManager durante o jogo (match_active=True)
        e no final da partida (match_active=False).
        """
        if match_active:
            self.frame_count += 1
            self.current_metrics["enemies_detected"] = max(
                self.current_metrics["enemies_detected"], enemies_detected
            )
            if action_taken:
                self.current_metrics["good_decisions"] += 1
            logger.debug(
                f"[REWARD_BRIDGE] Frame atualizado: enemies={enemies_detected}, "
                f"elapsed={elapsed_seconds:.1f}s"
            )
            return None
        else:
            result = "win" if win else ("draw" if draw else "loss")
            return self.end_match(result, final_position=0, survival_time=survival_time)

    def log_kill(self, count: int = 1) -> None:
        """Registra um kill."""
        self.current_metrics["kills"] += count
        logger.debug(f"[REWARD_BRIDGE] Kill registrado: total={self.current_metrics['kills']}")

    def log_death(self) -> None:
        """Registra uma morte."""
        self.current_metrics["deaths"] += 1

    def log_objective(self) -> None:
        """Registra conclusao de objetivo."""
        self.current_metrics["objectives_completed"] += 1

    def end_match(self, result: str, final_position: int = 0, survival_time: Optional[float] = None) -> Optional[Dict]:
        """Calcula reward final quando match termina."""
        if self.calculator is None or GameMetrics is None:
            logger.warning("[REWARD_BRIDGE] RewardCalculator nao disponivel")
            return None

        if survival_time is None:
            if self.match_start_time:
                survival_time = time.time() - self.match_start_time
            else:
                survival_time = 0.0

        # Calcula accuracy de decisao
        total_decisions = (
            self.current_metrics["good_decisions"] +
            self.current_metrics["bad_decisions"]
        )
        decision_accuracy = (
            self.current_metrics["good_decisions"] / total_decisions
            if total_decisions > 0 else 0.0
        )
        self.current_metrics["decision_accuracy"] = decision_accuracy

        metrics = GameMetrics(
            match_id=f"match_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            kills=self.current_metrics["kills"],
            deaths=self.current_metrics["deaths"],
            damage_dealt=self.current_metrics["damage_dealt"],
            damage_taken=self.current_metrics["damage_taken"],
            survival_time=survival_time,
            final_position=final_position,
            power_cubes_collected=self.current_metrics["power_cubes_collected"],
            power_cubes_denied=self.current_metrics["power_cubes_denied"],
            objectives_completed=self.current_metrics["objectives_completed"],
            enemies_detected=self.current_metrics["enemies_detected"],
            good_decisions=self.current_metrics["good_decisions"],
            bad_decisions=self.current_metrics["bad_decisions"],
            decision_accuracy=decision_accuracy,
        )

        reward = self.calculator.calculate_total_reward(metrics)
        self.last_reward = reward.total_reward

        logger.info(
            f"[REWARD_BRIDGE] Match finalizado: {result} "
            f"| Kills: {metrics.kills} | Deaths: {metrics.deaths} | "
            f"Reward: {reward.total_reward:.1f} "
            f"| Normalized: {reward.normalized_reward:.2f}"
        )

        # Loga no DataCollector se disponivel
        if self.data_collector:
            self.data_collector.log_frame(
                state="end",
                reward=reward.total_reward,
            )

        return {
            "total_reward": reward.total_reward,
            "normalized_reward": reward.normalized_reward,
            "kill_reward": reward.kill_reward,
            "survival_reward": reward.survival_reward,
            "damage_reward": reward.damage_reward,
            "decision_reward": reward.decision_reward,
            "death_penalty": reward.death_penalty,
            "metrics": self.current_metrics,
        }

    def get_session_summary(self) -> Dict:
        """Retorna sumário acumulado da sessão atual (não match)."""
        return {
            "current_metrics": dict(self.current_metrics),
            "frame_count": self.frame_count,
            "match_start_time": self.match_start_time,
            "elapsed": time.time() - self.match_start_time if self.match_start_time else 0.0,
            "last_reward": self.last_reward,
        }

    def reset(self) -> None:
        """Reseta estado completo para nova sessão."""
        self.reset_match_metrics()
        self.match_start_time = None
