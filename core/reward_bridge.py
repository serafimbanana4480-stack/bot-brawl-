"""
core/reward_bridge.py

Ponte de integracao entre o sistema de rewards e o gameplay.
Conecta StateManager, PlayLogic, MatchController e o RealRewardCalculator.

Funcionalidades:
- Calcula rewards em tempo real durante o gameplay (dense reward shaping)
- Acumula metricas por match
- Emite rewards por frame e por match completo
- Integra com DataCollector para logging
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

__all__ = ["RewardBridge", "RewardShapingConfig"]

logger = logging.getLogger(__name__)

# Lazy import para evitar circular imports
try:
    from training.real_reward_system import GameMetrics, RealRewardCalculator
except ImportError:
    RealRewardCalculator = None
    GameMetrics = None


@dataclass
class RewardShapingConfig:
    """Configuracao de pesos para dense reward shaping."""

    damage_dealt: float = 0.1
    power_cube_collected: float = 0.05
    kill: float = 0.2
    damage_taken: float = -0.1
    win: float = 1.0
    loss: float = -1.0
    timestep_penalty: float = -0.001
    survival_time: float = 0.05  # a cada 5 segundos
    survival_interval: float = 5.0

    # Normalizacao
    reward_clip: float = 1.0


class RewardBridge:
    """Ponte entre gameplay e sistema de rewards com dense shaping."""

    def __init__(self, data_collector=None, config: RewardShapingConfig | None = None):
        self.data_collector = data_collector
        self.calculator = None
        if RealRewardCalculator is not None:
            self.calculator = RealRewardCalculator()

        # Carrega shaping do config.json (campo rl.reward_shaping) com fallback
        self.shaping = self._load_shaping_config()
        if config is not None:
            # Override com dataclass se fornecido
            self.shaping.update({
                "damage_dealt": config.damage_dealt,
                "damage_taken": config.damage_taken,
                "power_cube_collected": config.power_cube_collected,
                "kill": config.kill,
                "win": config.win,
                "loss": config.loss,
                "timestep_penalty": config.timestep_penalty,
                "survival_time": config.survival_time,
            })
        self.config = config or RewardShapingConfig()

        # Metricas acumuladas por match
        self.reset_match_metrics()
        self.match_start_time: float | None = None
        self.frame_count = 0
        self.last_reward: float = 0.0

        # Running statistics for online normalisation
        self._reward_history: list[float] = []
        self._max_history = 1000

        # Frame-level delta tracking (for dense rewards)
        self._prev_metrics: dict | None = None
        self._last_survival_payout = 0.0

        logger.info("[REWARD_BRIDGE] Inicializado (dense shaping ativo)")

    # ------------------------------------------------------------------
    # Public API (mantida para compatibilidade)
    # ------------------------------------------------------------------

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
        self._prev_metrics = None
        self._last_survival_payout = 0.0

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
        action_taken: str | None = None,
        action_was_good: bool | None = None,
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
        action_taken: str | None = None,
        win: bool = False,
        draw: bool = False,
        survival_time: float = 0.0,
    ) -> dict | None:
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

    def end_match(self, result: str, final_position: int = 0, survival_time: float | None = None) -> dict | None:
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

    def get_session_summary(self) -> dict:
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

    # ------------------------------------------------------------------
    # Dense Reward Shaping (novo)
    # ------------------------------------------------------------------

    def _load_shaping_config(self) -> dict[str, float]:
        """Carrega pesos de reward shaping do config.json (campo rl.reward_shaping)."""
        import json
        from pathlib import Path
        try:
            with open(Path(__file__).parent.parent / "config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            raw = cfg.get("rl", {}).get("reward_shaping", {})
            # Mapeia nomes alternativos do JSON para nomes internos
            mapping = {
                "damage_received": "damage_taken",
                "cube_collected": "power_cube_collected",
                "survival_bonus": "survival_time",
            }
            shaped: dict[str, float] = {}
            for k, v in raw.items():
                key = mapping.get(k, k)
                shaped[key] = float(v)
            return shaped
        except Exception:
            return {}

    def reward_damage_dealt(self, damage: float) -> float:
        weight = self.shaping.get("damage_dealt", 0.1)
        return weight * (damage / 1000.0)

    def reward_damage_received(self, damage: float) -> float:
        weight = self.shaping.get("damage_taken", -0.1)
        return weight * (damage / 1000.0)

    def reward_cube_collected(self) -> float:
        return self.shaping.get("power_cube_collected", 0.05)

    def reward_kill(self) -> float:
        return self.shaping.get("kill", 0.2)

    def reward_win(self) -> float:
        return self.shaping.get("win", 1.0)

    def reward_loss(self) -> float:
        return self.shaping.get("loss", -1.0)

    def reward_survival_time(self, seconds: float) -> float:
        weight = self.shaping.get("survival_time", 0.05)
        return weight * (seconds / 5.0)

    def reward_timestep(self) -> float:
        return self.shaping.get("timestep_penalty", -0.001)

    def compute_total_reward(self, events: dict) -> float:
        """Soma rewards por evento e normaliza para [-1, 1]."""
        total = 0.0
        total += self.reward_damage_dealt(events.get("damage_dealt", 0.0))
        total += self.reward_damage_received(events.get("damage_received", 0.0))
        total += self.reward_cube_collected() * events.get("cubes_collected", 0)
        total += self.reward_kill() * events.get("kills", 0)
        total += self.reward_survival_time(events.get("survival_seconds", 0.0))
        total += self.reward_timestep() * events.get("timesteps", 0)
        if events.get("match_done", False):
            total += self.reward_win() if events.get("win", False) else self.reward_loss()
        # Normalizacao suave para [-1, 1]
        normalized = float(np.clip(total / (abs(total) + 1.0), -1.0, 1.0))
        logger.debug(f"[REWARD_SHAPING] raw={total:.4f} norm={normalized:.4f} events={events}")
        return normalized

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_reward(self, reward: float) -> float:
        """Normaliza reward para [-1, 1] usando clipping + running scale."""
        # Clip imediato para evitar outliers extremos
        clipped = np.clip(reward, -self.config.reward_clip, self.config.reward_clip)

        # Se temos histórico suficiente, usamos running std para scale suave
        if len(self._reward_history) >= 10:
            arr = np.array(self._reward_history[-100:], dtype=np.float32)
            mean = float(np.mean(arr))
            std = float(np.std(arr)) + 1e-6
            scaled = (clipped - mean) / (std * 3.0)  # ~3 sigma -> [-1, 1]
            return float(np.clip(scaled, -1.0, 1.0))

        # Fallback: simples clip + tanh-like scaling para intervalo pequeno
        return float(np.clip(reward / (abs(reward) + 1.0), -1.0, 1.0))

    def _update_reward_stats(self, reward: float) -> None:
        """Atualiza running statistics de rewards."""
        self._reward_history.append(reward)
        if len(self._reward_history) > self._max_history:
            self._reward_history.pop(0)
