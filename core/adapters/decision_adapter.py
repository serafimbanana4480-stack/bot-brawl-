"""
core/adapters/decision_adapter.py

Adapter: RLBridge / PlayLogic -> DecisionPort
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from core.ports.decision_port import Decision, DecisionContext, DecisionPort

logger = logging.getLogger(__name__)


class DecisionAdapter(DecisionPort):
    """Wraps RLBridge (NeuralPolicy + PPO + Q-Learning) to satisfy DecisionPort."""

    def __init__(self, rl_bridge: Any = None, play_logic: Any = None):
        self._rl = rl_bridge
        self._play = play_logic
        self._initialized = False

    def initialize(self) -> bool:
        if self._rl is None:
            try:
                from neural.rl_bridge import RLBridge
                self._rl = RLBridge(use_neural=True, q_learning_fallback=True)
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"[DECISION_ADAPTER] RLBridge init failed: {e}")
                return False
        self._initialized = True
        return True

    def decide(self, context: DecisionContext) -> Decision:
        if not self._initialized or self._rl is None:
            return Decision(action_type="idle", confidence=0.0, reasoning="Decision system unavailable")

        try:
            # Build discrete state for RL
            from pylaai_real.rl_engine import CombatQLearning
            state = CombatQLearning.discretize_state(
                player_hp_pct=context.player_hp,
                num_enemies=len(context.enemies),
                nearest_enemy_dist=self._nearest_enemy_dist(context),
                can_attack=context.can_attack,
                can_super=context.can_super,
            )

            # Get action from RL bridge
            action_name, confidence = self._rl.get_action(
                state,
                player_pos=context.player_pos,
                enemies=context.enemies,
            )

            # Map to Decision
            decision = Decision(
                action_type=action_name,
                confidence=confidence,
                reasoning=f"RL decision: state={state}, action={action_name}",
            )

            # Determine target position if moving/attacking
            if action_name in ("move_to_enemy", "attack") and context.enemies:
                decision.target_pos = self._select_target(context)
            elif action_name == "retreat":
                decision.target_pos = self._retreat_direction(context)

            return decision

        except (ImportError, ModuleNotFoundError) as e:
            logger.error(f"[DECISION_ADAPTER] Decide error: {e}")
            return Decision(action_type="idle", confidence=0.0, reasoning=f"Error: {e}")

    def learn(self, context: DecisionContext, decision: Decision, reward: float) -> None:
        if self._rl is None:
            return
        try:
            from pylaai_real.rl_engine import CombatQLearning
            state = CombatQLearning.discretize_state(
                player_hp_pct=context.player_hp,
                num_enemies=len(context.enemies),
                nearest_enemy_dist=self._nearest_enemy_dist(context),
                can_attack=context.can_attack,
                can_super=context.can_super,
            )
            next_state = state  # Simplified; should re-perceive
            self._rl.learn_from_frame(state, decision.action_type, reward, next_state)
        except (ImportError, ModuleNotFoundError) as e:
            logger.debug(f"[DECISION_ADAPTER] Learn error: {e}")

    def start_episode(self, brawler: str, map_name: Optional[str] = None) -> None:
        if self._rl is not None:
            try:
                self._rl.start_episode()
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.debug(f"[DECISION_ADAPTER] Start episode error: {e}")

    def end_episode(self, result: str, rank: int = 0) -> None:
        if self._rl is not None:
            try:
                result_reward = {"win": 10.0, "draw": 2.0, "loss": -5.0}.get(result, 0.0)
                self._rl.end_episode(result_reward)
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.debug(f"[DECISION_ADAPTER] End episode error: {e}")

    def health_check(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "rl_available": self._rl is not None,
            "stats": self._rl.get_stats() if self._rl else {},
        }

    def shutdown(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest_enemy_dist(context: DecisionContext) -> float:
        if not context.enemies:
            return 999.0
        px, py = context.player_pos
        import math
        return min(
            math.hypot(e.get("x", 0.5) - px, e.get("y", 0.5) - py)
            for e in context.enemies
        )

    @staticmethod
    def _select_target(context: DecisionContext) -> tuple:
        if not context.enemies:
            return context.player_pos
        # Target weakest/lowest HP enemy
        weakest = min(context.enemies, key=lambda e: e.get("hp_ratio", 1.0))
        return (weakest.get("x", 0.5), weakest.get("y", 0.5))

    @staticmethod
    def _retreat_direction(context: DecisionContext) -> tuple:
        px, py = context.player_pos
        if not context.enemies:
            return (0.5, 0.5)
        # Run away from nearest enemy
        nearest = min(context.enemies, key=lambda e: math.hypot(e.get("x", 0.5) - px, e.get("y", 0.5) - py))
        ex, ey = nearest.get("x", 0.5), nearest.get("y", 0.5)
        # Opposite direction
        rx = max(0.1, min(0.9, px + (px - ex) * 0.3))
        ry = max(0.1, min(0.9, py + (py - ey) * 0.3))
        return (rx, ry)
