"""
decision/combat_decision_bridge.py

Bridge que conecta MultiObjectiveOptimizer ao PlayLogic de combate.

O PlayLogic decide ações de combate (atacar, recuar, coletar, etc.).
Este bridge permite que o MultiObjectiveOptimizer influence ou substitua
essas decisões com base em múltiplos objetivos simultâneos.

Uso:
    bridge = CombatDecisionBridge(play_logic_instance)
    action = bridge.decide_combat_action(
        valid_actions=["attack", "retreat", "collect_cube"],
        game_context={"player_hp": 0.6, "enemies_nearby": 2, "current_apm": 35},
    )
"""

import logging
import random
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class CombatDecisionBridge:
    """
    Bridge entre PlayLogic e MultiObjectiveOptimizer.

    Combina a lógica existente do PlayLogic com a otimização multi-objetivo
    para decisões de combate mais robustas e anti-detecção.
    """

    def __init__(self, play_logic_instance=None, epsilon: float = 0.1):
        self.play_logic = play_logic_instance
        self.epsilon = epsilon
        self._moo: Optional[Any] = None
        self._action_history: List[str] = []
        self._max_history = 50

        try:
            from decision.multi_objective_rl import MultiObjectiveOptimizer
            self._moo = MultiObjectiveOptimizer()
            logger.info("[COMBAT_BRIDGE] MultiObjectiveOptimizer conectado")
        except Exception as e:
            logger.warning("[COMBAT_BRIDGE] MultiObjectiveOptimizer indisponível: %s", e)

    def decide_combat_action(
        self,
        valid_actions: List[str],
        game_context: Dict[str, Any],
        play_logic_recommendation: Optional[str] = None,
    ) -> str:
        """
        Decide ação de combate combinando PlayLogic + MultiObjective RL.

        Estratégia:
        1. Se MOO disponível, usa-o como decisão primária
        2. Se não disponível, usa recomendação do PlayLogic
        3. Se nenhum disponível, ação aleatória dos válidos
        """
        if self._moo and random.random() > 0.3:  # 70% confiança no MOO
            try:
                result = self._moo.select_action(
                    valid_actions=valid_actions,
                    context=game_context,
                    epsilon=self.epsilon,
                )
                action = result.action

                # Log da decisão
                logger.debug(
                    "[COMBAT_BRIDGE] MOO escolheu: %s (score=%.3f, pareto=%s)",
                    action, result.total_score, result.is_pareto_optimal
                )

                # Se o PlayLogic recomendou algo MUITO diferente, log warning
                if play_logic_recommendation and play_logic_recommendation != action:
                    logger.debug(
                        "[COMBAT_BRIDGE] Divergência: PlayLogic=%s vs MOO=%s",
                        play_logic_recommendation, action
                    )

                self._action_history.append(action)
                if len(self._action_history) > self._max_history:
                    self._action_history = self._action_history[-self._max_history:]

                return action

            except Exception as e:
                logger.warning("[COMBAT_BRIDGE] MOO falhou: %s", e)

        # Fallback para PlayLogic
        if play_logic_recommendation and play_logic_recommendation in valid_actions:
            logger.debug("[COMBAT_BRIDGE] Usando recomendação do PlayLogic: %s", play_logic_recommendation)
            return play_logic_recommendation

        # Fallback aleatório
        action = random.choice(valid_actions)
        logger.debug("[COMBAT_BRIDGE] Fallback aleatório: %s", action)
        return action

    def get_action_scores(self, action: str, game_context: Dict[str, Any]) -> Dict[str, float]:
        """Retorna scores individuais por objetivo para uma ação."""
        if self._moo:
            return self._moo.get_objective_scores(action, game_context)
        return {}

    def update_weights(self, objective_name: str, new_weight: float):
        """Atualiza peso de um objetivo (ex: aumentar survival se morrendo muito)."""
        if self._moo:
            from decision.multi_objective_rl import ObjectiveType
            try:
                obj_type = ObjectiveType(objective_name)
                self._moo.update_objective_weight(obj_type, new_weight)
                logger.info("[COMBAT_BRIDGE] Peso de %s atualizado para %.2f", objective_name, new_weight)
            except ValueError:
                logger.warning("[COMBAT_BRIDGE] Objetivo desconhecido: %s", objective_name)

    def get_status(self) -> Dict[str, Any]:
        """Status do bridge para dashboard."""
        return {
            "moo_available": self._moo is not None,
            "epsilon": self.epsilon,
            "recent_actions": self._action_history[-10:],
            "action_diversity": len(set(self._action_history)) if self._action_history else 0,
        }
