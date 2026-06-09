"""
decision/gradient_boosting_decisions.py

Gradient Boosting para Decisões Combinatórias.

DEPRECATED: Use pylaai_real.play.PlayLogic instead.
This module is kept for backward compatibility only.

Combina múltiplos "weak decision makers" (UtilityAI, MultiObjectiveRL,
BrawlerAdaptive, StickyTarget, etc.) numa decisão final robusta.

Ideia: cada subsistema dá um "voto" com peso. O sistema aprende os pesos
ótimo via gradient boosting (ou online weight adaptation).

Benefício: se um subsistema falha, os outros compensam.
"""

import warnings
import logging
import random
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

warnings.warn(
    "Deprecated: use pylaai_real.play.PlayLogic instead",
    DeprecationWarning,
    stacklevel=2,
)


class DecisionVoter:
    """Um votante que retorna uma ação e confiança."""

    def __init__(self, name: str, weight: float, decide_fn: Callable[[Dict], tuple]):
        self.name = name
        self.weight = weight
        self.decide_fn = decide_fn
        self._correct_predictions = 0
        self._total_predictions = 0

    def predict(self, context: Dict) -> tuple:
        """Retorna (action, confidence)."""
        return self.decide_fn(context)

    def record_outcome(self, predicted_action: str, actual_outcome: str):
        """Registra se a predição foi boa."""
        self._total_predictions += 1
        if predicted_action == actual_outcome:
            self._correct_predictions += 1

    @property
    def accuracy(self) -> float:
        if self._total_predictions == 0:
            return 0.5
        return self._correct_predictions / self._total_predictions


class GradientBoostingDecisionSystem:
    """
    Sistema de decisão que combina múltiplos votantes via boosting.

    Uso:
        gbd = GradientBoostingDecisionSystem()
        gbd.add_voter("utility", 1.0, utility_ai.decide)
        gbd.add_voter("moo", 1.0, moo.select_action)
        gbd.add_voter("brawler", 0.8, brawler_ctrl.get_strategy)

        action = gbd.decide(context)
    """

    def __init__(self, learning_rate: float = 0.1):
        self.voters: List[DecisionVoter] = []
        self.learning_rate = learning_rate
        self._action_history: List[str] = []
        self._max_history = 200

    def add_voter(self, name: str, initial_weight: float, decide_fn: Callable[[Dict], tuple]):
        """Adiciona um novo votante."""
        self.voters.append(DecisionVoter(name, initial_weight, decide_fn))
        logger.info("[GBD] Votante adicionado: %s (weight=%.2f)", name, initial_weight)

    def decide(self, context: Dict[str, Any]) -> str:
        """
        Decide ação combinando todos os votantes.

        Algoritmo:
        1. Cada votante retorna (ação, confiança)
        2. Votos são ponderados pelo weight * accuracy do votante
        3. Ação com maior score ponderado ganha
        """
        if not self.voters:
            return "idle"

        # Coletar predições
        predictions: Dict[str, float] = defaultdict(float)
        voter_results = []

        for voter in self.voters:
            try:
                action, confidence = voter.predict(context)
                effective_weight = voter.weight * (0.5 + 0.5 * voter.accuracy)
                score = confidence * effective_weight
                predictions[action] += score
                voter_results.append({
                    "name": voter.name,
                    "action": action,
                    "confidence": confidence,
                    "weight": voter.weight,
                    "accuracy": voter.accuracy,
                    "effective_score": score,
                })
            except Exception as e:
                logger.debug("[GBD] Votante %s falhou: %s", voter.name, e)

        if not predictions:
            return "idle"

        # Escolher ação com maior score
        best_action = max(predictions, key=predictions.get)
        best_score = predictions[best_action]

        # Logging
        logger.debug(
            "[GBD] Decisão: %s (score=%.2f) | Votos: %s",
            best_action, best_score,
            ", ".join(f"{r['name']}:{r['action']}({r['effective_score']:.2f})" for r in voter_results)
        )

        self._action_history.append(best_action)
        if len(self._action_history) > self._max_history:
            self._action_history = self._action_history[-self._max_history:]

        return best_action

    def update_weights_from_outcome(self, chosen_action: str, outcome: str, reward: float = 0.0):
        """
        Atualiza pesos dos votantes baseado no resultado.

        Se reward > 0: aumentar peso dos votantes que previram a ação correta
        Se reward < 0: diminuir peso dos votantes que previram a ação errada
        """
        for voter in self.voters:
            voter.record_outcome(chosen_action, outcome)

            # Gradient descent simples nos pesos
            if reward > 0:
                voter.weight = min(2.0, voter.weight * (1 + self.learning_rate))
            elif reward < 0:
                voter.weight = max(0.1, voter.weight * (1 - self.learning_rate))

    def get_voter_stats(self) -> List[Dict[str, Any]]:
        """Retorna estatísticas de cada votante."""
        return [
            {
                "name": v.name,
                "weight": round(v.weight, 3),
                "accuracy": round(v.accuracy, 3),
                "predictions": v._total_predictions,
            }
            for v in self.voters
        ]

    def get_status(self) -> Dict[str, Any]:
        return {
            "voter_count": len(self.voters),
            "voters": self.get_voter_stats(),
            "action_diversity": len(set(self._action_history)) if self._action_history else 0,
            "recent_actions": self._action_history[-10:],
        }
