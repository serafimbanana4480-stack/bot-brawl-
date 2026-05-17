"""
decision/multi_objective_rl.py

Multi-Objective RL para Soberana Omega.

Otimiza múltiplos objetivos conflitantes simultaneamente:
- Maximizar win rate (objetivo primário)
- Minimizar risco de detecção (anti-ban)
- Maximizar sobrevivência
- Maximizar coleta de recursos
- Minimizar uso de abilities desperdiçado

Cada ação é avaliada por weighted sum de objetivos,
permitindo trade-offs transparentes e ajustáveis.
"""

import random
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    WIN_RATE = "win_rate"
    SURVIVAL = "survival"
    DETECTION_RISK = "detection_risk"
    RESOURCE_COLLECTION = "resource_collection"
    DAMAGE_DEALT = "damage_dealt"
    ABILITY_EFFICIENCY = "ability_efficiency"


@dataclass
class Objective:
    """Um objetivo otimizável com peso dinâmico."""
    name: ObjectiveType
    weight: float
    value_fn: Callable[[Any, Any], float]  # (action, context) -> score [0,1]
    description: str = ""


@dataclass
class ActionEvaluation:
    """Avaliação de uma ação considerando múltiplos objetivos."""
    action: str
    action_id: int
    total_score: float
    objective_scores: Dict[str, float] = field(default_factory=dict)
    is_pareto_optimal: bool = False


class MultiObjectiveOptimizer:
    """
    Seleciona ações considerando múltiplos objetivos simultaneamente.

    Uso:
        moo = MultiObjectiveOptimizer()
        moo.register_objective(Objective(...))

        best_action = moo.select_action(
            valid_actions=["attack", "retreat", "collect"],
            game_state=current_state,
        )
    """

    def __init__(self):
        self.objectives: List[Objective] = []
        self._action_history: List[str] = []
        self._max_history = 100

        # Objetivos padrão
        self._setup_default_objectives()

        logger.info("[MULTI_OBJECTIVE] Inicializado com %d objetivos", len(self.objectives))

    def _setup_default_objectives(self):
        """Configura objetivos padrão do Soberana Omega."""
        self.objectives = [
            Objective(
                name=ObjectiveType.WIN_RATE,
                weight=0.60,
                value_fn=self._estimate_win_probability,
                description="Probabilidade de vitória dado estado e ação",
            ),
            Objective(
                name=ObjectiveType.DETECTION_RISK,
                weight=0.20,
                value_fn=self._estimate_detection_risk,
                description="Risco de detecção anti-ban (menor = melhor)",
            ),
            Objective(
                name=ObjectiveType.SURVIVAL,
                weight=0.10,
                value_fn=self._estimate_survival,
                description="HP do jogador e segurança posicional",
            ),
            Objective(
                name=ObjectiveType.RESOURCE_COLLECTION,
                weight=0.05,
                value_fn=self._estimate_resource_gain,
                description="Power cubes, gems, stars coletáveis",
            ),
            Objective(
                name=ObjectiveType.ABILITY_EFFICIENCY,
                weight=0.05,
                value_fn=self._estimate_ability_efficiency,
                description="Eficiência de uso de super/gadget",
            ),
        ]

    # ------------------------------------------------------------------
    # Value functions (placeholder — integrar com lógica real)
    # ------------------------------------------------------------------

    def _estimate_win_probability(self, action: str, context: Dict[str, Any]) -> float:
        """Estima probabilidade de vitória [0,1]."""
        base = 0.5
        if action == "retreat":
            # Recuar com baixo HP aumenta chance de sobrevivência
            if context.get("player_hp", 1.0) < 0.3:
                base += 0.3
            else:
                base -= 0.1
        elif action == "attack":
            # Atacar com vantagem
            if context.get("enemies_nearby", 0) == 1 and context.get("player_hp", 1.0) > 0.5:
                base += 0.2
            else:
                base -= 0.1
        elif action == "collect_cube":
            # Cubes são importantes no early/mid game
            if context.get("match_time", 0) < 60:
                base += 0.15
        return max(0.0, min(1.0, base))

    def _estimate_detection_risk(self, action: str, context: Dict[str, Any]) -> float:
        """
        Estima risco de detecção [0,1].
        Menor = melhor (o peso na weighted sum é invertido).
        """
        # Ações muito rápidas/repetitivas aumentam risco
        recent_actions = context.get("recent_actions", [])
        if len(recent_actions) >= 3 and all(a == action for a in recent_actions[-3:]):
            return 0.8  # Padrão repetitivo = alto risco

        # APM alto aumenta risco
        if context.get("current_apm", 0) > 50:
            return 0.7

        # Movimentos perfeitos (linha reta) são suspeitos
        if action == "move" and context.get("last_move_angle", 0) == context.get("current_move_angle", 0):
            return 0.6

        return 0.1  # Padrão normal

    def _estimate_survival(self, action: str, context: Dict[str, Any]) -> float:
        """Estima chance de sobrevivência [0,1]."""
        hp = context.get("player_hp", 1.0)
        enemies = context.get("enemies_nearby", 0)
        has_cover = context.get("near_cover", False)

        if action == "retreat":
            if hp < 0.3:
                return 0.9
            return 0.6
        elif action == "attack":
            if enemies >= 2 and hp < 0.5:
                return 0.2
            return 0.5
        elif action == "move_to_cover":
            if not has_cover:
                return 0.8
            return 0.4
        return hp

    def _estimate_resource_gain(self, action: str, context: Dict[str, Any]) -> float:
        """Estima ganho de recursos [0,1]."""
        if action == "collect_cube":
            cubes_nearby = context.get("power_cubes_nearby", 0)
            return min(1.0, cubes_nearby / 5.0)
        elif action == "collect_gem":
            return 0.7 if context.get("gems_nearby", 0) > 0 else 0.0
        return 0.1

    def _estimate_ability_efficiency(self, action: str, context: Dict[str, Any]) -> float:
        """Estima eficiência de habilidades [0,1]."""
        if action == "use_super":
            # Usar super sem alvos é ruim
            targets = context.get("super_targets", 0)
            if targets == 0:
                return 0.0
            return min(1.0, targets / 2.0)
        elif action == "use_gadget":
            # Gadget em cooldown ou sem necessidade
            if context.get("gadget_needed", False):
                return 0.9
            return 0.2
        return 0.5

    # ------------------------------------------------------------------
    # Seleção de ação
    # ------------------------------------------------------------------

    def select_action(
        self,
        valid_actions: List[str],
        context: Dict[str, Any],
        epsilon: float = 0.1,
    ) -> ActionEvaluation:
        """
        Seleciona a melhor ação usando multi-objective optimization.

        Args:
            valid_actions: Lista de ações possíveis
            context: Estado do jogo (HP, inimigos, posição, etc.)
            epsilon: Chance de exploração aleatória
        """
        if random.random() < epsilon:
            action = random.choice(valid_actions)
            return ActionEvaluation(
                action=action,
                action_id=valid_actions.index(action),
                total_score=0.0,
                is_pareto_optimal=False,
            )

        evaluations = []
        for action in valid_actions:
            scores = {}
            weighted_total = 0.0

            for obj in self.objectives:
                raw_score = obj.value_fn(action, context)
                # Para DETECTION_RISK, invertemos (menor risco = maior score)
                if obj.name == ObjectiveType.DETECTION_RISK:
                    raw_score = 1.0 - raw_score
                scores[obj.name.value] = raw_score
                weighted_total += obj.weight * raw_score

            evaluations.append(ActionEvaluation(
                action=action,
                action_id=valid_actions.index(action),
                total_score=weighted_total,
                objective_scores=scores,
            ))

        # Identificar soluções Pareto-ótimas
        pareto = self._find_pareto_front(evaluations)
        for ev in evaluations:
            ev.is_pareto_optimal = ev in pareto

        # Escolher a de maior score total (dentro do Pareto front se possível)
        best = max(pareto if pareto else evaluations, key=lambda x: x.total_score)

        self._action_history.append(best.action)
        if len(self._action_history) > self._max_history:
            self._action_history = self._action_history[-self._max_history:]

        logger.debug(
            "[MULTI_OBJECTIVE] Ação selecionada: %s (score=%.3f, pareto=%s)",
            best.action, best.total_score, best.is_pareto_optimal
        )
        return best

    def _find_pareto_front(self, evaluations: List[ActionEvaluation]) -> List[ActionEvaluation]:
        """Encontra soluções Pareto-ótimas (nenhum objetivo pode melhorar sem piorar outro)."""
        pareto = []
        for ev in evaluations:
            dominated = False
            for other in evaluations:
                if other is ev:
                    continue
                # other domina ev se for >= em todos os objetivos e > em pelo menos um
                if all(
                    other.objective_scores.get(k, 0) >= ev.objective_scores.get(k, 0)
                    for k in ev.objective_scores
                ) and any(
                    other.objective_scores.get(k, 0) > ev.objective_scores.get(k, 0)
                    for k in ev.objective_scores
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(ev)
        return pareto

    # ------------------------------------------------------------------
    # Gestão de objetivos
    # ------------------------------------------------------------------

    def update_objective_weight(self, objective_name: ObjectiveType, new_weight: float):
        """Atualiza peso de um objetivo (para adaptação dinâmica)."""
        for obj in self.objectives:
            if obj.name == objective_name:
                obj.weight = max(0.0, min(1.0, new_weight))
                logger.info("[MULTI_OBJECTIVE] Peso de %s atualizado para %.2f", objective_name.value, obj.weight)
                break
        self._normalize_weights()

    def _normalize_weights(self):
        """Normaliza pesos para somarem 1.0."""
        total = sum(obj.weight for obj in self.objectives)
        if total > 0:
            for obj in self.objectives:
                obj.weight /= total

    def add_objective(self, objective: Objective):
        """Adiciona um novo objetivo."""
        self.objectives.append(objective)
        self._normalize_weights()

    def get_objective_scores(self, action: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Retorna scores individuais por objetivo para uma ação."""
        return {
            obj.name.value: obj.value_fn(action, context)
            for obj in self.objectives
        }

    def get_status(self) -> Dict[str, Any]:
        """Status atual do otimizador."""
        return {
            "objectives": [
                {"name": obj.name.value, "weight": round(obj.weight, 3), "description": obj.description}
                for obj in self.objectives
            ],
            "recent_actions": self._action_history[-10:],
        }
