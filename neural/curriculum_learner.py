"""
neural/curriculum_learner.py

Curriculum Learning para Soberana Omega.

Treina o agente contra adversários progressivamente mais fortes:
- Início: bots aleatórios / fáceis (exploração do espaço de estados)
- Meio: bots reativos (aprende a punir erros)
- Fim: bots agressivos / human-like (refina estratégia)

Benefícios:
- Convergência mais rápida
- Menor chance de ficar preso em mínimos locais
- Melhor generalização
- Dificuldade adaptativa automática
"""

import random
import logging
import time
from typing import Dict, Optional, Any, List, Callable
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DifficultyLevel:
    """Um nível de dificuldade no currículo."""
    level: int  # 0-10
    name: str
    bot_behavior: str  # random | reactive | aggressive | human_like
    bot_accuracy: float  # 0-1 (chance de acerto)
    bot_reaction_ms: float  # Tempo de reação do bot
    bot_hp_multiplier: float  # Multiplicador de HP
    bot_damage_multiplier: float  # Multiplicador de dano
    win_rate_target: float  # Win rate esperado para avançar
    min_episodes: int  # Mínimo de episódios neste nível
    description: str = ""


class CurriculumLearner:
    """
    Gerencia curriculum learning com dificuldade adaptativa.

    Uso:
        curriculum = CurriculumLearner()
        for episode in range(10000):
            difficulty = curriculum.get_current_difficulty()
            result = run_episode(difficulty)
            curriculum.record_episode(result)
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        max_difficulty: int = 10,
        win_rate_window: int = 50,
        advancement_threshold: float = 0.70,
        regression_threshold: float = 0.30,
    ):
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.win_rate_window = win_rate_window
        self.advancement_threshold = advancement_threshold
        self.regression_threshold = regression_threshold

        self._win_history: deque = deque(maxlen=win_rate_window)
        self._difficulty_history: List[Dict[str, Any]] = []
        self._episode_count_at_level: Dict[int, int] = {}
        self._total_episodes = 0

        self._levels = self._build_curriculum_levels()

        logger.info("[CURRICULUM] Inicializado (nível=%d/%d)", self.current_difficulty, self.max_difficulty)

    def _build_curriculum_levels(self) -> Dict[int, DifficultyLevel]:
        """Constrói os níveis do currículo."""
        return {
            0: DifficultyLevel(0, "Sandbox", "random", 0.1, 800.0, 0.5, 0.5, 0.80, 10,
                              "Bots aleatórios, quase não atiram. Aprenda a se mover."),
            1: DifficultyLevel(1, "Tutorial", "random", 0.2, 700.0, 0.6, 0.6, 0.75, 15,
                               "Bots aleatórios com movimentação básica."),
            2: DifficultyLevel(2, "Beginner", "reactive", 0.3, 600.0, 0.7, 0.7, 0.70, 20,
                               "Bots reagem quando atacados. Aprenda kiting."),
            3: DifficultyLevel(3, "Easy", "reactive", 0.4, 500.0, 0.8, 0.8, 0.70, 25,
                               "Bots usam cover básico."),
            4: DifficultyLevel(4, "Normal", "aggressive", 0.5, 450.0, 1.0, 1.0, 0.65, 30,
                               "Bots agressivos com acerto médio."),
            5: DifficultyLevel(5, "Intermediate", "aggressive", 0.6, 400.0, 1.0, 1.0, 0.65, 30,
                               "Bots perseguem e usam super."),
            6: DifficultyLevel(6, "Hard", "human_like", 0.7, 350.0, 1.0, 1.1, 0.60, 40,
                               "Bots com delays humanos, previsão básica."),
            7: DifficultyLevel(7, "Expert", "human_like", 0.75, 300.0, 1.0, 1.1, 0.55, 40,
                               "Bots usam gadgets, jogo de equipe."),
            8: DifficultyLevel(8, "Master", "human_like", 0.8, 250.0, 1.0, 1.2, 0.55, 50,
                               "Bots avançados, combos, map awareness."),
            9: DifficultyLevel(9, "Champion", "human_like", 0.85, 200.0, 1.0, 1.2, 0.50, 50,
                               "Quase indistinguível de jogadores reais."),
            10: DifficultyLevel(10, "Legend", "human_like", 0.9, 150.0, 1.1, 1.3, 0.45, 50,
                                "Nível máximo. Apenas os melhores jogadores."),
        }

    # ------------------------------------------------------------------
    # Ciclo de episódio
    # ------------------------------------------------------------------

    def get_current_difficulty(self) -> DifficultyLevel:
        """Retorna configuração de dificuldade atual."""
        return self._levels.get(
            min(self.current_difficulty, self.max_difficulty),
            self._levels[0]
        )

    def record_episode(self, won: bool, metrics: Optional[Dict[str, Any]] = None):
        """
        Registra resultado de um episódio e ajusta dificuldade se necessário.

        Args:
            won: True se ganhou
            metrics: Métricas adicionais (kills, deaths, damage, etc.)
        """
        self._win_history.append(1 if won else 0)
        self._total_episodes += 1
        self._episode_count_at_level[self.current_difficulty] = self._episode_count_at_level.get(self.current_difficulty, 0) + 1

        # Verificar se pode avançar ou regredir
        self._maybe_adjust_difficulty()

        # Registrar histórico
        self._difficulty_history.append({
            "episode": self._total_episodes,
            "difficulty": self.current_difficulty,
            "won": won,
            "timestamp": time.time(),
            "metrics": metrics or {},
        })

    def _maybe_adjust_difficulty(self):
        """Avalia se deve subir, descer ou manter dificuldade."""
        current_level = self._levels.get(self.current_difficulty)
        if not current_level:
            return

        # Precisa de mínimo de episódios no nível atual
        episodes_here = self._episode_count_at_level.get(self.current_difficulty, 0)
        if episodes_here < current_level.min_episodes:
            return

        # Calcular win rate recente
        if len(self._win_history) < 10:
            return

        recent_wr = sum(self._win_history) / len(self._win_history)

        # Subir dificuldade se win rate > threshold
        if recent_wr >= self.advancement_threshold and self.current_difficulty < self.max_difficulty:
            self._advance_difficulty()

        # Descer dificuldade se win rate < threshold (não regredir do nível 0)
        elif recent_wr < self.regression_threshold and self.current_difficulty > 0:
            self._regress_difficulty()

    def _advance_difficulty(self):
        """Aumenta dificuldade."""
        old = self.current_difficulty
        self.current_difficulty = min(self.current_difficulty + 1, self.max_difficulty)
        logger.info("[CURRICULUM] Dificuldade aumentada: %d → %d (%s)",
                    old, self.current_difficulty, self._levels[self.current_difficulty].name)

    def _regress_difficulty(self):
        """Reduz dificuldade."""
        old = self.current_difficulty
        self.current_difficulty = max(self.current_difficulty - 1, 0)
        # Resetar contador de episódios no nível
        self._episode_count_at_level[self.current_difficulty] = 0
        logger.info("[CURRICULUM] Dificuldade reduzida: %d → %d (%s) — win rate muito baixo",
                    old, self.current_difficulty, self._levels[self.current_difficulty].name)

    # ------------------------------------------------------------------
    # API de treinamento
    # ------------------------------------------------------------------

    def get_opponent_config(self) -> Dict[str, Any]:
        """Retorna configuração do oponente simulado para o nível atual."""
        level = self.get_current_difficulty()
        return {
            "behavior": level.bot_behavior,
            "accuracy": level.bot_accuracy,
            "reaction_ms": level.bot_reaction_ms,
            "hp_multiplier": level.bot_hp_multiplier,
            "damage_multiplier": level.bot_damage_multiplier,
        }

    def is_max_difficulty(self) -> bool:
        """Verifica se está no nível máximo."""
        return self.current_difficulty >= self.max_difficulty

    def get_progress(self) -> Dict[str, Any]:
        """Retorna progresso no currículo."""
        recent_wr = sum(self._win_history) / len(self._win_history) if self._win_history else 0.0
        return {
            "current_level": self.current_difficulty,
            "current_name": self.get_current_difficulty().name,
            "total_episodes": self._total_episodes,
            "recent_win_rate": round(recent_wr, 3),
            "episodes_at_current_level": self._episode_count_at_level.get(self.current_difficulty, 0),
            "min_episodes_for_advance": self.get_current_difficulty().min_episodes,
            "is_max_level": self.is_max_difficulty(),
        }

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retorna histórico de mudanças de dificuldade."""
        return self._difficulty_history[-limit:]
