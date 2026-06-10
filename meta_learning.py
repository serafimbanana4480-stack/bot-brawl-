"""
meta_learning.py

Sistema de Meta-Learning para auto-otimização do bot.

O Meta-Learning ("learning to learn") adapta os hiperparâmetros do bot
baseado no desempenho observado, permitindo que o bot otimize a si mesmo.

Funcionalidades:
- Adaptação de hiperparâmetros (epsilon, learning rate, etc.)
- Seleção automática de estratégias por contexto
- Detecção de meta-padrões (o bot está a melhorar ou a piorar?)
- Ajuste de exploração vs exploitação
- Calibração de confiança
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import threading

logger = logging.getLogger("meta_learning")


class PerformanceTrend(Enum):
    """Tendência de performance."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    UNKNOWN = "unknown"


@dataclass
class Hyperparameters:
    """Hiperparâmetros atuais do sistema."""
    epsilon: float = 0.4
    learning_rate: float = 0.1
    gamma: float = 0.95
    batch_size: int = 32
    exploration_bonus: float = 0.1
    confidence_threshold: float = 0.6
    utility_ai_threshold: float = 0.70

    def to_dict(self) -> dict:
        return asdict(self)


class MetaLearningSystem:
    """
    Sistema de Meta-Learning que otimiza os hiperparâmetros do bot.

    Monitoriza o desempenho e ajusta:
    - Epsilon (exploração vs exploitação)
    - Learning rate do RL
    - Thresholds de confiança
    - Estratégias por contexto
    """

    def __init__(self, save_path: Path = Path("data/meta_learning.json")):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.hyperparams = Hyperparameters()

        # Histórico de métricas
        self.performance_history: List[Dict] = []
        self.max_history = 500

        # Tendência atual
        self.current_trend = PerformanceTrend.UNKNOWN

        # Lock para thread safety
        self._lock = threading.Lock()

        # Contexto atual
        self.current_context: Dict[str, Any] = {}

        # Contadores
        self.total_matches = 0
        self.wins = 0

        # Learning rate adaptation
        self._lr_adjustments = 0

        # Carregar se existir
        self._load()

        logger.info("[META_LEARNING] Sistema inicializado")

    def record_match_result(
        self,
        result: str,  # "win", "loss", "draw"
        brawler: str,
        map_name: str,
        performance_metrics: Optional[Dict] = None
    ):
        """
        Regista o resultado de uma partida para análise.

        Args:
            result: Resultado da partida
            brawler: Brawler usado
            map_name: Mapa jogado
            performance_metrics: Métricas adicionais (kills, deaths, damage, etc.)
        """
        with self._lock:
            self.total_matches += 1
            if result == "win":
                self.wins += 1

            # Calcular win rate
            win_rate = self.wins / max(1, self.total_matches)

            # Guardar no histórico
            entry = {
                "timestamp": time.time(),
                "result": result,
                "brawler": brawler,
                "map": map_name,
                "win_rate": win_rate,
                "total_matches": self.total_matches,
                "metrics": performance_metrics or {}
            }

            self.performance_history.append(entry)
            if len(self.performance_history) > self.max_history:
                self.performance_history = self.performance_history[-self.max_history:]

            # Analisar tendência
            self._analyze_trend()

            # Ajustar hiperparâmetros baseado na tendência
            self._adapt_hyperparameters()

            # Guardar periodicamente
            if self.total_matches % 10 == 0:
                self._save()

            logger.debug(
                f"[META_LEARNING] Partida #{self.total_matches}: {result} "
                f"(WR: {win_rate:.1%}, Trend: {self.current_trend.value})"
            )

    def _analyze_trend(self):
        """Analisa a tendência de performance."""
        if len(self.performance_history) < 10:
            self.current_trend = PerformanceTrend.UNKNOWN
            return

        # Calcular win rate das últimas 10 e 20 partidas
        recent_10 = self.performance_history[-10:]
        recent_20 = self.performance_history[-20:] if len(self.performance_history) >= 20 else self.performance_history

        wr_10 = sum(1 for e in recent_10 if e["result"] == "win") / len(recent_10)
        wr_20 = sum(1 for e in recent_20 if e["result"] == "win") / len(recent_20)

        # Comparar com baseline (win rate global)
        baseline = self.wins / max(1, self.total_matches)

        # Determinar tendência
        if wr_10 > baseline + 0.1:
            if wr_10 > wr_20 + 0.1:
                self.current_trend = PerformanceTrend.IMPROVING
            else:
                self.current_trend = PerformanceTrend.STABLE
        elif wr_10 < baseline - 0.1:
            self.current_trend = PerformanceTrend.DECLINING
        else:
            self.current_trend = PerformanceTrend.STABLE

    def _adapt_hyperparameters(self):
        """Adapta os hiperparâmetros baseado na tendência."""

        if self.current_trend == PerformanceTrend.IMPROVING:
            # Se está a melhorar, reduzir exploração (mais exploitação)
            self.hyperparams.epsilon = max(0.05, self.hyperparams.epsilon * 0.95)
            self.hyperparams.learning_rate = min(0.2, self.hyperparams.learning_rate * 1.05)

        elif self.current_trend == PerformanceTrend.DECLINING:
            # Se está a piorar, aumentar exploração
            self.hyperparams.epsilon = min(0.4, self.hyperparams.epsilon * 1.1)
            self.hyperparams.learning_rate = max(0.01, self.hyperparams.learning_rate * 0.9)
            self._lr_adjustments += 1

        elif self.current_trend == PerformanceTrend.UNKNOWN:
            # Manter valores padrão
            pass

        # Se fez muitas adaptações sem melhorar, pode estar num platô
        if self._lr_adjustments > 10:
            # Tentar mudar estratégia completamente
            self.hyperparams.exploration_bonus *= 1.5
            self._lr_adjustments = 0
            logger.info("[META_LEARNING] Platô detectado, aumentando exploração")

    def get_recommended_epsilon(self, context: Optional[Dict] = None) -> float:
        """
        Retorna epsilon recomendado para o contexto atual.

        Args:
            context: Contexto opcional (brawler, map, etc.)

        Returns:
            Epsilon otimizado
        """
        with self._lock:
            base_epsilon = self.hyperparams.epsilon

            # Ajustar por contexto se fornecido
            if context:
                # Brawler específico
                brawler = context.get("brawler", "")
                if brawler in ["colt", "brock", "piper"]:  # Snipers
                    base_epsilon *= 0.8  # Menos exploração
                elif brawler in ["mortis", "darryl", "leon"]:  # Assassins
                    base_epsilon *= 1.2  # Mais exploração

                # Fase do jogo
                phase = context.get("match_phase", "mid")
                if phase == "early":
                    base_epsilon *= 1.2  # Mais exploração no início
                elif phase == "late":
                    base_epsilon *= 0.9  # Menos exploração no final

            return base_epsilon

    def get_confidence_threshold(self) -> float:
        """Retorna threshold de confiança para decisões."""
        with self._lock:
            # Se está a melhorar, pode ser mais confiante
            if self.current_trend == PerformanceTrend.IMPROVING:
                return min(0.8, self.hyperparams.confidence_threshold * 1.1)
            elif self.current_trend == PerformanceTrend.DECLINING:
                return max(0.4, self.hyperparams.confidence_threshold * 0.9)
            return self.hyperparams.confidence_threshold

    def should_explore_new_strategy(self) -> bool:
        """Decide se deve explorar uma nova estratégia."""
        with self._lock:
            # Se está a declinar, aumentar exploração
            if self.current_trend == PerformanceTrend.DECLINING:
                return True

            # Se está estável há muito tempo, tentar algo novo
            if self.current_trend == PerformanceTrend.STABLE:
                if len(self.performance_history) > 50:
                    recent = self.performance_history[-20:]
                    wr = sum(1 for e in recent if e["result"] == "win") / len(recent)
                    if abs(wr - 0.5) < 0.1:  # Perto de 50%
                        return True  # Platô - tentar algo novo

            return False

    def get_context_strategy(self, context: Dict[str, Any]) -> str:
        """
        Retorna estratégia recomendada para o contexto.

        Args:
            context: Contexto (brawler, map, hp, enemies, etc.)

        Returns:
            Nome da estratégia ("aggressive", "defensive", "balanced")
        """
        brawler = context.get("brawler", "").lower()
        hp = context.get("hp", 1.0)
        enemies_nearby = context.get("enemies_nearby", 0)
        map_type = context.get("map_type", "").lower()

        # Brawler role-based
        if brawler in ["colt", "brock", "piper", "bea"]:  # Snipers
            if enemies_nearby > 2:
                return "defensive"
            return "balanced"
        elif brawler in ["mortis", "darryl", "leon", "crow"]:  # Assassins
            if hp < 0.5:
                return "defensive"
            return "aggressive"
        elif brawler in ["frank", "jacky", "rosa"]:  # Tanks
            return "aggressive"
        elif brawler in ["gene", "poco", "pam"]:  # Support
            return "defensive"

        # Default
        if hp < 0.3:
            return "defensive"
        elif hp > 0.7 and enemies_nearby <= 1:
            return "aggressive"
        return "balanced"

    def get_stats(self) -> Dict:
        """Retorna estatísticas do sistema."""
        with self._lock:
            win_rate = self.wins / max(1, self.total_matches)

            return {
                "total_matches": self.total_matches,
                "wins": self.wins,
                "win_rate": win_rate,
                "trend": self.current_trend.value,
                "hyperparams": self.hyperparams.to_dict(),
                "lr_adjustments": self._lr_adjustments,
            }

    def _save(self):
        """Guarda estado para disco."""
        try:
            data = {
                "hyperparams": self.hyperparams.to_dict(),
                "performance_history": self.performance_history[-100:],  # Só últimos 100
                "total_matches": self.total_matches,
                "wins": self.wins,
                "current_trend": self.current_trend.value,
                "_lr_adjustments": self._lr_adjustments,
            }

            temp_path = self.save_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            temp_path.rename(self.save_path)

            logger.debug(f"[META_LEARNING] Estado guardado em {self.save_path}")

        except Exception as e:
            logger.error(f"[META_LEARNING] Erro ao guardar: {e}")

    def _load(self):
        """Carrega estado do disco."""
        if not self.save_path.exists():
            return

        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.hyperparams = Hyperparameters(**data.get("hyperparams", {}))
            self.performance_history = data.get("performance_history", [])
            self.total_matches = data.get("total_matches", 0)
            self.wins = data.get("wins", 0)
            self.current_trend = PerformanceTrend(data.get("current_trend", "unknown"))
            self._lr_adjustments = data.get("_lr_adjustments", 0)

            logger.info(
                f"[META_LEARNING] Carregado: {self.total_matches} partidas, "
                f"WR: {self.wins/max(1,self.total_matches):.1%}"
            )

        except Exception as e:
            logger.error(f"[META_LEARNING] Erro ao carregar: {e}")