"""
real_reward_system.py

Sistema de rewards real baseado em performance de gameplay.
Calcula rewards baseados em métricas reais do jogo (wins, kills, sobrevivência, etc.).

Funcionalidades:
- Cálculo de rewards baseados em performance
- Tracking de métricas de gameplay
- Sistema de pontuação composto
- Histórico de performance
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GameMetrics:
    """Métricas de uma partida"""
    match_id: str
    timestamp: str
    
    # Combat metrics
    kills: int = 0
    deaths: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    
    # Survival metrics
    survival_time: float = 0.0  # seconds
    final_position: int = 0  # 1-10 (1 = winner)
    
    # Resource metrics
    power_cubes_collected: int = 0
    power_cubes_denied: int = 0
    
    # Objective metrics
    objectives_completed: int = 0
    team_contributions: float = 0.0
    
    # Vision metrics
    enemies_detected: int = 0
    detection_accuracy: float = 0.0
    
    # Decision metrics
    good_decisions: int = 0
    bad_decisions: int = 0
    decision_accuracy: float = 0.0


@dataclass
class RewardComponents:
    """Componentes do reward"""
    timestamp: str
    
    # Combat rewards
    kill_reward: float = 0.0
    survival_reward: float = 0.0
    damage_reward: float = 0.0
    
    # Objective rewards
    objective_reward: float = 0.0
    resource_reward: float = 0.0
    
    # Vision rewards
    detection_reward: float = 0.0
    
    # Decision rewards
    decision_reward: float = 0.0
    
    # Penalties
    death_penalty: float = 0.0
    bad_decision_penalty: float = 0.0
    
    # Total reward
    total_reward: float = 0.0
    
    # Normalized reward (0-1)
    normalized_reward: float = 0.0


class RealRewardCalculator:
    """Calculadora de rewards real baseada em métricas de gameplay"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.reward_history: List[RewardComponents] = []
        
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Carrega configuração de pesos de reward"""
        default_config = {
            "kill_weight": 10.0,
            "death_weight": -5.0,
            "survival_weight": 0.1,  # por segundo
            "damage_weight": 0.01,  # por ponto de dano
            "objective_weight": 5.0,
            "resource_weight": 1.0,
            "detection_weight": 2.0,
            "decision_weight": 3.0,
            "bad_decision_penalty": -2.0,
            "win_bonus": 50.0,
            "top_3_bonus": 20.0
        }
        
        if config_path and config_path.exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def calculate_combat_reward(self, metrics: GameMetrics) -> float:
        """Calcula reward de combate"""
        kill_reward = metrics.kills * self.config["kill_weight"]
        death_penalty = metrics.deaths * self.config["death_weight"]
        damage_reward = metrics.damage_dealt * self.config["damage_weight"]
        
        return kill_reward + death_penalty + damage_reward
    
    def calculate_survival_reward(self, metrics: GameMetrics) -> float:
        """Calcula reward de sobrevivência"""
        survival_reward = metrics.survival_time * self.config["survival_weight"]
        
        # Bonus por posição final
        if metrics.final_position == 1:
            survival_reward += self.config["win_bonus"]
        elif metrics.final_position <= 3:
            survival_reward += self.config["top_3_bonus"]
        
        return survival_reward
    
    def calculate_objective_reward(self, metrics: GameMetrics) -> float:
        """Calcula reward de objetivos"""
        objective_reward = metrics.objectives_completed * self.config["objective_weight"]
        resource_reward = (metrics.power_cubes_collected + metrics.power_cubes_denied) * self.config["resource_weight"]
        
        return objective_reward + resource_reward
    
    def calculate_vision_reward(self, metrics: GameMetrics) -> float:
        """Calcula reward de visão"""
        detection_reward = metrics.enemies_detected * self.config["detection_weight"]
        accuracy_bonus = metrics.detection_accuracy * 10.0  # bonus por alta precisão
        
        return detection_reward + accuracy_bonus
    
    def calculate_decision_reward(self, metrics: GameMetrics) -> float:
        """Calcula reward de decisão"""
        decision_reward = metrics.good_decisions * self.config["decision_weight"]
        bad_decision_penalty = metrics.bad_decisions * self.config["bad_decision_penalty"]
        accuracy_bonus = metrics.decision_accuracy * 5.0  # bonus por alta precisão
        
        return decision_reward + bad_decision_penalty + accuracy_bonus
    
    def calculate_total_reward(self, metrics: GameMetrics) -> RewardComponents:
        """Calcula reward total para uma partida"""
        timestamp = datetime.now().isoformat()
        
        # Calcular componentes individuais (sem dupla contagem)
        # kill_reward = kills * kill_weight (positivo)
        kill_reward = metrics.kills * self.config["kill_weight"]
        # damage_reward = damage_dealt * damage_weight
        damage_reward = metrics.damage_dealt * self.config["damage_weight"]
        # survival_reward inclui bônus de posição final
        survival_reward = self.calculate_survival_reward(metrics)
        # objective_reward = objectives_completed * weight (sem resource — separado abaixo)
        objective_reward = metrics.objectives_completed * self.config["objective_weight"]
        # resource_reward = power cubes
        resource_reward = (metrics.power_cubes_collected + metrics.power_cubes_denied) * self.config["resource_weight"]
        # detection_reward
        detection_reward = self.calculate_vision_reward(metrics)
        # decision_reward = good_decisions * weight + accuracy bonus (sem penalidade — separada abaixo)
        decision_reward = (
            metrics.good_decisions * self.config["decision_weight"]
            + metrics.decision_accuracy * 5.0
        )
        # Penalties (positivos para facilitar leitura; subtraídos no total)
        death_penalty = metrics.deaths * abs(self.config["death_weight"])
        bad_decision_penalty = metrics.bad_decisions * abs(self.config["bad_decision_penalty"])
        
        # Reward total — sem dupla contagem
        total_reward = (
            kill_reward + damage_reward + survival_reward +
            objective_reward + resource_reward + detection_reward +
            decision_reward - death_penalty - bad_decision_penalty
        )
        
        # Normalizar reward (0-1) dinamicamente a partir do histórico de rewards
        # Estima o máximo como o melhor total já visto (ou um patamar mínimo razoável)
        if self.reward_history:
            historical_max = max(r.total_reward for r in self.reward_history)
            normalization_max = max(historical_max * 1.1, 200.0)
        else:
            normalization_max = 200.0
        normalized_reward = max(0.0, min(1.0, total_reward / normalization_max))
        
        reward_components = RewardComponents(
            timestamp=timestamp,
            kill_reward=kill_reward,
            survival_reward=survival_reward,
            damage_reward=damage_reward,
            objective_reward=objective_reward,
            resource_reward=resource_reward,
            detection_reward=detection_reward,
            decision_reward=decision_reward,
            death_penalty=death_penalty,
            bad_decision_penalty=bad_decision_penalty,
            total_reward=total_reward,
            normalized_reward=normalized_reward
        )
        
        self.reward_history.append(reward_components)
        
        return reward_components
    
    def add_reward_score(self, score: float):
        """Adiciona uma pontuação de reward diretamente ao histórico.
        
        Útil para registrar rewards de fontes externas sem criar GameMetrics completas.
        
        Args:
            score: Valor numérico do reward a ser registrado.
        """
        timestamp = datetime.now().isoformat()
        # Estima a normalization_max da mesma forma que calculate_total_reward
        if self.reward_history:
            historical_max = max(r.total_reward for r in self.reward_history)
            normalization_max = max(historical_max * 1.1, 200.0)
        else:
            normalization_max = max(abs(score) * 1.1, 200.0)
        normalized = max(0.0, min(1.0, score / normalization_max))
        self.reward_history.append(
            RewardComponents(
                timestamp=timestamp,
                total_reward=score,
                normalized_reward=normalized
            )
        )

    def get_average_reward(self, last_n: Optional[int] = None) -> float:
        """Calcula reward médio"""
        if not self.reward_history:
            return 0.0
        
        history = self.reward_history[-last_n:] if last_n else self.reward_history
        return np.mean([r.total_reward for r in history])
    
    def get_reward_trend(self) -> str:
        """Analisa tendência de rewards"""
        if len(self.reward_history) < 10:
            return "insufficient_data"
        
        try:
            recent = np.mean([r.total_reward for r in self.reward_history[-10:]])
            older = np.mean([r.total_reward for r in self.reward_history[-20:-10]])
            
            if recent > older * 1.1:
                return "improving"
            elif recent < older * 0.9:
                return "declining"
            else:
                return "stable"
        except (ValueError, RuntimeWarning):
            return "insufficient_data"
    
    def save_reward_history(self, output_path: Path):
        """Salva histórico de rewards"""
        history_data = [asdict(r) for r in self.reward_history]
        
        with open(output_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Histórico de rewards salvo: {output_path}")
    
    def load_reward_history(self, input_path: Path):
        """Carrega histórico de rewards"""
        if not input_path.exists():
            logger.warning(f"Arquivo de histórico não encontrado: {input_path}")
            return
        
        with open(input_path) as f:
            history_data = json.load(f)
        
        self.reward_history = [RewardComponents(**r) for r in history_data]
        logger.info(f"Histórico de rewards carregado: {len(self.reward_history)} entradas")


def create_sample_metrics() -> GameMetrics:
    """Cria métricas de exemplo para teste"""
    return GameMetrics(
        match_id="test_match_001",
        timestamp=datetime.now().isoformat(),
        kills=random.randint(0, 10),
        deaths=random.randint(0, 5),
        damage_dealt=random.uniform(0, 5000),
        damage_taken=random.uniform(0, 3000),
        survival_time=random.uniform(30, 180),
        final_position=random.randint(1, 10),
        power_cubes_collected=random.randint(0, 20),
        power_cubes_denied=random.randint(0, 10),
        objectives_completed=random.randint(0, 5),
        team_contributions=random.uniform(0, 100),
        enemies_detected=random.randint(0, 30),
        detection_accuracy=random.uniform(0.5, 1.0),
        good_decisions=random.randint(0, 50),
        bad_decisions=random.randint(0, 20),
        decision_accuracy=random.uniform(0.5, 1.0)
    )


def main():
    """Função principal para teste"""
    import random
    
    # Criar calculadora de rewards
    calculator = RealRewardCalculator()
    
    # Simular algumas partidas
    logger.info("Simulando partidas para teste do sistema de rewards...")
    
    for i in range(10):
        metrics = create_sample_metrics()
        reward = calculator.calculate_total_reward(metrics)
        
        logger.info(f"Partida {i+1}: Reward={reward.total_reward:.2f}, Normalizado={reward.normalized_reward:.2f}")
    
    # Analisar tendência
    avg_reward = calculator.get_average_reward()
    trend = calculator.get_reward_trend()
    
    logger.info(f"\nReward médio: {avg_reward:.2f}")
    logger.info(f"Tendência: {trend}")
    
    # Salvar histórico
    output_path = Path("./training_reports/reward_history.json")
    output_path.parent.mkdir(exist_ok=True)
    calculator.save_reward_history(output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()