"""
auto_tuner.py

Auto-tuning de parâmetros baseado em performance histórica.
Analisa match history e ajusta parâmetros automaticamente para melhorar win rate.
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

try:
    from realtime_logs import get_log_manager
    log_manager = get_log_manager()
except (ImportError, Exception):
    log_manager = None
    logger.warning("[AUTO_TUNER] Log manager não disponível")


@dataclass
class TuningConfig:
    """Configuração do auto-tuner"""
    min_matches_for_tuning: int = 10  # Mínimo de partidas para ajustar
    tuning_interval_hours: int = 1  # Intervalo entre tunings
    max_parameter_change_percent: float = 20.0  # Máxima mudança por tuning
    win_rate_target: float = 0.6  # Win rate alvo (60%)
    
    # Limites de parâmetros
    min_attack_distance: int = 100
    max_attack_distance: int = 400
    min_shot_cooldown: float = 0.3
    max_shot_cooldown: float = 0.8
    min_safety_threshold: float = 0.3
    max_safety_threshold: float = 0.8


class AutoTuner:
    """Sistema de auto-tuning de parâmetros"""
    
    def __init__(self, match_controller, config: Optional[TuningConfig] = None):
        self.match_controller = match_controller
        self.config = config or TuningConfig()
        self.last_tuning_time = 0
        self.tuning_history: list = []
        
        # Parâmetros atuais (serão inicializados dos componentes)
        self.current_params = {
            "attack_distance": 200,
            "shot_cooldown": 0.45,
            "safety_threshold": 0.5,
            "aggressiveness": 0.5
        }
        
        logger.info("[AUTO_TUNER] Auto-tuner inicializado")
    
    def should_tune(self) -> bool:
        """Verifica se deve fazer tuning agora"""
        # Verificar intervalo de tempo
        hours_since_last_tuning = (time.time() - self.last_tuning_time) / 3600
        if hours_since_last_tuning < self.config.tuning_interval_hours:
            logger.debug(f"[AUTO_TUNER] Tuning não necessário (último tuning há {hours_since_last_tuning:.1f}h)")
            return False
        
        # Verificar número mínimo de partidas
        stats = self.match_controller.get_stats(last_n=self.config.min_matches_for_tuning)
        if stats["total"] < self.config.min_matches_for_tuning:
            logger.debug(f"[AUTO_TUNER] Partidas insuficientes para tuning ({stats['total']}/{self.config.min_matches_for_tuning})")
            return False
        
        return True
    
    def analyze_performance(self) -> Dict:
        """Analisa performance recente"""
        stats = self.match_controller.get_stats(last_n=50)
        
        analysis = {
            "win_rate": stats["win_rate"] / 100.0,  # Converter para 0-1
            "avg_kills_per_match": stats["total_kills"] / max(1, stats["total"]),
            "avg_damage_per_match": stats["total_damage"] / max(1, stats["total"]),
            "total_matches": stats["total"],
            "performance_rating": self._calculate_performance_rating(stats)
        }
        
        logger.info(f"[AUTO_TUNER] Análise de performance: win_rate={analysis['win_rate']:.2%}, "
                   f"avg_kills={analysis['avg_kills_per_match']:.1f}, rating={analysis['performance_rating']:.2f}")
        
        return analysis
    
    def _calculate_performance_rating(self, stats: Dict) -> float:
        """Calcula rating de performance (0-1)"""
        if stats["total"] == 0:
            return 0.5
        
        # Win rate (peso 0.5)
        win_rate = stats["win_rate"] / 100.0
        win_score = min(1.0, win_rate / self.config.win_rate_target)
        
        # Kills por partida (peso 0.3)
        avg_kills = stats["total_kills"] / stats["total"]
        kill_score = min(1.0, avg_kills / 3.0)  # 3 kills/partida é bom
        
        # Troféus ganhos (peso 0.2)
        avg_trophies = stats["total_trophies"] / stats["total"]
        trophy_score = min(1.0, max(0, avg_trophies / 20.0))  # 20 troféus/partida é bom
        
        rating = 0.5 * win_score + 0.3 * kill_score + 0.2 * trophy_score
        return rating
    
    def calculate_adjustments(self, analysis: Dict) -> Dict:
        """Calcula ajustes de parâmetros baseados na análise"""
        adjustments = {}
        
        # Se performance ruim, ajustar parâmetros
        if analysis["performance_rating"] < 0.5:
            logger.info("[AUTO_TUNER] Performance abaixo do alvo, ajustando parâmetros")
            
            # Win rate baixo: aumentar distância de ataque (mais seguro)
            if analysis["win_rate"] < self.config.win_rate_target:
                adjustment_percent = (self.config.win_rate_target - analysis["win_rate"]) * 50
                adjustment_percent = min(adjustment_percent, self.config.max_parameter_change_percent)
                
                adjustments["attack_distance"] = adjustment_percent  # Aumentar distância
                adjustments["shot_cooldown"] = -adjustment_percent * 0.5  # Diminuir cooldown (mais rápido)
                adjustments["safety_threshold"] = adjustment_percent * 0.5  # Aumentar segurança
                adjustments["aggressiveness"] = -adjustment_percent  # Menos agressivo
                
            # Kills baixos: aumentar agressividade
            if analysis["avg_kills_per_match"] < 1.5:
                adjustments["aggressiveness"] = adjustments.get("aggressiveness", 0) + 10  # Mais agressivo
                adjustments["shot_cooldown"] = adjustments.get("shot_cooldown", 0) - 5  # Mais rápido
        
        # Se performance muito boa, pode arriscar mais
        elif analysis["performance_rating"] > 0.7:
            logger.info("[AUTO_TUNER] Performance excelente, otimizando para mais agressividade")
            
            adjustments["attack_distance"] = -10  # Diminuir distância (mais perto)
            adjustments["shot_cooldown"] = -5  # Mais rápido
            adjustments["safety_threshold"] = -5  # Menos conservador
            adjustments["aggressiveness"] = 10  # Mais agressivo
        
        else:
            logger.info("[AUTO_TUNER] Performance adequada, ajustes mínimos")
            adjustments = {}
        
        logger.info(f"[AUTO_TUNER] Ajustes calculados: {adjustments}")
        return adjustments
    
    def apply_adjustments(self, adjustments: Dict, play_logic, safety_system) -> bool:
        """Aplica ajustes aos componentes"""
        try:
            # Aplicar ajustes ao play_logic
            if hasattr(play_logic, 'attack_distance'):
                old_value = getattr(play_logic, 'attack_distance', 200)
                if "attack_distance" in adjustments:
                    new_value = old_value * (1 + adjustments["attack_distance"] / 100)
                    new_value = max(self.config.min_attack_distance, 
                                  min(self.config.max_attack_distance, new_value))
                    setattr(play_logic, 'attack_distance', int(new_value))
                    logger.info(f"[AUTO_TUNER] attack_distance: {old_value} -> {new_value}")
                    self.current_params["attack_distance"] = new_value
            
            if hasattr(play_logic, 'shot_cooldown'):
                old_value = getattr(play_logic, 'shot_cooldown', 0.45)
                if "shot_cooldown" in adjustments:
                    new_value = old_value * (1 + adjustments["shot_cooldown"] / 100)
                    new_value = max(self.config.min_shot_cooldown,
                                  min(self.config.max_shot_cooldown, new_value))
                    setattr(play_logic, 'shot_cooldown', new_value)
                    logger.info(f"[AUTO_TUNER] shot_cooldown: {old_value} -> {new_value}")
                    self.current_params["shot_cooldown"] = new_value
            
            # Aplicar ajustes ao safety_system.
            # Suporta tanto a configuração completa quanto mocks simples usados nos testes.
            if "safety_threshold" in adjustments:
                old_value = None
                if hasattr(safety_system, "suspicion_threshold"):
                    old_value = getattr(safety_system, "suspicion_threshold")
                elif hasattr(safety_system, "config") and hasattr(safety_system.config, "suspicious_pattern_threshold"):
                    old_value = safety_system.config.suspicious_pattern_threshold

                if old_value is not None:
                    new_value = old_value * (1 + adjustments["safety_threshold"] / 100)
                    new_value = max(1, min(20, new_value))  # Limites razoáveis

                    if hasattr(safety_system, "suspicion_threshold"):
                        setattr(safety_system, "suspicion_threshold", int(new_value))
                    elif hasattr(safety_system, "config") and hasattr(safety_system.config, "suspicious_pattern_threshold"):
                        safety_system.config.suspicious_pattern_threshold = int(new_value)

                    logger.info(f"[AUTO_TUNER] safety_threshold: {old_value} -> {new_value}")
                    self.current_params["safety_threshold"] = new_value
            
            # Ajustar agressividade (parâmetro genérico)
            if "aggressiveness" in adjustments:
                old_value = self.current_params["aggressiveness"]
                new_value = old_value + adjustments["aggressiveness"] / 100
                new_value = max(0.0, min(1.0, new_value))
                self.current_params["aggressiveness"] = new_value
                logger.info(f"[AUTO_TUNER] aggressiveness: {old_value} -> {new_value}")
            
            # Registrar no histórico
            self.tuning_history.append({
                "timestamp": time.time(),
                "adjustments": adjustments,
                "params_after": self.current_params.copy()
            })
            
            self.last_tuning_time = time.time()
            logger.info("[AUTO_TUNER] Ajustes aplicados com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"[AUTO_TUNER] Erro ao aplicar ajustes: {e}")
            return False
    
    def tune(self, play_logic, safety_system) -> Dict:
        """Executa ciclo completo de tuning"""
        logger.info("[AUTO_TUNER] Iniciando ciclo de auto-tuning")
        if log_manager:
            log_manager.log(
                message="Iniciando ciclo de auto-tuning",
                level="INFO",
                category="auto_tuning",
                data={"action": "tune_start"}
            )
        if not self.should_tune():
            return {
                "success": False,
                "reason": "Tuning não necessário neste momento",
                "current_params": self.current_params
            }
        
        # Analisar performance
        analysis = self.analyze_performance()
        
        # Calcular ajustes
        adjustments = self.calculate_adjustments(analysis)
        
        if not adjustments:
            return {
                "success": False,
                "reason": "Nenhum ajuste necessário",
                "analysis": analysis,
                "current_params": self.current_params
            }
        
        # Aplicar ajustes
        success = self.apply_adjustments(adjustments, play_logic, safety_system)

        if log_manager:
            log_manager.log(
                message=f"Ciclo de auto-tuning concluído: success={success}",
                level="INFO",
                category="auto_tuning",
                data={"success": success, "adjustments": adjustments, "analysis": analysis}
            )

        return {
            "success": success,
            "analysis": analysis,
            "adjustments": adjustments,
            "current_params": self.current_params,
            "tuning_history_count": len(self.tuning_history)
        }
    
    def get_tuning_status(self) -> Dict:
        """Retorna status atual do tuning"""
        return {
            "last_tuning_time": self.last_tuning_time,
            "last_tuning_hours_ago": (time.time() - self.last_tuning_time) / 3600,
            "tuning_history_count": len(self.tuning_history),
            "current_params": self.current_params,
            "config": {
                "min_matches_for_tuning": self.config.min_matches_for_tuning,
                "tuning_interval_hours": self.config.tuning_interval_hours,
                "win_rate_target": self.config.win_rate_target
            }
        }
    
    def reset_params(self, play_logic, safety_system) -> bool:
        """Reseta parâmetros para valores padrão"""
        try:
            defaults = {
                "attack_distance": 200,
                "shot_cooldown": 0.45,
                "safety_threshold": 0.5,
                "aggressiveness": 0.5
            }
            
            if hasattr(play_logic, 'attack_distance'):
                setattr(play_logic, 'attack_distance', defaults["attack_distance"])
            if hasattr(play_logic, 'shot_cooldown'):
                setattr(play_logic, 'shot_cooldown', defaults["shot_cooldown"])
            
            # Resetar safety_threshold na config do safety_system ou mock equivalente
            if hasattr(safety_system, "suspicion_threshold"):
                safety_system.suspicion_threshold = 5
            elif hasattr(safety_system, 'config'):
                if hasattr(safety_system.config, 'suspicious_pattern_threshold'):
                    safety_system.config.suspicious_pattern_threshold = 5  # Valor padrão
            
            self.current_params = defaults.copy()
            self.tuning_history = []
            self.last_tuning_time = 0
            
            logger.info("[AUTO_TUNER] Parâmetros resetados para valores padrão")
            return True
            
        except Exception as e:
            logger.error(f"[AUTO_TUNER] Erro ao resetar parâmetros: {e}")
            return False
