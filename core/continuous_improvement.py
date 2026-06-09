"""
continuous_improvement.py

Sistema de Melhoria Contínua para o Bot Brawl Stars.
Monitora métricas em tempo real e ajusta parâmetros automaticamente.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BotMetrics:
    """Métricas do bot em uma sessão."""
    session_start: float = 0.0
    session_duration: float = 0.0
    
    # Transições de estado
    lobby_to_game_time: float = 0.0
    state_transitions: Dict[str, int] = None
    
    # Detecção
    detection_success_rate: float = 0.0
    detection_total: int = 0
    detection_success: int = 0
    
    # Combate
    actions_per_minute: float = 0.0
    total_combat_actions: int = 0
    combat_start_time: float = 0.0
    
    # Popups
    popups_closed: int = 0
    popup_failures: int = 0
    
    # Erros
    errors_count: Dict[str, int] = None
    
    # Performance
    avg_cycle_time: float = 0.0
    cycle_times: list = None
    
    def __post_init__(self):
        if self.state_transitions is None:
            self.state_transitions = {}
        if self.errors_count is None:
            self.errors_count = {}
        if self.cycle_times is None:
            self.cycle_times = []


class ContinuousImprovement:
    """
    Monitora métricas do bot e ajusta parâmetros automaticamente.
    
    Thresholds:
    - lobby_to_game < 10s (bom), > 20s (ruim)
    - actions_per_minute > 5 (bom), < 2 (ruim)
    - detection_rate > 80% (bom), < 50% (ruim)
    """
    
    THRESHOLDS = {
        'lobby_to_game_max': 10.0,
        'lobby_to_game_warning': 20.0,
        'actions_per_min_min': 5.0,
        'actions_per_min_warning': 2.0,
        'detection_rate_min': 0.8,
        'detection_rate_warning': 0.5,
        'cycle_time_max': 2.0,
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.metrics = BotMetrics(session_start=time.time())
        self.config_path = Path(config_path) if config_path else Path("bot_metrics.json")
        self.report_path = Path("improvement_report.json")
        self._last_save = 0.0
        self._improvements_applied = []
        
        logger.info("[IMPROVEMENT] Sistema de melhoria contínua iniciado")
    
    def record_state_transition(self, from_state: str, to_state: str, duration: float = 0.0):
        """Registra uma transição de estado."""
        key = f"{from_state}->{to_state}"
        self.metrics.state_transitions[key] = self.metrics.state_transitions.get(key, 0) + 1
        
        # Medir tempo lobby -> in_game
        if from_state == 'lobby' and to_state in ('in_game', 'loading', 'matchmaking'):
            self.metrics.lobby_to_game_time = duration
            if duration > self.THRESHOLDS['lobby_to_game_warning']:
                logger.warning(f"[IMPROVEMENT] Lobby->Jogo lento: {duration:.1f}s (threshold: {self.THRESHOLDS['lobby_to_game_warning']}s)")
                self._suggest_lobby_improvement()
    
    def record_detection(self, success: bool):
        """Registra sucesso/falha de detecção."""
        self.metrics.detection_total += 1
        if success:
            self.metrics.detection_success += 1
        self.metrics.detection_success_rate = self.metrics.detection_success / max(self.metrics.detection_total, 1)
    
    def record_combat_action(self, action_type: str):
        """Registra uma ação de combate."""
        if self.metrics.combat_start_time == 0.0:
            self.metrics.combat_start_time = time.time()
        
        self.metrics.total_combat_actions += 1
        elapsed = time.time() - self.metrics.combat_start_time
        if elapsed > 60:
            self.metrics.actions_per_minute = self.metrics.total_combat_actions / (elapsed / 60)
    
    def record_cycle_time(self, cycle_time: float):
        """Registra tempo de um ciclo do bot."""
        self.metrics.cycle_times.append(cycle_time)
        # Manter apenas últimos 100 ciclos
        if len(self.metrics.cycle_times) > 100:
            self.metrics.cycle_times.pop(0)
        self.metrics.avg_cycle_time = sum(self.metrics.cycle_times) / len(self.metrics.cycle_times)
    
    def record_error(self, error_type: str):
        """Registra um erro."""
        self.metrics.errors_count[error_type] = self.metrics.errors_count.get(error_type, 0) + 1
    
    def record_popup(self, closed: bool):
        """Registra resultado de fechamento de popup."""
        if closed:
            self.metrics.popups_closed += 1
        else:
            self.metrics.popup_failures += 1
    
    def _suggest_lobby_improvement(self):
        """Sugere melhorias para o lobby."""
        suggestion = {
            'timestamp': time.time(),
            'area': 'lobby',
            'issue': 'lobby_to_game_too_slow',
            'suggestions': [
                'Verificar coordenadas do botão Play',
                'Aumentar timeout de matchmaking',
                'Verificar se há popups bloqueando',
            ]
        }
        self._improvements_applied.append(suggestion)
        logger.info(f"[IMPROVEMENT] Sugestão gerada: {suggestion['issue']}")
    
    def _suggest_combat_improvement(self):
        """Sugere melhorias para o combate."""
        suggestion = {
            'timestamp': time.time(),
            'area': 'combat',
            'issue': 'actions_per_minute_too_low',
            'suggestions': [
                'Verificar se YOLO está detectando corretamente',
                'Aumentar fallback de movimento',
                'Revisar lógica de targeting',
            ]
        }
        self._improvements_applied.append(suggestion)
    
    def check_and_adjust(self) -> Dict[str, Any]:
        """
        Verifica métricas e retorna ajustes recomendados.
        Chamado periodicamente pelo StateManager.
        """
        adjustments = {}
        
        # Verificar actions_per_minute
        if self.metrics.actions_per_minute < self.THRESHOLDS['actions_per_min_warning']:
            adjustments['combat_fallback'] = True
            self._suggest_combat_improvement()
            logger.warning(f"[IMPROVEMENT] Ações/min baixas ({self.metrics.actions_per_minute:.1f}) - ativando fallback agressivo")
        
        # Verificar detection_rate
        if self.metrics.detection_success_rate < self.THRESHOLDS['detection_rate_warning']:
            adjustments['increase_detection_retry'] = True
            logger.warning(f"[IMPROVEMENT] Taxa de detecção baixa ({self.metrics.detection_success_rate:.1%}) - aumentando retries")
        
        # Verificar cycle_time
        if self.metrics.avg_cycle_time > self.THRESHOLDS['cycle_time_max']:
            adjustments['optimize_cycles'] = True
            logger.warning(f"[IMPROVEMENT] Ciclo lento ({self.metrics.avg_cycle_time:.2f}s) - otimizando")
        
        return adjustments
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo das métricas."""
        self.metrics.session_duration = time.time() - self.metrics.session_start
        return {
            'session_duration': self.metrics.session_duration,
            'lobby_to_game_time': self.metrics.lobby_to_game_time,
            'detection_rate': self.metrics.detection_success_rate,
            'actions_per_minute': self.metrics.actions_per_minute,
            'total_actions': self.metrics.total_combat_actions,
            'popups_closed': self.metrics.popups_closed,
            'popup_failures': self.metrics.popup_failures,
            'avg_cycle_time': self.metrics.avg_cycle_time,
            'errors': self.metrics.errors_count,
            'state_transitions': self.metrics.state_transitions,
            'improvements_suggested': len(self._improvements_applied),
        }
    
    def save_report(self):
        """Salva relatório de métricas e melhorias."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(self.metrics),
            'summary': self.get_summary(),
            'improvements': self._improvements_applied,
        }
        
        try:
            with open(self.report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"[IMPROVEMENT] Relatório salvo em {self.report_path}")
        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
            logger.error(f"[IMPROVEMENT] Erro ao salvar relatório: {e}")
    
    def periodic_save(self):
        """Salva métricas periodicamente (a cada 60s)."""
        if time.time() - self._last_save > 60:
            self.save_report()
            self._last_save = time.time()
