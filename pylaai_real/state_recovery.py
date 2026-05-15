"""
state_recovery.py

Sistema de recuperação de estados para lidar com situações desconhecidas.
Implementa estratégias de fallback quando o bot fica preso em estados não reconhecidos.

Funcionalidades:
- Stack de recuperação com múltiplas estratégias
- Detecção de loops (estado repetido muitas vezes)
- Backtracking automático para estados conhecidos
- Timeout para estados desconhecidos
- Logging detalhado para troubleshooting
"""

import time
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Ações de recuperação disponíveis."""
    PRESS_BACK = "press_back"
    WAIT = "wait"
    TAP_CENTER = "tap_center"
    TAP_TOP_LEFT = "tap_top_left"
    SWIPE_DOWN = "swipe_down"
    RESTART_GAME = "restart_game"
    CLEAR_CACHE = "clear_cache"


@dataclass
class RecoveryStep:
    """Um passo na sequência de recuperação."""
    action: RecoveryAction
    params: dict = field(default_factory=dict)
    description: str = ""
    timeout: float = 2.0


@dataclass
class StateHistory:
    """Histórico de estados para detecção de loops."""
    states: deque = field(default_factory=lambda: deque(maxlen=20))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=20))
    
    def add(self, state: str):
        """Adiciona estado ao histórico."""
        self.states.append(state)
        self.timestamps.append(time.time())
    
    def detect_loop(self, window_size: int = 5) -> bool:
        """Detecta se está em loop (mesmo estado repetindo)."""
        if len(self.states) < window_size:
            return False
        
        recent = list(self.states)[-window_size:]
        return len(set(recent)) == 1  # Todos os estados são iguais
    
    def detect_oscillation(self, window_size: int = 10) -> bool:
        """Detecta oscilação entre 2 estados."""
        if len(self.states) < window_size:
            return False
        
        recent = list(self.states)[-window_size:]
        unique_states = set(recent)
        
        # Oscila se alternar entre apenas 2 estados
        return len(unique_states) == 2 and len(recent) >= window_size
    
    def get_dominant_state(self, window_size: int = 10) -> Optional[str]:
        """Retorna o estado mais frequente na janela."""
        if len(self.states) < window_size:
            return None
        
        recent = list(self.states)[-window_size:]
        from collections import Counter
        counts = Counter(recent)
        return counts.most_common(1)[0][0]
    
    def time_in_current_state(self) -> float:
        """Tempo (segundos) no estado atual."""
        if len(self.timestamps) < 2:
            return 0.0
        
        return time.time() - self.timestamps[-1]


class StateRecoverySystem:
    """
    Sistema de recuperação de estados.
    
    Estratégias de recuperação (em ordem):
    1. Back simples (pressionar ESC/Back)
    2. Back múltiplo (pressionar Back várias vezes)
    3. Tap no centro (tentar interagir)
    4. Swipe para baixo (scroll)
    5. Restart do jogo (último recurso)
    """
    
    def __init__(
        self,
        emulator_controller=None,
        max_unknown_duration: float = 30.0,
        max_loop_duration: float = 15.0,
        enable_auto_restart: bool = False
    ):
        self.emulator_controller = emulator_controller
        self.max_unknown_duration = max_unknown_duration
        self.max_loop_duration = max_loop_duration
        self.enable_auto_restart = enable_auto_restart
        
        # Histórico de estados
        self.history = StateHistory()
        
        # Estado atual e timestamp
        self.current_state = "unknown"
        self.state_start_time = time.time()
        
        # Contador de tentativas de recuperação
        self.recovery_attempts = 0
        self.max_recovery_attempts = 5
        
        # Sequência de recuperação ativa
        self.active_recovery: Optional[List[RecoveryStep]] = None
        self.recovery_step_index = 0
        self.recovery_start_time = 0.0
        
        # Estratégias de recuperação pré-definidas
        self.recovery_strategies = {
            "unknown": self._get_unknown_recovery(),
            "stuck": self._get_stuck_recovery(),
            "loop": self._get_loop_recovery(),
            "popup": self._get_popup_recovery(),
            "network_error": self._get_network_error_recovery()
        }
        
        logger.info("[RECOVERY] Sistema de recuperação inicializado")
    
    def update_state(self, state: str, confidence: float = 1.0):
        """
        Atualiza estado atual e verifica se precisa de recuperação.
        
        Args:
            state: Estado atual detectado
            confidence: Confiança da detecção (0-1)
        """
        
        # Se estado mudou
        if state != self.current_state:
            self.history.add(state)
            self.current_state = state
            self.state_start_time = time.time()
            self.recovery_attempts = 0  # Reset contador
            logger.debug(f"[RECOVERY] Estado mudou para: {state}")
        
        # Verificar se precisa de recuperação
        if self._needs_recovery(state, confidence):
            self._trigger_recovery(state)
    
    def _needs_recovery(self, state: str, confidence: float) -> bool:
        """Verifica se o estado atual requer recuperação."""
        
        # 1. Estado desconhecido por muito tempo
        if state == "unknown" or confidence < 0.5:
            time_in_state = self.history.time_in_current_state()
            if time_in_state > self.max_unknown_duration:
                logger.warning(f"[RECOVERY] Estado desconhecido por {time_in_state:.1f}s")
                return True
        
        # 2. Loop detectado
        if self.history.detect_loop():
            time_in_state = self.history.time_in_current_state()
            if time_in_state > self.max_loop_duration:
                logger.warning(f"[RECOVERY] Loop detectado por {time_in_state:.1f}s")
                return True
        
        # 3. Oscilação detectada
        if self.history.detect_oscillation():
            time_in_state = self.history.time_in_current_state()
            if time_in_state > self.max_loop_duration:
                logger.warning(f"[RECOVERY] Oscilação detectada por {time_in_state:.1f}s")
                return True
        
        # 4. Muitas tentativas de recuperação sem sucesso
        if self.recovery_attempts >= self.max_recovery_attempts:
            logger.error(f"[RECOVERY] Máximo de tentativas de recuperação atingido")
            return True
        
        return False
    
    def _trigger_recovery(self, state: str):
        """Inicia sequência de recuperação."""
        
        # Determinar tipo de recuperação
        if state == "unknown":
            recovery_type = "unknown"
        elif self.history.detect_loop():
            recovery_type = "loop"
        elif self.history.detect_oscillation():
            recovery_type = "loop"
        else:
            recovery_type = "stuck"
        
        # Obter sequência de recuperação
        self.active_recovery = self.recovery_strategies.get(recovery_type, self._get_unknown_recovery())
        self.recovery_step_index = 0
        self.recovery_start_time = time.time()
        self.recovery_attempts += 1
        
        logger.warning(f"[RECOVERY] Iniciando recuperação tipo: {recovery_type} (tentativa {self.recovery_attempts})")
    
    def execute_recovery_step(self) -> bool:
        """
        Executa próximo passo da recuperação.
        
        Returns:
            True se recuperação está em progresso, False se completou
        """
        if self.active_recovery is None:
            return False
        
        # Verificar timeout da recuperação
        if time.time() - self.recovery_start_time > 30.0:
            logger.error("[RECOVERY] Timeout da recuperação")
            self.active_recovery = None
            return False
        
        # Executar passo atual
        if self.recovery_step_index >= len(self.active_recovery):
            logger.info("[RECOVERY] Sequência de recuperação completada")
            self.active_recovery = None
            return False
        
        step = self.active_recovery[self.recovery_step_index]
        logger.info(f"[RECOVERY] Executando passo {self.recovery_step_index + 1}/{len(self.active_recovery)}: {step.description}")
        
        # Executar ação
        self._execute_action(step.action, step.params)
        
        # Avançar para próximo passo
        self.recovery_step_index += 1
        
        # Esperar antes de próximo passo
        time.sleep(step.timeout)
        
        return True
    
    def _execute_action(self, action: RecoveryAction, params: dict):
        """Executa uma ação de recuperação."""
        
        if self.emulator_controller is None:
            logger.warning("[RECOVERY] EmulatorController não disponível, simulando ação")
            return
        
        try:
            if action == RecoveryAction.PRESS_BACK:
                self.emulator_controller.press_back()
            
            elif action == RecoveryAction.WAIT:
                duration = params.get("duration", 1.0)
                time.sleep(duration)
            
            elif action == RecoveryAction.TAP_CENTER:
                w, h = params.get("width", 1920), params.get("height", 1080)
                self.emulator_controller.tap(w // 2, h // 2)
            
            elif action == RecoveryAction.TAP_TOP_LEFT:
                self.emulator_controller.tap(100, 100)
            
            elif action == RecoveryAction.SWIPE_DOWN:
                w, h = params.get("width", 1920), params.get("height", 1080)
                self.emulator_controller.swipe(w // 2, h // 3, w // 2, h * 2 // 3, 500)
            
            elif action == RecoveryAction.RESTART_GAME:
                if self.enable_auto_restart:
                    self.emulator_controller.restart_game()
                else:
                    logger.warning("[RECOVERY] Auto-restart desabilitado")
            
            elif action == RecoveryAction.CLEAR_CACHE:
                self.emulator_controller.clear_app_cache()
            
            logger.debug(f"[RECOVERY] Ação executada: {action.value}")
        
        except Exception as e:
            logger.error(f"[RECOVERY] Erro ao executar ação {action.value}: {e}")
    
    # Estratégias de recuperação
    
    def _get_unknown_recovery(self) -> List[RecoveryStep]:
        """Estratégia para estado desconhecido."""
        return [
            RecoveryStep(
                action=RecoveryAction.WAIT,
                params={"duration": 2.0},
                description="Aguardar 2s"
            ),
            RecoveryStep(
                action=RecoveryAction.PRESS_BACK,
                description="Pressionar Back"
            ),
            RecoveryStep(
                action=RecoveryAction.WAIT,
                params={"duration": 1.0},
                description="Aguardar 1s"
            ),
            RecoveryStep(
                action=RecoveryAction.PRESS_BACK,
                description="Pressionar Back novamente"
            ),
            RecoveryStep(
                action=RecoveryAction.WAIT,
                params={"duration": 2.0},
                description="Aguardar 2s"
            ),
            RecoveryStep(
                action=RecoveryAction.TAP_CENTER,
                description="Tocar no centro"
            ),
        ]
    
    def _get_stuck_recovery(self) -> List[RecoveryStep]:
        """Estratégia para estado preso (mas conhecido)."""
        return [
            RecoveryStep(
                action=RecoveryAction.PRESS_BACK,
                description="Pressionar Back"
            ),
            RecoveryStep(
                action=RecoveryAction.WAIT,
                params={"duration": 1.5},
                description="Aguardar 1.5s"
            ),
            RecoveryStep(
                action=RecoveryAction.SWIPE_DOWN,
                description="Swipe para baixo"
            ),
        ]
    
    def _get_loop_recovery(self) -> List[RecoveryStep]:
        """Estratégia para loop/oscilação."""
        return [
            RecoveryStep(
                action=RecoveryAction.PRESS_BACK,
                description="Pressionar Back (loop)"
            ),
            RecoveryStep(
                action=RecoveryAction.WAIT,
                params={"duration": 0.5},
                description="Aguardar 0.5s"
            ),
            RecoveryStep(
                action=RecoveryAction.PRESS_BACK,
                description="Pressionar Back novamente"
            ),
            RecoveryStep(
                action=RecoveryAction.WAIT,
                params={"duration": 0.5},
                description="Aguardar 0.5s"
            ),
            RecoveryStep(
                action=RecoveryAction.PRESS_BACK,
                description="Pressionar Back terceira vez"
            ),
            RecoveryStep(
                action=RecoveryAction.WAIT,
                params={"duration": 2.0},
                description="Aguardar 2s"
            ),
            RecoveryStep(
                action=RecoveryAction.TAP_TOP_LEFT,
                description="Tocar no topo esquerdo"
            ),
        ]
    
    def _get_popup_recovery(self) -> List[RecoveryStep]:
        """Estratégia para popups inesperados."""
        return [
            RecoveryStep(
                action=RecoveryAction.TAP_TOP_LEFT,
                description="Tentar fechar popup (topo esquerdo)"
            ),
            RecoveryStep(
                action=RecoveryAction.WAIT,
                params={"duration": 1.0},
                description="Aguardar 1s"
            ),
            RecoveryStep(
                action=RecoveryAction.PRESS_BACK,
                description="Pressionar Back"
            ),
        ]
    
    def _get_network_error_recovery(self) -> List[RecoveryStep]:
        """Estratégia para erros de rede."""
        return [
            RecoveryStep(
                action=RecoveryAction.WAIT,
                params={"duration": 5.0},
                description="Aguardar 5s (reconexão)"
            ),
            RecoveryStep(
                action=RecoveryAction.PRESS_BACK,
                description="Pressionar Back"
            ),
            RecoveryStep(
                action=RecoveryAction.WAIT,
                params={"duration": 2.0},
                description="Aguardar 2s"
            ),
        ]
    
    def is_recovering(self) -> bool:
        """Verifica se recuperação está em progresso."""
        return self.active_recovery is not None
    
    def cancel_recovery(self):
        """Cancela recuperação em progresso."""
        if self.active_recovery:
            logger.info("[RECOVERY] Recuperação cancelada")
            self.active_recovery = None
            self.recovery_step_index = 0
    
    def get_recovery_status(self) -> dict:
        """Retorna status da recuperação."""
        return {
            "is_recovering": self.is_recovering(),
            "recovery_attempts": self.recovery_attempts,
            "current_step": self.recovery_step_index if self.active_recovery else 0,
            "total_steps": len(self.active_recovery) if self.active_recovery else 0,
            "current_state": self.current_state,
            "time_in_state": self.history.time_in_current_state(),
            "history_length": len(self.history.states),
            "dominant_state": self.history.get_dominant_state()
        }
    
    def reset(self):
        """Reseta o sistema de recuperação."""
        self.history = StateHistory()
        self.current_state = "unknown"
        self.state_start_time = time.time()
        self.recovery_attempts = 0
        self.active_recovery = None
        self.recovery_step_index = 0
        logger.info("[RECOVERY] Sistema resetado")
