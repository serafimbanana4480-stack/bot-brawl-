"""
state_manager.py

Gestor de estados do jogo — orquestrador.
Automatiza ações baseado no estado atual.

DEPRECATED: Use core.orchestrator.BotOrchestrator instead.
This module is kept for backward compatibility only.

Refactor: detection logic moved to core.state_detection,
transition handlers moved to core.state_transitions.
"""

import warnings
import time
import random
import logging
from typing import Dict, Callable, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import pyautogui
except ImportError:
    pyautogui = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

warnings.warn(
    "Deprecated: use core.orchestrator.BotOrchestrator instead",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from core.resolution_manager import ResolutionManager
    HAS_RESOLUTION_MANAGER = True
except ImportError:
    HAS_RESOLUTION_MANAGER = False
    ResolutionManager = None  # type: ignore[misc,assignment]

# Phase 10: LobbyFSM for hierarchical lobby state management
try:
    from core.lobby_fsm import HierarchicalFSM as LobbyFSM, TopLevelState, LobbyState
    HAS_LOBBY_FSM = True
except ImportError:
    HAS_LOBBY_FSM = False
    LobbyFSM = None
    TopLevelState = None
    LobbyState = None

# Phase 10: Lobby automation expanded (PlayAgain, TrainingCave, PvE)
try:
    from pylaai_real.lobby_automation_expanded import PlayAgainResult, PvEClassification
    HAS_LOBBY_EXPANDED = True
except ImportError:
    HAS_LOBBY_EXPANDED = False
    PlayAgainResult = None
    PvEClassification = None

try:
    from ..realtime_logs import get_log_manager
    log_manager = get_log_manager()
except ImportError:
    log_manager = None

try:
    from .lobby_automation_expanded import (
        PlayAgainHandler,
        DailyRewardsCollector,
        StarrRoadAutomation,
        ShopAutomation,
        QuestAutomation,
        MaintenanceHandler,
    )
    LOBBY_EXPANDED_AVAILABLE = True
except ImportError:
    LOBBY_EXPANDED_AVAILABLE = False
    PlayAgainHandler = None
    DailyRewardsCollector = None
    StarrRoadAutomation = None
    ShopAutomation = None
    QuestAutomation = None
    MaintenanceHandler = None

from core.state_detection import StateDetectionMixin
from core.state_transitions import StateTransitionsMixin


class StateManager(StateDetectionMixin, StateTransitionsMixin):
    """Gerencia estados do jogo e executa ações apropriadas (orquestrador)."""

    def __init__(
        self,
        screenshot_taker,
        state_finder,
        lobby_automator=None,
        progress_observer=None,
        play_logic=None,
        match_controller=None,
        emulator_controller=None,
        screen_automation=None,  # Mantido para compat, mas NÃO usado como thread
        movement=None,
        diagnostic_mode: bool = False,
        reward_bridge=None,
        data_collector=None,
        brawler_selector=None,
        observability=None,
        unified_state_detector=None,  # NOVO: detector unificado
        ocr_detector=None,  # NOVO: OCR detector para fallback híbrido
        lobby=None,
        rl_engine=None,  # NOVO: motor de RL online
        learning_mode_controller=None,  # NOVO: modo teste aprendizagem
        auto_fix_engine=None,  # NOVO: motor de auto-diagnóstico e recovery
    ):
        # Aceitar 'lobby' como alias de 'lobby_automator'
        if lobby is not None and lobby_automator is None:
            lobby_automator = lobby
        self.screenshot = screenshot_taker
        self.state_finder = state_finder
        self.lobby = lobby_automator
        self.progress = progress_observer
        self.play = play_logic
        self.match_controller = match_controller
        self.emulator_controller = emulator_controller
        self.screen_automation = screen_automation  # Mantido para hints apenas
        self.movement = movement
        self.diagnostic_mode = diagnostic_mode
        self.reward_bridge = reward_bridge
        self.data_collector = data_collector
        self.brawler_selector = brawler_selector
        self.observability = observability
        self.rl_engine = rl_engine  # NOVO
        self.learning_mode_controller = learning_mode_controller  # NOVO
        self.auto_fix = auto_fix_engine

        # NOVO: Detector unificado (prioridade sobre state_finder antigo)
        self.unified_detector = unified_state_detector
        self.ocr_detector = ocr_detector
        if self.unified_detector:
            logger.info("[STATE] Usando UnifiedStateDetector para deteção")
        else:
            logger.info("[STATE] Usando StateFinder legado para deteção")

        if self.rl_engine:
            logger.info("[STATE] Motor de RL online ativo")

        # Phase 10: LobbyFSM for hierarchical lobby management
        self.lobby_fsm = None
        if HAS_LOBBY_FSM:
            try:
                self.lobby_fsm = LobbyFSM()
                logger.info("[STATE] LobbyFSM inicializado")
            except Exception as e:
                logger.warning(f"[STATE] LobbyFSM indisponível: {e}")

        # Phase 10: Adaptive Screenshot Cache
        self._adaptive_cache = None
        try:
            from core.adaptive_screenshot import AdaptiveScreenshotCache
            self._adaptive_cache = AdaptiveScreenshotCache()
            logger.info("[STATE] AdaptiveScreenshotCache inicializado")
        except ImportError:
            logger.debug("[STATE] AdaptiveScreenshotCache não disponível")

        # Phase 10: State Persistence for recovery
        self._state_persistence = None
        try:
            from state_persistence import StatePersistence
            self._state_persistence = StatePersistence()
            logger.info("[STATE] StatePersistence inicializado")
        except ImportError:
            logger.debug("[STATE] StatePersistence não disponível")

        logger.debug(f"[STATE] StateManager inicializado com movement: {movement is not None}, "
                    f"reward_bridge: {reward_bridge is not None}, data_collector: {data_collector is not None}, "
                    f"brawler_selector: {brawler_selector is not None}, observability: {observability is not None}")

        # Mapear estados para handlers
        self.states: Dict[str, Callable] = {
            'lobby': self._handle_lobby,
            'loading': self._handle_loading,
            'matchmaking': self._handle_matchmaking,
            'connection_lost': self._handle_connection_lost,
            'brawler_selection': self._handle_brawler_selection,
            'shop': self._handle_shop,
            'popup': self._handle_popup,
            'end': self._handle_end_game,
            'in_game': self._handle_in_game,
            'unknown': self._handle_unknown,
            # NOVO: modo teste aprendizagem (Training Cave)
            'in_game_learning': self._handle_in_game_learning,
            # NOVOS estados adicionais (ANALISE_PROFUNDA)
            'tutorial': self._handle_tutorial,
            'news': self._handle_news,
            'brawler_unlock': self._handle_brawler_unlock,
            'season_reset': self._handle_season_reset,
            'event_screen': self._handle_event_screen,
            'starr_drop': self._handle_starr_drop,
        }

        self.running = False
        self.current_state = 'unknown'
        self.current_brawler = None  # Set by lobby automator when brawler is selected
        self._current_map = None  # Set by movement detector or lobby automator
        self.last_known_state = 'unknown'
        self.last_known_state_at = None
        self.unknown_streak = 0
        self.last_unknown_hint = None
        self._unknown_hold_cycles = 2
        self.unknown_since = None

        # Buffer de screenshot compartilhado (evita captura duplicada)
        self._last_screenshot = None
        self._last_screenshot_time = 0
        self._paused = False

        # State timeout tracking to prevent infinite loops
        self.state_start_time = None
        
        # ANTI-OSCILLATION: Cooldown entre transições e lock de estado
        self._last_transition_time = 0.0
        self._state_transition_cooldown = 3.0  # segundos mínimos entre transições
        self._in_game_lock = False  # Quando True, só sai de in_game para end/connection_lost
        self._in_game_min_duration = 5.0  # Mínimo de segundos em in_game antes de permitir saída
        
        # Continuous Improvement System
        try:
            from core.continuous_improvement import ContinuousImprovement
            self.improvement_system = ContinuousImprovement()
            logger.info("[STATE] Sistema de Melhoria Contínua ativado")
        except ImportError:
            self.improvement_system = None

        # Autonomous Tester (auto-diagnóstico)
        try:
            from autonomous_tester import AutonomousTester
            self.autonomous_tester = AutonomousTester(state_manager=self)
            logger.info("[STATE] AutonomousTester ativado")
        except ImportError:
            self.autonomous_tester = None
        
        self.state_timeouts = {
            'loading': 30,    # 30 seconds max in loading
            'lobby': 60,     # 60 seconds max in lobby
            'matchmaking': 60,  # 60 seconds max in matchmaking
            'brawler_selection': 30,  # 30 seconds max in brawler selection
            'in_game': 180,  # 180 seconds max in game (3 min) - forces end_match with "unknown"
            'end': 45,       # 45 seconds max in end screen
            'unknown': 60,   # 60 seconds max in unknown before forced reset
            'in_game_learning': 300,  # 5 min max por partida de treino
            'tutorial': 45,  # 45 seconds max in tutorial
            'news': 20,      # 20 seconds max in news (deve ser fechada rapido)
            'brawler_unlock': 30,  # 30 seconds max
            'season_reset': 20,  # 20 seconds max
        }

        # Valid state transitions (for validation and debugging)
        self.VALID_TRANSITIONS = {
            'unknown': ['lobby', 'in_game', 'in_game_learning', 'loading', 'matchmaking', 'brawler_selection', 'shop', 'popup', 'end', 'connection_lost', 'tutorial', 'news', 'brawler_unlock', 'season_reset', 'event_screen', 'starr_drop'],
            'lobby': ['loading', 'matchmaking', 'brawler_selection', 'shop', 'popup', 'unknown', 'in_game', 'in_game_learning', 'news', 'brawler_unlock', 'season_reset'],
            'loading': ['in_game', 'in_game_learning', 'lobby', 'unknown', 'connection_lost'],
            'matchmaking': ['in_game', 'in_game_learning', 'loading', 'lobby', 'unknown', 'connection_lost'],
            'brawler_selection': ['lobby', 'loading', 'matchmaking', 'unknown'],
            'in_game': ['end', 'connection_lost', 'unknown', 'lobby'],
            'in_game_learning': ['lobby', 'end', 'unknown', 'in_game_learning'],
            'end': ['lobby', 'brawler_selection', 'loading', 'unknown'],
            'shop': ['lobby', 'unknown'],
            'popup': ['lobby', 'unknown'],
            'connection_lost': ['lobby', 'loading', 'unknown'],
            'tutorial': ['lobby', 'unknown'],
            'news': ['lobby', 'unknown'],
            'brawler_unlock': ['lobby', 'unknown'],
            'season_reset': ['lobby', 'unknown'],
        }

    def run(self):
        """Loop principal do state manager - otimizado para performance."""
        self.running = True
        self.unknown_since = None
        logger.info("[STATE] State Manager iniciado")
        logger.info(f"[STATE] Handlers registrados: {list(self.states.keys())}")
        logger.info(f"[STATE] Modo diagnóstico: {self.diagnostic_mode}")

        while self.running:
            try:
                # Verificar pausa
                if self._is_paused():
                    time.sleep(0.5)
                    continue

                cycle_start = time.time()

                # === NOVO: Verificar manutencao/update antes de tudo ===
                if self.lobby and hasattr(self.lobby, 'handle_maintenance') and self.emulator_controller:
                    try:
                        screenshot = self.screenshot.take()
                        if screenshot is not None:
                            maint_handled = self.lobby.handle_maintenance(screenshot)
                            if maint_handled:
                                logger.warning("[STATE] Tela de manutencao/update detectada e tratada. Aguardando...")
                                time.sleep(5.0)
                                continue
                    except Exception as e:
                        logger.debug(f"[STATE] Erro ao verificar manutencao: {e}")

                # === NOVO: Verificar convites de partidas amistosas ===
                if self.lobby and hasattr(self.lobby, 'handle_friendly_invite') and self.emulator_controller:
                    try:
                        screenshot = self.screenshot.take()
                        if screenshot is not None:
                            invite_handled = self.lobby.handle_friendly_invite(screenshot, auto_accept=False)
                            if invite_handled:
                                logger.info("[STATE] Convite de partida amistosa recusado (auto_accept=False)")
                                time.sleep(0.5)
                                continue
                    except Exception as e:
                        logger.debug(f"[STATE] Erro ao verificar convites: {e}")

                # === NOVO: Auto-fix engine - diagnóstico e recovery automático ===
                if self.auto_fix:
                    try:
                        forced_state = self.auto_fix.tick(self.current_state)
                        if forced_state and forced_state != self.current_state:
                            logger.info(f"[STATE] AutoFix forçou estado: {self.current_state} -> {forced_state}")
                            self.current_state = forced_state
                            self.state_start_time = time.time()
                            if forced_state == 'lobby':
                                self._remember_known_state('lobby')
                    except Exception as e:
                        logger.debug(f"[STATE] AutoFix error: {e}")

                # FORCE LOADING TIMEOUT: if in loading for >5s, skip detection and force in_game
                logger.info(f"[STATE] DEBUG: current={self.current_state}, start_time={self.state_start_time}")
                if self.current_state == 'loading':
                    if not self.state_start_time:
                        self.state_start_time = time.time()
                        logger.info("[STATE] Initialized loading_start_time")
                    loading_elapsed = time.time() - self.state_start_time
                    logger.info(f"[STATE] Loading elapsed: {loading_elapsed:.1f}s / 5s timeout")
                    if loading_elapsed > 5:
                        logger.warning(f"[STATE] FORCE: Loading timeout ({loading_elapsed:.0f}s > 5s), forcing in_game")
                        self.current_state = 'in_game'
                        self.state_start_time = time.time()
                        self._remember_known_state('in_game')
                        self._forced_in_game_time = time.time()
                        logger.info("[STATE] Forçado in_game desde loading (main loop) - bloqueando retorno por 30s")
                        # Skip _process_cycle to avoid being reset back to loading
                        time.sleep(0.5)
                        continue
                
                self._process_cycle()

                # AutonomousTester: verificação periódica de saúde do bot
                if self.autonomous_tester:
                    try:
                        self.autonomous_tester.periodic_check()
                    except Exception as e:
                        logger.debug(f"[STATE] AutonomousTester error: {e}")

                # Delay adaptativo: mais rapido em in_game, mais lento em estados estaticos
                elapsed = time.time() - cycle_start
                if self.current_state == 'in_game':
                    # In-game: ciclo rapido para reagir a inimigos (0.15-0.25s)
                    delay = max(0.05, random.uniform(0.12, 0.25) - elapsed)
                elif self.current_state in ('loading', 'matchmaking'):
                    # Loading/matchmaking: verificar menos frequentemente
                    delay = max(0.5, 1.5 - elapsed)
                else:
                    # Lobby/outros: intervalo medio
                    delay = max(0.1, random.uniform(0.3, 0.6) - elapsed)
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Erro no state manager: {e}")
                time.sleep(random.uniform(0.5, 1.0))

    def stop(self):
        """Para o state manager"""
        self.running = False
        logger.info("State Manager parado")

    def pause(self):
        """Pausa o state manager (mantem thread viva mas nao processa ciclos)"""
        self._paused = True
        logger.info("[STATE] State Manager pausado")

    def resume(self):
        """Retoma o state manager apos pausa"""
        self._paused = False
        logger.info("[STATE] State Manager retomado")

    def _is_paused(self) -> bool:
        """Verifica se o state manager esta pausado"""
        return getattr(self, '_paused', False)

