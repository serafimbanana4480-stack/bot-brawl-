"""
state_manager.py

Gestor de estados do jogo.
Automatiza ações baseado no estado atual.

MELHORIA: Agora usa UnifiedStateDetector para deteção unificada.
ScreenAutomation thread NÃO é mais iniciada — toda a deteção acontece
neste ciclo principal, eliminando conflitos de cliques duplos.
"""

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


class StateManager:
    """Gerencia estados do jogo e executa ações apropriadas"""

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
            import sys
            from pathlib import Path
            _sp_path = Path(__file__).parent.parent
            if str(_sp_path) not in sys.path:
                sys.path.insert(0, str(_sp_path))
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

    def _remember_known_state(self, state: str):
        """Regista o último estado confiável para segurar breves oscilações em unknown."""
        self.last_known_state = state
        self.last_known_state_at = time.time()
        self.unknown_streak = 0
        self.last_unknown_hint = None
        self.unknown_since = None

    def _get_cached_screenshot(self, max_age: float = 0.15):
        """
        Reutiliza screenshot recente para evitar capturas duplicadas.
        Em in_game, max_age=0.15s mantem fluidez; em outros estados pode ser maior.
        Phase 10: Usa AdaptiveScreenshotCache se disponível para TTL adaptativo.
        """
        # Phase 10: Use adaptive cache if available
        if self._adaptive_cache is not None:
            try:
                game_state = self.current_state or "unknown"
                image = self._adaptive_cache.get_screenshot(
                    capture_fn=self.screenshot.take,
                    game_state=game_state,
                )
                if image is not None:
                    self._last_screenshot = image
                    self._last_screenshot_time = time.time()
                return image
            except Exception as e:
                # FIX #10: screenshot cache failure should be logged as error
                logger.error(f"[STATE] AdaptiveScreenshotCache failed: {e}")

        # Standard cache with fixed TTL
        now = time.time()
        if self._last_screenshot is not None and (now - self._last_screenshot_time) < max_age:
            # FIX #14: Return copy to prevent caller mutation from corrupting cache
            return self._last_screenshot.copy()
        image = self.screenshot.take()
        if image is not None:
            self._last_screenshot = image
            self._last_screenshot_time = now
        return image

    def _process_cycle(self):
        """Processa um ciclo de estado e executa o handler apropriado - otimizado."""
        logger.debug(f"[STATE] Iniciando ciclo de processamento, estado atual: {self.current_state}")
        image = self._get_cached_screenshot()
        if image is None:
            logger.warning("[STATE] Screenshot falhou - image é None")
            time.sleep(0.3)
            return None
        logger.debug(f"[STATE] Screenshot capturado com sucesso, shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")

        # === VERIFICAR POPUPS (v2) ===
        # Popups atrapalham TODOS os handlers. Verificar e fechar primeiro.
        # FIX #18: Track consecutive popup detections to prevent infinite sleep loops
        if not hasattr(self, '_popup_consecutive_count'):
            self._popup_consecutive_count = 0
        if not hasattr(self, '_last_popup_type'):
            self._last_popup_type = None
        if not hasattr(self, '_popup_ignore_until'):
            self._popup_ignore_until = 0.0

        # Verificar se estamos em cooldown de popup
        if time.time() < self._popup_ignore_until:
            logger.debug(f"[STATE] Popup check ignorado (cooldown até {self._popup_ignore_until - time.time():.0f}s)")
        elif self.lobby and hasattr(self.lobby, '_popup_manager') and self.lobby._popup_manager:
            try:
                import numpy as np
                if isinstance(image, np.ndarray):
                    popup = self.lobby._popup_manager.detect_popup(image)
                    if popup and popup.confidence > 0.3:
                        # Check if this is the same popup as last cycle
                        same_popup = (popup.popup_type == self._last_popup_type)
                        if same_popup:
                            self._popup_consecutive_count += 1
                        else:
                            self._popup_consecutive_count = 1
                            self._last_popup_type = popup.popup_type

                        logger.info(f"[STATE] Popup detectado antes de handler: {popup.popup_type}, "
                                    f"fechando... (consecutivo #{self._popup_consecutive_count})")

                        self.lobby._popup_manager.handle_popup(
                            popup,
                            click_func=self.lobby._click if hasattr(self.lobby, '_click') else lambda x, y: None,
                            key_func=self.lobby._key_press if hasattr(self.lobby, '_key_press') else lambda k: None
                        )

                        # FIX #18: Popup Loop Breaker - cooldown após 3 falhas
                        if self._popup_consecutive_count >= 3:
                            logger.warning(f"[STATE] Popup {popup.popup_type} NAO FECHOU apos 3 tentativas - IGNORANDO popups por 30s")
                            self._popup_ignore_until = time.time() + 30.0
                            self._popup_consecutive_count = 0
                        else:
                            time.sleep(random.uniform(0.3, 0.6))

                        # Se era um popup, ficamos em "unknown" e deixamos o ciclo seguinte resolver
                        if self.current_state != 'in_game':
                            return 'unknown'
            except Exception as e:
                logger.debug(f"[STATE] Erro na verificacao de popup: {e}")

        # Obter hint do screen_automation se disponível (apenas como SINAL, não como ação)
        screen_state_hint = None
        if self.screen_automation and hasattr(self.screen_automation, "get_current_state_name"):
            try:
                screen_state_hint = self.screen_automation.get_current_state_name()
                logger.debug(f"[STATE] Screen automation hint: {screen_state_hint}")
            except Exception as e:
                logger.debug(f"[STATE] Falha ao ler screen automation state: {e}")

        # NOTA: Não ativamos a janela antes da captura de screenshot porque
        # a captura via Win32 funciona em background. A ativação é feita
        # apenas antes de enviar inputs (toques), no play.py.

        # DETECÇÃO UNIFICADA: Usar UnifiedStateDetector se disponível
        if self.unified_detector:
            detection = self.unified_detector.detect(image, screen_hint=screen_state_hint)
            detected_state = detection.state
            detected_map = detection.map_name
            logger.info(f"[STATE] Unified detection: state={detected_state}, "
                        f"conf={detection.confidence:.2f}, method={detection.method}, "
                        f"button={detection.button_coords}")
        else:
            # Fallback para StateFinder legado
            detected_state_result = self.state_finder.get_state(image, screen_state_hint=screen_state_hint)
            detected_map = None
            if isinstance(detected_state_result, tuple) and len(detected_state_result) == 2:
                detected_state, detected_map = detected_state_result
            else:
                detected_state = detected_state_result

        # Aplicar mapa detectado ao movement
        if detected_map and self.movement:
            try:
                self.movement.set_current_map(detected_map)
                logger.info(f"[STATE] Mapa detectado: {detected_map}")
            except Exception as e:
                logger.warning(f"[STATE] Falha ao definir mapa detectado: {e}")

        logger.debug(f"[STATE] Estado detectado: {detected_state} (hint: {screen_state_hint})")

        if detected_state != 'unknown':
            if detected_state != self.current_state:
                # Validate transition
                valid_next = self.VALID_TRANSITIONS.get(self.current_state, [])

                # ANTI-OSCILLATION: Cooldown entre transições
                time_since_last_transition = time.time() - self._last_transition_time
                if time_since_last_transition < self._state_transition_cooldown:
                    logger.debug(f"[STATE] Transição bloqueada por cooldown ({time_since_last_transition:.1f}s < {self._state_transition_cooldown}s)")
                    detected_state = self.current_state

                # ANTI-OSCILLATION: Lock de in_game - só sai para end, connection_lost, ou unknown com confiança alta
                elif self.current_state == 'in_game' and detected_state not in ('end', 'connection_lost', 'unknown'):
                    time_in_game = time.time() - self.state_start_time if self.state_start_time else 0
                    if time_in_game < self._in_game_min_duration:
                        logger.warning(f"[STATE] BLOCKED: in_game -> {detected_state} (apenas {time_in_game:.1f}s em jogo, mínimo {self._in_game_min_duration}s)")
                        detected_state = self.current_state
                    else:
                        logger.warning(f"[STATE] BLOCKED: in_game -> {detected_state} (só permitido: end, connection_lost, unknown)")
                        detected_state = self.current_state

                elif valid_next and detected_state not in valid_next:
                    logger.warning(f"[STATE] Invalid transition: {self.current_state} -> {detected_state} (valid: {valid_next})")
                    # CRITICAL: Block impossible transitions that cause loops
                    if self.current_state == 'lobby' and detected_state == 'end':
                        logger.error("[STATE] BLOCKED impossible transition: lobby -> end. Staying in lobby.")
                        detected_state = 'lobby'  # Force stay in lobby
                    elif self.current_state == 'end' and detected_state == 'end':
                        logger.debug("[STATE] Already in end, ignoring repeated detection")
                        detected_state = self.current_state

                if detected_state != self.current_state:
                    logger.info(f"[STATE] Transição: {self.current_state} -> {detected_state}")
                    self._last_transition_time = time.time()
                if log_manager:
                    log_manager.log(
                        message=f"Transição de estado: {self.current_state} -> {detected_state}",
                        level="INFO",
                        category="state",
                        data={"from_state": self.current_state, "to_state": detected_state}
                    )
                logger.debug(f"[STATE] Motivo da transição: screen_hint={screen_state_hint}, template_match={detected_state}")
                if self.state_start_time:
                    time_in_previous_state = time.time() - self.state_start_time
                    logger.debug(f"[STATE] Tempo no estado anterior: {time_in_previous_state:.1f}s")
                    # Track transition for continuous improvement
                    if self.improvement_system:
                        self.improvement_system.record_state_transition(
                            self.current_state, detected_state, time_in_previous_state
                        )
                self._diag(f"transition_to={detected_state}, from={self.current_state}, reason={screen_state_hint}")
                # Reset state start time on transition
                self.state_start_time = time.time()
                # Reset in_game flag when leaving in_game state
                if self.current_state == 'in_game' and detected_state != 'in_game':
                    self._in_game_initialized = False
            elif self.state_start_time is None:
                # Initialize state start time if not set (e.g., on first run)
                self.state_start_time = time.time()
            else:
                current_time = time.time()
                time_in_state = current_time - self.state_start_time
                logger.debug(f"[STATE] Tempo no estado '{self.current_state}': {time_in_state:.1f}s")
                if time_in_state > 30:
                    logger.warning(f"[STATE] Estado '{self.current_state}' há muito tempo ({time_in_state:.1f}s)")
            self._remember_known_state(detected_state)
            self.current_state = detected_state
            logger.debug(f"[STATE] Estado confirmado: {detected_state}, streak resetado")
        else:
            self.unknown_streak += 1
            if self.last_known_state != 'unknown' and self.unknown_streak <= self._unknown_hold_cycles:
                self.last_unknown_hint = screen_state_hint
                self.current_state = self.last_known_state
                logger.warning(
                    f"[STATE] Unknown detectado ({self.unknown_streak}/{self._unknown_hold_cycles}); "
                    f"mantendo estado estável '{self.last_known_state}'"
                )
                self._diag(f"unknown_hold_state={self.last_known_state}")
                if screen_state_hint and screen_state_hint != 'unknown':
                    self._diag(f"unknown_hold_hint={screen_state_hint}")
                time.sleep(0.5)
                return self.current_state

            if detected_state != self.current_state:
                logger.info(f"Estado mudou: {self.current_state} -> {detected_state}")
            if self.current_state != 'unknown':
                self.unknown_since = time.time()
            self.current_state = 'unknown'

        if self.current_state == 'unknown' and self.unknown_since:
            elapsed = time.time() - self.unknown_since
            if elapsed > 10:
                self._try_intelligent_recovery(image, elapsed)

        # Check for state timeout to prevent infinite loops
        if self.current_state in self.state_timeouts and self.state_start_time:
            elapsed = time.time() - self.state_start_time
            timeout = self.state_timeouts[self.current_state]
            if elapsed > timeout:
                logger.warning(f"[STATE] Estado '{self.current_state}' preso por {elapsed:.1f}s (timeout: {timeout}s). Resetando para lobby.")
                if log_manager:
                    log_manager.log(
                        message=f"Estado preso por timeout, resetando para lobby: {self.current_state}",
                        level="WARNING",
                        category="state",
                        data={"state": self.current_state, "elapsed_time": elapsed, "timeout": timeout}
                    )
                # CRITICAL FIX: Force end_match with "unknown" result for in_game timeout
                # This ensures match statistics are not lost even if state detection fails
                if self.current_state == 'in_game' and self.match_controller:
                    logger.warning("[STATE] in_game timeout - forcing end_match with unknown result")
                    try:
                        self.match_controller.end_match("unknown")
                    except Exception as e:
                        logger.error(f"[STATE] Failed to end match on timeout: {e}")
                    if self.observability:
                        self.observability.record_match_result(
                            brawler=self.current_brawler or "unknown",
                            map_name=self._current_map or "unknown",
                            result="unknown",
                            duration=elapsed
                        )
                self.current_state = 'lobby'
                self.state_start_time = time.time()
                if self.match_controller:
                    logger.debug("[STATE] Resetando estado do MatchController devido a timeout")
                    self.match_controller.reset_match()

        if not hasattr(self, '_last_log_time'):
            self._last_log_time = 0
        if time.time() - self._last_log_time > 10:
            logger.info(f"Estado atual: {self.current_state}")
            self._last_log_time = time.time()

        if self.current_state in self.states:
            handler_name = self.states[self.current_state].__name__
            logger.info(f"[STATE] Executando handler: {handler_name} para estado {self.current_state}")
            logger.debug(f"[STATE] Handler disponível: {handler_name in self.states}")
            handler_start = time.time()
            try:
                if self.current_state in ('in_game', 'in_game_learning', 'unknown'):
                    self.states[self.current_state](image)
                else:
                    self.states[self.current_state]()
                handler_duration = time.time() - handler_start
                logger.info(f"[STATE] Handler {handler_name} concluído em {handler_duration:.2f}s")
            except Exception as e:
                logger.error(f"[STATE] Erro ao executar handler {handler_name}: {e}", exc_info=True)
                logger.error(f"[STATE] Estado atual: {self.current_state}")
                logger.error(f"[STATE] Image shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
        else:
            logger.warning(f"[STATE] Estado sem handler registrado: {self.current_state}")

        return self.current_state

    def _diag(self, message: str):
        """Emite logs diagnósticos apenas quando o modo está ativo."""
        if self.diagnostic_mode:
            logger.info(f"[STATE][DIAG] {message}")

    def _log_lobby_snapshot(self, context: str):
        """Mostra o último snapshot do lobby automator para localizar falhas rapidamente."""
        if not self.lobby or not hasattr(self.lobby, "get_diagnostic_report"):
            return

        try:
            report = self.lobby.get_diagnostic_report()
            logger.info(f"[STATE][DIAG] {context} lobby_snapshot={report}")
        except Exception as e:
            logger.debug(f"[STATE][DIAG] Falha ao ler snapshot do lobby: {e}")

    def _try_intelligent_recovery(self, image, elapsed: float):
        """
        Recovery inteligente quando o bot está preso em 'unknown'.
        Em vez de clicar no centro à sorte, analisa o screenshot e tenta
        ações específicas baseadas no que vê.
        """
        logger.warning(f"[STATE] Recovery: preso em 'unknown' por {elapsed:.1f}s")

        # Coordenadas dinâmicas (não hardcoded!)
        if self.movement and hasattr(self.movement, 'window_w'):
            w, h = self.movement.window_w, self.movement.window_h
        else:
            w, h = self._get_window_size()
        center_x, center_y = round(w / 2), round(h / 2)
        play_x, play_y = round(w * 0.9419), round(h * 0.8949)

        # Se temos o detector unificado, usar diagnóstico detalhado
        if self.unified_detector and image is not None:
            detection = self.unified_detector.detect(image)
            logger.info(f"[STATE] Recovery detection: state={detection.state}, "
                        f"conf={detection.confidence:.2f}, method={detection.method}, "
                        f"button={detection.button_coords}")

            if detection.state != "unknown" and detection.confidence > 0.2:
                # O detector encontrou algo — usar esse estado
                logger.info(f"[STATE] Recovery: estado recuperado = {detection.state}")
                self.current_state = detection.state
                self.unknown_since = None
                self.unknown_streak = 0
                return

            # Tentar usar find_play_button para localizar o botão visualmente
            if hasattr(self.unified_detector, 'find_play_button'):
                play_pos = self.unified_detector.find_play_button(image)
                if play_pos:
                    logger.info(f"[STATE] Recovery: Play button encontrado em {play_pos}")
                    if self.emulator_controller:
                        self.emulator_controller.tap_scaled(play_pos[0], play_pos[1])
                    time.sleep(1)
                    return

        # Se temos PopupManager, tentar detectar e fechar popups
        if self.lobby and hasattr(self.lobby, '_popup_manager') and image is not None:
            try:
                popup = self.lobby._popup_manager.detect_popup(image)
                if popup and popup.confidence > 0.25:
                    logger.info(f"[STATE] Recovery: popup detectado: {popup.popup_type}")
                    self.lobby._popup_manager.handle_popup(
                        popup,
                        click_func=self.lobby._click if hasattr(self.lobby, '_click') else lambda x, y: None,
                        key_func=self.lobby._key_press if hasattr(self.lobby, '_key_press') else lambda k: None
                    )
                    time.sleep(0.5)
                    return
            except Exception as e:
                logger.debug(f"[STATE] Recovery popup check failed: {e}")

        # Estratégia de recovery por tempo preso
        if elapsed < 15:
            # Curto: clicar no centro para fechar popup/recompensa
            logger.info("[STATE] Recovery: clicando centro para fechar popup")
            if self.emulator_controller:
                self.emulator_controller.tap_scaled(center_x, center_y)
            time.sleep(0.5)

        elif elapsed < 30:
            # Médio: tentar ESC + clicar Play
            logger.info("[STATE] Recovery: tentando ESC + Play")
            if self.emulator_controller:
                self.emulator_controller.keyevent(4)  # BACK
                time.sleep(0.5)
                self.emulator_controller.tap_scaled(play_x, play_y)
            time.sleep(1)

        elif elapsed < 60:
            # Longo: forçar reset para lobby
            logger.warning("[STATE] Recovery: forçando reset para lobby após 30s unknown")
            self.current_state = 'lobby'
            self.unknown_since = None
            self.unknown_streak = 0
            self.state_start_time = time.time()
            if self.match_controller:
                self.match_controller.reset_match()

        else:
            # Muito longo: reiniciar ciclo
            logger.error("[STATE] Recovery: unknown por >60s, reset completo")
            self.current_state = 'lobby'
            self.unknown_since = None
            self.unknown_streak = 0
            self.state_start_time = time.time()
            if self.match_controller:
                self.match_controller.reset_match()
            # Tentar pressionar Home no emulador
            if self.emulator_controller:
                self.emulator_controller.keyevent(3)  # HOME
                time.sleep(1)
                # Reabrir Brawl Stars (não implementado — requer ADB am start)

        # NOTA: NÃO resetamos unknown_since aqui para evitar loop infinito de recovery
        # O timer só é limpo quando o estado muda para algo que não 'unknown'

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

                # FORCE LOADING TIMEOUT: if in loading for >15s, skip detection and force in_game
                logger.info(f"[STATE] DEBUG: current={self.current_state}, start_time={self.state_start_time}")
                if self.current_state == 'loading':
                    if not self.state_start_time:
                        self.state_start_time = time.time()
                        logger.debug("[STATE] Initialized loading_start_time")
                    loading_elapsed = time.time() - self.state_start_time
                    if loading_elapsed > 15:
                        logger.warning(f"[STATE] FORCE: Loading timeout ({loading_elapsed:.0f}s), forcing in_game")
                        self.current_state = 'in_game'
                        self.state_start_time = time.time()
                        self._remember_known_state('in_game')
                        # Skip _process_cycle to avoid being reset back to loading
                        time.sleep(0.5)
                        continue
                
                self._process_cycle()
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

    def _wait_for_state(self, expected_state: str, timeout: float = 5.0, poll_interval: float = 0.5) -> bool:
        """Aguarda até o estado esperado aparecer antes de seguir com a automação."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                image = self.screenshot.take()
                screen_state_hint = None
                if self.screen_automation and hasattr(self.screen_automation, "get_current_state_name"):
                    try:
                        screen_state_hint = self.screen_automation.get_current_state_name()
                    except Exception:
                        screen_state_hint = None

                if image is not None and self.state_finder.get_state(image, screen_state_hint=screen_state_hint) == expected_state:
                    return True
            except Exception as e:
                logger.debug(f"[STATE] Erro ao aguardar estado '{expected_state}': {e}")
            time.sleep(poll_interval)
        return False

    def _handle_lobby(self):
        """No lobby - pressiona play com verificações proativas"""
        logger.info("[STATE] No lobby - a pressionar play")
        logger.info(f"[STATE] Lobby automator disponível: {self.lobby is not None}")
        self._diag("lobby_handler_start")

        # Verificar popups proativamente antes de tentar clicar no Play
        if self.lobby and hasattr(self.lobby, 'close_popup'):
            try:
                img = self._get_cached_screenshot()
                if img is not None:
                    from pylaai_real.unified_state_detector import UnifiedStateDetector
                    quick_det = UnifiedStateDetector(images_path=self.unified_detector.images_path if self.unified_detector else 'images')
                    det = quick_det.detect(img)
                    if det.state in ('popup', 'news', 'shop', 'tutorial'):
                        logger.info(f"[STATE] Detectado {det.state} no lobby, fechando antes de prosseguir")
                        self.lobby.close_popup()
                        time.sleep(random.uniform(0.5, 1.0))
                        self._diag(f"lobby_pre_popup_closed={det.state}")
            except Exception as e:
                logger.debug(f"[STATE] Verificação proativa de popup falhou: {e}")
        logger.info(f"[STATE] Emulator controller disponível: {self.emulator_controller is not None}")

        # Phase 10: LobbyFSM state tracking
        if self.lobby_fsm and HAS_LOBBY_FSM:
            try:
                current_fsm = self.lobby_fsm.get_state()
                if current_fsm.sub != LobbyState.PLAY_BUTTON:
                    self.lobby_fsm.transition(
                        TopLevelState.LOBBY, LobbyState.PLAY_BUTTON,
                        reason="entered_lobby_handler"
                    )
                # Stuck detection: if in PLAY_BUTTON too long, try recovery
                if self.lobby_fsm.is_stuck():
                    logger.warning("[STATE] LobbyFSM: stuck in PLAY_BUTTON, forcing state reset")
                    self.lobby_fsm.transition(
                        TopLevelState.LOBBY, LobbyState.IDLE,
                        reason="stuck_recovery"
                    )
                    # Try clicking harder or different area
                    if self.lobby and hasattr(self.lobby, '_force_play_click'):
                        self.lobby._force_play_click()
            except Exception as e:
                logger.debug(f"[STATE] LobbyFSM error: {e}")

        if self.screen_automation and hasattr(self.screen_automation, "get_current_state_name"):
            state_name = self.screen_automation.get_current_state_name()
            logger.info(f"[STATE] Screen automation state: {state_name}")
            self._diag(f"screen_automation_state={state_name}")

        if self.lobby is None:
            logger.warning("[STATE] Lobby automator não disponível, não é possível pressionar play")
            return

        # === NOVO: Learning Mode — ir diretamente para Training Cave ===
        if self.learning_mode_controller:
            logger.info("[STATE] Learning mode ativo — navegando para Training Cave")
            entered = self.learning_mode_controller.enter_training_cave()
            if entered:
                self.current_state = 'in_game_learning'
                self.state_start_time = time.time()
                self._diag("learning_mode_entered_training_cave")
                # Reset in_game flag to ensure clean state
                self._in_game_initialized = False
            else:
                logger.error("[STATE] Falha ao entrar na Training Cave; tentando fallback Play")
            return

        # === NOVO: Selecionar modo de jogo desejado antes de pressionar Play ===
        desired_mode = None
        if self.lobby and hasattr(self.lobby, 'queue') and self.lobby.queue:
            current = self.lobby.queue.get_current()
            if current:
                desired_mode = getattr(current, 'game_mode', None)
        if desired_mode and hasattr(self.lobby, 'select_game_mode'):
            try:
                mode_ok = self.lobby.select_game_mode(desired_mode)
                if mode_ok:
                    logger.info(f"[STATE] Modo de jogo selecionado: {desired_mode}")
                    self._diag(f"game_mode_selected={desired_mode}")
                else:
                    logger.warning(f"[STATE] Falha ao selecionar modo {desired_mode}, continuando com slot ativo")
            except Exception as e:
                logger.debug(f"[STATE] Erro ao selecionar modo: {e}")

        # === NOVO: Coletar recompensas automaticas no lobby ===
        if self.lobby and hasattr(self.lobby, 'collect_daily_rewards') and self.emulator_controller:
            try:
                screenshot = self.screenshot.take()
                if screenshot is not None:
                    collected = self.lobby.collect_daily_rewards(screenshot)
                    if collected:
                        logger.info("[STATE] Recompensas diarias coletadas")
                        self._diag("daily_rewards_collected=true")
                        time.sleep(0.5)
            except Exception as e:
                logger.debug(f"[STATE] Erro ao coletar recompensas diarias: {e}")

        if self.lobby and hasattr(self.lobby, 'collect_starr_road') and self.emulator_controller:
            try:
                screenshot = self.screenshot.take()
                if screenshot is not None:
                    collected = self.lobby.collect_starr_road(screenshot)
                    if collected:
                        logger.info("[STATE] Recompensas Starr Road coletadas")
                        self._diag("starr_road_collected=true")
                        time.sleep(0.5)
            except Exception as e:
                logger.debug(f"[STATE] Erro ao coletar Starr Road: {e}")

        if self.lobby and hasattr(self.lobby, 'collect_quest_rewards') and self.emulator_controller:
            try:
                screenshot = self.screenshot.take()
                if screenshot is not None:
                    collected = self.lobby.collect_quest_rewards(screenshot)
                    if collected:
                        logger.info("[STATE] Recompensas de missoes coletadas")
                        self._diag("quest_rewards_collected=true")
                        time.sleep(0.5)
            except Exception as e:
                logger.debug(f"[STATE] Erro ao coletar missoes: {e}")

        pressed = self.lobby.press_play()
        self._diag(f"press_play_result={pressed}")
        if not pressed:
            logger.warning("[STATE] press_play falhou; o fluxo parou no lobby")
            self._log_lobby_snapshot("press_play_failed")
            return

        # Reset MatchController se necessário para permitir novas partidas
        if self.match_controller:
            try:
                # Verificar se há partida pendente
                if hasattr(self.match_controller, 'current_match') and self.match_controller.current_match:
                    logger.warning("[STATE] MatchController tem partida pendente, resetando para permitir nova partida")
                    self.match_controller.reset_match()
            except Exception as e:
                logger.warning(f"[STATE] Falha ao verificar/resetar MatchController: {e}")

        if self.progress and hasattr(self.progress, "clear_last_result"):
            try:
                self.progress.clear_last_result()
            except Exception as e:
                logger.debug(f"[STATE] Falha ao limpar último resultado no início da partida: {e}")

        if self.match_controller and hasattr(self.lobby, "queue") and hasattr(self.lobby.queue, "get_current"):
            current = self.lobby.queue.get_current()
            if current:
                game_mode = getattr(current, "game_mode", None) or "showdown"
                if self.match_controller.start_match(game_mode, current.name):
                    self._diag(f"match_controller_start_match={current.name}")
                # Update current_brawler for dashboard
                self.current_brawler = current.name
                if self.play and hasattr(self.play, "set_current_game_mode"):
                    try:
                        self.play.set_current_game_mode(game_mode)
                    except Exception as e:
                        logger.debug(f"[STATE] Falha ao definir game mode no play logic: {e}")
        # Update current map for dashboard
        if self.movement and hasattr(self.movement, "current_map"):
            self._current_map = self.movement.current_map
        time.sleep(2)
        
        logger.info("[STATE] Handler lobby concluído")
        self._diag("lobby_handler_done")

    def _handle_brawler_selection(self):
        """Na seleção de brawler - seleciona, sai e inicia partida imediatamente"""
        logger.info("[STATE] Na seleção de brawler")
        self._diag("brawler_selection_start")

        # Phase 10: LobbyFSM state tracking
        if self.lobby_fsm and HAS_LOBBY_FSM:
            try:
                current_fsm = self.lobby_fsm.get_state()
                if current_fsm.sub != LobbyState.BRAWLER_SELECT:
                    self.lobby_fsm.transition(
                        TopLevelState.LOBBY, LobbyState.BRAWLER_SELECT,
                        reason="entered_brawler_selection"
                    )
                if self.lobby_fsm.is_stuck():
                    logger.warning("[STATE] LobbyFSM: stuck in BRAWLER_SELECT, forcing exit")
                    self.lobby_fsm.transition(
                        TopLevelState.LOBBY, LobbyState.IDLE,
                        reason="brawler_stuck_recovery"
                    )
            except Exception as e:
                logger.debug(f"[STATE] LobbyFSM error in brawler selection: {e}")

        # Intelligent brawler selection using BrawlerSelector
        if self.brawler_selector and hasattr(self.lobby, "queue") and hasattr(self.lobby.queue, "get_available_names"):
            try:
                available = self.lobby.queue.get_available_names()
                if available:
                    map_name = None
                    if self.movement and hasattr(self.movement, "current_map"):
                        map_name = self.movement.current_map
                    recommended = self.brawler_selector.select_brawler(available, map_name=map_name)
                    if recommended:
                        logger.info(f"[STATE] BrawlerSelector recomendou: {recommended} (mapa: {map_name})")
                        self._diag(f"brawler_selector_recommendation={recommended},map={map_name}")
                        if hasattr(self.lobby.queue, "set_current_by_name"):
                            self.lobby.queue.set_current_by_name(recommended)
            except Exception as e:
                logger.warning(f"[STATE] Falha ao usar BrawlerSelector: {e}")

        # Selecionar brawler atual (agora pode ser o recomendado)
        if self.lobby is None:
            logger.warning("[STATE] Lobby automator não disponível, não é possível selecionar brawler")
            return
        logger.info("[STATE] Selecionando brawler atual")
        selected = self.lobby.select_current_brawler(self.screenshot.take)
        self._diag(f"select_current_brawler_result={selected}")
        time.sleep(1)

        # Se falhou selecionar brawler, ainda tentar sair e entrar no jogo
        if not selected:
            logger.warning("[STATE] Falha ao selecionar brawler atual; tentando sair mesmo assim para entrar no jogo")
            self._log_lobby_snapshot("brawler_selection_failed")
            # Não return - continuar tentando sair da seleção

        # Sair da seleção de brawler
        logger.info("[STATE] Saindo da seleção de brawler")
        if self.emulator_controller:
            logger.info("[STATE] Usando emulator_controller para ESC")
            self.emulator_controller.keyevent(4) # ESC / BACK
        else:
            logger.info("[STATE] Usando pyautogui para ESC")
            if pyautogui:
                pyautogui.keyDown('esc')
                pyautogui.keyUp('esc')
        self._diag("returned_from_brawler_selection")
        
        time.sleep(1.5) # Esperar animação de volta ao lobby

        if not self._wait_for_state('lobby', timeout=4.0):
            logger.warning("[STATE] Lobby não foi confirmado após sair da seleção; tentando continuar com cautela")
            self._log_lobby_snapshot("lobby_not_confirmed_after_exit")

        # FORÇAR INÍCIO: Pressionar Play logo após sair da seleção
        if self.lobby is None:
            logger.warning("[STATE] Lobby automator não disponível após seleção")
            return
        logger.info("[STATE] Forçando início de partida após sair da seleção")
        # === NOVO: Coletar recompensas automaticas no lobby ===
        if self.lobby and hasattr(self.lobby, 'collect_daily_rewards') and self.emulator_controller:
            try:
                screenshot = self.screenshot.take()
                if screenshot is not None:
                    collected = self.lobby.collect_daily_rewards(screenshot)
                    if collected:
                        logger.info("[STATE] Recompensas diarias coletadas")
                        self._diag("daily_rewards_collected=true")
                        time.sleep(0.5)
            except Exception as e:
                logger.debug(f"[STATE] Erro ao coletar recompensas diarias: {e}")

        if self.lobby and hasattr(self.lobby, 'collect_starr_road') and self.emulator_controller:
            try:
                screenshot = self.screenshot.take()
                if screenshot is not None:
                    collected = self.lobby.collect_starr_road(screenshot)
                    if collected:
                        logger.info("[STATE] Recompensas Starr Road coletadas")
                        self._diag("starr_road_collected=true")
                        time.sleep(0.5)
            except Exception as e:
                logger.debug(f"[STATE] Erro ao coletar Starr Road: {e}")

        if self.lobby and hasattr(self.lobby, 'collect_quest_rewards') and self.emulator_controller:
            try:
                screenshot = self.screenshot.take()
                if screenshot is not None:
                    collected = self.lobby.collect_quest_rewards(screenshot)
                    if collected:
                        logger.info("[STATE] Recompensas de missoes coletadas")
                        self._diag("quest_rewards_collected=true")
                        time.sleep(0.5)
            except Exception as e:
                logger.debug(f"[STATE] Erro ao coletar missoes: {e}")

        pressed = self.lobby.press_play()
        self._diag(f"press_play_after_selection_result={pressed}")
        if not pressed:
            logger.warning("[STATE] press_play falhou após seleção; fluxo interrompido")
            self._log_lobby_snapshot("press_play_after_selection_failed")
            return
        time.sleep(2)
        
        logger.info("[STATE] Handler brawler selection concluído")
        self._diag("brawler_selection_handler_done")

    def _handle_loading(self):
        """Estado de loading - aguarda a transição para o jogo ou lobby."""
        logger.info("[STATE] Loading detectado - aguardando transição")
        self._diag("loading_handler_start")
        
        # Initialize loading timer if not set
        if not hasattr(self, '_loading_start_time') or self._loading_start_time is None:
            self._loading_start_time = time.time()
        
        # Check timeout: if loading for more than 15s, force to in_game
        elapsed = time.time() - self._loading_start_time
        if elapsed > 15:
            logger.warning(f"[STATE] Loading timeout ({elapsed:.0f}s), forcing transition to in_game")
            self._loading_start_time = None
            self.current_state = 'in_game'
            self._remember_known_state('in_game')
            self._diag("loading_handler_timeout_forced_in_game")
            return
        
        if self.screen_automation and hasattr(self.screen_automation, "get_current_state_name"):
            try:
                state_name = self.screen_automation.get_current_state_name()
                logger.info(f"[STATE] Loading hint via screen automation: {state_name}")
                self._diag(f"loading_screen_hint={state_name}")
            except Exception as e:
                logger.debug(f"[STATE] Falha ao ler hint de loading: {e}")
        time.sleep(1.0)
        self._diag("loading_handler_done")

    def _handle_matchmaking(self):
        """Estado de matchmaking - aguarda até a partida começar."""
        logger.info("[STATE] Matchmaking detectado - aguardando início da partida")
        self._diag("matchmaking_handler_start")

        # Verificar se a partida já começou (proactive detection)
        if self.state_start_time:
            elapsed = time.time() - self.state_start_time
            if elapsed > 20:
                logger.warning(f"[STATE] Matchmaking timeout ({elapsed:.0f}s > 20s) - forçando transição para in_game")
                self._diag(f"matchmaking_timeout_force_in_game={elapsed:.1f}")
                self.current_state = 'in_game'
                self.state_start_time = time.time()
                self._remember_known_state('in_game')
                return
            elif elapsed > 10:
                logger.info(f"[STATE] Matchmaking há {elapsed:.0f}s - verificando se partida começou")
                try:
                    img = self._get_cached_screenshot()
                    if img is not None and hasattr(self, 'unified_detector') and self.unified_detector:
                        result = self.unified_detector.detect(img)
                        if result.state in ('in_game', 'loading'):
                            logger.info(f"[STATE] Partida detectada via proactive check: {result.state}")
                            self._diag(f"matchmaking_proactive_detected={result.state}")
                            self.current_state = result.state
                            self.state_start_time = time.time()
                            return
                except Exception as e:
                    logger.debug(f"[STATE] Proactive matchmaking check falhou: {e}")

        # Map detection is now handled automatically in _process_cycle via screen automation hints
        # No need to set default map here anymore
        logger.debug("[STATE] Mapa será detectado automaticamente via screen automation hints")

        time.sleep(1.0)
        if self.screen_automation and hasattr(self.screen_automation, "get_current_state_name"):
            try:
                state_name = self.screen_automation.get_current_state_name()
                self._diag(f"matchmaking_screen_hint={state_name}")
            except Exception as e:
                logger.debug(f"[STATE] Falha ao ler hint de matchmaking: {e}")
        self._diag("matchmaking_handler_done")

    def _handle_connection_lost(self):
        """Estado de conexão perdida - tenta apenas estabilizar o fluxo."""
        logger.warning("[STATE] Connection lost detectado - tentando recuperar sem assumir gameplay")
        self._diag("connection_lost_handler_start")
        self._log_lobby_snapshot("connection_lost")
        if self.emulator_controller:
            try:
                self.emulator_controller.tap_scaled(960, 540)
                self._diag("connection_lost_center_tap")
            except Exception as e:
                logger.debug(f"[STATE] Falha ao clicar para recuperar conexão: {e}")
        time.sleep(1.0)
        self._diag("connection_lost_handler_done")

    def _handle_shop(self):
        """Na loja - coletar itens gratuitos e sair"""
        if self.lobby is None:
            logger.warning("[STATE] Lobby automator não disponível, não é possível sair da loja")
            return
        # === NOVO: Coletar itens gratuitos da loja ===
        if hasattr(self.lobby, 'collect_shop_items') and self.emulator_controller:
            try:
                screenshot = self.screenshot.take()
                if screenshot is not None:
                    collected = self.lobby.collect_shop_items(screenshot)
                    if collected:
                        logger.info("[STATE] Itens gratuitos da loja coletados")
                        self._diag("shop_free_items_collected=true")
                        time.sleep(0.5)
            except Exception as e:
                logger.debug(f"[STATE] Erro ao coletar itens da loja: {e}")
        self.lobby.quit_shop()

    def _handle_popup(self):
        """Com popup aberto - fechar"""
        if self.lobby is None:
            logger.warning("[STATE] Lobby automator não disponível, não é possível fechar popup")
            return
        self.lobby.close_popup()

    def _handle_in_game(self, image):
        """Durante a partida - executa lógica de jogo + RL online"""
        logger.info("[STATE] Em jogo - a executar lógica")
        logger.info(f"[STATE] Play logic disponível: {self.play is not None}")

        # Update current_brawler and _current_map for dashboard (real data)
        if self.lobby and hasattr(self.lobby, 'queue') and self.lobby.queue:
            current = self.lobby.queue.get_current()
            if current:
                self.current_brawler = current.name
        if self.movement and hasattr(self.movement, "current_map") and self.movement.current_map:
            self._current_map = self.movement.current_map

        # Reset tracker and combat state for new match (only on first entry to in_game)
        if self.play and not getattr(self, '_in_game_initialized', False):
            try:
                self.play.reset_for_new_match()
                logger.info("[STATE] Estado de combate resetado para nova partida")
                self._in_game_initialized = True
            except Exception as e:
                logger.warning(f"[STATE] Falha ao resetar estado de combate: {e}")

            # Iniciar episodio de RL com brawler e mapa atual
            if self.rl_engine:
                try:
                    brawler = self.lobby.queue.get_current().name if self.lobby and hasattr(self.lobby, 'queue') and self.lobby.queue else "unknown"
                    map_name = getattr(self, '_current_map', None)
                    self.rl_engine.start_episode(brawler, map_name)
                except Exception as e:
                    logger.debug(f"[STATE] Falha ao iniciar episodio RL: {e}")

        screenshot = self.screenshot.take()
        if screenshot is not None:
            logger.info(f"[STATE] Screenshot capturado: {screenshot.shape}")

            # === NOVO: Detectar Training Cave e PvE ===
            if self.lobby and hasattr(self.lobby, 'is_in_training_cave'):
                try:
                    in_cave = self.lobby.is_in_training_cave(screenshot)
                    if in_cave:
                        logger.info("[STATE] Training Cave detectada - modo de treino ativo")
                        self._diag("training_cave_active=true")
                except Exception as e:
                    logger.debug(f"[STATE] Erro ao detectar Training Cave: {e}")

            if self.lobby and hasattr(self.lobby, 'detect_pve'):
                try:
                    pve_result = self.lobby.detect_pve(screenshot=screenshot, game_mode=getattr(self, '_current_map', None))
                    if pve_result and getattr(pve_result, 'is_pve', False):
                        pve_type = getattr(pve_result, 'pve_type', 'unknown')
                        conf = getattr(pve_result, 'confidence', 0.0)
                        logger.info(f"[STATE] Partida PvE detectada: {pve_type} (conf={conf:.2f})")
                        self._diag(f"pve_detected={pve_type},confidence={conf:.2f}")
                        # Notificar play_logic sobre modo PvE para ajustar estrategia
                        if self.play and hasattr(self.play, 'set_pve_mode'):
                            self.play.set_pve_mode(pve_type)
                except Exception as e:
                    logger.debug(f"[STATE] Erro ao detectar PvE: {e}")

            result = self.play.play_round(screenshot)
            logger.info(f"[STATE] Play round resultado: {result}")
            
            # FALLBACK: Garantir que o bot nunca fica parado por muito tempo
            current_time = time.time()
            if not hasattr(self, '_last_combat_action_time'):
                self._last_combat_action_time = current_time
            
            # Verificar se houve ação real de combate
            action_taken = False
            if result and isinstance(result, dict):
                action_taken = result.get('attacked', False) or result.get('moved', False) or result.get('super_used', False)
            
            if action_taken:
                self._last_combat_action_time = current_time
                logger.info(f"[STATE] Ação de combate registrada: {result}")
                if self.improvement_system:
                    self.improvement_system.record_combat_action('attack' if result.get('attacked') else 'move')
            else:
                time_since_last_action = current_time - self._last_combat_action_time
                if time_since_last_action > 3.0:
                    logger.warning(f"[STATE] BOT PARADO há {time_since_last_action:.1f}s - FORÇANDO movimento de exploração")
                    # Movimento aleatório para explorar
                    if self.emulator_controller:
                        import random
                        # Mover para direção aleatória
                        angle = random.uniform(0, 2 * 3.14159)
                        distance = random.randint(100, 300)
                        # Centro do joystick + offset
                        joy_x, joy_y = 300, 900  # Coordenadas aproximadas do joystick
                        target_x = int(joy_x + distance * __import__('math').cos(angle))
                        target_y = int(joy_y + distance * __import__('math').sin(angle))
                        self.emulator_controller.swipe(joy_x, joy_y, target_x, target_y, duration=200)
                        logger.info(f"[STATE] Movimento de exploração: ({joy_x},{joy_y}) -> ({target_x},{target_y})")
                    self._last_combat_action_time = current_time

            # === RL ONLINE: aprender deste frame ===
            if self.rl_engine and self.play and hasattr(self.play, 'last_rl_transition') and self.play.last_rl_transition:
                try:
                    transition = self.play.last_rl_transition
                    # v2.3: Suporta tanto tupla (legacy) como dict (NeuralPolicy)
                    if isinstance(transition, dict):
                        self.rl_engine.learn_from_frame(
                            transition["state"],
                            transition["action"],
                            transition["reward"],
                            transition["next_state"],
                            player_pos=transition.get("player_pos"),
                            enemies=transition.get("enemies"),
                            detections=transition.get("detections"),
                        )
                    else:
                        state, action, reward, next_state = transition
                        self.rl_engine.learn_from_frame(state, action, reward, next_state)
                    logger.debug(f"[STATE] RL frame learned")
                except Exception as e:
                    logger.debug(f"[STATE] Falha ao aprender frame RL: {e}")

            # Record gameplay frame for dataset collection
            if self.data_collector is not None:
                try:
                    self.data_collector.record_frame(
                        screenshot=screenshot,
                        state="in_game",
                        action=result,
                    )
                except Exception as e:
                    logger.debug(f"[STATE] Falha ao gravar frame no data_collector: {e}")

            # Periodic reward update (survival time, match progression)
            if self.reward_bridge is not None:
                try:
                    self.reward_bridge.update_from_gameplay(
                        match_active=True,
                        elapsed_seconds=time.time() - self.state_start_time,
                    )
                except Exception as e:
                    logger.debug(f"[STATE] Falha ao atualizar reward_bridge: {e}")
        else:
            logger.warning("[STATE] Screenshot é None!")
        
        # Continuous Improvement: periodic check and save
        if self.improvement_system:
            try:
                self.improvement_system.periodic_save()
                adjustments = self.improvement_system.check_and_adjust()
                if adjustments:
                    logger.info(f"[IMPROVEMENT] Ajustes recomendados: {adjustments}")
            except Exception as e:
                logger.debug(f"[IMPROVEMENT] Erro no periodic check: {e}")
        
        time.sleep(0.1)

    def _handle_end_game(self):
        """No fim de uma partida - processar resultado e sair"""
        logger.debug("[STATE] Handler end_game iniciado")
        if log_manager:
            log_manager.log(
                message="Handler end_game iniciado",
                level="DEBUG",
                category="state",
                data={"action": "end_game_start"}
            )
        
        # Procurar resultado da partida
        found_result = False
        attempts = 0
        max_attempts = 3  # Timeout mais agressivo: apenas 3 tentativas
        screen_state_hint = None
        end_screen_start_time = time.time()
        max_end_screen_time = 15.0  # Timeout de 15 segundos no end screen

        if self.screen_automation and hasattr(self.screen_automation, "get_current_state_name"):
            try:
                screen_state_hint = self.screen_automation.get_current_state_name()
                logger.debug(f"[STATE] Screen automation hint em end_game: {screen_state_hint}")
            except Exception:
                screen_state_hint = None

        # Usar screen automation hint como primary para determinar quando sair do end screen
        while attempts < max_attempts:
            # Verificar timeout total no end screen
            if time.time() - end_screen_start_time > max_end_screen_time:
                logger.warning(f"[STATE] Timeout no end screen ({max_end_screen_time}s). Forçando reset para lobby.")
                if log_manager:
                    log_manager.log(
                        message=f"Timeout no end screen ({max_end_screen_time}s). Forçando reset para lobby.",
                        level="WARNING",
                        category="state",
                        data={"action": "end_screen_timeout", "timeout_seconds": max_end_screen_time}
                    )
                self.current_state = 'lobby'
                self.state_start_time = time.time()
                if self.match_controller:
                    self.match_controller.reset_match()
                break

            logger.info(f"[STATE] Tentativa {attempts + 1}/{max_attempts} para sair do end screen")

            # === NOVO: PlayAgainHandler inteligente (primeira tentativa) ===
            if self.lobby and hasattr(self.lobby, 'handle_end_screen_expanded') and attempts == 0:
                try:
                    screenshot = self.screenshot.take()
                    if screenshot is not None:
                        result = self.lobby.handle_end_screen_expanded(screenshot, window_size=self._get_window_size())
                        if result and getattr(result, 'success', False):
                            method = getattr(result, 'method_used', 'unknown')
                            logger.info(f"[STATE] PlayAgainHandler sucesso via: {method}")
                            self._diag(f"play_again_handler_success={method}")
                            if getattr(result, 'clicked_play_again', False):
                                logger.info("[STATE] Play Again clicado - reentrada rapida no mesmo modo")
                                time.sleep(2.0)
                                # Verificar se ja saiu do end screen
                                verify = self.screenshot.take()
                                if verify is not None:
                                    vstate = self.state_finder.get_state(verify)
                                    if vstate != 'end':
                                        logger.info("[STATE] Confirmado: saiu do end screen via Play Again")
                                        break
                except Exception as e:
                    logger.debug(f"[STATE] PlayAgainHandler falhou: {e}")

            # Verificar screen automation hint primeiro
            current_hint = None
            if self.screen_automation and hasattr(self.screen_automation, "get_current_state_name"):
                try:
                    current_hint = self.screen_automation.get_current_state_name()
                    logger.info(f"[STATE] Screen automation current hint: {current_hint}")
                except Exception as e:
                    logger.debug(f"[STATE] Falha ao ler hint de end screen: {e}")
            
            # Melhorar sincronização: Priorizar screen automation se for confiável
            # Se hint for 'idle', 'lobby', 'play', etc. (não end/loading), assumir que saiu
            if current_hint and current_hint not in ['end', 'loading', 'proceed']:
                logger.info(f"[STATE] Screen automation indica saída do end screen: {current_hint}")
                if log_manager:
                    log_manager.log(
                        message=f"Screen automation indica saída do end screen: {current_hint}",
                        level="INFO",
                        category="state",
                        data={"action": "screen_automation_exit", "hint": current_hint}
                    )
                time.sleep(0.5)  # Delay menor para resposta mais rápida
                # Verificar novamente com screenshot para confirmar
                try:
                    verify_screenshot = self.screenshot.take()
                    if verify_screenshot is not None:
                        verify_state = self.state_finder.get_state(verify_screenshot, screen_state_hint=current_hint)
                        logger.info(f"[STATE] Estado verificado após delay: {verify_state}")
                        if verify_state != 'end':
                            logger.info("[STATE] Confirmado saída do end screen via template matching")
                            if log_manager:
                                log_manager.log(
                                    message="Confirmado saída do end screen via template matching",
                                    level="INFO",
                                    category="state",
                                    data={"action": "end_screen_exit_confirmed", "verify_state": verify_state}
                                )
                            break
                        else:
                            logger.warning("[STATE] Screen automation disse que saiu, mas template ainda detecta end. Continuando tentativas.")
                            if log_manager:
                                log_manager.log(
                                    message="Screen automation disse que saiu, mas template ainda detecta end",
                                    level="WARNING",
                                    category="state",
                                    data={"action": "template_mismatch", "screen_hint": current_hint, "template_state": verify_state}
                                )
                except Exception as e:
                    logger.warning(f"[STATE] Falha ao verificar estado: {e}")
            
            # Se hint for 'proceed', tentar clicar uma vez e verificar
            if current_hint == 'proceed':
                logger.info("[STATE] Screen automation detecta 'proceed', tentando clicar")
                if log_manager:
                    log_manager.log(
                        message="Screen automation detecta 'proceed', tentando clicar",
                        level="INFO",
                        category="state",
                        data={"action": "proceed_detected", "click_position": [960, 950]}
                    )
                if self.emulator_controller:
                    self.emulator_controller.tap_scaled(960, 950)
                    time.sleep(0.5)
            
            if not found_result:
                try:
                    screenshot = self.screenshot.take()
                    if screenshot is not None:
                        found_result = self.progress.find_game_result(screenshot)
                        if np is not None and isinstance(found_result, np.ndarray):
                            # Se retornou array, verificar se não está vazio
                            found_result = bool(found_result.size > 0)
                        if found_result:
                            logger.info("[STATE] Resultado da partida encontrado")
                            if log_manager:
                                log_manager.log(
                                    message="Resultado da partida encontrado",
                                    level="INFO",
                                    category="state",
                                    data={"action": "result_found"}
                                )
                except Exception as e:
                    logger.warning(f"[STATE] Falha ao capturar screenshot para resultado: {e}")

            # Estratégia agressiva de saída: múltiplas tentativas
            if self.emulator_controller:
                logger.debug("[STATE] Tentando sair do end screen via ADB")
                # Tentativa 1: Clicar no centro inferior (botão continuar)
                self.emulator_controller.tap_scaled(960, 950)
                time.sleep(0.2)  # Delay menor
                # Tentativa 2: ESC/BACK key (mais eficiente que cliques múltiplos)
                self.emulator_controller.keyevent(4)
                time.sleep(0.3)
                # Tentativa 3: Clicar no canto superior direito (X ou fechar)
                self.emulator_controller.tap_scaled(1800, 100)
                time.sleep(0.2)
            else:
                logger.debug("[STATE] Tentando sair via pyautogui")
                if pyautogui:
                    pyautogui.keyDown('esc')
                    pyautogui.keyUp('esc')
                time.sleep(0.3)
            
            time.sleep(1.0)  # Esperar transição
            attempts += 1

        # Após tentativas, forçar reset para lobby
        logger.warning(f"[STATE] {attempts} tentativas para sair do end screen, forçando reset para lobby")
        if log_manager:
            log_manager.log(
                message=f"{attempts} tentativas para sair do end screen, forçando reset para lobby",
                level="WARNING",
                category="state",
                data={"action": "end_screen_force_reset", "attempts": attempts}
            )

        # Se OCR falhar, usar o último resultado apenas quando o hint ainda confirma fim de partida.
        fallback_result = None
        if not found_result and self.progress and hasattr(self.progress, "get_last_result"):
            hinted_end = False
            if self.state_finder and hasattr(self.state_finder, "_state_from_hint"):
                try:
                    hint_res = self.state_finder._state_from_hint(screen_state_hint)
                    hinted_state = hint_res[0] if isinstance(hint_res, tuple) else hint_res
                    hinted_end = hinted_state == 'end'
                except Exception:
                    hinted_end = False

            if hinted_end:
                fallback_result = self.progress.get_last_result()
                if fallback_result in {"win", "loss", "draw"}:
                    logger.warning("[STATE] OCR falhou, mas o hint de fim de partida permitiu reaproveitar o último resultado detectado")
                    found_result = True

        # Finalize match result BEFORE resetting match state
        match_result = None
        if found_result and self.match_controller and hasattr(self.progress, "get_last_result"):
            result = fallback_result or self.progress.get_last_result()
            if result in {"win", "loss", "draw"}:
                match_result = self.match_controller.end_match(result)
                self._diag(f"match_controller_end_match={result}")

                # Integrate with reward bridge for end-of-match reward calculation
                if self.reward_bridge is not None:
                    try:
                        win = result == "win"
                        draw = result == "draw"
                        survival_time = time.time() - self.state_start_time
                        self.reward_bridge.update_from_gameplay(
                            match_active=False,
                            win=win,
                            draw=draw,
                            survival_time=survival_time,
                        )
                        summary = self.reward_bridge.get_session_summary()
                        logger.info(f"[STATE] Reward summary: {summary}")
                        self._diag(f"reward_summary={summary}")
                    except Exception as e:
                        logger.warning(f"[STATE] Falha ao calcular reward final: {e}")

                # === RL ONLINE: finalizar episodio e atualizar ELO ===
                if self.rl_engine is not None:
                    try:
                        self.rl_engine.end_episode(
                            result=result,
                            rank=1 if result == "win" else (2 if result == "draw" else 10),
                        )
                        logger.info(f"[STATE] RL episodio finalizado: {result}")
                    except Exception as e:
                        logger.warning(f"[STATE] Falha ao finalizar episodio RL: {e}")

                # Flush data collector session
                if self.data_collector is not None:
                    try:
                        self.data_collector.end_episode(result=result)
                        logger.info("[STATE] Sessão de data collector finalizada")
                    except Exception as e:
                        logger.debug(f"[STATE] Falha ao finalizar data_collector: {e}")

                # Observability: record match result
                if self.observability is not None:
                    try:
                        map_name = None
                        if self.movement and hasattr(self.movement, "current_map"):
                            map_name = self.movement.current_map
                        brawler_name = None
                        if self.lobby and hasattr(self.lobby, "queue") and self.lobby.queue.get_current():
                            brawler_name = self.lobby.queue.get_current().name
                        self.observability.record_match_result(
                            result=result,
                            brawler=brawler_name,
                            map_name=map_name,
                        )
                        if self.reward_bridge:
                            summary = self.reward_bridge.get_session_summary()
                            if summary:
                                self.observability.record_reward(summary.get("last_reward", 0.0))
                        self.observability.save_to_disk()
                    except Exception as e:
                        logger.debug(f"[STATE] Falha ao registrar observabilidade: {e}")

                # Premium: Record match in BrawlerStatsTracker & TrophyTracker
                if hasattr(self, '_dashboard_bridge') and self._dashboard_bridge is not None:
                    try:
                        bname = brawler_name or "unknown"
                        mname = map_name or "unknown"
                        duration = time.time() - self.state_start_time
                        self._dashboard_bridge.brawler_tracker.record_match(
                            bname, mname, result, duration=duration
                        )
                        total = self._dashboard_bridge.brawler_tracker.get_total_trophies()
                        self._dashboard_bridge.trophy_tracker.record(total, bname)
                        analysis = self._dashboard_bridge.match_analyzer.analyze_match(
                            bname, mname, result, duration=duration
                        )
                        self._dashboard_bridge.update(
                            match_analysis=analysis,
                            recent_matches=[analysis],
                            coach_tips=self._dashboard_bridge.match_analyzer.get_coach_tips(bname),
                        )
                    except Exception as e:
                        logger.debug(f"[STATE] Premium tracking error: {e}")

        # Now reset state to lobby
        self.current_state = 'lobby'
        self.state_start_time = time.time()
        if self.match_controller:
            logger.debug("[STATE] Resetando estado do MatchController após end screen")
            self.match_controller.reset_match()
        
        logger.info("[STATE] Handler end_game concluído com reset forçado")

        if not found_result:
            logger.warning("Não foi possível ler resultado da partida")
            return

        # Se não houver match_controller, manter a atualização direta do progresso.
        if not self.match_controller:
            current = self.lobby.queue.get_current() if self.lobby and hasattr(self.lobby, "queue") else None
            if current:
                stats = self.progress.get_stats()
                current.current_trophies = stats['trophies']
                current.current_wins = stats['wins']

    def _handle_unknown(self, image):
        """Estado desconhecido - diagnosticar, aguardar, e forçar reset se persistir."""
        self._diag("unknown_handler_start")
        if self.state_finder and hasattr(self.state_finder, "get_diagnostic_report"):
            try:
                logger.info(f"[STATE][DIAG] unknown_state_snapshot={self.state_finder.get_diagnostic_report()}")
            except Exception as e:
                logger.debug(f"[STATE][DIAG] Falha ao ler diagnóstico de estado: {e}")

        # Check if we've been in unknown too long - force reset
        if self.unknown_since:
            unknown_elapsed = time.time() - self.unknown_since
            max_unknown = self.state_timeouts.get('unknown', 60)
            if unknown_elapsed > max_unknown:
                logger.error(f"[STATE] Unknown state timeout ({unknown_elapsed:.0f}s > {max_unknown}s), forcing reset to lobby")
                self.current_state = 'lobby'
                self.unknown_since = None
                self.unknown_streak = 0
                return

        logger.warning("[STATE] Estado desconhecido - sem acao agressiva para evitar suposicoes erradas")
        time.sleep(0.5)
        self._diag("unknown_handler_done")

    # === NOVOS HANDLERS PARA ESTADOS ADICIONAIS (ANALISE_PROFUNDA) ===

    def _get_window_size(self) -> tuple:
        """Helper para obter dimensoes da janela de forma segura."""
        # Priority 1: ResolutionManager (fonte centralizada e atualizada)
        if HAS_RESOLUTION_MANAGER:
            try:
                rm = ResolutionManager()
                rm.detect()
                if rm.profile.is_reasonable():
                    return rm.actual_resolution
            except Exception as e:
                logger.debug(f"[STATE] ResolutionManager fallback: {e}")
        # Priority 2: MovementEngine
        if self.movement and hasattr(self.movement, 'window_w'):
            return (self.movement.window_w, self.movement.window_h)
        # Priority 3: Screenshot
        try:
            img = self.screenshot.take()
            if img is not None and hasattr(img, 'shape'):
                h, w = img.shape[:2]
                return (w, h)
        except Exception as e:
            logger.error(f"[STATE] _get_window_size failed: {e}")
        return (1920, 1080)  # Fallback canonico

    def _handle_tutorial(self):
        """Tutorial detectado - tentar passar rapidamente clicando na area indicada."""
        self._diag("tutorial_handler_start")
        logger.info("[STATE] Tutorial detectado - tentando passar...")
        w, h = self._get_window_size()
        if self.emulator_controller:
            try:
                # Clicar no centro inferior onde geralmente ha setas de tutorial
                self.emulator_controller.tap_scaled(round(w * 0.5), round(h * 0.85))
                time.sleep(0.5)
                # Tentar clicar no centro tambem (alguns tutoriais usam centro)
                self.emulator_controller.tap_scaled(round(w * 0.5), round(h * 0.5))
                time.sleep(0.3)
                self._diag("tutorial_tap_attempted")
            except Exception as e:
                logger.debug(f"[STATE] Erro ao passar tutorial: {e}")
        # Tentar voltar para lobby com ESC/BACK
        self._safe_back_to_lobby()
        self._diag("tutorial_handler_done")

    def _handle_news(self):
        """News/Brawl Talk detectado - fechar clicando no X."""
        self._diag("news_handler_start")
        logger.info("[STATE] News/Brawl Talk detectado - fechando...")
        w, h = self._get_window_size()
        if self.emulator_controller:
            try:
                # Clicar no X do canto superior direito
                self.emulator_controller.tap_scaled(round(w * 0.97), round(h * 0.065))
                time.sleep(0.5)
                # Tentar ESC/BACK como fallback
                self.emulator_controller.keyevent(4)
                time.sleep(0.3)
                self._diag("news_closed")
            except Exception as e:
                logger.debug(f"[STATE] Erro ao fechar news: {e}")
        # Forcar transicao para lobby
        self.current_state = 'lobby'
        self.state_start_time = time.time()
        self._diag("news_handler_done")

    def _handle_brawler_unlock(self):
        """Tela de desbloqueio de brawler - clicar em Proceed/Continue."""
        self._diag("brawler_unlock_handler_start")
        logger.info("[STATE] Brawler unlock detectado - prosseguindo...")
        w, h = self._get_window_size()
        if self.emulator_controller:
            try:
                # Clicar no botao Proceed (area inferior direita tipica)
                self.emulator_controller.tap_scaled(round(w * 0.81), round(h * 0.92))
                time.sleep(1.0)
                # Tentar tambem no centro (algumas telas usam botao central)
                self.emulator_controller.tap_scaled(round(w * 0.50), round(h * 0.90))
                time.sleep(0.5)
                self._diag("brawler_unlock_proceed_clicked")
            except Exception as e:
                logger.debug(f"[STATE] Erro no brawler unlock: {e}")
        # Voltar para lobby
        self._safe_back_to_lobby()
        self._diag("brawler_unlock_handler_done")

    def _handle_season_reset(self):
        """Season reset detectado - clicar em Proceed para continuar."""
        self._diag("season_reset_handler_start")
        logger.info("[STATE] Season reset detectado - prosseguindo...")
        w, h = self._get_window_size()
        if self.emulator_controller:
            try:
                # Clicar no botao Proceed (area inferior central)
                self.emulator_controller.tap_scaled(round(w * 0.50), round(h * 0.90))
                time.sleep(1.0)
                # Tentar tambem na area de Proceed conhecida
                self.emulator_controller.tap_scaled(round(w * 0.81), round(h * 0.92))
                time.sleep(0.5)
                self._diag("season_reset_proceed_clicked")
            except Exception as e:
                logger.debug(f"[STATE] Erro no season reset: {e}")
        # Voltar para lobby
        self._safe_back_to_lobby()
        self._diag("season_reset_handler_done")


    def _handle_event_screen(self):
        """Tela de evento (Starrnova, etc.) - clicar em GOT IT! ou centro."""
        self._diag("event_screen_handler_start")
        logger.info("[STATE] Event screen detectado - fechando...")
        w, h = self._get_window_size()
        if self.emulator_controller:
            try:
                # Tentar clicar no botao verde (GOT IT!) na parte inferior central
                self.emulator_controller.tap_scaled(round(w * 0.50), round(h * 0.82))
                time.sleep(0.5)
                # Fallback: clicar no centro
                self.emulator_controller.tap_scaled(round(w * 0.50), round(h * 0.50))
                time.sleep(0.3)
                self._diag("event_screen_tapped")
            except Exception as e:
                logger.debug(f"[STATE] Erro ao fechar event screen: {e}")
        self.current_state = 'lobby'
        self.state_start_time = time.time()
        self._diag("event_screen_handler_done")

    def _handle_starr_drop(self):
        """Starr Drop - abrir clicando no centro, depois fechar."""
        self._diag("starr_drop_handler_start")
        logger.info("[STATE] Starr Drop detectado - abrindo...")
        w, h = self._get_window_size()
        if self.emulator_controller:
            try:
                # Clicar no centro para abrir
                self.emulator_controller.tap_scaled(round(w * 0.50), round(h * 0.50))
                time.sleep(1.5)
                # Tentar clicar em Claim/Collect se aparecer
                self.emulator_controller.tap_scaled(round(w * 0.50), round(h * 0.85))
                time.sleep(0.5)
                # ESC para fechar
                self.emulator_controller.keyevent(4)
                time.sleep(0.3)
                self._diag("starr_drop_handled")
            except Exception as e:
                logger.debug(f"[STATE] Erro no starr drop: {e}")
        self.current_state = 'lobby'
        self.state_start_time = time.time()
        self._diag("starr_drop_handler_done")

    def _handle_in_game_learning(self, image):
        """Modo teste aprendizagem na Training Cave — executa frame-a-frame com métricas."""
        logger.debug("[STATE] Handler in_game_learning iniciado")

        if self.learning_mode_controller is None:
            logger.error("[STATE] learning_mode_controller não disponível — saindo para lobby")
            self.current_state = 'lobby'
            return

        # Inicializar match na primeira entrada
        if not getattr(self, '_learning_match_initialized', False):
            brawler = "unknown"
            if self.lobby and hasattr(self.lobby, 'queue') and self.lobby.queue:
                current = self.lobby.queue.get_current()
                if current:
                    brawler = current.name
            self.learning_mode_controller.start_match(brawler)
            self._learning_match_initialized = True
            if self.play and hasattr(self.play, 'set_pve_mode'):
                try:
                    self.play.set_pve_mode('training_cave')
                except Exception as e:
                    logger.debug(f"[STATE] Falha ao definir PvE mode: {e}")

        # Executar um frame de gameplay
        if image is not None:
            self.learning_mode_controller.run_frame(image)

        # Verificar se a partida terminou (morte, timeout, tela de restart)
        ended, reason = self.learning_mode_controller.is_match_ended(image)
        if ended:
            self.learning_mode_controller.end_match(reason)
            self._learning_match_initialized = False
            if self.learning_mode_controller.should_continue():
                logger.info("[STATE] Reiniciando treino na Training Cave (%s/%s)",
                            self.learning_mode_controller._current_match_index,
                            self.learning_mode_controller.max_matches)
                self.learning_mode_controller.restart_training()
                self.state_start_time = time.time()
                # Start next match immediately on next cycle
                self._learning_match_initialized = False
            else:
                logger.info("[STATE] Modo de aprendizagem concluído (%s partidas). Voltando ao lobby.",
                            self.learning_mode_controller.max_matches)
                self.learning_mode_controller.exit_training_cave()
                self.current_state = 'lobby'
                self.state_start_time = time.time()
                return

        time.sleep(0.1)
        logger.debug("[STATE] Handler in_game_learning concluído")

    def _safe_back_to_lobby(self, max_attempts: int = 3):
        """Helper: tentar voltar para lobby de forma segura usando ESC/BACK."""
        for i in range(max_attempts):
            if self.emulator_controller:
                try:
                    self.emulator_controller.keyevent(4)  # BACK
                    time.sleep(0.4)
                except Exception as e:
                    logger.debug(f"[STATE] BACK key failed: {e}")
            # Verificar se ja chegou ao lobby
            if self.current_state == 'lobby':
                break
