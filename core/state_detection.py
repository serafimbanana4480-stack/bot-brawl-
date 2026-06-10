"""
state_detection.py

State detection logic extracted from state_manager.py.
Provides StateDetectionMixin with screenshot caching, state detection,
and intelligent recovery.
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

try:
    from ..realtime_logs import get_log_manager
    log_manager = get_log_manager()
except ImportError:
    log_manager = None


class StateDetectionMixin:
    """Mixin providing state detection, screenshot caching, and recovery."""

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
                # FIX: Permitir lobby/loading/matchmaking após tempo mínimo em jogo
                # FIX: Bloquear in_game -> loading a menos que screen_state_hint confirme (evita falsos positivos durante gameplay)
                elif self.current_state == 'in_game' and detected_state not in ('end', 'connection_lost', 'unknown', 'lobby', 'loading', 'matchmaking'):
                    time_in_game = time.time() - self.state_start_time if self.state_start_time else 0
                    if time_in_game < self._in_game_min_duration:
                        logger.warning(f"[STATE] BLOCKED: in_game -> {detected_state} (apenas {time_in_game:.1f}s em jogo, mínimo {self._in_game_min_duration}s)")
                        detected_state = self.current_state
                    else:
                        logger.warning(f"[STATE] BLOCKED: in_game -> {detected_state} (só permitido: end, connection_lost, unknown, lobby, loading, matchmaking)")
                        detected_state = self.current_state
                elif self.current_state == 'in_game' and detected_state == 'loading':
                    # CRITICAL: Nunca permitir in_game -> loading a menos que screen_state_hint confirme
                    # AND unless forced in_game timeout has expired
                    forced_time = getattr(self, '_forced_in_game_time', 0)
                    if forced_time and (time.time() - forced_time) < 30:
                        logger.warning(f"[STATE] BLOCKED: in_game -> loading (forçado há {time.time() - forced_time:.0f}s, bloqueado por 30s)")
                        detected_state = self.current_state
                    elif screen_state_hint and screen_state_hint in ('loading', 'detecting'):
                        logger.info(f"[STATE] Permitindo in_game -> loading (screen_state_hint={screen_state_hint} confirmou)")
                    else:
                        logger.warning(f"[STATE] BLOCKED: in_game -> loading (screen_state_hint={screen_state_hint}, não confirmado - provável falso positivo)")
                        detected_state = self.current_state

                elif self.current_state == 'loading' and detected_state == 'lobby':
                    # CRITICAL: Bloquear loading -> lobby a menos que tempo em loading seja muito longo (>20s)
                    # ou seja um erro genuíno. Isto evita que o detector volte ao lobby prematuramente
                    # durante o carregamento da partida.
                    time_in_loading = time.time() - self.state_start_time if self.state_start_time else 0
                    if time_in_loading < 20:
                        logger.warning(f"[STATE] BLOCKED: loading -> lobby (apenas {time_in_loading:.1f}s em loading, mínimo 20s)")
                        detected_state = self.current_state
                    else:
                        logger.info(f"[STATE] Permitindo loading -> lobby após {time_in_loading:.1f}s (possível erro/cancelamento)")

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
            # FIX: Nao sobrescrever estado se handler o forcou (ex: matchmaking -> in_game)
            if getattr(self, '_handler_forced_state', False):
                logger.info(f"[STATE] Handler forcou estado, ignorando deteccao: {self.current_state}")
                self._handler_forced_state = False
            else:
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

        # === GLOBAL STUCK DETECTION ===
        # Se estamos no mesmo estado há muito tempo e o handler não resolveu,
        # forçar uma ação de recovery autónoma
        if self.state_start_time:
            stuck_elapsed = time.time() - self.state_start_time
            if stuck_elapsed > 18 and self.current_state in ('lobby', 'matchmaking', 'loading', 'brawler_selection', 'end', 'unknown'):
                logger.warning(f"[STATE] STUCK DETECTION: {self.current_state} há {stuck_elapsed:.0f}s - forçando recovery")
                self._diag(f"stuck_recovery={self.current_state},elapsed={stuck_elapsed:.1f}")
                if self.current_state == 'lobby':
                    # Se preso no lobby, tentar clicar Play diretamente nas coordenadas padrão
                    self._force_click_play()
                elif self.current_state == 'matchmaking':
                    # Se preso em matchmaking, forçar in_game imediatamente
                    self.current_state = 'in_game'
                    self.state_start_time = time.time()
                    self._forced_in_game_time = time.time()
                    self._matchmaking_enter_time = None
                    return
                elif self.current_state == 'loading':
                    # Se preso em loading, forçar in_game
                    self.current_state = 'in_game'
                    self.state_start_time = time.time()
                    self._forced_in_game_time = time.time()
                    return
                elif self.current_state == 'brawler_selection':
                    # Se preso em brawler selection, tentar confirmar e voltar ao lobby
                    self.current_state = 'lobby'
                    self.state_start_time = time.time()
                    return
                elif self.current_state == 'end':
                    # Se preso no end screen, forçar lobby
                    logger.warning("[STATE] STUCK: end screen, forçando lobby")
                    self.current_state = 'lobby'
                    self.state_start_time = time.time()
                    if hasattr(self, '_matchmaking_enter_time'):
                        self._matchmaking_enter_time = None
                    return
                elif self.current_state == 'unknown':
                    # Se preso em unknown, forçar lobby
                    logger.warning("[STATE] STUCK: unknown, forçando lobby")
                    self.current_state = 'lobby'
                    self.state_start_time = time.time()
                    self.unknown_since = None
                    return

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
        play_x, play_y = round(w * 0.9119), round(h * 0.9122)

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

    def _force_click_play(self):
        """Força um clique no botão Play usando coordenadas dinâmicas - último recurso autónomo."""
        logger.warning("[STATE] Forçando clique no botão Play (stuck recovery)")
        try:
            if self.movement and hasattr(self.movement, 'window_w'):
                w, h = self.movement.window_w, self.movement.window_h
            else:
                w, h = self._get_window_size()
            play_x = int(w * 0.9119)
            play_y = int(h * 0.9122)
            if self.emulator_controller and hasattr(self.emulator_controller, 'tap'):
                self.emulator_controller.tap(play_x, play_y)
                logger.info(f"[STATE] Clique forçado em Play: ({play_x}, {play_y})")
            elif hasattr(self, '_click') and self._click:
                self._click(play_x, play_y)
                logger.info(f"[STATE] Clique forçado em Play via _click: ({play_x}, {play_y})")
            else:
                logger.warning("[STATE] Não foi possível forçar clique - sem controller disponível")
        except Exception as e:
            logger.error(f"[STATE] Erro ao forçar clique Play: {e}")

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

