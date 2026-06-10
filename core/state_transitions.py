"""
state_transitions.py

State transition handlers extracted from state_manager.py.
Provides StateTransitionsMixin with all state-specific handlers.
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


class StateTransitionsMixin:
    """Mixin providing all state-specific transition handlers."""

    def _handle_lobby(self):
        """No lobby - pressiona play com verificações proativas e recovery autónomo."""
        logger.info("[STATE] No lobby - a pressionar play")
        logger.info(f"[STATE] Lobby automator disponível: {self.lobby is not None}")
        self._diag("lobby_handler_start")

        # Se estamos no lobby há muito tempo, limpar estado de matchmaking para permitir novo ciclo
        if hasattr(self, '_matchmaking_enter_time') and self._matchmaking_enter_time is not None:
            self._matchmaking_enter_time = None

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

        # SEMPRE tentar clicar no Play, mesmo sem lobby automator
        if self.lobby is None:
            logger.warning("[STATE] Lobby automator não disponível - usando clique direto nas coordenadas padrão")
            self._force_click_play()
            time.sleep(0.8)
            self.current_state = 'loading'
            self.state_start_time = time.time()
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
            logger.warning("[STATE] press_play falhou; tentando fallback inteligente")
            self._log_lobby_snapshot("press_play_failed")
            # FALLBACK INTELIGENTE: usar coordenadas do detector se disponíveis
            if self.emulator_controller:
                try:
                    fallback_coords = None
                    
                    # Estratégia 1: Usar coordenadas detetadas pelo UnifiedStateDetector
                    if self.unified_detector:
                        img = self._get_cached_screenshot()
                        if img is not None:
                            det = self.unified_detector.detect(img)
                            if det.state == 'lobby' and det.button_coords and det.confidence > 0.2:
                                fallback_coords = det.button_coords
                                logger.info(f"[STATE] Fallback usando coordenadas do detector: {fallback_coords}")
                    
                    # Estratégia 2: Usar SmartPlayButtonDetector diretamente
                    if fallback_coords is None:
                        from pylaai_real.lobby_navigator import SmartPlayButtonDetector
                        play_det = SmartPlayButtonDetector(
                            self.unified_detector.images_path if self.unified_detector else 'images'
                        )
                        img = self._get_cached_screenshot()
                        if img is not None:
                            play_result = play_det.find_play_button(img)
                            if play_result.found and play_result.coords:
                                fallback_coords = play_result.coords
                                logger.info(f"[STATE] Fallback usando SmartPlayButtonDetector: {fallback_coords}")
                    
                    # Estratégia 3: Coordenadas hardcoded (último recurso)
                    if fallback_coords is None:
                        w, h = self._get_window_size()
                        fallback_coords = (round(w * 0.9119), round(h * 0.9122))
                        logger.info(f"[STATE] Fallback usando coordenadas hardcoded: {fallback_coords}")
                    
                    self.emulator_controller.tap_scaled(*fallback_coords)
                    logger.info(f"[STATE] Fallback clicando em: {fallback_coords}")
                    time.sleep(1.5)
                except Exception as e:
                    logger.warning(f"[STATE] Fallback inteligente falhou: {e}")
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

        # Registar a partida antes de abandonar o lobby.
        # O return abaixo é necessário para evitar repetir a pressão no Play
        # no mesmo ciclo, mas o match deve ser marcado antes disso.
        if self.match_controller and hasattr(self.lobby, "queue") and hasattr(self.lobby.queue, "get_current"):
            current = self.lobby.queue.get_current()
            if current:
                game_mode = getattr(current, "game_mode", None) or "showdown"
                if self.match_controller.start_match(game_mode, current.name):
                    self._diag(f"match_controller_start_match={current.name}")
                self.current_brawler = current.name
                if self.play and hasattr(self.play, "set_current_game_mode"):
                    try:
                        self.play.set_current_game_mode(game_mode)
                    except Exception as e:
                        logger.debug(f"[STATE] Falha ao definir game mode no play logic: {e}")

        if self.progress and hasattr(self.progress, "clear_last_result"):
            try:
                self.progress.clear_last_result()
            except Exception as e:
                logger.debug(f"[STATE] Falha ao limpar último resultado no início da partida: {e}")

        # NOVO: Forçar estado para loading após clicar no Play
        # Isto evita que o bot fique oscilando entre lobby e unknown
        # quando o detector não consegue reconhecer a tela de loading
        if pressed or True:  # Sempre forçar, mesmo que press_play tenha falhado no fallback
            logger.info("[STATE] Play pressionado. Forcando estado para loading por 5s...")
            self.current_state = 'loading'
            self.state_start_time = time.time()
            self._remember_known_state('loading')
            # Dar tempo para o jogo iniciar o loading
            time.sleep(2.0)
            return
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
            logger.warning("[STATE] press_play falhou após seleção; tentando fallback direto")
            self._log_lobby_snapshot("press_play_after_selection_failed")
            if self.emulator_controller:
                try:
                    w, h = self._get_window_size()
                    play_x = round(w * 0.9119)
                    play_y = round(h * 0.9122)
                    self.emulator_controller.tap_scaled(play_x, play_y)
                    logger.info(f"[STATE] Fallback direto no Play após seleção: ({play_x},{play_y})")
                    time.sleep(1.5)
                except Exception as e:
                    logger.warning(f"[STATE] Fallback direto após seleção falhou: {e}")
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
        
        # Check timeout: if loading for more than 5s, force to in_game
        # (Brawl Stars loading typically takes 3-8s; 5s is aggressive but safe)
        elapsed = time.time() - self._loading_start_time
        logger.info(f"[STATE] _handle_loading elapsed: {elapsed:.1f}s / 5s timeout")
        if elapsed > 5:
            logger.warning(f"[STATE] Loading timeout ({elapsed:.0f}s), forcing transition to in_game")
            self._loading_start_time = None
            self.current_state = 'in_game'
            self.state_start_time = time.time()
            self._remember_known_state('in_game')
            self._diag("loading_handler_timeout_forced_in_game")
            # CRITICAL: Mark that we forced in_game from loading - block return to loading for 30s
            self._forced_in_game_time = time.time()
            logger.info("[STATE] Forçado in_game desde loading - bloqueando retorno a loading por 30s")
            return
        
        # Reset timer if state changed naturally (not via timeout)
        # This prevents stale timer if we come back to loading later
        if self.current_state != 'loading':
            self._loading_start_time = None
        
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

        # CRITICAL FIX: Usar múltiplas fontes de tempo para garantir timeout
        # state_start_time é o mais fiável (vem do ciclo principal)
        # _matchmaking_enter_time é backup local
        state_elapsed = 0.0
        if self.state_start_time:
            state_elapsed = time.time() - self.state_start_time

        # Inicializar _matchmaking_enter_time se não existe ou se é inválido (None)
        if not hasattr(self, '_matchmaking_enter_time') or self._matchmaking_enter_time is None:
            self._matchmaking_enter_time = time.time()
            logger.info(f"[STATE] Matchmaking enter time inicializado: {self._matchmaking_enter_time:.3f}")

        try:
            matchmaking_elapsed = time.time() - self._matchmaking_enter_time
        except TypeError:
            # Se _matchmaking_enter_time é None ou inválido, usar state_start_time
            self._matchmaking_enter_time = self.state_start_time or time.time()
            matchmaking_elapsed = time.time() - self._matchmaking_enter_time
            logger.warning("[STATE] _matchmaking_enter_time inválido, usando state_start_time")

        # Usar o MAIOR dos dois elapsed (mais conservador, evita falsos negativos)
        effective_elapsed = max(matchmaking_elapsed, state_elapsed)
        logger.debug(f"[STATE] Matchmaking elapsed: local={matchmaking_elapsed:.1f}s, state={state_elapsed:.1f}s, effective={effective_elapsed:.1f}s")

        # Verificar se a partida já começou (proactive detection) - a cada ciclo
        try:
            img = self._get_cached_screenshot()
            if img is not None and hasattr(self, 'unified_detector') and self.unified_detector:
                result = self.unified_detector.detect(img)
                if result.state in ('in_game', 'loading'):
                    logger.info(f"[STATE] Partida detectada via proactive check: {result.state}")
                    self._diag(f"matchmaking_proactive_detected={result.state}")
                    self.current_state = result.state
                    self.state_start_time = time.time()
                    self._matchmaking_enter_time = None  # Reset
                    return
        except Exception as e:
            logger.debug(f"[STATE] Proactive matchmaking check falhou: {e}")

        # Verificar se a partida já começou (proactive detection via pixels crus)
        if effective_elapsed > 6:
            logger.info(f"[STATE] Matchmaking há {matchmaking_elapsed:.0f}s - verificando pixels crus")
            try:
                img = self._get_cached_screenshot()
                if img is not None and np is not None:
                    h, w = img.shape[:2]
                    # Verificar joystick area (escura = in_game)
                    joy_y, joy_x = int(h * 0.75), int(w * 0.10)
                    joy_region = img[max(0,joy_y-25):min(h,joy_y+25), max(0,joy_x-25):min(w,joy_x+25)]
                    if joy_region.size > 0 and np.mean(joy_region) < 100:
                        logger.info("[STATE] Joystick escuro detetado - forçando in_game desde matchmaking")
                        self.current_state = 'in_game'
                        self.state_start_time = time.time()
                        self._remember_known_state('in_game')
                        self._forced_in_game_time = time.time()
                        self._matchmaking_enter_time = None
                        self._handler_forced_state = True
                        return
            except Exception as e:
                logger.debug(f"[STATE] Proactive pixel check falhou: {e}")

        # TIMEOUT AGRESSIVO: forçar in_game após 8s (Brawl Stars matchmaking raramente demora mais)
        if effective_elapsed > 8:
            logger.warning(f"[STATE] Matchmaking timeout ({effective_elapsed:.0f}s > 8s) - FORÇANDO in_game")
            self._diag(f"matchmaking_timeout_force_in_game={effective_elapsed:.1f}")
            self.current_state = 'in_game'
            self.state_start_time = time.time()
            self._remember_known_state('in_game')
            self._forced_in_game_time = time.time()
            self._matchmaking_enter_time = None
            logger.info("[STATE] Forçado in_game desde matchmaking - bloqueando retorno por 25s")
            return
        
        # Reset timer if state changed naturally (not via timeout)
        if self.current_state != 'matchmaking':
            self._matchmaking_enter_time = None

        time.sleep(0.8)
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
        
        # Reset combat action time to prevent immediate fallback on entry
        self._last_combat_action_time = time.time()

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

            # GARANTIR que play_round é chamado e retorna resultado válido
            result = None
            try:
                result = self.play.play_round(screenshot)
                logger.info(f"[STATE] Play round resultado: {result}")
            except Exception as e:
                logger.error(f"[STATE] ERRO CRÍTICO em play_round: {e}", exc_info=True)
                result = {"attacked": False, "moved": False, "super_used": False, "success": False, "error": str(e)}

            # FALLBACK IMEDIATO: Se play_round falhou (ex: sem modelo), forçar ação imediatamente
            if result and isinstance(result, dict) and not result.get('success', True):
                logger.warning(f"[STATE] Play round falhou: {result.get('error', 'unknown')} - forçando ação imediata")
                current_time = time.time()
                if not hasattr(self, '_last_combat_action_time'):
                    self._last_combat_action_time = current_time
                # Forçar movimento imediato
                joy_x, joy_y = 300, 900
                if self.movement and hasattr(self.movement, 'joystick_center_x'):
                    joy_x = self.movement.joystick_center_x
                    joy_y = self.movement.joystick_center_y
                if self.emulator_controller:
                    import random
                    angle = random.uniform(0, 2 * 3.14159)
                    distance = random.randint(150, 350)
                    target_x = int(joy_x + distance * __import__('math').cos(angle))
                    target_y = int(joy_y + distance * __import__('math').sin(angle))
                    try:
                        self.emulator_controller.swipe_scaled(joy_x, joy_y, target_x, target_y, duration=200)
                        logger.info(f"[STATE] Movimento imediato: ({joy_x},{joy_y}) -> ({target_x},{target_y})")
                    except Exception as e:
                        logger.warning(f"[STATE] Falha no swipe imediato: {e}")
                    # Forçar ataque imediato
                    try:
                        atk_x, atk_y = 1750, 850
                        if self.movement and hasattr(self.movement, 'window_w'):
                            atk_x = round(self.movement.window_w * 0.90)
                            atk_y = round(self.movement.window_h * 0.82)
                        self.emulator_controller.tap_scaled(atk_x, atk_y)
                        logger.info(f"[STATE] Ataque imediato em ({atk_x},{atk_y})")
                        if hasattr(self.play, 'last_shot_time'):
                            self.play.last_shot_time = current_time
                    except Exception as e:
                        logger.warning(f"[STATE] Falha no ataque imediato: {e}")
                self._last_combat_action_time = current_time
                # Pular o resto do ciclo para evitar delays
                return

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
                    logger.warning(f"[STATE] BOT PARADO há {time_since_last_action:.1f}s - FORÇANDO ação de combate")
                    # Usar coordenadas dinâmicas do joystick
                    joy_x, joy_y = 300, 900
                    if self.movement and hasattr(self.movement, 'joystick_center_x'):
                        joy_x = self.movement.joystick_center_x
                        joy_y = self.movement.joystick_center_y
                    
                    # FORÇAR movimento aleatório
                    if self.emulator_controller:
                        import random
                        angle = random.uniform(0, 2 * 3.14159)
                        distance = random.randint(150, 350)
                        target_x = int(joy_x + distance * __import__('math').cos(angle))
                        target_y = int(joy_y + distance * __import__('math').sin(angle))
                        try:
                            self.emulator_controller.swipe_scaled(joy_x, joy_y, target_x, target_y, duration=200)
                            logger.info(f"[STATE] Movimento forçado: ({joy_x},{joy_y}) -> ({target_x},{target_y})")
                        except Exception as e:
                            logger.warning(f"[STATE] Falha no swipe forçado: {e}")
                    
                    # FORÇAR ataque se houver emulador e não atacou recentemente
                    if self.emulator_controller and self.play and hasattr(self.play, 'last_shot_time'):
                        time_since_shot = current_time - self.play.last_shot_time
                        if time_since_shot > 1.0:
                            try:
                                # Coordenadas dinâmicas do botão de ataque
                                atk_x, atk_y = 1750, 850
                                if self.movement and hasattr(self.movement, 'window_w'):
                                    atk_x = round(self.movement.window_w * 0.90)
                                    atk_y = round(self.movement.window_h * 0.82)
                                self.emulator_controller.tap_scaled(atk_x, atk_y)
                                logger.info(f"[STATE] Ataque forçado em ({atk_x},{atk_y})")
                                self.play.last_shot_time = current_time
                            except Exception as e:
                                logger.warning(f"[STATE] Falha no ataque forçado: {e}")
                    
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
            # NOTA: Se o rl_engine (OnlineLearner) tem gameplay_collector ativo,
            # a coleta ja e feita no learn_from_frame() com dados RL completos.
            # Neste caso, nao duplicamos a coleta aqui.
            has_rl_collector = (
                self.rl_engine is not None
                and getattr(self.rl_engine, "gameplay_collector", None) is not None
            )
            if self.data_collector is not None and not has_rl_collector:
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
        """No fim de uma partida - processar resultado e sair com recovery autónomo."""
        logger.debug("[STATE] Handler end_game iniciado")

        # Reset matchmaking timer para permitir novo ciclo
        if hasattr(self, '_matchmaking_enter_time'):
            self._matchmaking_enter_time = None

        # Se estamos no end há muito tempo, forçar retorno ao lobby
        if self.state_start_time and (time.time() - self.state_start_time) > 12:
            logger.warning("[STATE] End screen timeout - forçando retorno ao lobby")
            self.current_state = 'lobby'
            self.state_start_time = time.time()
            if hasattr(self, '_matchmaking_enter_time'):
                self._matchmaking_enter_time = None
            return
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

        # NOVO: Tentar usar UnifiedStateDetector com fallback inteligente
        if self.unified_detector and image is not None:
            try:
                detection = self.unified_detector.detect(image)
                if detection.state != 'unknown' and detection.confidence > 0.2:
                    logger.info(f"[STATE] Unknown handler: detector recuperou estado {detection.state} "
                               f"(conf={detection.confidence:.2f}, method={detection.method})")
                    self.current_state = detection.state
                    self._remember_known_state(detection.state)
                    if detection.state == 'lobby' and detection.button_coords and self.emulator_controller:
                        logger.info(f"[STATE] Clicando no botao Play detectado em {detection.button_coords}")
                        self.emulator_controller.tap_scaled(*detection.button_coords)
                        time.sleep(1.5)
                    return
            except Exception as e:
                logger.debug(f"[STATE] Unknown handler: detector fallback falhou: {e}")

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
                # Estratégia agressiva: múltiplos cliques em posições comuns de botões
                tap_positions = [
                    (0.50, 0.82),  # Botão verde GOT IT inferior central
                    (0.50, 0.75),  # Botão verde um pouco mais acima
                    (0.50, 0.88),  # Botão verde mais abaixo
                    (0.50, 0.50),  # Centro (fallback)
                    (0.85, 0.10),  # X no canto superior direito
                    (0.15, 0.10),  # X no canto superior esquerdo
                ]
                for i, (x_pct, y_pct) in enumerate(tap_positions):
                    self.emulator_controller.tap_scaled(round(w * x_pct), round(h * y_pct))
                    logger.info(f"[STATE] Event screen tap {i+1}/{len(tap_positions)}: ({x_pct},{y_pct})")
                    time.sleep(0.4)
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

