"""
lobby_automator.py

Automatiza ações no lobby, incluindo seleção de brawlers.
Com suporte a fila de brawlers (melhoria sobre o original).

VERSAO 2.0 - Navegacao inteligente com:
- PopupManager: fecha popups automaticamente
- SmartPlayButtonDetector: encontra Play em qualquer situacao
- EventDetector: identifica Starr Nova e eventos
- BrawlerSelectorFast: selecao rapida com cache
"""

import time
import random
from collections import deque
from typing import Optional, List, Dict, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
import logging

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for heavy UI/vision libraries to avoid startup hangs
pyautogui = None
np = None  # type: ignore
Image = None
ImageOps = None

def _ensure_imports():
    global pyautogui, np, Image, ImageOps
    if pyautogui is None:
        try:
            import pyautogui as _pyautogui
            pyautogui = _pyautogui
        except Exception as e:
            logger.warning(f"[LOBBY] pyautogui não disponível: {e}")
    if np is None:
        try:
            import numpy as _np
            np = _np
        except Exception as e:
            logger.warning(f"[LOBBY] numpy não disponível: {e}")
    if Image is None or ImageOps is None:
        try:
            from PIL import Image as _Image, ImageOps as _ImageOps
            Image = _Image
            ImageOps = _ImageOps
        except Exception as e:
            logger.warning(f"[LOBBY] PIL não disponível: {e}")

try:
    from realtime_logs import get_log_manager
    log_manager = get_log_manager()
except ImportError:
    log_manager = None
    logger.warning("[LOBBY] Log manager não disponível")

# Importar navegacao inteligente do lobby (v2)
try:
    from .lobby_navigator import PopupManager, SmartPlayButtonDetector, EventDetector, BrawlerSelectorFast
    LOBBY_NAV_AVAILABLE = True
except ImportError:
    LOBBY_NAV_AVAILABLE = False
    logger.warning("[LOBBY] lobby_navigator nao disponivel, usando metodo legado")

try:
    from .lobby_automation_expanded import (
        EventSlotNavigator, ModeSelectionResolver,
        LobbyAutomationExpanded, TrainingCaveResult,
        PlayAgainResult, PvEClassification,
    )
    LOBBY_EXPANDED_AVAILABLE = True
except ImportError:
    LOBBY_EXPANDED_AVAILABLE = False
    LobbyAutomationExpanded = None
    TrainingCaveResult = None
    PlayAgainResult = None
    PvEClassification = None
    logger.warning("[LOBBY] lobby_automation_expanded nao disponivel")


@dataclass
class BrawlerConfig:
    """Configuração de um brawler na fila"""
    name: str
    current_trophies: int = 0
    target_trophies: int = 350
    current_wins: int = 0
    target_wins: int = 10
    priority: int = 1  # 1-5, maior = mais prioritário
    enabled: bool = True
    game_mode: Optional[str] = None  # Modo de jogo preferido (showdown, gem_grab, etc.)


class BrawlerQueue:
    """Fila de brawlers para farm automático (melhoria PylaAI)"""

    def __init__(self):
        self.brawlers: List[BrawlerConfig] = []
        self.current_index = 0

    def add_brawler(self, config: BrawlerConfig):
        """Adiciona brawler à fila"""
        self.brawlers.append(config)
        self._sort_by_priority()

    def remove_brawler(self, index: int) -> bool:
        """Remove brawler da fila pelo índice"""
        if 0 <= index < len(self.brawlers):
            self.brawlers.pop(index)
            if self.current_index >= len(self.brawlers):
                self.current_index = 0
            return True
        return False

    def should_switch(self, current_result=None, history=None) -> bool:
        """Compatibilidade com o MatchController: decide troca por metas ou sequência recente de derrotas."""
        if self.check_goals():
            return True

        current = self.get_current()
        if not current or not history:
            return False

        recent_matches = history.matches[-3:]
        recent_losses = sum(
            1 for match in recent_matches
            if match.result == "loss" and match.brawler == current.name
        )
        if recent_losses >= 3:
            logger.info(f"Brawler {current.name} com 3 derrotas seguidas, trocando...")
            current.enabled = False
            return True

        return False

    def reorder(self, new_order: List[int]):
        """Reordena a fila (recebe lista de índices)"""
        if len(new_order) == len(self.brawlers):
            self.brawlers = [self.brawlers[i] for i in new_order]
            self.current_index = 0

    def get_current(self) -> Optional[BrawlerConfig]:
        """Retorna brawler atual"""
        if not self.brawlers:
            return None
        return self.brawlers[self.current_index]

    def next(self) -> Optional[BrawlerConfig]:
        """Avança para o próximo brawler ativo"""
        if not self.brawlers:
            return None

        # Encontrar próximo brawler ativo
        checked = 0
        while checked < len(self.brawlers):
            self.current_index = (self.current_index + 1) % len(self.brawlers)
            current = self.brawlers[self.current_index]
            if current.enabled:
                return current
            checked += 1

        return None  # Nenhum brawler ativo

    def peek_next(self) -> Optional[BrawlerConfig]:
        """Observa o próximo brawler ativo sem alterar o índice atual"""
        if not self.brawlers:
            return None

        idx = self.current_index
        checked = 0
        while checked < len(self.brawlers):
            idx = (idx + 1) % len(self.brawlers)
            current = self.brawlers[idx]
            if current.enabled:
                return current
            checked += 1

        return None

    def set_current_by_name(self, name: str) -> bool:
        """Define o brawler atual pelo nome."""
        for idx, brawler in enumerate(self.brawlers):
            if brawler.name == name:
                self.current_index = idx
                return True
        return False

    def get_available_names(self) -> List[str]:
        """Retorna nomes de brawlers ativos na fila."""
        return [b.name for b in self.brawlers if b.enabled]

    def check_goals(self) -> bool:
        """Verifica se brawler atual atingiu metas"""
        current = self.get_current()
        if not current:
            return False

        if current.current_trophies >= current.target_trophies:
            logger.info(f"Brawler {current.name} atingiu meta de troféus!")
            current.enabled = False
            return True

        if current.current_wins >= current.target_wins:
            logger.info(f"Brawler {current.name} atingiu meta de vitórias!")
            current.enabled = False
            return True

        return False

    def _sort_by_priority(self):
        """Ordena por prioridade (maior primeiro)"""
        self.brawlers.sort(key=lambda b: b.priority, reverse=True)

    def get_queue(self) -> List[Dict]:
        """Retorna fila como lista de dicts"""
        return [
            {
                'name': b.name,
                'current_trophies': b.current_trophies,
                'target_trophies': b.target_trophies,
                'current_wins': b.current_wins,
                'target_wins': b.target_wins,
                'priority': b.priority,
                'enabled': b.enabled,
                'game_mode': b.game_mode,
                'is_current': i == self.current_index
            }
            for i, b in enumerate(self.brawlers)
        ]


def _retry_on_failure(max_retries: int = 3, backoff: float = 0.5):
    """Decorator simples de retry com backoff exponencial."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait = backoff * (2 ** (attempt - 1))
                    logger.warning(f"[LOBBY] Retry {attempt}/{max_retries} para {func.__name__}: {e}")
                    time.sleep(wait)
            logger.error(f"[LOBBY] Falha após {max_retries} tentativas em {func.__name__}: {last_exception}")
            return False
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


class LobbyConfig:
    """
    Coordenadas dinâmicas para UI do Brawl Stars.
    Todas como % da janela (como BrawlStarsBot) para funcionar em qualquer resolução.
    
    Updated 2025-05: Coordenadas mapeadas ao vivo no BlueStacks 1600x900.
    O layout do Brawl Stars mudou - o Play button agora fica na barra de navegação
    inferior, nao no canto inferior direito como antes.
    """
    # Play button in nav bar (updated from live mapping)
    # Old: 0.9419 x, 0.8949 y (canto inferior direito - nao funciona mais)
    # New: 0.9119 x, 0.9122 y (tab na barra de navegacao)
    PLAY_BTN_X_PCT = 0.9119
    PLAY_BTN_Y_PCT = 0.9122
    CENTER_X_PCT = 0.50
    CENTER_Y_PCT = 0.50
    PLAY_FALLBACK_Y_PCT = 0.93
    CLOSE_X_PCT = 0.94
    CLOSE_Y_PCT = 0.09
    SWIPE_DOWN_START_Y_PCT = 0.74
    SWIPE_DOWN_END_Y_PCT = 0.28
    GRID_CENTRE_Y_PCT = 0.50
    GRID_LEFT_X_PCT = 0.40
    GRID_RIGHT_X_PCT = 0.65
    GRID_TOP_Y_PCT = 0.28
    LIST_SCROLL_AMOUNT = -5

    # Fight button on game mode selection screen (updated from live mapping)
    # This is the button that appears after clicking Play tab
    FIGHT_BTN_X_PCT = 0.4950
    FIGHT_BTN_Y_PCT = 0.9356

    # Play again button (end of match)
    PLAY_AGAIN_X_PCT = 0.5903
    PLAY_AGAIN_Y_PCT = 0.9197

    # Exit button (when defeated)
    EXIT_X_PCT = 0.493
    EXIT_Y_PCT = 0.9187

    # Proceed button
    PROCEED_X_PCT = 0.8093
    PROCEED_Y_PCT = 0.9165

    # Brawler icon in lobby (to open brawler selection)
    BRAWLER_ICON_X_PCT = 0.3494
    BRAWLER_ICON_Y_PCT = 0.4489

    def __init__(self, window_w: int = 1920, window_h: int = 1080):
        self.w = window_w
        self.h = window_h
        self.W = window_w  # Alias para compatibilidade com código legado
        self.H = window_h  # Alias para compatibilidade com código legado
        self.recompute()

    def recompute(self):
        """Recalcula coordenadas absolutas baseado na resolução."""
        self.PLAY_BTN_X = round(self.w * self.PLAY_BTN_X_PCT)
        self.PLAY_BTN_Y = round(self.h * self.PLAY_BTN_Y_PCT)
        self.CENTER_X = round(self.w * self.CENTER_X_PCT)
        self.CENTER_Y = round(self.h * self.CENTER_Y_PCT)
        self.PLAY_FALLBACK_Y = round(self.h * self.PLAY_FALLBACK_Y_PCT)
        self.CLOSE_X = round(self.w * self.CLOSE_X_PCT)
        self.CLOSE_Y = round(self.h * self.CLOSE_Y_PCT)
        self.SWIPE_DOWN_START_Y = round(self.h * self.SWIPE_DOWN_START_Y_PCT)
        self.SWIPE_DOWN_END_Y = round(self.h * self.SWIPE_DOWN_END_Y_PCT)
        self.GRID_CENTRE_Y = round(self.h * self.GRID_CENTRE_Y_PCT)
        self.GRID_LEFT_X = round(self.w * self.GRID_LEFT_X_PCT)
        self.GRID_RIGHT_X = round(self.w * self.GRID_RIGHT_X_PCT)
        self.GRID_TOP_Y = round(self.h * self.GRID_TOP_Y_PCT)
        self.FIGHT_BTN_X = round(self.w * self.FIGHT_BTN_X_PCT)
        self.FIGHT_BTN_Y = round(self.h * self.FIGHT_BTN_Y_PCT)
        self.PLAY_AGAIN_X = round(self.w * self.PLAY_AGAIN_X_PCT)
        self.PLAY_AGAIN_Y = round(self.h * self.PLAY_AGAIN_Y_PCT)
        self.EXIT_X = round(self.w * self.EXIT_X_PCT)
        self.EXIT_Y = round(self.h * self.EXIT_Y_PCT)
        self.PROCEED_X = round(self.w * self.PROCEED_X_PCT)
        self.PROCEED_Y = round(self.h * self.PROCEED_Y_PCT)
        self.BRAWLER_ICON_X = round(self.w * self.BRAWLER_ICON_X_PCT)
        self.BRAWLER_ICON_Y = round(self.h * self.BRAWLER_ICON_Y_PCT)


class LobbyAutomator:
    """Automatiza ações no lobby"""

    def __init__(self, queue: Optional[BrawlerQueue] = None, emulator_controller = None, screen_automation=None, diagnostic_mode: bool = False, play_logic=None, window_w: int = 1920, window_h: int = 1080, images_path: str = None):
        self.queue = queue or BrawlerQueue()
        self.emulator_controller = emulator_controller
        self.screen_automation = screen_automation
        self.diagnostic_mode = diagnostic_mode
        self.play_logic = play_logic
        self.lobby_config = LobbyConfig(window_w, window_h)
        logger.debug(f"[LOBBY] LobbyAutomator inicializado com play_logic: {play_logic is not None}")
        self.scroll_attempts = 0
        self.max_scrolls = 10
        self.last_diagnostic = {
            "flow": "idle",
            "step": "created",
            "details": {},
            "error": None,
            "updated_at": None,
        }
        # Simple state machine for lobby flow
        self._state = "idle"
        self._state_history: deque = deque(maxlen=100)
        # Visual verification support
        self._state_detector = None  # Will be set via set_state_detector()
        self._screenshot_func = None  # Will be set via set_screenshot_func()

        # NOVO: Navegacao inteligente do lobby v2
        self._images_path = Path(images_path) if images_path else Path(__file__).parent.parent / "images"
        if LOBBY_NAV_AVAILABLE:
            logger.info("[LOBBY] Inicializando navegacao inteligente v2")
            self._popup_manager = PopupManager(self._images_path)
            self._play_detector = SmartPlayButtonDetector(self._images_path)
            self._event_detector = EventDetector(self._images_path)
            self._brawler_selector_fast = BrawlerSelectorFast(self._images_path)
        else:
            self._popup_manager = None
            self._play_detector = None
            self._event_detector = None
            self._brawler_selector_fast = None

        if LOBBY_EXPANDED_AVAILABLE:
            logger.info("[LOBBY] Inicializando sistemas expandidos (EventSlotNavigator, ModeSelectionResolver)")
            self._slot_navigator = EventSlotNavigator(self._images_path)
            self._mode_resolver = ModeSelectionResolver
            self._expanded = LobbyAutomationExpanded(self._images_path)
        else:
            self._slot_navigator = None
            self._mode_resolver = None
            self._expanded = None

        logger.debug(f"[LOBBY] LobbyAutomator inicializado: diagnostic_mode={diagnostic_mode}, emulator_controller={'available' if emulator_controller else 'unavailable'}, window={window_w}x{window_h}, nav_v2={LOBBY_NAV_AVAILABLE}")

    @property
    def state(self) -> str:
        return self._state

    def _transition(self, new_state: str):
        if self._state != new_state:
            logger.info(f"[LOBBY] State transition: {self._state} -> {new_state}")
            self._state_history.append((self._state, new_state, time.time()))
            self._state = new_state

    def set_screen_automation(self, screen_automation):
        """Atualiza a automação de tela associada ao lobby."""
        self.screen_automation = screen_automation

    def set_state_detector(self, detector):
        """Define o detector de estado unificado para verificação visual."""
        self._state_detector = detector

    def set_screenshot_func(self, func):
        """Define a função de captura de screenshot para verificação visual."""
        self._screenshot_func = func

    def update_window_size(self, w: int, h: int):
        """Atualiza coordenadas quando a janela muda de tamanho."""
        self.lobby_config = LobbyConfig(w, h)
        logger.info(f"[LOBBY] Coordenadas atualizadas para {w}x{h}")

    def set_diagnostic_mode(self, enabled: bool):
        """Ativa ou desativa o modo diagnóstico em runtime."""
        self.diagnostic_mode = enabled
        logger.info(f"[LOBBY][DIAG] Diagnostic mode {'enabled' if enabled else 'disabled'}")

    def _update_diagnostic(self, step: str, **details):
        """Guarda um snapshot curto do passo atual do fluxo do lobby."""
        self.last_diagnostic = {
            "flow": "lobby",
            "step": step,
            "details": details,
            "error": details.get("error"),
            "updated_at": time.time(),
        }
        if self.diagnostic_mode:
            logger.info(f"[LOBBY][DIAG] step={step} details={details}")

    def get_diagnostic_report(self) -> Dict[str, object]:
        """Retorna o último snapshot diagnóstico do lobby."""
        return dict(self.last_diagnostic)

    # ------------------------------------------------------------------
    # Delegacoes para LobbyAutomationExpanded (sistemas expandidos)
    # ------------------------------------------------------------------

    def handle_end_screen_expanded(self, screenshot, window_size=(1920, 1080)):
        """Usa PlayAgainHandler inteligente para sair do end screen."""
        if self._expanded and self.emulator_controller:
            return self._expanded.handle_end_screen(
                screenshot=screenshot,
                emulator_controller=self.emulator_controller,
                screen_automation=self.screen_automation,
                window_size=window_size,
            )
        return None

    def enter_training_cave(self, screenshot, window_size=(1920, 1080)):
        """Entra na Training Cave."""
        if self._expanded and self.emulator_controller:
            return self._expanded.enter_training_cave(
                screenshot=screenshot,
                emulator_controller=self.emulator_controller,
                window_size=window_size,
            )
        return None

    def exit_training_cave(self, window_size=(1920, 1080)):
        """Sai da Training Cave."""
        if self._expanded and self.emulator_controller:
            return self._expanded.exit_training_cave(
                emulator_controller=self.emulator_controller,
                window_size=window_size,
            )
        return None

    def is_in_training_cave(self, screenshot) -> bool:
        """Verifica se estamos na Training Cave."""
        if self._expanded:
            return self._expanded.is_in_training_cave(screenshot)
        return False

    def detect_pve(self, screenshot=None, game_mode=None, enemy_detections=None):
        """Classifica se a partida atual e PvE."""
        if self._expanded:
            return self._expanded.detect_pve(
                screenshot=screenshot,
                game_mode=game_mode,
                enemy_detections=enemy_detections,
            )
        return None

    def handle_friendly_invite(self, screenshot, auto_accept=False):
        """Trata convites de partidas amistosas."""
        if self._expanded and self.emulator_controller:
            return self._expanded.handle_friendly_invite(
                screenshot=screenshot,
                emulator_controller=self.emulator_controller,
                auto_accept=auto_accept,
            )
        return False

    def collect_daily_rewards(self, screenshot, window_size=(1920, 1080)):
        """Coleta recompensas diarias se disponivel."""
        if self._expanded and self.emulator_controller:
            return self._expanded.collect_daily_rewards(
                screenshot=screenshot,
                emulator_controller=self.emulator_controller,
                window_size=window_size,
            )
        return False

    def collect_starr_road(self, screenshot, window_size=(1920, 1080)):
        """Coleta recompensas da Starr Road."""
        if self._expanded and self.emulator_controller:
            return self._expanded.collect_starr_road(
                screenshot=screenshot,
                emulator_controller=self.emulator_controller,
                window_size=window_size,
            )
        return False

    def collect_quest_rewards(self, screenshot, window_size=(1920, 1080)):
        """Coleta recompensas de missoes/quests."""
        if self._expanded and self.emulator_controller:
            return self._expanded.collect_quest_rewards(
                screenshot=screenshot,
                emulator_controller=self.emulator_controller,
                window_size=window_size,
            )
        return False

    def collect_shop_items(self, screenshot, window_size=(1920, 1080)):
        """Coleta itens gratuitos da loja."""
        if self._expanded and self.emulator_controller:
            return self._expanded.collect_shop_items(
                screenshot=screenshot,
                emulator_controller=self.emulator_controller,
                window_size=window_size,
            )
        return False

    def handle_maintenance(self, screenshot, window_size=(1920, 1080)):
        """Trata tela de manutencao/update."""
        if self._expanded and self.emulator_controller:
            return self._expanded.handle_maintenance(
                screenshot=screenshot,
                emulator_controller=self.emulator_controller,
                window_size=window_size,
            )
        return False

    def detect_maintenance(self, screenshot, window_size=(1920, 1080)):
        """Deteta se ha tela de manutencao."""
        if self._expanded:
            return self._expanded.detect_maintenance(screenshot, window_size)
        return False, "none"

    def _click(self, x: int, y: int):
        """Executa clique via EmulatorController (ADB) ou pyautogui (fallback)"""
        logger.debug(f"[LOBBY] Executando clique em ({x}, {y})")
        if self.emulator_controller:
            # ADB tap é independente da posição da janela no PC
            # Usa tap_scaled para garantir que 1920x1080 -> resolução real
            logger.debug("[LOBBY] Usando EmulatorController para clique")
            self.emulator_controller.tap_scaled(x, y)
        else:
            # PyAutoGUI depende da posição da janela (não recomendado para ADB bots)
            _ensure_imports()
            logger.debug("[LOBBY] Usando pyautogui fallback para clique")
            if pyautogui:
                pyautogui.click(x, y)
            else:
                logger.error("[LOBBY] pyautogui não disponível para clique fallback")

    def _key_press(self, key: str):
        """Executa pressionamento de tecla via EmulatorController (ADB) ou pyautogui"""
        logger.debug(f"[LOBBY] Pressionando tecla: {key}")
        if self.emulator_controller:
            # Mapear chaves comuns para keycodes Android
            keymap = {
                'esc': 4,      # BACK
                'enter': 66,   # ENTER
                'home': 3,     # HOME
                'backspace': 67 # DEL
            }
            keycode = keymap.get(key.lower(), 4)
            logger.debug(f"[LOBBY] Enviando keyevent {keycode} para tecla '{key}'")
            self.emulator_controller.keyevent(keycode)
        else:
            _ensure_imports()
            logger.debug(f"[LOBBY] Usando pyautogui fallback para tecla '{key}'")
            if pyautogui:
                pyautogui.press(key)
            else:
                logger.error("[LOBBY] pyautogui não disponível para key press fallback")

    def _get_ocr_reader(self):
        """Inicializa o OCR com fallback EasyOCR -> pytesseract -> indisponível."""
        if hasattr(self, '_ocr_reader') and self._ocr_reader is not None:
            logger.debug(f"[LOBBY][OCR] Reutilizando reader existente: {self._ocr_backend}")
            return self._ocr_reader, getattr(self, '_ocr_backend', 'easyocr')

        logger.debug("[LOBBY][OCR] Tentando inicializar EasyOCR")
        try:
            import easyocr
            self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self._ocr_backend = 'easyocr'
            self._update_diagnostic('ocr_reader_ready', backend='easyocr')
            logger.info("[LOBBY][OCR] EasyOCR inicializado com sucesso")
            return self._ocr_reader, self._ocr_backend
        except ImportError:
            logger.debug("[LOBBY][OCR] EasyOCR não disponível, tentando pytesseract")
            try:
                import pytesseract  # type: ignore
                self._ocr_reader = pytesseract
                self._ocr_backend = 'pytesseract'
                self._update_diagnostic('ocr_reader_ready', backend='pytesseract')
                logger.info("[LOBBY][OCR] pytesseract inicializado com sucesso")
                return self._ocr_reader, self._ocr_backend
            except ImportError:
                logger.warning("[LOBBY][OCR] Nenhum backend OCR disponível (EasyOCR nem pytesseract)")
                self._ocr_reader = None
                self._ocr_backend = 'none'
                return None, self._ocr_backend

    def _preprocess_for_ocr(self, img):
        """Melhora contraste antes da leitura de texto."""
        _ensure_imports()
        if Image is None or ImageOps is None or np is None:
            logger.warning("[LOBBY] PIL/numpy não disponível para preprocessamento OCR")
            return img
        pil_img = Image.fromarray(img)
        gray = ImageOps.grayscale(pil_img)
        gray = ImageOps.autocontrast(gray)
        return np.array(gray)

    def _read_text_candidates(self, img):
        """Extrai texto com o backend OCR disponível."""
        _ensure_imports()
        if np is None:
            logger.warning("[LOBBY] numpy não disponível para OCR")
            return []
        logger.debug(f"[LOBBY][OCR] Lendo texto com backend")
        reader, backend = self._get_ocr_reader()
        if reader is None or backend == 'none':
            logger.debug("[LOBBY][OCR] Nenhum reader OCR disponível")
            return []

        logger.debug(f"[LOBBY][OCR] Pré-processando imagem para OCR")
        processed = self._preprocess_for_ocr(img)
        try:
            if backend == 'easyocr':
                logger.debug("[LOBBY][OCR] Usando EasyOCR para leitura")
                results = reader.readtext(processed)
                logger.debug(f"[LOBBY][OCR] EasyOCR retornou {len(results)} resultados")
                return [(bbox, text, conf) for bbox, text, conf in results]
            logger.debug("[LOBBY][OCR] Usando pytesseract para leitura")
            text = reader.image_to_string(processed)
            if text:
                logger.debug(f"[LOBBY][OCR] pytesseract encontrou texto: {text[:50]}...")
                # pytesseract não fornece bbox confiável aqui; usamos bbox vazio.
                return [((0, 0, 0, 0), text, 1.0)]
        except Exception as e:
            logger.error(f"[LOBBY][OCR] Erro ao ler texto com backend '{backend}': {e}")
        return []

    def _classify_card_position(self, center_x: Optional[float], image_width: int) -> str:
        """Classifica a posição horizontal de um card/texto visível."""
        if center_x is None or image_width <= 0:
            return "unknown"

        third = image_width / 3.0
        if center_x < third:
            return "left"
        if center_x < third * 2:
            return "center"
        return "right"

    def _summarize_candidates(self, results, image_width: int, image_height: int):
        """Normaliza OCR em um resumo mais legível para diagnóstico e navegação."""
        summary = []
        for bbox, text, conf in results:
            center_x = None
            center_y = None
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and isinstance(bbox[0], (list, tuple)):
                xs = [pt[0] for pt in bbox if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                ys = [pt[1] for pt in bbox if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                if xs and ys:
                    center_x = sum(xs) / len(xs)
                    center_y = sum(ys) / len(ys)

            position = self._classify_card_position(center_x, image_width)
            summary.append({
                "text": text,
                "confidence": conf,
                "center_x": center_x,
                "center_y": center_y,
                "position": position,
                "visible": True,
            })

        layout_hint = "grid"
        positions = {item["position"] for item in summary if item["position"] in {"left", "center", "right"}}
        if len(positions) <= 1:
            layout_hint = "list"

        return summary, layout_hint

    def _grid_navigation_step(self, attempt: int):
        """Executa um passo de navegação horizontal/vertical para telas em grade."""
        cfg = self.lobby_config
        pattern = attempt % 4
        if pattern == 0:
            direction = "right"
            coords = (cfg.GRID_LEFT_X, cfg.GRID_CENTRE_Y, cfg.GRID_RIGHT_X, cfg.GRID_CENTRE_Y)
        elif pattern == 1:
            direction = "down"
            coords = (cfg.CENTER_X, cfg.SWIPE_DOWN_START_Y, cfg.CENTER_X, cfg.SWIPE_DOWN_END_Y)
        elif pattern == 2:
            direction = "left"
            coords = (cfg.GRID_RIGHT_X, cfg.GRID_CENTRE_Y, cfg.GRID_LEFT_X, cfg.GRID_CENTRE_Y)
        else:
            direction = "up"
            coords = (cfg.CENTER_X, cfg.GRID_TOP_Y, cfg.CENTER_X, cfg.SWIPE_DOWN_START_Y)

        logger.debug(f"[LOBBY] Navegação em grade passo {attempt + 1}: direção={direction}, coords={coords}")
        if self.emulator_controller:
            self._update_diagnostic("grid_navigation", attempt=attempt + 1, direction=direction, method="emulator_controller", coords=coords)
            self.emulator_controller.swipe_scaled(*coords)
        else:
            _ensure_imports()
            self._update_diagnostic("grid_navigation", attempt=attempt + 1, direction=direction, method="pyautogui", coords=coords)
            if pyautogui:
                pyautogui.moveTo(cfg.CENTER_X, cfg.CENTER_Y)
                pyautogui.dragTo(coords[2], coords[3], duration=0.2, button='left')

    def _list_navigation_step(self, attempt: int):
        """Executa scroll vertical para layouts em lista."""
        cfg = self.lobby_config
        logger.debug(f"[LOBBY] Navegação em lista passo {attempt + 1}")
        if self.emulator_controller:
            self._update_diagnostic("list_navigation", attempt=attempt + 1, method="emulator_controller", direction="down")
            self.emulator_controller.swipe_scaled(cfg.CENTER_X, cfg.SWIPE_DOWN_START_Y, cfg.CENTER_X, cfg.SWIPE_DOWN_END_Y)
        else:
            _ensure_imports()
            self._update_diagnostic("list_navigation", attempt=attempt + 1, method="pyautogui", direction="down")
            if pyautogui:
                pyautogui.scroll(cfg.LIST_SCROLL_AMOUNT, cfg.CENTER_X, cfg.CENTER_Y)

    def _confirm_selected_brawler(self, brawler_name: str, screenshot_func, backend: str) -> bool:
        """Confirma após o clique que o brawler selecionado continua visível na tela."""
        _ensure_imports()
        if Image is None or np is None:
            logger.warning("[LOBBY] PIL/numpy não disponíveis para confirmação de brawler")
            return False
        confirmation_image = screenshot_func()
        if confirmation_image is None:
            self._update_diagnostic("brawler_confirmation_missing_screenshot", target=brawler_name, backend=backend)
            return False

        pil_img = Image.fromarray(confirmation_image)
        w, h = pil_img.size
        pil_img = pil_img.resize((int(w * 0.65), int(h * 0.65)))
        results = self._read_text_candidates(np.array(pil_img))
        resized_w, resized_h = pil_img.size
        summary, layout_hint = self._summarize_candidates(results, resized_w, resized_h)
        self._update_diagnostic(
            "brawler_confirmation_scan",
            target=brawler_name,
            backend=backend,
            layout_hint=layout_hint,
            candidates=summary,
        )

        for candidate in summary:
            if candidate["confidence"] > 0.6 and brawler_name.lower() in candidate["text"].lower():
                self._update_diagnostic(
                    "brawler_selected_confirmed",
                    target=brawler_name,
                    backend=backend,
                    position=candidate["position"],
                    confidence=candidate["confidence"],
                )
                return True

        self._update_diagnostic("brawler_confirmation_failed", target=brawler_name, backend=backend, layout_hint=layout_hint)
        return False

    @_retry_on_failure(max_retries=2, backoff=0.5)
    def select_brawler(self, brawler_name: str, screenshot_func) -> bool:
        self._transition("selecting_brawler")
        logger.info(f"[LOBBY] Iniciando selecao de brawler v2: {brawler_name}")

        # === TENTAR SELECAO RAPIDA (v2) PRIMEIRO ===
        if self._brawler_selector_fast and LOBBY_NAV_AVAILABLE:
            try:
                screenshot = screenshot_func()
                if screenshot is not None:
                    result = self._brawler_selector_fast.select_brawler(
                        brawler_name, screenshot,
                        click_func=self._click,
                        swipe_func=lambda x1, y1, x2, y2: self._swipe(x1, y1, x2, y2) if hasattr(self, '_swipe') else None
                    )
                    if result:
                        logger.info(f"[LOBBY] Brawler {brawler_name} selecionado via v2 (rapido)")
                        self._transition("idle")
                        return True
                    logger.info(f"[LOBBY] v2 falhou, caindo para OCR legado")
            except Exception as e:
                logger.warning(f"[LOBBY] Erro na selecao rapida v2: {e}, fallback para OCR")

        # === FALLBACK: METODO LEGADO (OCR + scroll) ===
        logger.info(f"[LOBBY] Iniciando selecao OCR legada: {brawler_name}")
        if log_manager:
            log_manager.log(
                message=f"Iniciando seleção de brawler: {brawler_name}",
                level="INFO",
                category="lobby",
                data={"brawler_name": brawler_name, "action": "selection_start"}
            )
        try:
            _ensure_imports()
            if Image is None or np is None:
                logger.warning("[LOBBY] PIL/numpy não disponíveis para seleção de brawler")
                self._update_diagnostic("pil_numpy_missing", target=brawler_name)
                self._transition("idle")
                return False
            self._update_diagnostic("select_brawler_start", target=brawler_name, scroll_attempts=self.scroll_attempts)
            reader, backend = self._get_ocr_reader()
            if reader is None:
                logger.warning("[LOBBY] Nenhum OCR disponível. Não foi possível selecionar brawler por nome.")
                self._update_diagnostic("ocr_missing", target=brawler_name, error="no_ocr_backend")
                self._transition("idle")
                return False
            logger.debug(f"[LOBBY] Usando backend OCR: {backend}")

            for attempt in range(self.max_scrolls):
                self._update_diagnostic("capture_screenshot", target=brawler_name, attempt=attempt + 1, max_scrolls=self.max_scrolls)
                img = screenshot_func()
                if img is None:
                    self._update_diagnostic("screenshot_missing", target=brawler_name, attempt=attempt + 1)
                    continue

                pil_img = Image.fromarray(img)
                w, h = pil_img.size
                pil_img = pil_img.resize((int(w * 0.65), int(h * 0.65)))

                results = self._read_text_candidates(np.array(pil_img))
                resized_w, resized_h = pil_img.size
                summary, layout_hint = self._summarize_candidates(results, resized_w, resized_h)
                self._update_diagnostic(
                    "ocr_completed",
                    target=brawler_name,
                    attempt=attempt + 1,
                    matches=len(results),
                    backend=backend,
                    layout_hint=layout_hint,
                    candidates=summary,
                )

                for candidate in summary:
                    text = candidate["text"]
                    conf = candidate["confidence"]
                    center_x = candidate["center_x"]
                    center_y = candidate["center_y"]
                    position = candidate["position"]
                    if conf > 0.6 and brawler_name.lower() in text.lower():
                        if center_x is not None and center_y is not None:
                            x = int(center_x / 0.65)
                            y = int(center_y / 0.65)
                        else:
                            x = self.lobby_config.CENTER_X
                            y = self.lobby_config.CENTER_Y
                        self._update_diagnostic(
                            "brawler_match_found",
                            target=brawler_name,
                            attempt=attempt + 1,
                            confidence=conf,
                            click_x=x,
                            click_y=y,
                            matched_text=text,
                            card_position=position,
                        )
                        self._click(x, y)
                        time.sleep(0.5)
                        logger.info(f"Brawler {brawler_name} selected via OCR backend={backend}")
                        if log_manager:
                            log_manager.log(
                                message=f"Brawler selecionado via OCR: {brawler_name}",
                                level="INFO",
                                category="lobby",
                                data={
                                    "brawler_name": brawler_name,
                                    "ocr_backend": backend,
                                    "confidence": conf,
                                    "click_x": x,
                                    "click_y": y,
                                    "attempt": attempt + 1
                                }
                            )
                        confirmed = self._confirm_selected_brawler(brawler_name, screenshot_func, backend)

                        # Set current brawler in play logic for brawler-specific strategies
                        if confirmed and self.play_logic:
                            try:
                                self.play_logic.set_current_brawler(brawler_name)
                                logger.info(f"[LOBBY] Brawler {brawler_name} definido no play_logic para estratégias específicas")
                            except Exception as e:
                                logger.warning(f"[LOBBY] Falha ao definir brawler no play_logic: {e}")

                        self._update_diagnostic(
                            "brawler_selected",
                            target=brawler_name,
                            attempt=attempt + 1,
                            backend=backend,
                            confirmed=confirmed,
                            card_position=position,
                        )
                        self._transition("idle")
                        return confirmed

                if layout_hint == "grid":
                    self._grid_navigation_step(attempt)
                else:
                    self._list_navigation_step(attempt)
                time.sleep(0.3)

            logger.warning(f"Brawler {brawler_name} not found after {self.max_scrolls} scroll attempts")
            self._update_diagnostic("brawler_not_found", target=brawler_name, attempts=self.max_scrolls)
            if log_manager:
                log_manager.log(
                    message=f"Brawler não encontrado após {self.max_scrolls} tentativas: {brawler_name}",
                    level="WARNING",
                    category="lobby",
                    data={"brawler_name": brawler_name, "attempts": self.max_scrolls, "success": False}
                )
            self._transition("idle")
            return False

        except Exception as e:
            logger.error(f"Error selecting brawler '{brawler_name}': {e}")
            self._update_diagnostic("brawler_selection_error", target=brawler_name, error=str(e))
            self._transition("idle")
            return False

    def scan_unlocked_brawlers(self, screenshot_func) -> List[str]:
        """
        Escaneia a tela de selecao de brawlers e retorna nomes dos desbloqueados.
        Atualiza a lista interna de brawlers bloqueados.
        """
        if not self._brawler_selector_fast:
            return []
        screenshot = screenshot_func()
        if screenshot is None:
            return []
        brawlers = self._brawler_selector_fast.scan_available_brawlers(screenshot)
        unlocked_names = [b["name"] for b in brawlers if b["unlocked"] and b["name"]]
        locked_names = [b["name"] for b in brawlers if not b["unlocked"] and b["name"]]
        if locked_names:
            logger.info(f"[LOBBY] Brawlers bloqueados detectados: {locked_names}")
        if unlocked_names:
            logger.info(f"[LOBBY] Brawlers desbloqueados visiveis: {unlocked_names}")
        return unlocked_names

    def mark_brawler_locked(self, brawler_name: str):
        """Marca um brawler como bloqueado na fila e no selector rapido."""
        # Marcar no selector rapido
        if self._brawler_selector_fast:
            self._brawler_selector_fast.mark_brawler_locked(brawler_name)
        # Desativar na fila
        for b in self.queue.brawlers:
            if b.name.lower() == brawler_name.lower():
                b.enabled = False
                logger.info(f"[LOBBY] Brawler {brawler_name} desativado na fila (bloqueado)")
                break

    @_retry_on_failure(max_retries=2, backoff=0.5)
    def select_current_brawler(self, screenshot_func, _depth=0) -> bool:
        """Seleciona o brawler atual da fila, pulando bloqueados"""
        if _depth > 10:
            logger.warning("[LOBBY] Profundidade maxima de recursao atingida - todos os brawlers podem estar bloqueados")
            self._transition("idle")
            return False

        self._transition("selecting_brawler")
        current = self.queue.get_current()
        if not current:
            logger.warning("Nenhum brawler na fila!")
            self._update_diagnostic("no_brawler_in_queue")
            self._transition("idle")
            return False

        # Verificar se o brawler atual esta na lista de bloqueados
        if (self._brawler_selector_fast and
                current.name.lower() in self._brawler_selector_fast.get_locked_brawlers()):
            logger.warning(f"[LOBBY] Brawler {current.name} esta bloqueado, avancando para o proximo")
            self._update_diagnostic("brawler_locked", target=current.name)
            next_brawler = self.queue.next()
            if next_brawler:
                return self.select_current_brawler(screenshot_func, _depth=_depth + 1)
            logger.warning("[LOBBY] Nenhum brawler desbloqueado na fila!")
            self._transition("idle")
            return False

        self._update_diagnostic("select_current_brawler", target=current.name, queue_index=self.queue.current_index)
        result = self.select_brawler(current.name, screenshot_func)

        # Se falhou e o selector rapido marcou como bloqueado, tentar proximo
        if not result and self._brawler_selector_fast:
            locked = self._brawler_selector_fast.get_locked_brawlers()
            if current.name.lower() in locked:
                logger.warning(f"[LOBBY] Selecao falhou - {current.name} bloqueado, tentando proximo")
                self.mark_brawler_locked(current.name)
                next_brawler = self.queue.next()
                if next_brawler:
                    return self.select_current_brawler(screenshot_func, _depth=_depth + 1)

        self._transition("idle")
        return result

    def check_and_switch_if_needed(self, progress_observer, screenshot_func) -> bool:
        """Verifica se deve trocar de brawler e troca se necessário"""
        if self.queue.check_goals():
            next_brawler = self.queue.peek_next()
            if next_brawler:
                logger.info(f"Trocando para {next_brawler.name}")
                if log_manager:
                    log_manager.log(
                        message=f"Trocando para brawler: {next_brawler.name}",
                        level="INFO",
                        category="lobby",
                        data={"from_brawler": self.queue.get_current().name if self.queue.get_current() else None, "to_brawler": next_brawler.name, "reason": "goals_achieved"}
                    )
                self._update_diagnostic("switch_brawler_start", target=next_brawler.name)
                if self.select_brawler(next_brawler.name, screenshot_func):
                    self.queue.next()
                    if log_manager:
                        log_manager.log(
                            message=f"Troca de brawler concluída: {next_brawler.name}",
                            level="INFO",
                            category="lobby",
                            data={"brawler_name": next_brawler.name, "success": True}
                        )
                    self._update_diagnostic("switch_brawler_done", target=next_brawler.name)
                    return True
                logger.warning(f"Falha ao selecionar {next_brawler.name}; fila não avançada")
                if log_manager:
                    log_manager.log(
                        message=f"Falha ao selecionar brawler: {next_brawler.name}",
                        level="WARNING",
                        category="lobby",
                        data={"brawler_name": next_brawler.name, "success": False}
                    )
                self._update_diagnostic("switch_brawler_failed", target=next_brawler.name)
        return False

    @_retry_on_failure(max_retries=3, backoff=0.5)
    def press_play(self):
        """
        Pressiona o botao Play com navegacao inteligente v2.

        Fluxo:
        1. Verificar popups e fechar
        2. Verificar evento ativo (Starr Nova, etc.)
        3. Encontrar botao Play via deteccao visual (multi-regiao)
        4. Clicar no Play encontrado
        5. Verificar se funcionou
        """
        self._transition("pressing_play")
        cfg = self.lobby_config
        logger.info("[LOBBY] press_play v2: navegacao inteligente iniciada")

        try:
            # === PASSO 1: Verificar e fechar popups ===
            screenshot = None
            if self._popup_manager and self._screenshot_func:
                screenshot = self._screenshot_func()
                if screenshot is not None:
                    popup = self._popup_manager.detect_popup(screenshot)
                    if popup:
                        logger.info(f"[LOBBY] Popup detectado: {popup.popup_type}, fechando...")
                        self._popup_manager.handle_popup(
                            popup,
                            click_func=self._click,
                            key_func=self._key_press
                        )
                        time.sleep(random.uniform(0.5, 1.0))
                        # Screenshot fresh after popup closed
                        screenshot = self._screenshot_func()

            # === PASSO 2: Verificar evento ativo ===
            if self._event_detector and screenshot is not None:
                event = self._event_detector.detect_event(screenshot)
                if event.event_active:
                    logger.info(f"[LOBBY] Evento ativo detectado: {event.event_type} (conf={event.confidence:.2f})")
                    if event.play_in_event_coords:
                        ex, ey = event.play_in_event_coords
                        logger.info(f"[LOBBY] Clicando 'Jogar' do evento em ({ex}, {ey})")
                        self._click(ex, ey)
                        time.sleep(random.uniform(1.0, 1.5))
                        if self._verify_state_changed("lobby"):
                            logger.info("[LOBBY] Evento iniciado com sucesso")
                            self._update_diagnostic("press_play_event", event_type=event.event_type)
                            self._transition("idle")
                            return True

            # === PASSO 2.5: Usar UnifiedStateDetector (mais preciso) ===
            if self._state_detector and screenshot is not None:
                try:
                    det = self._state_detector.detect(screenshot)
                    if det.state == 'lobby' and det.button_coords and det.confidence > 0.5:
                        ux, uy = det.button_coords
                        logger.info(f'[LOBBY] Play detectado via UnifiedStateDetector em ({ux},{uy}) conf={det.confidence:.2f}')
                        self._click(ux, uy)
                        time.sleep(random.uniform(1.0, 1.5))
                        if self._verify_state_changed('lobby'):
                            logger.info('[LOBBY] Play clicado com sucesso (UnifiedStateDetector)')
                            self._update_diagnostic('press_play_unified_success', coords=(ux,uy), conf=det.confidence)
                            self._transition('idle')
                            return True
                except Exception as e:
                    logger.debug(f'[LOBBY] UnifiedStateDetector falhou em press_play: {e}')

            # === PASSO 3: Encontrar botao Play com deteccao inteligente (fallback) ===
            if self._play_detector and screenshot is not None:
                play_result = self._play_detector.find_play_button(screenshot)
                if play_result.found and play_result.coords:
                    bx, by = play_result.coords
                    logger.info(f"[LOBBY] Play detectado via {play_result.region} em ({bx}, {by}) conf={play_result.confidence:.2f}")
                    # Jitter no clique para parecer humano
                    jitter_x = random.randint(-10, 10)
                    jitter_y = random.randint(-10, 10)
                    self._click(bx + jitter_x, by + jitter_y)
                    time.sleep(random.uniform(1.0, 1.5))

                    if self._verify_state_changed("lobby"):
                        logger.info("[LOBBY] Play clicado com sucesso (deteccao visual v2)")
                        self._update_diagnostic("press_play_v2_success", region=play_result.region, conf=play_result.confidence)
                        self._transition("idle")
                        return True

            # === FALLBACK: Metodo legado com coordenadas dinamicas ===
            logger.info("[LOBBY] Deteccao visual falhou, usando fallback legado")

            # Tentar múltiplas coordenadas de fallback (Play button pode estar em várias posições)
            fallback_coords = [
                (cfg.PLAY_BTN_X, cfg.PLAY_BTN_Y),  # Centro inferior
                (round(cfg.W * 0.85), round(cfg.H * 0.88)),  # Direita inferior (Starr Nova)
                (round(cfg.W * 0.75), round(cfg.H * 0.85)),  # Meio-direita
                (round(cfg.W * 0.5), round(cfg.H * 0.85)),   # Centro
            ]

            for attempt, (fx, fy) in enumerate(fallback_coords, 1):
                logger.info(f"[LOBBY] Fallback tentativa {attempt}/{len(fallback_coords)}: clicando em ({fx}, {fy})")
                self._click(fx, fy)
                time.sleep(random.uniform(1.0, 1.5))

                if self._verify_state_changed("lobby"):
                    logger.info(f"[LOBBY] Play funcionou (fallback {attempt})")
                    self._update_diagnostic("press_play_fallback", attempt=attempt, coords=(fx, fy))
                    self._transition("idle")
                    return True

            # === ULTIMA TENTATIVA: ESC + retry + múltiplas posições ===
            logger.info("[LOBBY] Tentando ESC + retry com múltiplas posições")
            self._key_press('esc')
            time.sleep(random.uniform(0.3, 0.6))

            # Tentar todas as posições novamente após ESC
            for attempt, (fx, fy) in enumerate(fallback_coords, 1):
                logger.info(f"[LOBBY] ESC+retry tentativa {attempt}: clicando em ({fx}, {fy})")
                self._click(fx, fy)
                time.sleep(random.uniform(1.0, 1.5))

                if self._verify_state_changed("lobby"):
                    logger.info(f"[LOBBY] Play funcionou apos ESC + retry ({attempt})")
                    self._update_diagnostic("press_play_esc_retry", attempt=attempt)
                    self._transition("idle")
                    return True

                # Se não funcionou, tentar ESC novamente entre cliques
                if attempt < len(fallback_coords):
                    self._key_press('esc')
                    time.sleep(random.uniform(0.2, 0.4))

            logger.warning("[LOBBY] press_play falhou apos TODAS as tentativas")
            self._update_diagnostic("press_play_all_failed")
            self._transition("idle")
            return False

        except Exception as e:
            logger.error(f"[LOBBY] press_play v2 falhou: {e}", exc_info=True)
            self._update_diagnostic("press_play_error", error=str(e))
            self._transition("idle")
            return False

    def select_game_mode(self, desired_mode: str) -> bool:
        """
        Navega para o modo de jogo desejado.

        Args:
            desired_mode: modo desejado (showdown, gem_grab, knockout, etc.)

        Returns:
            True se o slot correto esta ativo
        """
        if not LOBBY_EXPANDED_AVAILABLE or not self._slot_navigator:
            logger.warning("[LOBBY] EventSlotNavigator nao disponivel, pulando navegacao de modo")
            return False

        if not self._screenshot_func:
            logger.warning("[LOBBY] Sem funcao de screenshot, pulando navegacao de modo")
            return False

        if not desired_mode:
            logger.info("[LOBBY] Nenhum modo especificado, usando slot ativo")
            return True

        resolved = self._mode_resolver.resolve(desired_mode)
        logger.info(f"[LOBBY] Navegando para modo: {desired_mode} -> {resolved}")

        screenshot = self._screenshot_func()
        if screenshot is None:
            logger.warning("[LOBBY] Falha ao capturar screenshot para navegacao de modo")
            return False

        def click_func(x, y):
            self._click(x, y)

        def swipe_func(x1, y1, x2, y2, duration=0.35):
            if self.emulator_controller:
                self.emulator_controller.swipe_scaled(x1, y1, x2, y2, duration=duration)

        success = self._slot_navigator.navigate_to_mode(
            screenshot=screenshot,
            desired_mode=resolved,
            click_func=click_func,
            swipe_func=swipe_func,
            max_swipes=3,
        )

        if success:
            logger.info(f"[LOBBY] Modo '{resolved}' selecionado com sucesso")
        else:
            logger.warning(f"[LOBBY] Falha ao selecionar modo '{resolved}'")

        return success

    def _verify_state_changed(self, previous_state: str, timeout: float = 3.0) -> bool:
        """
        Verifica visualmente se o estado do jogo mudou após uma ação.
        Usa o detector unificado para confirmar que a ação funcionou.
        """
        if not self._state_detector or not self._screenshot_func:
            # Sem verificação visual, assumir sucesso
            logger.debug("[LOBBY] Sem verificação visual disponível")
            return True

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                screenshot = self._screenshot_func()
                if screenshot is not None:
                    detection = self._state_detector.detect(screenshot)
                    # FIXED: Consider any state change from lobby as valid, even "unknown",
                    # because matchmaking/loading screens may not be recognized yet.
                    if detection.state != previous_state:
                        logger.info(f"[LOBBY] Estado mudou: {previous_state} -> {detection.state}")
                        return True
            except Exception as e:
                logger.debug(f"[LOBBY] Erro na verificação visual: {e}")
            time.sleep(0.3)

        logger.debug(f"[LOBBY] Estado não mudou após {timeout}s")
        return False

    @_retry_on_failure(max_retries=2, backoff=0.3)
    def _swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300):
        """Executa swipe via EmulatorController."""
        if self.emulator_controller:
            self.emulator_controller.swipe_scaled(x1, y1, x2, y2, duration=duration)
        else:
            logger.debug("[LOBBY] Swipe nao disponivel (sem emulator_controller)")

    def close_popup(self):
        """Fecha popups usando PopupManager v2 (inteligente) ou fallback legado."""
        self._transition("closing_popup")
        cfg = self.lobby_config
        logger.info("[LOBBY] close_popup: detectando tipo de popup...")

        try:
            # === V2: Usar PopupManager para deteccao inteligente ===
            if self._popup_manager and self._screenshot_func:
                screenshot = self._screenshot_func()
                if screenshot is not None:
                    popup = self._popup_manager.detect_popup(screenshot)
                    if popup:
                        logger.info(f"[LOBBY] Popup v2: {popup.popup_type} -> acao={popup.action}")
                        closed = self._popup_manager.handle_popup(
                            popup, click_func=self._click, key_func=self._key_press
                        )
                        if closed:
                            self._transition("idle")
                            return True

            # === FALLBACK: Metodo legado ===
            logger.info("[LOBBY] Popup fallback: clicando X, centro, ESC")
            self._click(cfg.CLOSE_X, cfg.CLOSE_Y)
            time.sleep(random.uniform(0.3, 0.6))
            self._click(cfg.CENTER_X, cfg.CENTER_Y)
            time.sleep(random.uniform(0.3, 0.6))
            self._key_press('esc')
            self._transition("idle")
            return True

        except Exception as e:
            logger.warning(f"[LOBBY] close_popup error: {e}")
            self._transition("idle")
            return False
