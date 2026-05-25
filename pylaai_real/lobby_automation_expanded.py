"""
lobby_automation_expanded.py - Sistemas de automação expandida para o bot

Sistemas implementados:
1. EventSlotNavigator    - Navegação de slots de modo de jogo
2. PlayAgainHandler      - Click inteligente no botão Play Again
3. DailyRewardsCollector - Coleta automática de recompensas diárias
4. StarrRoadAutomation  - Navegação e coleta da Starr Road
5. ShopAutomation       - Automação básica da loja (itens gratuitos)
6. QuestAutomation      - Coleta de recompensas de missões
7. MaintenanceHandler    - Tratamento de tela de manutenção/update
8. ModeSelectionResolver - Utilitário para resolver nomes de modo

Integração:
- Adicionar EventSlotNavigator ao LobbyAutomator
- Adicionar PlayAgainHandler ao StateManager._handle_end_game
- Adicionar DailyRewardsCollector ao StateManager
- Adicionar MaintenanceHandler ao StateManager
"""

import time
import random
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import cv2
except ImportError:
    np = None
    cv2 = None


# ---------------------------------------------------------------------------
# ModeSelectionResolver
# ---------------------------------------------------------------------------

class ModeSelectionResolver:
    """Resolve e normaliza nomes de modos de jogo."""

    MODE_ALIASES = {
        "sd": "showdown",
        "solo": "showdown_solo",
        "solo_sd": "showdown_solo",
        "duo": "showdown_duo",
        "duo_sd": "showdown_duo",
        "showdown_solo": "showdown_solo",
        "showdown_duo": "showdown_duo",
        "gg": "gem_grab",
        "gemgrab": "gem_grab",
        "gems": "gem_grab",
        "gem_grab": "gem_grab",
        "bs": "bot_smac",
        "botsmac": "bot_smac",
        "bot_smac": "bot_smac",
        "ko": "knockout",
        "knockout": "knockout",
        "heist": "heist",
        "bounty": "bounty",
        "bb": "brawl_ball",
        "brawlball": "brawl_ball",
        "ball": "brawl_ball",
        "brawl_ball": "brawl_ball",
        "hz": "hot_zone",
        "hotzone": "hot_zone",
        "hot_zone": "hot_zone",
        "pl": "power_league",
        "powerleague": "power_league",
        "power_league": "power_league",
        "cl": "club_league",
        "clubleague": "club_league",
        "club_league": "club_league",
    }

    @classmethod
    def resolve(cls, mode: str) -> str:
        if not mode:
            return "showdown"
        normalized = mode.lower().replace(" ", "_").replace("-", "_")
        return cls.MODE_ALIASES.get(normalized, normalized)

    @classmethod
    def is_showdown_variant(cls, mode: str) -> bool:
        resolved = cls.resolve(mode)
        return "showdown" in resolved


# ---------------------------------------------------------------------------
# EventSlotNavigator
# ---------------------------------------------------------------------------

class EventSlotNavigator:
    """
    Navega pelos slots de modo de jogo no lobby.

    Deteta slots visiveis, classifica o modo pela cor, e clica
    no slot desejado se ainda não estiver ativo.
    """

    def __init__(self, images_path: Optional[Path] = None):
        self.images_path = Path(images_path) if images_path else None
        self._slot_cache: List[Tuple[int, int, int, int]] = []
        self._last_scan_time: float = 0
        self._cache_ttl: float = 1.5

    def navigate_to_mode(
        self,
        screenshot: np.ndarray,
        desired_mode: str,
        click_func: Callable[[int, int], None],
        swipe_func: Optional[Callable[[int, int, int, int, float], None]] = None,
        max_swipes: int = 3,
    ) -> bool:
        """
        Navega para o modo desejado.

        Args:
            screenshot: screenshot atual do lobby
            desired_mode: modo desejado (showdown, gem_grab, etc.)
            click_func: funcao(x, y) para clicar
            swipe_func: funcao(x1, y1, x2, y2, duration) para swipe
            max_swipes: maximo de swipes horizontais

        Returns:
            True se o slot correto está agora ativo
        """
        if screenshot is None or np is None:
            return False

        h, w = screenshot.shape[:2]
        target_mode = ModeSelectionResolver.resolve(desired_mode)

        for attempt in range(max_swipes + 1):
            slots = self._scan_slots(screenshot)

            for idx, (x, y, sw, sh) in enumerate(slots):
                slot_region = screenshot[y:y + sh, x:x + sw]
                mode = self._classify_slot(slot_region)
                if mode == target_mode:
                    center_x, center_y = x + sw // 2, y + sh // 2
                    is_active = self._is_slot_active(slot_region)

                    if is_active:
                        logger.info(f"[SLOT] Modo '{mode}' ja ativo em ({center_x}, {center_y})")
                        return True

                    logger.info(f"[SLOT] Clicando slot {idx} -> modo '{mode}' em ({center_x}, {center_y})")
                    click_func(center_x, center_y)
                    time.sleep(0.5)
                    return True

            if swipe_func and attempt < max_swipes:
                logger.info(f"[SLOT] Slot '{target_mode}' nao visivel, swipe para esquerda (tentativa {attempt + 1}/{max_swipes})")
                swipe_func(w // 2, h // 2, -300, 0, duration=0.35)
                time.sleep(0.7)

        logger.warning(f"[SLOT] Modo '{target_mode}' nao encontrado apos {max_swipes} swipes")
        return False

    def _scan_slots(self, screenshot: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Escaneia regioes provaveis de slots na tela."""
        if screenshot is None or cv2 is None:
            return []

        now = time.time()
        if now - self._last_scan_time < self._cache_ttl and self._slot_cache:
            return self._slot_cache

        h, w = screenshot.shape[:2]
        regions: List[Tuple[int, int, int, int]] = []

        gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = w * h * 0.008
        max_area = w * h * 0.18

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            aspect = cw / max(ch, 1)
            if min_area < area < max_area and 1.2 < aspect < 5.0 and ch > h * 0.04:
                regions.append((x, y, cw, ch))

        if not regions:
            sw = int(w * 0.16)
            sh = int(h * 0.08)
            base_y = int(h * 0.17)
            gap = int(w * 0.015)
            for i in range(6):
                sx = gap + i * (sw + gap)
                if sx + sw < w - gap:
                    regions.append((sx, base_y, sw, sh))

        regions.sort(key=lambda r: r[0])
        self._slot_cache = regions[:5]
        self._last_scan_time = now
        return self._slot_cache

    def _is_slot_active(self, slot_region: np.ndarray) -> bool:
        """Deteta se um slot esta ativo (cor brilhante/amarela)."""
        if slot_region is None or cv2 is None:
            return False
        hsv = cv2.cvtColor(slot_region, cv2.COLOR_RGB2HSV)
        yellow_mask = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 40) &
                       (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 140))
        return np.sum(yellow_mask) / max(yellow_mask.size, 1) > 0.01

    def _classify_slot(self, slot_region: np.ndarray) -> Optional[str]:
        """Classifica o modo de jogo de um slot pela cor dominante (HSV)."""
        if slot_region is None or cv2 is None or np is None:
            return None
        hsv = cv2.cvtColor(slot_region, cv2.COLOR_RGB2HSV)
        avg_h = hsv[:, :, 0].mean()
        avg_s = hsv[:, :, 1].mean()
        avg_v = hsv[:, :, 2].mean()

        if avg_s < 35:
            return None

        if 125 <= avg_h <= 175 and avg_s > 45:
            return "showdown"
        if 10 <= avg_h <= 50 and avg_s > 60 and avg_v > 130:
            return "gem_grab"
        if 160 <= avg_h <= 200 and avg_s > 40:
            return "knockout"
        if 40 <= avg_h <= 90 and avg_s > 55:
            return "hot_zone"
        if 0 <= avg_h <= 20 and avg_s > 75:
            return "bounty"
        if 10 <= avg_h <= 45 and avg_s > 70 and avg_v > 100:
            return "heist"
        if 85 <= avg_h <= 130 and avg_s > 50:
            return "brawl_ball"
        if 130 <= avg_h <= 175 and avg_s > 60:
            return "power_league"

        return None

    def get_active_mode(self, screenshot: np.ndarray) -> Optional[str]:
        """Retorna o modo ativo atualmente ou None."""
        if screenshot is None:
            return None
        for (x, y, sw, sh) in self._scan_slots(screenshot):
            if self._is_slot_active(screenshot[y:y + sh, x:x + sw]):
                return self._classify_slot(screenshot[y:y + sh, x:x + sw])
        return None

    def get_play_button_coords(self, screenshot: np.ndarray) -> Tuple[int, int]:
        """Retorna coordenadas do botao Play (slot ativo)."""
        if screenshot is None:
            return (0, 0)
        h, w = screenshot.shape[:2]
        return (int(w * 0.87), int(h * 0.90))


# ---------------------------------------------------------------------------
# PlayAgainHandler
# ---------------------------------------------------------------------------

@dataclass
class PlayAgainResult:
    success: bool
    clicked_play_again: bool
    clicked_proceed: bool
    exited_via_esc: bool
    method_used: str


class PlayAgainHandler:
    """
    Handler inteligente para o botao Play Again no end screen.

    Estrategia:
    1. Verificar screen automation para play_again / proceed
    2. Clicar no botao Play Again se detectado
    3. Clicar em proceed/continuar como segunda opcao
    4. Fallback: ESC ou cliques genericos
    """

    PLAY_AGAIN_COLORS = [
        ((15, 40, 140), (40, 100, 255)),
        ((20, 50, 150), (45, 110, 255)),
    ]
    PROCEED_COLORS = [
        ((200, 150, 0), (255, 210, 50)),
        ((180, 130, 0), (240, 190, 40)),
    ]

    def __init__(self, images_path: Optional[Path] = None):
        self.images_path = Path(images_path) if images_path else None

    def handle(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        screen_automation=None,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> PlayAgainResult:
        """
        Tenta sair do end screen usando a melhor estrategia disponivel.

        Returns:
            PlayAgainResult com detalhes do que foi feito
        """
        w, h = window_size
        result = PlayAgainResult(
            success=False,
            clicked_play_again=False,
            clicked_proceed=False,
            exited_via_esc=False,
            method_used="none",
        )

        if screenshot is None or emulator_controller is None:
            return result

        if screen_automation is not None:
            hint = None
            if hasattr(screen_automation, "get_current_state_name"):
                try:
                    hint = screen_automation.get_current_state_name()
                except Exception:
                    pass

            if hint == "play_again":
                logger.info("[PLAYAGAIN] ScreenAutomation detecta play_again, usando coordenadas do ScreenAutomation")
                if hasattr(screen_automation, "play_again_button"):
                    px, py = screen_automation.play_again_button
                    emulator_controller.tap_scaled(px, py)
                    result.clicked_play_again = True
                    result.success = True
                    result.method_used = "screen_automation_play_again"
                    return result

        if self._try_click_play_again(screenshot, emulator_controller, w, h):
            result.clicked_play_again = True
            result.success = True
            result.method_used = "pixel_detection_play_again"
            return result

        if self._try_click_proceed(screenshot, emulator_controller, w, h):
            result.clicked_proceed = True
            result.success = True
            result.method_used = "pixel_detection_proceed"
            return result

        emulator_controller.tap_scaled(w // 2, int(h * 0.90))
        time.sleep(0.2)
        emulator_controller.keyevent(4)
        time.sleep(0.3)
        emulator_controller.tap_scaled(int(w * 0.95), int(h * 0.05))
        result.exited_via_esc = True
        result.success = True
        result.method_used = "fallback_esc"
        return result

    def _try_click_play_again(
        self, screenshot: np.ndarray, emulator_controller, w: int, h: int
    ) -> bool:
        """Deteta e clica no botao Play Again pela cor."""
        if screenshot is None or cv2 is None:
            return False

        h_s, w_s = screenshot.shape[:2]
        scale_x = w / w_s
        scale_y = h / h_s

        bottom_area = screenshot[int(h_s * 0.80):h_s, int(w_s * 0.35):int(w_s * 0.75)]
        if bottom_area.size == 0:
            return False

        for (lower, upper) in self.PLAY_AGAIN_COLORS:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(cv2.cvtColor(bottom_area, cv2.COLOR_RGB2HSV), lower_np, upper_np)

            if np.sum(mask > 0) > 500:
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    cx = int(np.median(xs) * scale_x) + int(w_s * 0.35)
                    cy = int(np.median(ys) * scale_y) + int(h_s * 0.80)
                    logger.info(f"[PLAYAGAIN] Botao Play Again detectado em ({cx}, {cy})")
                    emulator_controller.tap_scaled(cx, cy)
                    return True

        return False

    def _try_click_proceed(
        self, screenshot: np.ndarray, emulator_controller, w: int, h: int
    ) -> bool:
        """Deteta e clica no botao Proceed/Continuar pela cor."""
        if screenshot is None or cv2 is None:
            return False

        h_s, w_s = screenshot.shape[:2]
        scale_x = w / w_s
        scale_y = h / h_s

        bottom_right = screenshot[int(h_s * 0.80):h_s, int(w_s * 0.70):w_s]
        if bottom_right.size == 0:
            return False

        for (lower, upper) in self.PROCEED_COLORS:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(cv2.cvtColor(bottom_right, cv2.COLOR_RGB2HSV), lower_np, upper_np)

            if np.sum(mask > 0) > 300:
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    cx = int(np.median(xs) * scale_x) + int(w_s * 0.70)
                    cy = int(np.median(ys) * scale_y) + int(h_s * 0.80)
                    logger.info(f"[PLAYAGAIN] Botao Proceed detectado em ({cx}, {cy})")
                    emulator_controller.tap_scaled(cx, cy)
                    return True

        return False


# ---------------------------------------------------------------------------
# DailyRewardsCollector
# ---------------------------------------------------------------------------

class DailyRewardsCollector:
    """
    Coleta automaticamente recompensas diarias.

    Deteta telas de login streak e clicks em "Claim".
    """

    CLAIM_COLORS = [((15, 50, 200), (40, 120, 255)), ((20, 80, 200), (50, 150, 255))]
    STREAK_BAR_COLORS = [((200, 180, 50), (255, 230, 120))]

    def __init__(self, images_path: Optional[Path] = None):
        self.images_path = Path(images_path) if images_path else None

    def try_collect(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> bool:
        """
        Tenta coletar recompensas diarias se estiver na tela apropriada.

        Returns:
            True se coletou alguma recompensa
        """
        if screenshot is None or emulator_controller is None or cv2 is None:
            return False

        w, h = window_size
        h_s, w_s = screenshot.shape[:2]
        scale_x = w / w_s
        scale_y = h / h_s

        claimed = False

        claim_pos = self._find_claim_button(screenshot, w, h, scale_x, scale_y)
        if claim_pos:
            emulator_controller.tap_scaled(*claim_pos)
            logger.info(f"[REWARDS] Clicou em Claim em {claim_pos}")
            time.sleep(0.5)
            claimed = True

        if self._is_streak_screen(screenshot):
            logger.info("[REWARDS] Tela de streak detectada")
            for _ in range(3):
                claim_pos = self._find_claim_button(screenshot, w, h, scale_x, scale_y)
                if claim_pos:
                    emulator_controller.tap_scaled(*claim_pos)
                    time.sleep(0.4)
                    claimed = True
                else:
                    break

        return claimed

    def _is_streak_screen(self, screenshot: np.ndarray) -> bool:
        """Deteta se estamos na tela de login streak."""
        if screenshot is None or cv2 is None or np is None:
            return False
        h_s, w_s = screenshot.shape[:2]
        center = screenshot[h_s // 3:2 * h_s // 3, w_s // 4:3 * w_s // 4]
        if center.size == 0:
            return False
        hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
        streak_mask = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 45) &
                       (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 120))
        return np.sum(streak_mask) / max(streak_mask.size, 1) > 0.08

    def _find_claim_button(
        self, screenshot: np.ndarray, w: int, h: int, scale_x: float, scale_y: float
    ) -> Optional[Tuple[int, int]]:
        """Encontra botao Claim pela cor."""
        if screenshot is None or cv2 is None or np is None:
            return None

        h_s, w_s = screenshot.shape[:2]

        for (lower, upper) in self.CLAIM_COLORS:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)

            mask = cv2.inRange(cv2.cvtColor(screenshot, cv2.COLOR_RGB2HSV), lower_np, upper_np)
            if np.sum(mask > 0) > 800:
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    cx = int(np.median(xs) * scale_x)
                    cy = int(np.median(ys) * scale_y)
                    return (cx, cy)

        return None


# ---------------------------------------------------------------------------
# StarrRoadAutomation
# ---------------------------------------------------------------------------

class StarrRoadAutomation:
    """
    Automacao da Starr Road.

    Navega pela Starr Road e coleta recompensas disponiveis.
    """

    COLLECT_COLORS = [((50, 200, 50), (100, 255, 100)), ((30, 180, 30), (80, 255, 80))]
    MILESTONE_COLORS = [((200, 150, 0), (255, 210, 80))]

    def __init__(self, images_path: Optional[Path] = None):
        self.images_path = Path(images_path) if images_path else None
        self._claimed_positions: set = set()

    def try_collect(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> bool:
        """
        Tenta coletar recompensas da Starr Road.

        Returns:
            True se coletou alguma recompensa
        """
        if screenshot is None or emulator_controller is None or cv2 is None:
            return False

        w, h = window_size
        h_s, w_s = screenshot.shape[:2]
        scale_x = w / w_s
        scale_y = h / h_s

        collected = False

        for (lower, upper) in self.COLLECT_COLORS:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(cv2.cvtColor(screenshot, cv2.COLOR_RGB2HSV), lower_np, upper_np)

            ys, xs = np.where(mask > 0)
            for i in range(len(xs)):
                cx = int(xs[i] * scale_x)
                cy = int(ys[i] * scale_y)
                pos_key = (cx // 50, cy // 50)
                if pos_key not in self._claimed_positions:
                    emulator_controller.tap_scaled(cx, cy)
                    logger.info(f"[STARR] Coleta em ({cx}, {cy})")
                    self._claimed_positions.add(pos_key)
                    time.sleep(0.3)
                    collected = True

        return collected

    def is_starr_road_screen(self, screenshot: np.ndarray) -> bool:
        """Deteta se estamos na tela da Starr Road."""
        if screenshot is None or cv2 is None or np is None:
            return False
        h_s, w_s = screenshot.shape[:2]
        top_area = screenshot[0:h_s // 4, w_s // 4:3 * w_s // 4]
        if top_area.size == 0:
            return False
        hsv = cv2.cvtColor(top_area, cv2.COLOR_RGB2HSV)
        star_mask = ((hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 150) &
                     (((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 45)) |
                      ((hsv[:, :, 0] >= 50) & (hsv[:, :, 0] <= 70))))
        return np.sum(star_mask) / max(star_mask.size, 1) > 0.05

    def reset(self):
        """Reset posicoes ja coletadas (nova sessao)."""
        self._claimed_positions.clear()


# ---------------------------------------------------------------------------
# ShopAutomation
# ---------------------------------------------------------------------------

class ShopAutomation:
    """
    Automacao basica da loja.

    Coleta itens diarios gratuitos e ofertas especiais.
    """

    FREE_ITEM_COLORS = [((100, 200, 50), (150, 255, 100)), ((80, 180, 30), (130, 255, 80))]
    COLLECT_COLORS = [((220, 180, 0), (255, 230, 80)), ((200, 150, 0), (250, 210, 60))]

    def __init__(self, images_path: Optional[Path] = None):
        self.images_path = Path(images_path) if images_path else None
        self._collected_today: set = set()

    def try_collect_free_items(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> bool:
        """
        Tenta coletar itens gratuitos na loja.

        Returns:
            True se coletou algum item
        """
        if screenshot is None or emulator_controller is None or cv2 is None:
            return False

        w, h = window_size
        h_s, w_s = screenshot.shape[:2]
        scale_x = w / w_s
        scale_y = h / h_s

        collected = False

        for (lower, upper) in self.COLLECT_COLORS:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(cv2.cvtColor(screenshot, cv2.COLOR_RGB2HSV), lower_np, upper_np)

            ys, xs = np.where(mask > 0)
            for i in range(len(xs)):
                cx = int(xs[i] * scale_x)
                cy = int(ys[i] * scale_y)
                pos_key = (cx // 80, cy // 80)
                if pos_key not in self._collected_today:
                    emulator_controller.tap_scaled(cx, cy)
                    logger.info(f"[SHOP] Coletou item gratuito em ({cx}, {cy})")
                    self._collected_today.add(pos_key)
                    time.sleep(0.4)
                    collected = True

        return collected

    def reset_daily(self):
        """Reset itens ja coletados (novo dia)."""
        self._collected_today.clear()


# ---------------------------------------------------------------------------
# QuestAutomation
# ---------------------------------------------------------------------------

class QuestAutomation:
    """
    Automacao de missoes/quests.

    Deteta missoes completadas e coleta recompensas.
    """

    CLAIM_COLORS = [((50, 180, 220), (100, 240, 255)), ((40, 160, 200), (90, 220, 250))]
    QUEST_TAB_COLORS = [((150, 100, 200), (200, 160, 255))]

    def __init__(self, images_path: Optional[Path] = None):
        self.images_path = Path(images_path) if images_path else None
        self._claimed_quests: set = set()

    def try_collect_quests(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> bool:
        """
        Tenta coletar recompensas de missoes completadas.

        Returns:
            True se coletou alguma recompensa
        """
        if screenshot is None or emulator_controller is None or cv2 is None:
            return False

        w, h = window_size
        h_s, w_s = screenshot.shape[:2]
        scale_x = w / w_s
        scale_y = h / h_s

        collected = False

        for (lower, upper) in self.CLAIM_COLORS:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(cv2.cvtColor(screenshot, cv2.COLOR_RGB2HSV), lower_np, upper_np)

            ys, xs = np.where(mask > 0)
            for i in range(len(xs)):
                cx = int(xs[i] * scale_x)
                cy = int(ys[i] * scale_y)
                pos_key = (cx // 60, cy // 60)
                if pos_key not in self._claimed_quests:
                    emulator_controller.tap_scaled(cx, cy)
                    logger.info(f"[QUEST] Coletou missao em ({cx}, {cy})")
                    self._claimed_quests.add(pos_key)
                    time.sleep(0.4)
                    collected = True

        return collected

    def is_quest_screen(self, screenshot: np.ndarray) -> bool:
        """Deteta se estamos na tela de missoes."""
        if screenshot is None or cv2 is None or np is None:
            return False
        h_s, w_s = screenshot.shape[:2]
        left_area = screenshot[0:h_s, 0:w_s // 4]
        if left_area.size == 0:
            return False
        hsv = cv2.cvtColor(left_area, cv2.COLOR_RGB2HSV)
        quest_mask = ((hsv[:, :, 0] >= 130) & (hsv[:, :, 0] <= 170) &
                      (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 100))
        return np.sum(quest_mask) / max(quest_mask.size, 1) > 0.03

    def reset(self):
        """Reset missoes ja coletadas."""
        self._claimed_quests.clear()


# ---------------------------------------------------------------------------
# MaintenanceHandler
# ---------------------------------------------------------------------------

class MaintenanceHandler:
    """
    Trata telas de manutencao e atualizacao obrigatoria.

    Deteta telas de "Update Required" e "Server Under Maintenance".
    """

    UPDATE_COLORS = [((200, 50, 50), (255, 100, 100))]
    MAINTENANCE_COLORS = [((150, 100, 50), (200, 150, 100))]
    UPDATE_TEXT_HUES = [(0, 15), (170, 180)]

    def __init__(self, images_path: Optional[Path] = None):
        self.images_path = Path(images_path) if images_path else None

    def detect(
        self,
        screenshot: np.ndarray,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> Tuple[bool, str]:
        """
        Deteta tela de manutencao/update.

        Returns:
            (is_maintenance, maintenance_type) onde type e "update", "maintenance" ou "none"
        """
        if screenshot is None or cv2 is None or np is None:
            return False, "none"

        h_s, w_s = screenshot.shape[:2]

        center = screenshot[h_s // 4:3 * h_s // 4, w_s // 4:3 * w_s // 4]
        if center.size == 0:
            return False, "none"

        hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)

        for (lower, upper) in self.UPDATE_COLORS:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            if np.sum(mask > 0) > 2000:
                return True, "update"

        for (lower, upper) in self.MAINTENANCE_COLORS:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            if np.sum(mask > 0) > 2000:
                return True, "maintenance"

        return False, "none"

    def handle(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> bool:
        """
        Trata a tela de manutencao.

        Returns:
            True se tratou a tela (clicou em algo ou saiu)
        """
        is_maint, maint_type = self.detect(screenshot, window_size)
        if not is_maint:
            return False

        w, h = window_size
        logger.warning(f"[MAINT] Tela de {maint_type} detectada")

        emulator_controller.keyevent(4)
        time.sleep(0.5)

        return True


# ---------------------------------------------------------------------------
# LobbyAutomationExpanded - Orquestrador de todos os sistemas
# ---------------------------------------------------------------------------

class LobbyAutomationExpanded:
    """
    Orquestrador de todos os sistemas de automacao expandida.

    Combina:
    - EventSlotNavigator (navegacao de modos)
    - PlayAgainHandler (end screen)
    - DailyRewardsCollector (recompensas diarias)
    - StarrRoadAutomation (Starr Road)
    - ShopAutomation (loja)
    - QuestAutomation (missoes)
    - MaintenanceHandler (manutencao/update)
    """

    def __init__(self, images_path: Optional[Path] = None):
        self.images_path = Path(images_path) if images_path else None
        self.slot_navigator = EventSlotNavigator(images_path)
        self.play_again_handler = PlayAgainHandler(images_path)
        self.daily_collector = DailyRewardsCollector(images_path)
        self.starr_road = StarrRoadAutomation(images_path)
        self.shop = ShopAutomation(images_path)
        self.quest = QuestAutomation(images_path)
        self.maintenance = MaintenanceHandler(images_path)

    def navigate_to_game_mode(
        self,
        screenshot: np.ndarray,
        desired_mode: str,
        emulator_controller,
    ) -> bool:
        """
        Navega para o modo de jogo desejado no lobby.

        Args:
            screenshot: screenshot atual do lobby
            desired_mode: modo desejado
            emulator_controller: controlador do emulador

        Returns:
            True se o modo esta ativo
        """
        return self.slot_navigator.navigate_to_mode(
            screenshot=screenshot,
            desired_mode=desired_mode,
            click_func=emulator_controller.tap_scaled,
            swipe_func=emulator_controller.swipe_scaled,
            max_swipes=3,
        )

    def handle_end_screen(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        screen_automation=None,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> PlayAgainResult:
        """Trata o end screen com estrategia inteligente."""
        return self.play_again_handler.handle(
            screenshot=screenshot,
            emulator_controller=emulator_controller,
            screen_automation=screen_automation,
            window_size=window_size,
        )

    def collect_daily_rewards(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> bool:
        """Coleta recompensas diarias se disponivel."""
        return self.daily_collector.try_collect(
            screenshot=screenshot,
            emulator_controller=emulator_controller,
            window_size=window_size,
        )

    def collect_starr_road(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> bool:
        """Coleta recompensas da Starr Road se disponivel."""
        return self.starr_road.try_collect(
            screenshot=screenshot,
            emulator_controller=emulator_controller,
            window_size=window_size,
        )

    def collect_shop_items(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> bool:
        """Coleta itens gratuitos da loja."""
        return self.shop.try_collect_free_items(
            screenshot=screenshot,
            emulator_controller=emulator_controller,
            window_size=window_size,
        )

    def collect_quest_rewards(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> bool:
        """Coleta recompensas de missoes."""
        return self.quest.try_collect_quests(
            screenshot=screenshot,
            emulator_controller=emulator_controller,
            window_size=window_size,
        )

    def handle_maintenance(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> bool:
        """Trata tela de manutencao/update."""
        return self.maintenance.handle(
            screenshot=screenshot,
            emulator_controller=emulator_controller,
            window_size=window_size,
        )

    def detect_maintenance(
        self,
        screenshot: np.ndarray,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> Tuple[bool, str]:
        """Deteta se ha tela de manutencao."""
        return self.maintenance.detect(screenshot, window_size)

    def reset_daily(self):
        """Reset collectores diarios (chamar a cada novo dia)."""
        self.shop.reset_daily()
        self.starr_road.reset()
        self.quest.reset()
