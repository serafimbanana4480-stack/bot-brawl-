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
from dataclasses import dataclass, field

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
# TrainingCaveNavigator
# ---------------------------------------------------------------------------

@dataclass
class TrainingCaveResult:
    success: bool
    entered: bool
    exited: bool
    method_used: str
    error: Optional[str] = None


class TrainingCaveNavigator:
    """
    Navega para a Training Cave e gerencia o modo de treino.

    Fluxo de entrada:
    1. No lobby, clicar no botao de menu (canto superior esquerdo)
    2. Clicar em "Training Cave" / "Treino"
    3. Selecionar brawler para teste
    4. Clicar em "Train"

    Fluxo de saida:
    1. Clicar no botao de pause (canto superior direito)
    2. Clicar em "Exit" / "Quit"
    3. Confirmar se necessario
    """

    # Cores caracteristicas da Training Cave
    CAVE_COLORS = {
        "training_button": [((100, 180, 50), (160, 255, 120))],  # Verde botao Train
        "pause_button": [((200, 200, 200), (255, 255, 255))],     # Branco pause
        "exit_button": [((200, 50, 50), (255, 100, 100))],        # Vermelho Exit
    }

    # Coordenadas normalizadas (0-1) para elementos da Training Cave
    TRAIN_BTN_PCT = (0.50, 0.88)
    PAUSE_BTN_PCT = (0.94, 0.06)
    EXIT_BTN_PCT = (0.50, 0.55)
    CONFIRM_EXIT_PCT = (0.60, 0.58)
    RESTART_BTN_PCT = (0.50, 0.50)

    def __init__(self, images_path: Optional[Path] = None):
        self.images_path = Path(images_path) if images_path else None

    def enter_training_cave(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> TrainingCaveResult:
        """
        Tenta entrar na Training Cave a partir do lobby.

        Estrategia:
        1. Clicar no menu (canto superior esquerdo)
        2. Procurar e clicar em "Training Cave" / "Practice"
        3. Clicar em "Train" para iniciar
        """
        w, h = window_size
        result = TrainingCaveResult(success=False, entered=False, exited=False, method_used="none")

        if screenshot is None or emulator_controller is None:
            result.error = "screenshot or controller missing"
            return result

        # Passo 1: Clicar no menu (canto superior esquerdo)
        menu_x, menu_y = int(w * 0.06), int(h * 0.08)
        logger.info(f"[TRAIN] Clicando menu em ({menu_x}, {menu_y})")
        emulator_controller.tap_scaled(menu_x, menu_y)
        time.sleep(1.0)

        # Passo 2: Procurar botao Training/Practice pela cor (verde)
        if cv2 is not None and np is not None:
            h_s, w_s = screenshot.shape[:2]
            scale_x = w / w_s
            scale_y = h / h_s

            # Area central-inferior onde geralmente esta o botao Training Cave
            center_bottom = screenshot[int(h_s * 0.35):int(h_s * 0.75), int(w_s * 0.15):int(w_s * 0.85)]
            if center_bottom.size > 0:
                hsv = cv2.cvtColor(center_bottom, cv2.COLOR_RGB2HSV)
                # Verde caracteristico do botao de treino
                green_mask = ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) &
                              (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 80))
                if np.sum(green_mask) > 500:
                    ys, xs = np.where(green_mask)
                    if len(xs) > 0:
                        cx = int(np.median(xs) * scale_x) + int(w_s * 0.15)
                        cy = int(np.median(ys) * scale_y) + int(h_s * 0.35)
                        logger.info(f"[TRAIN] Botao Training Cave detectado em ({cx}, {cy})")
                        emulator_controller.tap_scaled(cx, cy)
                        time.sleep(1.5)
                        result.entered = True
                        result.success = True
                        result.method_used = "menu_navigation"
                        return result

        # Fallback: clicar em coordenadas tipicas do botao Training Cave
        tx, ty = int(w * 0.50), int(h * 0.55)
        logger.info(f"[TRAIN] Fallback: clicando Training Cave em ({tx}, {ty})")
        emulator_controller.tap_scaled(tx, ty)
        time.sleep(1.5)

        # Clicar em Train
        train_x, train_y = int(w * self.TRAIN_BTN_PCT[0]), int(h * self.TRAIN_BTN_PCT[1])
        emulator_controller.tap_scaled(train_x, train_y)
        time.sleep(2.0)

        result.entered = True
        result.success = True
        result.method_used = "fallback_coords"
        return result

    def exit_training_cave(
        self,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> TrainingCaveResult:
        """
        Sai da Training Cave.

        Estrategia:
        1. Clicar no pause (canto superior direito)
        2. Clicar em Exit/Quit
        3. Confirmar se aparecer dialogo
        """
        w, h = window_size
        result = TrainingCaveResult(success=False, entered=False, exited=False, method_used="none")

        if emulator_controller is None:
            result.error = "controller missing"
            return result

        # Clicar pause
        px, py = int(w * self.PAUSE_BTN_PCT[0]), int(h * self.PAUSE_BTN_PCT[1])
        logger.info(f"[TRAIN] Clicando pause em ({px}, {py})")
        emulator_controller.tap_scaled(px, py)
        time.sleep(0.6)

        # Clicar Exit
        ex, ey = int(w * self.EXIT_BTN_PCT[0]), int(h * self.EXIT_BTN_PCT[1])
        logger.info(f"[TRAIN] Clicando exit em ({ex}, {ey})")
        emulator_controller.tap_scaled(ex, ey)
        time.sleep(0.6)

        # Confirmar se necessario
        cx, cy = int(w * self.CONFIRM_EXIT_PCT[0]), int(h * self.CONFIRM_EXIT_PCT[1])
        emulator_controller.tap_scaled(cx, cy)
        time.sleep(0.5)

        # ESC como garantia
        emulator_controller.keyevent(4)
        time.sleep(0.3)

        result.exited = True
        result.success = True
        result.method_used = "pause_exit"
        return result

    def is_in_training_cave(self, screenshot: np.ndarray) -> bool:
        """
        Deteta se estamos na Training Cave.

        Heuristicas:
        - Fundo claro/cinza (ambiente de treino)
        - Ausencia de timer de partida
        - Presenca de bots estáticos (inimigos que nao se movem)
        - Botao de pause visivel no canto superior direito
        """
        if screenshot is None or cv2 is None or np is None:
            return False

        h_s, w_s = screenshot.shape[:2]

        # Verificar area do pause button (canto superior direito)
        pause_area = screenshot[0:int(h_s * 0.15), int(w_s * 0.85):w_s]
        if pause_area.size == 0:
            return False

        hsv = cv2.cvtColor(pause_area, cv2.COLOR_RGB2HSV)
        # Branco/cinza claro do botao pause
        pause_mask = ((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 150))
        has_pause = np.sum(pause_mask) / max(pause_mask.size, 1) > 0.02

        # Verificar se ha joystick (in-game) mas nao ha timer de partida
        joystick_area = screenshot[int(h_s * 0.55):h_s, 0:int(w_s * 0.25)]
        has_joystick = False
        if joystick_area.size > 0:
            j_hsv = cv2.cvtColor(joystick_area, cv2.COLOR_RGB2HSV)
            # Cores do joystick (cinza/azulado)
            joy_mask = ((j_hsv[:, :, 1] < 60) & (j_hsv[:, :, 2] > 100))
            has_joystick = np.sum(joy_mask) / max(joy_mask.size, 1) > 0.03

        # Verificar ausencia de timer (area superior central)
        timer_area = screenshot[0:int(h_s * 0.12), int(w_s * 0.35):int(w_s * 0.65)]
        has_timer = False
        if timer_area.size > 0:
            t_hsv = cv2.cvtColor(timer_area, cv2.COLOR_RGB2HSV)
            # Timer geralmente branco com fundo escuro
            timer_mask = ((t_hsv[:, :, 1] < 30) & (t_hsv[:, :, 2] > 180))
            has_timer = np.sum(timer_mask) / max(timer_mask.size, 1) > 0.05

        # Training cave = joystick visivel + pause visivel + sem timer
        return has_joystick and has_pause and not has_timer

    def restart_training(
        self,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> bool:
        """Reinicia o treino na Training Cave."""
        if emulator_controller is None:
            return False

        w, h = window_size
        # Clicar pause
        px, py = int(w * self.PAUSE_BTN_PCT[0]), int(h * self.PAUSE_BTN_PCT[1])
        emulator_controller.tap_scaled(px, py)
        time.sleep(0.5)

        # Clicar restart (geralmente no centro)
        rx, ry = int(w * self.RESTART_BTN_PCT[0]), int(h * self.RESTART_BTN_PCT[1])
        emulator_controller.tap_scaled(rx, ry)
        time.sleep(2.0)

        return True


# ---------------------------------------------------------------------------
# PvEDetector - Deteta partidas PvE (vs bots)
# ---------------------------------------------------------------------------

@dataclass
class PvEClassification:
    is_pve: bool
    pve_type: Optional[str]  # "training_cave", "robo_rumble", "big_game", "boss_fight", "practice", "unknown"
    confidence: float
    details: Dict = field(default_factory=dict)


class PvEDetector:
    """
    Classifica se uma partida atual e PvE (Player vs Environment/Bots).

    Usa multiplas heuristicas:
    1. Nome do mapa/modo (Robo Rumble, Big Game, Boss Fight = PvE)
    2. Comportamento dos inimigos (bots = padroes repetitivos, movimentos previsiveis)
    3. Ausencia de nomes de jogador (bots tem nomes genericos)
    4. Training Cave (ambiente de treino)
    5. Modo Practice (amistoso contra bots)
    """

    PVE_MODES = {
        "robo_rumble", "big_game", "boss_fight", "training_cave",
        "practice", "bot_smac", "friendly",
    }

    BOT_NAME_PATTERNS = [
        "bot", "cpu", "training", "practice", "test",
        "player1", "player2", "player3",
    ]

    def __init__(self):
        self._enemy_history: List[Dict] = []
        self._history_max = 60  # frames

    def classify(
        self,
        screenshot: Optional[np.ndarray] = None,
        game_mode: Optional[str] = None,
        enemy_detections: Optional[List[Dict]] = None,
        player_hp: Optional[float] = None,
    ) -> PvEClassification:
        """
        Classifica se a partida atual e PvE.

        Args:
            screenshot: screenshot atual (opcional)
            game_mode: modo de jogo atual (ex: "robo_rumble")
            enemy_detections: lista de deteccoes de inimigos do YOLO
            player_hp: HP atual do jogador (0.0 - 1.0)

        Returns:
            PvEClassification com resultado
        """
        details: Dict[str, Any] = {}

        # 1. Verificar modo de jogo
        if game_mode:
            mode_normalized = game_mode.lower().replace(" ", "_").replace("-", "_")
            if mode_normalized in self.PVE_MODES:
                return PvEClassification(
                    is_pve=True,
                    pve_type=mode_normalized,
                    confidence=0.95,
                    details={"reason": "known_pve_mode", "mode": game_mode}
                )
            details["mode"] = game_mode

        # 2. Verificar Training Cave por screenshot
        if screenshot is not None and cv2 is not None:
            training_nav = TrainingCaveNavigator()
            if training_nav.is_in_training_cave(screenshot):
                return PvEClassification(
                    is_pve=True,
                    pve_type="training_cave",
                    confidence=0.90,
                    details={"reason": "training_cave_detected"}
                )

        # 3. Analisar comportamento dos inimigos
        bot_score = 0.0
        if enemy_detections:
            bot_score = self._analyze_enemy_behavior(enemy_detections)
            details["bot_behavior_score"] = round(bot_score, 2)

            if bot_score > 0.85:
                return PvEClassification(
                    is_pve=True,
                    pve_type="practice",
                    confidence=bot_score,
                    details={"reason": "bot_behavior", "score": bot_score}
                )

        # 4. Heuristica combinada
        confidence = bot_score
        if game_mode and "showdown" in game_mode.lower() and bot_score > 0.5:
            # Showdown com comportamento bot-like = provavelmente modo treino
            confidence = max(confidence, 0.70)
            return PvEClassification(
                is_pve=True,
                pve_type="practice",
                confidence=confidence,
                details={"reason": "showdown_bot_behavior", "score": bot_score}
            )

        return PvEClassification(
            is_pve=False,
            pve_type=None,
            confidence=1.0 - confidence,
            details=details
        )

    def _analyze_enemy_behavior(self, enemy_detections: List[Dict]) -> float:
        """
        Analisa padroes de comportamento dos inimigos para detectar bots.

        Bots tendem a:
        - Movimentos lineares (sem curvas humanas)
        - Parar subitamente
        - Nao usar cobertura
        - Padroes repetitivos de ataque
        """
        if not enemy_detections:
            return 0.0

        self._enemy_history.append({
            "timestamp": time.time(),
            "enemies": enemy_detections,
        })
        if len(self._enemy_history) > self._history_max:
            self._enemy_history.pop(0)

        if len(self._enemy_history) < 10:
            return 0.0  # Nao tem dados suficientes

        # Analisar padrao de movimento
        linear_moves = 0
        total_moves = 0
        sudden_stops = 0

        for i in range(1, len(self._enemy_history)):
            prev = self._enemy_history[i - 1]["enemies"]
            curr = self._enemy_history[i]["enemies"]

            for j in range(min(len(prev), len(curr))):
                p_pos = prev[j].get("center", (0, 0))
                c_pos = curr[j].get("center", (0, 0))

                if p_pos == (0, 0) or c_pos == (0, 0):
                    continue

                dx = c_pos[0] - p_pos[0]
                dy = c_pos[1] - p_pos[1]
                dist = (dx ** 2 + dy ** 2) ** 0.5

                if dist > 5:  # Movimento significativo
                    total_moves += 1
                    # Movimento muito linear (dx ou dy quase zero)
                    if abs(dx) < 2 or abs(dy) < 2:
                        linear_moves += 1
                    # Parada subita (distancia pequena apos movimento grande)
                    if i > 1:
                        pp_pos = self._enemy_history[i - 2]["enemies"][j].get("center", (0, 0))
                        if pp_pos != (0, 0):
                            prev_dist = ((p_pos[0] - pp_pos[0]) ** 2 + (p_pos[1] - pp_pos[1]) ** 2) ** 0.5
                            if prev_dist > 20 and dist < 3:
                                sudden_stops += 1

        if total_moves == 0:
            return 0.0

        linear_ratio = linear_moves / total_moves
        stop_ratio = sudden_stops / total_moves

        # Bots tem mais movimentos lineares e paradas subitas
        bot_score = linear_ratio * 0.4 + stop_ratio * 0.6
        return min(bot_score, 1.0)

    def reset(self):
        """Reseta historico de analise."""
        self._enemy_history.clear()


# ---------------------------------------------------------------------------
# FriendlyGameHandler
# ---------------------------------------------------------------------------

class FriendlyGameHandler:
    """
    Gerencia partidas amistosas e convites.

    Deteta:
    - Tela de convite de amigo
    - Sala de equipe (team room)
    - Ready status
    """

    INVITE_COLORS = [((50, 180, 50), (100, 255, 100))]   # Verde do botao Accept
    DECLINE_COLORS = [((200, 50, 50), (255, 100, 100))]  # Vermelho Decline

    def __init__(self, images_path: Optional[Path] = None):
        self.images_path = Path(images_path) if images_path else None

    def detect_invite(self, screenshot: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Deteta se ha um convite de partida amistosa.

        Returns:
            (has_invite, accept_coords, decline_coords)
        """
        if screenshot is None or cv2 is None or np is None:
            return False, None, None

        h_s, w_s = screenshot.shape[:2]
        center = screenshot[int(h_s * 0.25):int(h_s * 0.75), int(w_s * 0.20):int(w_s * 0.80)]
        if center.size == 0:
            return False, None, None

        hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)

        # Procurar botao verde (Accept)
        accept_coords = None
        for (lower, upper) in self.INVITE_COLORS:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            if np.sum(mask > 0) > 300:
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    cx = int(np.median(xs)) + int(w_s * 0.20)
                    cy = int(np.median(ys)) + int(h_s * 0.25)
                    accept_coords = (cx, cy)
                    break

        # Procurar botao vermelho (Decline)
        decline_coords = None
        for (lower, upper) in self.DECLINE_COLORS:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            if np.sum(mask > 0) > 300:
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    cx = int(np.median(xs)) + int(w_s * 0.20)
                    cy = int(np.median(ys)) + int(h_s * 0.25)
                    decline_coords = (cx, cy)
                    break

        has_invite = accept_coords is not None
        return has_invite, accept_coords, decline_coords

    def handle_invite(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        auto_accept: bool = False,
    ) -> bool:
        """
        Trata um convite de partida amistosa.

        Args:
            auto_accept: se True, aceita automaticamente; se False, recusa
        """
        has_invite, accept_coords, decline_coords = self.detect_invite(screenshot)
        if not has_invite:
            return False

        if auto_accept and accept_coords:
            logger.info(f"[FRIENDLY] Aceitando convite em {accept_coords}")
            emulator_controller.tap_scaled(*accept_coords)
        elif decline_coords:
            logger.info(f"[FRIENDLY] Recusando convite em {decline_coords}")
            emulator_controller.tap_scaled(*decline_coords)
        else:
            # Fallback: ESC para recusar
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
        self.training_cave = TrainingCaveNavigator(images_path)
        self.pve_detector = PvEDetector()
        self.friendly_handler = FriendlyGameHandler(images_path)

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

    def enter_training_cave(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> TrainingCaveResult:
        """Entra na Training Cave para treino de brawler."""
        return self.training_cave.enter_training_cave(
            screenshot=screenshot,
            emulator_controller=emulator_controller,
            window_size=window_size,
        )

    def exit_training_cave(
        self,
        emulator_controller,
        window_size: Tuple[int, int] = (1920, 1080),
    ) -> TrainingCaveResult:
        """Sai da Training Cave."""
        return self.training_cave.exit_training_cave(
            emulator_controller=emulator_controller,
            window_size=window_size,
        )

    def is_in_training_cave(self, screenshot: np.ndarray) -> bool:
        """Verifica se estamos na Training Cave."""
        return self.training_cave.is_in_training_cave(screenshot)

    def detect_pve(
        self,
        screenshot: Optional[np.ndarray] = None,
        game_mode: Optional[str] = None,
        enemy_detections: Optional[List[Dict]] = None,
    ) -> PvEClassification:
        """Classifica se a partida atual e PvE."""
        return self.pve_detector.classify(
            screenshot=screenshot,
            game_mode=game_mode,
            enemy_detections=enemy_detections,
        )

    def handle_friendly_invite(
        self,
        screenshot: np.ndarray,
        emulator_controller,
        auto_accept: bool = False,
    ) -> bool:
        """Trata convites de partidas amistosas."""
        return self.friendly_handler.handle_invite(
            screenshot=screenshot,
            emulator_controller=emulator_controller,
            auto_accept=auto_accept,
        )

    def reset_daily(self):
        """Reset collectores diarios (chamar a cada novo dia)."""
        self.shop.reset_daily()
        self.starr_road.reset()
        self.quest.reset()
        self.pve_detector.reset()
