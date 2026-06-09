"""
unified_state_detector.py

Sistema UNIFICADO de deteção de estado do jogo.
Substitui o ScreenAutomation (thread conflituosa) e o StateFinder (template matching frágil)
por um único detector que combina:

1. Pixel color matching (do BrawlStarsBot) - rápido, funciona em qualquer resolução
2. Template matching (OpenCV) - preciso, robusto a variações
3. Screen automation hints - apenas como SINAL, nunca como AÇÃO

PRINCÍPIO: Um único ponto de decisão. Nunca dois sistemas a clicar em simultâneo.
"""

import time
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
import logging

try:
    from .ocr_state_detector import OCRStateDetector
except ImportError:
    OCRStateDetector = None

logger = logging.getLogger(__name__)


@dataclass
class DetectedState:
    """Resultado completo da deteção de estado."""
    state: str = "unknown"
    confidence: float = 0.0
    method: str = "none"  # "pixel", "template", "combined"
    map_name: Optional[str] = None
    button_coords: Optional[Tuple[int, int]] = None  # Coords do botão a clicar
    details: Dict = field(default_factory=dict)


class DynamicCoordinates:
    """
    Coordenadas dinâmicas que se adaptam a qualquer resolução.
    Todas as coordenadas são calculadas como % da janela, como o BrawlStarsBot faz.
    """

    def __init__(self, window_w: int = 1920, window_h: int = 1080,
                 offset_x: int = 0, offset_y: int = 0):
        self.w = window_w
        self.h = window_h
        self.offset_x = offset_x
        self.offset_y = offset_y
        self._compute()

    def update_resolution(self, window_w: int, window_h: int, offset_x: int = 0, offset_y: int = 0):
        """Atualiza resolução e recalcula todas as coordenadas."""
        self.w = window_w
        self.h = window_h
        self.offset_x = offset_x
        self.offset_y = offset_y
        self._compute()
        logger.info(f"[COORDS] Atualizado: {window_w}x{window_h} offset=({offset_x},{offset_y})")

    def _compute(self):
        """Calcula todas as coordenadas baseado na resolução atual."""
        w, h = self.w, self.h
        ox, oy = self.offset_x, self.offset_y

        # --- Pixel matching coordinates (from BrawlStarsBot) ---
        # Defeated corners (red pixels)
        self.defeated1 = (round(w * 0.9656) + ox, round(h * 0.152) + oy)
        self.defeated2 = (round(w * 0.993) + ox, round(h * 0.2046) + oy)

        # Star drop
        self.star_drop1 = (round(w * 0.488) + ox, round(h * 0.9303) + oy)
        self.star_drop2 = (round(w * 0.5228) + ox, round(h * 0.9296) + oy)

        # Play again button (end of match)
        self.play_again_button = (round(w * 0.5903) + ox, round(h * 0.9197) + oy)

        # Main play button (lobby)
        self.play_button = (round(w * 0.9119) + ox, round(h * 0.9122) + oy)

        # Exit button (when defeated)
        self.exit_button = (round(w * 0.493) + ox, round(h * 0.9187) + oy)

        # Loading indicator
        self.load_button = (round(w * 0.8057) + ox, round(h * 0.9675) + oy)

        # Proceed button (after match rewards)
        self.proceed_button = (round(w * 0.8093) + ox, round(h * 0.9165) + oy)

        # Connection lost
        self.connection_lost_cord = (round(w * 0.4912) + ox, round(h * 0.5525) + oy)
        self.reload_button = (round(w * 0.2824) + ox, round(h * 0.5812) + oy)

        # --- NOVOS: Estados adicionais (ANALISE_PROFUNDA) ---
        # News/Brawl Talk (botao X no canto superior direito, tipicamente vermelho/branco)
        self.news_close_button = (round(w * 0.973) + ox, round(h * 0.065) + oy)

        # Shop (icones de moedas/gemas no topo - area de deteccao)
        self.shop_icon_area = (round(w * 0.85) + ox, round(h * 0.04) + oy)

        # Tutorial (setas indicadoras, texto "Tap" - centro inferior)
        self.tutorial_tap_area = (round(w * 0.50) + ox, round(h * 0.85) + oy)

        # Brawler unlock (tela de desbloqueio - area central com texto dourado)
        self.brawler_unlock_area = (round(w * 0.50) + ox, round(h * 0.20) + oy)

        # Season reset (fundo especial, texto "Season")
        self.season_reset_area = (round(w * 0.50) + ox, round(h * 0.12) + oy)

        # Battle log (botao no canto superior esquerdo)
        self.battle_log_button = (round(w * 0.08) + ox, round(h * 0.08) + oy)

        # Social/Friends (icone de pessoas no topo)
        self.social_button = (round(w * 0.12) + ox, round(h * 0.04) + oy)

        # Settings (engrenagem no canto)
        self.settings_button = (round(w * 0.95) + ox, round(h * 0.04) + oy)

        # In-game: HP bar do jogador (topo esquerdo)
        self.player_hp_bar = (round(w * 0.08) + ox, round(h * 0.04) + oy)  # Calibrado: y=0.04 (43px @ 1080p)

        # In-game: Timer do match (topo centro)
        self.match_timer = (round(w * 0.50) + ox, round(h * 0.04) + oy)

        # --- Template matching regions (relative to window) ---
        # Play button search area
        self.play_button_region = (
            round(w * 0.40), round(h * 0.80),
            round(w * 0.60), round(h * 0.95)
        )

        # End of match area (bottom half)
        self.end_match_region = (
            round(w * 0.05), round(h * 0.70),
            round(w * 0.95), round(h * 0.98)
        )

        # Joystick area (bottom-left quadrant)
        self.joystick_region = (
            round(w * 0.02), round(h * 0.55),
            round(w * 0.20), round(h * 0.90)
        )

        # Brawler select area (center)
        self.brawler_select_region = (
            round(w * 0.20), round(h * 0.50),
            round(w * 0.80), round(h * 0.80)
        )

        # Attack button area (bottom-right)
        self.attack_button_region = (
            round(w * 0.80), round(h * 0.65),
            round(w * 0.98), round(h * 0.95)
        )

        # Super button area
        self.super_button_region = (
            round(w * 0.72), round(h * 0.70),
            round(w * 0.88), round(h * 0.90)
        )

        # Gadget button area
        self.gadget_button_region = (
            round(w * 0.75), round(h * 0.55),
            round(w * 0.88), round(h * 0.70)
        )

        # Virtual joystick center (for movement)
        self.joystick_center = (round(w * 0.10), round(h * 0.75))

        # Attack button center
        self.attack_button_center = (round(w * 0.90), round(h * 0.82))

    def update_window(self, w: int, h: int, offset_x: int = 0, offset_y: int = 0):
        """Atualiza coordenadas quando a janela muda de tamanho."""
        self.w = w
        self.h = h
        self.offset_x = offset_x
        self.offset_y = offset_y
        self._compute()
        logger.info(f"[COORDS] Atualizado: {w}x{h} offset=({offset_x},{offset_y})")

    def scale_to_emulator(self, x_1080: int, y_1080: int) -> Tuple[int, int]:
        """Converte coordenadas 1920x1080 para a resolução atual do emulador."""
        sx = round(x_1080 * self.w / 1920)
        sy = round(y_1080 * self.h / 1080)
        return (sx, sy)


    def scale_from_emulator(self, x: int, y: int) -> Tuple[int, int]:
        """Converte coordenadas da resolução atual do emulador para 1920x1080."""
        sx = round(x * 1920 / self.w) if self.w > 0 else x
        sy = round(y * 1080 / self.h) if self.h > 0 else y
        return (sx, sy)


class UnifiedStateDetector:
    """
    Detector unificado de estado do jogo.

    Combina pixel matching + template matching num único fluxo:
    1. Pixel matching (rápido) → detecta estados óbvios
    2. Template matching (preciso) → confirma ou refina
    3. Retorna estado + coordenadas do botão a clicar

    NUNCA clica automaticamente. Apenas detecta e informa.
    O StateManager decide quando e como agir.
    """

    # RGB reference colors (from BrawlStarsBot, calibrated)
    _DEFEATED_COLOR = (62, 0, 0)
    # UPDATED: Modern Brawl Stars Play button is yellow-orange gradient ~RGB(255,200,50)
    # Old value (224, 186, 8) was too dark for current UI
    _PLAY_COLOR = (240, 200, 10)      # Calibrated against real screenshots (2025 UI)
    _PLAY_COLOR_ALT = (230, 190, 5)   # Alternative shade for gradient variation
    _LOAD_COLOR = (50, 255, 50)  # Verde brilhante do spinner de loading
    _PROCEED_COLOR = (35, 115, 255)
    _CONNECTION_LOST_COLOR = (66, 66, 66)
    _STAR_DROP_COLOR = (222, 72, 227)

    # NOVOS: Cores para estados adicionais
    _NEWS_CLOSE_COLOR = (255, 50, 50)       # Botao X vermelho/branco
    _SHOP_GOLD_COLOR = (255, 200, 50)       # Moedas douradas no topo
    _TUTORIAL_ARROW_COLOR = (50, 200, 255) # Setas azuis de tutorial
    _BRAWLER_UNLOCK_GOLD = (255, 215, 0)   # Texto dourado de unlock
    _SEASON_RESET_BLUE = (100, 180, 255)    # Azul season reset
    _PLAYER_HP_BAR = (6, 140, 247)         # HP bar azul/ciano (Brawl Stars modern UI)
    _TIMER_WHITE = (240, 240, 240)          # Timer branco no topo

    # Tolerâncias calibradas por tipo de deteção
    _TOLERANCES = {
        'defeated': 12,
        'play': 35,  # Aumentado para acomodar gradiente amarelo-laranja moderno
        'load': 25,  # Tolerancia reduzida para evitar falsos positivos em pixels escuros
        'proceed': 30,
        'connection': 10,
        'star_drop': 20,
        'news': 25,
        'shop': 30,
        'tutorial': 25,
        'unlock': 25,
        'season': 20,
        'hp': 20,
        'timer': 15,
    }

    def __init__(self, images_path: Path, window_w: int = 1920, window_h: int = 1080, ocr_detector=None):
        self.images_path = images_path
        self.coords = DynamicCoordinates(window_w, window_h)
        self._template_cache: Dict[str, Optional[np.ndarray]] = {}
        self.region_data = self._load_regions()
        self.last_detection: DetectedState = DetectedState()
        self._detection_history: List[DetectedState] = []
        self._max_history = 50

        # Smoothing: evitar oscilacoes rapidas entre estados
        self._state_votes: Dict[str, int] = {}
        self._vote_window = 5  # Usar ultimas 5 detecoes para decidir
        self._min_votes_to_change = 3  # Precisa de 3/5 votos para mudar de estado
        self._current_stable_state = "unknown"
        self.ocr_detector = ocr_detector if ocr_detector is not None else (OCRStateDetector() if OCRStateDetector else None)

        # Matchmaking timeout intelligence
        self._matchmaking_start_time: Optional[float] = None
        self._ever_seen_lobby = False
        self._matchmaking_timeout_seconds = 15.0
        self._last_screenshot_size: Tuple[int, int] = (window_w, window_h)

        logger.info(f"[UNIFIED_DETECTOR] Inicializado: {window_w}x{window_h}, "
                     f"images_path={images_path}")

    def _load_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Carrega regiões de lobby.toml com fallback."""
        regions = {}
        lobby_toml_path = self.images_path.parent / "lobby.toml"
        if lobby_toml_path.exists():
            try:
                import toml
                config = toml.load(str(lobby_toml_path))
                for name, data in config.get("regions", {}).items():
                    if all(k in data for k in ("x1", "y1", "x2", "y2")):
                        regions[name] = (data["x1"], data["y1"], data["x2"], data["y2"])
                if regions:
                    logger.info(f"[UNIFIED_DETECTOR] {len(regions)} regiões de lobby.toml")
                    return regions
            except Exception as e:
                logger.warning(f"[UNIFIED_DETECTOR] Erro lobby.toml: {e}")

        # Fallback: usar coordenadas dinâmicas
        return {
            'thumbs_down': self.coords.end_match_region,
            'play_button': self.coords.play_button_region,
            'brawler_select': self.coords.brawler_select_region,
            'virtual_joystick': self.coords.joystick_region,
            'attack_button': self.coords.attack_button_region,
            'super_button': self.coords.super_button_region,
            'gadget_button': self.coords.gadget_button_region,
        }

    def update_window_size(self, w: int, h: int, offset_x: int = 0, offset_y: int = 0):
        """Atualiza coordenadas quando a janela muda."""
        self.coords.update_window(w, h, offset_x, offset_y)
        # Recarregar regiões com novas coordenadas
        self.region_data = self._load_regions()

    def _get_template(self, template_name: str) -> Optional[np.ndarray]:
        """Carrega template com cache."""
        if template_name in self._template_cache:
            return self._template_cache[template_name]

        template_path = self.images_path / template_name
        if not template_path.exists():
            self._template_cache[template_name] = None
            return None

        template = cv2.imread(str(template_path))
        if template is None:
            self._template_cache[template_name] = None
            return None

        self._template_cache[template_name] = template
        logger.debug(f"[UNIFIED_DETECTOR] Template carregado: {template_name}")
        return template

    def _pixel_match(self, image: np.ndarray, x: int, y: int,
                     expected_rgb: Tuple[int, int, int], tolerance: int = 15) -> bool:
        """
        Verifica se o pixel numa posição da imagem (screenshot) corresponde à cor esperada.
        Ao contrário do ScreenAutomation antigo que usava pyautogui.pixelMatchesColor
        (que lia o ecrã do PC), este método lê diretamente da screenshot capturada.
        """
        try:
            h, w = image.shape[:2]
            if y >= h or x >= w or y < 0 or x < 0:
                return False
            pixel = image[y, x]
            # Screenshots vêm da PIL/Win32 em RGB (não BGR)
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
            er, eg, eb = expected_rgb
            return (abs(r - er) <= tolerance and
                    abs(g - eg) <= tolerance and
                    abs(b - eb) <= tolerance)
        except (IndexError, TypeError):
            return False

    def _pixel_match_region(self, image: np.ndarray, x: int, y: int,
                            expected_rgb: Tuple[int, int, int],
                            tolerance: int = 15,
                            sample_radius: int = 3) -> float:
        """
        Verifica pixels numa pequena região e retorna a fração que corresponde.
        Mais robusto que um único pixel.
        """
        h, w = image.shape[:2]
        matches = 0
        total = 0
        for dy in range(-sample_radius, sample_radius + 1, 2):
            for dx in range(-sample_radius, sample_radius + 1, 2):
                px, py = x + dx, y + dy
                if 0 <= px < w and 0 <= py < h:
                    total += 1
                    if self._pixel_match(image, px, py, expected_rgb, tolerance):
                        matches += 1
        return matches / total if total > 0 else 0.0

    def _template_match(self, image: np.ndarray, template_name: str,
                        region: Optional[Tuple[int, int, int, int]] = None,
                        threshold: float = 0.7) -> Tuple[bool, float, Optional[Tuple[int, int]]]:
        """
        Template matching com retorno de confiança e posição.
        Retorna (found, confidence, center_coords).

        As regiões já são calculadas pela DynamicCoordinates para a resolução actual,
        por isso NÃO voltam a ser escaladas aqui.
        """
        template = self._get_template(template_name)
        if template is None:
            return False, 0.0, None

        h, w = image.shape[:2]

        # Escalar template para a resolução atual (templates são 1920x1080)
        scale = w / 1920.0
        if abs(scale - 1.0) > 0.01:
            tw = max(1, int(template.shape[1] * scale))
            th = max(1, int(template.shape[0] * scale))
            template = cv2.resize(template, (tw, th))

        # Recortar região se especificada
        # NOTA: as regiões já vêm escaladas pela DynamicCoordinates
        if region:
            rx1, ry1, rx2, ry2 = region
            # Clamp to image bounds
            rx1 = max(0, min(rx1, w))
            rx2 = max(rx1 + 1, min(rx2, w))
            ry1 = max(0, min(ry1, h))
            ry2 = max(ry1 + 1, min(ry2, h))
            search_img = image[ry1:ry2, rx1:rx2]
            offset_x, offset_y = rx1, ry1
        else:
            search_img = image
            offset_x, offset_y = 0, 0

        if search_img.size == 0:
            return False, 0.0, None

        th, tw = template.shape[:2]
        sh, sw = search_img.shape[:2]
        if th > sh or tw > sw:
            scale_fit = min(sw / tw, sh / th) * 0.9
            if scale_fit < 0.1:
                return False, 0.0, None
            template = cv2.resize(template, (max(1, int(tw * scale_fit)), max(1, int(th * scale_fit))))

        try:
            result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold:
                th, tw = template.shape[:2]
                center_x = offset_x + max_loc[0] + tw // 2
                center_y = offset_y + max_loc[1] + th // 2
                return True, max_val, (center_x, center_y)

            return False, max_val, None
        except cv2.error as e:
            logger.debug(f"[UNIFIED_DETECTOR] Template match cv2 error: {e}")
            return False, 0.0, None
        except Exception as e:
            logger.debug(f"[UNIFIED_DETECTOR] Template match error: {e}")
            return False, 0.0, None

    def _detect_by_pixels(self, image: np.ndarray, screen_hint: Optional[str] = None) -> DetectedState:
        """
        Detecção rápida por pixel matching (como BrawlStarsBot).
        Usa a screenshot capturada, não o ecrã do PC.
        As coordenadas já são calculadas pela DynamicCoordinates para a resolução actual.
        """
        c = self.coords
        h, w = image.shape[:2]

        # Scale coordinates to actual image size (in case screenshot differs from expected)
        def sc(coord):
            return (min(round(coord[0] * w / c.w), w - 1),
                    min(round(coord[1] * h / c.h), h - 1))

        # Pre-compute common regions to avoid NameError in later checks
        joy_x, joy_y = sc(c.joystick_center)
        atk_x, atk_y = sc(c.attack_button_center)
        hp_x, hp_y = sc(c.player_hp_bar)
        timer_x, timer_y = sc(c.match_timer)

        joy_region = None
        atk_region = None
        if (0 < joy_x < w and 0 < joy_y < h and
            0 < atk_x < w and 0 < atk_y < h):
            joy_region = image[max(0, joy_y - 20):min(h, joy_y + 20),
                               max(0, joy_x - 20):min(w, joy_x + 20)]
            atk_region = image[max(0, atk_y - 30):min(h, atk_y + 30),
                               max(0, atk_x - 30):min(w, atk_x + 30)]

        # --- HELPER: Check if lobby indicators are present ---
        def _lobby_indicators_present() -> Tuple[bool, Dict]:
            """Returns (True, details) if clear lobby UI elements are detected."""
            indicators = {}
            # Play button color (modern yellow-orange)
            play_match = self._pixel_match_region(
                image, sc(c.play_button)[0], sc(c.play_button)[1], self._PLAY_COLOR,
                tolerance=self._TOLERANCES['play'], sample_radius=6
            )
            play_match_alt = self._pixel_match_region(
                image, sc(c.play_button)[0], sc(c.play_button)[1], self._PLAY_COLOR_ALT,
                tolerance=self._TOLERANCES['play'], sample_radius=6
            )
            indicators['play_match'] = max(play_match, play_match_alt)

            # Shop gold icon in top-right
            shop_match = self._pixel_match_region(
                image, sc(c.shop_icon_area)[0], sc(c.shop_icon_area)[1], self._SHOP_GOLD_COLOR,
                tolerance=self._TOLERANCES['shop'], sample_radius=6
            )
            indicators['shop_match'] = shop_match

            # News close X in top-right
            news_match = self._pixel_match_region(
                image, sc(c.news_close_button)[0], sc(c.news_close_button)[1], self._NEWS_CLOSE_COLOR,
                tolerance=self._TOLERANCES['news'], sample_radius=4
            )
            indicators['news_match'] = news_match

            # Battle log button (top-left area, should be visible in lobby)
            battle_log_match = self._pixel_match_region(
                image, sc(c.battle_log_button)[0], sc(c.battle_log_button)[1], (255, 255, 255),
                tolerance=30, sample_radius=4
            )
            indicators['battle_log_match'] = battle_log_match

            # HSV-based play button detection (more robust to gradient)
            play_x, play_y = sc(c.play_button)
            hsv_play = self._detect_play_by_hsv(
                image,
                (max(0, play_x - 30), max(0, play_y - 30),
                 min(w, play_x + 30), min(h, play_y + 30)),
                v_min=150
            )
            indicators['hsv_play'] = hsv_play

            is_lobby = (
                indicators['play_match'] > 0.25 or
                indicators['hsv_play'] > 0.3 or
                indicators['shop_match'] > 0.25 or
                indicators['news_match'] > 0.3 or
                indicators['battle_log_match'] > 0.4
            )
            return is_lobby, indicators

        # 1. Play Again button (end of match, yellow)
        match_frac = self._pixel_match_region(
            image, *sc(c.play_again_button), self._PLAY_COLOR,
            tolerance=self._TOLERANCES['play']
        )
        if match_frac > 0.3:
            return DetectedState(
                state="end",
                confidence=match_frac,
                method="pixel",
                button_coords=sc(c.play_again_button),
                details={"sub_type": "play_again", "match_fraction": match_frac}
            )

        # 2. Defeated (red corners)
        d1_frac = self._pixel_match_region(
            image, *sc(c.defeated1), self._DEFEATED_COLOR,
            tolerance=self._TOLERANCES['defeated']
        )
        d2_frac = self._pixel_match_region(
            image, *sc(c.defeated2), self._DEFEATED_COLOR,
            tolerance=self._TOLERANCES['defeated']
        )
        # FIX NameError: use screen_hint parameter instead of undefined screen_state_hint
        hint_ok = screen_hint in ("in_game", "end") if screen_hint else False
        if (d1_frac > 0.6 or d2_frac > 0.6) and hint_ok:
            return DetectedState(
                state="end",
                confidence=max(d1_frac, d2_frac),
                method="pixel",
                button_coords=sc(c.exit_button),
                details={"sub_type": "defeated", "match_fraction": max(d1_frac, d2_frac)}
            )

        # 4. Star drop (purple)
        s1_frac = self._pixel_match_region(
            image, *sc(c.star_drop1), self._STAR_DROP_COLOR,
            tolerance=self._TOLERANCES['star_drop']
        )
        s2_frac = self._pixel_match_region(
            image, *sc(c.star_drop2), self._STAR_DROP_COLOR,
            tolerance=self._TOLERANCES['star_drop']
        )
        if s1_frac > 0.3 or s2_frac > 0.3:
            return DetectedState(
                state="end",
                confidence=max(s1_frac, s2_frac),
                method="pixel",
                button_coords=sc(c.star_drop1),
                details={"sub_type": "star_drop", "match_fraction": max(s1_frac, s2_frac)}
            )

        # 5. Play button (main lobby, yellow-orange) — CHECK LOBBY BEFORE in_game
        match_frac = self._pixel_match_region(
            image, *sc(c.play_button), self._PLAY_COLOR,
            tolerance=self._TOLERANCES['play']
        )
        match_frac_alt = self._pixel_match_region(
            image, *sc(c.play_button), self._PLAY_COLOR_ALT,
            tolerance=self._TOLERANCES['play']
        )
        play_match = max(match_frac, match_frac_alt)
        if play_match > 0.25:
            # Cross-check: if joystick is very dark AND HP bar exists, this is in-game
            joy_dark = joy_region is not None and joy_region.size > 0 and np.mean(joy_region) < 60
            hp_exists = False
            hp_frac_verify = 0.0
            if 0 < hp_x < w and 0 < hp_y < h:
                hp_frac_verify = self._pixel_match_region(
                    image, hp_x, hp_y, self._PLAYER_HP_BAR,
                    tolerance=self._TOLERANCES['hp'], sample_radius=5
                )
                hp_exists = hp_frac_verify > 0.20
            if joy_dark and hp_exists:
                return DetectedState(
                    state="in_game",
                    confidence=0.7,
                    method="pixel",
                    details={"sub_type": "play_button_override_in_game", "play_match": float(play_match), "hp_match": float(hp_frac_verify)}
                )
            return DetectedState(
                state="lobby",
                confidence=play_match,
                method="pixel",
                button_coords=sc(c.play_button),
                details={"sub_type": "play_button", "match_fraction": play_match}
            )

        # 5b. HSV fallback for lobby detection (catches gradient play buttons that RGB misses)
        play_x, play_y = sc(c.play_button)
        hsv_play_conf = self._detect_play_by_hsv(
            image,
            (max(0, play_x - 30), max(0, play_y - 30),
             min(w, play_x + 30), min(h, play_y + 30)),
            v_min=150
        )
        if hsv_play_conf > 0.35:
            # Verify not in-game
            joy_dark = joy_region is not None and joy_region.size > 0 and np.mean(joy_region) < 60
            hp_exists = False
            if 0 < hp_x < w and 0 < hp_y < h:
                hp_frac_verify = self._pixel_match_region(
                    image, hp_x, hp_y, self._PLAYER_HP_BAR,
                    tolerance=self._TOLERANCES['hp'], sample_radius=5
                )
                hp_exists = hp_frac_verify > 0.20
            if not (joy_dark and hp_exists):
                return DetectedState(
                    state="lobby",
                    confidence=hsv_play_conf,
                    method="pixel",
                    button_coords=sc(c.play_button),
                    details={"sub_type": "play_button_hsv", "match_fraction": hsv_play_conf}
                )

        # 6. In-game heuristic — only after lobby is ruled out
        in_game_conf = 0.0
        in_game_details = {}

        if joy_region is not None and joy_region.size > 0:
            joy_brightness = np.mean(joy_region)
            atk_brightness = np.mean(atk_region) if atk_region is not None and atk_region.size > 0 else 255
            atk_std = np.std(atk_region) if atk_region is not None and atk_region.size > 0 else 0
            in_game_details["brightness"] = float(joy_brightness)
            in_game_details["atk_brightness"] = float(atk_brightness)
            in_game_details["atk_std"] = float(atk_std)
            if joy_brightness < 60:
                in_game_conf = 0.35
                if atk_region is not None and atk_region.size > 0 and atk_std > 15:
                    in_game_conf = 0.55

        # Verificar HP bar (azul/ciano no topo esquerdo) para confirmar in_game
        if 0 < hp_x < w and 0 < hp_y < h:
            hp_frac = self._pixel_match_region(
                image, hp_x, hp_y, self._PLAYER_HP_BAR,
                tolerance=self._TOLERANCES['hp'], sample_radius=5
            )
            if hp_frac > 0.20:
                in_game_conf = max(in_game_conf, 0.65)
                in_game_details["hp_match"] = float(hp_frac)

        # Timer heuristic DISABLED — too many false positives in lobby
        # (lobby UI elements in top-center trigger brightness>100 and std>25)

        if in_game_conf >= 0.35:
            return DetectedState(
                state="in_game",
                confidence=in_game_conf,
                method="pixel",
                details={"sub_type": "joystick_heuristic", **in_game_details}
            )

        # 6b. HSV fallback for lobby detection (catches gradient play buttons that RGB misses)
        play_x, play_y = sc(c.play_button)
        hsv_play_conf = self._detect_play_by_hsv(
            image,
            (max(0, play_x - 30), max(0, play_y - 30),
             min(w, play_x + 30), min(h, play_y + 30)),
            v_min=150
        )
        if hsv_play_conf > 0.35:
            # Verify not in-game
            joy_dark = joy_region is not None and joy_region.size > 0 and np.mean(joy_region) < 150
            hp_exists = False
            if 0 < hp_x < w and 0 < hp_y < h:
                hp_frac_verify = self._pixel_match_region(
                    image, hp_x, hp_y, self._PLAYER_HP_BAR,
                    tolerance=self._TOLERANCES['hp'], sample_radius=5
                )
                hp_exists = hp_frac_verify > 0.15
            if not (joy_dark and hp_exists):
                return DetectedState(
                    state="lobby",
                    confidence=hsv_play_conf,
                    method="pixel",
                    button_coords=sc(c.play_button),
                    details={"sub_type": "play_button_hsv", "match_fraction": hsv_play_conf}
                )

        # 7. Proceed button (blue)
        match_frac = self._pixel_match_region(
            image, *sc(c.proceed_button), self._PROCEED_COLOR,
            tolerance=self._TOLERANCES['proceed']
        )
        if match_frac > 0.3:
            return DetectedState(
                state="end",
                confidence=match_frac,
                method="pixel",
                button_coords=sc(c.proceed_button),
                details={"sub_type": "proceed", "match_fraction": match_frac}
            )

        # 8. Connection lost (gray)
        match_frac = self._pixel_match_region(
            image, *sc(c.connection_lost_cord), self._CONNECTION_LOST_COLOR,
            tolerance=self._TOLERANCES['connection']
        )
        if match_frac > 0.3:
            return DetectedState(
                state="connection_lost",
                confidence=match_frac,
                method="pixel",
                button_coords=sc(c.reload_button),
                details={"sub_type": "connection_lost", "match_fraction": match_frac}
            )

        # 9. Loading (green spinner)
        match_frac = self._pixel_match_region(
            image, *sc(c.load_button), self._LOAD_COLOR,
            tolerance=self._TOLERANCES['load']
        )
        if match_frac > 0.2:
            return DetectedState(
                state="loading",
                confidence=match_frac,
                method="pixel",
                details={"sub_type": "loading", "match_fraction": match_frac}
            )

        # 10. Matchmaking detection: dark screen with loading spinner or player icons
        # CRITICAL FIX: Strong negative checks to exclude lobby before calling matchmaking.
        center_region = image[h // 3:2 * h // 3, w // 3:2 * w // 3]
        if center_region.size > 0:
            center_brightness = float(np.mean(center_region))
            center_std = float(np.std(center_region))

            # Strong negative check: if lobby UI indicators are present, this is NOT matchmaking
            is_lobby_ui, lobby_indicators = _lobby_indicators_present()
            if not is_lobby_ui:
                # NO green load indicator in the bottom-right area
                load_px = sc(c.load_button)
                load_match = self._pixel_match_region(
                    image, load_px[0], load_px[1], self._LOAD_COLOR,
                    tolerance=self._TOLERANCES['load'], sample_radius=6
                )

                # NO defeated red corners
                defeated_match = self._pixel_match_region(
                    image, sc(c.defeated1)[0], sc(c.defeated1)[1], self._DEFEATED_COLOR,
                    tolerance=self._TOLERANCES['defeated'], sample_radius=6
                )

                if (20 < center_brightness < 100 and center_std > 8 and
                        load_match < 0.15 and defeated_match < 0.3):
                    return DetectedState(
                        state="matchmaking",
                        confidence=0.45,
                        method="pixel",
                        details={"sub_type": "matchmaking_dark_screen",
                                 "center_brightness": center_brightness,
                                 "center_std": center_std,
                                 "load_match": float(load_match)}
                    )
            else:
                logger.debug(f"[UNIFIED_DETECTOR] Matchmaking heuristic rejected: lobby indicators present {lobby_indicators}")

        # 11. Tutorial (setas azuis no centro inferior)
        tut_frac = self._pixel_match_region(
            image, *sc(c.tutorial_tap_area), self._TUTORIAL_ARROW_COLOR,
            tolerance=self._TOLERANCES['tutorial'], sample_radius=8
        )
        if tut_frac > 0.2:
            return DetectedState(
                state="tutorial",
                confidence=tut_frac,
                method="pixel",
                button_coords=sc(c.tutorial_tap_area),
                details={"sub_type": "tutorial_arrow", "match_fraction": tut_frac}
            )

        # 12. Shop (moedas douradas no topo direito)
        shop_frac = self._pixel_match_region(
            image, *sc(c.shop_icon_area), self._SHOP_GOLD_COLOR,
            tolerance=self._TOLERANCES['shop'], sample_radius=6
        )
        if shop_frac > 0.25:
            return DetectedState(
                state="shop",
                confidence=shop_frac,
                method="pixel",
                button_coords=sc(c.shop_icon_area),
                details={"sub_type": "shop_gold_icon", "match_fraction": shop_frac}
            )

        # 13. News/Brawl Talk (botao X vermelho no canto superior direito)
        news_frac = self._pixel_match_region(
            image, *sc(c.news_close_button), self._NEWS_CLOSE_COLOR,
            tolerance=self._TOLERANCES['news'], sample_radius=4
        )
        if news_frac > 0.3:
            return DetectedState(
                state="news",
                confidence=news_frac,
                method="pixel",
                button_coords=sc(c.news_close_button),
                details={"sub_type": "news_close_x", "match_fraction": news_frac}
            )

        # 14. Brawler unlock (texto dourado central)
        unlock_frac = self._pixel_match_region(
            image, *sc(c.brawler_unlock_area), self._BRAWLER_UNLOCK_GOLD,
            tolerance=self._TOLERANCES['unlock'], sample_radius=10
        )
        if unlock_frac > 0.15:
            return DetectedState(
                state="brawler_unlock",
                confidence=unlock_frac,
                method="pixel",
                button_coords=sc(c.proceed_button),
                details={"sub_type": "unlock_gold_text", "match_fraction": unlock_frac}
            )

        # 15. Season reset (azul no topo central)
        season_frac = self._pixel_match_region(
            image, *sc(c.season_reset_area), self._SEASON_RESET_BLUE,
            tolerance=self._TOLERANCES['season'], sample_radius=8
        )
        if season_frac > 0.2:
            return DetectedState(
                state="season_reset",
                confidence=season_frac,
                method="pixel",
                button_coords=sc(c.proceed_button),
                details={"sub_type": "season_blue_bg", "match_fraction": season_frac}
            )

        return DetectedState(state="unknown", confidence=0.0, method="pixel")

    def _detect_by_templates(self, image: np.ndarray) -> DetectedState:
        """
        Detecção precisa por template matching.
        Mais lento mas mais robusto que pixel matching.
        Tenta múltiplos templates para cada estado.
        """
        # 1. Check play button (lobby) — PRIORITY over other templates
        play_region = self.region_data.get('play_button')
        found, conf, pos = self._template_match(
            image, 'play_button.png', play_region, threshold=0.30
        )
        if found and conf > 0.30:
            return DetectedState(
                state="lobby",
                confidence=conf,
                method="template",
                button_coords=pos,
                details={"template": "play_button", "position": pos}
            )

        # 2. Check joystick (in-game indicator) — HIGH threshold because
        # matchmaking overlay can still show joystick underneath
        joystick_region = self.region_data.get('virtual_joystick')
        found, conf, pos = self._template_match(
            image, 'joystick.png', joystick_region, threshold=0.80
        )
        if found and conf > 0.75:
            # Negative check: if play button is also detected, prefer lobby
            play_found, play_conf, _ = self._template_match(
                image, 'play_button.png', play_region, threshold=0.25
            )
            if play_found and play_conf > 0.25:
                logger.debug("[UNIFIED_DETECTOR] Joystick detected but play button also present -> lobby")
                return DetectedState(
                    state="lobby",
                    confidence=play_conf,
                    method="template",
                    button_coords=pos,
                    details={"template": "play_button_overrides_joystick", "position": pos}
                )
            return DetectedState(
                state="in_game",
                confidence=conf,
                method="template",
                details={"template": "joystick", "position": pos}
            )

        # 3. Check thumbs down (end of match)
        end_region = self.region_data.get('thumbs_down')
        found, conf, pos = self._template_match(
            image, 'thumbs_down.png', end_region, threshold=0.4
        )
        if found and conf > 0.4:
            return DetectedState(
                state="end",
                confidence=conf,
                method="template",
                button_coords=pos,
                details={"template": "thumbs_down", "position": pos}
            )

        # 4. Check brawler select
        brawler_region = self.region_data.get('brawler_select')
        found, conf, pos = self._template_match(
            image, 'brawler_select.png', brawler_region, threshold=0.4
        )
        if found and conf > 0.4:
            return DetectedState(
                state="brawler_selection",
                confidence=conf,
                method="template",
                details={"template": "brawler_select", "position": pos}
            )

        return DetectedState(state="unknown", confidence=0.0, method="template")

    def _extract_map_name(self, image: np.ndarray) -> Optional[str]:
        """Tenta extrair nome do mapa da screenshot via OCR ou heurísticas."""
        # TODO: Implementar OCR para nome do mapa
        # Por agora, usar heurísticas de cor da área do mapa
        return None

    def _smooth_state(self, detected_state: str, confidence: float) -> str:
        """
        Sistema de votação para evitar oscilacoes rapidas entre estados.
        So muda de estado se tiver votos suficientes na janela temporal.
        """
        # Atualizar votos
        self._state_votes[detected_state] = self._state_votes.get(detected_state, 0) + 1

        # Manter apenas votos da janela (remover o mais antigo se necessario)
        # Simplificacao: decrementar todos periodicamente (decay)
        if sum(self._state_votes.values()) > self._vote_window * 2:
            for k in list(self._state_votes.keys()):
                self._state_votes[k] = max(0, self._state_votes[k] - 1)
                if self._state_votes[k] == 0:
                    del self._state_votes[k]

        # Verificar se o estado detectado tem votos suficientes para mudar
        current_votes = self._state_votes.get(detected_state, 0)
        current_stable_votes = self._state_votes.get(self._current_stable_state, 0)

        # Se estado atual ainda tem mais votos, manter
        if (self._current_stable_state != detected_state and
            self._current_stable_state != "unknown" and
            current_stable_votes >= self._min_votes_to_change):
            # Mudanca requer que novo estado tenha mais votos que o atual
            if current_votes <= current_stable_votes:
                return self._current_stable_state

        # Se novo estado tem votos suficientes, atualizar
        if current_votes >= self._min_votes_to_change:
            if self._current_stable_state != detected_state:
                logger.info(f"[UNIFIED_DETECTOR] Estado estabilizado: "
                           f"{self._current_stable_state} -> {detected_state} "
                           f"(votes: {current_votes})")
                self._current_stable_state = detected_state
            return detected_state

        # Fallback: manter estado atual se tiver votos
        if self._current_stable_state != "unknown":
            return self._current_stable_state

        return detected_state

    def _finalize_detection(self, detection: DetectedState, log_prefix: str = "") -> DetectedState:
        """Aplica smoothing e regista a deteção no histórico."""
        smoothed = self._smooth_state(detection.state, detection.confidence)
        if smoothed != detection.state:
            detection.confidence *= 0.8
        detection.state = smoothed
        if log_prefix:
            logger.info(log_prefix + f" {detection.state} (conf={detection.confidence:.2f})")
        self._record_detection(detection)
        return detection

    def update_resolution(self, window_w: int, window_h: int):
        """Atualiza resolucao do detector e recalcula coordenadas dinamicas."""
        self.coords.update_resolution(window_w, window_h)
        logger.info(f"[UNIFIED_DETECTOR] Resolucao atualizada: {window_w}x{window_h}")

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """Detecta se imagem está em BGR (OpenCV padrão) e converte para RGB."""
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
        h, w = image.shape[:2]
        pb_x = min(round(self.coords.play_button[0] * w / self.coords.w), w - 1)
        pb_y = min(round(self.coords.play_button[1] * h / self.coords.h), h - 1)
        pixel = image[pb_y, pb_x].astype(np.float32)
        expected = np.array(self._PLAY_COLOR, dtype=np.float32)
        dist_rgb = np.linalg.norm(pixel - expected)
        bgr_expected = np.array([expected[2], expected[1], expected[0]], dtype=np.float32)
        dist_bgr = np.linalg.norm(pixel - bgr_expected)
        if dist_bgr < dist_rgb:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def detect(self, image: np.ndarray, screen_hint: Optional[str] = None) -> DetectedState:
        """
        Detecção unificada: combina pixel matching + template matching + OCR + hint + smoothing.

        Fluxo:
        1. Verificar resolução e recalibrar se necessário
        2. Pixel matching (rápido) → se alta confiança (>0.5), usar
        3. Template matching (preciso) → se alta confiança (>0.5), usar
        4. Se pixel_conf < 0.35 E template_conf < 0.35 → OCR fallback
        5. Screen hint (auxiliar) → apenas como desempate
        6. Smoothing (votação) → evitar oscilações rápidas

        Retorna DetectedState com estado, confiança, método e coords do botão.
        """
        image = self._ensure_rgb(image)
        if image is None or image.size == 0:
            if screen_hint:
                hinted = self._hint_to_state(screen_hint)
                if hinted:
                    hinted.method = "hint_only"
                    hinted.details["empty_image"] = True
                    return hinted
            return DetectedState(state="unknown", confidence=0.0, method="none",
                               details={"reason": "empty_image"})

        # --- RESOLUTION AUTO-CALIBRATION ---
        h, w = image.shape[:2]
        if (w, h) != self._last_screenshot_size:
            logger.info(f"[UNIFIED_DETECTOR] Screenshot size changed from {self._last_screenshot_size} to ({w}, {h}); recalibrating coordinates")
            self.update_window_size(w, h)
            self._last_screenshot_size = (w, h)

        # Passo 1: Pixel matching (rápido)
        pixel_result = self._detect_by_pixels(image, screen_hint=screen_hint)

        # Se pixel matching tem alta confiança, usar diretamente
        if pixel_result.confidence > 0.5:
            return self._finalize_detection(
                pixel_result,
                f"[UNIFIED_DETECTOR] Pixel match:"
                f" (sub={pixel_result.details.get('sub_type', 'unknown')}),"
            )

        # Passo 2: Template matching (preciso)
        template_result = self._detect_by_templates(image)

        # Se template tem alta confiança, usar diretamente
        if template_result.confidence > 0.5:
            template_name = template_result.details.get("template", "unknown")
            return self._finalize_detection(
                template_result,
                f"[UNIFIED_DETECTOR] Template match: (template={template_name}),"
            )

        # Extrair confianças efetivas (unknown conta como 0)
        pixel_conf = pixel_result.confidence if pixel_result.state != "unknown" else 0.0
        template_conf = template_result.confidence if template_result.state != "unknown" else 0.0

        # Passo 3: Se ambos detectaram o mesmo estado, combinar
        if pixel_result.state != "unknown" and pixel_result.state == template_result.state:
            combined = DetectedState(
                state=pixel_result.state,
                confidence=max(pixel_conf, template_conf),
                method="combined",
                button_coords=pixel_result.button_coords or template_result.button_coords,
                details={"pixel_conf": pixel_conf, "template_conf": template_conf,
                         "sub_type": pixel_result.details.get("sub_type", "unknown")}
            )
            return self._finalize_detection(combined, "[UNIFIED_DETECTOR] Combined match:")

        # Passo 4: Se pelo menos um tem confiança decente (>=0.35), usar o melhor
        if pixel_conf >= 0.35 or template_conf >= 0.35:
            if pixel_conf >= template_conf:
                return self._finalize_detection(pixel_result, "[UNIFIED_DETECTOR] Pixel match (low confidence):")
            else:
                return self._finalize_detection(template_result, "[UNIFIED_DETECTOR] Template match (low confidence):")

        # Passo 5: OCR fallback — ambos pixel e template abaixo de 0.35
        ocr_result = self._ocr_to_state(image)
        if ocr_result and ocr_result.confidence >= 0.35:
            logger.info(
                f"[UNIFIED_DETECTOR] OCR fallback triggered"
                f" (pixel_conf={pixel_conf:.2f}, template_conf={template_conf:.2f})"
                f" → {ocr_result.state} (conf={ocr_result.confidence:.2f})"
            )
            return self._finalize_detection(ocr_result, "[UNIFIED_DETECTOR] OCR fallback:")

        # Se OCR falhou mas existe um resultado não-unknown, usar o melhor disponível
        if pixel_result.state != "unknown":
            logger.debug(f"[UNIFIED_DETECTOR] Keeping pixel result despite low confidence ({pixel_conf:.2f}) after OCR miss")
            return self._finalize_detection(pixel_result)
        if template_result.state != "unknown":
            logger.debug(f"[UNIFIED_DETECTOR] Keeping template result despite low confidence ({template_conf:.2f}) after OCR miss")
            return self._finalize_detection(template_result)

        # Passo 6: SmartPlayButtonDetector fallback
        smart_result = self._smart_play_fallback(image)
        if smart_result and smart_result.state != "unknown":
            return self._finalize_detection(
                smart_result,
                f"[UNIFIED_DETECTOR] SmartPlay fallback: (coords={smart_result.button_coords}),"
            )

        # Passo 7: Verificar se o jogo está visível
        game_visible = self._check_game_visible(image)
        if not game_visible:
            result = DetectedState(
                state="unknown",
                confidence=0.0,
                method="none",
                details={
                    "pixel_state": pixel_result.state,
                    "template_state": template_result.state,
                    "game_visible": False,
                    "reason": "game_not_visible_or_dark_screen"
                }
            )
            self._record_detection(result)
            return result

        # Passo 8: Usar hint como último recurso
        if screen_hint:
            hinted = self._hint_to_state(screen_hint)
            if hinted and hinted.state != "unknown":
                hinted.method = "hint"
                return self._finalize_detection(hinted, "[UNIFIED_DETECTOR] Hint fallback:")

        # Nenhuma detecção
        result = DetectedState(state="unknown", confidence=0.0, method="none",
                              details={"pixel_state": pixel_result.state,
                                       "template_state": template_result.state,
                                       "ocr_state": ocr_result.state if ocr_result else "none"})
        self._record_detection(result)
        return result

    def _detect_play_by_hsv(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None, v_min: int = 80) -> float:
        """Detecta botão Play usando cor HSV (amarelo/laranja). Retorna confiança 0-1."""
        try:
            if region:
                rx1, ry1, rx2, ry2 = region
                h, w = image.shape[:2]
                rx1 = max(0, min(rx1, w))
                rx2 = max(rx1 + 1, min(rx2, w))
                ry1 = max(0, min(ry1, h))
                ry2 = max(ry1 + 1, min(ry2, h))
                search_img = image[ry1:ry2, rx1:rx2]
            else:
                search_img = image

            if search_img.size == 0:
                return 0.0

            hsv = cv2.cvtColor(search_img, cv2.COLOR_RGB2HSV)
            # Amarelo/laranja em HSV
            lower = np.array([15, 80, v_min])
            upper = np.array([40, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            total_pixels = mask.size
            yellow_pixels = cv2.countNonZero(mask)
            if total_pixels == 0:
                return 0.0

            ratio = yellow_pixels / total_pixels
            # Normalizar: um botão play típico ocupa ~5-20% da região
            if ratio < 0.02:
                return 0.0
            confidence = min(1.0, ratio / 0.15)
            return confidence
        except Exception as e:
            logger.debug(f"[UNIFIED_DETECTOR] _detect_play_by_hsv error: {e}")
            return 0.0

    def _smart_play_fallback(self, image: np.ndarray) -> Optional[DetectedState]:
        """
        Fallback que usa SmartPlayButtonDetector para encontrar o botao Play.
        Se encontrar, assume que estamos no lobby.
        """
        try:
            from pylaai_real.lobby_navigator import SmartPlayButtonDetector
            detector = SmartPlayButtonDetector(self.images_path)
            result = detector.find_play_button(image)
            if result.found and result.coords and result.confidence > 0.25:
                return DetectedState(
                    state="lobby",
                    confidence=result.confidence,
                    method="smart_play",
                    button_coords=result.coords,
                    details={
                        "sub_type": "play_button_found",
                        "region": result.region,
                        "screenshot_verified": result.screenshot_verified
                    }
                )
        except Exception as e:
            logger.debug(f"[UNIFIED_DETECTOR] SmartPlay fallback falhou: {e}")
        return None

    def _check_game_visible(self, image: np.ndarray) -> bool:
        """
        Heuristica para verificar se o jogo esta realmente visivel na screenshot.
        Detecta telas pretas, brancas, ou telas do emulador sem o jogo.
        """
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            return False

        # Verificar se o centro nao e completamente preto (pode ser tela de loading)
        center = image[h//2-50:h//2+50, w//2-50:w//2+50]
        center_brightness = float(np.mean(center))

        # Verificar se ha variacao de cor (tela preta/branca pura e invalida)
        std = float(np.std(image))

        # Verificar cantos - se todos os cantos forem identicos, pode ser screenshot congelada
        corners = [
            image[0:10, 0:10],
            image[0:10, w-10:w],
            image[h-10:h, 0:10],
            image[h-10:h, w-10:w]
        ]
        corner_means = [float(np.mean(c)) for c in corners]
        corner_variance = float(np.std(corner_means))

        # Regras:
        # - Se o centro for muito escuro (< 5) e std baixo -> tela preta
        # - Se std muito baixo (< 3) -> tela congelada ou uniforme
        # - Se cantos todos iguais (variance < 1) e std baixo -> screenshot invalida
        if center_brightness < 5 and std < 10:
            logger.warning("[UNIFIED_DETECTOR] Tela aparentemente preta - jogo nao visivel")
            return False
        if std < 3:
            logger.warning("[UNIFIED_DETECTOR] Screenshot uniforme - possivelmente congelada")
            return False
        if corner_variance < 1 and std < 15:
            logger.warning("[UNIFIED_DETECTOR] Cantos identicos e baixa variacao - screenshot suspeita")
            return False

        return True

    def _hint_to_state(self, hint: str) -> Optional[DetectedState]:
        """Converte screen hint em DetectedState."""
        if not hint:
            return None

        normalized = str(hint).strip().lower().replace(" ", "_").replace("-", "_")
        hint_map = {
            "play": "lobby",
            "idle": "lobby",
            "detecting": "matchmaking",
            "loading": "loading",
            "connection_lost": "connection_lost",
            "connection": "connection_lost",
            "exiting": "end",
            "play_again": "end",
            "playagain": "end",
            "star_drop": "end",
            "stardrop": "end",
            "proceed": "end",
            "game": "in_game",
        }

        mapped = hint_map.get(normalized)
        if mapped:
            return DetectedState(
                state=mapped,
                confidence=0.3,
                method="hint",
                details={"original_hint": hint}
            )
        return None

    def _ocr_to_state(self, image: np.ndarray) -> Optional[DetectedState]:
        """Usa OCR como fallback quando pixel/template não chegam com confiança suficiente."""
        if not self.ocr_detector:
            return None

        try:
            state_name, confidence = self.ocr_detector.detect_state_from_text(image)
        except Exception as e:
            logger.debug(f"[UNIFIED_DETECTOR] OCR fallback falhou: {e}")
            return None

        if not state_name or state_name == "unknown" or confidence < 0.3:
            return None

        mapped = self._hint_to_state(state_name)
        if mapped is None:
            mapped = DetectedState(state=state_name, confidence=confidence, method="ocr", details={})
        else:
            mapped.confidence = max(mapped.confidence, confidence)
            mapped.method = "ocr"

        if mapped.state in ("victory", "defeat"):
            mapped.details["ocr_signal"] = state_name
        return mapped

    def _record_detection(self, detection: DetectedState):
        """Guarda histórico de deteções para análise."""
        self.last_detection = detection
        self._detection_history.append(detection)
        if len(self._detection_history) > self._max_history:
            self._detection_history = self._detection_history[-self._max_history:]

    def get_diagnostic_report(self) -> Dict:
        """Retorna diagnóstico detalhado da última deteção."""
        d = self.last_detection
        return {
            "state": d.state,
            "confidence": d.confidence,
            "method": d.method,
            "map_name": d.map_name,
            "button_coords": d.button_coords,
            "details": d.details,
            "window_size": f"{self.coords.w}x{self.coords.h}",
            "detection_history_size": len(self._detection_history),
        }

    def get_state_name(self) -> str:
        """Retorna o nome do estado atual (compatibilidade com StateFinder)."""
        return self.last_detection.state

    # --- Compatibilidade com StateFinder antigo ---
    def get_state(self, image: np.ndarray, screen_state_hint: Optional[str] = None) -> str:
        """Interface compatível com StateFinder.get_state()."""
        result = self.detect(image, screen_hint=screen_state_hint)
        return result.state

    def is_in_end_of_match(self, image: np.ndarray, scale_factor: float = 1.0) -> bool:
        """Compatibilidade com StateFinder."""
        result = self.detect(image)
        return result.state == "end"

    def is_in_main_lobby(self, image: np.ndarray, scale_factor: float = 1.0) -> bool:
        """Compatibilidade com StateFinder."""
        result = self.detect(image)
        return result.state == "lobby"

    def is_in_game(self, image: np.ndarray, scale_factor: float = 1.0) -> bool:
        """Compatibilidade com StateFinder."""
        result = self.detect(image)
        return result.state == "in_game"

    def is_in_brawler_select(self, image: np.ndarray, scale_factor: float = 1.0) -> bool:
        """Compatibilidade com StateFinder."""
        result = self.detect(image)
        return result.state == "brawler_selection"

    # --- Métodos de busca visual para o LobbyAutomator ---

    def find_play_button(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Encontra a posição exacta do botão Play na screenshot.
        Usa template matching primeiro, depois pixel matching como fallback.
        Retorna (x, y) do centro do botão, ou None se não encontrado.
        """
        if image is None or image.size == 0:
            return None

        # 1. Template matching (mais preciso)
        play_region = self.region_data.get('play_button')
        found, conf, pos = self._template_match(
            image, 'play_button.png', play_region, threshold=0.5
        )
        if found and pos:
            logger.debug(f"[UNIFIED_DETECTOR] Play button encontrado por template: {pos} (conf={conf:.2f})")
            return pos

        # 2. Pixel matching fallback (procurar cor amarela na região do play)
        c = self.coords
        h, w = image.shape[:2]
        region = play_region or c.play_button_region
        rx1, ry1, rx2, ry2 = region
        # Clamp
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        search_area = image[ry1:ry2, rx1:rx2]
        if search_area.size == 0:
            return None

        # Procurar pixels com cor do Play button (amarelo)
        hsv = cv2.cvtColor(search_area, cv2.COLOR_RGB2HSV)
        # Amarelo em HSV: H=20-35, S=100-255, V=100-255
        mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Usar o maior contorno
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:  # Mínimo de área
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + rx1
                    cy = int(M["m01"] / M["m00"]) + ry1
                    logger.debug(f"[UNIFIED_DETECTOR] Play button encontrado por cor: ({cx}, {cy})")
                    return (cx, cy)

        # 3. Fallback: usar coordenada dinâmica
        logger.debug(f"[UNIFIED_DETECTOR] Play button não encontrado visualmente, usando coordenada dinâmica")
        return (c.play_button[0], c.play_button[1])

    def find_template_with_location(self, template_name: str, image: np.ndarray,
                                     region: Optional[Tuple[int, int, int, int]] = None,
                                     threshold: float = 0.5) -> Optional[Tuple[int, int]]:
        """
        Encontra a posição de um template na screenshot.
        Retorna (x, y) do centro ou None.
        """
        if image is None or image.size == 0:
            return None

        found, conf, pos = self._template_match(image, template_name, region, threshold)
        if found and pos:
            return pos
        return None

    def detect_attack_button(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """Encontra a posição do botão de ataque."""
        if image is None or image.size == 0:
            return None
        attack_region = self.region_data.get('attack_button')
        pos = self.find_template_with_location('attack_button.png', image, attack_region)
        if pos:
            return pos
        # Fallback: coordenada dinâmica
        return self.coords.attack_button_center

    def detect_super_button(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """Encontra a posição do botão Super (quando carregado)."""
        if image is None or image.size == 0:
            return None
        super_region = self.region_data.get('super_button')
        pos = self.find_template_with_location('super_button.png', image, super_region)
        return pos  # None se não encontrado (super não carregado)

    def detect_gadget_button(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """Encontra a posição do botão Gadget."""
        if image is None or image.size == 0:
            return None
        gadget_region = self.region_data.get('gadget_button')
        pos = self.find_template_with_location('gadget_button.png', image, gadget_region)
        return pos
