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
        self.play_button = (round(w * 0.9419) + ox, round(h * 0.8949) + oy)

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
        self.player_hp_bar = (round(w * 0.08) + ox, round(h * 0.06) + oy)

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
    _PLAY_COLOR = (224, 186, 8)
    _LOAD_COLOR = (0, 1, 0)
    _PROCEED_COLOR = (35, 115, 255)
    _CONNECTION_LOST_COLOR = (66, 66, 66)
    _STAR_DROP_COLOR = (222, 72, 227)

    # NOVOS: Cores para estados adicionais
    _NEWS_CLOSE_COLOR = (255, 50, 50)       # Botao X vermelho/branco
    _SHOP_GOLD_COLOR = (255, 200, 50)       # Moedas douradas no topo
    _TUTORIAL_ARROW_COLOR = (50, 200, 255) # Setas azuis de tutorial
    _BRAWLER_UNLOCK_GOLD = (255, 215, 0)   # Texto dourado de unlock
    _SEASON_RESET_BLUE = (100, 180, 255)    # Azul season reset
    _PLAYER_HP_GREEN = (50, 255, 50)       # HP bar verde
    _TIMER_WHITE = (240, 240, 240)          # Timer branco no topo

    # Tolerâncias calibradas por tipo de deteção
    _TOLERANCES = {
        'defeated': 12,
        'play': 20,
        'load': 35,
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

    def __init__(self, images_path: Path, window_w: int = 1920, window_h: int = 1080):
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
            # OpenCV usa BGR, converter para RGB
            b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
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

    def _detect_by_pixels(self, image: np.ndarray) -> DetectedState:
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

        # 2. Loading (green pixel)
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

        # 3. Defeated (red corners)
        d1_frac = self._pixel_match_region(
            image, *sc(c.defeated1), self._DEFEATED_COLOR,
            tolerance=self._TOLERANCES['defeated']
        )
        d2_frac = self._pixel_match_region(
            image, *sc(c.defeated2), self._DEFEATED_COLOR,
            tolerance=self._TOLERANCES['defeated']
        )
        if (d1_frac > 0.6 or d2_frac > 0.6) and (screen_state_hint == "in_game" or screen_state_hint == "end"):
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

        # 5. Play button (main lobby, yellow)
        match_frac = self._pixel_match_region(
            image, *sc(c.play_button), self._PLAY_COLOR,
            tolerance=self._TOLERANCES['play']
        )
        if match_frac > 0.3:
            return DetectedState(
                state="lobby",
                confidence=match_frac,
                method="pixel",
                button_coords=sc(c.play_button),
                details={"sub_type": "play_button", "match_fraction": match_frac}
            )

        # 6. Proceed button (blue)
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

        # 7. Connection lost (gray)
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

        # 8. In-game heuristic: check for joystick area (dark region) + attack button area
        # If joystick area is mostly dark AND attack button area has distinctive color, we're in-game
        joy_x, joy_y = sc(c.joystick_center)
        atk_x, atk_y = sc(c.attack_button_center)
        hp_x, hp_y = sc(c.player_hp_bar)
        timer_x, timer_y = sc(c.match_timer)

        in_game_conf = 0.0
        in_game_details = {}

        if (0 < joy_x < w and 0 < joy_y < h and
            0 < atk_x < w and 0 < atk_y < h):
            # Sample joystick area - should be dark in-game
            joy_region = image[max(0,joy_y-20):min(h,joy_y+20),
                               max(0,joy_x-20):min(w,joy_x+20)]
            if joy_region.size > 0:
                joy_brightness = np.mean(joy_region)
                # Also check attack button area - should have distinctive color in-game
                atk_region = image[max(0,atk_y-30):min(h,atk_y+30),
                                   max(0,atk_x-30):min(w,atk_x+30)]
                atk_brightness = np.mean(atk_region) if atk_region.size > 0 else 255
                atk_std = np.std(atk_region) if atk_region.size > 0 else 0
                in_game_details["brightness"] = float(joy_brightness)
                in_game_details["atk_brightness"] = float(atk_brightness)
                in_game_details["atk_std"] = float(atk_std)
                # In-game: joystick area is dark (brightness < 80) AND attack area has some content
                if joy_brightness < 80:
                    in_game_conf = 0.3
                    if atk_region.size > 0 and atk_std > 20:
                        in_game_conf = 0.5

        # Verificar HP bar (verde no topo esquerdo) para confirmar in_game
        if 0 < hp_x < w and 0 < hp_y < h:
            hp_frac = self._pixel_match_region(
                image, hp_x, hp_y, self._PLAYER_HP_GREEN,
                tolerance=self._TOLERANCES['hp'], sample_radius=5
            )
            if hp_frac > 0.2:
                in_game_conf = max(in_game_conf, 0.6)
                in_game_details["hp_match"] = float(hp_frac)

        # Verificar timer no topo centro (branco)
        if 0 < timer_x < w and 0 < timer_y < h:
            timer_region = image[max(0,timer_y-8):min(h,timer_y+8),
                                 max(0,timer_x-25):min(w,timer_x+25)]
            if timer_region.size > 0:
                timer_brightness = np.mean(timer_region)
                timer_std = np.std(timer_region)
                # Timer: area pequena branca com texto (alto contraste)
                if timer_brightness > 180 and timer_std > 30:
                    in_game_conf = max(in_game_conf, 0.55)
                    in_game_details["timer_brightness"] = float(timer_brightness)
                    in_game_details["timer_std"] = float(timer_std)

        if in_game_conf >= 0.3:
            return DetectedState(
                state="in_game",
                confidence=in_game_conf,
                method="pixel",
                details={"sub_type": "joystick_heuristic", **in_game_details}
            )

        # 9. Tutorial (setas azuis no centro inferior)
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

        # 10. Shop (moedas douradas no topo direito)
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

        # 11. News/Brawl Talk (botao X vermelho no canto superior direito)
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

        # 12. Brawler unlock (texto dourado central)
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

        # 13. Season reset (azul no topo central)
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
        # 1. Check play button (lobby)
        play_region = self.region_data.get('play_button')
        found, conf, pos = self._template_match(
            image, 'play_button.png', play_region, threshold=0.5
        )
        if found and conf > 0.5:
            return DetectedState(
                state="lobby",
                confidence=conf,
                method="template",
                button_coords=pos,
                details={"template": "play_button", "position": pos}
            )

        
# 2. Check joystick first (in-game indicator) (in-game indicator)
        joystick_region = self.region_data.get('virtual_joystick')
        found, conf, pos = self._template_match(
            image, 'joystick.png', joystick_region, threshold=0.8
        )
        if found and conf > 0.4:
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

    def detect(self, image: np.ndarray, screen_hint: Optional[str] = None) -> DetectedState:
        """
        Detecção unificada: combina pixel matching + template matching + hint + smoothing.

        Fluxo:
        1. Pixel matching (rapido) → se alta confianca, usar
        2. Template matching (preciso) → confirmar ou refinar
        3. Screen hint (auxiliar) → apenas como desempate
        4. Smoothing (votacao) → evitar oscilacoes rapidas

        Retorna DetectedState com estado, confianca, metodo e coords do botao.
        """
        if image is None or image.size == 0:
            # Usar hint se disponivel
            if screen_hint:
                hinted = self._hint_to_state(screen_hint)
                if hinted:
                    hinted.method = "hint_only"
                    hinted.details["empty_image"] = True
                    return hinted
            return DetectedState(state="unknown", confidence=0.0, method="none",
                               details={"reason": "empty_image"})

        # Passo 1: Pixel matching (rapido)
        pixel_result = self._detect_by_pixels(image)

        # Se pixel matching tem alta confianca, usar diretamente
        if pixel_result.confidence > 0.5:
            smoothed = self._smooth_state(pixel_result.state, pixel_result.confidence)
            if smoothed != pixel_result.state:
                pixel_result.confidence *= 0.8  # Penalizar confianca quando smoothed
            pixel_result.state = smoothed
            logger.info(f"[UNIFIED_DETECTOR] Pixel match: {pixel_result.state} "
                        f"(conf={pixel_result.confidence:.2f}, "
                        f"sub={pixel_result.details.get('sub_type', 'unknown')})")
            self._record_detection(pixel_result)
            return pixel_result

        # Passo 2: Template matching (preciso)
        template_result = self._detect_by_templates(image)

        # Se template tem boa confianca, usar
        if template_result.confidence > 0.4:
            smoothed = self._smooth_state(template_result.state, template_result.confidence)
            if smoothed != template_result.state:
                template_result.confidence *= 0.8
            template_result.state = smoothed
            template_name = template_result.details.get("template", "unknown")
            logger.info(f"[UNIFIED_DETECTOR] Template match: {template_result.state} "
                        f"(conf={template_result.confidence:.2f}, template={template_name})")
            self._record_detection(template_result)
            return template_result

        # Passo 3: Combinar resultados
        # Se ambos detectaram algo, usar o de maior confianca
        if pixel_result.state != "unknown" and template_result.state != "unknown":
            if pixel_result.confidence >= template_result.confidence:
                result = pixel_result
                result.method = "combined_pixel"
            else:
                result = template_result
                result.method = "combined_template"
            smoothed = self._smooth_state(result.state, result.confidence)
            if smoothed != result.state:
                result.confidence *= 0.8
            result.state = smoothed
            logger.info(f"[UNIFIED_DETECTOR] Combined: {result.state} "
                        f"(conf={result.confidence:.2f}, method={result.method})")
            self._record_detection(result)
            return result

        # Se so um detectou, usar esse
        if pixel_result.state != "unknown":
            smoothed = self._smooth_state(pixel_result.state, pixel_result.confidence)
            if smoothed != pixel_result.state:
                pixel_result.confidence *= 0.8
            pixel_result.state = smoothed
            self._record_detection(pixel_result)
            return pixel_result
        if template_result.state != "unknown":
            smoothed = self._smooth_state(template_result.state, template_result.confidence)
            if smoothed != template_result.state:
                template_result.confidence *= 0.8
            template_result.state = smoothed
            self._record_detection(template_result)
            return template_result

        # Passo 4: Usar hint como ultimo recurso
        if screen_hint:
            hinted = self._hint_to_state(screen_hint)
            if hinted and hinted.state != "unknown":
                hinted.method = "hint"
                logger.info(f"[UNIFIED_DETECTOR] Hint fallback: {hinted.state}")
                self._record_detection(hinted)
                return hinted

        # Nenhuma deteccao
        result = DetectedState(state="unknown", confidence=0.0, method="none",
                              details={"pixel_state": pixel_result.state,
                                       "template_state": template_result.state})
        self._record_detection(result)
        return result

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
