"""
lobby_navigator.py - Navegacao Inteligente no Lobby

Sistema profissional para navegacao no lobby do Brawl Stars:
1. PopupManager - deteta e fecha popups automaticamente
2. SmartPlayButtonDetector - encontra o botao Play em qualquer situacao
3. EventDetector - identifica eventos ativos (Starr Nova, etc.)
4. BrawlerSelectorFast - selecao rapida de brawler com cache

Integrado no LobbyAutomator para fluxo completo lobby->jogo.
"""

import time
import random
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import numpy as np

try:
    import numpy as np
    import cv2
except ImportError:
    np = None  # type: ignore
    cv2 = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PopupManager - Deteta e fecha popups automaticamente
# ---------------------------------------------------------------------------

@dataclass
class PopupDetection:
    popup_type: str          # e.g., "reward", "news", "starr_drop", "unknown"
    confidence: float
    close_coords: Optional[Tuple[int, int]] = None  # onde clicar para fechar
    action: str = "click"    # "click", "esc", "swipe", "wait"
    details: Dict = None


class PopupManager:
    """
    Deteta e fecha popups de forma inteligente.
    Usa heuristica de pixels + template matching para identificar popups.
    """

    def __init__(self, images_path: Path):
        self.images_path = Path(images_path)
        self._template_cache: Dict[str, Optional[np.ndarray]] = {}
        self._last_popup_time = 0
        self._popup_cooldown = 1.5  # segundos entre checks

        # Cores caracteristicas de popups comuns
        self._popup_signatures = {
            "reward": {
                # Popup de recompensa: fundo escurecido + botao "Claim"
                "darkened_background": True,
                "center_bright": True,
                "close_button_top_right": True,
            },
            "news": {
                # Noticias: overlay escurecido, X no canto, texto central
                "top_right_x": True,
                "title_area": True,
            },
            "starr_drop": {
                # Starr Drop: roxo/lilas dominante, texto "Starr"
                "purple_dominant": True,
                "tap_to_open": True,
            },
            "brawl_pass": {
                # Passe de batalha: barra de progresso, cores vibrantes
                "progress_bar": True,
                "reward_track": True,
            },
        }

    def _load_template(self, name: str) -> Optional[np.ndarray]:
        if name in self._template_cache:
            return self._template_cache[name]
        path = self.images_path / name
        if path.exists() and cv2:
            img = cv2.imread(str(path))
            self._template_cache[name] = img
            return img
        self._template_cache[name] = None
        return None

    def detect_popup(self, screenshot: np.ndarray) -> Optional[PopupDetection]:
        """
        Analisa screenshot para detectar popup.
        Retorna PopupDetection ou None se nenhum popup encontrado.
        """
        if screenshot is None or cv2 is None or np is None:
            return None

        h, w = screenshot.shape[:2]

        # 1. Heuristica: popup tipicamente tem fundo escurecido (overlay)
        # Verificar cantos da tela (deveriam estar mais escuros se houver overlay)
        corner_brightness = self._sample_brightness(screenshot, 0, 0, w // 10, h // 10)
        center_brightness = self._sample_brightness(screenshot, w // 3, h // 3, 2 * w // 3, 2 * h // 3)

        # Se o centro e significativamente mais brilhante que os cantos -> overlay escurecido
        has_overlay = center_brightness > corner_brightness + 20

        if not has_overlay:
            # Sem overlay, provavelmente nao ha popup
            return None

        # 2. Verificar X no canto superior direito (popup comum)
        x_region = screenshot[0:h // 8, 7 * w // 8:w]
        x_found, x_conf = self._detect_close_x(x_region)
        if x_found and x_conf > 0.6:
            close_x = 15 * w // 16
            close_y = h // 16
            return PopupDetection(
                popup_type="generic_popup",
                confidence=x_conf,
                close_coords=(close_x, close_y),
                action="click",
                details={"has_x": True}
            )

        # 3. Verificar Starr Drop (cor roxa/lilas dominante)
        starr_conf = self._detect_starr_drop(screenshot)
        if starr_conf > 0.4:
            return PopupDetection(
                popup_type="starr_drop",
                confidence=starr_conf,
                close_coords=(w // 2, h // 2),  # Clicar centro para "abrir"
                action="click",
                details={"tap_to_open": True}
            )

        # 4. Verificar texto "CLAIM" ou "Collect" (recompensa)
        claim_conf = self._detect_claim_button(screenshot)
        if claim_conf > 0.5:
            # Recompensa - clicar no centro para coletar
            return PopupDetection(
                popup_type="reward",
                confidence=claim_conf,
                close_coords=(w // 2, h // 2),
                action="click",
                details={"reward_type": "unknown"}
            )

        # 5. Verificar se e news/noticia (titulo grande no centro)
        news_conf = self._detect_news(screenshot)
        if news_conf > 0.5:
            return PopupDetection(
                popup_type="news",
                confidence=news_conf,
                close_coords=(15 * w // 16, h // 16),
                action="click",
                details={}
            )

        # 6. Verificar botao verde (GOT IT!, OK, Continue)
        green_conf = self._detect_green_button(screenshot)
        if green_conf > 0.15:
            return PopupDetection(
                popup_type="green_button",
                confidence=green_conf,
                close_coords=(w // 2, 8 * h // 10),
                action="click",
                details={"button_type": "green"}
            )

        # 7. Verificar tela de evento (painel central, fundo escurecido)
        event_conf = self._detect_event_screen(screenshot)
        if event_conf > 0.5:
            # Se tambem ha botao verde, clicar nele; senao clicar centro
            if green_conf > 0.1:
                return PopupDetection(
                    popup_type="event_screen",
                    confidence=event_conf,
                    close_coords=(w // 2, 8 * h // 10),
                    action="click",
                    details={"has_green_button": True}
                )
            return PopupDetection(
                popup_type="event_screen",
                confidence=event_conf,
                close_coords=(w // 2, h // 2),
                action="click",
                details={}
            )

        # 8. Verificar overlay escuro forte (provavelmente popup modal)
        overlay_conf = self._detect_dark_overlay(screenshot)
        if overlay_conf > 0.7:
            return PopupDetection(
                popup_type="dark_overlay",
                confidence=overlay_conf,
                close_coords=(w // 2, h // 2),
                action="click",
                details={"dark_overlay": True}
            )

        # Popup desconhecido com overlay -> clicar centro como fallback
        return PopupDetection(
            popup_type="unknown",
            confidence=0.3,
            close_coords=(w // 2, h // 2),
            action="click",
            details={"has_overlay": True}
        )

    def handle_popup(self, popup: PopupDetection, click_func, key_func) -> bool:
        """
        Executa a acao apropriada para fechar o popup.
        Retorna True se conseguiu fechar.
        """
        if popup is None:
            return False

        if time.time() - self._popup_cooldown < self._last_popup_time:
            logger.debug("[POPUP] Cooldown ativo, skipping")
            return False

        self._last_popup_time = time.time()

        logger.info(f"[POPUP] Detectado: {popup.popup_type} (conf={popup.confidence:.2f}) -> acao={popup.action}")

        if popup.action == "click" and popup.close_coords:
            # Jitter no clique para parecer humano
            x, y = popup.close_coords
            jitter_x = random.randint(-15, 15)
            jitter_y = random.randint(-15, 15)
            click_func(x + jitter_x, y + jitter_y)
            time.sleep(random.uniform(0.4, 0.8))
            return True

        elif popup.action == "esc":
            key_func('esc')
            time.sleep(random.uniform(0.3, 0.6))
            return True

        return False

    def _sample_brightness(self, img, x1, y1, x2, y2) -> float:
        region = img[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if len(region.shape) == 3 else region
        return float(np.mean(gray))

    def _detect_close_x(self, region: np.ndarray) -> Tuple[bool, float]:
        """Detecta X de fechar no canto superior direito."""
        if region.size == 0:
            return False, 0.0
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        # X tipicamente tem bordas diagonais
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        # X tem ~15-30% de edges na regiao pequena
        if 0.05 < edge_ratio < 0.40:
            return True, edge_ratio * 2.5  # scale to 0-1
        return False, 0.0

    def _detect_starr_drop(self, screenshot: np.ndarray) -> float:
        """Detecta Starr Drop pela cor roxa/lilas dominante."""
        h, w = screenshot.shape[:2]
        center = screenshot[h//4:3*h//4, w//4:3*w//4]
        if center.size == 0:
            return 0.0
        hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
        # Roxo/lilas em HSV: H ~130-160
        purple_mask = ((hsv[:, :, 0] > 130) & (hsv[:, :, 0] < 170) &
                       (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 80))
        purple_ratio = np.sum(purple_mask) / purple_mask.size
        return purple_ratio

    def _detect_claim_button(self, screenshot: np.ndarray) -> float:
        """Detecta botao de recompensa (amarelo/dourado no centro inferior)."""
        h, w = screenshot.shape[:2]
        bottom_center = screenshot[3*h//4:h, w//3:2*w//3]
        if bottom_center.size == 0:
            return 0.0
        hsv = cv2.cvtColor(bottom_center, cv2.COLOR_RGB2HSV)
        # Amarelo/dourado em HSV
        yellow_mask = ((hsv[:, :, 0] > 20) & (hsv[:, :, 0] < 40) &
                       (hsv[:, :, 1] > 150) & (hsv[:, :, 2] > 150))
        yellow_ratio = np.sum(yellow_mask) / yellow_mask.size
        return yellow_ratio

    def _detect_news(self, screenshot: np.ndarray) -> float:
        """Detecta popup de noticias (overlay + texto grande)."""
        h, w = screenshot.shape[:2]
        # Verificar se ha texto grande no centro (noticias tipicamente tem titulo grande)
        center = screenshot[h//5:2*h//5, w//5:4*w//5]
        gray = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)
        # Texto tem alto contraste
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_ratio = np.sum(binary > 0) / binary.size
        # Noticias tem ~20-40% de texto na regiao do titulo
        if 0.15 < text_ratio < 0.45:
            return text_ratio * 1.5
        return 0.0



    def _detect_green_button(self, screenshot: np.ndarray) -> float:
        """Detecta botao verde tipo 'GOT IT!' no centro inferior."""
        h, w = screenshot.shape[:2]
        bottom_center = screenshot[7*h//10:h, w//4:3*w//4]
        if bottom_center.size == 0:
            return 0.0
        hsv = cv2.cvtColor(bottom_center, cv2.COLOR_RGB2HSV)
        # Verde brilhante em HSV (GOT IT!, OK, Continue)
        green_mask = ((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) &
                      (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 100))
        green_ratio = np.sum(green_mask) / green_mask.size
        return green_ratio

    def _detect_event_screen(self, screenshot: np.ndarray) -> float:
        """Detecta telas de evento (painel colorido no centro, fundo escurecido)."""
        h, w = screenshot.shape[:2]
        # Verificar se ha um painel claro no centro cercado por fundo escuro
        center_region = screenshot[h//4:3*h//4, w//4:3*w//4]
        edge_region = screenshot[0:h//10, 0:w//10]  # canto superior esquerdo
        if center_region.size == 0 or edge_region.size == 0:
            return 0.0
        center_gray = cv2.cvtColor(center_region, cv2.COLOR_RGB2GRAY)
        edge_gray = cv2.cvtColor(edge_region, cv2.COLOR_RGB2GRAY)
        center_mean = float(np.mean(center_gray))
        edge_mean = float(np.mean(edge_gray))
        # Painel central significativamente mais brilhante que os cantos
        if center_mean > edge_mean + 30:
            # Verificar se ha texto grande no centro (titulo do evento)
            _, binary = cv2.threshold(center_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text_ratio = np.sum(binary > 0) / binary.size
            if text_ratio > 0.1:
                return min(1.0, (center_mean - edge_mean) / 100.0)
        return 0.0

    def _detect_dark_overlay(self, screenshot: np.ndarray) -> float:
        """Detecta overlay escuro (fundo do jogo escurecido por modal)."""
        h, w = screenshot.shape[:2]
        # Verificar 4 cantos - se estao todos escuros, e overlay
        corners = [
            screenshot[0:h//10, 0:w//10],
            screenshot[0:h//10, 9*w//10:w],
            screenshot[9*h//10:h, 0:w//10],
            screenshot[9*h//10:h, 9*w//10:w],
        ]
        corner_means = []
        for corner in corners:
            if corner.size > 0:
                gray = cv2.cvtColor(corner, cv2.COLOR_RGB2GRAY) if len(corner.shape) == 3 else corner
                corner_means.append(float(np.mean(gray)))
        if not corner_means:
            return 0.0
        avg_corner = sum(corner_means) / len(corner_means)
        # Cantos escuros (< 80) indicam overlay
        if avg_corner < 80:
            return 1.0 - (avg_corner / 80.0)
        return 0.0

# ---------------------------------------------------------------------------
# SmartPlayButtonDetector - Encontra Play em qualquer situacao
# ---------------------------------------------------------------------------

@dataclass
class PlayButtonResult:
    found: bool
    coords: Optional[Tuple[int, int]]
    region: str          # "primary", "event", "spectator", "event_screen", etc.
    confidence: float
    screenshot_verified: bool = False


class SmartPlayButtonDetector:
    """
    Encontra o botao Play em qualquer situacao do lobby.
    Usa multiplas regioes + template matching + heuristica de cor.
    """

    def __init__(self, images_path: Path):
        self.images_path = Path(images_path)
        self._template_cache: Dict[str, Optional[np.ndarray]] = {}
        self._last_known_play_pos: Optional[Tuple[int, int]] = None
        self._last_known_time = 0
        self._position_ttl = 300  # segundos

    def _load_template(self, name: str) -> Optional[np.ndarray]:
        if name in self._template_cache:
            return self._template_cache[name]
        path = self.images_path / name
        if path.exists() and cv2:
            img = cv2.imread(str(path))
            self._template_cache[name] = img
            return img
        self._template_cache[name] = None
        return None

    def find_play_button(self, screenshot: np.ndarray) -> PlayButtonResult:
        """
        Procura botao Play em multiplas regioes possiveis.
        Retorna coordenadas + confianca + regiao onde foi encontrado.
        """
        if screenshot is None or cv2 is None or np is None:
            logger.warning("[PLAY_BTN] Screenshot ou CV2/Numpy indisponivel")
            return PlayButtonResult(found=False, coords=None, region="none", confidence=0.0)

        h, w = screenshot.shape[:2]
        logger.debug(f"[PLAY_BTN] Procurando em screenshot {w}x{h}")

        # 1. Verificar cache de posicao recente
        if self._last_known_play_pos and (time.time() - self._last_known_time) < self._position_ttl:
            lx, ly = self._last_known_play_pos
            if 0 <= lx < w and 0 <= ly < h:
                region = screenshot[max(0, ly-20):min(h, ly+20), max(0, lx-30):min(w, lx+30)]
                if self._is_yellow_play_button(region):
                    logger.info(f"[PLAY_BTN] Encontrado via cache em ({lx}, {ly})")
                    return PlayButtonResult(
                        found=True, coords=(lx, ly), region="cached",
                        confidence=0.7, screenshot_verified=True
                    )

        # 2. Template matching em multiplas regioes
        regions = [
            (0.82, 0.78, 1.0, 1.0, "primary"),      # Lobby normal (canto inferior direito)
            (0.70, 0.72, 0.95, 0.95, "event"),     # Evento ativo (Play pode estar mais a esquerda)
            (0.55, 0.75, 0.80, 1.0, "event_screen"), # Dentro da tela de evento
            (0.40, 0.80, 0.70, 1.0, "spectator"),  # Modo espectador
        ]

        best_result = None
        best_conf = 0.0
        template = self._load_template("play_button_real.png")
        if template is None:
            logger.warning("[PLAY_BTN] Template 'play_button_real.png' NAO ENCONTRADO")
        else:
            logger.debug(f"[PLAY_BTN] Template carregado: {template.shape}")

        for rx1_pct, ry1_pct, rx2_pct, ry2_pct, label in regions:
            rx1 = int(rx1_pct * w)
            ry1 = int(ry1_pct * h)
            rx2 = int(rx2_pct * w)
            ry2 = int(ry2_pct * h)

            if template is not None:
                found, conf, pos = self._template_match_in_region(
                    screenshot, template, (rx1, ry1, rx2, ry2), threshold=0.30
                )
                logger.debug(f"[PLAY_BTN] Regiao {label}: found={found}, conf={conf:.3f}, pos={pos}")
                if found and conf > best_conf:
                    best_conf = conf
                    best_result = PlayButtonResult(
                        found=True, coords=pos, region=label,
                        confidence=conf, screenshot_verified=True
                    )

        if best_result:
            self._last_known_play_pos = best_result.coords
            self._last_known_time = time.time()
            logger.info(f"[PLAY_BTN] Encontrado na regiao '{best_result.region}' em {best_result.coords} (conf={best_result.confidence:.2f})")
            return best_result

        # 3. Fallback: heuristica de cor amarela na area inferior direita
        fallback = self._detect_yellow_button_fallback(screenshot)
        if fallback:
            self._last_known_play_pos = fallback
            self._last_known_time = time.time()
            logger.info(f"[PLAY_BTN] Encontrado via fallback de cor em {fallback}")
            return PlayButtonResult(
                found=True, coords=fallback, region="fallback_color",
                confidence=0.45, screenshot_verified=False
            )

        logger.warning("[PLAY_BTN] Botao Play NAO ENCONTRADO em nenhuma regiao")
        return PlayButtonResult(found=False, coords=None, region="none", confidence=0.0)

    def _template_match_in_region(self, image, template, region, threshold=0.45):
        """Template matching numa regiao especifica."""
        rx1, ry1, rx2, ry2 = region
        rx1 = max(0, min(rx1, image.shape[1]))
        rx2 = max(rx1 + 1, min(rx2, image.shape[1]))
        ry1 = max(0, min(ry1, image.shape[0]))
        ry2 = max(ry1 + 1, min(ry2, image.shape[0]))

        search_img = image[ry1:ry2, rx1:rx2]
        if search_img.size == 0 or template is None:
            return False, 0.0, None

        # Escalar template
        scale = (rx2 - rx1) / 1920.0
        if abs(scale - 1.0) > 0.01:
            tw = max(1, int(template.shape[1] * scale * 1.5))  # Ajuste fino
            th = max(1, int(template.shape[0] * scale * 1.5))
            template = cv2.resize(template, (tw, th))

        th, tw = template.shape[:2]
        sh, sw = search_img.shape[:2]
        if th > sh or tw > sw:
            return False, 0.0, None

        try:
            result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val >= threshold:
                cx = rx1 + max_loc[0] + tw // 2
                cy = ry1 + max_loc[1] + th // 2
                return True, max_val, (cx, cy)
            return False, max_val, None
        except cv2.error:
            return False, 0.0, None

    def _is_yellow_play_button(self, region) -> bool:
        """Verifica se a regiao tem a cor amarela caracteristica do botao Play."""
        if region.size == 0 or cv2 is None:
            return False
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        # Amarelo do Brawl Stars: H=20-35, S>150, V>150
        yellow_mask = ((hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 38) &
                       (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 120))
        return np.mean(yellow_mask) > 0.15

    def _detect_yellow_button_fallback(self, screenshot: np.ndarray) -> Optional[Tuple[int, int]]:
        """Fallback: encontra maior blob amarelo na area inferior direita."""
        h, w = screenshot.shape[:2]
        bottom_right = screenshot[3*h//4:h, 3*w//4:w]
        if bottom_right.size == 0 or cv2 is None:
            return None

        hsv = cv2.cvtColor(bottom_right, cv2.COLOR_RGB2HSV)
        yellow_mask = ((hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 38) &
                       (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 100))

        # Encontrar contornos
        contours, _ = cv2.findContours(yellow_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Maior contorno = provavelmente o botao Play
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 200:  # Muito pequeno, nao e botao
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"]) + 3*w//4
        cy = int(M["m01"] / M["m00"]) + 3*h//4
        return (cx, cy)


# ---------------------------------------------------------------------------
# EventDetector - Identifica eventos ativos (Starr Nova, etc.)
# ---------------------------------------------------------------------------

@dataclass
class EventDetection:
    event_active: bool
    event_type: Optional[str]    # "starr_nova", "power_league", "club", "weekend", etc.
    play_in_event_coords: Optional[Tuple[int, int]]  # onde clicar para jogar o evento
    confidence: float


class EventDetector:
    """
    Detecta se ha um evento ativo no lobby e onde esta o botao de jogar o evento.
    """

    def __init__(self, images_path: Path):
        self.images_path = Path(images_path)
        self._template_cache: Dict[str, Optional[np.ndarray]] = {}

    def detect_event(self, screenshot: np.ndarray) -> EventDetection:
        """
        Analisa o lobby para detectar eventos ativos.
        """
        if screenshot is None or cv2 is None or np is None:
            return EventDetection(event_active=False, event_type=None,
                                 play_in_event_coords=None, confidence=0.0)

        h, w = screenshot.shape[:2]

        # 1. Verificar Starr Nova: cor roxa/lilas + texto na area superior
        starr_conf = self._detect_starr_nova(screenshot)
        if starr_conf > 0.35:
            # Botao "Jogar" do evento tipicamente na area inferior
            play_coords = self._find_event_play_button(screenshot)
            return EventDetection(
                event_active=True, event_type="starr_nova",
                play_in_event_coords=play_coords,
                confidence=starr_conf
            )

        # 2. Verificar evento ativo generico: botao colorido na area inferior
        event_conf = self._detect_generic_event(screenshot)
        if event_conf > 0.4:
            play_coords = self._find_event_play_button(screenshot)
            return EventDetection(
                event_active=True, event_type="generic_event",
                play_in_event_coords=play_coords,
                confidence=event_conf
            )

        # 3. Verificar se estamos DENTRO de uma tela de evento (nao no lobby)
        inside_event = self._detect_inside_event_screen(screenshot)
        if inside_event > 0.5:
            play_coords = self._find_event_play_button(screenshot)
            return EventDetection(
                event_active=True, event_type="event_screen",
                play_in_event_coords=play_coords,
                confidence=inside_event
            )

        return EventDetection(event_active=False, event_type=None,
                             play_in_event_coords=None, confidence=0.0)

    def _detect_starr_nova(self, screenshot: np.ndarray) -> float:
        """Detecta Starr Nova pela cor roxa dominante na area superior."""
        h, w = screenshot.shape[:2]
        top_area = screenshot[0:h//5, 0:w]
        if top_area.size == 0:
            return 0.0
        hsv = cv2.cvtColor(top_area, cv2.COLOR_RGB2HSV)
        purple_mask = ((hsv[:, :, 0] > 130) & (hsv[:, :, 0] < 170) &
                       (hsv[:, :, 1] > 60))
        purple_ratio = np.sum(purple_mask) / purple_mask.size
        return purple_ratio

    def _detect_generic_event(self, screenshot: np.ndarray) -> float:
        """Detecta evento generico: botao nao-amarelo na area inferior."""
        h, w = screenshot.shape[:2]
        bottom = screenshot[3*h//4:h, w//2:w]
        if bottom.size == 0:
            return 0.0
        hsv = cv2.cvtColor(bottom, cv2.COLOR_RGB2HSV)
        # Nao amarelo (play normal) -> provavelmente evento
        non_yellow = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 45)) & (hsv[:, :, 1] > 80)
        non_yellow_ratio = np.sum(non_yellow) / non_yellow.size
        return non_yellow_ratio if non_yellow_ratio > 0.3 else 0.0

    def _detect_inside_event_screen(self, screenshot: np.ndarray) -> float:
        """Detecta se estamos dentro da tela de um evento (nao no lobby principal)."""
        h, w = screenshot.shape[:2]
        # Dentro de evento: Play button NAO esta no canto inferior direito
        # E ha elementos de UI especificos do evento
        bottom_right = screenshot[3*h//4:h, 4*w//5:w]
        if bottom_right.size == 0:
            return 0.0
        # Se nao ha amarelo no canto inferior direito -> provavelmente nao e lobby
        hsv = cv2.cvtColor(bottom_right, cv2.COLOR_RGB2HSV)
        yellow_mask = ((hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 38) &
                       (hsv[:, :, 1] >= 100))
        yellow_ratio = np.sum(yellow_mask) / yellow_mask.size
        # Se pouco amarelo no canto inferior direito -> pode ser tela de evento
        if yellow_ratio < 0.05:
            # Verificar se ha algum botao grande na area inferior central
            bottom_center = screenshot[3*h//4:h, w//4:3*w//4]
            gray = cv2.cvtColor(bottom_center, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bright_ratio = np.sum(binary > 0) / binary.size
            return bright_ratio * 1.5 if bright_ratio > 0.15 else 0.0
        return 0.0

    def _find_event_play_button(self, screenshot: np.ndarray) -> Optional[Tuple[int, int]]:
        """Procura botao 'Jogar' dentro da tela de evento."""
        h, w = screenshot.shape[:2]
        # Area inferior central onde tipicamente esta o botao
        bottom_center = screenshot[3*h//4:h, w//4:3*w//4]
        if bottom_center.size == 0:
            return None

        # Procurar blob mais brilhante/colorido
        hsv = cv2.cvtColor(bottom_center, cv2.COLOR_RGB2HSV)
        # Qualquer cor saturada e brilhante = provavel botao
        bright_mask = ((hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 120))
        if np.sum(bright_mask) < 100:
            return None

        # Centro do blob
        ys, xs = np.where(bright_mask)
        if len(xs) == 0:
            return None
        cx = int(np.median(xs)) + w//4
        cy = int(np.median(ys)) + 3*h//4
        return (cx, cy)


# ---------------------------------------------------------------------------
# BrawlerSelectorFast - Selecao rapida com cache e pesquisa
# ---------------------------------------------------------------------------

class BrawlerSelectorFast:
    """
    Seleciona brawlers de forma rapida:
    1. Verifica se ja esta selecionado (evita cliques desnecessarios)
    2. Verifica se o brawler esta desbloqueado (nao tenta selecionar locked)
    3. Tenta pesquisa direta (mais rapida que OCR+scroll)
    4. Usa cache de coordenadas conhecidas
    5. Fallback para OCR otimizado (apenas na regiao relevante)
    """

    def __init__(self, images_path: Path):
        self.images_path = Path(images_path)
        self._brawler_coords_cache: Dict[str, Tuple[int, int]] = {}  # nome -> (x, y)
        self._ocr_reader = None
        self._ocr_backend = None
        self._locked_brawlers: set = set()  # brawlers known to be locked

    def select_brawler(self, brawler_name: str, screenshot: np.ndarray,
                       click_func, swipe_func) -> bool:
        """
        Seleciona brawler usando estrategia rapida.
        screenshot deve ser da TELA DE SELECAO DE BRAWLER.
        """
        if screenshot is None or cv2 is None or np is None:
            return False

        h, w = screenshot.shape[:2]

        # 0. Verificar se o brawler esta na lista de bloqueados
        if brawler_name.lower() in self._locked_brawlers:
            logger.warning(f"[BRAWLER] {brawler_name} esta bloqueado (locked), nao tentando selecionar")
            return False

        # 1. Verificar se ja esta selecionado (indicador de selecao visivel)
        if self._is_brawler_already_selected(brawler_name, screenshot):
            logger.info(f"[BRAWLER] {brawler_name} ja esta selecionado")
            return True

        # 2. Tentar cache de coordenadas
        if brawler_name.lower() in self._brawler_coords_cache:
            x, y = self._brawler_coords_cache[brawler_name.lower()]
            logger.info(f"[BRAWLER] Usando cache: {brawler_name} em ({x}, {y})")
            click_func(x, y)
            time.sleep(random.uniform(0.4, 0.7))
            if self._is_brawler_already_selected(brawler_name, screenshot):
                return True
            # Cache invalidado
            del self._brawler_coords_cache[brawler_name.lower()]

        # 3. Tentar pesquisa direta (se houver caixa de pesquisa)
        if self._try_search_box(brawler_name, screenshot, click_func):
            return True

        # 4. Fallback: OCR rapido em regiao otimizada
        result = self._fast_ocr_scan(brawler_name, screenshot, click_func, swipe_func)
        if result:
            # Guardar no cache (is_brawler_unlocked ja verificado dentro de _fast_ocr_scan)
            self._brawler_coords_cache[brawler_name.lower()] = result
        return result is not None

    def is_brawler_unlocked(self, screenshot: np.ndarray, cx: int, cy: int, radius: int = 30) -> bool:
        """
        Verifica se um brawler na posicao (cx, cy) esta desbloqueado.
        Brawlers bloqueados tem baixa saturacao (cinza/borrado).
        Brawlers desbloqueados tem alta saturacao (colorido).
        """
        if screenshot is None or cv2 is None:
            return True  # Assumir desbloqueado se nao puder verificar

        h, w = screenshot.shape[:2]
        r1 = max(0, cy - radius)
        r2 = min(h, cy + radius)
        c1 = max(0, cx - radius)
        c2 = min(w, cx + radius)
        region = screenshot[r1:r2, c1:c2]

        if region.size == 0:
            return True

        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        avg_saturation = hsv[:, :, 1].mean()
        avg_value = hsv[:, :, 2].mean()

        # Brawlers desbloqueados: saturacao alta (> 40)
        # Brawlers bloqueados: saturacao baixa (< 40) + valor medio-alto (> 80)
        is_unlocked = avg_saturation > 40

        if not is_unlocked:
            logger.info(f"[BRAWLER] Brawler em ({cx},{cy}) parece BLOQUEADO (sat={avg_saturation:.0f}, val={avg_value:.0f})")

        return is_unlocked

    def scan_available_brawlers(self, screenshot: np.ndarray) -> List[Dict]:
        """
        Escaneia a tela de selecao de brawlers e retorna lista de brawlers
        visiveis com suas posicoes e status (locked/unlocked).
        """
        if screenshot is None or cv2 is None:
            return []

        h, w = screenshot.shape[:2]
        gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=80,
            param1=50, param2=25, minRadius=25, maxRadius=60
        )

        brawlers = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0]:
                cx, cy, radius = c
                unlocked = self.is_brawler_unlocked(screenshot, cx, cy, radius)

                # Tentar OCR no nome do brawler (area abaixo do portrait)
                name = None
                if cy + radius + 20 < h:
                    name_region = screenshot[cy + radius:cy + radius + 30, cx - 40:cx + 40]
                    if name_region.size > 0:
                        try:
                            import easyocr
                            if self._ocr_reader is None:
                                self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                            results = self._ocr_reader.readtext(name_region)
                            for (_, text, conf) in results:
                                if conf > 0.4:
                                    name = text.strip()
                                    break
                        except Exception as e:
                            logger.debug(f"[BRAWLER] OCR failed: {e}")

                brawlers.append({
                    "x": int(cx),
                    "y": int(cy),
                    "radius": int(radius),
                    "unlocked": unlocked,
                    "name": name
                })

                if not unlocked and name:
                    self._locked_brawlers.add(name.lower())

        logger.info(f"[BRAWLER] Scan: {len(brawlers)} brawlers, "
                     f"{sum(1 for b in brawlers if b['unlocked'])} unlocked, "
                     f"{sum(1 for b in brawlers if not b['unlocked'])} locked")
        return brawlers

    def get_locked_brawlers(self) -> set:
        """Retorna set de nomes de brawlers bloqueados conhecidos."""
        return self._locked_brawlers.copy()

    def mark_brawler_locked(self, name: str):
        """Marca um brawler como bloqueado."""
        self._locked_brawlers.add(name.lower())
        if name.lower() in self._brawler_coords_cache:
            del self._brawler_coords_cache[name.lower()]
        logger.info(f"[BRAWLER] {name} marcado como bloqueado")

    def _is_brawler_already_selected(self, brawler_name: str, screenshot: np.ndarray) -> bool:
        """
        Verifica se o brawler ja esta selecionado procurando o nome na area
        onde tipicamente aparece o brawler selecionado (centro inferior).
        """
        h, w = screenshot.shape[:2]
        # Area onde o nome do brawler selecionado tipicamente aparece
        name_area = screenshot[3*h//5:4*h//5, w//3:2*w//3]
        if name_area.size == 0 or cv2 is None:
            return False

        # Usar OCR simples na regiao pequena
        try:
            import easyocr
            if self._ocr_reader is None:
                self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            results = self._ocr_reader.readtext(name_area)
            for (_, text, conf) in results:
                if conf > 0.5 and brawler_name.lower() in text.lower():
                    return True
        except Exception as e:
            logger.debug(f"[BRAWLER] Selection OCR failed: {e}")
        return False

    def _try_search_box(self, brawler_name: str, screenshot: np.ndarray,
                        click_func) -> bool:
        """
        Tenta usar a caixa de pesquisa para encontrar o brawler diretamente.
        Clica na caixa de pesquisa, escreve o nome, seleciona.
        """
        h, w = screenshot.shape[:2]
        # Caixa de pesquisa tipicamente no topo da tela de selecao
        # Procurar por icon de lupa ou caixa de texto
        top_area = screenshot[0:h//6, 0:w]
        gray = cv2.cvtColor(top_area, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Procurar regiao retangular clara (caixa de input)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)
            area = cw * ch
            # Caixa de pesquisa: retangulo largo, area media, no topo
            if 3 < aspect < 12 and 500 < area < 8000 and y < h // 10:
                search_x = x + cw // 2
                search_y = y + ch // 2
                logger.info(f"[BRAWLER] Caixa de pesquisa encontrada em ({search_x}, {search_y})")
                click_func(search_x, search_y)
                time.sleep(random.uniform(0.3, 0.5))
                # Digitar nome (simulado via ADB)
                # TODO: Implementar typing via ADB
                return False  # Por enquanto, nao implementamos typing

        return False

    def _fast_ocr_scan(self, brawler_name: str, screenshot: np.ndarray,
                       click_func, swipe_func) -> Optional[Tuple[int, int]]:
        """
        OCR otimizado: apenas na regiao central onde os brawlers aparecem.
        Sem redimensionamento agressivo. Tenta ate 3 vezes com scroll.
        """
        h, w = screenshot.shape[:2]

        try:
            import easyocr
            if self._ocr_reader is None:
                self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

            for scroll_attempt in range(3):
                # Area central onde os brawlers aparecem (nao redimensionar!)
                brawler_area = screenshot[h//5:4*h//5, 0:w]
                if brawler_area.size == 0:
                    return None

                results = self._ocr_reader.readtext(brawler_area)
                for (bbox, text, conf) in results:
                    if conf > 0.45 and brawler_name.lower() in text.lower():
                        # Calcular centro do bbox
                        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                            xs = [pt[0] for pt in bbox if isinstance(pt, (list, tuple))]
                            ys = [pt[1] for pt in bbox if isinstance(pt, (list, tuple))]
                            if xs and ys:
                                cx = int(sum(xs) / len(xs))
                                cy = int(sum(ys) / len(ys)) + h // 5
                                # Verificar se o brawler esta desbloqueado antes de clicar
                                if not self.is_brawler_unlocked(screenshot, cx, cy):
                                    logger.warning(f"[BRAWLER] {brawler_name} encontrado via OCR em ({cx},{cy}) mas esta BLOQUEADO (sat baixa)")
                                    self._locked_brawlers.add(brawler_name.lower())
                                    return None
                                logger.info(f"[BRAWLER] Encontrado via OCR: {brawler_name} em ({cx}, {cy}) conf={conf:.2f}")
                                click_func(cx, cy)
                                time.sleep(random.uniform(0.4, 0.7))
                                return (cx, cy)

                # Nao encontrou -> scroll e tentar novamente
                logger.info(f"[BRAWLER] {brawler_name} nao encontrado (tentativa {scroll_attempt+1}/3), fazendo scroll...")
                center_x = w // 2
                if swipe_func:
                    swipe_func(center_x, 3*h//4, center_x, h//4)  # Swipe para cima
                time.sleep(random.uniform(0.5, 0.8))

        except Exception as e:
            logger.warning(f"[BRAWLER] Erro no OCR rapido: {e}")

        return None


# ---------------------------------------------------------------------------
# EventSlotNavigator - Navegacao de slots de evento/modo de jogo
# ---------------------------------------------------------------------------

class EventSlotNavigator:
    """
    Navega pelos slots de modo de jogo no lobby.

    Responsabilidades:
    1. Detetar slots visiveis e seu modo (gem grab, showdown, etc.)
    2. Clicar no slot desejado se nao estiver ativo
    3. Swipe horizontal para revelar slots fora da tela

    Uso:
        slot_nav = EventSlotNavigator(images_path)
        slot_nav.navigate_to_mode(screenshot, "showdown", click_func=emulator.tap_scaled)
    """

    SHOWDOWN_COLORS = [(130, 170, 255), (100, 140, 220)]
    GEM_GRAB_COLORS = [(40, 200, 200), (20, 180, 180)]
    KNOCKOUT_COLORS = [(200, 100, 255), (180, 80, 230)]
    HEIST_COLORS = [(200, 180, 60), (180, 160, 40)]
    HOT_ZONE_COLORS = [(80, 180, 80), (60, 160, 60)]
    BOUNTY_COLORS = [(220, 180, 60), (200, 160, 40)]
    BRAWL_BALL_COLORS = [(200, 80, 80), (180, 60, 60)]
    POWER_LEAGUE_COLORS = [(160, 80, 255), (140, 60, 230)]

    def __init__(self, images_path: Optional[Path] = None):
        self.images_path = Path(images_path) if images_path else None
        self._slot_cache: List[Tuple[int, int, int, int]] = []
        self._last_scan_time: float = 0
        self._cache_ttl: float = 1.5

    def navigate_to_mode(
        self,
        screenshot: np.ndarray,
        desired_mode: str,
        click_func,
        swipe_func=None,
        max_swipes: int = 3,
    ) -> bool:
        """
        Navega para o modo desejado.

        Args:
            screenshot: screenshot atual do lobby
            desired_mode: modo desejado (showdown, gem_grab, knockout, etc.)
            click_func: funcao(x, y) para clicar
            swipe_func: funcao(x1, y1, x2, y2, duration) para swipe (opcional)
            max_swipes: maximo de swipes horizontais para encontrar o slot

        Returns:
            True se o slot correto esta agora ativo
        """
        if screenshot is None or np is None:
            return False

        h, w = screenshot.shape[:2]
        target_mode = desired_mode.lower().replace(" ", "_")

        for attempt in range(max_swipes + 1):
            slots = self._scan_slots(screenshot)

            for idx, (x, y, sw, sh) in enumerate(slots):
                mode = self._classify_slot(screenshot[y:y+sh, x:x+sw])
                if mode == target_mode:
                    center_x, center_y = x + sw // 2, y + sh // 2
                    is_active = self._is_slot_active(screenshot[y:y+sh, x:x+sw])

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
        """Classifica o modo de jogo de um slot pela cor dominante."""
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
            if self._is_slot_active(screenshot[y:y+sh, x:x+sw]):
                return self._classify_slot(screenshot[y:y+sh, x:x+sw])
        return None

    def get_play_button_coords(self, screenshot: np.ndarray) -> Tuple[int, int]:
        """Retorna coordenadas do botao Play (slot ativo)."""
        if screenshot is None:
            return (0, 0)
        h, w = screenshot.shape[:2]
        return (int(w * 0.87), int(h * 0.90))


# ---------------------------------------------------------------------------
# GameModeResolver - Utilitario para resolver nomes de modo
# ---------------------------------------------------------------------------

class GameModeResolver:
    """Resolve nomes de modo de jogo em varios formatos."""

    MODE_ALIASES = {
        "sd": "showdown",
        "solo": "showdown_solo",
        "solo_sd": "showdown_solo",
        "duo": "showdown_duo",
        "duo_sd": "showdown_duo",
        "gg": "gem_grab",
        "gemgrab": "gem_grab",
        "gems": "gem_grab",
        "bs": "bot_smac",
        "botsmac": "bot_smac",
        "ko": "knockout",
        "knockout": "knockout",
        "heist": "heist",
        "bounty": "bounty",
        "bb": "brawl_ball",
        "brawlball": "brawl_ball",
        "ball": "brawl_ball",
        "hz": "hot_zone",
        "hotzone": "hot_zone",
        "pl": "power_league",
        "powerleague": "power_league",
        "cl": "club_league",
        "clubleague": "club_league",
    }

    @classmethod
    def resolve(cls, mode: str) -> str:
        if not mode:
            return "unknown"
        normalized = mode.lower().replace(" ", "_").replace("-", "_")
        return cls.MODE_ALIASES.get(normalized, normalized)
