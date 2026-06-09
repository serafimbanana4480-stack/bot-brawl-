"""
screenshot_analyzer.py

Sistema de análise automática de screenshots para diagnóstico e validação.
Detecta problemas comuns (cores trocadas, screenshot preta, janela minimizada,
emulador não visível) e fornece métricas de qualidade.

DEPRECATED: Use pylaai_real.unified_state_detector.UnifiedStateDetector instead.
This module is kept for backward compatibility only.
"""

import warnings
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

warnings.warn(
    "Deprecated: use pylaai_real.unified_state_detector.UnifiedStateDetector instead",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class ScreenshotAnalysis:
    """Resultado da análise de um screenshot."""
    valid: bool = False
    is_black: bool = False
    is_white: bool = False
    is_frozen: bool = False
    color_space: str = "unknown"  # "rgb", "bgr", "unknown"
    avg_brightness: float = 0.0
    std_brightness: float = 0.0
    dominant_color: Tuple[int, int, int] = (0, 0, 0)
    region_health: Dict[str, float] = field(default_factory=dict)
    issues: list = field(default_factory=list)
    timestamp: float = 0.0


class ScreenshotAnalyzer:
    """
    Analisa screenshots para detectar problemas de captura e validar
    se o detector de estado está a funcionar corretamente.
    """

    def __init__(self):
        self._last_screenshot: Optional[np.ndarray] = None
        self._last_analysis: Optional[ScreenshotAnalysis] = None
        self._history: list = []
        self._max_history = 50

    def analyze(self, screenshot: Optional[np.ndarray]) -> ScreenshotAnalysis:
        """
        Analisa um screenshot e retorna métricas de qualidade.
        """
        result = ScreenshotAnalysis(timestamp=time.time())

        if screenshot is None:
            result.issues.append("screenshot_is_none")
            result.valid = False
            return result

        if screenshot.size == 0:
            result.issues.append("screenshot_empty")
            result.valid = False
            return result

        h, w = screenshot.shape[:2]
        if len(screenshot.shape) < 3:
            result.issues.append("grayscale_screenshot")
            result.valid = False
            return result

        # Verificar se é preto ou branco (problema comum de captura)
        gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY) if cv2 else np.mean(screenshot, axis=2)
        result.avg_brightness = float(np.mean(gray))
        result.std_brightness = float(np.std(gray))

        if result.avg_brightness < 5:
            result.is_black = True
            result.issues.append("screenshot_nearly_black")
            result.valid = False
            return result

        if result.avg_brightness > 250 and result.std_brightness < 10:
            result.is_white = True
            result.issues.append("screenshot_nearly_white")
            result.valid = False
            return result

        # Detetar color space (RGB vs BGR heurística)
        result.color_space = self._detect_color_space(screenshot)

        # Verificar regiões chave do Brawl Stars
        result.region_health = self._analyze_regions(screenshot)

        # Verificar se o screenshot está "congelado" (igual ao anterior)
        if self._last_screenshot is not None:
            if self._last_screenshot.shape == screenshot.shape:
                diff = np.mean(np.abs(self._last_screenshot.astype(float) - screenshot.astype(float)))
                if diff < 2.0:
                    result.is_frozen = True
                    result.issues.append("screenshot_frozen")

        self._last_screenshot = screenshot.copy()

        # Verificar saúde do detector (cores dos pixels chave)
        play_pixel = self._sample_pixel(screenshot, 0.9119, 0.9122)
        joy_pixel = self._sample_pixel(screenshot, 0.10, 0.75)
        hp_pixel = self._sample_pixel(screenshot, 0.08, 0.06)

        result.dominant_color = self._compute_dominant_color(screenshot)

        # Se o play button é amarelo, provavelmente estamos no lobby
        is_play_yellow = self._is_yellow(play_pixel)
        is_joy_dark = np.mean(joy_pixel) < 30
        is_hp_white = np.mean(hp_pixel) > 240

        result.region_health["play_button_yellow"] = 1.0 if is_play_yellow else 0.0
        result.region_health["joystick_dark"] = 1.0 if is_joy_dark else 0.0
        result.region_health["hp_bar_white"] = 1.0 if is_hp_white else 0.0

        result.valid = True
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return result

    def _detect_color_space(self, image: np.ndarray) -> str:
        """
        Heurística para detetar se a imagem é RGB ou BGR.
        O botão Play do Brawl Stars é tipicamente amarelo (R alto, G médio, B baixo).
        Se o pixel do play button tiver B alto, provavelmente é BGR.
        """
        h, w = image.shape[:2]
        px = image[int(h * 0.9122), int(w * 0.9119)]
        r, g, b = int(px[0]), int(px[1]), int(px[2])

        # Amarelo em RGB: R > 200, G > 150, B < 50
        # Amarelo em BGR: B > 200, G > 150, R < 50
        rgb_yellow_score = (r > 200 and g > 150 and b < 80)
        bgr_yellow_score = (b > 200 and g > 150 and r < 80)

        if rgb_yellow_score and not bgr_yellow_score:
            return "rgb"
        if bgr_yellow_score and not rgb_yellow_score:
            return "bgr"
        return "unknown"

    def _sample_pixel(self, image: np.ndarray, rx: float, ry: float) -> np.ndarray:
        h, w = image.shape[:2]
        x = min(int(w * rx), w - 1)
        y = min(int(h * ry), h - 1)
        return image[y, x]

    def _is_yellow(self, pixel) -> bool:
        r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
        return r > 200 and g > 150 and b < 80

    def _analyze_regions(self, image: np.ndarray) -> Dict[str, float]:
        h, w = image.shape[:2]
        regions = {}

        # Região do botão play (inferior direito)
        play_region = image[int(h * 0.85):h, int(w * 0.85):w]
        if play_region.size > 0:
            play_hsv = cv2.cvtColor(play_region, cv2.COLOR_RGB2HSV) if cv2 else None
            if play_hsv is not None:
                yellow_mask = ((play_hsv[:, :, 0] >= 18) & (play_hsv[:, :, 0] <= 38) &
                               (play_hsv[:, :, 1] >= 80) & (play_hsv[:, :, 2] >= 100))
                regions["play_yellow_ratio"] = float(np.sum(yellow_mask)) / yellow_mask.size
            else:
                regions["play_yellow_ratio"] = 0.0
        else:
            regions["play_yellow_ratio"] = 0.0

        # Região central (para popups)
        center_region = image[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
        if center_region.size > 0:
            regions["center_brightness"] = float(np.mean(center_region))
        else:
            regions["center_brightness"] = 0.0

        return regions

    def _compute_dominant_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        # Downsample para performance
        small = cv2.resize(image, (50, 50)) if cv2 else image[::20, ::20]
        avg = np.mean(small, axis=(0, 1))
        return (int(avg[0]), int(avg[1]), int(avg[2]))

    def is_lobby_likely(self, screenshot: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Retorna (provavelmente_lobby, confiança) baseado na análise visual.
        """
        if screenshot is not None:
            analysis = self.analyze(screenshot)
        elif self._last_analysis is not None:
            analysis = self._last_analysis
        else:
            return False, 0.0

        if not analysis.valid:
            return False, 0.0

        score = 0.0
        if analysis.region_health.get("play_button_yellow", 0) > 0.5:
            score += 0.5
        if analysis.region_health.get("joystick_dark", 0) > 0.5:
            score += 0.3
        if analysis.region_health.get("hp_bar_white", 0) > 0.5:
            score += 0.2

        return score > 0.4, score

    def get_streak_issues(self) -> list:
        """Retorna problemas que ocorrem em sequência."""
        if len(self._history) < 5:
            return []
        issues = []
        recent = self._history[-5:]
        if all(a.is_frozen for a in recent):
            issues.append("frozen_streak")
        if all(not a.valid for a in recent):
            issues.append("invalid_streak")
        if all(a.color_space == "bgr" for a in recent if a.color_space != "unknown"):
            issues.append("bgr_detected_streak")
        return issues
