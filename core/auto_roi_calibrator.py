"""
core/auto_roi_calibrator.py

Auto-Calibração de ROIs por Resolução.

O Brawl Stars pode rodar em diferentes resoluções (1080p, 1440p, 720p, etc.).
As ROIs do OCR e dos detectores precisam ser ajustadas proporcionalmente.

Este módulo:
1. Detecta resolução atual da tela
2. Mantém ROIs canônicas (normalizadas 0-1 ou em 1920x1080)
3. Escala automaticamente para a resolução atual
4. Valida se ROIs fazem sentido na resolução alvo
5. Cache de calibrações por resolução

Integra com ResolutionManager existente.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ROIDefinition:
    """Definição de uma ROI canônica."""
    name: str
    # Coordenadas canônicas (baseadas em 1920x1080)
    x: int
    y: int
    w: int
    h: int
    description: str = ""


@dataclass
class CalibratedROI:
    """ROI calibrada para uma resolução específica."""
    name: str
    x: int
    y: int
    w: int
    h: int
    source_resolution: tuple[int, int]
    target_resolution: tuple[int, int]
    scale_x: float
    scale_y: float


class AutoROICalibrator:
    """
    Calibra ROIs automaticamente para qualquer resolução.

    Uso:
        calibrator = AutoROICalibrator()
        roi = calibrator.get_roi("hp_bar", current_width, current_height)
    """

    CANONICAL_W = 1920
    CANONICAL_H = 1080

    # ROIs canônicas para Brawl Stars (1920x1080)
    DEFAULT_ROIS = {
        # HUD
        "hp_bar": ROIDefinition("hp_bar", 70, 980, 200, 30, "Barra de HP do jogador"),
        "ammo_bar": ROIDefinition("ammo_bar", 1600, 980, 200, 30, "Barra de munição"),
        "super_bar": ROIDefinition("super_bar", 850, 980, 220, 30, "Barra de super"),
        "timer": ROIDefinition("timer", 900, 20, 120, 40, "Timer da partida"),
        "score": ROIDefinition("score", 850, 60, 220, 35, "Placar"),
        "cube_count": ROIDefinition("cube_count", 30, 150, 80, 40, "Contador de power cubes"),
        "gem_count": ROIDefinition("gem_count", 30, 200, 80, 40, "Contador de gems"),

        # Brawler select
        "brawler_name": ROIDefinition("brawler_name", 800, 150, 320, 50, "Nome do brawler selecionado"),
        "brawler_power": ROIDefinition("brawler_power", 920, 220, 80, 30, "Power level do brawler"),

        # Lobby
        "play_button": ROIDefinition("play_button", 1550, 900, 300, 120, "Botão Play"),
        "event_banner": ROIDefinition("event_banner", 50, 150, 400, 200, "Banner de evento"),

        # Map
        "map_name": ROIDefinition("map_name", 700, 50, 520, 40, "Nome do mapa no loading"),

        # Death/respawn
        "death_timer": ROIDefinition("death_timer", 860, 500, 200, 60, "Timer de respawn"),
        "spectate_button": ROIDefinition("spectate_button", 800, 850, 320, 60, "Botão Spectate"),
    }

    def __init__(self, cache_dir: Path = Path("pylaai_workspace/roi_cache")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._canonical_rois = dict(self.DEFAULT_ROIS)
        self._calibration_cache: dict[tuple[int, int], dict[str, CalibratedROI]] = {}
        self._load_cache()

    def _load_cache(self):
        """Carrega cache de calibrações anteriores."""
        cache_file = self.cache_dir / "calibrations.json"
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    data = json.load(f)
                for key, rois in data.items():
                    res = tuple(map(int, key.split("x")))
                    self._calibration_cache[res] = {
                        name: CalibratedROI(**roi) for name, roi in rois.items()
                    }
                logger.info("[ROI_CALIBRATOR] %d calibrações carregadas", len(self._calibration_cache))
            except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning("[ROI_CALIBRATOR] Erro ao carregar cache: %s", e)

    def _save_cache(self):
        """Salva cache no disco."""
        try:
            data = {}
            for res, rois in self._calibration_cache.items():
                key = f"{res[0]}x{res[1]}"
                data[key] = {name: asdict(roi) for name, roi in rois.items()}
            cache_file = self.cache_dir / "calibrations.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.warning("[ROI_CALIBRATOR] Erro ao salvar cache: %s", e)

    def get_roi(self, name: str, target_w: int, target_h: int) -> tuple[int, int, int, int] | None:
        """
        Retorna ROI calibrada para a resolução alvo.
        Retorna (x, y, w, h) ou None se ROI não existir.
        """
        canonical = self._canonical_rois.get(name)
        if not canonical:
            logger.warning("[ROI_CALIBRATOR] ROI '%s' não definida", name)
            return None

        target_res = (target_w, target_h)

        # Verificar cache
        if target_res in self._calibration_cache:
            cached = self._calibration_cache[target_res].get(name)
            if cached:
                return (cached.x, cached.y, cached.w, cached.h)

        # Calcular escala
        scale_x = target_w / self.CANONICAL_W
        scale_y = target_h / self.CANONICAL_H

        # Calibração
        calibrated = CalibratedROI(
            name=name,
            x=int(canonical.x * scale_x),
            y=int(canonical.y * scale_y),
            w=int(canonical.w * scale_x),
            h=int(canonical.h * scale_y),
            source_resolution=(self.CANONICAL_W, self.CANONICAL_H),
            target_resolution=target_res,
            scale_x=scale_x,
            scale_y=scale_y,
        )

        # Validar
        if not self._validate_roi(calibrated, target_w, target_h):
            logger.warning(
                "[ROI_CALIBRATOR] ROI '%s' inválida em %dx%d, ajustando...",
                name, target_w, target_h
            )
            calibrated = self._clamp_roi(calibrated, target_w, target_h)

        # Cache
        if target_res not in self._calibration_cache:
            self._calibration_cache[target_res] = {}
        self._calibration_cache[target_res][name] = calibrated
        self._save_cache()

        return (calibrated.x, calibrated.y, calibrated.w, calibrated.h)

    def get_all_rois(self, target_w: int, target_h: int) -> dict[str, tuple[int, int, int, int]]:
        """Retorna todas as ROIs calibradas para a resolução."""
        return {
            name: roi for name in self._canonical_rois
            if (roi := self.get_roi(name, target_w, target_h)) is not None
        }

    def _validate_roi(self, roi: CalibratedROI, screen_w: int, screen_h: int) -> bool:
        """Verifica se ROI está dentro da tela e tem tamanho mínimo."""
        if roi.x < 0 or roi.y < 0:
            return False
        if roi.x + roi.w > screen_w or roi.y + roi.h > screen_h:
            return False
        if roi.w < 10 or roi.h < 10:
            return False
        return True

    def _clamp_roi(self, roi: CalibratedROI, screen_w: int, screen_h: int) -> CalibratedROI:
        """Ajusta ROI para caber na tela."""
        x = max(0, min(roi.x, screen_w - 10))
        y = max(0, min(roi.y, screen_h - 10))
        w = min(roi.w, screen_w - x)
        h = min(roi.h, screen_h - y)
        return CalibratedROI(
            name=roi.name, x=x, y=y, w=w, h=h,
            source_resolution=roi.source_resolution,
            target_resolution=roi.target_resolution,
            scale_x=roi.scale_x, scale_y=roi.scale_y,
        )

    def add_roi(self, name: str, x: int, y: int, w: int, h: int, description: str = ""):
        """Adiciona uma nova ROI canônica."""
        self._canonical_rois[name] = ROIDefinition(name, x, y, w, h, description)
        # Invalidar cache para esta ROI
        for res_cache in self._calibration_cache.values():
            res_cache.pop(name, None)
        logger.info("[ROI_CALIBRATOR] ROI '%s' adicionada (%dx%d+%d+%d)", name, x, y, w, h)

    def get_calibration_info(self, target_w: int, target_h: int) -> dict[str, Any]:
        """Retorna informações de calibração para uma resolução."""
        scale_x = target_w / self.CANONICAL_W
        scale_y = target_h / self.CANONICAL_H
        return {
            "canonical_resolution": (self.CANONICAL_W, self.CANONICAL_H),
            "target_resolution": (target_w, target_h),
            "scale_x": round(scale_x, 4),
            "scale_y": round(scale_y, 4),
            "roi_count": len(self._canonical_rois),
            "cached": (target_w, target_h) in self._calibration_cache,
        }
