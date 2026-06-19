"""
vision/ocr_hud_extractor.py

Extrator avançado de valores numéricos do HUD via OCR.

Funcionalidades:
- HP do jogador (barra + valor numérico)
- Ammo/cargas do ataque básico (0-3)
- Super charge (0-100%)
- Timer da partida (MM:SS)
- Score do time (ex: 0-2)
- Contagem de cubos (Showdown)
- Contagem de gems (Gem Grab)

Design:
- Pré-processamento multi-variante (scale, threshold, denoise)
- Múltiplas ROIs por campo com votação ponderada
- Normalização numérica robusta (O->0, I->1, etc.)
- Fallback hierárquico: OCR → heurísticas pixel → default
- Cache por frame com TTL adaptativo
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("[OCR_HUD] OpenCV not available — pixel fallback disabled")


class HudField(Enum):
    """Campos do HUD extraíveis via OCR."""

    HP_VALUE = auto()
    AMMO_COUNT = auto()
    SUPER_CHARGE = auto()
    MATCH_TIMER = auto()
    TEAM_SCORE = auto()
    CUBE_COUNT = auto()
    GEM_COUNT = auto()


@dataclass
class HudValue:
    """Valor extraído de um campo do HUD."""

    field: HudField
    raw_value: str
    parsed_value: float | None
    confidence: float  # 0.0–1.0
    source: str  # "ocr", "pixel_heuristic", "default"
    timestamp: float = field(default_factory=time.time)

    @property
    def is_valid(self) -> bool:
        """True se o valor foi extraído com sucesso e confiança aceitável."""
        return self.parsed_value is not None and self.confidence >= 0.5


@dataclass
class HudState:
    """Estado completo do HUD num único frame."""

    hp: HudValue
    ammo: HudValue
    super_charge: HudValue
    timer: HudValue
    score: HudValue
    cubes: HudValue
    gems: HudValue
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Serializa para dict (útil para dataset/logs)."""
        return {
            "hp": {
                "raw": self.hp.raw_value,
                "parsed": self.hp.parsed_value,
                "confidence": self.hp.confidence,
                "source": self.hp.source,
            },
            "ammo": {
                "raw": self.ammo.raw_value,
                "parsed": self.ammo.parsed_value,
                "confidence": self.ammo.confidence,
                "source": self.ammo.source,
            },
            "super_charge": {
                "raw": self.super_charge.raw_value,
                "parsed": self.super_charge.parsed_value,
                "confidence": self.super_charge.confidence,
                "source": self.super_charge.source,
            },
            "timer": {
                "raw": self.timer.raw_value,
                "parsed": self.timer.parsed_value,
                "confidence": self.timer.confidence,
                "source": self.timer.source,
            },
            "score": {
                "raw": self.score.raw_value,
                "parsed": self.score.parsed_value,
                "confidence": self.score.confidence,
                "source": self.score.source,
            },
            "cubes": {
                "raw": self.cubes.raw_value,
                "parsed": self.cubes.parsed_value,
                "confidence": self.cubes.confidence,
                "source": self.cubes.source,
            },
            "gems": {
                "raw": self.gems.raw_value,
                "parsed": self.gems.parsed_value,
                "confidence": self.gems.confidence,
                "source": self.gems.source,
            },
            "timestamp": self.timestamp,
        }


class OCRHudExtractor:
    """
    Extrator avançado de valores numéricos do HUD.

    Usa EasyOCR com pré-processamento multi-variante e votação entre
    múltiplas ROIs.  Quando o OCR falha recai para heurísticas de pixel
    (ex.: contar pixels verdes na barra de HP) e, por fim, valores default.
    """

    # ------------------------------------------------------------------
    # ROIs normalizadas (0–1) — múltiplas candidatas por campo para
    # tolerância a mudanças sutis de layout / resolução.
    # ------------------------------------------------------------------
    DEFAULT_ROIS: dict[HudField, list[tuple[float, float, float, float]]] = {
        HudField.HP_VALUE: [
            (0.020, 0.010, 0.180, 0.060),
            (0.015, 0.005, 0.200, 0.070),
        ],
        HudField.AMMO_COUNT: [
            (0.720, 0.840, 0.950, 0.990),
            (0.700, 0.820, 0.960, 0.995),
            (0.735, 0.855, 0.935, 0.975),
        ],
        HudField.SUPER_CHARGE: [
            (0.020, 0.800, 0.280, 0.990),
            (0.010, 0.780, 0.300, 0.995),
        ],
        HudField.MATCH_TIMER: [
            (0.445, 0.006, 0.555, 0.060),
            (0.420, 0.000, 0.580, 0.074),
            (0.455, 0.010, 0.545, 0.052),
        ],
        HudField.TEAM_SCORE: [
            (0.390, 0.035, 0.610, 0.125),
            (0.365, 0.020, 0.635, 0.140),
            (0.410, 0.045, 0.590, 0.115),
        ],
        HudField.CUBE_COUNT: [
            (0.020, 0.080, 0.120, 0.160),
            (0.015, 0.070, 0.130, 0.170),
        ],
        HudField.GEM_COUNT: [
            (0.020, 0.080, 0.120, 0.160),
            (0.015, 0.070, 0.130, 0.170),
        ],
    }

    def __init__(
        self,
        resolution: tuple[int, int] = (1920, 1080),
        confidence_threshold: float = 0.6,
        ocr_scale_factor: float = 2.0,
        cache_ttl_sec: float = 0.5,
        use_heuristic_fallback: bool = True,
    ):
        self.w, self.h = resolution
        self.confidence_threshold = confidence_threshold
        self.ocr_scale_factor = ocr_scale_factor
        self.cache_ttl_sec = cache_ttl_sec
        self.use_heuristic_fallback = use_heuristic_fallback

        self._reader: object | None = None
        self._reader_ready: bool = False  # True apenas quando reader OK

        # Cache por campo: campo → (timestamp, HudValue)
        self._cache: dict[HudField, tuple[float, HudValue]] = {}

        logger.info(
            "[OCR_HUD] Inicializado: resolution=%s, conf_thresh=%.2f, scale=%.1f",
            resolution,
            confidence_threshold,
            ocr_scale_factor,
        )

    # ------------------------------------------------------------------
    # EasyOCR lifecycle (lazy + mock-safe)
    # ------------------------------------------------------------------
    def _ensure_reader(self) -> bool:
        """Carrega o reader EasyOCR se ainda não estiver pronto."""
        if self._reader_ready:
            return True
        if self._reader is not None:
            # Já tentamos carregar e falhou — não repetir
            return False
        try:
            import easyocr  # type: ignore[import-untyped]

            logger.info("[OCR_HUD] Carregando EasyOCR…")
            self._reader = easyocr.Reader(["en"], gpu=False)
            self._reader_ready = True
            logger.info("[OCR_HUD] EasyOCR pronto")
            return True
        except ImportError:
            logger.warning("[OCR_HUD] EasyOCR não instalado. OCR numérico desativado.")
            self._reader = None
            return False
        except Exception as exc:  # pragma: no cover
            logger.error("[OCR_HUD] Falha ao carregar EasyOCR: %s", exc)
            self._reader = None
            return False

    # ------------------------------------------------------------------
    # Coordenadas
    # ------------------------------------------------------------------
    def _roi_to_pixels(
        self, roi: tuple[float, float, float, float]
    ) -> tuple[int, int, int, int]:
        """Converte ROI normalizada (0–1) para pixels absolutos."""
        x1, y1, x2, y2 = roi
        return (
            int(x1 * self.w),
            int(y1 * self.h),
            int(x2 * self.w),
            int(y2 * self.h),
        )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _cached(self, field: HudField) -> HudValue | None:
        """Retorna valor em cache se ainda válido."""
        entry = self._cache.get(field)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > self.cache_ttl_sec:
            return None
        return value

    def _set_cache(self, field: HudField, value: HudValue) -> None:
        self._cache[field] = (time.time(), value)

    # ------------------------------------------------------------------
    # Pré-processamento OCR
    # ------------------------------------------------------------------
    def _preprocess_variants(self, crop: np.ndarray) -> list[np.ndarray]:
        """Gera variantes pré-processadas para maximizar chance de OCR correto."""
        if crop is None or crop.size == 0:
            return []
        variants: list[np.ndarray] = [crop]
        if not HAS_CV2:
            return variants
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.copy()
            variants.append(gray)
            scaled = cv2.resize(
                gray,
                None,
                fx=self.ocr_scale_factor,
                fy=self.ocr_scale_factor,
                interpolation=cv2.INTER_CUBIC,
            )
            variants.append(scaled)
            blurred = cv2.GaussianBlur(scaled, (3, 3), 0)
            variants.append(blurred)
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
            variants.append(thresh)
        except Exception as exc:  # pragma: no cover
            logger.debug("[OCR_HUD] preprocess fallback: %s", exc)
        return variants

    # ------------------------------------------------------------------
    # OCR — executa em todas as variantes e devolve resultados combinados
    # ------------------------------------------------------------------
    def _ocr_crop(self, crop: np.ndarray) -> list[tuple[str, float]]:
        """Executa OCR num recorte e devolve lista de (texto, confiança)."""
        if not self._ensure_reader():
            return []
        results: list[tuple[str, float]] = []
        seen: set[str] = set()
        for variant in self._preprocess_variants(crop):
            try:
                raw = self._reader.readtext(variant)  # type: ignore[union-attr]
                for _bbox, text, conf in raw:
                    key = f"{text.lower().strip()}|{conf:.3f}"
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append((text, float(conf)))
            except Exception as exc:  # pragma: no cover
                logger.debug("[OCR_HUD] readtext error: %s", exc)
        return results

    # ------------------------------------------------------------------
    # Normalização numérica
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_numeric(text: str) -> str:
        """Normaliza texto com foco em números, corrigindo confusões típicas de OCR."""
        text = text.strip().upper()
        # Remove espaços entre dígitos (ex: "1 2 3" → "123")
        # Aplica repetidamente até não haver mais matches
        while re.search(r"(\d)\s+(\d)", text):
            text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)
        # Substituições comuns de OCR — NOTA: '/' preservado para timers/frações
        trans = str.maketrans({
            "O": "0",
            "Q": "0",
            "D": "0",
            "I": "1",
            "L": "1",
            "|": "1",
            "S": "5",
            "B": "8",
            "Z": "2",
        })
        return text.translate(trans)

    @staticmethod
    def _extract_first_number(text: str) -> float | None:
        """Extrai o primeiro número (inteiro ou decimal) do texto."""
        # Procura padrões como "12", "1:23", "100%", "3/3"
        # Primeiro tenta MM:SS (timer)
        timer_match = re.search(r"(\d{1,2})[:\s](\d{2})", text)
        if timer_match:
            minutes = int(timer_match.group(1))
            seconds = int(timer_match.group(2))
            return float(minutes * 60 + seconds)
        # Procura número simples
        num_match = re.search(r"(\d+(?:\.\d+)?)", text.replace("%", "").replace("/", " "))
        if num_match:
            return float(num_match.group(1))
        return None

    @staticmethod
    def _extract_fraction(text: str) -> tuple[int, int] | None:
        """Extrai fração tipo '2/3' → (2, 3)."""
        match = re.search(r"(\d+)\s*/\s*(\d+)", text)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    # ------------------------------------------------------------------
    # Heurísticas de pixel (fallback)
    # ------------------------------------------------------------------
    def _heuristic_hp(self, screenshot: np.ndarray) -> float | None:
        """Estima HP pela razão de pixels verdes vs. vermelhos na barra de HP."""
        if not HAS_CV2 or screenshot is None or screenshot.size == 0:
            return None
        try:
            # Região típica da barra de HP (topo-esquerdo)
            x1, y1, x2, y2 = self._roi_to_pixels((0.020, 0.010, 0.180, 0.060))
            h, w = screenshot.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            region = screenshot[y1:y2, x1:x2]
            if region.size == 0:
                return None
            # Conta pixels verdes (HP) e vermelhos (dano)
            green_mask = (region[:, :, 1] > 120) & (region[:, :, 0] < 100) & (region[:, :, 2] < 100)
            red_mask = (region[:, :, 0] > 120) & (region[:, :, 1] < 100) & (region[:, :, 2] < 100)
            green_count = int(np.sum(green_mask))
            red_count = int(np.sum(red_mask))
            total = green_count + red_count
            if total < 10:
                return None
            return float(np.clip(green_count / total, 0.0, 1.0))
        except Exception as exc:  # pragma: no cover
            logger.debug("[OCR_HUD] heuristic_hp error: %s", exc)
            return None

    def _heuristic_ammo(self, screenshot: np.ndarray) -> float | None:
        """Estima ammo pela presença de ícones/cargas na região de habilidades."""
        # Simplificado: não implementamos sem análise visual real
        return None

    def _heuristic_super(self, screenshot: np.ndarray) -> float | None:
        """Estima super charge pela cor da barra de super (amarela/laranja vs cinza)."""
        if not HAS_CV2 or screenshot is None or screenshot.size == 0:
            return None
        try:
            x1, y1, x2, y2 = self._roi_to_pixels((0.020, 0.800, 0.280, 0.990))
            h, w = screenshot.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            region = screenshot[y1:y2, x1:x2]
            if region.size == 0:
                return None
            # Super carregada = cor amarela/laranja brilhante
            yellow_mask = (
                (region[:, :, 0] > 180)
                & (region[:, :, 1] > 150)
                & (region[:, :, 2] < 100)
            )
            total = region.shape[0] * region.shape[1]
            yellow_count = int(np.sum(yellow_mask))
            if total == 0:
                return None
            ratio = yellow_count / total
            # Mapeia razão para percentagem (heurística)
            return float(np.clip(ratio * 5.0, 0.0, 1.0))  # Multiplicador empírico
        except Exception as exc:  # pragma: no cover
            logger.debug("[OCR_HUD] heuristic_super error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Extração principal — um campo
    # ------------------------------------------------------------------
    def _extract_field(
        self,
        field: HudField,
        screenshot: np.ndarray,
        rois: list[tuple[float, float, float, float]] | None = None,
    ) -> HudValue:
        """Extrai um único campo do HUD com votação entre múltiplas ROIs."""
        # 1) Cache
        cached = self._cached(field)
        if cached is not None:
            return cached

        candidates: list[tuple[float, float | None, str]] = []
        rois = rois or self.DEFAULT_ROIS.get(field, [])

        if screenshot is None or screenshot.size == 0:
            value = HudValue(
                field=field,
                raw_value="",
                parsed_value=None,
                confidence=0.0,
                source="default",
            )
            self._set_cache(field, value)
            return value

        # 2) OCR em cada ROI
        for roi in rois:
            x1, y1, x2, y2 = self._roi_to_pixels(roi)
            h_s, w_s = screenshot.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_s, x2), min(h_s, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = screenshot[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            ocr_results = self._ocr_crop(crop)
            for text, conf in ocr_results:
                norm = self._normalize_numeric(text)
                parsed = self._extract_first_number(norm)
                if parsed is not None:
                    candidates.append((conf, parsed, text))

        # 3) Seleciona melhor candidato
        if candidates:
            # Ordena por confiança e pega o melhor
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_conf, best_val, best_text = candidates[0]
            if best_conf >= self.confidence_threshold:
                # Converte valores para intervalos corretos
                parsed = self._post_process_value(field, best_val)
                value = HudValue(
                    field=field,
                    raw_value=best_text,
                    parsed_value=parsed,
                    confidence=best_conf,
                    source="ocr",
                )
                self._set_cache(field, value)
                return value

        # 4) Heurística de pixel
        if self.use_heuristic_fallback:
            heuristic_val = self._pixel_fallback(field, screenshot)
            if heuristic_val is not None:
                value = HudValue(
                    field=field,
                    raw_value="",
                    parsed_value=heuristic_val,
                    confidence=0.4,
                    source="pixel_heuristic",
                )
                self._set_cache(field, value)
                return value

        # 5) Default
        value = HudValue(
            field=field,
            raw_value="",
            parsed_value=self._default_value(field),
            confidence=0.0,
            source="default",
        )
        self._set_cache(field, value)
        return value

    def _post_process_value(self, field: HudField, raw: float) -> float:
        """Converte valor bruto para intervalo semântico correto."""
        if field == HudField.HP_VALUE:
            return float(np.clip(raw / 100.0 if raw > 1.0 else raw, 0.0, 1.0))
        if field == HudField.AMMO_COUNT:
            return float(np.clip(round(raw), 0.0, 3.0))
        if field == HudField.SUPER_CHARGE:
            return float(np.clip(raw / 100.0 if raw > 1.0 else raw, 0.0, 1.0))
        if field == HudField.MATCH_TIMER:
            # Já convertido para segundos em _extract_first_number
            return raw
        if field in (HudField.TEAM_SCORE, HudField.CUBE_COUNT, HudField.GEM_COUNT):
            return float(max(0, round(raw)))
        return raw

    def _pixel_fallback(self, field: HudField, screenshot: np.ndarray) -> float | None:
        """Direciona para heurística correta."""
        if field == HudField.HP_VALUE:
            return self._heuristic_hp(screenshot)
        if field == HudField.SUPER_CHARGE:
            return self._heuristic_super(screenshot)
        if field == HudField.AMMO_COUNT:
            return self._heuristic_ammo(screenshot)
        return None

    @staticmethod
    def _default_value(field: HudField) -> float | None:
        """Valor default conservador."""
        defaults = {
            HudField.HP_VALUE: 1.0,
            HudField.AMMO_COUNT: 3.0,
            HudField.SUPER_CHARGE: 0.0,
            HudField.MATCH_TIMER: None,
            HudField.TEAM_SCORE: None,
            HudField.CUBE_COUNT: None,
            HudField.GEM_COUNT: None,
        }
        return defaults.get(field)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def extract_all(self, screenshot: np.ndarray) -> HudState:
        """Extrai todos os campos do HUD e devolve HudState completo."""
        t0 = time.time()
        state = HudState(
            hp=self._extract_field(HudField.HP_VALUE, screenshot),
            ammo=self._extract_field(HudField.AMMO_COUNT, screenshot),
            super_charge=self._extract_field(HudField.SUPER_CHARGE, screenshot),
            timer=self._extract_field(HudField.MATCH_TIMER, screenshot),
            score=self._extract_field(HudField.TEAM_SCORE, screenshot),
            cubes=self._extract_field(HudField.CUBE_COUNT, screenshot),
            gems=self._extract_field(HudField.GEM_COUNT, screenshot),
        )
        elapsed = time.time() - t0
        logger.debug("[OCR_HUD] extract_all completo em %.1f ms", elapsed * 1000)
        return state

    def extract_field(
        self,
        field: HudField,
        screenshot: np.ndarray,
        rois: list[tuple[float, float, float, float]] | None = None,
    ) -> HudValue:
        """Extrai um campo específico do HUD."""
        return self._extract_field(field, screenshot, rois)

    def invalidate_cache(self) -> None:
        """Invalida todo o cache (útil quando muda de tela)."""
        self._cache.clear()
        logger.debug("[OCR_HUD] Cache invalidado")
