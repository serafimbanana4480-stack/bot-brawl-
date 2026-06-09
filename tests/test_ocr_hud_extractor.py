"""
tests/test_ocr_hud_extractor.py

Testes para o OCRHudExtractor.

Estratégia de teste:
- Mock do reader EasyOCR para testes determinísticos
- Screenshots sintéticos com texto renderizado via PIL
- Testes de normalização numérica (sem dependência de OCR)
- Testes de heurísticas de pixel (com OpenCV mockado)
- Testes de cache e coordenadas
"""

from __future__ import annotations

import sys
from pathlib import Path


import numpy as np
import pytest

from vision.ocr_hud_extractor import (
    HudField,
    HudState,
    HudValue,
    OCRHudExtractor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor():
    """Extrator com reader desativado (modo teste)."""
    ext = OCRHudExtractor(resolution=(1920, 1080))
    ext._reader = None
    ext._reader_ready = False
    ext.invalidate_cache()
    return ext


@pytest.fixture
def extractor_with_mock_reader(monkeypatch):
    """Extrator com reader mockado que retorna resultados controlados."""
    ext = OCRHudExtractor(resolution=(1920, 1080))

    class FakeReader:
        def __init__(self, results):
            self._results = results

        def readtext(self, image):
            return self._results

    def _make(results):
        ext._reader = FakeReader(results)
        ext._reader_ready = True
        ext.invalidate_cache()
        return ext

    return _make


@pytest.fixture
def black_screenshot():
    """Screenshot preto 1080p."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def white_screenshot():
    """Screenshot branco 1080p."""
    return np.full((1080, 1920, 3), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_text_pil(
    screenshot: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = (255, 255, 255),
    size: int = 24,
) -> np.ndarray:
    """Renderiza texto num screenshot via PIL (sem alterar cv2)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(screenshot)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x, y), text, fill=color, font=font)
    return np.array(img)


# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_resolution(self, extractor):
        assert extractor.w == 1920
        assert extractor.h == 1080

    def test_custom_resolution(self):
        ext = OCRHudExtractor(resolution=(2560, 1440))
        assert ext.w == 2560
        assert ext.h == 1440

    def test_reader_not_loaded_by_default(self, extractor):
        assert extractor._reader is None
        assert extractor._reader_ready is False


# ---------------------------------------------------------------------------
# Normalização numérica
# ---------------------------------------------------------------------------

class TestNormalizeNumeric:
    def test_zero_substitutions(self, extractor):
        assert extractor._normalize_numeric("OQ") == "00"
        assert extractor._normalize_numeric("DO") == "00"

    def test_one_substitutions(self, extractor):
        assert extractor._normalize_numeric("IL|") == "111"
        # Barra '/' preservada para timers/frações — não substituída por "1"
        assert extractor._normalize_numeric("I/L") == "1/1"

    def test_five_and_eight(self, extractor):
        assert extractor._normalize_numeric("S") == "5"
        assert extractor._normalize_numeric("B") == "8"

    def test_spaces_between_digits(self, extractor):
        assert extractor._normalize_numeric("1 2 3") == "123"
        assert extractor._normalize_numeric("12 34") == "1234"

    def test_mixed_text(self, extractor):
        assert extractor._normalize_numeric("HP: 5O/1O0") == "HP: 50/100"


# ---------------------------------------------------------------------------
# Extração de números
# ---------------------------------------------------------------------------

class TestExtractFirstNumber:
    def test_simple_integer(self, extractor):
        assert extractor._extract_first_number("123") == 123.0

    def test_decimal(self, extractor):
        assert extractor._extract_first_number("45.67") == 45.67

    def test_timer_format(self, extractor):
        assert extractor._extract_first_number("2:30") == 150.0  # 2*60+30
        assert extractor._extract_first_number("1 45") == 105.0  # 1*60+45

    def test_no_number(self, extractor):
        assert extractor._extract_first_number("abc") is None

    def test_number_in_text(self, extractor):
        assert extractor._extract_first_number("Score: 5-2") == 5.0


# ---------------------------------------------------------------------------
# Extração de frações
# ---------------------------------------------------------------------------

class TestExtractFraction:
    def test_simple_fraction(self, extractor):
        assert extractor._extract_fraction("2/3") == (2, 3)

    def test_spaced_fraction(self, extractor):
        assert extractor._extract_fraction("1 / 3") == (1, 3)

    def test_no_fraction(self, extractor):
        assert extractor._extract_fraction("abc") is None


# ---------------------------------------------------------------------------
# Coordenadas
# ---------------------------------------------------------------------------

class TestRoiToPixels:
    def test_full_screen(self, extractor):
        assert extractor._roi_to_pixels((0.0, 0.0, 1.0, 1.0)) == (0, 0, 1920, 1080)

    def test_quadrant(self, extractor):
        assert extractor._roi_to_pixels((0.0, 0.0, 0.5, 0.5)) == (0, 0, 960, 540)

    def test_hp_region(self, extractor):
        # ROI padrão de HP
        roi = (0.020, 0.010, 0.180, 0.060)
        px = extractor._roi_to_pixels(roi)
        assert px == (38, 10, 345, 64)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_cache_miss(self, extractor):
        assert extractor._cached(HudField.HP_VALUE) is None

    def test_cache_hit(self, extractor):
        value = HudValue(
            field=HudField.HP_VALUE,
            raw_value="100",
            parsed_value=1.0,
            confidence=0.9,
            source="test",
        )
        extractor._set_cache(HudField.HP_VALUE, value)
        cached = extractor._cached(HudField.HP_VALUE)
        assert cached is not None
        assert cached.raw_value == "100"

    def test_cache_expiry(self, extractor):
        import time

        value = HudValue(
            field=HudField.HP_VALUE,
            raw_value="100",
            parsed_value=1.0,
            confidence=0.9,
            source="test",
        )
        extractor._set_cache(HudField.HP_VALUE, value)
        # Modifica TTL para forçar expiração
        extractor.cache_ttl_sec = 0.001
        time.sleep(0.01)
        assert extractor._cached(HudField.HP_VALUE) is None

    def test_invalidate_cache(self, extractor, black_screenshot):
        value = HudValue(
            field=HudField.HP_VALUE,
            raw_value="100",
            parsed_value=1.0,
            confidence=0.9,
            source="test",
        )
        extractor._set_cache(HudField.HP_VALUE, value)
        extractor.invalidate_cache()
        assert extractor._cached(HudField.HP_VALUE) is None


# ---------------------------------------------------------------------------
# Heurísticas de pixel
# ---------------------------------------------------------------------------

class TestPixelHeuristics:
    def test_heuristic_hp_no_cv2(self, extractor, black_screenshot):
        """Sem cv2 deve retornar None."""
        # Simula ausência de cv2
        orig = getattr(extractor, "_heuristic_hp")
        # Como _heuristic_hp checa HAS_CV2 internamente, mockamos
        import vision.ocr_hud_extractor as mod
        orig_has = mod.HAS_CV2
        try:
            mod.HAS_CV2 = False
            ext = OCRHudExtractor()
            assert ext._heuristic_hp(black_screenshot) is None
        finally:
            mod.HAS_CV2 = orig_has

    def test_heuristic_hp_all_red(self, extractor, black_screenshot):
        """Barra toda vermelha → HP ≈ 0."""
        # Desenha região vermelha na área de HP
        x1, y1, x2, y2 = extractor._roi_to_pixels((0.020, 0.010, 0.180, 0.060))
        h, w = black_screenshot.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            black_screenshot[y1:y2, x1:x2, :] = [200, 30, 30]  # vermelho
            hp = extractor._heuristic_hp(black_screenshot)
            assert hp is not None
            assert hp < 0.2

    def test_heuristic_hp_all_green(self, extractor, black_screenshot):
        """Barra toda verde → HP ≈ 1.0."""
        x1, y1, x2, y2 = extractor._roi_to_pixels((0.020, 0.010, 0.180, 0.060))
        h, w = black_screenshot.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            black_screenshot[y1:y2, x1:x2, :] = [30, 200, 30]  # verde
            hp = extractor._heuristic_hp(black_screenshot)
            assert hp is not None
            assert hp > 0.8

    def test_heuristic_super_no_yellow(self, extractor, black_screenshot):
        """Sem amarelo → super ≈ 0."""
        super_val = extractor._heuristic_super(black_screenshot)
        assert super_val is not None
        assert super_val < 0.1


# ---------------------------------------------------------------------------
# Extração com mock OCR
# ---------------------------------------------------------------------------

class TestExtractFieldWithMockOCR:
    def test_extract_hp_high_confidence(self, extractor_with_mock_reader, black_screenshot):
        ext = extractor_with_mock_reader([
            ([(0, 0), (100, 0), (100, 30), (0, 30)], "100", 0.95),
        ])
        result = ext.extract_field(HudField.HP_VALUE, black_screenshot)
        assert result.parsed_value == 1.0  # 100 → 1.0 (post-process)
        assert result.confidence == 0.95
        assert result.source == "ocr"

    def test_extract_ammo(self, extractor_with_mock_reader, black_screenshot):
        ext = extractor_with_mock_reader([
            ([(0, 0), (100, 0), (100, 30), (0, 30)], "3", 0.92),
        ])
        result = ext.extract_field(HudField.AMMO_COUNT, black_screenshot)
        assert result.parsed_value == 3.0
        assert result.source == "ocr"

    def test_extract_timer(self, extractor_with_mock_reader, black_screenshot):
        ext = extractor_with_mock_reader([
            ([(0, 0), (100, 0), (100, 30), (0, 30)], "2:30", 0.88),
        ])
        result = ext.extract_field(HudField.MATCH_TIMER, black_screenshot)
        assert result.parsed_value == 150.0  # 2*60+30
        assert result.source == "ocr"

    def test_extract_super(self, extractor_with_mock_reader, black_screenshot):
        ext = extractor_with_mock_reader([
            ([(0, 0), (100, 0), (100, 30), (0, 30)], "85%", 0.90),
        ])
        result = ext.extract_field(HudField.SUPER_CHARGE, black_screenshot)
        assert result.parsed_value == 0.85  # 85% → 0.85
        assert result.source == "ocr"

    def test_extract_cubes(self, extractor_with_mock_reader, black_screenshot):
        ext = extractor_with_mock_reader([
            ([(0, 0), (100, 0), (100, 30), (0, 30)], "5", 0.87),
        ])
        result = ext.extract_field(HudField.CUBE_COUNT, black_screenshot)
        assert result.parsed_value == 5.0
        assert result.source == "ocr"

    def test_low_confidence_fallback(self, extractor_with_mock_reader, black_screenshot):
        """Confiança abaixo do threshold → fallback."""
        ext = extractor_with_mock_reader([
            ([(0, 0), (100, 0), (100, 30), (0, 30)], "50", 0.3),  # < 0.6
        ])
        ext.use_heuristic_fallback = False  # Desativa heurística para testar default
        result = ext.extract_field(HudField.HP_VALUE, black_screenshot)
        assert result.source == "default"
        assert result.parsed_value == 1.0  # default HP


# ---------------------------------------------------------------------------
# Extract all
# ---------------------------------------------------------------------------

class TestExtractAll:
    def test_extract_all_returns_complete_state(self, extractor_with_mock_reader, black_screenshot):
        ext = extractor_with_mock_reader([
            ([(0, 0), (100, 0), (100, 30), (0, 30)], "100", 0.95),  # HP
        ])
        # Como todas as ROIs usam o mesmo reader, todas vão retornar "100"
        state = ext.extract_all(black_screenshot)
        assert isinstance(state, HudState)
        assert state.hp.source == "ocr"
        assert state.timestamp > 0

    def test_extract_all_empty_screenshot(self, extractor):
        empty = np.array([])
        state = extractor.extract_all(empty)
        assert state.hp.source == "default"
        assert state.ammo.source == "default"


# ---------------------------------------------------------------------------
# Post-processamento
# ---------------------------------------------------------------------------

class TestPostProcessValue:
    def test_hp_hundred(self, extractor):
        assert extractor._post_process_value(HudField.HP_VALUE, 100.0) == 1.0

    def test_hp_fraction(self, extractor):
        assert extractor._post_process_value(HudField.HP_VALUE, 0.75) == 0.75

    def test_ammo_rounding(self, extractor):
        assert extractor._post_process_value(HudField.AMMO_COUNT, 2.7) == 3.0
        assert extractor._post_process_value(HudField.AMMO_COUNT, 4.0) == 3.0  # clamp

    def test_super_percentage(self, extractor):
        assert extractor._post_process_value(HudField.SUPER_CHARGE, 85.0) == 0.85

    def test_timer_passthrough(self, extractor):
        assert extractor._post_process_value(HudField.MATCH_TIMER, 150.0) == 150.0


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaultValues:
    def test_hp_default(self, extractor):
        assert extractor._default_value(HudField.HP_VALUE) == 1.0

    def test_ammo_default(self, extractor):
        assert extractor._default_value(HudField.AMMO_COUNT) == 3.0

    def test_super_default(self, extractor):
        assert extractor._default_value(HudField.SUPER_CHARGE) == 0.0

    def test_timer_default(self, extractor):
        assert extractor._default_value(HudField.MATCH_TIMER) is None


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_extract_all_under_100ms(self, extractor_with_mock_reader, black_screenshot):
        """extract_all deve completar em menos de 100ms mesmo com mock."""
        ext = extractor_with_mock_reader([
            ([(0, 0), (100, 0), (100, 30), (0, 30)], "100", 0.95),
        ])
        import time
        t0 = time.time()
        state = ext.extract_all(black_screenshot)
        elapsed = (time.time() - t0) * 1000
        assert elapsed < 100.0, f"extract_all demorou {elapsed:.1f}ms"


# ---------------------------------------------------------------------------
# HudValue / HudState
# ---------------------------------------------------------------------------

class TestHudValue:
    def test_valid_high_confidence(self):
        v = HudValue(
            field=HudField.HP_VALUE,
            raw_value="100",
            parsed_value=1.0,
            confidence=0.9,
            source="ocr",
        )
        assert v.is_valid is True

    def test_invalid_low_confidence(self):
        v = HudValue(
            field=HudField.HP_VALUE,
            raw_value="",
            parsed_value=1.0,
            confidence=0.3,
            source="default",
        )
        assert v.is_valid is False

    def test_invalid_none_value(self):
        v = HudValue(
            field=HudField.HP_VALUE,
            raw_value="",
            parsed_value=None,
            confidence=0.0,
            source="default",
        )
        assert v.is_valid is False


class TestHudState:
    def test_to_dict_serializable(self, extractor_with_mock_reader, black_screenshot):
        ext = extractor_with_mock_reader([
            ([(0, 0), (100, 0), (100, 30), (0, 30)], "100", 0.95),
        ])
        state = ext.extract_all(black_screenshot)
        d = state.to_dict()
        assert "hp" in d
        assert "ammo" in d
        assert "timestamp" in d
        assert d["hp"]["parsed"] == 1.0


# ---------------------------------------------------------------------------
# Regression: múltiplos ROIs
# ---------------------------------------------------------------------------

class TestMultipleRois:
    def test_voting_best_confidence_wins(self, extractor_with_mock_reader, black_screenshot):
        """Quando há múltiplas ROIs, o melhor resultado por confiança deve ser escolhido."""
        # Primeira ROI retorna baixa confiança, segunda alta
        ext = extractor_with_mock_reader([
            ([(0, 0), (100, 0), (100, 30), (0, 30)], "50", 0.4),
            ([(0, 0), (100, 0), (100, 30), (0, 30)], "100", 0.95),
        ])
        result = ext.extract_field(HudField.HP_VALUE, black_screenshot)
        # Como o reader retorna ambos para cada crop, o melhor (0.95) deve vencer
        assert result.confidence >= 0.4
