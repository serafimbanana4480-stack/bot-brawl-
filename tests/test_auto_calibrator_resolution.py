"""
tests/test_auto_calibrator_resolution.py

Testes para a integracao do AutoCalibrator com ResolutionManager.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from pylaai_real.auto_calibrator import AutoCalibrator, CalibratedCoords


class TestAutoCalibratorResolutionAware:
    """Testes de resolucao-aware para AutoCalibrator."""

    @pytest.fixture
    def calibrator_with_rm(self):
        """AutoCalibrator com ResolutionManager mock."""
        rm_mock = MagicMock()
        rm_mock.actual_resolution = (2560, 1440)
        rm_mock.check_for_changes.return_value = False
        cal = AutoCalibrator(
            templates_dir=Path("nonexistent"),
            enable_cache=False,
            resolution_manager=rm_mock,
        )
        return cal

    def test_scale_template_for_resolution(self, calibrator_with_rm):
        cal = calibrator_with_rm
        template = np.zeros((100, 100, 3), dtype=np.uint8)
        scaled = cal._scale_template_for_resolution(template)
        # Template canónico 100x100 em 2560x1440 deve ser ~133x133
        assert scaled.shape[0] == 133 or scaled.shape[0] == 134
        assert scaled.shape[1] == 133 or scaled.shape[1] == 134

    def test_coords_to_canonical(self, calibrator_with_rm):
        cal = calibrator_with_rm
        screenshot = np.zeros((1440, 2560, 3), dtype=np.uint8)
        actual = CalibratedCoords(
            element_name="test",
            x=1280, y=720,
            confidence=0.9,
            method="test",
            bbox=(0, 0, 2560, 1440),
        )
        canonical = cal._coords_to_canonical(actual, screenshot)
        assert canonical.x == 960   # 1280 * 1920 / 2560
        assert canonical.y == 540  # 720 * 1080 / 1440
        assert canonical.bbox == (0, 0, 1920, 1080)

    def test_coords_from_canonical(self, calibrator_with_rm):
        cal = calibrator_with_rm
        screenshot = np.zeros((1440, 2560, 3), dtype=np.uint8)
        canonical = CalibratedCoords(
            element_name="test",
            x=960, y=540,
            confidence=0.9,
            method="test",
            bbox=(0, 0, 1920, 1080),
        )
        actual = cal._coords_from_canonical(canonical, screenshot)
        assert actual.x == 1280  # 960 * 2560 / 1920
        assert actual.y == 720   # 540 * 1440 / 1080
        assert actual.bbox == (0, 0, 2560, 1440)

    def test_validate_coords_in_bounds(self, calibrator_with_rm):
        cal = calibrator_with_rm
        screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        coords = CalibratedCoords(element_name="test", x=100, y=100, confidence=0.9, method="test")
        assert cal._validate_coords(coords, screenshot.shape) is True

    def test_validate_coords_out_of_bounds(self, calibrator_with_rm):
        cal = calibrator_with_rm
        screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        coords = CalibratedCoords(element_name="test", x=2000, y=100, confidence=0.9, method="test")
        assert cal._validate_coords(coords, screenshot.shape) is False

    def test_cache_stores_canonical(self, calibrator_with_rm):
        """Verifica que o cache guarda coordenadas canónicas, não reais."""
        cal = calibrator_with_rm
        screenshot = np.zeros((1440, 2560, 3), dtype=np.uint8)
        # Simular deteccao
        detected = CalibratedCoords(
            element_name="play_button",
            x=1280, y=720,
            confidence=0.9,
            method="template",
        )
        # Guardar manualmente como o detect_element faz
        canonical = cal._coords_to_canonical(detected, screenshot)
        cal.coords_cache["play_button"] = canonical

        # Recuperar deve converter de volta
        cached = cal.coords_cache["play_button"]
        actual = cal._coords_from_canonical(cached, screenshot)
        assert actual.x == 1280
        assert actual.y == 720

    def test_detect_element_returns_actual_coords(self, calibrator_with_rm):
        """Verifica que detect_element retorna coordenadas em pixels da screenshot."""
        cal = calibrator_with_rm
        screenshot = np.zeros((1440, 2560, 3), dtype=np.uint8)
        # Mock do template detection para retornar coordenadas reais
        with patch.object(cal, '_detect_template', return_value=CalibratedCoords(
            element_name="play_button",
            x=1280, y=720,
            confidence=0.8,
            method="template",
        )):
            result = cal.detect_element(screenshot, "play_button")
        assert result is not None
        assert result.x == 1280
        assert result.y == 720

    def test_detect_element_with_fallback_coords(self, calibrator_with_rm):
        """Fallback coords sao fornecidas em canónico e convertidas para actual."""
        cal = calibrator_with_rm
        screenshot = np.zeros((1440, 2560, 3), dtype=np.uint8)
        # Forcar falha em todos os metodos
        with patch.object(cal, '_detect_template', return_value=None):
            with patch.object(cal, '_detect_color', return_value=None):
                with patch.object(cal, '_detect_ocr', return_value=None):
                    result = cal.detect_element(
                        screenshot, "play_button",
                        fallback_coords=(960, 540)  # canónico
                    )
        assert result is not None
        # Deve estar convertido para actual: 960 * 2560/1920 = 1280
        assert result.x == 1280
        assert result.y == 720
