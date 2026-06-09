"""
tests/test_esp_overlay.py

Testes para o ESP Overlay.
"""

import sys, os

import pytest
from unittest.mock import patch, MagicMock

try:
    from core.esp_overlay import COLORS, LABELS
    HAS_PYWIN32 = True
except ImportError:
    HAS_PYWIN32 = False
    COLORS = LABELS = None

pytestmark = pytest.mark.skipif(not HAS_PYWIN32, reason="pywin32 não disponível")


class TestESPOverlay:
    @patch("core.esp_overlay.win32gui.FindWindow", return_value=12345)
    @patch("core.esp_overlay.win32gui.GetWindowRect", return_value=(100, 100, 500, 400))
    def test_init(self, mock_rect, mock_find):
        from core.esp_overlay import ESPOverlay
        esp = ESPOverlay(window_title="FakeWindow")
        assert esp.hwnd is not None
        assert esp.enabled is False
        assert esp.detections == []

    @patch("core.esp_overlay.win32gui.FindWindow", return_value=12345)
    @patch("core.esp_overlay.win32gui.GetWindowRect", return_value=(100, 100, 500, 400))
    def test_toggle(self, mock_rect, mock_find):
        from core.esp_overlay import ESPOverlay
        esp = ESPOverlay(window_title="FakeWindow")
        assert esp.toggle(True) is True
        assert esp.enabled is True
        assert esp.toggle(False) is True
        assert esp.enabled is False
        assert esp.toggle() is True
        assert esp.enabled is True

    @patch("core.esp_overlay.win32gui.FindWindow", return_value=12345)
    @patch("core.esp_overlay.win32gui.GetWindowRect", return_value=(100, 100, 500, 400))
    def test_update_detections(self, mock_rect, mock_find):
        from core.esp_overlay import ESPOverlay
        esp = ESPOverlay(window_title="FakeWindow")
        dets = [
            {"class_name": "enemy", "confidence": 0.9, "x": 10, "y": 20, "width": 30, "height": 40},
            {"class_name": "bush", "confidence": 0.8, "x": 100, "y": 200, "width": 50, "height": 50},
        ]
        esp.update_detections(dets)
        assert len(esp.detections) == 2
        assert esp.detections[0]["class_name"] == "enemy"

    def test_color_for_class(self):
        from core.esp_overlay import COLOR_FOR_CLASS
        assert COLOR_FOR_CLASS("enemy") == COLORS["red"]
        assert COLOR_FOR_CLASS("bush") == COLORS["green"]
        assert COLOR_FOR_CLASS("unknown") == COLORS["white"]

    def test_text_for_class(self):
        from core.esp_overlay import TEXT_FOR_CLASS
        assert TEXT_FOR_CLASS("enemy", 0.91) == "enemy 91%"
        assert TEXT_FOR_CLASS("wall", 0.75) == "wall"
