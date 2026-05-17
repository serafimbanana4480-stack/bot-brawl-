"""
tests/test_resolution_manager.py

Testes para o sistema centralizado de gestão de resolução.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from core.resolution_manager import (
    ResolutionManager,
    ResolutionProfile,
    CANONICAL_W,
    CANONICAL_H,
)


class TestResolutionProfile:
    """Testes para ResolutionProfile dataclass."""

    def test_basic_properties(self):
        p = ResolutionProfile(actual_width=2560, actual_height=1440)
        assert p.actual_resolution == (2560, 1440)
        assert p.canonical_resolution == (CANONICAL_W, CANONICAL_H)
        assert p.aspect_ratio == pytest.approx(2560 / 1440, 0.01)

    def test_scale_factors(self):
        p = ResolutionProfile(actual_width=2560, actual_height=1440)
        assert p.scale_x == pytest.approx(2560 / CANONICAL_W, 0.001)
        assert p.scale_y == pytest.approx(1440 / CANONICAL_H, 0.001)

    def test_reasonable_resolution(self):
        # Resolucao razoavel
        p = ResolutionProfile(actual_width=1920, actual_height=1080)
        assert p.is_reasonable() is True

    def test_unreasonable_too_small(self):
        p = ResolutionProfile(actual_width=320, actual_height=240)
        assert p.is_reasonable() is False

    def test_unreasonable_too_large(self):
        p = ResolutionProfile(actual_width=8000, actual_height=4500)
        assert p.is_reasonable() is False

    def test_unreasonable_bad_aspect(self):
        p = ResolutionProfile(actual_width=100, actual_height=1000)
        assert p.is_reasonable() is False


class TestResolutionManager:
    """Testes para ResolutionManager."""

    def test_init_defaults(self):
        rm = ResolutionManager()
        assert rm.window_title == "auto"
        assert rm.canonical_resolution == (CANONICAL_W, CANONICAL_H)
        assert rm._profile is None

    def test_fallback_profile(self):
        rm = ResolutionManager()
        profile = rm._fallback_profile()
        assert profile.actual_resolution == (CANONICAL_W, CANONICAL_H)
        assert profile.source == "fallback"

    def test_to_canonical(self):
        rm = ResolutionManager()
        # Forcar perfil 2560x1440
        rm._profile = ResolutionProfile(actual_width=2560, actual_height=1440)
        cx, cy = rm.to_canonical(1280, 720)
        assert cx == round(1280 * CANONICAL_W / 2560)
        assert cy == round(720 * CANONICAL_H / 1440)

    def test_from_canonical(self):
        rm = ResolutionManager()
        rm._profile = ResolutionProfile(actual_width=2560, actual_height=1440)
        ax, ay = rm.from_canonical(960, 540)
        assert ax == round(960 * 2560 / CANONICAL_W)
        assert ay == round(540 * 1440 / CANONICAL_H)

    def test_scale_relative_to_actual(self):
        rm = ResolutionManager()
        rm._profile = ResolutionProfile(actual_width=1920, actual_height=1080)
        ax, ay = rm.scale_relative_to_actual(0.5, 0.5)
        assert ax == 960
        assert ay == 540

    def test_scale_roi_to_actual(self):
        rm = ResolutionManager()
        rm._profile = ResolutionProfile(actual_width=1920, actual_height=1080)
        roi = rm.scale_roi_to_actual((0.1, 0.1, 0.9, 0.9))
        assert roi == (192, 108, 1728, 972)

    def test_scale_roi_to_canonical(self):
        rm = ResolutionManager()
        roi = rm.scale_roi_to_canonical((0.1, 0.1, 0.9, 0.9))
        assert roi == (
            round(0.1 * CANONICAL_W),
            round(0.1 * CANONICAL_H),
            round(0.9 * CANONICAL_W),
            round(0.9 * CANONICAL_H),
        )

    def test_check_for_changes_no_changes(self):
        rm = ResolutionManager(enable_change_detection=True, change_check_interval_sec=0)
        rm._profile = ResolutionProfile(actual_width=1920, actual_height=1080)
        rm._last_check_time = 0
        with patch.object(rm, '_detect_win32', return_value=None):
            with patch.object(rm, '_detect_adb', return_value=None):
                result = rm.check_for_changes()
        assert result is False  # fallback é igual ao anterior

    def test_change_detection_callback(self):
        callback_called = False
        def callback(profile):
            nonlocal callback_called
            callback_called = True

        rm = ResolutionManager(
            on_resolution_change=callback,
            enable_change_detection=True,
            change_check_interval_sec=0,
        )
        rm._profile = ResolutionProfile(actual_width=1920, actual_height=1080)
        rm._last_check_time = 0

        new_profile = ResolutionProfile(actual_width=2560, actual_height=1440, source="win32")
        with patch.object(rm, '_detect_win32', return_value=new_profile):
            rm.check_for_changes()

        assert callback_called is True
        assert rm._profile.change_detected is True
        assert rm._profile.previous_actual == (1920, 1080)

    def test_update_window_title_invalidates_cache(self):
        rm = ResolutionManager(window_title="LDPlayer")
        rm._window_cache["LDPlayer"] = 12345
        rm._profile = ResolutionProfile(actual_width=1920, actual_height=1080)
        rm.update_window_title("BlueStacks")
        assert rm._window_cache == {}
        assert rm._profile is None

    def test_validate_coords_in_bounds(self):
        rm = ResolutionManager()
        rm._profile = ResolutionProfile(actual_width=1920, actual_height=1080)
        assert rm.profile.is_reasonable() is True

    def test_detect_adb_parsing(self):
        rm = ResolutionManager()
        mock_output = "Physical size: 2560x1440\n"
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=mock_output,
                stderr="",
            )
            profile = rm._detect_adb()
        assert profile is not None
        assert profile.actual_resolution == (2560, 1440)
        assert profile.source == "adb"

    def test_detect_adb_failure(self):
        rm = ResolutionManager()
        with patch('subprocess.run', side_effect=Exception("adb not found")):
            profile = rm._detect_adb()
        assert profile is None


class TestResolutionManagerIntegration:
    """Testes de integracao com outros modulos."""

    def test_canonical_roundtrip(self):
        """Verifica que actual -> canonical -> actual preserva valores aproximados."""
        rm = ResolutionManager()
        rm._profile = ResolutionProfile(actual_width=2560, actual_height=1440)

        original = (1280, 720)
        canonical = rm.to_canonical(*original)
        back = rm.from_canonical(*canonical)

        # Deve estar dentro de 1 pixel devido a arredondamento
        assert abs(back[0] - original[0]) <= 1
        assert abs(back[1] - original[1]) <= 1

    def test_different_resolutions_produce_different_scales(self):
        rm = ResolutionManager()

        rm._profile = ResolutionProfile(actual_width=1280, actual_height=720)
        s1 = rm.scale_x

        rm._profile = ResolutionProfile(actual_width=2560, actual_height=1440)
        s2 = rm.scale_x

        assert s1 != s2
        assert s2 == pytest.approx(s1 * 2, 0.01)
