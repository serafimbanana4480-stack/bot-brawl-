"""
tests/test_mode_controller.py

Testes para o controlador de modos operacionais.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock

from core.mode_controller import ModeController, ModeStatus


class TestModeStatus:
    def test_default_values(self):
        s = ModeStatus()
        assert s.active_mode is None
        assert s.training_active is False
        assert s.farm_active is False
        assert s.learn_active is False


class TestModeController:
    def test_init(self):
        mc = ModeController()
        assert mc.get_status()["active_mode"] is None

    def test_start_training_success(self):
        wrapper = MagicMock()
        wrapper.toggle_learning_mode = MagicMock(return_value=True)
        mc = ModeController(wrapper_ref=wrapper)
        ok = mc.start_mode("training", {"max_matches": 3})
        assert ok is True
        assert mc.get_status()["training_active"] is True
        assert mc.get_status()["matches_target"] == 3
        wrapper.toggle_learning_mode.assert_called_once_with(enabled=True, max_matches=3)

    def test_stop_training(self):
        wrapper = MagicMock()
        wrapper.toggle_learning_mode = MagicMock(return_value=True)
        mc = ModeController(wrapper_ref=wrapper)
        mc.start_mode("training", {"max_matches": 3})
        ok = mc.stop_mode("training")
        assert ok is True
        assert mc.get_status()["training_active"] is False

    def test_invalid_mode(self):
        mc = ModeController()
        ok = mc.start_mode("invalid")
        assert ok is False

    def test_is_any_active(self):
        mc = ModeController()
        assert mc.is_any_active() is False
        mc._status.active_mode = "farm"
        assert mc.is_any_active() is True

    def test_update_match_count(self):
        mc = ModeController()
        mc.update_match_count(2)
        assert mc.get_status()["matches_completed"] == 2
