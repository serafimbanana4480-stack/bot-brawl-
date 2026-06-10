"""Tests for core.movement.movement_engine module."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from core.movement.movement_engine import MovementEngineMixin


class DummyPlayLogic(MovementEngineMixin):
    """Dummy class for testing MovementEngineMixin."""

    def __init__(self):
        self.detect_main = MagicMock()
        self.movement = MagicMock()
        self.emulator_controller = None
        self.humanization = None
        self.current_brawler = None
        self.brawler_strategy = {}
        self._last_action = "idle"
        self.last_combat_snapshot = {
            "state": "idle",
            "player": None,
            "enemies": 0,
            "bushes": 0,
            "power_cubes": 0,
            "move_key": "",
            "attack_taken": False,
            "super_taken": False,
            "window_active": None,
            "window_title": None,
            "target_position": None,
            "last_error": None,
        }


class TestMovementEngineMixin:
    """Tests for MovementEngineMixin movement methods."""

    def test_distance_calculation(self):
        """_distance should compute Euclidean distance correctly."""
        pl = DummyPlayLogic()
        assert pl._distance((0, 0), (3, 4)) == 5.0
        assert pl._distance((0, 0), (0, 0)) == 0.0

    def test_find_player_none(self):
        """_find_player should return None when no player detected."""
        pl = DummyPlayLogic()
        pl.detect_main.detect.return_value = {"player": []}
        result = pl._find_player(MagicMock())
        assert result is None

    def test_find_bushes_empty(self):
        """_find_bushes should return empty list when no bushes."""
        pl = DummyPlayLogic()
        pl.detect_main.detect.return_value = {"bushes": []}
        result = pl._find_bushes(MagicMock())
        assert result == []

    def test_find_power_cubes_empty(self):
        """_find_power_cubes should return empty list when no cubes."""
        pl = DummyPlayLogic()
        pl.detect_main.detect.return_value = {"power_cubes": []}
        result = pl._find_power_cubes(MagicMock())
        assert result == []

    def test_execute_movement_no_crash(self):
        """_execute_movement should not raise with mocked emulator_controller."""
        pl = DummyPlayLogic()
        pl.emulator_controller = MagicMock()
        pl._execute_movement("W")
        pl.emulator_controller.ensure_window_active.assert_called_once()

    def test_is_in_bush_no_bushes(self):
        """_is_in_bush should return False when no bushes detected."""
        pl = DummyPlayLogic()
        result = pl._is_in_bush((100, 100, 200, 200), None)
        assert result is False
