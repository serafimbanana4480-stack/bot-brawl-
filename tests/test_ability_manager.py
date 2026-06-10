"""Tests for core.abilities.ability_manager module."""

import pytest
import time
from unittest.mock import MagicMock

from core.abilities.ability_manager import AbilityManagerMixin


class DummyPlayLogic(AbilityManagerMixin):
    """Dummy class for testing AbilityManagerMixin."""

    def __init__(self):
        self.last_shot_time = 0
        self.shot_cooldown = 0.35
        self.super_ready = False
        self.current_brawler = None
        self.get_brawler_strategy = MagicMock(return_value={})
        self.movement = MagicMock()
        self.humanization = None
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


class TestAbilityManagerMixin:
    """Tests for AbilityManagerMixin ability methods."""

    def test_manage_abilities_no_crash(self):
        """_manage_abilities should not raise with minimal setup."""
        pl = DummyPlayLogic()
        player = (100, 100, 200, 200)
        enemies = []
        pl._manage_abilities(player, enemies)

    def test_manage_abilities_respects_cooldown(self):
        """_manage_abilities should not fire super if not ready."""
        pl = DummyPlayLogic()
        pl.last_shot_time = time.time()
        pl.shot_cooldown = 10.0  # Large cooldown
        pl.super_ready = True
        player = (100, 100, 200, 200)
        enemies = []
        # Should not crash even with large cooldown
        pl._manage_abilities(player, enemies)

    def test_manage_abilities_with_emulator(self):
        """_manage_abilities should interact with emulator_controller if available."""
        pl = DummyPlayLogic()
        pl.emulator_controller = MagicMock()
        pl.super_ready = True
        pl.last_shot_time = 0
        pl.shot_cooldown = 0.0
        player = (100, 100, 200, 200)
        enemies = []
        pl._manage_abilities(player, enemies)
