"""Tests for core.combat.combat_engine module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from core.combat.combat_engine import CombatEngineMixin


class DummyPlayLogic(CombatEngineMixin):
    """Dummy class for testing CombatEngineMixin."""

    def __init__(self):
        self.enemy_history = {}
        self.last_shot_time = 0
        self.shot_cooldown = 0.35
        self.shot_cooldown_jitter = 0.15
        self.super_ready = False
        self._human_pause = MagicMock()
        self._apm_action_count = 0
        self._apm_window_start = 0
        self._last_rl_state = None
        self._last_rl_action = None
        self.last_rl_transition = None
        self.current_game_mode = "showdown"
        self.pve_mode = None
        self._combat_strategy = None
        self._leading_engine = None
        self._combo_manager = None
        self._utility_ai = None
        self._sticky_target = None
        self._intent_system = None
        self.meta_awareness = None
        self.cover_system = None
        self._feature_extractor = None
        self._occupancy_grid = None
        self._enable_utility_ai = True
        self._utility_ai_threshold = 0.70
        self._enable_intent_system = True
        self._enable_enemy_intention = True
        self.attack_distance = 200
        self.aggressiveness = 0.5
        self.enemy_tracker = None
        self.current_brawler = None
        self.brawler_strategy = None
        self.brawler_strategies = {}
        self._last_action = "idle"
        self._last_enemies = 0
        self._game_phase = "early"
        self._match_start_time = None
        self._power_cubes_collected = 0
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
        self.detect_main = MagicMock()
        self.detect_enemies = MagicMock()
        self.movement = MagicMock()
        self.humanization = None
        self.emulator_controller = None
        self.rl_engine = None
        self.central_coordinator = None
        self.world_model = None
        self.pressure_map = None
        self.enemy_intention = None
        self._normalize_bbox = lambda bbox: bbox if bbox and len(bbox) == 4 else None


class TestCombatEngineMixin:
    """Tests for CombatEngineMixin combat methods."""

    def test_set_current_brawler(self):
        """set_current_brawler should update current_brawler."""
        pl = DummyPlayLogic()
        pl.set_current_brawler("colt")
        assert pl.current_brawler == "colt"

    def test_set_current_game_mode(self):
        """set_current_game_mode should normalize mode name."""
        pl = DummyPlayLogic()
        pl.set_current_game_mode("Gem Grab")
        assert pl.current_game_mode == "gem_grab"

    def test_set_pve_mode(self):
        """set_pve_mode should set pve_mode."""
        pl = DummyPlayLogic()
        pl.set_pve_mode("training_cave")
        assert pl.pve_mode == "training_cave"

    def test_get_brawler_strategy_default(self):
        """get_brawler_strategy should return default if none set."""
        pl = DummyPlayLogic()
        strategy = pl.get_brawler_strategy()
        assert isinstance(strategy, dict)

    def test_reset_for_new_match(self):
        """reset_for_new_match should clear enemy history."""
        pl = DummyPlayLogic()
        pl.enemy_history[1] = [(0, 0, 0)]
        pl.reset_for_new_match()
        assert len(pl.enemy_history) == 0

    def test_get_last_combat_snapshot(self):
        """get_last_combat_snapshot should return dict."""
        pl = DummyPlayLogic()
        snap = pl.get_last_combat_snapshot()
        assert isinstance(snap, dict)
        assert "state" in snap

    def test_get_enemy_id_consistency(self):
        """_get_enemy_id should return consistent IDs for same bbox."""
        pl = DummyPlayLogic()
        bbox = [100, 100, 200, 200]
        id1 = pl._get_enemy_id(bbox)
        id2 = pl._get_enemy_id(bbox)
        assert id1 == id2

    def test_find_enemies_empty(self):
        """_find_enemies should return empty list when no detections."""
        pl = DummyPlayLogic()
        pl.detect_enemies.detect.return_value = []
        enemies = pl._find_enemies(MagicMock())
        assert enemies == []

    def test_estimate_enemy_hp_default(self):
        """_estimate_enemy_hp should return default for unknown."""
        pl = DummyPlayLogic()
        hp = pl._estimate_enemy_hp([100, 100, 200, 200])
        assert hp == 1.0
