"""
tests/test_learning_mode.py

Tests for the LearningModeController.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.learning_mode import LearningModeController
from core.learning_metrics import LearningMetricsCollector


@pytest.fixture
def controller(tmp_path):
    lobby = MagicMock()
    lobby.lobby_config = MagicMock()
    lobby.lobby_config.w = 1920
    lobby.lobby_config.h = 1080
    # Simulate successful Training Cave entry via expanded module
    lobby.enter_training_cave.return_value = MagicMock(success=True, method_used="menu_navigation")
    lobby.exit_training_cave.return_value = MagicMock(success=True)
    lobby.restart_training.return_value = True

    emu = MagicMock()
    screenshot = MagicMock()
    screenshot.take.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)

    state_finder = MagicMock()
    state_finder.get_state.return_value = "in_game_learning"

    play = MagicMock()
    play.play_round.return_value = {"action": "attack"}
    play.last_combat_snapshot = {"enemies": 2, "player": {"x": 100, "y": 200}}
    play.set_pve_mode = MagicMock()

    metrics = LearningMetricsCollector(output_dir=tmp_path)

    ctrl = LearningModeController(
        lobby_automator=lobby,
        emulator_controller=emu,
        screenshot_taker=screenshot,
        state_finder=state_finder,
        play_logic=play,
        metrics_collector=metrics,
        max_matches=2,
        match_timeout_seconds=5.0,
    )
    return ctrl


class TestLearningModeController:
    def test_enter_training_cave_success(self, controller):
        ok = controller.enter_training_cave()
        assert ok is True
        controller.lobby.enter_training_cave.assert_called_once()

    def test_enter_training_cave_fallback(self, controller):
        controller.lobby.enter_training_cave.return_value = None
        ok = controller.enter_training_cave()
        assert ok is True
        assert controller.emulator_controller.tap_scaled.call_count >= 2

    def test_start_match(self, controller):
        controller.start_match("colt")
        assert controller._match_active is True
        assert controller.metrics.current_match.brawler == "colt"

    def test_run_frame(self, controller):
        controller.start_match("colt")
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = controller.run_frame(img)
        assert controller.play.play_round.called
        assert controller.metrics.current_match.actions_attack == 1

    def test_is_match_ended_timeout(self, controller):
        controller.start_match("colt")
        controller._match_start_time = time.time() - 10.0
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ended, reason = controller.is_match_ended(img)
        assert ended is True
        assert reason == "timeout"

    def test_is_match_ended_by_state(self, controller):
        controller.start_match("colt")
        controller.state_finder.get_state.return_value = "lobby"
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ended, reason = controller.is_match_ended(img)
        assert ended is True
        assert "state_changed" in reason

    def test_restart_training(self, controller):
        ok = controller.restart_training()
        assert ok is True
        controller.lobby.restart_training.assert_called_once()

    def test_should_continue(self, controller):
        assert controller.should_continue() is True
        controller._current_match_index = 2
        assert controller.should_continue() is False

    def test_print_summary(self, controller):
        controller.start_match("shelly")
        controller.end_match("completed")
        # Should not raise
        controller.print_summary()
