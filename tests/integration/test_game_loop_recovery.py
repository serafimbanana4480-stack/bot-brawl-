"""Integration tests for AutoPlayBot game-loop recovery logic."""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from auto_play import AutoPlayBot, FALLBACK


@pytest.fixture
def bot():
    with patch("pylaai_real.screenshot_taker.ScreenshotTaker"), patch(
        "pylaai_real.unified_state_detector.UnifiedStateDetector"
    ), patch("emulator_controller.EmulatorController"):
        instance = AutoPlayBot()
        instance.emulator = Mock()
        instance.detector = Mock()
        instance.screenshot_taker = Mock()
        instance._adb_shell = Mock(return_value=Mock(returncode=0))
        instance._restart_app = Mock()
        instance._tap_btn = Mock(return_value=True)
        instance._tap_1080 = Mock(return_value=True)
        instance._tap_raw = Mock(return_value=True)
        instance._keyevent = Mock(return_value=True)
        instance._swipe_1080 = Mock(return_value=True)
        return instance


def test_matchmaking_timeout_triggers_cancel(bot):
    bot.transition_to("matchmaking")
    bot.state_start_time = time.time() - 35
    with patch("auto_play.time.sleep"):
        bot.handle_matchmaking(np.zeros((1080, 1920, 3), dtype=np.uint8), (960, 900))
    assert bot.state == "lobby"
    bot._tap_btn.assert_called_once_with((960, 900), "cancel_button")


def test_unknown_timeout_triggers_recovery_sequence(bot):
    bot.transition_to("unknown")
    bot.state_start_time = time.time() - 15
    with patch("auto_play.time.sleep"):
        bot.handle_unknown(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert bot.recovery_actions == 1
    assert bot._recovery_attempts == 1
    bot._tap_1080.assert_called_once_with(*FALLBACK["center_screen"])
    bot._keyevent.assert_called_once_with(4)


def test_unknown_recovery_fails_after_three_attempts_triggers_restart(bot):
    bot.transition_to("unknown")
    bot.state_start_time = time.time() - 15
    bot._recovery_attempts = 2
    with patch("auto_play.time.sleep"):
        bot.handle_unknown(np.zeros((1080, 1920, 3), dtype=np.uint8))
    bot._restart_app.assert_called_once()


def test_loading_frozen_triggers_restart(bot):
    bot.transition_to("loading")
    bot.state_start_time = time.time() - 30
    frozen_screen = np.ones((1080, 1920, 3), dtype=np.uint8) * 100
    bot._last_screenshot = frozen_screen.copy()
    with patch("auto_play.time.sleep"):
        bot.handle_loading(frozen_screen.copy())
    bot._restart_app.assert_called_once()


def test_loading_not_frozen_transitions_to_ingame(bot):
    bot.transition_to("loading")
    bot.state_start_time = time.time() - 30
    bot._last_screenshot = np.ones((1080, 1920, 3), dtype=np.uint8) * 100
    changed_screen = np.ones((1080, 1920, 3), dtype=np.uint8) * 200
    with patch("auto_play.time.sleep"):
        bot.handle_loading(changed_screen)
    assert bot.state == "in_game"
    bot._restart_app.assert_not_called()


def test_state_transition_sequence(bot):
    bot.transition_to("lobby")
    assert bot.state == "lobby"
    bot.transition_to("loading")
    assert bot.state == "loading"
    bot.transition_to("matchmaking")
    assert bot.state == "matchmaking"
    bot.transition_to("in_game")
    assert bot.state == "in_game"
    bot.transition_to("end")
    assert bot.state == "end"
    assert bot.state_start_time > 0


def test_update_smoothing_transitions_after_three_detections(bot):
    bot.state = "unknown"
    bot._pending_state = "unknown"
    bot._pending_count = 0
    for _ in range(3):
        bot._update_smoothing("lobby")
    assert bot.state == "lobby"


def test_run_cycle_calls_detector_and_increments_cycle(bot):
    bot.state = "lobby"
    bot.detector.detect.return_value = Mock(
        state="lobby", confidence=0.9, button_coords=(100, 100)
    )
    bot.screenshot_taker.take.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    with patch("auto_play.time.sleep"), patch.object(bot, "handle_lobby"):
        bot.run_cycle()
    assert bot.cycle_count == 1
    bot.detector.detect.assert_called_once()


def test_handle_lobby_clicks_play_and_transitions(bot):
    bot.state = "lobby"
    with patch("auto_play.time.sleep"):
        bot.handle_lobby(np.zeros((1080, 1920, 3), dtype=np.uint8), (1751, 985))
    assert bot.state == "loading"
    bot._tap_btn.assert_called_once()


def test_handle_end_increments_matches_and_returns_to_lobby(bot):
    bot.state = "end"
    with patch("auto_play.time.sleep"):
        bot.handle_end(np.zeros((1080, 1920, 3), dtype=np.uint8), (1554, 990))
    assert bot.state == "lobby"
    assert bot.matches_completed == 1
    bot._keyevent.assert_called_once_with(4)
