"""Tests for core.state_detection module."""

import pytest
import time
from unittest.mock import MagicMock, patch

from core.state_detection import StateDetectionMixin


class DummyStateManager(StateDetectionMixin):
    """Dummy class for testing StateDetectionMixin."""

    def __init__(self):
        self.screenshot = MagicMock()
        self.state_finder = MagicMock()
        self.unified_detector = None
        self.lobby = None
        self.movement = None
        self.screen_automation = None
        self.current_state = "unknown"
        self.last_known_state = "unknown"
        self.last_known_state_at = None
        self.unknown_streak = 0
        self.last_unknown_hint = None
        self.unknown_since = None
        self._unknown_hold_cycles = 2
        self._last_screenshot = None
        self._last_screenshot_time = 0
        self._paused = False
        self._adaptive_cache = None
        self.state_start_time = None
        self._last_transition_time = 0.0
        self._state_transition_cooldown = 3.0
        self._in_game_lock = False
        self._in_game_min_duration = 5.0
        self.VALID_TRANSITIONS = {"unknown": ["lobby"], "lobby": ["loading"]}
        self.state_timeouts = {"lobby": 60}
        self.improvement_system = None
        self.diagnostic_mode = False


class TestStateDetectionMixin:
    """Tests for StateDetectionMixin."""

    def test_remember_known_state(self):
        """_remember_known_state should update tracking fields."""
        sm = DummyStateManager()
        sm._remember_known_state("lobby")
        assert sm.last_known_state == "lobby"
        assert sm.unknown_streak == 0
        assert sm.last_known_state_at is not None

    def test_get_cached_screenshot_fresh(self):
        """Should return cached screenshot if fresh."""
        sm = DummyStateManager()
        fake_img = MagicMock()
        sm._last_screenshot = fake_img
        sm._last_screenshot_time = time.time()
        result = sm._get_cached_screenshot(max_age=1.0)
        # Method returns a copy; MagicMock.copy() returns a new MagicMock
        assert result is not None

    def test_get_cached_screenshot_expired(self):
        """Should call screenshot.take if cache expired."""
        sm = DummyStateManager()
        fake_img = MagicMock()
        sm.screenshot.take.return_value = fake_img
        sm._last_screenshot_time = time.time() - 10.0
        result = sm._get_cached_screenshot(max_age=1.0)
        sm.screenshot.take.assert_called_once()
        assert result is fake_img

    def test_diag_no_crash(self):
        """_diag should not raise."""
        sm = DummyStateManager()
        sm._diag("test_message")

    def test_log_lobby_snapshot_no_crash(self):
        """_log_lobby_snapshot should not raise."""
        sm = DummyStateManager()
        sm._log_lobby_snapshot("test_context")

    def test_wait_for_state_timeout(self):
        """_wait_for_state should return False on timeout."""
        sm = DummyStateManager()
        sm.current_state = "lobby"
        result = sm._wait_for_state("loading", timeout=0.01)
        assert result is False

    def test_force_click_play_no_crash(self):
        """_force_click_play should not raise."""
        sm = DummyStateManager()
        sm.lobby = MagicMock()
        sm.lobby._click = MagicMock()
        sm._force_click_play()
