"""Tests for core.state_transitions module."""

import pytest
from unittest.mock import MagicMock, patch

from core.state_transitions import StateTransitionsMixin


class DummyStateManager(StateTransitionsMixin):
    """Dummy class for testing StateTransitionsMixin."""

    def __init__(self):
        self.lobby = MagicMock()
        self.play = MagicMock()
        self.match_controller = MagicMock()
        self.emulator_controller = MagicMock()
        self.screen_automation = None
        self.movement = MagicMock()
        self.reward_bridge = None
        self.data_collector = None
        self.brawler_selector = None
        self.observability = None
        self.rl_engine = None
        self.learning_mode_controller = None
        self.auto_fix = None
        self.unified_detector = None
        self.ocr_detector = None
        self.lobby_fsm = None
        self._adaptive_cache = None
        self._state_persistence = None
        self.improvement_system = None
        self.progress = MagicMock()
        self._get_window_size = MagicMock(return_value=(1920, 1080))
        self.screenshot = MagicMock()
        self.state_finder = MagicMock()
        self.current_state = "unknown"
        self.current_brawler = None
        self._current_map = None
        self.last_known_state = "unknown"
        self.last_known_state_at = None
        self.unknown_streak = 0
        self.last_unknown_hint = None
        self.unknown_since = None
        self._unknown_hold_cycles = 2
        self._last_screenshot = None
        self._last_screenshot_time = 0
        self._paused = False
        self.state_start_time = None
        self._last_transition_time = 0.0
        self._state_transition_cooldown = 3.0
        self._in_game_lock = False
        self._in_game_min_duration = 5.0
        self._forced_in_game_time = None
        self._matchmaking_enter_time = None
        self._in_game_initialized = False
        self._popup_consecutive_count = 0
        self._last_popup_type = None
        self._popup_ignore_until = 0.0
        self.VALID_TRANSITIONS = {"unknown": ["lobby"], "lobby": ["loading"]}
        self.state_timeouts = {"lobby": 60}

    def _remember_known_state(self, state):
        self.last_known_state = state
        self.last_known_state_at = 0
        self.unknown_streak = 0

    def _get_cached_screenshot(self, max_age=0.15):
        return MagicMock()

    def _diag(self, msg):
        pass

    def _log_lobby_snapshot(self, context=""):
        pass

    def _wait_for_state(self, state, timeout=10):
        return False

    def _force_click_play(self):
        pass


class TestStateTransitionsMixin:
    """Tests for StateTransitionsMixin handlers."""

    def test_handle_lobby_no_crash(self):
        """_handle_lobby should not raise with mocked dependencies."""
        sm = DummyStateManager()
        sm.lobby.get_next_action.return_value = ("click", 100, 200)
        sm._handle_lobby()

    def test_handle_brawler_selection_no_crash(self):
        """_handle_brawler_selection should not raise."""
        sm = DummyStateManager()
        sm._handle_brawler_selection()

    def test_handle_loading_no_crash(self):
        """_handle_loading should not raise."""
        sm = DummyStateManager()
        sm._handle_loading()

    def test_handle_matchmaking_no_crash(self):
        """_handle_matchmaking should not raise."""
        sm = DummyStateManager()
        sm._handle_matchmaking()

    def test_handle_connection_lost_no_crash(self):
        """_handle_connection_lost should not raise."""
        sm = DummyStateManager()
        sm._handle_connection_lost()

    def test_handle_shop_no_crash(self):
        """_handle_shop should not raise."""
        sm = DummyStateManager()
        sm._handle_shop()

    def test_handle_popup_no_crash(self):
        """_handle_popup should not raise."""
        sm = DummyStateManager()
        sm._handle_popup()

    def test_handle_in_game_no_crash(self):
        """_handle_in_game should not raise with mocked dependencies."""
        sm = DummyStateManager()
        fake_img = MagicMock()
        sm.play.play_round.return_value = {"action": "attack"}
        sm._handle_in_game(fake_img)

    def test_handle_end_game_no_crash(self):
        """_handle_end_game should not raise (StopIteration is expected flow control)."""
        sm = DummyStateManager()
        try:
            sm._handle_end_game()
        except StopIteration:
            pass

    def test_handle_unknown_no_crash(self):
        """_handle_unknown should not raise."""
        sm = DummyStateManager()
        fake_img = MagicMock()
        sm._handle_unknown(fake_img)

    def test_handle_tutorial_no_crash(self):
        """_handle_tutorial should not raise."""
        sm = DummyStateManager()
        sm._handle_tutorial()

    def test_handle_news_no_crash(self):
        """_handle_news should not raise."""
        sm = DummyStateManager()
        sm._handle_news()

    def test_handle_brawler_unlock_no_crash(self):
        """_handle_brawler_unlock should not raise."""
        sm = DummyStateManager()
        sm._handle_brawler_unlock()

    def test_handle_season_reset_no_crash(self):
        """_handle_season_reset should not raise."""
        sm = DummyStateManager()
        sm._handle_season_reset()

    def test_handle_event_screen_no_crash(self):
        """_handle_event_screen should not raise."""
        sm = DummyStateManager()
        sm._handle_event_screen()

    def test_handle_starr_drop_no_crash(self):
        """_handle_starr_drop should not raise."""
        sm = DummyStateManager()
        sm._handle_starr_drop()

    def test_handle_in_game_learning_no_crash(self):
        """_handle_in_game_learning should not raise."""
        sm = DummyStateManager()
        fake_img = MagicMock()
        sm._handle_in_game_learning(fake_img)

    def test_safe_back_to_lobby_no_crash(self):
        """_safe_back_to_lobby should not raise."""
        sm = DummyStateManager()
        sm._safe_back_to_lobby()
