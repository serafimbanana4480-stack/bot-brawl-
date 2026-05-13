"""
Tests for tracker reset integration.

Verifies that tracker reset works correctly when called between matches.
"""

import pytest
from unittest.mock import Mock


def test_tracker_reset_clears_tracks():
    """Verifies that reset() clears tracks and frame_count."""
    from brawl_bot.tracker import MultiObjectTracker

    # Create tracker
    tracker = MultiObjectTracker(max_age=30, min_hits=3)

    # Add some tracks
    tracker.tracks = [Mock(id=1), Mock(id=2)]
    tracker.frame_count = 100
    tracker.next_id = 3

    # Reset tracker
    tracker.reset()

    # Verify tracks are cleared
    assert len(tracker.tracks) == 0
    # Verify frame_count is reset
    assert tracker.frame_count == 0
    # Verify next_id is reset
    assert tracker.next_id == 1


def test_play_logic_resets_tracker_on_new_match():
    """Verifies that reset_for_new_match() is called and resets tracker."""
    from brawl_bot.pylai_real.play import PlayLogic
    from unittest.mock import Mock

    # Create mock tracker
    enemy_tracker = Mock()
    enemy_tracker.reset = Mock()

    # Create play logic with tracker
    play_logic = PlayLogic(
        detect_main=None,
        movement=None,
        enemy_tracker=enemy_tracker
    )

    # Call reset_for_new_match
    play_logic.reset_for_new_match()

    # Verify tracker reset was called
    enemy_tracker.reset.assert_called_once()

    # Verify enemy_history is cleared
    assert len(play_logic.enemy_history) == 0


def test_play_logic_clears_enemy_history():
    """Verifies that reset_for_new_match() clears enemy_history."""
    from brawl_bot.pylai_real.play import PlayLogic

    # Create play logic without tracker
    play_logic = PlayLogic(
        detect_main=None,
        movement=None,
        enemy_tracker=None
    )

    # Add some enemy history
    play_logic.enemy_history = {
        1: [(100, 200, 100.0), (105, 205, 100.1)],
        2: [(300, 400, 100.0)]
    }

    # Call reset_for_new_match
    play_logic.reset_for_new_match()

    # Verify enemy_history is cleared
    assert len(play_logic.enemy_history) == 0


def test_state_manager_triggers_reset_on_in_game():
    """Verifies that StateManager calls reset_for_new_match() when entering in_game state."""
    from brawl_bot.pylai_real.state_manager import StateManager
    from unittest.mock import Mock

    # Create mock play_logic
    play_logic = Mock()
    play_logic.reset_for_new_match = Mock()

    # Create mock other dependencies
    screenshot_taker = Mock()
    state_finder = Mock()
    lobby = Mock()
    progress_observer = Mock()
    match_controller = Mock()
    emulator_controller = Mock()
    screen_automation = Mock()
    movement = Mock()

    # Create state manager
    state_manager = StateManager(
        screenshot_taker=screenshot_taker,
        state_finder=state_finder,
        lobby=lobby,
        progress_observer=progress_observer,
        play_logic=play_logic,
        match_controller=match_controller,
        emulator_controller=emulator_controller,
        screen_automation=screen_automation,
        movement=movement
    )

    # Mock state_finder to return in_game
    state_finder.get_state = Mock(return_value="in_game")
    screenshot_taker.take = Mock(return_value=Mock())

    # Process a cycle to trigger _handle_in_game
    state_manager._process_cycle()

    # Verify reset_for_new_match was called
    play_logic.reset_for_new_match.assert_called_once()


def test_tracker_reset_does_not_crash_without_tracker():
    """Verifies that reset_for_new_match() works without tracker."""
    from brawl_bot.pylai_real.play import PlayLogic

    # Create play logic without tracker
    play_logic = PlayLogic(
        detect_main=None,
        movement=None,
        enemy_tracker=None
    )

    # Call reset_for_new_match - should not crash
    play_logic.reset_for_new_match()

    # Verify enemy_history is still cleared
    assert len(play_logic.enemy_history) == 0


def test_tracker_stats_after_reset():
    """Verifies that get_stats() returns correct values after reset."""
    from brawl_bot.tracker import MultiObjectTracker

    # Create tracker
    tracker = MultiObjectTracker(max_age=30, min_hits=3)

    # Add some tracks
    tracker.tracks = [Mock(id=1), Mock(id=2)]
    tracker.frame_count = 100
    tracker.next_id = 3

    # Get stats before reset
    stats_before = tracker.get_stats()
    assert stats_before["active_tracks"] == 2
    assert stats_before["frame_count"] == 100
    assert stats_before["next_track_id"] == 3

    # Reset tracker
    tracker.reset()

    # Get stats after reset
    stats_after = tracker.get_stats()
    assert stats_after["active_tracks"] == 0
    assert stats_after["frame_count"] == 0
    assert stats_after["next_track_id"] == 1
