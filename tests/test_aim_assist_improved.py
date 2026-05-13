"""
Tests for improved aim assist integration.

Verifies that predict_position() uses tracker and get_velocity() when available.
"""

import pytest
from unittest.mock import Mock


def test_predict_position_uses_tracker_when_available():
    """Verifies that _predict_position() uses tracker.predict_position() when available."""
    from brawl_bot.pylai_real.play import PlayLogic

    # Create mock tracker
    enemy_tracker = Mock()
    enemy_tracker.get_all_tracks = Mock(return_value=[
        Mock(id=1, confirmed=True, bbox=(100, 100, 200, 200))
    ])
    enemy_tracker.predict_position = Mock(return_value=(150, 150))

    # Create play logic with tracker
    play_logic = PlayLogic(
        detect_main=None,
        movement=None,
        enemy_tracker=enemy_tracker
    )

    # Call _predict_position
    enemy_bbox = (100, 100, 200, 200)
    pred_pos = play_logic._predict_position(enemy_bbox, time_ahead=0.25)

    # Verify predict_position was called
    enemy_tracker.predict_position.assert_called_once()
    # Verify predicted position is returned
    assert pred_pos == (150, 150)


def test_predict_position_fallback_to_simple_estimation():
    """Verifies that _predict_position() falls back to simple estimation when tracker is not available."""
    from brawl_bot.pylai_real.play import PlayLogic

    # Create play logic without tracker
    play_logic = PlayLogic(
        detect_main=None,
        movement=None,
        enemy_tracker=None
    )

    # Call _predict_position
    enemy_bbox = (100, 100, 200, 200)
    pred_pos = play_logic._predict_position(enemy_bbox, time_ahead=0.25)

    # Should return center position (no history yet)
    assert pred_pos == (150, 150)


def test_predict_position_uses_velocity():
    """Verifies that _predict_position() uses get_velocity() when available."""
    from brawl_bot.pylai_real.play import PlayLogic

    # Create mock tracker
    enemy_tracker = Mock()
    enemy_tracker.get_all_tracks = Mock(return_value=[
        Mock(id=1, confirmed=True, bbox=(100, 100, 200, 200))
    ])
    enemy_tracker.predict_position = Mock(return_value=None)  # predict_position fails
    enemy_tracker.get_velocity = Mock(return_value=(50.0, 30.0))  # velocity available

    # Create play logic with tracker
    play_logic = PlayLogic(
        detect_main=None,
        movement=None,
        enemy_tracker=enemy_tracker
    )

    # Call _predict_position
    enemy_bbox = (100, 100, 200, 200)
    pred_pos = play_logic._predict_position(enemy_bbox, time_ahead=0.25)

    # Verify get_velocity was called
    enemy_tracker.get_velocity.assert_called_once()
    # Verify prediction uses velocity: 150 + 50*0.25 = 162.5, 150 + 30*0.25 = 157.5
    assert pred_pos == (162, 157)


def test_get_track_info_returns_track_info():
    """Verifies that _get_track_info() returns track information when available."""
    from brawl_bot.pylai_real.play import PlayLogic

    # Create mock tracker
    enemy_tracker = Mock()
    enemy_tracker.get_all_tracks = Mock(return_value=[
        Mock(id=1, confirmed=True, hit_streak=3, age=10, bbox=(100, 100, 200, 200))
    ])

    # Create play logic with tracker
    play_logic = PlayLogic(
        detect_main=None,
        movement=None,
        enemy_tracker=enemy_tracker
    )

    # Call _get_track_info
    enemy_bbox = (100, 100, 200, 200)
    track_info = play_logic._get_track_info(enemy_bbox)

    # Verify track info is returned
    assert track_info is not None
    assert track_info["id"] == 1
    assert track_info["hit_streak"] == 3
    assert track_info["age"] == 10


def test_get_track_info_returns_none_without_tracker():
    """Verifies that _get_track_info() returns None when tracker is not available."""
    from brawl_bot.pylai_real.play import PlayLogic

    # Create play logic without tracker
    play_logic = PlayLogic(
        detect_main=None,
        movement=None,
        enemy_tracker=None
    )

    # Call _get_track_info
    enemy_bbox = (100, 100, 200, 200)
    track_info = play_logic._get_track_info(enemy_bbox)

    # Verify None is returned
    assert track_info is None


def test_try_smart_attack_uses_track_info():
    """Verifies that _try_smart_attack() uses track info to prioritize targets."""
    from brawl_bot.pylai_real.play import PlayLogic
    from unittest.mock import patch

    # Create mock tracker
    enemy_tracker = Mock()
    enemy_tracker.get_all_tracks = Mock(return_value=[
        Mock(id=1, confirmed=True, hit_streak=3, bbox=(100, 100, 200, 200)),
        Mock(id=2, confirmed=True, hit_streak=0, bbox=(300, 300, 400, 400))
    ])

    # Create play logic with tracker
    play_logic = PlayLogic(
        detect_main=None,
        movement=None,
        enemy_tracker=enemy_tracker
    )

    # Mock get_brawler_strategy
    play_logic.get_brawler_strategy = Mock(return_value={"recommended_distance": 400})

    # Call _try_smart_attack with two enemies
    player = (500, 500)
    enemies = [(100, 100, 200, 200), (300, 300, 400, 400)]

    with patch.object(play_logic, '_distance') as mock_distance, \
         patch.object(play_logic, '_get_track_info', wraps=play_logic._get_track_info) as mock_get_track_info:
        mock_distance.side_effect = lambda p, e: 200  # Same distance for both

        play_logic._try_smart_attack(player, enemies)

        # Verify _get_track_info was called for both enemies
        assert mock_get_track_info.call_count == 2


def test_predict_position_with_enemy_history():
    """Verifies that _predict_position() uses enemy_history when tracker is not available."""
    from brawl_bot.pylai_real.play import PlayLogic

    # Create play logic without tracker
    play_logic = PlayLogic(
        detect_main=None,
        movement=None,
        enemy_tracker=None
    )

    # Add enemy history (moving only in x, y stays at 155)
    enemy_id = play_logic._get_enemy_id((100, 100, 200, 200))
    play_logic.enemy_history[enemy_id] = [
        (150, 155, 100.0),
        (155, 155, 100.1)
    ]

    # Call _predict_position
    enemy_bbox = (100, 100, 200, 200)
    pred_pos = play_logic._predict_position(enemy_bbox, time_ahead=0.25)

    # Should predict based on velocity from history
    # Velocity: (155-150)/0.1 = 50, (155-155)/0.1 = 0
    # Prediction: 155 + 50*0.25 = 167.5, 155 + 0*0.25 = 155
    assert pred_pos == (167, 155)
