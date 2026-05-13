"""
Tests for map detection integration.

Verifies that map detection via screen automation hints works correctly.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch


def test_state_finder_detects_map_from_hint():
    """Verifies that StateFinder extracts map name from screen automation hint."""
    from brawl_bot.pylai_real.state_finder import StateFinder
    from pathlib import Path

    # Create a mock images path
    images_path = Path("/tmp/test_images")

    # Create state finder
    state_finder = StateFinder(images_path)

    # Test map detection from hint
    state, map_name = state_finder._state_from_hint("island invasion")

    # Verify state is detected
    assert state == "matchmaking"
    # Verify map name is extracted
    assert map_name == "Island Invasion"


def test_state_finder_detects_canyon_crossing():
    """Verifies that StateFinder detects Canyon Crossing map."""
    from brawl_bot.pylai_real.state_finder import StateFinder
    from pathlib import Path

    # Create a mock images path
    images_path = Path("/tmp/test_images")

    # Create state finder
    state_finder = StateFinder(images_path)

    # Test map detection from hint
    state, map_name = state_finder._state_from_hint("canyon crossing")

    # Verify state is detected
    assert state == "matchmaking"
    # Verify map name is extracted
    assert map_name == "Canyon Crossing"


def test_state_finder_no_map_in_hint():
    """Verifies that StateFinder returns None when no map is in hint."""
    from brawl_bot.pylai_real.state_finder import StateFinder
    from pathlib import Path

    # Create a mock images path
    images_path = Path("/tmp/test_images")

    # Create state finder
    state_finder = StateFinder(images_path)

    # Test hint without map keyword
    state, map_name = state_finder._state_from_hint("loading")

    # Verify state is detected
    assert state == "loading"
    # Verify map name is None
    assert map_name is None


def test_state_manager_sets_map_on_detection():
    """Verifies that StateManager calls set_current_map() when map is detected."""
    from brawl_bot.pylai_real.state_manager import StateManager
    from unittest.mock import Mock

    # Create mock movement
    movement = Mock()
    movement.set_current_map = Mock()

    # Create mock other dependencies
    screenshot_taker = Mock()
    state_finder = Mock()
    lobby = Mock()
    progress_observer = Mock()
    play_logic = Mock()
    match_controller = Mock()
    emulator_controller = Mock()
    screen_automation = Mock()

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

    # Simulate map detection in _process_cycle
    # Mock state_finder to return (state, map_name) tuple
    state_finder.get_state = Mock(return_value=("matchmaking", "Island Invasion"))

    # Process a cycle
    state_manager._process_cycle()

    # Verify set_current_map was called
    movement.set_current_map.assert_called_once_with("Island Invasion")


def test_movement_loads_map_strategy():
    """Verifies that Movement loads strategy when map is set."""
    from brawl_bot.pylai_real.movement import Movement
    from unittest.mock import patch, Mock

    # Mock the config loading
    with patch("brawl_bot.pylai_real.movement.toml.load") as mock_toml:
        mock_toml.return_value = {
            "maps": {
                "island_invasion": {
                    "strategy": "aggressive",
                    "priority": "cover"
                },
                "default": {
                    "strategy": "balanced"
                }
            }
        }

        # Create movement
        movement = Movement()

        # Set current map
        movement.set_current_map("Island Invasion")

        # Verify strategy is loaded
        assert movement.current_map == "Island Invasion"
        assert movement.map_strategy is not None
        assert movement.map_strategy.get("strategy") == "aggressive"


def test_movement_fallback_to_default_strategy():
    """Verifies that Movement falls back to default strategy for unknown maps."""
    from brawl_bot.pylai_real.movement import Movement
    from unittest.mock import patch

    # Mock the config loading
    with patch("brawl_bot.pylai_real.movement.toml.load") as mock_toml:
        mock_toml.return_value = {
            "maps": {
                "island_invasion": {
                    "strategy": "aggressive"
                },
                "default": {
                    "strategy": "balanced"
                }
            }
        }

        # Create movement
        movement = Movement()

        # Set unknown map
        movement.set_current_map("Unknown Map")

        # Verify fallback to default strategy
        assert movement.map_strategy is not None
        assert movement.map_strategy.get("strategy") == "balanced"
