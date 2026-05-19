"""
Tests for behavioral biometrics integration.

Verifies that record_tap() and record_swipe() are called in EmulatorController.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


def test_emulator_controller_records_tap():
    """Verifies that tap_scaled() calls record_tap() when safety_system is available."""
    from brawl_bot.emulator_controller import EmulatorController, EmulatorConfig

    # Create a mock safety system
    safety_system = Mock()
    safety_system.record_tap = Mock()

    # Create emulator controller with safety system
    config = EmulatorConfig(
        window_title="Test Emulator",
        adb_path=None,
        window_width=1280,
        window_height=720
    )
    controller = EmulatorController(config, safety_system=safety_system)

    # Mock the ADB tap to return success
    controller.adb = Mock()
    controller.adb.tap = Mock(return_value=True)

    # Call tap_scaled
    controller.tap_scaled(100, 200)

    # Verify record_tap was called
    safety_system.record_tap.assert_called_once_with(100, 200)


def test_emulator_controller_records_swipe():
    """Verifies that swipe_scaled() calls record_swipe() when safety_system is available."""
    from brawl_bot.emulator_controller import EmulatorController, EmulatorConfig

    # Create a mock safety system
    safety_system = Mock()
    safety_system.record_swipe = Mock()

    # Create emulator controller with safety system
    config = EmulatorConfig(
        window_title="Test Emulator",
        adb_path=None,
        window_width=1280,
        window_height=720
    )
    controller = EmulatorController(config, safety_system=safety_system)

    # Mock the ADB swipe to return success
    controller.adb = Mock()
    controller.adb.swipe = Mock(return_value=True)

    # Call swipe_scaled
    controller.swipe_scaled(100, 200, 300, 400, duration=300)

    # Verify record_swipe was called with real coordinates
    safety_system.record_swipe.assert_called_once()
    call_args = safety_system.record_swipe.call_args
    assert call_args[0][0] == 100  # x1
    assert call_args[0][1] == 200  # y1
    assert call_args[0][2] == 300  # x2
    assert call_args[0][3] == 400  # y2
    assert call_args[0][4] == pytest.approx(0.3, abs=0.03)  # duration in seconds (±30ms jitter)


def test_emulator_controller_no_safety_system():
    """Verifies that tap_scaled() and swipe_scaled() work without safety_system."""
    from brawl_bot.emulator_controller import EmulatorController, EmulatorConfig

    # Create emulator controller without safety system
    config = EmulatorConfig(
        window_title="Test Emulator",
        adb_path=None,
        window_width=1280,
        window_height=720
    )
    controller = EmulatorController(config, safety_system=None)

    # Mock the ADB tap to return success
    controller.adb = Mock()
    controller.adb.tap = Mock(return_value=True)
    controller.adb.swipe = Mock(return_value=True)

    # Call tap_scaled - should not crash
    controller.tap_scaled(100, 200)

    # Call swipe_scaled - should not crash
    controller.swipe_scaled(100, 200, 300, 400, duration=300)

    # Verify ADB was still called
    controller.adb.tap.assert_called_once()
    controller.adb.swipe.assert_called_once()


def test_emulator_controller_uses_humanization_delays():
    """Verifies that EmulatorController consults the humanization engine when available."""
    from brawl_bot.emulator_controller import EmulatorController, EmulatorConfig

    humanization = Mock()
    humanization.config.enabled = True
    humanization.get_tremor = Mock(return_value=(2.2, -1.4))
    humanization.get_delay = Mock(return_value=0.01)

    config = EmulatorConfig(
        window_title="Test Emulator",
        adb_path=None,
        window_width=1280,
        window_height=720
    )
    controller = EmulatorController(config, safety_system=None, humanization_system=humanization)

    controller.adb = Mock()
    controller.adb.tap = Mock(return_value=True)
    controller.adb.swipe = Mock(return_value=True)

    controller.tap_scaled(100, 200)
    controller.swipe_scaled(100, 200, 300, 400, duration=300)

    assert humanization.get_tremor.called
    assert humanization.get_delay.call_count >= 2


def test_safety_system_receives_tap_data():
    """Verifies that safety_system receives correct tap data."""
    from brawl_bot.safety_system import SafetySystem, SafetyConfig

    # Create safety system
    config = SafetyConfig()
    safety_system = SafetySystem(config)
    safety_system.start_session()

    # Record a tap
    safety_system.record_tap(100, 200)

    # Verify tap was recorded
    assert len(safety_system.movement_analyzer.taps) > 0


def test_safety_system_receives_swipe_data():
    """Verifies that safety_system receives correct swipe data."""
    from brawl_bot.safety_system import SafetySystem, SafetyConfig

    # Create safety system
    config = SafetyConfig()
    safety_system = SafetySystem(config)
    safety_system.start_session()

    # Record a swipe
    safety_system.record_swipe(100, 200, 300, 400, 0.3)

    # Verify swipe was recorded
    assert len(safety_system.movement_analyzer.swipes) > 0
