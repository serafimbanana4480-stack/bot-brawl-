import os
import sys
from pathlib import Path

# Ensure the repository root is importable.
_this_dir = os.path.dirname(__file__)

from backend.brawl_bot.pylaai_real.lobby_automator import LobbyAutomator, BrawlerQueue, BrawlerConfig


class DummyScreenAutomation:
    def __init__(self):
        self.bot_stopped = True

    def set_bot_stopped(self, stopped):
        self.bot_stopped = stopped


class DummyController:
    def __init__(self):
        self.taps = []

    def tap_scaled(self, x, y):
        self.taps.append((x, y))


def test_press_play_records_diagnostic_steps(monkeypatch):
    queue = BrawlerQueue()
    queue.add_brawler(BrawlerConfig(name="Colt"))

    lobby = LobbyAutomator(
        queue=queue,
        emulator_controller=DummyController(),
        screen_automation=DummyScreenAutomation(),
        diagnostic_mode=True,
    )

    # Avoid real waiting during the test.
    monkeypatch.setattr("backend.brawl_bot.pylaai_real.lobby_automator.time.sleep", lambda *_args, **_kwargs: None)

    result = lobby.press_play()

    assert result is True
    report = lobby.get_diagnostic_report()
    assert report["flow"] == "lobby"
    # After rewrite, press_play uses visual verification and dynamic coords.
    # The step name changed from "press_play_done" to "press_play_state_changed"
    # or "press_play_verified" depending on the detection method.
    assert report["step"] in ("press_play_state_changed", "press_play_verified",
                               "press_play_after_clear", "press_play_fallback")
    assert report["error"] is None
    assert isinstance(report["updated_at"], float)

    # Ensure a play tap happened (coordinates are now dynamic, not hardcoded 960,950).
    assert len(lobby.emulator_controller.taps) > 0
