import os
import sys
from pathlib import Path

import numpy as np

# Ensure the repository root is importable.
_this_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_this_dir, '..', '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from backend.brawl_bot.pylaai_real.state_finder import StateFinder
from backend.brawl_bot.pylaai_real.state_manager import StateManager


class DummyScreenshot:
    def take(self):
        return np.zeros((1080, 1920, 3), dtype=np.uint8)


class DummyFinder:
    def __init__(self, state="unknown"):
        self.state = state
        self.calls = []

    def get_state(self, image, screen_state_hint=None):
        self.calls.append(screen_state_hint)
        return self.state

    def get_diagnostic_report(self):
        return {"state": self.state, "reason": "test", "details": {}}


class DummyScreenAutomation:
    def __init__(self, state_name="loading"):
        self.state_name = state_name

    def get_current_state_name(self):
        return self.state_name


class DummyLobby:
    def __init__(self):
        self.press_play_calls = 0
        self.select_calls = 0

    def press_play(self):
        self.press_play_calls += 1
        return True

    def select_current_brawler(self, screenshot_func):
        self.select_calls += 1
        return True

    def quit_shop(self):
        return None

    def close_popup(self):
        return None


class DummyProgress:
    def __init__(self):
        self.calls = 0

    def find_game_result(self, screenshot):
        self.calls += 1
        return False

    def get_stats(self):
        return {}


class DummyPlay:
    def __init__(self):
        self.calls = 0

    def play_round(self, screenshot):
        self.calls += 1
        return False


class DummyEmulatorController:
    def __init__(self):
        self.taps = []
        self.keyevents = []

    def tap_scaled(self, x, y):
        self.taps.append((x, y))

    def keyevent(self, code):
        self.keyevents.append(code)


class SequenceFinder:
    def __init__(self, states):
        self.states = list(states)
        self.calls = []

    def get_state(self, image, screen_state_hint=None):
        self.calls.append(screen_state_hint)
        if self.states:
            return self.states.pop(0)
        return self.states[-1] if self.states else "unknown"

    def get_diagnostic_report(self):
        return {"state": self.states[0] if self.states else "unknown", "reason": "test", "details": {}}


class DummyScreenshotSequence:
    def __init__(self, frames):
        self.frames = list(frames)

    def take(self):
        if self.frames:
            return self.frames.pop(0)
        return np.zeros((1080, 1920, 3), dtype=np.uint8)


class DummyMatchController:
    def __init__(self):
        self.started = []
        self.ended = []
        self.is_in_match = True
        self.current_match = {"id": "test", "game_mode": "gem_grab", "brawler": "Colt"}

    def start_match(self, *args, **kwargs):
        self.started.append((args, kwargs))
        return True

    def end_match(self, result):
        self.ended.append(result)
        self.is_in_match = False
        self.current_match = None
        return True

    def reset_match(self):
        self.is_in_match = False
        self.current_match = None


class DummyEndProgress(DummyProgress):
    def __init__(self, last_result="win", find_result=False):
        super().__init__()
        self.last_result = last_result
        self.find_result = find_result

    def find_game_result(self, screenshot):
        self.calls += 1
        return self.find_result

    def get_last_result(self):
        return self.last_result

    def clear_last_result(self):
        self.last_result = None


def test_state_finder_maps_explicit_screen_hints_to_non_unknown_states():
    finder = StateFinder(Path(_repo_root) / "missing-images")
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    assert finder.get_state(image, screen_state_hint="loading") == "loading"
    assert finder.get_state(image, screen_state_hint="detecting") == "matchmaking"
    # play_again maps to 'end' which is handled by the screen_automation_end path
    assert finder.get_state(image, screen_state_hint="play_again") == "end"
    # The last call used play_again→end, which uses the screen_automation_end reason
    assert finder.get_diagnostic_report()["reason"] == "screen_automation_end"
    # Verify non-end hints use the fallback path
    finder.get_state(image, screen_state_hint="loading")
    assert finder.get_diagnostic_report()["reason"] == "screen_state_hint_fallback"


def test_unknown_handler_is_conservative_and_does_not_invoke_play_logic():
    finder = DummyFinder()
    progress = DummyProgress()
    play = DummyPlay()
    manager = StateManager(
        screenshot_taker=DummyScreenshot(),
        state_finder=finder,
        lobby_automator=DummyLobby(),
        progress_observer=progress,
        play_logic=play,
        screen_automation=DummyScreenAutomation(),
        diagnostic_mode=False,
    )

    manager._handle_unknown(np.zeros((1080, 1920, 3), dtype=np.uint8))

    assert play.calls == 0
    assert progress.calls == 0


def test_state_manager_holds_unknown_for_a_few_cycles_after_connection_lost():
    finder = SequenceFinder(["connection_lost", "unknown", "unknown", "unknown"])
    progress = DummyProgress()
    play = DummyPlay()
    manager = StateManager(
        screenshot_taker=DummyScreenshot(),
        state_finder=finder,
        lobby_automator=DummyLobby(),
        progress_observer=progress,
        play_logic=play,
        emulator_controller=DummyEmulatorController(),
        screen_automation=DummyScreenAutomation(),
        diagnostic_mode=False,
    )

    state_1 = manager._process_cycle()
    state_2 = manager._process_cycle()
    state_3 = manager._process_cycle()
    state_4 = manager._process_cycle()

    assert state_1 == "connection_lost"
    assert state_2 == "connection_lost"
    assert state_3 == "connection_lost"
    assert state_4 == "unknown"
    assert manager.last_known_state == "connection_lost"
    assert manager.unknown_streak == 3


def test_state_manager_keeps_brawler_selection_to_lobby_flow_stable():
    finder = SequenceFinder([
        "brawler_selection",
        "lobby",
        "lobby",
    ])
    progress = DummyProgress()
    play = DummyPlay()
    lobby = DummyLobby()
    manager = StateManager(
        screenshot_taker=DummyScreenshotSequence([
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            np.zeros((1080, 1920, 3), dtype=np.uint8),
        ]),
        state_finder=finder,
        lobby_automator=lobby,
        progress_observer=progress,
        play_logic=play,
        screen_automation=DummyScreenAutomation(),
        diagnostic_mode=False,
    )

    first_state = manager._process_cycle()
    second_state = manager._process_cycle()

    assert first_state == "brawler_selection"
    assert second_state == "lobby"
    assert lobby.select_calls == 1
    assert lobby.press_play_calls == 2


def test_end_game_uses_last_result_when_ocr_fails_but_hint_still_confirms_end():
    finder = StateFinder(Path(_repo_root) / "missing-images")
    progress = DummyEndProgress(last_result="win", find_result=False)
    match_controller = DummyMatchController()
    manager = StateManager(
        screenshot_taker=DummyScreenshot(),
        state_finder=finder,
        lobby_automator=DummyLobby(),
        progress_observer=progress,
        play_logic=DummyPlay(),
        match_controller=match_controller,
        screen_automation=DummyScreenAutomation("play_again"),
        diagnostic_mode=False,
    )

    manager._handle_end_game()

    assert match_controller.ended == ["win"]
