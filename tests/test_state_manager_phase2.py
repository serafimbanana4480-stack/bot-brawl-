import os
import sys

# Ensure the repository root is importable.
_this_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_this_dir, '..', '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from backend.brawl_bot.pylaai_real.state_manager import StateManager


class DummyScreenshot:
    def take(self):
        return object()


class DummyFinder:
    def __init__(self):
        self.calls = []

    def get_state(self, image, screen_state_hint=None):
        self.calls.append(screen_state_hint)
        return "unknown"


class DummyScreenAutomation:
    def get_current_state_name(self):
        return "play"


class DummyLobby:
    def press_play(self):
        return True

    def select_current_brawler(self, screenshot_func):
        return True

    def quit_shop(self):
        return None

    def close_popup(self):
        return None


class DummyProgress:
    def find_game_result(self, screenshot):
        return False

    def get_stats(self):
        return {}


class DummyPlay:
    def play_round(self, screenshot):
        return False


def test_state_manager_passes_screen_hint_to_state_finder():
    finder = DummyFinder()
    manager = StateManager(
        screenshot_taker=DummyScreenshot(),
        state_finder=finder,
        lobby_automator=DummyLobby(),
        progress_observer=DummyProgress(),
        play_logic=DummyPlay(),
        screen_automation=DummyScreenAutomation(),
        diagnostic_mode=False,
    )

    assert manager._wait_for_state("lobby", timeout=0.01, poll_interval=0.0) is False
    assert finder.calls
    assert finder.calls[0] == "play"
