import os
import sys
from pathlib import Path

import numpy as np

# Ensure the repository root is importable.
_this_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_this_dir, '..', '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from backend.brawl_bot.match_controller import MatchController, MatchResult
from backend.brawl_bot.pylaai_real.lobby_automator import BrawlerQueue, BrawlerConfig
from backend.brawl_bot.pylaai_real.state_manager import StateManager


class DummyScreenshot:
    def take(self):
        return np.zeros((1080, 1920, 3), dtype=np.uint8)


class DummyFinder:
    def __init__(self, state="end"):
        self.state = state

    def get_state(self, image, screen_state_hint=None):
        return self.state


class DummyLobby:
    def __init__(self, queue):
        self.queue = queue

    def press_play(self):
        return True

    def select_current_brawler(self, screenshot_func):
        return True

    def quit_shop(self):
        return None

    def close_popup(self):
        return None


class DummyProgress:
    def __init__(self, result="win"):
        self.result = result
        self.find_calls = 0

    def find_game_result(self, screenshot):
        self.find_calls += 1
        return True

    def get_last_result(self):
        return self.result

    def get_stats(self):
        return {
            "trophies": 8 if self.result == "win" else -6,
            "wins": 1 if self.result == "win" else 0,
            "losses": 1 if self.result == "loss" else 0,
            "draws": 1 if self.result == "draw" else 0,
        }


class DummyPlay:
    def play_round(self, screenshot):
        return False


def _make_queue():
    queue = BrawlerQueue()
    queue.add_brawler(BrawlerConfig(name="Colt", current_trophies=0, target_trophies=10, current_wins=0, target_wins=2, priority=2))
    queue.add_brawler(BrawlerConfig(name="Shelly", current_trophies=0, target_trophies=10, current_wins=0, target_wins=2, priority=1))
    return queue


def test_match_controller_rejects_duplicate_start_and_allows_restart_after_end(tmp_path):
    controller = MatchController(tmp_path)

    assert controller.start_match("gem_grab", "Colt") is True
    assert controller.start_match("gem_grab", "Colt") is False

    result = controller.end_match("win")
    assert result is not None
    assert result.result == "win"
    assert controller.is_in_match is False
    assert controller.current_match is None

    assert controller.start_match("brawl_ball", "Shelly") is True


def test_match_controller_advances_shared_queue_after_loss_streak(tmp_path):
    controller = MatchController(tmp_path)
    queue = _make_queue()
    controller.brawler_queue = queue

    controller.history.matches = [
        MatchResult(
            match_id=f"m{i}",
            timestamp="2026-05-06T00:00:00",
            game_mode="gem_grab",
            brawler="Colt",
            result="loss",
            trophies_change=-6,
            duration_seconds=60.0,
            kills=0,
            damage_dealt=0,
            powerups_collected=0,
            star_player=False,
        )
        for i in range(3)
    ]

    assert controller.start_match("gem_grab", "Colt") is True
    result = controller.end_match("loss")

    assert result is not None
    assert result.result == "loss"
    assert queue.get_current().name == "Shelly"
    assert controller.is_in_match is False
    assert controller.current_match is None


def test_state_manager_finalizes_confirmed_result_with_match_controller(tmp_path, monkeypatch):
    queue = _make_queue()
    controller = MatchController(tmp_path)
    controller.brawler_queue = queue
    controller.start_match("unknown", queue.get_current().name)

    progress = DummyProgress(result="win")
    manager = StateManager(
        screenshot_taker=DummyScreenshot(),
        state_finder=DummyFinder(),
        lobby_automator=DummyLobby(queue),
        progress_observer=progress,
        play_logic=DummyPlay(),
        match_controller=controller,
        diagnostic_mode=False,
    )

    monkeypatch.setattr("backend.brawl_bot.pylaai_real.state_manager.time.sleep", lambda *_args, **_kwargs: None)

    manager._handle_end_game()

    assert progress.find_calls == 1
    assert controller.is_in_match is False
    assert controller.history.matches[-1].result == "win"
    assert queue.get_current().current_wins == 1
    assert queue.get_current().current_trophies == 8
