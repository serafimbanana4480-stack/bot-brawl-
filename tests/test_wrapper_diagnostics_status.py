import os
import sys
import time

# Ensure repository root is importable.
_this_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_this_dir, '..', '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from backend.brawl_bot.wrapper import PylaAIEnhanced
from backend.brawl_bot.pylaai_real.lobby_automator import BrawlerQueue, BrawlerConfig
from backend.brawl_bot.diagnostic_overlay import DiagnosticOverlay


class DummySafety:
    def get_status(self):
        return {"safe": True}


class DummyProgress:
    def get_stats(self):
        return {"wins": 2, "losses": 1}


class DummyScreenAutomation:
    def get_current_state_name(self):
        return "play"


class DummyWindowController:
    def get_status_snapshot(self):
        return {"window_active": True, "window_title": "BlueStacks App Player"}


class DummyPlayLogic:
    def get_last_combat_snapshot(self):
        return {
            "state": "combat_ok",
            "enemies": 2,
            "move_key": "W",
            "attack_taken": True,
            "super_taken": False,
            "target_position": (120, 240),
        }


class DummyLobby:
    def __init__(self):
        self._diag = {"flow": "lobby", "step": "press_play_done", "details": {}, "error": None, "updated_at": 123.0}

    def get_diagnostic_report(self):
        return self._diag


def test_get_status_exposes_diagnostics_snapshot():
    import threading
    bot = PylaAIEnhanced.__new__(PylaAIEnhanced)
    bot._running_lock = threading.Lock()
    bot.running = True
    bot.state_manager = type("StateManagerStub", (), {"current_state": "lobby", "screen_automation": DummyScreenAutomation()})()
    bot.safety = DummySafety()
    bot.progress = DummyProgress()
    bot.lobby = DummyLobby()
    bot.session_start = time.time() - 120
    bot.matches_played = 3
    bot.emulator_controller = DummyWindowController()
    bot.detect_main = object()
    bot.play_logic = DummyPlayLogic()
    bot.diagnostic_mode = True
    bot.error_recovery = None
    bot.enable_error_recovery = False
    bot.auto_calibrator = None
    bot.state_recovery = None
    bot.ocr_detector = None
    bot.debug_visualizer = None
    bot.rl_engine = None
    bot.dashboard = None

    queue = BrawlerQueue()
    queue.add_brawler(BrawlerConfig(name="Colt"))
    bot.brawler_queue = queue

    status = bot.get_status()

    assert status["running"] is True
    assert status["current_brawler"] == "Colt"
    assert status["window_active"] is True
    assert status["window_title"] == "BlueStacks App Player"
    assert status["diagnostics"]["diagnostic_mode"] is True
    assert status["diagnostics"]["lobby"]["step"] == "press_play_done"
    assert status["diagnostics"]["screen_state"] == "play"
    assert status["diagnostics"]["progress"] == {"wins": 2, "losses": 1}
    assert status["diagnostics"]["combat"]["state"] == "combat_ok"


def test_diagnostic_overlay_formats_key_state_fields():
    lines = DiagnosticOverlay.format_status({
        "current_state": "lobby",
        "last_known_state": "lobby",
        "unknown_streak": 1,
        "last_unknown_hint": "play",
        "current_brawler": "Colt",
        "matches_played": 4,
        "session_duration_minutes": 12.5,
        "window_active": True,
        "window_title": "BlueStacks App Player",
        "diagnostics": {
            "lobby": {"step": "press_play_done"},
            "screen_state": "play",
            "progress": {"total_games": 7},
            "match": {"active": True},
            "combat": {
                "state": "combat_ok",
                "enemies": 2,
                "move_key": "W",
                "attack_taken": True,
                "super_taken": False,
                "target_position": (120, 240),
            },
        },
    })

    assert "State: lobby" in lines[0]
    assert any("Unknown streak: 1" in line for line in lines)
    assert any("Unknown hint: play" in line for line in lines)
    assert any("Brawler: Colt" in line for line in lines)
    assert any("Window active: True" in line for line in lines)
    assert any("Combat state: combat_ok" in line for line in lines)
    assert any("Enemies: 2" in line for line in lines)
    assert any("Lobby: press_play_done" in line for line in lines)
