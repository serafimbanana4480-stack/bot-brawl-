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


def test_state_finder_uses_screen_hint_as_safe_lobby_fallback():
    finder = StateFinder(Path(_repo_root) / "missing-images")
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    state_without_hint = finder.get_state(image)
    report_without_hint = finder.get_diagnostic_report()

    state_with_hint = finder.get_state(image, screen_state_hint="play")
    report_with_hint = finder.get_diagnostic_report()

    assert state_without_hint == "unknown"
    assert report_without_hint["reason"] == "no_template_match"
    assert state_with_hint == "lobby"
    assert report_with_hint["reason"] == "screen_state_hint_fallback"
    assert report_with_hint["details"]["screen_state_hint"] == "play"


def test_state_finder_uses_normalized_hint_for_empty_screens():
    finder = StateFinder(Path(_repo_root) / "missing-images")

    state = finder.get_state(np.array([]), screen_state_hint="Play Again")
    report = finder.get_diagnostic_report()

    assert state == "end"
    assert report["reason"] == "screen_state_hint_fallback"
    assert report["details"]["screen_state_hint"] == "Play Again"
