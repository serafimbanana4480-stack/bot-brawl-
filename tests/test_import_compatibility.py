"""Compatibility tests for legacy import paths."""

from pathlib import Path
import sys




def test_brawl_bot_compatibility_imports():
    import brawl_bot.auto_tuner as auto_tuner
    import backend.brawl_bot.safety_system as safety_system

    assert hasattr(auto_tuner, "AutoTuner")
    assert hasattr(safety_system, "SafetySystem")
