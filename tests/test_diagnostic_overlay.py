"""
Testes para diagnostic_overlay.py
"""
import sys
from pathlib import Path


import pytest
from diagnostic_overlay import DiagnosticOverlay


class TestDiagnosticOverlay:
    def test_format_status_basic(self):
        lines = DiagnosticOverlay.format_status({
            "current_state": "lobby",
            "matches_played": 5,
            "session_duration_minutes": 12.5,
        })
        assert any("State: lobby" in line for line in lines)
        assert any("Matches: 5" in line for line in lines)
        assert any("Session: 12.5" in line for line in lines)

    def test_format_status_with_combat(self):
        lines = DiagnosticOverlay.format_status({
            "current_state": "in_game",
            "diagnostics": {
                "combat": {
                    "state": "combat_ok",
                    "enemies": 2,
                    "move_key": "W",
                    "attack_taken": True,
                },
            },
        })
        assert any("Combat state: combat_ok" in line for line in lines)
        assert any("Enemies: 2" in line for line in lines)

    def test_format_status_empty(self):
        lines = DiagnosticOverlay.format_status({})
        assert isinstance(lines, list)
        assert len(lines) > 0
