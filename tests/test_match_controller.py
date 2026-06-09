"""
Testes para match_controller.py
"""
import sys
from pathlib import Path
import tempfile


import pytest
from match_controller import MatchController, MatchResult, MatchHistory


class TestMatchController:
    def test_start_end_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MatchController(Path(tmpdir))
            assert mc.start_match("Island", "Colt")
            assert mc.current_match is not None
            res = mc.end_match("win", kills=3, damage=1000)
            assert res is not None
            assert res.result == "win"
            assert mc.current_match is None

    def test_history_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MatchController(Path(tmpdir))
            mc.start_match("Island", "Colt")
            mc.end_match("win", kills=3, damage=1000)
            mc.start_match("Island", "Colt")
            mc.end_match("loss", kills=1, damage=500)
            stats = mc.history.get_stats(last_n=10)
            assert stats["total"] == 2
            assert stats["wins"] == 1
            assert stats["losses"] == 1

    def test_limit_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MatchController(Path(tmpdir))
            for i in range(1005):
                mc.start_match("Island", "Colt")
                mc.end_match("win", kills=0, damage=0)
            assert len(mc.history.matches) <= 1000

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MatchController(Path(tmpdir))
            mc.start_match("Island", "Colt")
            mc.end_match("win", kills=3, damage=1000)
            # Recriar e carregar
            mc2 = MatchController(Path(tmpdir))
            assert len(mc2.history.matches) == 1
            assert mc2.history.matches[0].brawler == "Colt"

    def test_get_session_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MatchController(Path(tmpdir))
            info = mc.get_session_info()
            assert "total_trophies" in info
            assert "session_matches" in info
            assert "is_in_match" in info
