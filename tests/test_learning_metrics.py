"""
tests/test_learning_metrics.py

Tests for the LearningMetricsCollector and LearningMatchResult.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.learning_metrics import LearningMatchResult, LearningMetricsCollector


class TestLearningMatchResult:
    def test_defaults(self):
        r = LearningMatchResult(match_id="m1", brawler="colt", start_time="2026-01-01T00:00:00")
        assert r.kills == 0
        assert r.damage_dealt == 0.0
        assert r.result == "unknown"

    def test_to_dict(self):
        r = LearningMatchResult(match_id="m1", brawler="colt", start_time="t1", kills=3)
        d = r.to_dict()
        assert d["match_id"] == "m1"
        assert d["kills"] == 3
        assert "notes" in d


class TestLearningMetricsCollector:
    @pytest.fixture
    def collector(self, tmp_path: Path):
        return LearningMetricsCollector(output_dir=tmp_path)

    def test_start_match(self, collector):
        collector.start_match("shelly")
        assert collector.current_match is not None
        assert collector.current_match.brawler == "shelly"

    def test_log_frame(self, collector):
        collector.start_match("colt")
        collector.log_frame(enemies_detected=2, player_detected=True, action_taken="attack")
        assert collector.current_match.detections_enemies == 2
        assert collector.current_match.actions_attack == 1

    def test_log_kill_and_death(self, collector):
        collector.start_match("colt")
        collector.log_kill(2)
        collector.log_death()
        assert collector.current_match.kills == 2
        assert collector.current_match.deaths == 1
        assert collector.current_match.result == "died"

    def test_end_match(self, collector):
        collector.start_match("bull")
        collector.log_kill(1)
        result = collector.end_match(result="completed", duration=120.0)
        assert result.result == "completed"
        assert result.duration_seconds == 120.0
        assert result.kills == 1
        assert len(collector.matches) == 1
        assert collector.current_match is None

    def test_persistence(self, collector, tmp_path: Path):
        collector.start_match("dynamike")
        collector.end_match(result="timeout", duration=60.0)
        data = json.loads((tmp_path / "learning_sessions.json").read_text(encoding="utf-8"))
        assert data["total_sessions"] == 1
        assert data["matches"][0]["brawler"] == "dynamike"

    def test_summary_empty(self, collector):
        summary = collector.get_summary()
        assert summary["total_matches"] == 0

    def test_summary_with_matches(self, collector):
        collector.start_match("crow")
        collector.log_kill(3)
        collector.end_match(result="completed", duration=100.0)
        collector.start_match("crow")
        collector.log_kill(1)
        collector.log_death()
        collector.end_match(result="died", duration=50.0)
        summary = collector.get_summary()
        assert summary["total_matches"] == 2
        assert summary["total_kills"] == 4
        assert summary["total_deaths"] == 1
        assert summary["kdr"] == 4.0
        assert summary["completed"] == 1
        assert summary["deaths_by_died"] == 1
