"""Tests for core.dashboard_logic module."""

import pytest
import json
import time
from pathlib import Path
from datetime import datetime
from collections import deque

from core.dashboard_logic import (
    BotLiveData,
    DashboardDataBridge,
    ReplayFrame,
    ReplayRecorder,
    ABTestVariant,
    ABTestManager,
    BrawlerStatsTracker,
    MatchAnalyzer,
    TrophyTracker,
)


class TestBotLiveData:
    """Tests for BotLiveData dataclass."""

    def test_default_initialization(self):
        """BotLiveData should initialize with sensible defaults."""
        data = BotLiveData()
        assert data.timestamp == 0.0
        assert data.running is False
        assert data.current_state == "unknown"
        assert data.matches_total == 0
        assert data.wins == 0
        assert data.losses == 0
        assert data.win_rate == 0.0

    def test_field_mutation(self):
        """Fields should be mutable after creation."""
        data = BotLiveData()
        data.running = True
        data.current_state = "lobby"
        data.matches_total = 5
        assert data.running is True
        assert data.current_state == "lobby"
        assert data.matches_total == 5


class TestDashboardDataBridge:
    """Tests for DashboardDataBridge thread-safe data bridge."""

    def test_initialization(self):
        """Bridge should initialize with empty data and history."""
        bridge = DashboardDataBridge()
        snap = bridge.get_snapshot()
        assert snap["running"] is False
        assert snap["current_state"] == "unknown"

    def test_update_and_snapshot(self):
        """Update should reflect in snapshot."""
        bridge = DashboardDataBridge()
        bridge.update(running=True, current_state="in_game", matches_total=3)
        snap = bridge.get_snapshot()
        assert snap["running"] is True
        assert snap["current_state"] == "in_game"
        assert snap["matches_total"] == 3

    def test_history_tracking(self):
        """History should track multiple updates."""
        bridge = DashboardDataBridge()
        for i in range(5):
            bridge.update(matches_total=i)
        history = bridge.get_history()
        assert len(history) == 5

    def test_rewards_history(self):
        """Rewards history should track reward points."""
        bridge = DashboardDataBridge()
        bridge.add_reward_point(1.0)
        bridge.add_reward_point(-0.5)
        rewards = bridge.get_rewards_history()
        assert len(rewards) == 2
        assert rewards[0]["r"] == 1.0
        assert rewards[1]["r"] == -0.5


class TestReplayRecorder:
    """Tests for ReplayRecorder."""

    def test_initialization(self, tmp_path):
        """Recorder should initialize with save directory."""
        recorder = ReplayRecorder(save_dir=tmp_path / "replays")
        assert recorder.save_dir.exists()
        assert recorder._active is False

    def test_start_stop(self, tmp_path):
        """Start and stop should manage active state."""
        recorder = ReplayRecorder(save_dir=tmp_path / "replays")
        recorder.start(name="test_replay")
        assert recorder._active is True
        recorder.stop()
        assert recorder._active is False

    def test_list_replays_empty(self, tmp_path):
        """Empty recorder should return empty list."""
        recorder = ReplayRecorder(save_dir=tmp_path / "replays")
        assert recorder.list_replays() == []


class TestABTestManager:
    """Tests for ABTestManager."""

    def test_initialization(self, tmp_path):
        """Manager should initialize inactive with no variants."""
        mgr = ABTestManager(save_path=tmp_path / "ab_tests.json")
        assert mgr.active is False
        assert mgr.get_summary() == {}

    def test_define_variants(self, tmp_path):
        """Should accept variant definitions."""
        mgr = ABTestManager(save_path=tmp_path / "ab_tests.json")
        mgr.define_variants({
            "control": {"aggression": 0.5},
            "experimental": {"aggression": 0.8},
        })
        assert len(mgr.variants) == 2

    def test_start_test_insufficient_variants(self, tmp_path):
        """Should refuse to start with < 2 variants."""
        mgr = ABTestManager(save_path=tmp_path / "ab_tests.json")
        mgr.define_variants({"control": {}})
        mgr.start_test()
        assert mgr.active is False

    def test_round_robin_variant(self, tmp_path):
        """Should cycle through variants round-robin."""
        mgr = ABTestManager(save_path=tmp_path / "ab_tests.json")
        mgr.define_variants({"A": {}, "B": {}})
        mgr.start_test()
        v1 = mgr.next_match_variant()
        v2 = mgr.next_match_variant()
        v3 = mgr.next_match_variant()
        assert v1 == "A"
        assert v2 == "B"
        assert v3 == "A"

    def test_record_result(self, tmp_path):
        """Should record match results."""
        mgr = ABTestManager(save_path=tmp_path / "ab_tests.json")
        mgr.define_variants({"A": {}})
        mgr.record_result("A", "win", reward=1.0)
        summary = mgr.get_summary()
        assert summary["variants"]["A"]["wins"] == 1
        assert summary["variants"]["A"]["matches"] == 1


class TestBrawlerStatsTracker:
    """Tests for BrawlerStatsTracker."""

    def test_record_match_win(self, tmp_path):
        """Win should increment wins and trophies."""
        tracker = BrawlerStatsTracker(save_path=tmp_path / "stats.json")
        tracker.record_match("colt", "gem_grab", "win", kills=3, deaths=1)
        stats = tracker.get_all_stats()
        assert len(stats) == 1
        assert stats[0]["wins"] == 1
        assert stats[0]["trophies"] == 3

    def test_record_match_loss(self, tmp_path):
        """Loss should increment losses without negative trophies."""
        tracker = BrawlerStatsTracker(save_path=tmp_path / "stats.json")
        tracker.record_match("colt", "gem_grab", "loss", kills=1, deaths=3)
        stats = tracker.get_all_stats()
        assert stats[0]["losses"] == 1
        assert stats[0]["trophies"] == 0

    def test_total_trophies(self, tmp_path):
        """Total trophies should sum across brawlers."""
        tracker = BrawlerStatsTracker(save_path=tmp_path / "stats.json")
        tracker.record_match("colt", "map1", "win")
        tracker.record_match("shelly", "map2", "win")
        assert tracker.get_total_trophies() == 6


class TestMatchAnalyzer:
    """Tests for MatchAnalyzer."""

    def test_suggest_pick_empty(self):
        """Empty available list should still return a suggestion dict."""
        analyzer = MatchAnalyzer()
        result = analyzer.suggest_pick("gem_grab", [])
        # Method returns a dict even for empty list (fallback to default brawler)
        assert isinstance(result, dict)
        assert "brawler" in result

    def test_predict_win(self):
        """Predict win should return a float between 0 and 1."""
        analyzer = MatchAnalyzer()
        score = analyzer.predict_win("colt", "gem_grab")
        assert 0.0 <= score <= 1.0

    def test_get_coach_tips(self):
        """Should return list of tips."""
        analyzer = MatchAnalyzer()
        tips = analyzer.get_coach_tips("colt")
        assert isinstance(tips, list)


class TestTrophyTracker:
    """Tests for TrophyTracker."""

    def test_record_and_history(self, tmp_path):
        """Record should appear in history."""
        tracker = TrophyTracker(save_path=tmp_path / "trophies.json")
        tracker.record(100)
        history = tracker.get_trophy_history(days=30)
        assert len(history) >= 1
        assert history[0]["total_trophies"] == 100

    def test_daily_evolution(self, tmp_path):
        """Daily evolution should compute differences."""
        tracker = TrophyTracker(save_path=tmp_path / "trophies.json")
        tracker.record(100)
        tracker.record(110)
        evo = tracker.get_daily_evolution(days=30)
        assert len(evo) >= 1
