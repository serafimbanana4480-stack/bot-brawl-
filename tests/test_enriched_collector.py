"""
tests/test_enriched_collector.py

Testes para o EnrichedGameplayCollector.
"""

from __future__ import annotations

import sys
from pathlib import Path


import numpy as np
import pytest

from dataset.enriched_collector import (
    EnrichedFrameRecord,
    EnrichedGameplayCollector,
)


@pytest.fixture
def black_screenshot():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def collector(tmp_path):
    return EnrichedGameplayCollector(
        base_dir=tmp_path,
        enable_multimodal=False,
        resolution=(1920, 1080),
    )


class TestEnrichedFrameRecord:
    def test_to_dict(self):
        r = EnrichedFrameRecord(
            timestamp=12345.0,
            frame_id=1,
            state="in_game",
            reward=10.0,
            vision_latency_ms=5.0,
        )
        d = r.to_dict()
        assert d["frame_id"] == 1
        assert d["state"] == "in_game"
        assert d["reward"] == 10.0
        assert d["vision_latency_ms"] == 5.0


class TestCollectorInit:
    def test_default(self, collector):
        assert collector.frames_collected == 0
        assert collector.episodes_collected == 0
        assert collector.enable_multimodal is False

    def test_base_dir_created(self, tmp_path):
        c = EnrichedGameplayCollector(base_dir=tmp_path / "test_dir")
        assert c.base_dir.exists()


class TestRecordFrame:
    def test_record_basic(self, collector, black_screenshot):
        dets = [{"class": 0, "bbox": [100, 100, 200, 200], "confidence": 0.9}]
        record = collector.record_frame(
            screenshot=black_screenshot,
            yolo_detections=dets,
            game_state_hint="in_game",
            frame_id=1,
        )
        assert record.frame_id == 1
        assert record.state == "in_game"
        assert collector.frames_collected == 1

    def test_record_with_action(self, collector, black_screenshot):
        record = collector.record_frame(
            screenshot=black_screenshot,
            yolo_detections=[],
            game_state_hint="lobby",
            action={"type": "tap", "x": 100, "y": 200},
            reward=5.0,
            frame_id=2,
        )
        assert record.action["type"] == "tap"
        assert record.reward == 5.0


class TestEpisode:
    def test_start_episode(self, collector):
        ep_id = collector.start_episode()
        assert ep_id.startswith("ep_")

    def test_end_episode(self, collector, black_screenshot, tmp_path):
        ep_id = collector.start_episode("test_ep")
        collector.record_frame(black_screenshot, [], "in_game", frame_id=1)
        collector.record_frame(black_screenshot, [], "in_game", frame_id=2)
        path = collector.end_episode(ep_id, result="win")
        assert path.exists()
        assert collector.episodes_collected == 1

    def test_end_episode_data(self, collector, black_screenshot, tmp_path):
        ep_id = collector.start_episode("data_test")
        collector.record_frame(black_screenshot, [], "in_game", frame_id=1)
        path = collector.end_episode(ep_id, result="loss", metrics={"kills": 3})
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["episode_id"] == "data_test"
        assert data["result"] == "loss"
        assert data["metrics"]["kills"] == 3
        assert data["frame_count"] == 1
        assert len(data["frames"]) == 1


class TestStats:
    def test_stats_initial(self, collector):
        stats = collector.get_stats()
        assert stats["frames_collected"] == 0
        assert stats["avg_vision_latency_ms"] == 0.0
        assert stats["multimodal_enabled"] is False

    def test_stats_after_frames(self, collector, black_screenshot):
        collector.record_frame(black_screenshot, [], "in_game", frame_id=1)
        collector.record_frame(black_screenshot, [], "in_game", frame_id=2)
        stats = collector.get_stats()
        assert stats["frames_collected"] == 2


class TestReset:
    def test_reset(self, collector, black_screenshot):
        collector.record_frame(black_screenshot, [], "in_game", frame_id=1)
        collector.reset()
        assert collector.frames_collected == 0
        assert collector.episodes_collected == 0


class TestPerformance:
    def test_record_frame_under_5ms(self, collector, black_screenshot):
        import time
        t0 = time.time()
        collector.record_frame(black_screenshot, [], "in_game", frame_id=1)
        elapsed = (time.time() - t0) * 1000
        assert elapsed < 5.0, f"record_frame demorou {elapsed:.1f}ms"
