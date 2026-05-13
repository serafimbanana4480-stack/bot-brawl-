"""
Testes para dataset/collector.py
"""
import sys
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from dataset.collector import GameplayCollector, FrameRecord, EpisodeRecord


class TestDatasetCollector:
    def test_initialization(self, tmp_path):
        gc = GameplayCollector(base_dir=tmp_path)
        assert gc.base_dir == tmp_path
        assert not gc.is_collecting

    def test_start_end_session(self, tmp_path):
        gc = GameplayCollector(base_dir=tmp_path)
        gc.start_episode(brawler_name="shelly", map_name="test_map")
        assert gc.is_collecting
        assert gc.current_episode is not None
        assert gc.current_episode.episode_id == "episode_0001"
        episode = gc.current_episode
        gc.end_episode(result="win", metrics={"kills": 3})
        assert not gc.is_collecting
        assert episode.result == "win"

    def test_record_frame(self, tmp_path):
        gc = GameplayCollector(base_dir=tmp_path)
        gc.start_episode()
        gc.log_frame(
            screenshot=None,
            state="in_game",
            action={"type": "attack"},
            reward=1.5,
        )
        assert len(gc.current_episode.frames) == 1
        frame = gc.current_episode.frames[0]
        assert frame.state == "in_game"
        assert frame.action == {"type": "attack"}
        assert frame.reward == 1.5

    def test_rotated_path(self, tmp_path):
        gc = GameplayCollector(base_dir=tmp_path, max_file_size_mb=0.001)
        gc.start_episode()
        for i in range(100):
            gc.log_frame(state="in_game", action={"type": "attack"})
        gc.end_episode("win")
        # end_episode already exports; check bc dir
        bc_file = tmp_path / "bc" / "episodes.jsonl"
        assert bc_file.exists()

    def test_export_yolo(self, tmp_path):
        import numpy as np
        gc = GameplayCollector(base_dir=tmp_path)
        gc.start_episode()
        gc.log_frame(
            screenshot=np.zeros((1080, 1920, 3), dtype=np.uint8),
            state="in_game",
            detections={"enemy": [[100, 200, 150, 250]]},
        )
        gc.end_episode("win")
        # Screenshot should be saved under screenshots_dir
        assert any((tmp_path / "screenshots").rglob("*.jpg"))
