"""
tests/test_multimodal_pipeline.py

Testes para o MultimodalPipeline e GameState.
"""

from __future__ import annotations

import sys
from pathlib import Path


import numpy as np
import pytest

from vision.game_state import GameState, HudValues, PlayerStatus, YoloDetection
from vision.multimodal_pipeline import MultimodalPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def black_screenshot():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def pipeline_disabled():
    """Pipeline com tudo desativado para testes de estrutura."""
    return MultimodalPipeline(
        resolution=(1920, 1080),
        enable_ocr=False,
        enable_player_state=False,
        enable_hud=False,
    )


@pytest.fixture
def pipeline():
    return MultimodalPipeline(
        resolution=(1920, 1080),
        enable_ocr=False,
        enable_player_state=False,
        enable_hud=False,
    )


@pytest.fixture
def sample_detections():
    return [
        {"class": 0, "bbox": [900, 500, 1000, 600], "confidence": 0.95},
        {"class": 1, "bbox": [1100, 600, 1200, 700], "confidence": 0.90},
        {"class": 3, "bbox": [500, 300, 550, 350], "confidence": 0.85},
    ]


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

class TestGameState:
    def test_default(self):
        gs = GameState()
        assert gs.game_state == "unknown"
        assert gs.is_in_game is False
        assert gs.can_act is False
        assert gs.enemy_count == 0

    def test_is_in_game(self):
        gs = GameState(game_state="in_game")
        assert gs.is_in_game is True

    def test_can_act_alive(self):
        gs = GameState(
            game_state="in_game",
            player=PlayerStatus(life="alive"),
        )
        assert gs.can_act is True

    def test_can_act_dead(self):
        gs = GameState(
            game_state="in_game",
            player=PlayerStatus(life="dead"),
        )
        assert gs.can_act is False

    def test_to_dict_serializable(self):
        gs = GameState(
            frame_id=1,
            game_state="in_game",
            detections=[YoloDetection(class_id=0, class_name="player", bbox=(0.1, 0.2, 0.3, 0.4), confidence=0.9)],
            player=PlayerStatus(life="alive", hp=0.8),
            hud=HudValues(hp=0.8, hp_confidence=0.9),
        )
        d = gs.to_dict()
        assert d["frame_id"] == 1
        assert d["game_state"] == "in_game"
        assert d["player"]["life"] == "alive"
        assert d["hud"]["hp"] == 0.8
        assert len(d["detections"]) == 1

    def test_enemy_count(self):
        gs = GameState(
            enemy_detections=[
                YoloDetection(class_id=1, class_name="enemy", bbox=(0, 0, 0, 0), confidence=0.9),
                YoloDetection(class_id=1, class_name="enemy", bbox=(0, 0, 0, 0), confidence=0.8),
            ],
        )
        assert gs.enemy_count == 2

    def test_is_super_ready(self):
        gs = GameState(player=PlayerStatus(super_ready=True))
        assert gs.is_super_ready is True

        gs2 = GameState(player=PlayerStatus(super_charge=1.0))
        assert gs2.is_super_ready is True

        gs3 = GameState(player=PlayerStatus(super_charge=0.5))
        assert gs3.is_super_ready is False

    def test_has_ammo(self):
        assert GameState(player=PlayerStatus(ammo=2)).has_ammo is True
        assert GameState(player=PlayerStatus(ammo=0)).has_ammo is False
        assert GameState(player=PlayerStatus(ammo=-1)).has_ammo is True  # unknown = assume ok


# ---------------------------------------------------------------------------
# YoloDetection
# ---------------------------------------------------------------------------

class TestYoloDetection:
    def test_to_dict(self):
        yd = YoloDetection(
            class_id=0,
            class_name="player",
            bbox=(0.1, 0.2, 0.3, 0.4),
            confidence=0.95,
            center=(0.2, 0.3),
        )
        d = yd.to_dict()
        assert d["class_id"] == 0
        assert d["class_name"] == "player"
        assert d["confidence"] == 0.95


# ---------------------------------------------------------------------------
# HudValues
# ---------------------------------------------------------------------------

class TestHudValues:
    def test_default_none(self):
        hud = HudValues()
        assert hud.hp is None
        assert hud.ammo is None

    def test_to_dict(self):
        hud = HudValues(hp=0.8, ammo=2, super_charge=0.5)
        d = hud.to_dict()
        assert d["hp"] == 0.8
        assert d["ammo"] == 2
        assert d["super_charge"] == 0.5


# ---------------------------------------------------------------------------
# PlayerStatus
# ---------------------------------------------------------------------------

class TestPlayerStatus:
    def test_default(self):
        ps = PlayerStatus()
        assert ps.life == "unknown"
        assert ps.super_ready is False

    def test_to_dict(self):
        ps = PlayerStatus(life="alive", hp=0.7, super_ready=True)
        d = ps.to_dict()
        assert d["life"] == "alive"
        assert d["hp"] == 0.7
        assert d["super_ready"] is True


# ---------------------------------------------------------------------------
# MultimodalPipeline init
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_default(self, pipeline):
        assert pipeline.resolution == (1920, 1080)
        assert pipeline.frame_count == 0

    def test_custom_resolution(self):
        p = MultimodalPipeline(resolution=(2560, 1440))
        assert p.resolution == (2560, 1440)

    def test_disabled_ocr(self, pipeline_disabled):
        assert pipeline_disabled.enable_ocr is False
        assert pipeline_disabled._has_ocr is False


# ---------------------------------------------------------------------------
# Process — desativado (testes estruturais)
# ---------------------------------------------------------------------------

class TestPipelineProcess:
    def test_empty_screenshot(self, pipeline):
        empty = np.array([])
        gs = pipeline.process(empty, [], game_state_hint="lobby", frame_id=0)
        assert isinstance(gs, GameState)
        assert gs.game_state == "lobby"
        assert gs.frame_id == 0

    def test_with_detections(self, pipeline, black_screenshot, sample_detections):
        gs = pipeline.process(
            black_screenshot,
            sample_detections,
            game_state_hint="in_game",
            game_state_confidence=0.9,
            frame_id=42,
        )
        assert gs.frame_id == 42
        assert gs.game_state == "in_game"
        assert gs.game_state_confidence == 0.9
        assert len(gs.detections) == 3
        assert gs.player_detection is not None
        assert gs.player_detection.class_name == "player"
        assert len(gs.enemy_detections) == 1
        assert len(gs.powerup_detections) == 1
        assert gs.enemy_count == 1

    def test_player_detection_none(self, pipeline, black_screenshot):
        dets = [{"class": 1, "bbox": [100, 100, 200, 200], "confidence": 0.9}]
        gs = pipeline.process(black_screenshot, dets, game_state_hint="in_game", frame_id=1)
        assert gs.player_detection is None
        assert len(gs.enemy_detections) == 1

    def test_bbox_normalization(self, pipeline, black_screenshot):
        dets = [{"class": 0, "bbox": [960, 540, 1060, 640], "confidence": 0.95}]
        gs = pipeline.process(black_screenshot, dets, frame_id=1)
        pd = gs.player_detection
        assert pd is not None
        # 960/1920 = 0.5, 540/1080 = 0.5
        assert abs(pd.bbox[0] - 0.5) < 0.01
        assert abs(pd.bbox[1] - 0.5) < 0.01

    def test_latency_recorded(self, pipeline, black_screenshot, sample_detections):
        gs = pipeline.process(black_screenshot, sample_detections, frame_id=1)
        assert gs.latency_ms >= 0.0

    def test_frame_count_increments(self, pipeline, black_screenshot):
        assert pipeline.frame_count == 0
        pipeline.process(black_screenshot, [], frame_id=1)
        assert pipeline.frame_count == 1
        pipeline.process(black_screenshot, [], frame_id=2)
        assert pipeline.frame_count == 2


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestPipelineStats:
    def test_stats_after_frames(self, pipeline, black_screenshot):
        pipeline.process(black_screenshot, [], frame_id=1)
        pipeline.process(black_screenshot, [], frame_id=2)
        stats = pipeline.get_stats()
        assert stats["frame_count"] == 2
        assert stats["avg_latency_ms"] >= 0.0
        assert stats["ocr_available"] is False  # desativado
        assert stats["player_detector_available"] is False

    def test_stats_initial(self, pipeline):
        stats = pipeline.get_stats()
        assert stats["frame_count"] == 0
        assert stats["avg_latency_ms"] == 0.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestPipelineReset:
    def test_reset_clears(self, pipeline, black_screenshot):
        pipeline.process(black_screenshot, [], frame_id=1)
        pipeline.process(black_screenshot, [], frame_id=2)
        pipeline.reset()
        assert pipeline.frame_count == 0
        assert pipeline.total_latency_ms == 0.0


# ---------------------------------------------------------------------------
# Merge HUD → Player
# ---------------------------------------------------------------------------

class TestMergeHudIntoPlayer:
    def test_hp_override(self):
        hud = HudValues(hp=0.5, hp_confidence=0.9)
        player = PlayerStatus(hp=1.0)
        merged = MultimodalPipeline._merge_hud_into_player(hud, player)
        assert merged.hp == 0.5

    def test_hp_low_confidence_no_override(self):
        hud = HudValues(hp=0.5, hp_confidence=0.2)
        player = PlayerStatus(hp=1.0)
        merged = MultimodalPipeline._merge_hud_into_player(hud, player)
        assert merged.hp == 1.0  # não sobrescreve

    def test_ammo_override(self):
        hud = HudValues(ammo=1, ammo_confidence=0.9)
        player = PlayerStatus(ammo=3)
        merged = MultimodalPipeline._merge_hud_into_player(hud, player)
        assert merged.ammo == 1

    def test_super_sets_ready(self):
        hud = HudValues(super_charge=1.0, super_confidence=0.9)
        player = PlayerStatus(super_ready=False, super_charge=0.0)
        merged = MultimodalPipeline._merge_hud_into_player(hud, player)
        assert merged.super_charge == 1.0
        assert merged.super_ready is True


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPipelinePerformance:
    def test_process_under_10ms(self, pipeline, black_screenshot, sample_detections):
        import time
        t0 = time.time()
        gs = pipeline.process(black_screenshot, sample_detections, frame_id=1)
        elapsed = (time.time() - t0) * 1000
        assert elapsed < 10.0, f"process() demorou {elapsed:.1f}ms"
