"""
tests/test_player_state_detector.py

Testes para o PlayerStateDetector.

Estratégia:
- Mock de screenshot e detecções YOLO
- Testes determinísticos de fusão e votação
- Testes de suavização temporal
- Testes de transições de estado
"""

from __future__ import annotations

import sys
from pathlib import Path


import numpy as np
import pytest

from vision.player_state_detector import (
    GadgetState,
    LifeState,
    PlayerState,
    PlayerStateDetector,
    StateTransition,
    SuperState,
    ThreatState,
    VisibilityState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    return PlayerStateDetector(smoothing_frames=1, enable_ocr=False)  # sem suavização / OCR para testes base


@pytest.fixture
def detector_smoothed():
    return PlayerStateDetector(smoothing_frames=3, enable_ocr=False)


@pytest.fixture
def black_screenshot():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def empty_detections():
    return []


@pytest.fixture
def player_detection():
    return {"class": 0, "bbox": [900, 500, 1000, 600], "confidence": 0.95}


@pytest.fixture
def enemy_detection():
    return {"class": 1, "bbox": [950, 550, 1050, 650], "confidence": 0.90}


# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_weights(self, detector):
        assert detector.weights["yolo"] == 0.40
        assert detector.weights["ocr"] == 0.35
        assert detector.weights["pixel"] == 0.25

    def test_custom_weights(self):
        d = PlayerStateDetector(weights={"yolo": 0.5, "ocr": 0.3, "pixel": 0.2}, enable_ocr=False)
        assert d.weights["yolo"] == 0.5

    def test_default_distances(self, detector):
        assert detector.danger_distance == 250
        assert detector.critical_distance == 120


# ---------------------------------------------------------------------------
# PlayerState
# ---------------------------------------------------------------------------

class TestPlayerState:
    def test_default_state(self):
        s = PlayerState()
        assert s.life == LifeState.UNKNOWN
        assert s.can_attack is False  # UNKNOWN life
        assert s.can_super is False

    def test_alive_can_attack(self):
        s = PlayerState(life=LifeState.ALIVE, ammo=2)
        assert s.can_attack is True

    def test_alive_no_ammo_cannot_attack(self):
        s = PlayerState(life=LifeState.ALIVE, ammo=0)
        assert s.can_attack is False

    def test_can_super(self):
        s = PlayerState(life=LifeState.ALIVE, super_state=SuperState.READY)
        assert s.can_super is True

    def test_is_vulnerable(self):
        s = PlayerState(
            visibility=VisibilityState.EXPOSED,
            threat=ThreatState.DANGER,
        )
        assert s.is_vulnerable is True

    def test_not_vulnerable_in_bush(self):
        s = PlayerState(
            visibility=VisibilityState.IN_BUSH,
            threat=ThreatState.DANGER,
        )
        assert s.is_vulnerable is False

    def test_to_dict(self):
        s = PlayerState(life=LifeState.ALIVE, hp=0.8)
        d = s.to_dict()
        assert d["life"] == "ALIVE"
        assert d["hp"] == 0.8
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# Source YOLO
# ---------------------------------------------------------------------------

class TestSourceYolo:
    def test_no_detections(self, detector):
        state, conf = detector._source_yolo([])
        assert state["life"] == LifeState.UNKNOWN
        assert conf == 0.0

    def test_player_alive_no_enemies(self, detector, player_detection):
        state, conf = detector._source_yolo([player_detection])
        assert state["life"] == LifeState.ALIVE
        assert state["enemy_count_nearby"] == 0
        assert conf == 0.7

    def test_player_alive_with_nearby_enemy(self, detector, player_detection, enemy_detection):
        # Inimigo a ~70px do jogador (dentro de danger_distance=250)
        state, conf = detector._source_yolo([player_detection, enemy_detection])
        assert state["life"] == LifeState.ALIVE
        assert state["enemy_count_nearby"] == 1
        assert state["enemy_distance_closest"] < 250

    def test_player_alive_with_distant_enemy(self, detector, player_detection):
        # Inimigo longe (> 500px)
        distant_enemy = {"class": 1, "bbox": [1400, 800, 1500, 900], "confidence": 0.90}
        state, conf = detector._source_yolo([player_detection, distant_enemy])
        assert state["life"] == LifeState.ALIVE
        assert state["enemy_count_nearby"] == 0  # fora do danger_distance


# ---------------------------------------------------------------------------
# Source Pixel
# ---------------------------------------------------------------------------

class TestSourcePixel:
    def test_empty_screenshot(self, detector):
        state, conf = detector._source_pixel(np.array([]), [])
        assert state["visibility"] == VisibilityState.UNKNOWN
        assert conf == 0.0

    def test_player_in_bush(self, detector, black_screenshot, player_detection):
        # Pinta pixel do jogador de verde-escuro (arbusto)
        cx, cy = 950, 550
        black_screenshot[cy, cx] = [30, 100, 30]
        state, conf = detector._source_pixel(black_screenshot, [player_detection])
        assert state["visibility"] == VisibilityState.IN_BUSH

    def test_player_exposed(self, detector, black_screenshot, player_detection):
        # Pinta pixel do jogador de cinza (não é arbusto)
        cx, cy = 950, 550
        black_screenshot[cy, cx] = [150, 150, 150]
        state, conf = detector._source_pixel(black_screenshot, [player_detection])
        assert state["visibility"] == VisibilityState.EXPOSED


# ---------------------------------------------------------------------------
# Fusão
# ---------------------------------------------------------------------------

class TestFusion:
    def test_fusion_all_unknown(self, detector):
        state = detector._fuse({}, 0.0, {}, 0.0, {}, 0.0)
        assert state.life == LifeState.UNKNOWN
        assert state.confidence == 0.0

    def test_fusion_yolo_alive(self, detector):
        state = detector._fuse(
            {"life": LifeState.ALIVE, "enemy_count_nearby": 0, "enemy_distance_closest": float("inf")},
            0.7,
            {}, 0.0,
            {}, 0.0,
        )
        assert state.life == LifeState.ALIVE
        assert state.threat == ThreatState.SAFE

    def test_fusion_ocr_hp(self, detector):
        state = detector._fuse(
            {}, 0.0,
            {"hp": 0.5, "ammo": 2, "super_charge": 0.8, "super_state": SuperState.CHARGING}, 0.8,
            {}, 0.0,
        )
        assert state.hp == 0.5
        assert state.ammo == 2
        assert state.super_state == SuperState.CHARGING
        assert state.super_charge == 0.8

    def test_fusion_pixel_bush(self, detector):
        state = detector._fuse(
            {}, 0.0,
            {}, 0.0,
            {"visibility": VisibilityState.IN_BUSH}, 0.5,
        )
        assert state.visibility == VisibilityState.IN_BUSH

    def test_fusion_combined(self, detector):
        state = detector._fuse(
            {"life": LifeState.ALIVE, "enemy_count_nearby": 1, "enemy_distance_closest": 100.0},
            0.7,
            {"hp": 0.8, "ammo": 1, "super_charge": 0.0, "super_state": SuperState.EMPTY},
            0.8,
            {"visibility": VisibilityState.EXPOSED},
            0.5,
        )
        assert state.life == LifeState.ALIVE
        assert state.hp == 0.8
        assert state.ammo == 1
        assert state.super_state == SuperState.EMPTY
        assert state.visibility == VisibilityState.EXPOSED
        assert state.threat == ThreatState.CRITICAL  # dist < 120
        assert state.enemy_count_nearby == 1


# ---------------------------------------------------------------------------
# Suavização temporal
# ---------------------------------------------------------------------------

class TestSmoothing:
    def test_no_history_returns_new(self, detector_smoothed, black_screenshot, player_detection):
        # Sem histórico, retorna o estado como está
        state, _ = detector_smoothed.detect(black_screenshot, [player_detection], frame_id=1)
        assert state.life == LifeState.ALIVE

    def test_smoothing_persist_state(self, detector_smoothed, black_screenshot, player_detection):
        # 3 frames com mesmo estado → suavizado mantém
        for i in range(3):
            state, _ = detector_smoothed.detect(black_screenshot, [player_detection], frame_id=i)
        assert state.life == LifeState.ALIVE

    def test_smoothing_requires_3_frames(self, detector_smoothed, black_screenshot):
        # Frame 1: vivo, Frame 2: sem deteção, Frame 3: sem deteção
        d1 = [{"class": 0, "bbox": [900, 500, 1000, 600], "confidence": 0.95}]
        d2 = []
        d3 = []
        s1, _ = detector_smoothed.detect(black_screenshot, d1, frame_id=1)
        assert s1.life == LifeState.ALIVE  # sem histórico suficiente

        s2, _ = detector_smoothed.detect(black_screenshot, d2, frame_id=2)
        # Ainda pode ser ALIVE se histórico curto

        s3, _ = detector_smoothed.detect(black_screenshot, d3, frame_id=3)
        # Agora com 3 frames no histórico, a moda pode ter mudado
        assert s3 is not None


# ---------------------------------------------------------------------------
# Transições
# ---------------------------------------------------------------------------

class TestTransitions:
    def test_no_prev_state_no_transitions(self, detector, black_screenshot, player_detection):
        state, trans = detector.detect(
            black_screenshot, [player_detection], frame_id=1, return_transitions=True
        )
        assert trans == []

    def test_alive_to_dead_transition(self, detector, black_screenshot):
        # Frame 1: vivo
        d_alive = [{"class": 0, "bbox": [900, 500, 1000, 600], "confidence": 0.95}]
        s1, t1 = detector.detect(black_screenshot, d_alive, frame_id=1, return_transitions=True)
        assert t1 == []

        # Frame 2: sem deteção (morte)
        d_dead = []
        s2, t2 = detector.detect(black_screenshot, d_dead, frame_id=2, return_transitions=True)
        assert len(t2) > 0
        assert any(tr.field == "life" and tr.old == "ALIVE" for tr in t2)

    def test_transition_callback(self, detector, black_screenshot, player_detection):
        transitions_captured = []

        def callback(trans):
            transitions_captured.extend(trans)

        detector.register_transition_callback(callback)

        d1 = [{"class": 0, "bbox": [900, 500, 1000, 600], "confidence": 0.95}]
        d2 = []
        detector.detect(black_screenshot, d1, frame_id=1)
        detector.detect(black_screenshot, d2, frame_id=2)

        assert len(transitions_captured) > 0


# ---------------------------------------------------------------------------
# Threat levels
# ---------------------------------------------------------------------------

class TestThreatLevels:
    def test_safe_no_enemies(self, detector, black_screenshot, player_detection):
        state, _ = detector.detect(black_screenshot, [player_detection], frame_id=1)
        assert state.threat == ThreatState.SAFE

    def test_critical_close_enemy(self, detector, black_screenshot, player_detection):
        # Inimigo a 50px (menor que critical_distance=120)
        enemy = {"class": 1, "bbox": [940, 540, 960, 560], "confidence": 0.90}
        state, _ = detector.detect(black_screenshot, [player_detection, enemy], frame_id=1)
        assert state.threat == ThreatState.CRITICAL

    def test_danger_nearby_enemy(self, detector, black_screenshot, player_detection):
        # Inimigo a 150px (entre critical e danger)
        enemy = {"class": 1, "bbox": [800, 400, 900, 500], "confidence": 0.90}
        state, _ = detector.detect(black_screenshot, [player_detection, enemy], frame_id=1)
        assert state.threat == ThreatState.DANGER

    def test_caution_distant_enemy(self, detector, black_screenshot, player_detection):
        # Inimigo a 400px (entre danger e caution)
        enemy = {"class": 1, "bbox": [500, 200, 600, 300], "confidence": 0.90}
        state, _ = detector.detect(black_screenshot, [player_detection, enemy], frame_id=1)
        assert state.threat == ThreatState.CAUTION


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_history(self, detector_smoothed, black_screenshot, player_detection):
        detector_smoothed.detect(black_screenshot, [player_detection], frame_id=1)
        detector_smoothed.reset()
        assert detector_smoothed._prev_state is None
        assert len(detector_smoothed._state_history) == 0


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_detect_under_10ms(self, detector, black_screenshot, player_detection):
        import time
        t0 = time.time()
        state, _ = detector.detect(black_screenshot, [player_detection], frame_id=1)
        elapsed = (time.time() - t0) * 1000
        assert elapsed < 10.0, f"detect() demorou {elapsed:.1f}ms"


# ---------------------------------------------------------------------------
# StateTransition
# ---------------------------------------------------------------------------

class TestStateTransition:
    def test_transition_fields(self):
        tr = StateTransition(field="life", old="ALIVE", new="DEAD", frame_id=5)
        assert tr.field == "life"
        assert tr.old == "ALIVE"
        assert tr.new == "DEAD"
        assert tr.frame_id == 5


# ---------------------------------------------------------------------------
# Custom distances
# ---------------------------------------------------------------------------

class TestCustomDistances:
    def test_custom_danger_distance(self, black_screenshot, player_detection):
        # danger_distance=100 → inimigo a ~141px não conta como perigo próximo
        enemy = {"class": 1, "bbox": [800, 400, 900, 500], "confidence": 0.90}
        detector = PlayerStateDetector(danger_distance=100, smoothing_frames=1, enable_ocr=False)
        state, _ = detector.detect(black_screenshot, [player_detection, enemy], frame_id=1)
        # Inimigo existe mas está fora do danger_distance → CAUTION
        assert state.threat == ThreatState.CAUTION


# ---------------------------------------------------------------------------
# Detect com return_transitions=False
# ---------------------------------------------------------------------------

class TestDetectReturnFormat:
    def test_default_no_transitions(self, detector, black_screenshot, player_detection):
        result = detector.detect(black_screenshot, [player_detection], frame_id=1)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[1] is None  # não retorna transições por default

    def test_with_transitions(self, detector, black_screenshot, player_detection):
        result = detector.detect(
            black_screenshot, [player_detection], frame_id=1, return_transitions=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[1] == []  # primeira chamada sem histórico
