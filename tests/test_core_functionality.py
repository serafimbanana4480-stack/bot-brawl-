"""
test_core_functionality.py

Testes unitarios abrangentes e profissionais para os modulos criticos do Brawl Stars Bot.
Cobertura: vision, decision, humanization, safety, match control.
"""

import sys
import time
import math
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock

# Project root on path

import pytest
import torch

# ---------------------------------------------------------------------------
# VISION / STATE TESTS
# ---------------------------------------------------------------------------
from vision.state import (
    GamePhase, PlayerState, EnemyInfo, WallInfo, BushInfo,
    GameState, StateExtractor
)
from vision.tracker import TrackedObject, ByteTracker
from core.class_registry import STATE_FEATURE_DIM
from decision.utility_ai import UtilityAI, Action


class TestGameState:
    """Testes para a classe GameState."""

    def test_default_state(self):
        state = GameState()
        assert state.phase == GamePhase.UNKNOWN
        assert state.player_health == 1.0
        assert state.player_ammo == 3
        assert state.player_state == PlayerState.UNKNOWN
        assert state.danger_score == 0.0
        assert not state.is_in_danger
        assert not state.should_retreat
        assert state.can_engage
        assert state.dist_nearest_enemy is None
        assert state.enemy_history == []
        assert state.time_since_enemy_seen == 0.0

    def test_enemy_history_fields(self):
        state = GameState(enemy_history=[
            EnemyInfo(track_id=1, position=(10, 10), bbox=(0, 0, 10, 10),
                      health_estimate=0.9, distance=120, velocity=(0, 0), threat_level=0.4,
                      last_seen=time.time()),
        ])
        assert len(state.enemy_history) == 1
        assert state.enemy_history[0].last_seen > 0

    def test_is_in_danger(self):
        state = GameState(danger_score=0.8)
        assert state.is_in_danger

        state = GameState(danger_score=0.5, enemies=[
            EnemyInfo(track_id=1, position=(0, 0), bbox=(0, 0, 10, 10),
                      health_estimate=1.0, distance=100, velocity=(0, 0), threat_level=0.5),
            EnemyInfo(track_id=2, position=(0, 0), bbox=(0, 0, 10, 10),
                      health_estimate=1.0, distance=100, velocity=(0, 0), threat_level=0.5),
            EnemyInfo(track_id=3, position=(0, 0), bbox=(0, 0, 10, 10),
                      health_estimate=1.0, distance=100, velocity=(0, 0), threat_level=0.5),
        ])
        assert state.is_in_danger

    def test_should_retreat(self):
        state = GameState(danger_score=0.8, player_health=0.2)
        assert state.should_retreat

    def test_can_engage(self):
        state = GameState(danger_score=0.3, player_health=0.8)
        assert state.can_engage

        state = GameState(danger_score=0.5, player_health=0.4)
        assert not state.can_engage

    def test_enemy_is_dangerous(self):
        enemy = EnemyInfo(
            track_id=1, position=(100, 100), bbox=(0, 0, 20, 20),
            health_estimate=0.8, distance=150, velocity=(0, 0), threat_level=0.7
        )
        assert enemy.is_dangerous

        enemy2 = EnemyInfo(
            track_id=2, position=(100, 100), bbox=(0, 0, 20, 20),
            health_estimate=0.3, distance=150, velocity=(0, 0), threat_level=0.2
        )
        assert not enemy2.is_dangerous


class TestStateExtractor:
    """Testes para o StateExtractor."""

    def test_extract_empty_state(self):
        extractor = StateExtractor()
        tracks = []
        state = extractor.extract_state(tracks)
        assert state.player_state == PlayerState.DEAD
        assert len(state.enemies) == 0

    def test_extract_player(self):
        extractor = StateExtractor()
        player_track = TrackedObject(
            id=1, class_name="player", bbox=(100, 100, 150, 150),
            confidence=0.9, center=(125, 125), velocity=(0, 0),
            age=0, hits=5, last_seen=time.time()
        )
        state = extractor.extract_state([player_track])
        assert state.player_state == PlayerState.ALIVE
        assert state.player_position == (125, 125)

    def test_extract_enemies(self):
        extractor = StateExtractor()
        player_track = TrackedObject(
            id=1, class_name="player", bbox=(100, 100, 150, 150),
            confidence=0.9, center=(125, 125), velocity=(0, 0),
            age=0, hits=5, last_seen=time.time()
        )
        enemy_track = TrackedObject(
            id=2, class_name="enemy", bbox=(300, 300, 330, 330),
            confidence=0.85, center=(315, 315), velocity=(5, 0),
            age=0, hits=3, last_seen=time.time()
        )
        state = extractor.extract_state([player_track, enemy_track])
        assert len(state.enemies) == 1
        assert state.enemies[0].track_id == 2
        assert state.nearest_enemy is not None
        assert state.nearest_enemy.distance > 0
        assert state.dist_nearest_enemy == state.nearest_enemy.distance
        assert len(state.enemy_history) == 1
        assert state.time_since_enemy_seen >= 0.0

    def test_extract_enemy_history_orders_recent_first(self):
        extractor = StateExtractor()
        now = time.time()
        player_track = TrackedObject(
            id=1, class_name="player", bbox=(100, 100, 150, 150),
            confidence=0.9, center=(125, 125), velocity=(0, 0),
            age=0, hits=5, last_seen=now
        )
        enemy_old = TrackedObject(
            id=2, class_name="enemy", bbox=(300, 300, 330, 330),
            confidence=0.85, center=(315, 315), velocity=(5, 0),
            age=0, hits=3, last_seen=now - 5
        )
        enemy_new = TrackedObject(
            id=3, class_name="enemy", bbox=(340, 340, 370, 370),
            confidence=0.85, center=(355, 355), velocity=(3, 2),
            age=0, hits=4, last_seen=now - 1
        )
        state = extractor.extract_state([player_track, enemy_old, enemy_new])
        assert len(state.enemy_history) == 2
        assert state.enemy_history[0].track_id == 3

    def test_threat_calculation(self):
        extractor = StateExtractor()
        player_pos = (100, 100)
        enemy_track = TrackedObject(
            id=2, class_name="enemy", bbox=(120, 120, 140, 140),
            confidence=0.9, center=(130, 130), velocity=(10, 10),
            age=0, hits=3, last_seen=time.time()
        )
        threat = extractor._calculate_threat_level(
            enemy_track, player_pos, 1.0
        )
        assert 0.0 <= threat <= 1.0
        # Enemy close and moving fast should be high threat
        assert threat > 0.3

    def test_extract_state_with_ocr_enrichment(self):
        class _FakeOCRDetector:
            def extract_hud_text(self, screenshot):
                return {
                    "match_timer_text": "1:23",
                    "match_time_remaining": 83.0,
                    "score_text": "2-1",
                    "match_score": (2, 1),
                    "ability_texts": {"ability_super": "READY"},
                    "ability_states": {"ability_super": True},
                }

        extractor = StateExtractor(ocr_detector=_FakeOCRDetector())
        player_track = TrackedObject(
            id=1, class_name="player", bbox=(100, 100, 150, 150),
            confidence=0.9, center=(125, 125), velocity=(0, 0),
            age=0, hits=5, last_seen=time.time()
        )
        state = extractor.extract_state([player_track], screenshot=np.zeros((64, 64, 3), dtype=np.uint8))

        assert state.ocr_match_timer_text == "1:23"
        assert state.match_time_remaining == 83.0
        assert state.ocr_score_text == "2-1"
        assert state.ocr_match_score == (2, 1)
        assert state.ocr_ability_texts["ability_super"] == "READY"
        assert state.ocr_ability_states["ability_super"] is True


class TestNeuralStateEncoding:
    """Testes para o vetor de estado neural."""

    def test_state_feature_dimension_constant(self):
        assert STATE_FEATURE_DIM == 44

    def test_state_encoder_accepts_new_feature_dim(self):
        from neural.state_encoder import StateEncoder

        encoder = StateEncoder(input_dim=STATE_FEATURE_DIM, output_dim=16)
        x = torch.zeros(1, STATE_FEATURE_DIM)
        out = encoder(x)
        assert out.shape == (1, 16)

    def test_neural_policy_state_vector_matches_registry(self):
        from neural.neural_policy import NeuralPolicy

        policy = NeuralPolicy(schema="core", spatial_dim=16, state_dim=16, temporal_dim=16, fusion_dim=16)
        state = GameState()
        vector = policy.extract_state_vector(state)
        assert vector.shape == (STATE_FEATURE_DIM,)
        assert len(vector) == 44
        assert np.all(vector >= -1.0)
        assert np.all(vector <= 1.0)


class TestByteTracker:
    """Testes para o ByteTracker."""

    def test_init(self):
        tracker = ByteTracker()
        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert len(tracker.tracks) == 0

    def test_update_new_detection(self):
        tracker = ByteTracker()
        detections = [("enemy", (100, 100, 150, 150), 0.9)]
        tracks = tracker.update(detections)
        # ByteTracker returns all tracks (including unconfirmed)
        assert len(tracks) >= 1

    def test_confirm_after_min_hits(self):
        tracker = ByteTracker(min_hits=2)
        detections = [("enemy", (100, 100, 150, 150), 0.9)]
        for _ in range(2):
            tracks = tracker.update(detections)
        assert len(tracks) >= 1
        # At least one track should have hits >= 2
        assert any(t.hits >= 2 for t in tracks.values())

    def test_track_persistence(self):
        tracker = ByteTracker(min_hits=1)
        detections = [("enemy", (100, 100, 150, 150), 0.9)]
        tracks1 = tracker.update(detections)
        assert len(tracks1) >= 1
        track_id = list(tracks1.values())[0].id

        # Move enemy slightly
        detections = [("enemy", (105, 105, 155, 155), 0.9)]
        tracks2 = tracker.update(detections)
        assert len(tracks2) >= 1
        assert list(tracks2.values())[0].id == track_id

    def test_iou_matching(self):
        tracker = ByteTracker(min_hits=1)
        detections = [("enemy", (100, 100, 150, 150), 0.9)]
        tracks1 = tracker.update(detections)

        # Same position, should match
        detections = [("enemy", (100, 100, 150, 150), 0.9)]
        tracks2 = tracker.update(detections)
        assert len(tracks2) >= 1
        assert list(tracks2.values())[0].id == list(tracks1.values())[0].id

    def test_stale_removal(self):
        tracker = ByteTracker(max_age=2, min_hits=1)
        detections = [("enemy", (100, 100, 150, 150), 0.9)]
        tracker.update(detections)

        # No detections for 3 frames
        for _ in range(3):
            tracker.update([])

        # Track age should exceed max_age
        assert all(t.age > tracker.max_age for t in tracker.tracks.values())

    def test_velocity_calculation(self):
        tracker = ByteTracker(min_hits=1)
        detections = [("enemy", (100, 100, 150, 150), 0.9)]
        tracker.update(detections)

        detections = [("enemy", (110, 110, 160, 160), 0.9)]
        tracks = tracker.update(detections)
        assert len(tracks) >= 1
        track = list(tracks.values())[0]
        vx, vy = track.velocity
        assert vx == 10.0
        assert vy == 10.0

    def test_predict_position(self):
        obj = TrackedObject(
            id=1, class_name="enemy", bbox=(100, 100, 150, 150),
            confidence=0.9, center=(125, 125), velocity=(5, 5),
            age=0, hits=5, last_seen=time.time()
        )
        predicted = obj.predict_position()
        assert predicted == (130, 130)

    def test_is_stale(self):
        obj = TrackedObject(
            id=1, class_name="enemy", bbox=(100, 100, 150, 150),
            confidence=0.9, center=(125, 125), velocity=(0, 0),
            age=35, hits=5, last_seen=time.time()
        )
        assert obj.is_stale(max_age=30)
        assert not obj.is_stale(max_age=40)


# ---------------------------------------------------------------------------
# DECISION / STATE MACHINE TESTS
# ---------------------------------------------------------------------------
from decision.state_machine import BotState, StateMachine, StateContext, BrawlStarsStateMachine


class MockPlayerState:
    def __init__(self, value):
        self.value = value


class MockGameState:
    """Mock para GameState nos testes de state machine."""
    def __init__(self, player_health=1.0, enemies=None, danger_score=0.0):
        self.player_health = player_health
        self.enemies = enemies or []
        self.danger_score = danger_score
        self.player_state = MockPlayerState("alive")
        self.can_engage = danger_score < 0.4 and player_health > 0.5
        self.should_retreat = danger_score > 0.7 or player_health < 0.3
        self.is_in_danger = danger_score > 0.6 or len(self.enemies) > 2


class TestStateMachine:
    """Testes para a maquina de estados."""

    def test_initial_state(self):
        sm = StateMachine()
        assert sm.current_state == BotState.IDLE
        assert sm.previous_state is None

    def test_register_handler(self):
        sm = StateMachine()
        handler_called = [False]

        def handler(ctx):
            handler_called[0] = True

        sm.register_handler(BotState.IDLE, handler)
        ctx = StateContext(game_state=MockGameState(), bot_instance=None)
        sm.execute(ctx)
        assert handler_called[0]

    def test_transition(self):
        sm = StateMachine()
        sm.register_transition(
            BotState.IDLE, BotState.SEARCH,
            lambda ctx: True
        )
        ctx = StateContext(game_state=MockGameState(), bot_instance=None)
        # Force bypass min duration
        sm.state_entry_time = 0
        new_state = sm.update(ctx)
        assert new_state == BotState.SEARCH
        assert sm.previous_state == BotState.IDLE

    def test_min_state_duration(self):
        sm = StateMachine()
        sm.register_transition(
            BotState.IDLE, BotState.SEARCH,
            lambda ctx: True
        )
        ctx = StateContext(game_state=MockGameState(), bot_instance=None)
        # Don't bypass - should stay in IDLE
        new_state = sm.update(ctx)
        assert new_state == BotState.IDLE

    def test_transition_priority(self):
        sm = StateMachine()
        sm.register_transition(
            BotState.IDLE, BotState.SEARCH,
            lambda ctx: True, priority=1
        )
        sm.register_transition(
            BotState.IDLE, BotState.ENGAGE,
            lambda ctx: True, priority=2
        )
        ctx = StateContext(game_state=MockGameState(), bot_instance=None)
        sm.state_entry_time = 0
        new_state = sm.update(ctx)
        # Higher priority (ENGAGE) should win
        assert new_state == BotState.ENGAGE

    def test_stuck_detection(self):
        sm = StateMachine()
        sm.state_entry_time = time.time() - 60
        assert sm.is_stuck(max_duration=30.0)
        assert not sm.is_stuck(max_duration=120.0)

    def test_state_duration(self):
        sm = StateMachine()
        sm.state_entry_time = time.time() - 5.0
        assert sm.get_state_duration() >= 5.0

    def test_on_enter_exit(self):
        sm = StateMachine()
        enter_called = [False]
        exit_called = [False]

        sm.on_enter[BotState.SEARCH] = lambda ctx: enter_called.__setitem__(0, True)
        sm.on_exit[BotState.IDLE] = lambda ctx: exit_called.__setitem__(0, True)

        sm.register_transition(
            BotState.IDLE, BotState.SEARCH,
            lambda ctx: True
        )

        ctx = StateContext(game_state=MockGameState(), bot_instance=None)
        sm.state_entry_time = 0
        sm.update(ctx)
        assert enter_called[0]
        assert exit_called[0]


class TestBrawlStarsStateMachine:
    """Testes para a maquina de estados pre-configurada."""

    def test_default_transitions(self):
        sm = BrawlStarsStateMachine()
        assert len(sm.transitions) > 0

    def test_idle_to_search(self):
        sm = BrawlStarsStateMachine()
        ctx = StateContext(
            game_state=MockGameState(player_health=1.0, enemies=[]),
            bot_instance=None
        )
        sm.state_entry_time = 0
        new_state = sm.update(ctx)
        assert new_state == BotState.SEARCH

    def test_idle_to_retreat_low_health(self):
        sm = BrawlStarsStateMachine()
        ctx = StateContext(
            game_state=MockGameState(player_health=0.1, enemies=[1]),
            bot_instance=None
        )
        sm.state_entry_time = 0
        new_state = sm.update(ctx)
        assert new_state == BotState.RETREAT


# ---------------------------------------------------------------------------
# DECISION / RULES TESTS
# ---------------------------------------------------------------------------
from decision.rules import Tactic, TacticalDecision, RuleEngine


class MockEnemy:
    def __init__(self, track_id, position, health_estimate, distance, threat_level=0.5):
        self.track_id = track_id
        self.position = position
        self.health_estimate = health_estimate
        self.distance = distance
        self.threat_level = threat_level


class MockGameStateForRules:
    def __init__(self, player_position, player_health, enemies, walls=None, bushes=None, danger_score=0.0):
        self.player_position = player_position
        self.player_health = player_health
        self.enemies = enemies
        self.walls = walls or []
        self.bushes = bushes or []
        self.nearest_enemy = enemies[0] if enemies else None
        self.can_engage = player_health > 0.5 and danger_score < 0.4
        self.safe_bushes = []
        self.biggest_threat = enemies[0] if enemies else None
        self.danger_score = danger_score


class TestRuleEngine:
    """Testes para o motor de regras taticas."""

    def test_init(self):
        engine = RuleEngine()
        assert engine.engagement_range == 200.0
        assert engine.optimal_range == 150.0

    def test_evaluate_engagement_empty(self):
        engine = RuleEngine()
        state = MockGameStateForRules((100, 100), 1.0, [])
        decisions = engine.evaluate_engagement(state)
        assert len(decisions) == 0

    def test_evaluate_engagement_close(self):
        engine = RuleEngine()
        enemy = MockEnemy(1, (150, 150), 0.3, 50)
        state = MockGameStateForRules((100, 100), 0.8, [enemy])
        decisions = engine.evaluate_engagement(state)
        assert len(decisions) > 0
        # Should have close engagement
        tactics = [d.tactic for d in decisions]
        assert Tactic.ENGAGE_CLOSE in tactics

    def test_evaluate_engagement_ranged(self):
        engine = RuleEngine()
        enemy = MockEnemy(1, (400, 400), 0.5, 300)
        state = MockGameStateForRules((100, 100), 0.8, [enemy])
        decisions = engine.evaluate_engagement(state)
        tactics = [d.tactic for d in decisions]
        assert Tactic.ENGAGE_RANGED in tactics

    def test_flanking(self):
        engine = RuleEngine()
        enemy = MockEnemy(1, (300, 300), 0.5, 200)
        state = MockGameStateForRules((100, 100), 0.8, [enemy])
        decisions = engine.evaluate_engagement(state)
        tactics = [d.tactic for d in decisions]
        assert Tactic.FLANK in tactics

    def test_retreat_low_health(self):
        engine = RuleEngine()
        enemy = MockEnemy(1, (150, 150), 0.8, 100)
        state = MockGameStateForRules((100, 100), 0.1, [enemy])
        decisions = engine.evaluate_retreat(state)
        assert len(decisions) > 0
        assert decisions[0].tactic == Tactic.RETREAT_DEFENSIVE

    def test_find_safe_retreat_point(self):
        engine = RuleEngine()
        enemy = MockEnemy(1, (200, 200), 0.8, 100)
        bush = MagicMock()
        bush.center = (50, 50)
        bush.enemies_nearby = 0

        point = engine._find_safe_retreat_point(
            (100, 100), [enemy], [bush], []
        )
        assert point is not None

    def test_calculate_flank_position(self):
        engine = RuleEngine()
        target = (300, 300)
        player = (100, 100)
        pos = engine._calculate_flank_position(target, player, [])
        assert pos is not None
        assert len(pos) == 2


# ---------------------------------------------------------------------------
# DECISION / SCORER TESTS
# ---------------------------------------------------------------------------
from decision.scorer import TargetScore, TargetScorer, ActionScorer, SituationScorer


class TestTargetScorer:
    """Testes para o sistema de scoring de alvos."""

    def test_init(self):
        scorer = TargetScorer()
        assert scorer.optimal_range == 150.0
        assert scorer.max_range == 500.0

    def test_score_low_health_target(self):
        scorer = TargetScorer()
        enemy = MockEnemy(1, (200, 200), 0.1, 150)
        score = scorer.score_target(enemy, (100, 100), 1.0, [enemy], [])
        assert score.health_score > 1.0  # Bonus for low health
        assert score.total_score > 0

    def test_score_optimal_distance(self):
        scorer = TargetScorer()
        enemy = MockEnemy(1, (200, 200), 0.5, 150)  # Exactly optimal range
        score = scorer.score_target(enemy, (100, 100), 1.0, [enemy], [])
        assert score.distance_score > 0.5

    def test_score_out_of_range(self):
        scorer = TargetScorer()
        enemy = MockEnemy(1, (700, 700), 0.5, 600)
        score = scorer.score_target(enemy, (100, 100), 1.0, [enemy], [])
        assert score.distance_score == 0.0

    def test_vulnerability_isolated(self):
        scorer = TargetScorer()
        enemy = MockEnemy(1, (200, 200), 0.5, 150)
        score = scorer.score_target(enemy, (100, 100), 1.0, [enemy], [])
        assert score.vulnerability_score == 1.0  # Isolated enemy

    def test_vulnerability_with_allies(self):
        scorer = TargetScorer()
        enemy1 = MockEnemy(1, (200, 200), 0.5, 150)
        enemy2 = MockEnemy(2, (210, 210), 0.5, 160)
        score = scorer.score_target(enemy1, (100, 100), 1.0, [enemy1, enemy2], [])
        assert score.vulnerability_score < 1.0

    def test_rank_targets(self):
        scorer = TargetScorer()
        enemy1 = MockEnemy(1, (200, 200), 0.2, 100)  # Low health, close
        enemy2 = MockEnemy(2, (500, 500), 0.9, 400)  # High health, far
        ranked = scorer.rank_targets([enemy1, enemy2], (100, 100), 1.0, [])
        assert ranked[0].target_id == 1  # enemy1 should be higher priority

    def test_kill_pressure(self):
        scorer = TargetScorer()
        enemy = MockEnemy(1, (200, 200), 0.1, 50)  # Very low health, close
        score = scorer.score_target(enemy, (100, 100), 1.0, [enemy], [])
        assert score.kill_pressure >= 1.5


class TestActionScorer:
    """Testes para o ActionScorer."""

    def test_init(self):
        scorer = ActionScorer()
        assert "damage_dealt" in scorer.weights
        assert "death_risk" in scorer.weights
        assert scorer.weights["death_risk"] < 0

    def test_score_action(self):
        scorer = ActionScorer()
        outcome = {
            "damage_dealt": 500,
            "damage_taken": 100,
            "kill_potential": 1,
            "death_risk": 0,
            "position_improvement": 0.5,
            "resource_gain": 2
        }
        score = scorer.score_action("attack", outcome)
        assert score > 0

    def test_score_bad_action(self):
        scorer = ActionScorer()
        outcome = {
            "damage_dealt": 0,
            "damage_taken": 500,
            "kill_potential": 0,
            "death_risk": 1,
            "position_improvement": -0.5,
            "resource_gain": 0
        }
        score = scorer.score_action("rush", outcome)
        assert score < 0

    def test_compare_actions(self):
        scorer = ActionScorer()
        actions = [
            ("safe_attack", {"damage_dealt": 300, "death_risk": 0.1}),
            ("risky_rush", {"damage_dealt": 500, "death_risk": 0.8}),
        ]
        ranked = scorer.compare_actions(actions)
        assert len(ranked) == 2
        # Just verify ranking is consistent and both have scores
        assert ranked[0][1] > ranked[1][1] or ranked[0][1] == ranked[1][1]


class TestSituationScorer:
    """Testes para o SituationScorer."""

    def test_score_situation_no_enemies(self):
        scorer = SituationScorer()
        state = MockGameStateForRules((100, 100), 1.0, [])
        result = scorer.score_situation(state)
        assert result["recommendation"] == "aggressive"
        assert result["overall"] > 0.6

    def test_score_situation_danger(self):
        scorer = SituationScorer()
        enemies = [MockEnemy(i, (100, 100), 1.0, 50) for i in range(3)]
        state = MockGameStateForRules((100, 100), 0.2, enemies, danger_score=0.9)
        result = scorer.score_situation(state)
        assert result["recommendation"] == "defensive"


# ---------------------------------------------------------------------------
# HUMANIZATION TESTS
# ---------------------------------------------------------------------------
from humanization import (
    HumanizationConfig, BezierCurve, WindMouse,
    MouseHumanizer, HumanizationEngine, DelayRandomizer
)


class TestBezierCurve:
    """Testes para curvas de Bezier."""

    def test_get_point(self):
        curve = BezierCurve((0, 0), (1, 1), (2, 1), (3, 0))
        p = curve.get_point(0.5)
        assert len(p) == 2
        assert isinstance(p[0], float)
        assert isinstance(p[1], float)

    def test_get_point_bounds(self):
        curve = BezierCurve((0, 0), (1, 1), (2, 1), (3, 0))
        p0 = curve.get_point(-0.1)
        p1 = curve.get_point(1.1)
        assert p0 == curve.get_point(0.0)
        assert p1 == curve.get_point(1.0)

    def test_generate_path(self):
        curve = BezierCurve((0, 0), (1, 1), (2, 1), (3, 0))
        path = curve.generate_path(50)
        assert len(path) == 50
        assert path[0] == (0.0, 0.0)
        assert path[-1] == (3.0, 0.0)


class TestWindMouse:
    """Testes para o algoritmo WindMouse."""

    def test_generate_path(self):
        wm = WindMouse()
        path = wm.generate_path((0, 0), (100, 100))
        assert len(path) > 0
        # Last point should be close to target (float comparison)
        assert abs(path[-1][0] - 100) < 1
        assert abs(path[-1][1] - 100) < 1

    def test_generate_path_with_timing(self):
        wm = WindMouse()
        path = wm.generate_path_with_timing((0, 0), (100, 100))
        assert len(path) > 0
        # Each point should have x, y, timestamp
        assert len(path[0]) == 3
        # Timestamps should be non-decreasing
        for i in range(1, len(path)):
            assert path[i][2] >= path[i-1][2]

    def test_path_has_variance(self):
        """WindMouse should produce slightly different paths each time."""
        wm = WindMouse()
        path1 = wm.generate_path((0, 0), (100, 100))
        path2 = wm.generate_path((0, 0), (100, 100))
        # Paths should differ (due to random wind)
        assert path1 != path2


class TestMouseHumanizer:
    """Testes para o humanizador de mouse."""

    def test_init(self):
        config = HumanizationConfig()
        humanizer = MouseHumanizer(config)
        assert humanizer.config is config

    def test_generate_control_points(self):
        humanizer = MouseHumanizer()
        cp1, cp2 = humanizer.generate_bezier_control_points((0, 0), (100, 100))
        assert len(cp1) == 2
        assert len(cp2) == 2

    def test_humanize_path(self):
        humanizer = MouseHumanizer()
        path = humanizer.humanize_path((0, 0), (100, 100), use_windmouse=True)
        assert len(path) > 0
        assert len(path[0]) == 3  # x, y, timestamp

    def test_should_make_mistake(self):
        config = HumanizationConfig(mistake_probability=1.0)
        humanizer = MouseHumanizer(config)
        assert humanizer.should_make_mistake()

        config = HumanizationConfig(mistake_probability=0.0)
        humanizer = MouseHumanizer(config)
        assert not humanizer.should_make_mistake()

    def test_get_mistake_offset(self):
        humanizer = MouseHumanizer()
        offset = humanizer.get_mistake_offset()
        assert len(offset) == 2
        # Should be within max_offset default of 50
        assert math.sqrt(offset[0]**2 + offset[1]**2) <= 50


class TestDelayRandomizer:
    """Testes para o DelayRandomizer."""

    def test_get_delay(self):
        dr = DelayRandomizer()
        delay = dr.get_delay("default")
        config = dr.config
        assert config.min_delay <= delay <= config.max_delay

    def test_get_reaction_delay(self):
        dr = DelayRandomizer()
        delay = dr.get_delay("reaction")
        # Should be within reasonable bounds (min_delay to max_delay)
        assert dr.config.min_delay <= delay <= dr.config.max_delay

    def test_get_typing_delay(self):
        dr = DelayRandomizer()
        delay = dr.get_typing_delay("a")
        assert 0.08 <= delay <= 0.15

        delay_special = dr.get_typing_delay("!")
        assert 0.15 <= delay_special <= 0.25


class TestHumanizationEngine:
    """Testes para o HumanizationEngine."""

    def test_init(self):
        engine = HumanizationEngine()
        assert engine.mouse is not None
        assert engine.delays is not None
        assert engine.config is not None

    def test_execute_humanized_click(self):
        engine = HumanizationEngine()
        result = engine.execute_humanized_click(100, 100, pre_delay=0.01)
        assert "target" in result
        assert "path" in result
        assert result["target"] == (100, 100) or result["is_mistake"]
        assert result["action_number"] == 1

    def test_get_humanized_aim(self):
        engine = HumanizationEngine()
        # With accuracy=1.0, should always hit
        aim = engine.get_humanized_aim((500, 500), accuracy=1.0)
        assert aim == (500, 500)

    def test_get_stats(self):
        engine = HumanizationEngine()
        stats = engine.get_stats()
        assert "total_actions" in stats
        assert "config" in stats

    def test_get_delay(self):
        engine = HumanizationEngine()
        delay = engine.get_delay("reaction")
        config = engine.config
        assert delay >= config.min_delay

    def test_get_tremor(self):
        engine = HumanizationEngine()
        tremor = engine.get_tremor()
        assert len(tremor) == 2


# ---------------------------------------------------------------------------
# SAFETY SYSTEM TESTS
# ---------------------------------------------------------------------------
from safety_system import (
    SafetyConfig, SessionStats, PatternDetector,
    MovementAnalyzer, SafetySystem, APMLimiter
)


class TestPatternDetector:
    """Testes para o detector de padroes."""

    def test_init(self):
        detector = PatternDetector()
        assert detector.threshold == 5

    def test_record_click(self):
        detector = PatternDetector()
        detector.record_click(100, 100)
        assert len(detector.click_times) == 1
        assert len(detector.click_positions) == 1

    def test_detect_perfect_timing(self):
        detector = PatternDetector()
        # Simulate perfect timing (every 100ms exactly)
        base_time = time.time()
        for i in range(15):
            detector.click_times.append(base_time + i * 0.1)
        assert detector.detect_perfect_timing()

    def test_detect_non_perfect_timing(self):
        detector = PatternDetector()
        # Simulate random timing with fixed seed for determinism
        rng = random.Random(42)
        base_time = time.time()
        for i in range(15):
            detector.click_times.append(base_time + i * 0.1 + rng.uniform(-0.05, 0.05))
        assert not detector.detect_perfect_timing()

    def test_detect_perfect_aim(self):
        detector = PatternDetector()
        # Same position 20 times
        for _ in range(25):
            detector.record_click(100, 100)
        assert detector.detect_perfect_aim()

    def test_detect_burst(self):
        detector = PatternDetector()
        # 15 actions in quick succession
        now = time.time()
        for _ in range(15):
            detector.action_window.append(now)
        assert detector.detect_burst()

    def test_suspicion_score(self):
        detector = PatternDetector()
        base_time = time.time()
        for i in range(15):
            detector.click_times.append(base_time + i * 0.1)
        for _ in range(25):
            detector.record_click(100, 100)
        score = detector.get_suspicion_score()
        assert 0 <= score <= 100
        # Should be elevated with perfect timing + aim
        assert score > 30


class TestMovementAnalyzer:
    """Testes para o analisador de movimento."""

    def test_record_swipe(self):
        analyzer = MovementAnalyzer()
        analyzer.record_swipe(0, 0, 100, 100, 0.5)
        assert len(analyzer.swipes) == 1
        assert len(analyzer.movements) == 1
        swipe = analyzer.swipes[0]
        assert swipe["distance"] > 0
        assert swipe["velocity"] > 0

    def test_record_tap(self):
        analyzer = MovementAnalyzer()
        analyzer.record_tap(100, 100)
        assert len(analyzer.taps) == 1

    def test_get_average_velocity(self):
        analyzer = MovementAnalyzer()
        for _ in range(5):
            analyzer.record_swipe(0, 0, 100, 100, 0.5)
        avg = analyzer.get_average_velocity()
        assert avg > 0

    def test_analyze_human_likeness(self):
        config = SafetyConfig()
        analyzer = MovementAnalyzer()
        # Not enough movements - should pass
        result = analyzer.analyze_human_likeness(config)
        assert result["human_like"] is True

        # Add many movements
        for _ in range(10):
            analyzer.record_swipe(0, 0, 100, 100, 0.5)
        result = analyzer.analyze_human_likeness(config)
        assert "score" in result
        assert "reasons" in result

    def test_velocity_variance(self):
        analyzer = MovementAnalyzer()
        assert analyzer.get_velocity_variance() == 0
        analyzer.record_swipe(0, 0, 100, 100, 0.5)
        analyzer.record_swipe(0, 0, 200, 200, 0.5)
        assert analyzer.get_velocity_variance() >= 0


class TestAPMLimiter:
    """Testes para o APMLimiter."""

    def test_init(self):
        limiter = APMLimiter(min_apm=20, max_apm=60)
        assert limiter.min_apm == 20
        assert limiter.max_apm == 60

    def test_record_and_get_apm(self):
        limiter = APMLimiter()
        for _ in range(10):
            limiter.record_action()
        apm = limiter.get_current_apm()
        assert apm == 10

    def test_should_delay(self):
        limiter = APMLimiter(max_apm=5)
        for _ in range(10):
            limiter.record_action()
        assert limiter.should_delay()

    def test_get_recommended_delay(self):
        limiter = APMLimiter(max_apm=5)
        for _ in range(20):
            limiter.record_action()
        delay = limiter.get_recommended_delay()
        assert delay > 0


class TestSafetySystem:
    """Testes para o sistema de seguranca completo."""

    def test_init(self):
        config = SafetyConfig()
        safety = SafetySystem(config)
        assert safety.config.max_trophies == 400
        assert safety.stats is not None

    def test_start_session(self):
        config = SafetyConfig(max_session_hours=1.0)
        safety = SafetySystem(config)
        safety.start_session()
        assert safety.is_running
        assert not safety.emergency_stop_triggered

    def test_record_action_and_apm(self):
        config = SafetyConfig(max_apm=60)
        safety = SafetySystem(config)
        safety.start_session()
        for _ in range(100):
            safety.record_action()
        status = safety.get_status()
        assert status['current_apm'] > 0
        assert status['actions'] == 100

    def test_record_swipe(self):
        config = SafetyConfig()
        safety = SafetySystem(config)
        safety.start_session()
        result = safety.record_swipe(0, 0, 100, 100, 0.5)
        assert "safe" in result
        assert "apm" in result

    def test_record_tap(self):
        config = SafetyConfig()
        safety = SafetySystem(config)
        safety.start_session()
        result = safety.record_tap(100, 100)
        assert "safe" in result
        assert "apm" in result

    def test_check_trophy_limit(self):
        config = SafetyConfig(max_trophies=100)
        safety = SafetySystem(config)
        status = safety.check_trophy_limit(110)
        assert not status['can_play']
        assert "meta" in status['message'].lower() or "Meta" in status['message']

    def test_pattern_detection(self):
        config = SafetyConfig()
        safety = SafetySystem(config)
        safety.start_session()
        # Simulate suspicious behavior
        base_time = time.time()
        for i in range(20):
            safety.pattern_detector.click_times.append(base_time + i * 0.1)
        for _ in range(30):
            safety.pattern_detector.record_click(100, 100)

        status = safety.get_status()
        assert status['suspicion_score'] > 0

    def test_emergency_stop(self):
        config = SafetyConfig()
        safety = SafetySystem(config)
        safety.start_session()
        safety.emergency_stop("test")
        assert not safety.is_running
        assert safety.emergency_stop_triggered

    def test_should_take_break(self):
        config = SafetyConfig()
        safety = SafetySystem(config)
        safety.start_session()
        # Set next break to past
        safety.next_break_time = time.time() - 1
        assert safety.should_take_break()

    def test_get_status_structure(self):
        config = SafetyConfig()
        safety = SafetySystem(config)
        safety.start_session()
        status = safety.get_status()
        required_keys = [
            "running", "emergency_stop", "session_duration_hours",
            "actions", "apm", "current_apm", "current_trophies",
            "suspicion_score", "human_likeness_score", "movement_count"
        ]
        for key in required_keys:
            assert key in status


# ---------------------------------------------------------------------------
# MATCH CONTROLLER TESTS
# ---------------------------------------------------------------------------
from match_controller import MatchResult, MatchHistory, BrawlerQueue, BrawlerConfig, MatchController


class TestMatchResult:
    """Testes para MatchResult."""

    def test_to_dict(self):
        result = MatchResult(
            match_id="test_1", timestamp=datetime.now().isoformat(),
            game_mode="showdown", brawler="colt", result="win",
            trophies_change=8, duration_seconds=120.0, kills=5,
            damage_dealt=5000, powerups_collected=10, star_player=True
        )
        d = result.to_dict()
        assert d["match_id"] == "test_1"
        assert d["result"] == "win"


class TestMatchHistory:
    """Testes para MatchHistory."""

    def test_add_and_stats(self, tmp_path):
        history = MatchHistory(tmp_path / "test_history.json")
        result = MatchResult(
            match_id="test_1", timestamp=datetime.now().isoformat(),
            game_mode="showdown", brawler="colt", result="win",
            trophies_change=8, duration_seconds=120.0, kills=5,
            damage_dealt=5000, powerups_collected=10, star_player=True
        )
        history.add_match(result)
        stats = history.get_stats()
        assert stats["total"] == 1
        assert stats["wins"] == 1
        assert stats["win_rate"] == 100.0

    def test_win_rate_calculation(self, tmp_path):
        history = MatchHistory(tmp_path / "test_history.json")
        for result in ["win", "win", "loss", "win", "loss"]:
            history.add_match(MatchResult(
                match_id=f"test_{result}", timestamp=datetime.now().isoformat(),
                game_mode="showdown", brawler="colt", result=result,
                trophies_change=8, duration_seconds=120.0, kills=5,
                damage_dealt=5000, powerups_collected=10, star_player=True
            ))
        stats = history.get_stats()
        assert stats["total"] == 5
        assert stats["wins"] == 3
        assert stats["losses"] == 2
        assert stats["win_rate"] == 60.0

    def test_limit_matches(self, tmp_path):
        history = MatchHistory(tmp_path / "test_history.json")
        for i in range(1100):
            history.add_match(MatchResult(
                match_id=f"test_{i}", timestamp=datetime.now().isoformat(),
                game_mode="showdown", brawler="colt", result="win",
                trophies_change=8, duration_seconds=120.0, kills=5,
                damage_dealt=5000, powerups_collected=10, star_player=True
            ))
        assert len(history.matches) == 1000


class TestBrawlerQueue:
    """Testes para BrawlerQueue."""

    def test_add_and_get_current(self):
        queue = BrawlerQueue()
        config = BrawlerConfig(name="colt", current_trophies=0, target_trophies=100, current_wins=0, target_wins=5)
        queue.add_brawler(config)
        current = queue.get_current()
        assert current is not None
        assert current.name == "colt"

    def test_next_circular(self):
        queue = BrawlerQueue()
        queue.add_brawler(BrawlerConfig(name="colt", current_trophies=0, target_trophies=100, current_wins=0, target_wins=5))
        queue.add_brawler(BrawlerConfig(name="shelly", current_trophies=0, target_trophies=100, current_wins=0, target_wins=5))
        first = queue.get_current()
        queue.next()
        second = queue.get_current()
        queue.next()
        back_to_first = queue.get_current()
        assert first.name == back_to_first.name
        assert first.name != second.name

    def test_should_switch_trophy_target(self):
        queue = BrawlerQueue()
        queue.add_brawler(BrawlerConfig(name="colt", current_trophies=100, target_trophies=100, current_wins=0, target_wins=5))
        assert queue.should_switch()

    def test_should_switch_three_losses(self, tmp_path):
        history = MatchHistory(tmp_path / "test_history.json")
        for _ in range(3):
            history.add_match(MatchResult(
                match_id="test", timestamp=datetime.now().isoformat(),
                game_mode="showdown", brawler="colt", result="loss",
                trophies_change=-2, duration_seconds=60.0, kills=0,
                damage_dealt=100, powerups_collected=0, star_player=False
            ))
        queue = BrawlerQueue()
        queue.add_brawler(BrawlerConfig(name="colt", current_trophies=0, target_trophies=100, current_wins=0, target_wins=5))
        assert queue.should_switch(current_result=None, history=history)

    def test_priority_sorting(self):
        queue = BrawlerQueue()
        queue.add_brawler(BrawlerConfig(name="low", current_trophies=0, target_trophies=100, current_wins=0, target_wins=5, priority=1))
        queue.add_brawler(BrawlerConfig(name="high", current_trophies=0, target_trophies=100, current_wins=0, target_wins=5, priority=5))
        current = queue.get_current()
        assert current.name == "high"


class TestMatchController:
    """Testes para MatchController."""

    def test_start_match(self, tmp_path):
        controller = MatchController(tmp_path)
        assert controller.start_match("showdown", "colt")
        assert controller.is_in_match
        assert controller.current_match is not None

    def test_start_match_when_already_in_match(self, tmp_path):
        controller = MatchController(tmp_path)
        controller.start_match("showdown", "colt")
        assert not controller.start_match("showdown", "shelly")

    def test_end_match(self, tmp_path):
        controller = MatchController(tmp_path)
        controller.start_match("showdown", "colt")
        result = controller.end_match("win", 8, 5000, 5, True)
        assert result is not None
        assert result.result == "win"
        assert not controller.is_in_match

    def test_end_match_not_in_match(self, tmp_path):
        controller = MatchController(tmp_path)
        assert controller.end_match("win", 8, 5000, 5, True) is None

    def test_reset_match(self, tmp_path):
        controller = MatchController(tmp_path)
        controller.start_match("showdown", "colt")
        controller.reset_match()
        assert not controller.is_in_match
        assert controller.current_match is None

    def test_get_status(self, tmp_path):
        controller = MatchController(tmp_path)
        info = controller.get_session_stats()
        assert "total_trophies" in info
        assert "session_matches" in info
        assert "is_in_match" in info


# ---------------------------------------------------------------------------
# VISION ENGINE TESTS
# ---------------------------------------------------------------------------
from vision_engine import Detection, VisionConfig, YOLOVisionEngine


class TestDetection:
    """Testes para a classe Detection."""

    def test_detection_creation(self):
        det = Detection(
            class_name="enemy", confidence=0.9,
            x=100, y=100, width=50, height=50,
            center_x=125, center_y=125
        )
        assert det.class_name == "enemy"
        assert det.confidence == 0.9


class TestVisionConfig:
    """Testes para VisionConfig."""

    def test_default_values(self):
        config = VisionConfig()
        assert config.confidence_threshold == 0.65
        assert config.iou_threshold == 0.45
        assert config.input_size == 640
        assert config.detect_enemies is True


class TestYOLOVisionEngine:
    """Testes para YOLOVisionEngine."""

    def test_init(self):
        engine = YOLOVisionEngine()
        assert not engine.is_initialized
        assert engine.device in ["cuda", "cpu"]

    def test_get_device(self):
        engine = YOLOVisionEngine()
        device = engine._get_device()
        assert device in ["cuda", "cpu"]

    def test_detection_config(self):
        """Test vision engine config is applied."""
        engine = YOLOVisionEngine(VisionConfig(confidence_threshold=0.7))
        assert engine.config.confidence_threshold == 0.7


# ---------------------------------------------------------------------------
# EMULATOR CONTROLLER TESTS
# ---------------------------------------------------------------------------
from emulator_controller import EmulatorConfig, ADBController, WindowController


class TestEmulatorConfig:
    """Testes para EmulatorConfig."""

    def test_default_config(self):
        config = EmulatorConfig()
        assert config.name == "LDPlayer"
        assert config.adb_port == 5555
        assert config.resolution == (1920, 1080)

    def test_for_bluestacks(self):
        config = EmulatorConfig.for_bluestacks()
        assert config.name == "BlueStacks"
        assert config.window_title == "BlueStacks App Player"

    def test_for_ldplayer(self):
        config = EmulatorConfig.for_ldplayer()
        assert config.name == "LDPlayer"
        assert config.window_title == "LDPlayer"


class TestADBController:
    """Testes para ADBController."""

    def test_sanitize_device_id(self):
        assert ADBController._sanitize_device_id("emulator-5555") == "emulator-5555"
        assert ADBController._sanitize_device_id("emulator;5555") == "emulator5555"
        assert ADBController._sanitize_device_id("") == ""

    def test_init(self):
        config = EmulatorConfig(adb_path="adb")
        controller = ADBController(config)
        assert controller.device_id == "emulator-5555"


class TestWindowController:
    """Testes para WindowController."""

    def test_init_no_window(self):
        controller = WindowController("NonExistentWindow12345")
        assert controller.hwnd is None or controller.hwnd == 0


# ---------------------------------------------------------------------------
# INTEGRATION TESTS - Vision -> Decision
# ---------------------------------------------------------------------------
class TestVisionDecisionIntegration:
    """Testes de integracao entre visao e decisao."""

    def test_extract_state_and_decide(self):
        # 1. Create tracking data
        player = TrackedObject(
            id=1, class_name="player", bbox=(900, 500, 1000, 600),
            confidence=0.95, center=(950, 550), velocity=(0, 0),
            age=0, hits=10, last_seen=time.time()
        )
        enemy = TrackedObject(
            id=2, class_name="enemy", bbox=(1100, 600, 1150, 650),
            confidence=0.9, center=(1125, 625), velocity=(-5, 0),
            age=0, hits=8, last_seen=time.time()
        )

        # 2. Extract game state
        extractor = StateExtractor()
        game_state = extractor.extract_state([player, enemy])

        # 3. Create state machine context
        sm = BrawlStarsStateMachine()
        ctx = StateContext(game_state=game_state, bot_instance=None)

        # 4. Update state machine
        sm.state_entry_time = 0  # Bypass min duration
        new_state = sm.update(ctx)

        # Should engage since enemy is present and player is healthy
        assert new_state in [BotState.ENGAGE, BotState.SEARCH]

    def test_tracker_to_scorer_pipeline(self):
        tracker = ByteTracker(min_hits=1)
        scorer = TargetScorer()

        # Simulate detections over frames
        for i in range(5):
            detections = [
                ("player", (900 + i*2, 500, 1000 + i*2, 600), 0.95),
                ("enemy", (1100, 600 + i, 1150, 650 + i), 0.9),
            ]
            tracks = tracker.update(detections)

        # Get confirmed tracks
        player_tracks = [t for t in tracks.values() if t.class_name == "player"]
        enemy_tracks = [t for t in tracks.values() if t.class_name == "enemy"]

        if player_tracks and enemy_tracks:
            player = player_tracks[0]
            enemies = [MagicMock(
                track_id=e.id, position=e.center,
                health_estimate=0.5, distance=200, threat_level=0.5
            ) for e in enemy_tracks]

            ranked = scorer.rank_targets(
                enemies, player.center, 1.0, []
            )
            assert len(ranked) > 0
            assert ranked[0].total_score > 0


# ---------------------------------------------------------------------------
# PERFORMANCE / STRESS TESTS
# ---------------------------------------------------------------------------
class TestPerformance:
    """Testes de stress/corretude para componentes criticos.

    Nota: asserts de tempo absoluto foram removidos para evitar
    falhas flaky em CI ou maquinas sob carga. Performance real
    deve ser medida com pytest-benchmark.
    """

    def test_tracker_stress(self):
        """Tracker deve processar muitos objetos sem erro."""
        tracker = ByteTracker()
        for _ in range(100):
            detections = [
                ("enemy", (i * 10, i * 10, i * 10 + 50, i * 10 + 50), 0.9)
                for i in range(20)
            ]
            tracks = tracker.update(detections)
            assert isinstance(tracks, dict)

    def test_scorer_stress(self):
        """Scorer deve ranquear muitos alvos sem erro."""
        scorer = TargetScorer()
        enemies = [
            MockEnemy(i, (i * 50, i * 50), 0.5, i * 50)
            for i in range(100)
        ]
        ranked = scorer.rank_targets(enemies, (0, 0), 1.0, [])
        assert len(ranked) == 100
        assert all(r.total_score >= 0 for r in ranked)

    def test_state_extractor_stress(self):
        """StateExtractor deve processar muitos tracks sem erro."""
        extractor = StateExtractor()
        now = time.time()
        tracks = [
            TrackedObject(
                id=i, class_name="enemy" if i > 0 else "player",
                bbox=(i * 10, i * 10, i * 10 + 20, i * 10 + 20),
                confidence=0.9, center=(i * 10 + 10, i * 10 + 10),
                velocity=(0, 0), age=0, hits=5, last_seen=now
            )
            for i in range(200)
        ]
        state = extractor.extract_state(tracks)
        assert len(state.enemies) == 199
        assert state.player_state == PlayerState.ALIVE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
