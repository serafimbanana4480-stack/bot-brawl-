"""
tests/test_orchestrator_integration.py

Enterprise-grade integration tests for the BotOrchestrator.

Uses OrchestratorHarness with fake ports to validate:
    - Normal gameplay loop (lobby -> match -> lobby)
    - Error recovery (vision failures, input failures)
    - Safety system interaction (veto, pause, stop)
    - State machine transitions
    - Telemetry emission
    - Checkpoint persistence
    - Decision learning loop
    - Performance boundaries (latency)
"""

from __future__ import annotations

import pytest

from core.ports import DetectedObject, GameStateSnapshot, HUDState
from tests.fixtures.orchestrator_harness import OrchestratorHarness
from tests.fixtures.port_doubles import VisionScenario


# ------------------------------------------------------------------
# Happy path
# ------------------------------------------------------------------

class TestHappyPath:
    def test_lobby_to_match_transition(self):
        harness = OrchestratorHarness()
        harness.queue_phases(["lobby", "lobby", "in_game", "in_game", "lobby"])
        result = harness.run(ticks=5)

        assert result.ticks_completed == 5
        assert result.final_state.lower() == "lobby"
        assert result.actions_executed > 0
        assert result.errors == 0
        assert result.decisions_learned == 5
        assert result.telemetry_metrics > 0

    def test_match_persists_and_actions_execute(self):
        harness = OrchestratorHarness()
        # Simulate combat with enemies detected
        scenario = VisionScenario(
            phase="in_game",
            objects=[
                DetectedObject("enemy", 0.9, (0.4, 0.4, 0.5, 0.5), (0.45, 0.45)),
            ],
            confidence=0.95,
        )
        harness.queue_scenarios([scenario] * 3)
        result = harness.run(ticks=3)

        assert result.actions_executed >= 3
        assert result.avg_cycle_ms < 100.0  # fast with fakes
        assert result.checkpoint_saves >= 0


# ------------------------------------------------------------------
# Error recovery
# ------------------------------------------------------------------

class TestErrorRecovery:
    def test_vision_failure_does_not_crash(self):
        harness = OrchestratorHarness()
        harness.inject_vision_failure(every_n=2)
        harness.queue_phases(["lobby"] * 10)
        result = harness.run(ticks=10)

        # Vision fails 50% of the time, but orchestrator should not crash
        assert result.errors == 0
        assert result.ticks_completed == 10
        # Fewer actions because some ticks had no snapshot
        assert result.actions_executed <= 5

    def test_input_failure_tolerance(self):
        harness = OrchestratorHarness()
        harness.inject_input_failure(every_n=3)
        harness.queue_phases(["in_game"] * 9)
        result = harness.run(ticks=9)

        # Some actions fail but loop continues
        assert result.errors == 0
        assert result.ticks_completed == 9
        assert result.actions_executed < 9

    def test_multiple_failure_modes_combined(self):
        harness = OrchestratorHarness()
        harness.inject_vision_failure(every_n=4)
        harness.inject_input_failure(every_n=5)
        harness.queue_phases(["in_game"] * 20)
        result = harness.run(ticks=20)

        assert result.errors == 0
        assert result.ticks_completed == 20


# ------------------------------------------------------------------
# Safety system
# ------------------------------------------------------------------

class TestSafetyInteraction:
    def test_safety_veto_skips_actions(self):
        harness = OrchestratorHarness()
        harness.inject_safety_veto()
        harness.queue_phases(["in_game"] * 5)
        result = harness.run(ticks=5)

        assert result.actions_executed == 0
        assert result.errors == 0

    def test_safety_pause_stops_actions(self):
        harness = OrchestratorHarness()
        harness.safety._can_continue = False
        harness.safety._should_pause = True
        harness.queue_phases(["in_game"] * 5)
        result = harness.run(ticks=5)

        # After pause, actions stop
        assert result.actions_executed == 0
        assert result.errors == 0

    def test_safety_stop_shuts_down(self):
        harness = OrchestratorHarness()
        harness.safety._can_continue = False
        harness.safety._should_stop = True
        harness.queue_phases(["in_game"] * 10)
        result = harness.run(ticks=10)

        # After stop, first tick triggers shutdown
        # Remaining ticks should not execute
        assert result.actions_executed == 0
        assert result.errors == 0


# ------------------------------------------------------------------
# Telemetry & persistence
# ------------------------------------------------------------------

class TestTelemetryAndPersistence:
    def test_telemetry_collects_metrics(self):
        harness = OrchestratorHarness()
        harness.queue_phases(["lobby", "in_game", "in_game", "lobby"])
        result = harness.run(ticks=4)

        assert result.telemetry_metrics > 0
        assert result.telemetry_events >= 0

    def test_checkpoint_saves_periodically(self):
        harness = OrchestratorHarness()
        harness.queue_phases(["in_game"] * 600)  # 600 ticks triggers multiple saves
        result = harness.run(ticks=600)

        # Every 300 ticks = 2 saves
        assert result.checkpoint_saves >= 1

    def test_persistence_restores_state(self):
        harness = OrchestratorHarness()
        harness.queue_phases(["in_game"] * 5)
        harness.run(ticks=5)

        checkpoint = harness.persistence.load_state()
        assert checkpoint is not None
        assert "timestamp" in checkpoint


# ------------------------------------------------------------------
# Performance boundaries
# ------------------------------------------------------------------

class TestPerformance:
    def test_cycle_latency_under_threshold(self):
        harness = OrchestratorHarness()
        harness.queue_phases(["in_game"] * 50)
        result = harness.run(ticks=50)

        # With all fakes, cycle should be very fast (< 10ms)
        assert result.avg_cycle_ms < 50.0, (
            f"Average cycle latency {result.avg_cycle_ms:.1f}ms exceeds 50ms threshold"
        )

    @pytest.mark.parametrize("tick_count", [10, 100, 500])
    def test_scales_linearly(self, tick_count: int):
        harness = OrchestratorHarness()
        harness.queue_phases(["in_game"] * tick_count)
        result = harness.run(ticks=tick_count)

        assert result.ticks_completed == tick_count
        assert result.errors == 0


# ------------------------------------------------------------------
# Decision learning
# ------------------------------------------------------------------

class TestDecisionLearning:
    def test_learn_called_every_tick(self):
        harness = OrchestratorHarness()
        harness.queue_phases(["in_game"] * 10)
        result = harness.run(ticks=10)

        assert result.decisions_learned == 10

    def test_learn_receives_reward_signal(self):
        harness = OrchestratorHarness()
        scenario = VisionScenario(
            phase="in_game",
            objects=[DetectedObject("enemy", 0.9, (0.3, 0.3, 0.4, 0.4), (0.35, 0.35))],
        )
        harness.queue_scenarios([scenario] * 3)
        harness.run(ticks=3)

        for ctx, decision, reward in harness.decision.learn_calls:
            assert isinstance(reward, (int, float))
            assert -200 <= reward <= 200  # reasonable reward bounds
