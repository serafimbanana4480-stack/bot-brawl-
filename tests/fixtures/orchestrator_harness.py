"""
tests/fixtures/orchestrator_harness.py

Headless orchestrator harness for enterprise integration testing.

Provides a pre-wired BotOrchestrator with all fakes, plus helpers for:
    - Scenario playback (vision snapshots queued sequentially)
    - Action assertion (verify what the bot did)
    - Error injection (test resilience)
    - Performance benchmarking (cycle latency stats)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.orchestrator import BotOrchestrator
from core.ports import GameStateSnapshot, HUDState
from tests.fixtures.port_doubles import (
    FakeVision,
    FakeInput,
    FakeDecision,
    FakeSafety,
    FakeTelemetry,
    FakePersistence,
    VisionScenario,
)


@dataclass
class HarnessResult:
    """Result of a harness run."""
    ticks_completed: int
    actions_executed: int
    errors: int
    avg_cycle_ms: float
    final_state: str
    telemetry_metrics: int
    telemetry_events: int
    checkpoint_saves: int
    decisions_learned: int


class OrchestratorHarness:
    """
    Headless test harness for the BotOrchestrator.

    Usage:
        harness = OrchestratorHarness()
        harness.queue_phases(["lobby", "lobby", "in_game", "in_game", "lobby"])
        result = harness.run(ticks=10)
        assert result.final_state == "LOBBY"
        assert result.actions_executed > 0
    """

    def __init__(self, vision_confidence: float = 0.9):
        self.vision = FakeVision(
            scenario=VisionScenario(confidence=vision_confidence)
        )
        self.input_ = FakeInput()
        self.decision = FakeDecision()
        self.safety = FakeSafety()
        self.telemetry = FakeTelemetry()
        self.persistence = FakePersistence()

        self.orchestrator = BotOrchestrator(
            vision=self.vision,
            input_=self.input_,
            decision=self.decision,
            safety=self.safety,
            telemetry=self.telemetry,
            persistence=self.persistence,
            config={},
        )
        self._latencies: List[float] = []

    def queue_phases(self, phases: List[str]) -> None:
        """Queue a sequence of game phases as snapshots."""
        snapshots = []
        for phase in phases:
            snapshots.append(GameStateSnapshot(
                screenshot=None,
                game_phase=phase,
                metadata={"state_confidence": self.vision.scenario.confidence},
            ))
        self.vision.queue_scenarios(snapshots)

    def queue_scenarios(self, scenarios: List[VisionScenario]) -> None:
        """Queue rich vision scenarios."""
        snapshots = []
        for s in scenarios:
            snapshots.append(GameStateSnapshot(
                screenshot=None,
                detected_objects=list(s.objects),
                hud=HUDState(),
                game_phase=s.phase,
                latency_ms=s.latency_ms,
                resolution=s.resolution,
                metadata={"state_confidence": s.confidence},
            ))
        self.vision.queue_scenarios(snapshots)

    def run(self, ticks: int = 10) -> HarnessResult:
        """Run the orchestrator for N ticks and return aggregated results."""
        self.orchestrator.initialize()

        for _ in range(ticks):
            t0 = time.time()
            self.orchestrator._tick()
            self._latencies.append((time.time() - t0) * 1000)

        status = self.orchestrator.status()
        return HarnessResult(
            ticks_completed=ticks,
            actions_executed=len(self.input_.actions),
            errors=getattr(status, "error_count", 0),
            avg_cycle_ms=sum(self._latencies) / len(self._latencies) if self._latencies else 0.0,
            final_state=getattr(status, "state", "UNKNOWN"),
            telemetry_metrics=len(self.telemetry.metrics),
            telemetry_events=len(self.telemetry.events),
            checkpoint_saves=self.persistence.save_count,
            decisions_learned=len(self.decision.learn_calls),
        )

    def inject_vision_failure(self, every_n: int = 3) -> None:
        """Make vision fail periodically."""
        self.vision.error_every_n = every_n

    def inject_input_failure(self, every_n: int = 5) -> None:
        """Make input execution fail periodically."""
        self.input_.fail_every_n = every_n

    def inject_safety_veto(self) -> None:
        """Make safety veto every action."""
        self.safety._can_continue = False

    def set_safety_pause(self) -> None:
        """Make safety pause the bot."""
        self.safety._should_pause = True

    def set_safety_stop(self) -> None:
        """Make safety stop the bot."""
        self.safety._should_stop = True

    def reset(self) -> None:
        """Reset all fakes to clean state."""
        self.vision._call_count = 0
        self.vision._last_snapshot = None
        self.vision._scenario_queue = []
        self.input_.actions = []
        self.input_._call_count = 0
        self.telemetry.metrics = []
        self.telemetry.events = []
        self.telemetry.flushed = False
        self.persistence._data = {}
        self.persistence.save_count = 0
        self.persistence.load_count = 0
        self.decision.learn_calls = []
        self._latencies = []
        self.orchestrator._error_count = 0
        self.orchestrator._state = self.orchestrator._state.IDLE
