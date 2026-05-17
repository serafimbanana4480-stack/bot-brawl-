"""
tests/fixtures/port_doubles.py

Production-quality test doubles (fakes/mocks) for every Port interface.

These are NOT simple MagicMocks — they maintain realistic invariants
(e.g., state transitions, latency statistics, error injection) so that
orchestrator integration tests behave like real subsystems.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.ports import (
    DetectedObject,
    GameStateSnapshot,
    HUDState,
    Decision,
    DecisionContext,
    InputAction,
    MetricEvent,
    SafetyStatus,
    VisionPort,
    InputPort,
    DecisionPort,
    SafetyPort,
    TelemetryPort,
    PersistencePort,
)


# ------------------------------------------------------------------
# Vision double
# ------------------------------------------------------------------

@dataclass
class VisionScenario:
    """Predefined vision scenario for replay testing."""
    phase: str = "lobby"
    objects: List[DetectedObject] = field(default_factory=list)
    latency_ms: float = 35.0
    resolution: Tuple[int, int] = (1920, 1080)
    confidence: float = 0.9


class FakeVision(VisionPort):
    """
    Deterministic VisionPort double.

    Supports:
        - Scenario playback (queue of GameStateSnapshots)
        - Error injection (every Nth call returns None)
        - Latency simulation
    """

    def __init__(
        self,
        scenario: Optional[VisionScenario] = None,
        error_every_n: int = 0,
        latency_ms: float = 0.0,
    ):
        self.scenario = scenario or VisionScenario()
        self.error_every_n = error_every_n
        self.latency_ms = latency_ms
        self._call_count = 0
        self._last_snapshot: Optional[GameStateSnapshot] = None
        self._scenario_queue: List[GameStateSnapshot] = []

    def initialize(self) -> bool:
        return True

    def queue_scenarios(self, scenarios: List[GameStateSnapshot]) -> None:
        """Queue multiple snapshots for sequential playback."""
        self._scenario_queue = list(scenarios)

    def capture_and_perceive(self) -> Optional[GameStateSnapshot]:
        self._call_count += 1
        if self.error_every_n > 0 and self._call_count % self.error_every_n == 0:
            return None

        if self._scenario_queue:
            snapshot = self._scenario_queue.pop(0)
        else:
            img = np.zeros((*self.scenario.resolution[::-1], 3), dtype=np.uint8)
            snapshot = GameStateSnapshot(
                screenshot=img,
                detected_objects=list(self.scenario.objects),
                game_phase=self.scenario.phase,
                resolution=self.scenario.resolution,
                latency_ms=self.scenario.latency_ms,
                timestamp=time.time(),
                metadata={"state_confidence": self.scenario.confidence},
            )

        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        self._last_snapshot = snapshot
        return snapshot

    def get_detected_objects(self, class_filter: Optional[List[str]] = None) -> List[DetectedObject]:
        if self._last_snapshot is None:
            return []
        objects = self._last_snapshot.detected_objects
        if class_filter:
            objects = [o for o in objects if o.class_name in class_filter]
        return objects

    def health_check(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "call_count": self._call_count,
            "last_latency_ms": self._last_snapshot.latency_ms if self._last_snapshot else 0.0,
        }

    def shutdown(self) -> None:
        pass


# ------------------------------------------------------------------
# Input double
# ------------------------------------------------------------------

class FakeInput(InputPort):
    """
    Captures all executed actions for inspection.
    Supports failure injection.
    """

    def __init__(self, fail_every_n: int = 0):
        self.actions: List[InputAction] = []
        self.fail_every_n = fail_every_n
        self._call_count = 0

    def initialize(self) -> bool:
        return True

    def execute(self, action: InputAction) -> bool:
        self._call_count += 1
        if self.fail_every_n > 0 and self._call_count % self.fail_every_n == 0:
            return False
        self.actions.append(action)
        return True

    def tap(self, x: float, y: float, duration_ms: int = 100) -> bool:
        return self.execute(InputAction(action_type="tap", x=x, y=y, duration_ms=duration_ms))

    def swipe(self, x1: float, y1: float, x2: float, y2: float, duration_ms: int = 300) -> bool:
        return self.execute(InputAction(
            action_type="swipe", x=x1, y=y1, x2=x2, y2=y2, duration_ms=duration_ms
        ))

    def health_check(self) -> Dict[str, Any]:
        return {"ok": True, "actions_captured": len(self.actions)}

    def shutdown(self) -> None:
        pass


# ------------------------------------------------------------------
# Decision double
# ------------------------------------------------------------------

class FakeDecision(DecisionPort):
    """
    Preprogrammed decision engine.
    Returns decisions from a queue, or generates them from a rule.
    """

    def __init__(
        self,
        decisions: Optional[List[Decision]] = None,
        default_action: str = "attack",
        default_confidence: float = 0.8,
    ):
        self._queue = list(decisions) if decisions else []
        self.default_action = default_action
        self.default_confidence = default_confidence
        self.learn_calls: List[Tuple[DecisionContext, Decision, float]] = []

    def initialize(self) -> bool:
        return True

    def decide(self, context: DecisionContext) -> Decision:
        if self._queue:
            return self._queue.pop(0)
        return Decision(
            action_type=self.default_action,
            confidence=self.default_confidence,
            target_pos=(0.5, 0.5),
        )

    def learn(self, context: DecisionContext, decision: Decision, reward: float) -> None:
        self.learn_calls.append((context, decision, reward))

    def start_episode(self, brawler: str, map_name: Optional[str] = None) -> None:
        pass

    def end_episode(self, result: str, rank: int = 0) -> None:
        pass

    def health_check(self) -> Dict[str, Any]:
        return {"ok": True, "queue_remaining": len(self._queue)}

    def shutdown(self) -> None:
        pass


# ------------------------------------------------------------------
# Safety double
# ------------------------------------------------------------------

class FakeSafety(SafetyPort):
    """
    Configurable safety double.
    """

    def __init__(
        self,
        can_continue: bool = True,
        should_stop: bool = False,
        should_pause: bool = False,
    ):
        self._can_continue = can_continue
        self._should_stop = should_stop
        self._should_pause = should_pause
        self.actions_recorded: List[str] = []

    def initialize(self) -> bool:
        return True

    def check_before_action(self, action_type: str) -> SafetyStatus:
        return SafetyStatus(
            can_continue=self._can_continue,
            should_stop=self._should_stop,
            should_pause=self._should_pause,
        )

    def check_before_match(self) -> SafetyStatus:
        return SafetyStatus(
            can_continue=self._can_continue,
            should_stop=self._should_stop,
            should_pause=self._should_pause,
        )

    def record_action(self, action_type: str, duration_ms: float = 0.0) -> None:
        self.actions_recorded.append(action_type)

    def record_match_end(self, result: str, duration_sec: float = 0.0) -> None:
        pass

    def health_check(self) -> Dict[str, Any]:
        return {"ok": True}

    def shutdown(self) -> None:
        pass


# ------------------------------------------------------------------
# Telemetry double
# ------------------------------------------------------------------

class FakeTelemetry(TelemetryPort):
    """Captures all metrics and events for assertions."""

    def __init__(self):
        self.metrics: List[MetricEvent] = []
        self.events: List[Tuple[str, Dict[str, Any]]] = []
        self.flushed = False

    def initialize(self) -> bool:
        return True

    def record_metric(self, event: MetricEvent) -> None:
        self.metrics.append(event)

    def record_event(self, event_name: str, details: Dict[str, Any]) -> None:
        self.events.append((event_name, details))

    def flush(self) -> None:
        self.flushed = True

    def health_check(self) -> Dict[str, Any]:
        return {"ok": True, "metrics_count": len(self.metrics)}

    def shutdown(self) -> None:
        pass


# ------------------------------------------------------------------
# Persistence double
# ------------------------------------------------------------------

class FakePersistence(PersistencePort):
    """In-memory persistence for tests."""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self.save_count = 0
        self.load_count = 0

    def initialize(self) -> bool:
        return True

    def save_state(self, state: Dict[str, Any], label: str = "checkpoint") -> bool:
        self._data = dict(state)
        self.save_count += 1
        return True

    def load_state(self, label: str = "checkpoint") -> Optional[Dict[str, Any]]:
        self.load_count += 1
        return dict(self._data) if self._data else None

    def list_checkpoints(self) -> Dict[str, Any]:
        return {"checkpoints": ["checkpoint"] if self._data else []}

    def health_check(self) -> Dict[str, Any]:
        return {"ok": True, "has_data": bool(self._data)}

    def shutdown(self) -> None:
        pass
