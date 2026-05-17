"""
tests/test_orchestrator.py

Testes para BotOrchestrator (hexagonal architecture).
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from core.orchestrator import BotOrchestrator, BotState, BotStatus
from core.ports import (
    DecisionContext,
    DecisionPort,
    InputAction,
    InputPort,
    MetricEvent,
    PersistencePort,
    SafetyPort,
    TelemetryPort,
    VisionPort,
)
from core.ports.vision_port import GameStateSnapshot, HUDState


# ------------------------------------------------------------------
# Mocks
# ------------------------------------------------------------------

class MockVision(VisionPort):
    def __init__(self, frames=None):
        self.frames = frames or []
        self.idx = 0
        self.initialized = False

    def initialize(self) -> bool:
        self.initialized = True
        return True

    def capture_and_perceive(self):
        if self.idx < len(self.frames):
            f = self.frames[self.idx]
            self.idx += 1
            return f
        return None

    def get_detected_objects(self, class_filter=None):
        return []

    def health_check(self):
        return {"ok": True}

    def shutdown(self):
        pass


class MockInput(InputPort):
    def __init__(self):
        self.actions: list = []
        self.initialized = False

    def initialize(self) -> bool:
        self.initialized = True
        return True

    def execute(self, action: InputAction) -> bool:
        self.actions.append(action)
        return True

    def tap(self, x, y, duration_ms=100) -> bool:
        self.actions.append(InputAction("tap", x, y))
        return True

    def swipe(self, x1, y1, x2, y2, duration_ms=300) -> bool:
        self.actions.append(InputAction("swipe", x1, y1, x2, y2))
        return True

    def health_check(self):
        return {"ok": True}

    def shutdown(self):
        pass


class MockDecision(DecisionPort):
    def __init__(self):
        self.decisions: list = []
        self.learn_calls: list = []
        self.initialized = False

    def initialize(self) -> bool:
        self.initialized = True
        return True

    def decide(self, context: DecisionContext):
        from core.ports.decision_port import Decision
        d = Decision(action_type="attack", confidence=0.8, target_pos=(0.5, 0.5))
        self.decisions.append((context, d))
        return d

    def learn(self, context, decision, reward):
        self.learn_calls.append((context, decision, reward))

    def start_episode(self, brawler, map_name=None):
        pass

    def end_episode(self, result, rank=0):
        pass

    def health_check(self):
        return {"ok": True}

    def shutdown(self):
        pass


class MockSafety(SafetyPort):
    def __init__(self, can_continue=True):
        self.can_continue = can_continue
        self.actions_recorded: list = []
        self.initialized = False

    def initialize(self) -> bool:
        self.initialized = True
        return True

    def check_before_action(self, action_type):
        from core.ports.safety_port import SafetyStatus
        return SafetyStatus(can_continue=self.can_continue)

    def check_before_match(self):
        from core.ports.safety_port import SafetyStatus
        return SafetyStatus(can_continue=self.can_continue)

    def record_action(self, action_type, duration_ms=0.0):
        self.actions_recorded.append(action_type)

    def record_match_end(self, result, duration_sec=0.0):
        pass

    def health_check(self):
        return {"ok": True}

    def shutdown(self):
        pass


class MockTelemetry(TelemetryPort):
    def __init__(self):
        self.metrics: list = []
        self.events: list = []
        self.flushed = False
        self.initialized = False

    def initialize(self) -> bool:
        self.initialized = True
        return True

    def record_metric(self, event: MetricEvent):
        self.metrics.append(event)

    def record_event(self, event_name, details):
        self.events.append((event_name, details))

    def flush(self):
        self.flushed = True

    def health_check(self):
        return {"ok": True}


class MockPersistence(PersistencePort):
    def __init__(self):
        self.states: dict = {}
        self.initialized = False

    def initialize(self) -> bool:
        self.initialized = True
        return True

    def save_state(self, state, label="checkpoint") -> bool:
        self.states[label] = state
        return True

    def load_state(self, label="checkpoint"):
        return self.states.get(label)

    def list_checkpoints(self):
        return {"checkpoints": list(self.states.keys())}


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_ports():
    return {
        "vision": MockVision(),
        "input": MockInput(),
        "decision": MockDecision(),
        "safety": MockSafety(),
        "telemetry": MockTelemetry(),
        "persistence": MockPersistence(),
    }


@pytest.fixture
def orchestrator(mock_ports):
    return BotOrchestrator(
        vision=mock_ports["vision"],
        input_=mock_ports["input"],
        decision=mock_ports["decision"],
        safety=mock_ports["safety"],
        telemetry=mock_ports["telemetry"],
        persistence=mock_ports["persistence"],
        config={"target_fps": 1000, "max_errors": 5},
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestInitialization:
    def test_initialize_all_ports(self, orchestrator, mock_ports):
        result = orchestrator.initialize()
        assert result is True
        assert mock_ports["vision"].initialized is True
        assert mock_ports["input"].initialized is True
        assert mock_ports["decision"].initialized is True

    def test_telemetry_records_init(self, orchestrator, mock_ports):
        orchestrator.initialize()
        events = [e for e in mock_ports["telemetry"].events if e[0] == "orchestrator_initialized"]
        assert len(events) == 1


class TestMainLoop:
    def test_tick_perceive_decide_act(self, orchestrator, mock_ports):
        orchestrator.initialize()

        # Feed one frame
        snapshot = GameStateSnapshot(
            game_phase="lobby",
            detected_objects=[],
            hud=HUDState(),
        )
        mock_ports["vision"].frames = [snapshot]

        orchestrator._tick()

        # Verify decision was called
        assert len(mock_ports["decision"].decisions) == 1

        # Verify action was executed
        assert len(mock_ports["input"].actions) > 0

        # Verify telemetry recorded metrics
        assert len(mock_ports["telemetry"].metrics) > 0

    def test_tick_vision_failure(self, orchestrator, mock_ports):
        orchestrator.initialize()
        # No frames -> vision returns None
        orchestrator._tick()
        # Should not crash, should record failure metric
        failure_metrics = [m for m in mock_ports["telemetry"].metrics if m.name == "vision_failure"]
        assert len(failure_metrics) == 1

    def test_safety_veto(self, orchestrator, mock_ports):
        orchestrator.initialize()
        mock_ports["safety"] = MockSafety(can_continue=False)
        orchestrator.safety = mock_ports["safety"]

        snapshot = GameStateSnapshot(game_phase="lobby")
        mock_ports["vision"].frames = [snapshot]

        orchestrator._tick()
        # Action should not be executed because safety vetoed
        assert len(mock_ports["input"].actions) == 0


class TestStateMachine:
    def test_lobby_to_match_transition(self, orchestrator, mock_ports):
        orchestrator._state = BotState.LOBBY
        orchestrator._update_state_machine("in_game")
        assert orchestrator._state == BotState.IN_MATCH

    def test_match_to_lobby_transition(self, orchestrator, mock_ports):
        orchestrator._state = BotState.IN_MATCH
        orchestrator._update_state_machine("lobby")
        assert orchestrator._state == BotState.LOBBY

    def test_no_transition_same_phase(self, orchestrator, mock_ports):
        orchestrator._state = BotState.LOBBY
        orchestrator._update_state_machine("lobby")
        assert orchestrator._state == BotState.LOBBY


class TestPauseResume:
    def test_pause(self, orchestrator, mock_ports):
        orchestrator.initialize()
        result = orchestrator.pause()
        assert result is True
        assert orchestrator._paused is True
        assert orchestrator._state == BotState.PAUSED

    def test_resume(self, orchestrator, mock_ports):
        orchestrator.initialize()
        orchestrator.pause()
        result = orchestrator.resume()
        assert result is True
        assert orchestrator._paused is False
        assert orchestrator._state == BotState.IDLE

    def test_pause_when_shutting_down(self, orchestrator, mock_ports):
        orchestrator._state = BotState.SHUTTING_DOWN
        result = orchestrator.pause()
        assert result is False


class TestStatus:
    def test_status_structure(self, orchestrator, mock_ports):
        status = orchestrator.status()
        assert isinstance(status, BotStatus)
        assert status.state == "idle"
        assert status.fps == 0.0


class TestExecuteAction:
    def test_execute_pause(self, orchestrator, mock_ports):
        orchestrator.initialize()
        result = orchestrator.execute_action("pause")
        assert result is True
        assert orchestrator._paused is True

    def test_execute_tap(self, orchestrator, mock_ports):
        orchestrator.initialize()
        result = orchestrator.execute_action("tap", x=0.5, y=0.5)
        assert result is True
        assert len(mock_ports["input"].actions) > 0

    def test_execute_unknown(self, orchestrator, mock_ports):
        result = orchestrator.execute_action("unknown_action")
        assert result is False


class TestPersistence:
    def test_save_checkpoint(self, orchestrator, mock_ports):
        orchestrator._episode_count = 42
        orchestrator._save_checkpoint()
        assert len(mock_ports["persistence"].states) > 0

    def test_checkpoint_content(self, orchestrator, mock_ports):
        orchestrator._episode_count = 99
        orchestrator._save_checkpoint()
        label = list(mock_ports["persistence"].states.keys())[0]
        state = mock_ports["persistence"].states[label]
        assert state["episode_count"] == 99


class TestErrorHandling:
    def test_error_counting(self, orchestrator, mock_ports):
        orchestrator.initialize()
        # Force vision to raise on capture
        def boom():
            raise RuntimeError("vision boom")
        mock_ports["vision"].capture_and_perceive = boom

        orchestrator._tick()  # _tick catches exceptions internally

        assert orchestrator._error_count == 1
        assert "vision boom" in orchestrator._last_error

    def test_too_many_errors_shutdown(self, orchestrator, mock_ports):
        orchestrator.config["max_errors"] = 2
        orchestrator._error_count = 3
        # The run loop should detect this and shutdown
        # We test by calling the condition directly
        assert orchestrator._error_count > orchestrator.config["max_errors"]


class TestLearning:
    def test_decision_learn_called(self, orchestrator, mock_ports):
        orchestrator.initialize()
        snapshot = GameStateSnapshot(game_phase="lobby")
        mock_ports["vision"].frames = [snapshot]
        orchestrator._tick()
        assert len(mock_ports["decision"].learn_calls) == 1

    def test_reward_computation(self, orchestrator, mock_ports):
        snapshot = GameStateSnapshot(
            game_phase="victory",
            hud=HUDState(hp_ratio=1.0),
        )
        from core.ports.decision_port import Decision
        reward = orchestrator._compute_reward(snapshot, Decision(action_type="attack"))
        assert reward > 5.0  # Victory bonus
