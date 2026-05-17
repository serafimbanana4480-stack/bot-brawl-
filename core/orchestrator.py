"""
core/orchestrator.py

BotOrchestrator — Hexagonal architecture for Soberana Omega.

The orchestrator knows ONLY about Port interfaces (abstract).
It does NOT depend on:
    - YOLO/Ultralytics
    - ADB/Win32
    - Q-Learning/NeuralPolicy specifics
    - Screenshot mechanism
    - Any safety or telemetry backend

Ports (injected):
    - vision: VisionPort
    - input: InputPort
    - decision: DecisionPort
    - safety: SafetyPort
    - telemetry: TelemetryPort
    - persistence: PersistencePort

Usage (via factory):
    from core.factory import create_orchestrator
    bot = create_orchestrator(config)
    bot.run()          # blocking monitor loop
    bot.execute_action("pause")
    status = bot.status()
"""

from __future__ import annotations

import abc
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.opentelemetry_tracing import span, record_error
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

logger = logging.getLogger(__name__)


class BotState(Enum):
    """Internal orchestrator state machine."""
    IDLE = auto()
    INITIALIZING = auto()
    CONNECTING = auto()
    LOBBY = auto()
    MATCHMAKING = auto()
    IN_MATCH = auto()
    PAUSED = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()


@dataclass
class BotStatus:
    """Public status snapshot."""
    state: str = "idle"
    fps: float = 0.0
    cycle_time_ms: float = 0.0
    vision_latency_ms: float = 0.0
    decision_confidence: float = 0.0
    safety_ok: bool = True
    episode_count: int = 0
    last_error: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class BotOrchestrator:
    """
    Hexagonal orchestrator for the Brawl Stars bot.

    Responsibilities:
    1. Initialize all ports (graceful degradation if any fail)
    2. Run the main monitor loop (capture -> perceive -> decide -> act)
    3. Track state machine (lobby -> match -> combat -> lobby)
    4. Record telemetry
    5. Handle pause/resume/shutdown
    6. Delegate all domain logic to Ports
    """

    def __init__(
        self,
        vision: VisionPort,
        input_: InputPort,
        decision: DecisionPort,
        safety: SafetyPort,
        telemetry: TelemetryPort,
        persistence: PersistencePort,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.vision = vision
        self.input_ = input_
        self.decision = decision
        self.safety = safety
        self.telemetry = telemetry
        self.persistence = persistence
        self.config = config or {}

        # State machine
        self._state = BotState.IDLE
        self._running = False
        self._paused = False
        self._shutdown_requested = False
        self._lock = threading.Lock()

        # Metrics
        self._episode_count = 0
        self._frame_count = 0
        self._error_count = 0
        self._last_error = ""
        self._fps = 0.0
        self._cycle_time_ms = 0.0

        # Hooks
        self._shutdown_hooks: List[Callable] = []

        logger.info("[ORCHESTRATOR] Created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Initialize all ports with graceful degradation."""
        self._state = BotState.INITIALIZING
        success = True

        ports = [
            ("vision", self.vision),
            ("input", self.input_),
            ("decision", self.decision),
            ("safety", self.safety),
            ("telemetry", self.telemetry),
            ("persistence", self.persistence),
        ]

        for name, port in ports:
            try:
                port.initialize()
                logger.info(f"[ORCHESTRATOR] {name} initialized")
            except Exception as e:
                logger.error(f"[ORCHESTRATOR] {name} init failed: {e}")
                success = False

        if success:
            self._state = BotState.IDLE
            self.telemetry.record_event("orchestrator_initialized", {})
        else:
            self._state = BotState.ERROR
            self._last_error = "Some ports failed to initialize"

        return success

    def run(self) -> None:
        """Blocking main loop."""
        if self._state == BotState.ERROR:
            logger.error("[ORCHESTRATOR] Cannot run in ERROR state")
            return

        self._running = True
        self._state = BotState.CONNECTING
        logger.info("[ORCHESTRATOR] Main loop started")

        target_cycle_time = 1.0 / self.config.get("target_fps", 10)
        last_time = time.time()

        while self._running and not self._shutdown_requested:
            if self._paused:
                time.sleep(0.1)
                continue

            cycle_start = time.time()

            try:
                self._tick()
            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                logger.error(f"[ORCHESTRATOR] Tick error: {e}")
                if self._error_count > self.config.get("max_errors", 10):
                    logger.critical("[ORCHESTRATOR] Too many errors, shutting down")
                    self._shutdown_requested = True

            # FPS throttling
            elapsed = time.time() - cycle_start
            self._cycle_time_ms = elapsed * 1000
            if elapsed < target_cycle_time:
                time.sleep(target_cycle_time - elapsed)

            # FPS tracking
            self._frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                self._fps = self._frame_count / (now - last_time)
                self._frame_count = 0
                last_time = now

        self._shutdown()

    def stop(self) -> None:
        """Request graceful shutdown."""
        logger.info("[ORCHESTRATOR] Stop requested")
        self._shutdown_requested = True
        self._running = False

    def pause(self) -> bool:
        """Pause the main loop."""
        with self._lock:
            if self._state not in (BotState.PAUSED, BotState.SHUTTING_DOWN):
                self._paused = True
                self._state = BotState.PAUSED
                self.telemetry.record_event("orchestrator_paused", {})
                return True
        return False

    def resume(self) -> bool:
        """Resume the main loop."""
        with self._lock:
            if self._state == BotState.PAUSED:
                self._paused = False
                self._state = BotState.IDLE
                self.telemetry.record_event("orchestrator_resumed", {})
                return True
        return False

    # ------------------------------------------------------------------
    # Main tick (the core loop)
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """One iteration: perceive -> decide -> act -> learn."""
        try:
            self._do_tick()
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.error(f"[ORCHESTRATOR] Tick error: {e}")

    def _do_tick(self) -> None:
        """Actual tick logic (wrapped by _tick for error handling)."""
        with span("orchestrator.tick", {"frame": self._frame_count}) as tick_span:
            # 1. PERCEIVE
            with span("vision.capture") as vision_span:
                snapshot = self.vision.capture_and_perceive()
            if snapshot is None:
                self.telemetry.record_metric(MetricEvent("vision_failure", 1.0))
                if tick_span:
                    tick_span.set_attribute("vision_failure", True)
                return

            self.telemetry.record_metric(MetricEvent("vision_latency_ms", snapshot.latency_ms))
            if tick_span:
                tick_span.set_attribute("vision_latency_ms", snapshot.latency_ms)
                tick_span.set_attribute("game_phase", snapshot.game_phase)

            # 2. SAFETY CHECK (pre-decision)
            with span("safety.check") as safety_span:
                safety_status = self.safety.check_before_action("frame_tick")
            if not safety_status.can_continue:
                if safety_status.should_stop:
                    self._shutdown_requested = True
                    if safety_span:
                        safety_span.set_attribute("should_stop", True)
                    return
                if safety_status.should_pause:
                    self.pause()
                    if safety_span:
                        safety_span.set_attribute("should_pause", True)
                    return
                # Safety veto without stop/pause - skip this tick
                if safety_span:
                    safety_span.set_attribute("veto", True)
                return

            # 3. DECIDE
            with span("decision.decide") as decision_span:
                context = self._build_decision_context(snapshot)
                decision = self.decision.decide(context)
                if decision_span:
                    decision_span.set_attribute("confidence", decision.confidence)
                    decision_span.set_attribute("action_type", decision.action_type)

            self.telemetry.record_metric(MetricEvent("decision_confidence", decision.confidence))
            if decision.confidence < 0.3:
                self.telemetry.record_event("low_confidence_decision", {
                    "action": decision.action_type,
                    "confidence": decision.confidence,
                })

            # 4. ACT
            if decision.target_pos:
                with span("input.execute", {"action_type": decision.action_type}) as act_span:
                    action = InputAction(
                        action_type="tap",
                        x=decision.target_pos[0],
                        y=decision.target_pos[1],
                        duration_ms=100,
                    )
                    success = self.input_.execute(action)
                    if act_span:
                        act_span.set_attribute("success", success)
                self.safety.record_action(decision.action_type)
                if not success:
                    self.telemetry.record_metric(MetricEvent("input_failure", 1.0))

            # 5. LEARN (online RL)
            with span("decision.learn") as learn_span:
                reward = self._compute_reward(snapshot, decision)
                self.decision.learn(context, decision, reward)
                if learn_span:
                    learn_span.set_attribute("reward", reward)

            # 6. STATE MACHINE UPDATE
            self._update_state_machine(snapshot.game_phase)

            # 7. PERSISTENCE (periodic)
            if self._frame_count % 300 == 0:  # every ~30s at 10 FPS
                with span("persistence.save_checkpoint"):
                    self._save_checkpoint()

    # ------------------------------------------------------------------
    # State Machine
    # ------------------------------------------------------------------

    def _update_state_machine(self, game_phase: str) -> None:
        transitions = {
            (BotState.IDLE, "lobby"): BotState.LOBBY,
            (BotState.CONNECTING, "lobby"): BotState.LOBBY,
            (BotState.LOBBY, "in_game"): BotState.IN_MATCH,
            (BotState.LOBBY, "match_loading"): BotState.MATCHMAKING,
            (BotState.MATCHMAKING, "in_game"): BotState.IN_MATCH,
            (BotState.IN_MATCH, "lobby"): BotState.LOBBY,
            (BotState.IN_MATCH, "victory"): BotState.LOBBY,
            (BotState.IN_MATCH, "defeat"): BotState.LOBBY,
        }

        new_state = transitions.get((self._state, game_phase))
        if new_state and new_state != self._state:
            old = self._state.name
            self._state = new_state
            logger.info(f"[ORCHESTRATOR] State: {old} -> {new_state.name} (phase={game_phase})")
            self.telemetry.record_event("state_transition", {
                "from": old,
                "to": new_state.name,
                "game_phase": game_phase,
            })

            if new_state == BotState.IN_MATCH:
                self._episode_count += 1
                self.decision.start_episode(
                    brawler=self.config.get("brawler", "default"),
                    map_name=self.config.get("map", None),
                )
            elif self._state == BotState.LOBBY and old == "IN_MATCH":
                self.decision.end_episode(result="unknown")

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def _build_decision_context(self, snapshot) -> DecisionContext:
        enemies = [
            {"x": obj.center[0], "y": obj.center[1], "hp_ratio": obj.hp_ratio or 1.0}
            for obj in snapshot.detected_objects
            if obj.class_name == "enemy"
        ]
        return DecisionContext(
            player_hp=snapshot.hud.hp_ratio,
            player_pos=snapshot.player_pos,
            enemies=enemies,
            detected_objects=snapshot.detected_objects,
            hud_state=snapshot.hud,
            game_phase=snapshot.game_phase,
            match_time_remaining=snapshot.hud.match_time,
        )

    def _compute_reward(self, snapshot, decision) -> float:
        """Heuristic reward for online learning."""
        reward = 0.0
        # Survival
        reward += snapshot.hud.hp_ratio * 0.1
        # Combat
        if decision.action_type in ("attack", "super"):
            reward += 0.5
        # Objective
        if snapshot.game_phase in ("victory",):
            reward += 10.0
        elif snapshot.game_phase in ("defeat",):
            reward -= 5.0
        return reward

    # ------------------------------------------------------------------
    # Status / Introspection
    # ------------------------------------------------------------------

    def status(self) -> BotStatus:
        return BotStatus(
            state=self._state.name.lower(),
            fps=self._fps,
            cycle_time_ms=self._cycle_time_ms,
            vision_latency_ms=self.vision.health_check().get("last_latency_ms", 0.0),
            decision_confidence=self.decision.health_check().get("last_confidence", 0.0),
            safety_ok=self.safety.health_check().get("can_continue", True),
            episode_count=self._episode_count,
            last_error=self._last_error,
            metrics={
                "error_count": self._error_count,
                "frame_count": self._frame_count,
            },
        )

    def execute_action(self, action_name: str, **kwargs) -> bool:
        """Execute a manual action (pause, resume, tap, etc.)."""
        if action_name == "pause":
            return self.pause()
        elif action_name == "resume":
            return self.resume()
        elif action_name == "stop":
            self.stop()
            return True
        elif action_name == "tap":
            x, y = kwargs.get("x", 0.5), kwargs.get("y", 0.5)
            return self.input_.tap(x, y)
        elif action_name == "status":
            return True
        else:
            logger.warning(f"[ORCHESTRATOR] Unknown action: {action_name}")
            return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        state = {
            "episode_count": self._episode_count,
            "error_count": self._error_count,
            "config": self.config,
            "timestamp": time.time(),
        }
        self.persistence.save_state(state, label=f"auto_{int(time.time())}")

    def _shutdown(self) -> None:
        self._state = BotState.SHUTTING_DOWN
        logger.info("[ORCHESTRATOR] Shutting down...")

        for hook in self._shutdown_hooks:
            try:
                hook()
            except Exception as e:
                logger.warning(f"[ORCHESTRATOR] Shutdown hook error: {e}")

        for port in [self.vision, self.input_, self.decision, self.safety, self.telemetry, self.persistence]:
            try:
                port.shutdown()
            except Exception:
                pass

        self.telemetry.record_event("orchestrator_shutdown", {})
        self.telemetry.flush()
        logger.info("[ORCHESTRATOR] Shutdown complete")

    def register_shutdown_hook(self, hook: Callable) -> None:
        self._shutdown_hooks.append(hook)
