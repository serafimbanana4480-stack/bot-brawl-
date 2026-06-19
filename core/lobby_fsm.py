"""
core/lobby_fsm.py

Hierarchical Lobby FSM + Repetition Guard for Brawl Stars bot.

DEPRECATED: Use core.orchestrator.BotOrchestrator instead.
This module is kept for backward compatibility only.

Repetition Guard has been extracted to core.repetition_guard.
"""

import logging
import threading
import time
import warnings
from dataclasses import dataclass
from enum import Enum

from core.repetition_guard import ActionRecord, RepetitionGuard

logger = logging.getLogger(__name__)

warnings.warn(
    "Deprecated: use core.orchestrator.BotOrchestrator instead",
    DeprecationWarning,
    stacklevel=2,
)


# ===== Backward-compatible re-exports =====
# RepetitionGuard and ActionRecord now live in core.repetition_guard.
__all__ = [
    "LobbyState", "LoadingState", "InGameState", "PostGameState",
    "TopLevelState", "FSMState", "HierarchicalFSM",
    "RepetitionGuard", "ActionRecord", "LobbySystem",
]

# ===== Hierarchical FSM =====

class LobbyState(Enum):
    """Top-level lobby states."""
    IDLE = "idle"
    PLAY_BUTTON = "play_button"
    EVENT_SELECT = "event_select"
    BRAWLER_SELECT = "brawler_select"
    CONFIRM = "confirm"
    POPUP = "popup"           # Handling popups (news, shop, etc.)
    SETTINGS = "settings"     # Settings menu


class LoadingState(Enum):
    """Loading sub-states."""
    MATCHMAKING = "matchmaking"
    LOADING_SCREEN = "loading_screen"


class InGameState(Enum):
    """In-game sub-states."""
    SPAWNING = "spawning"
    COMBAT = "combat"
    FARMING = "farming"
    RETREATING = "retreating"
    DYING = "dying"
    RESPAWNING = "respawning"


class PostGameState(Enum):
    """Post-game sub-states."""
    VICTORY = "victory"
    DEFEAT = "defeat"
    STAR_TOKENS = "star_tokens"
    RESULTS = "results"


class TopLevelState(Enum):
    """Top-level game states."""
    LOBBY = "lobby"
    LOADING = "loading"
    IN_GAME = "in_game"
    POST_GAME = "post_game"
    UNKNOWN = "unknown"


@dataclass
class FSMState:
    """Current FSM state with timing info."""
    top: TopLevelState = TopLevelState.UNKNOWN
    sub: Enum | None = None
    entered_at: float = 0.0
    transitions: int = 0


class HierarchicalFSM:
    """
    Hierarchical finite state machine for game state management.

    Provides:
    - Nested state tracking (top-level + sub-state)
    - State duration tracking
    - Transition validation
    - Stale state detection (stuck in one state too long)
    - Automatic recovery from stuck states
    """

    # Maximum time allowed in each state before considering it stuck
    STATE_TIMEOUTS = {
        LobbyState.IDLE: 30.0,
        LobbyState.PLAY_BUTTON: 10.0,
        LobbyState.EVENT_SELECT: 15.0,
        LobbyState.BRAWLER_SELECT: 15.0,
        LobbyState.CONFIRM: 10.0,
        LobbyState.POPUP: 10.0,
        LoadingState.MATCHMAKING: 60.0,
        LoadingState.LOADING_SCREEN: 30.0,
        InGameState.SPAWNING: 5.0,
        InGameState.COMBAT: 300.0,   # Long timeout for combat
        InGameState.FARMING: 300.0,
        InGameState.RETREATING: 10.0,
        InGameState.DYING: 5.0,
        InGameState.RESPAWNING: 5.0,
        PostGameState.VICTORY: 10.0,
        PostGameState.DEFEAT: 10.0,
        PostGameState.STAR_TOKENS: 5.0,
        PostGameState.RESULTS: 10.0,
    }

    def __init__(self):
        self._state = FSMState(entered_at=time.time())
        self._lock = threading.RLock()
        self._transition_history: list[dict] = []

        logger.info("[HFSM] Initialized")

    def transition(self, top: TopLevelState, sub: Enum | None = None,
                   reason: str = ""):
        """
        Transition to a new state.

        Args:
            top: Top-level state
            sub: Sub-state (specific enum value)
            reason: Why the transition happened
        """
        now = time.time()
        with self._lock:
            old_top = self._state.top
            old_sub = self._state.sub

            self._state = FSMState(
                top=top,
                sub=sub,
                entered_at=now,
                transitions=self._state.transitions + 1,
            )

            # Record transition
            self._transition_history.append({
                "from_top": old_top.value,
                "from_sub": old_sub.value if old_sub else None,
                "to_top": top.value,
                "to_sub": sub.value if sub else None,
                "reason": reason,
                "timestamp": now,
            })
            if len(self._transition_history) > 100:
                self._transition_history = self._transition_history[-100:]

            logger.debug("[HFSM] %s/%s → %s/%s (%s)",
                         old_top.value, old_sub.value if old_sub else "-",
                         top.value, sub.value if sub else "-",
                         reason)

    def get_state(self) -> FSMState:
        """Get current state."""
        with self._lock:
            return self._state

    def is_stuck(self) -> bool:
        """Check if we've been in the current state too long."""
        with self._lock:
            now = time.time()
            duration = now - self._state.entered_at

            timeout = self.STATE_TIMEOUTS.get(self._state.sub, 60.0)
            return duration > timeout

    def get_state_duration(self) -> float:
        """Get how long we've been in the current state (seconds)."""
        with self._lock:
            return time.time() - self._state.entered_at

    def get_transition_history(self, limit: int = 20) -> list[dict]:
        """Get recent transition history."""
        with self._lock:
            return self._transition_history[-limit:]


# ===== Combined Lobby System =====

class LobbySystem:
    """
    Combined hierarchical FSM + repetition guard for lobby management.

    Provides a unified interface for:
    - State tracking and transitions
    - Action validation (cooldown + loop detection)
    - Stuck state recovery
    - Statistics
    """

    def __init__(self):
        self.fsm = HierarchicalFSM()
        self.guard = RepetitionGuard()
        self._recovery_attempts = 0
        self._max_recovery_attempts = 5

        logger.info("[LOBBY_SYSTEM] Initialized")

    def can_act(self, action: str, target: str = "") -> tuple[bool, str]:
        """Check if an action can be performed."""
        return self.guard.can_execute(action, target)

    def execute(self, action: str, target: str = "",
                result: str = "success"):
        """Record an executed action."""
        self.guard.record_action(action, target, result)

    def transition(self, top: TopLevelState, sub: Enum | None = None,
                   reason: str = ""):
        """Transition to a new state."""
        self.fsm.transition(top, sub, reason)
        # Reset loop detection on state change
        if self.guard.is_loop_detected():
            self.guard.reset_loop()

    def check_stuck(self) -> str | None:
        """
        Check if the system is stuck and return a recovery action.

        Returns None if not stuck, or a recovery action string.
        """
        if self.fsm.is_stuck():
            self._recovery_attempts += 1
            if self._recovery_attempts > self._max_recovery_attempts:
                # Force reset to lobby
                self.transition(TopLevelState.LOBBY, LobbyState.IDLE,
                               "forced_reset")
                self._recovery_attempts = 0
                return "force_reset_to_lobby"

            state = self.fsm.get_state()
            self.fsm.get_state_duration()

            # State-specific recovery
            if state.sub == LobbyState.PLAY_BUTTON:
                return "try_alternative_play_location"
            elif state.sub == LobbyState.BRAWLER_SELECT:
                return "select_random_brawler"
            elif state.sub == LoadingState.MATCHMAKING:
                return "cancel_and_retry"
            elif state.sub == LobbyState.POPUP:
                return "close_popup_esc"
            else:
                return "tap_center_screen"

        if self.guard.is_loop_detected():
            self._recovery_attempts += 1
            # Loop recovery: go back to idle
            self.transition(TopLevelState.LOBBY, LobbyState.IDLE, "loop_recovery")
            self.guard.reset_loop()
            return "loop_recovery_reset"

        self._recovery_attempts = 0
        return None

    def get_state_info(self) -> dict:
        """Get comprehensive state information."""
        return {
            "fsm": {
                "top": self.fsm.get_state().top.value,
                "sub": self.fsm.get_state().sub.value if self.fsm.get_state().sub else None,
                "duration_s": round(self.fsm.get_state_duration(), 1),
                "is_stuck": self.fsm.is_stuck(),
            },
            "guard": self.guard.get_stats(),
            "loop": self.guard.get_loop_info(),
            "recovery_attempts": self._recovery_attempts,
        }
