"""
core/lobby_fsm.py

Hierarchical Lobby FSM + Repetition Guard for Brawl Stars bot.

Solves two problems:
1. "No hierarchical lobby FSM" — lobby_navigator is blocked by .gitignore
   and the existing state machine is flat (no sub-states)
2. "No repetition guard" — the bot can get stuck in loops
   (clicking the same button repeatedly, retrying the same action)

Hierarchical FSM:
  LOBBY
    ├── IDLE (waiting for input)
    ├── PLAY_BUTTON (finding/clicking play)
    ├── EVENT_SELECT (selecting game mode)
    ├── BRAWLER_SELECT (choosing brawler)
    └── CONFIRM (confirming selection)
  
  LOADING
    ├── MATCHMAKING (searching for match)
    └── LOADING_SCREEN (map loading)
  
  IN_GAME
    ├── SPAWNING (just spawned)
    ├── COMBAT (fighting enemies)
    ├── FARMING (collecting cubes)
    └── DYING (about to die / respawning)
  
  POST_GAME
    ├── VICTORY
    ├── DEFEAT
    ├── STAR_TOKENS
    └── RESULTS_SCREEN

Repetition Guard:
  - Tracks recent actions and their timestamps
  - Blocks repeated identical actions within a cooldown
  - Detects loops (same sequence of actions repeating)
  - Forces state reset when loop detected
  - Provides escape hatches for stuck states
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


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
    sub: Optional[Enum] = None
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
        self._transition_history: List[Dict] = []

        logger.info("[HFSM] Initialized")

    def transition(self, top: TopLevelState, sub: Optional[Enum] = None,
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

    def get_transition_history(self, limit: int = 20) -> List[Dict]:
        """Get recent transition history."""
        with self._lock:
            return self._transition_history[-limit:]


# ===== Repetition Guard =====

@dataclass
class ActionRecord:
    """Record of a performed action."""
    action: str
    target: str = ""         # What was clicked/interacted with
    timestamp: float = 0.0
    result: str = ""         # "success", "fail", "timeout"


class RepetitionGuard:
    """
    Guards against repetitive actions and stuck loops.
    
    Features:
    - Action cooldown: prevents repeating the same action too quickly
    - Loop detection: detects when the same sequence of actions repeats
    - Stuck recovery: forces state reset when loop detected
    - Action history tracking
    """

    # Minimum time between identical actions (seconds)
    DEFAULT_COOLDOWN = 2.0

    # Action-specific cooldowns
    ACTION_COOLDOWNS = {
        "click_play": 3.0,
        "select_brawler": 2.0,
        "click_confirm": 2.0,
        "close_popup": 1.5,
        "tap_screen": 0.5,
        "swipe": 0.3,
    }

    # Loop detection: if this many identical sequences occur, it's a loop
    LOOP_THRESHOLD = 3

    # Maximum sequence length to check for loops
    MAX_SEQUENCE_CHECK = 6

    def __init__(self):
        self._action_history: List[ActionRecord] = []
        self._cooldown_timers: Dict[str, float] = {}
        self._loop_detected = False
        self._loop_count = 0
        self._lock = threading.RLock()

        logger.info("[REPETITION_GUARD] Initialized")

    def can_execute(self, action: str, target: str = "") -> Tuple[bool, str]:
        """
        Check if an action can be executed (not on cooldown, not in a loop).
        
        Returns (can_execute, reason).
        """
        with self._lock:
            # Check cooldown
            cooldown = self.ACTION_COOLDOWNS.get(action, self.DEFAULT_COOLDOWN)
            key = f"{action}:{target}" if target else action
            last_time = self._cooldown_timers.get(key, 0)
            
            now = time.time()
            if now - last_time < cooldown:
                remaining = cooldown - (now - last_time)
                return (False, f"cooldown:{remaining:.1f}s")

            # Check for active loop
            if self._loop_detected:
                return (False, "loop_detected")

            return (True, "ok")

    def record_action(self, action: str, target: str = "",
                      result: str = "success"):
        """
        Record that an action was performed.
        
        Args:
            action: Action name
            target: Target identifier
            result: "success", "fail", "timeout"
        """
        now = time.time()
        
        with self._lock:
            # Update cooldown timer
            key = f"{action}:{target}" if target else action
            self._cooldown_timers[key] = now

            # Record action
            self._action_history.append(ActionRecord(
                action=action,
                target=target,
                timestamp=now,
                result=result,
            ))
            if len(self._action_history) > 200:
                self._action_history = self._action_history[-200:]

            # Check for loops
            if result == "fail" or result == "timeout":
                self._check_for_loops()

    def force_cooldown(self, action: str, cooldown_seconds: float):
        """Force a specific cooldown for an action."""
        with self._lock:
            self._cooldown_timers[action] = time.time()
            self.ACTION_COOLDOWNS[action] = cooldown_seconds

    def reset_loop(self):
        """Reset loop detection state."""
        with self._lock:
            self._loop_detected = False
            self._loop_count = 0
            logger.info("[REPETITION_GUARD] Loop state reset")

    def is_loop_detected(self) -> bool:
        """Check if a loop is currently detected."""
        with self._lock:
            return self._loop_detected

    def get_loop_info(self) -> Dict:
        """Get information about detected loops."""
        with self._lock:
            return {
                "loop_detected": self._loop_detected,
                "loop_count": self._loop_count,
                "recent_actions": [
                    {"action": a.action, "target": a.target, "result": a.result}
                    for a in self._action_history[-10:]
                ],
            }

    def get_stats(self) -> Dict:
        """Get guard statistics."""
        with self._lock:
            total = len(self._action_history)
            failures = sum(1 for a in self._action_history if a.result != "success")
            
            return {
                "total_actions": total,
                "failures": failures,
                "failure_rate": round(failures / max(1, total) * 100, 1),
                "loops_detected": self._loop_count,
                "active_cooldowns": len(self._cooldown_timers),
            }

    # --- Internal ---

    def _check_for_loops(self):
        """Check recent action history for repeating sequences."""
        if len(self._action_history) < 4:
            return

        # Get recent failed actions
        recent = self._action_history[-self.MAX_SEQUENCE_CHECK * 2:]
        
        # Try different sequence lengths
        for seq_len in range(2, self.MAX_SEQUENCE_CHECK + 1):
            if len(recent) < seq_len * self.LOOP_THRESHOLD:
                continue

            # Get the last sequence
            last_seq = [(a.action, a.target) for a in recent[-seq_len:]]
            
            # Check if this sequence repeats
            matches = 0
            for i in range(len(recent) - seq_len, -1, -seq_len):
                check_seq = [(a.action, a.target) for a in recent[i:i + seq_len]]
                if check_seq == last_seq:
                    matches += 1
                else:
                    break

            if matches >= self.LOOP_THRESHOLD:
                self._loop_detected = True
                self._loop_count += 1
                logger.warning(
                    "[REPETITION_GUARD] Loop detected! Sequence of %d actions "
                    "repeated %d times: %s",
                    seq_len, matches, last_seq
                )
                return


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

    def can_act(self, action: str, target: str = "") -> Tuple[bool, str]:
        """Check if an action can be performed."""
        return self.guard.can_execute(action, target)

    def execute(self, action: str, target: str = "",
                result: str = "success"):
        """Record an executed action."""
        self.guard.record_action(action, target, result)

    def transition(self, top: TopLevelState, sub: Optional[Enum] = None,
                   reason: str = ""):
        """Transition to a new state."""
        self.fsm.transition(top, sub, reason)
        # Reset loop detection on state change
        if self.guard.is_loop_detected():
            self.guard.reset_loop()

    def check_stuck(self) -> Optional[str]:
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
            duration = self.fsm.get_state_duration()

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

    def get_state_info(self) -> Dict:
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
