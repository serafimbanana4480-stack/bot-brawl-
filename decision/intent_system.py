"""
decision/intent_system.py

Intent System for persistent strategic goals.

Solves the "no intent system" problem where tactics exist as enums
but don't persist — the bot doesn't "plan" or have goals that span
multiple seconds/minutes.

Intents are high-level strategic goals that persist across frames:
- FARM: Collect power cubes, build advantage
- SURVIVE: Stay alive, avoid fights
- AGGRESSIVE: Push and kill enemies
- CONTROL: Hold zone/position
- AMBUSH: Wait in bush for opportunity
- RETREAT: Get to safety
- SUPPORT: Help teammates

The intent system:
1. Evaluates the world state to determine the best intent
2. Persists the intent across frames (doesn't flip-flop)
3. Provides intent to UtilityAI for action weighting
4. Tracks intent history for learning
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Strategic intents the bot can have."""
    FARM = "farm"               # Collect power cubes
    SURVIVE = "survive"         # Stay alive at all costs
    AGGRESSIVE = "aggressive"   # Push and kill enemies
    CONTROL = "control"         # Hold zone/position
    AMBUSH = "ambush"           # Wait in bush for opportunity
    RETREAT = "retreat"         # Get to safety
    SUPPORT = "support"         # Help teammates
    NEUTRAL = "neutral"         # No strong intent


@dataclass
class IntentState:
    """Current intent with reasoning and persistence."""
    intent: Intent = Intent.NEUTRAL
    confidence: float = 0.0     # How confident we are in this intent
    set_at: float = 0.0        # When this intent was set
    reason: str = ""
    previous_intent: Optional[Intent] = None
    persistence_bonus: float = 0.0  # Bonus for staying with current intent


class IntentSystem:
    """
    Persistent intent system for strategic decision making.

    Unlike frame-by-frame decisions, intents persist for seconds to minutes.
    They represent the bot's current strategic goal and influence all
    downstream decisions (action selection, movement, target priority).

    Features:
    - Intent persistence: minimum duration before switching
    - Hysteresis: must have strong reason to change intent
    - Game-phase awareness: different default intents per phase
    - Match-mode awareness: Gem Grab vs Showdown vs Brawl Ball
    - Integration with WorldModel for long-term memory
    """

    # Minimum time before allowing intent switch (ms)
    MIN_INTENT_DURATION_MS = 2000.0

    # Maximum intent duration before forced re-evaluation (ms)
    MAX_INTENT_DURATION_MS = 15000.0

    # Hysteresis: new intent must score this much higher than current
    HYSTERESIS_MARGIN = 0.2

    # Game mode default intents
    MODE_DEFAULTS = {
        "showdown": Intent.FARM,       # Showdown: farm first, fight later
        "gem_grab": Intent.CONTROL,    # Gem Grab: hold the center
        "brawl_ball": Intent.AGGRESSIVE,  # Brawl Ball: push forward
        "heist": Intent.AGGRESSIVE,    # Heist: attack the safe
        "hot_zone": Intent.CONTROL,    # Hot Zone: hold zones
        "knockout": Intent.AMBUSH,     # Knockout: careful, one life
        "bounty": Intent.SURVIVE,      # Bounty: don't die
    }

    def __init__(self, game_mode: str = "showdown"):
        self.game_mode = game_mode
        self._state = IntentState(
            intent=self.MODE_DEFAULTS.get(game_mode, Intent.NEUTRAL),
            set_at=time.time(),
            reason="initial_default",
        )
        self._intent_history: List[Dict] = []
        self._lock = threading.RLock()

        logger.info("[INTENT] Initialized for mode=%s, default intent=%s",
                     game_mode, self._state.intent.value)

    def set_game_mode(self, mode: str):
        """Update the game mode (called when match starts)."""
        self.game_mode = mode.lower().replace(" ", "_")
        default = self.MODE_DEFAULTS.get(self.game_mode, Intent.NEUTRAL)
        with self._lock:
            self._state = IntentState(
                intent=default,
                set_at=time.time(),
                reason=f"mode_default:{self.game_mode}",
            )
        logger.info("[INTENT] Mode changed to %s, intent set to %s",
                     self.game_mode, default.value)

    def evaluate(self, context: Dict) -> Intent:
        """
        Evaluate and return the current intent.

        May switch to a new intent if conditions warrant it.
        May maintain current intent if within persistence window.

        Args:
            context: World state dict with:
                - health: float (0-1)
                - enemies_nearby: int
                - pressure: float
                - match_phase: str ("early", "mid", "late")
                - has_super: bool
                - cube_count: int
                - alive_allies: int
                - gem_count: int (Gem Grab)
                - zone_control: float (0-1, Hot Zone)
                - is_in_bush: bool
                - brawler_role: str

        Returns:
            Current Intent
        """
        now = time.time()

        with self._lock:
            # Check persistence window
            age_ms = (now - self._state.set_at) * 1000

            if age_ms < self.MIN_INTENT_DURATION_MS:
                # Within minimum duration — keep current intent
                return self._state.intent

            # Evaluate all intents
            scores = self._score_all_intents(context)

            # Add persistence bonus to current intent
            current_idx = self._state.intent
            if current_idx in scores:
                persistence_bonus = min(0.3, age_ms / 10000.0)  # Up to 0.3 bonus
                scores[current_idx] += persistence_bonus

            # Find best intent
            best_intent = max(scores, key=scores.get)
            best_score = scores[best_intent]
            current_score = scores.get(self._state.intent, 0.0)

            # Check hysteresis
            should_switch = False
            reason = ""

            if best_intent != self._state.intent:
                improvement = best_score - current_score
                if improvement > self.HYSTERESIS_MARGIN:
                    should_switch = True
                    reason = f"better:{improvement:.2f}"
                elif age_ms > self.MAX_INTENT_DURATION_MS:
                    should_switch = True
                    reason = "max_duration_exceeded"

            if should_switch:
                previous = self._state.intent
                self._state = IntentState(
                    intent=best_intent,
                    confidence=best_score,
                    set_at=now,
                    reason=reason,
                    previous_intent=previous,
                )
                self._intent_history.append({
                    "intent": best_intent.value,
                    "previous": previous.value if previous else None,
                    "score": round(best_score, 2),
                    "reason": reason,
                    "timestamp": now,
                })
                if len(self._intent_history) > 50:
                    self._intent_history = self._intent_history[-50:]

                logger.info("[INTENT] Switched: %s → %s (%.2f, reason=%s)",
                            previous.value if previous else "None",
                            best_intent.value, best_score, reason)

            return self._state.intent

    def get_current_intent(self) -> Intent:
        """Get current intent without re-evaluation."""
        with self._lock:
            return self._state.intent

    def get_intent_info(self) -> Dict:
        """Get detailed intent state."""
        with self._lock:
            now = time.time()
            return {
                "intent": self._state.intent.value,
                "confidence": round(self._state.confidence, 2),
                "age_ms": round((now - self._state.set_at) * 1000),
                "reason": self._state.reason,
                "previous": self._state.previous_intent.value
                    if self._state.previous_intent else None,
            }

    def get_intent_history(self, limit: int = 10) -> List[Dict]:
        """Get recent intent history."""
        with self._lock:
            return self._intent_history[-limit:]

    def force_intent(self, intent: Intent, reason: str = "forced"):
        """Force a specific intent (e.g., from external coordinator)."""
        with self._lock:
            previous = self._state.intent
            self._state = IntentState(
                intent=intent,
                confidence=1.0,
                set_at=time.time(),
                reason=reason,
                previous_intent=previous,
            )
        logger.info("[INTENT] Forced: %s → %s (reason=%s)",
                     previous.value, intent.value, reason)

    # --- Intent scoring ---

    def _score_all_intents(self, ctx: Dict) -> Dict[Intent, float]:
        """Score all possible intents given the current context."""
        scores = {}

        health = ctx.get("health", 1.0)
        enemies = ctx.get("enemies_nearby", 0)
        pressure = ctx.get("pressure", 0.0)
        phase = ctx.get("match_phase", "early")
        has_super = ctx.get("has_super", False)
        cubes = ctx.get("cube_count", 0)
        allies = ctx.get("alive_allies", 0)
        gems = ctx.get("gem_count", 0)
        zone_ctrl = ctx.get("zone_control", 0.5)
        in_bush = ctx.get("is_in_bush", False)
        role = ctx.get("brawler_role", "damage")

        # FARM: Good early, when safe, when few cubes
        farm_score = 0.0
        if phase == "early":
            farm_score += 0.5
        elif phase == "mid":
            farm_score += 0.2
        if enemies == 0:
            farm_score += 0.3
        if cubes < 3:
            farm_score += 0.3
        if pressure < 1.0:
            farm_score += 0.2
        if role in ("tank", "assassin"):
            farm_score += 0.1  # Need cubes more
        scores[Intent.FARM] = farm_score

        # SURVIVE: Good when low health, high pressure
        survive_score = 0.0
        if health < 0.3:
            survive_score += 0.8
        elif health < 0.5:
            survive_score += 0.4
        if pressure > 3.0:
            survive_score += 0.4
        if enemies > 2:
            survive_score += 0.2
        if self.game_mode == "bounty":
            survive_score += 0.3  # Don't die in bounty
        if gems > 5:
            survive_score += 0.3  # Protect gems in Gem Grab
        scores[Intent.SURVIVE] = survive_score

        # AGGRESSIVE: Good when healthy, have super, enemies nearby
        aggro_score = 0.0
        if health > 0.7:
            aggro_score += 0.3
        if has_super:
            aggro_score += 0.3
        if enemies > 0 and enemies < 3:
            aggro_score += 0.2
        if role in ("assassin", "tank"):
            aggro_score += 0.2
        if phase == "late" and self.game_mode == "showdown":
            aggro_score += 0.3  # Final circle, must fight
        if self.game_mode in ("brawl_ball", "heist"):
            aggro_score += 0.2
        scores[Intent.AGGRESSIVE] = aggro_score

        # CONTROL: Good for zone-based modes
        control_score = 0.0
        if self.game_mode in ("gem_grab", "hot_zone"):
            control_score += 0.4
        if zone_ctrl < 0.5:
            control_score += 0.3  # Need to gain control
        if health > 0.5:
            control_score += 0.2
        if role in ("control", "support"):
            control_score += 0.2
        scores[Intent.CONTROL] = control_score

        # AMBUSH: Good when in bush, enemy approaching
        ambush_score = 0.0
        if in_bush:
            ambush_score += 0.4
        if enemies > 0 and enemies < 3:
            ambush_score += 0.2
        if role == "assassin":
            ambush_score += 0.3
        if self.game_mode == "knockout":
            ambush_score += 0.2
        scores[Intent.AMBUSH] = ambush_score

        # RETREAT: Good when overwhelmed
        retreat_score = 0.0
        if health < 0.3:
            retreat_score += 0.5
        if pressure > 4.0:
            retreat_score += 0.4
        if enemies > 3:
            retreat_score += 0.3
        if allies == 0 and enemies > 1:
            retreat_score += 0.2
        scores[Intent.RETREAT] = retreat_score

        # SUPPORT: Good when allies need help
        support_score = 0.0
        if role == "support":
            support_score += 0.3
        if allies > 0:
            support_score += 0.1
        if self.game_mode in ("gem_grab", "brawl_ball"):
            support_score += 0.2
        scores[Intent.SUPPORT] = support_score

        # NEUTRAL: Baseline
        scores[Intent.NEUTRAL] = 0.1

        return scores
