"""
core/central_coordinator.py

Central Coordinator for resolving conflicts between modules.

Solves the "no central coordinator" problem where modules can contradict
each other. For example:
- Scorer says ATTACK but pressure map says RETREAT
- Movement wants to go left but cover engine says go right
- Target selector picks enemy A but kiting engine wants to move away from A
- Intent says FARM but match phase says late game should fight

The CentralCoordinator:
1. Collects recommendations from all subsystems
2. Resolves conflicts using priority rules and context
3. Issues a single unified decision
4. Tracks decision history for consistency

This is NOT a replacement for UtilityAI — it's a higher-level
conflict resolver that sits above the individual subsystems.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of decisions the coordinator can make."""
    ACTION = auto()         # What action to take
    MOVEMENT = auto()       # Where to move
    TARGET = auto()         # Who to target
    ABILITY = auto()        # When to use super/ability
    POSITIONING = auto()    # Where to position on the map


class Priority(Enum):
    """Decision priority levels."""
    CRITICAL = 4    # Survival-critical (about to die)
    HIGH = 3        # Important (enemy rushing, super available)
    MEDIUM = 2      # Normal (standard combat decisions)
    LOW = 1         # Minor (farming, positioning)


@dataclass
class Recommendation:
    """A recommendation from a subsystem."""
    source: str                     # Module name (e.g., "pressure_map", "utility_ai")
    decision_type: DecisionType     # What kind of decision
    action: Any                     # Recommended action/value
    priority: Priority              # How important
    confidence: float               # 0-1, how confident the subsystem is
    reason: str = ""                # Why this recommendation
    context: Dict = None            # Additional context

    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class CoordinatedDecision:
    """The final coordinated decision."""
    decision_type: DecisionType
    action: Any
    source: str                     # Which recommendation won
    overridden: List[str]           # Which sources were overridden
    reason: str = ""
    timestamp: float = 0.0


class CentralCoordinator:
    """
    Central decision coordinator that resolves conflicts between subsystems.

    Priority rules:
    1. CRITICAL survival decisions always win
    2. HIGH-priority recommendations override MEDIUM/LOW
    3. When same priority: confidence-weighted voting
    4. Intent system provides tie-breaking bias
    5. Consistency bias: slight preference for maintaining current decision

    Integration points:
    - Receives recommendations from: UtilityAI, PressureMap, IntentSystem,
      StickyTarget, EnemyIntentionPredictor, BehavioralProfile, WorldModel
    - Outputs unified decisions to: wrapper.py / play.py
    """

    # How much to bias toward maintaining current decision
    CONSISTENCY_BIAS = 0.15

    # Minimum confidence difference to override a higher-priority source
    OVERRIDE_CONFIDENCE_MARGIN = 0.3

    def __init__(self):
        self._recommendations: List[Recommendation] = []
        self._last_decisions: Dict[DecisionType, CoordinatedDecision] = {}
        self._decision_history: List[CoordinatedDecision] = []
        self._lock = threading.RLock()

        # Current intent (set externally by IntentSystem)
        self._current_intent: Optional[str] = None

        logger.info("[COORDINATOR] Initialized")

    def set_intent(self, intent: str):
        """Set the current strategic intent (from IntentSystem)."""
        self._current_intent = intent

    def submit_recommendation(self, rec: Recommendation):
        """
        Submit a recommendation from a subsystem.

        Called by each subsystem during a frame's decision cycle.
        """
        with self._lock:
            self._recommendations.append(rec)

    def submit_batch(self, recommendations: List[Recommendation]):
        """Submit multiple recommendations at once."""
        with self._lock:
            self._recommendations.extend(recommendations)

    def resolve(self) -> Dict[DecisionType, CoordinatedDecision]:
        """
        Resolve all submitted recommendations into final decisions.

        Called once per frame after all subsystems have submitted.
        Clears the recommendation queue.

        Returns:
            Dict mapping DecisionType to final CoordinatedDecision
        """
        with self._lock:
            decisions = {}

            # Group recommendations by decision type
            by_type: Dict[DecisionType, List[Recommendation]] = {}
            for rec in self._recommendations:
                dt = rec.decision_type
                if dt not in by_type:
                    by_type[dt] = []
                by_type[dt].append(rec)

            # Resolve each decision type independently
            for dt, recs in by_type.items():
                decision = self._resolve_decision_type(dt, recs)
                decisions[dt] = decision

                # Track history
                self._decision_history.append(decision)
                if len(self._decision_history) > 100:
                    self._decision_history = self._decision_history[-100:]

            # Clear recommendations for next frame
            self._recommendations.clear()

            # Update last decisions
            self._last_decisions.update(decisions)

            return decisions

    def get_last_decision(self, decision_type: DecisionType) -> Optional[CoordinatedDecision]:
        """Get the last decision for a given type."""
        with self._lock:
            return self._last_decisions.get(decision_type)

    def get_decision_history(self, limit: int = 20) -> List[Dict]:
        """Get recent decision history."""
        with self._lock:
            return [
                {
                    "type": d.decision_type.name,
                    "action": str(d.action),
                    "source": d.source,
                    "overridden": d.overridden,
                    "reason": d.reason,
                }
                for d in self._decision_history[-limit:]
            ]

    def get_stats(self) -> Dict:
        """Get coordinator statistics."""
        with self._lock:
            override_counts = {}
            for d in self._decision_history:
                for src in d.overridden:
                    override_counts[src] = override_counts.get(src, 0) + 1

            return {
                "total_decisions": len(self._decision_history),
                "override_counts": override_counts,
                "current_intent": self._current_intent,
            }

    # --- Internal ---

    def _resolve_decision_type(self, dt: DecisionType,
                                recs: List[Recommendation]) -> CoordinatedDecision:
        """Resolve recommendations for a single decision type."""
        if not recs:
            return CoordinatedDecision(
                decision_type=dt, action=None, source="none",
                overridden=[], reason="no_recommendations",
                timestamp=time.time(),
            )

        if len(recs) == 1:
            rec = recs[0]
            return CoordinatedDecision(
                decision_type=dt, action=rec.action, source=rec.source,
                overridden=[], reason=rec.reason,
                timestamp=time.time(),
            )

        # Sort by priority (highest first), then confidence
        recs_sorted = sorted(recs, key=lambda r: (r.priority.value, r.confidence),
                             reverse=True)

        # Check for CRITICAL priority — always wins
        critical = [r for r in recs_sorted if r.priority == Priority.CRITICAL]
        if critical:
            best = critical[0]
            overridden = [r.source for r in recs if r.source != best.source]
            return CoordinatedDecision(
                decision_type=dt, action=best.action, source=best.source,
                overridden=overridden,
                reason=f"critical:{best.reason}",
                timestamp=time.time(),
            )

        # No critical — use weighted voting
        # Group by action value
        action_scores: Dict[Any, float] = {}
        action_sources: Dict[Any, List[str]] = {}

        for rec in recs_sorted:
            action_key = str(rec.action)  # Normalize for grouping
            score = rec.priority.value * rec.confidence

            # Intent alignment bonus
            if self._current_intent and self._is_intent_aligned(rec):
                score += 0.3

            # Consistency bias: prefer current decision
            last = self._last_decisions.get(dt)
            if last and str(last.action) == action_key:
                score += self.CONSISTENCY_BIAS

            if action_key not in action_scores:
                action_scores[action_key] = 0.0
                action_sources[action_key] = []
            action_scores[action_key] += score
            action_sources[action_key].append(rec.source)

        # Select highest-scoring action
        best_action_key = max(action_scores, key=action_scores.get)

        # Find the original recommendation for this action
        best_rec = None
        for rec in recs_sorted:
            if str(rec.action) == best_action_key:
                best_rec = rec
                break

        if best_rec is None:
            best_rec = recs_sorted[0]

        # Determine overridden sources
        overridden = [r.source for r in recs if r.source != best_rec.source
                      and str(r.action) != best_action_key]

        return CoordinatedDecision(
            decision_type=dt, action=best_rec.action, source=best_rec.source,
            overridden=overridden,
            reason=f"voted:{best_rec.reason} (score={action_scores[best_action_key]:.2f})",
            timestamp=time.time(),
        )

    def _is_intent_aligned(self, rec: Recommendation) -> bool:
        """Check if a recommendation aligns with the current intent."""
        if not self._current_intent:
            return False

        # Intent → action alignment mapping
        alignment = {
            "farm": {"collect_cube": True, "retreat": True},
            "survive": {"retreat": True, "take_cover": True, "heal_up": True},
            "aggressive": {"attack": True, "chase": True, "use_super": True},
            "control": {"hold_position": True, "kite": True},
            "ambush": {"ambush": True, "take_cover": True},
            "retreat": {"retreat": True, "take_cover": True},
        }

        intent_map = alignment.get(self._current_intent, {})
        action_str = str(rec.action).lower()
        return intent_map.get(action_str, False)
