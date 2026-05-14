"""
decision/sticky_target.py

Sticky Target + Commitment System for Brawl Stars bot.

Solves the "no target commitment" problem where the scorer re-picks
the best target every frame, causing:
- Target thrashing: switching between equally-good targets
- Direction jitter: movement direction changes every frame
- Wasted ammo: shooting at different enemies each shot
- Inconsistent pressure: never focusing one enemy down

Solution:
- Target commitment: once a target is selected, stick with it for
  a minimum duration unless a much better target appears
- Direction smoothing: movement direction changes gradually
- Hysteresis: new target must be significantly better to switch
- Focus bonus: staying on one target increases effectiveness
"""

import math
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CommitmentReason(Enum):
    """Why we committed to this target."""
    NEW_BEST = "new_best"           # Best target when no commitment
    MAINTAINED = "maintained"       # Still committed, not switching
    SWITCHED_BETTER = "switched"    # Switched to significantly better
    SWITCHED_LOST = "lost_switch"   # Previous target lost, new one
    SWITCHED_DEAD = "dead_switch"   # Previous target likely dead
    OVERRIDE = "override"           # External override (e.g., super)


@dataclass
class TargetInfo:
    """Information about a tracked target."""
    track_id: int
    x: float
    y: float
    width: float
    height: float
    class_name: str = ""
    health: float = 1.0  # Estimated 0-1
    velocity_x: float = 0.0  # pixels/frame
    velocity_y: float = 0.0
    confidence: float = 1.0
    last_seen: float = 0.0
    threat_score: float = 0.0


@dataclass
class CommitmentState:
    """Current target commitment state."""
    target_id: Optional[int] = None
    target_info: Optional[TargetInfo] = None
    committed_at: float = 0.0
    reason: CommitmentReason = CommitmentReason.NEW_BEST
    focus_time: float = 0.0        # Total time focused on this target
    shots_at_target: int = 0       # Shots directed at this target
    hits_on_target: int = 0        # Estimated hits on this target
    effectiveness: float = 0.0     # Hit rate on this target


class StickyTarget:
    """
    Target commitment system with hysteresis and focus tracking.

    Once a target is selected, the bot commits to it for a minimum
    duration. Switching requires the new target to be significantly
    better (hysteresis margin).

    Config:
        min_commitment_ms: Minimum time to stay on a target (default 800ms)
        hysteresis_margin: New target must be this much better (0-1, default 0.3)
        max_commitment_ms: Maximum time on one target (default 5000ms)
        lost_timeout_ms: Time before declaring target lost (default 500ms)
        focus_bonus: Score bonus per second of focus (default 0.1/s)
    """

    def __init__(
        self,
        min_commitment_ms: float = 800.0,
        hysteresis_margin: float = 0.3,
        max_commitment_ms: float = 5000.0,
        lost_timeout_ms: float = 500.0,
        focus_bonus: float = 0.1,
    ):
        self.min_commitment_ms = min_commitment_ms
        self.hysteresis_margin = hysteresis_margin
        self.max_commitment_ms = max_commitment_ms
        self.lost_timeout_ms = lost_timeout_ms
        self.focus_bonus = focus_bonus

        self._state = CommitmentState()
        self._previous_targets: Dict[int, TargetInfo] = {}  # Track history

        # Direction smoothing
        self._current_direction: Optional[Tuple[float, float]] = None
        self._direction_smoothing = 0.3  # Lower = smoother (0-1)

        # Stats
        self._total_switches = 0
        self._total_commits = 0

        logger.info("[STICKY_TARGET] Initialized (commit=%dms, hysteresis=%.2f)",
                     min_commitment_ms, hysteresis_margin)

    def select_target(self, candidates: List[TargetInfo],
                      scorer_fn=None) -> Tuple[Optional[TargetInfo], CommitmentReason]:
        """
        Select the best target with commitment logic.

        Args:
            candidates: List of potential targets
            scorer_fn: Optional function(TargetInfo) -> float for scoring.
                       Default uses built-in scoring.

        Returns:
            (selected_target, reason) tuple
        """
        now = time.time()

        if not candidates:
            # No candidates — check if committed target is still visible
            if self._state.target_info is not None:
                age_ms = (now - self._state.target_info.last_seen) * 1000
                if age_ms > self.lost_timeout_ms:
                    self._release_commitment(CommitmentReason.SWITCHED_LOST)
                    return (None, CommitmentReason.SWITCHED_LOST)
                # Target lost but within timeout — keep commitment
                return (self._state.target_info, CommitmentReason.MAINTAINED)
            return (None, CommitmentReason.NEW_BEST)

        # Score all candidates
        scored = []
        for c in candidates:
            score = scorer_fn(c) if scorer_fn else self._default_scorer(c)
            # Add focus bonus for current target
            if c.track_id == self._state.target_id:
                focus_seconds = self._state.focus_time
                score += self.focus_bonus * focus_seconds
                # Effectiveness bonus: if we're hitting this target, keep focusing
                if self._state.effectiveness > 0.5:
                    score += 0.2
            scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_candidate, best_score = scored[0]

        # No current commitment — commit to best
        if self._state.target_id is None:
            self._commit(best_candidate, CommitmentReason.NEW_BEST, now)
            return (best_candidate, CommitmentReason.NEW_BEST)

        # Currently committed — check if we should switch
        commitment_age_ms = (now - self._state.committed_at) * 1000

        # Update target info if same target is still visible
        if best_candidate.track_id == self._state.target_id:
            self._state.target_info = best_candidate
            self._state.focus_time = now - self._state.committed_at
            return (best_candidate, CommitmentReason.MAINTAINED)

        # Different target — check commitment duration
        if commitment_age_ms < self.min_commitment_ms:
            # Still in minimum commitment period — don't switch
            # But update our target info if we can still see it
            current_visible = [c for c in candidates
                               if c.track_id == self._state.target_id]
            if current_visible:
                self._state.target_info = current_visible[0]
                self._state.focus_time = now - self._state.committed_at
                return (current_visible[0], CommitmentReason.MAINTAINED)
            else:
                # Current target not visible but within commitment — check timeout
                if self._state.target_info:
                    age_ms = (now - self._state.target_info.last_seen) * 1000
                    if age_ms < self.lost_timeout_ms:
                        return (self._state.target_info, CommitmentReason.MAINTAINED)
                # Target lost and timeout exceeded — switch
                self._commit(best_candidate, CommitmentReason.SWITCHED_LOST, now)
                return (best_candidate, CommitmentReason.SWITCHED_LOST)

        # Past minimum commitment — check hysteresis
        current_score = 0.0
        current_visible = [c for c in candidates
                           if c.track_id == self._state.target_id]
        if current_visible:
            current_score = scorer_fn(current_visible[0]) if scorer_fn else \
                            self._default_scorer(current_visible[0])
            current_score += self.focus_bonus * self._state.focus_time
        else:
            # Current target not visible — easy to switch
            self._commit(best_candidate, CommitmentReason.SWITCHED_LOST, now)
            return (best_candidate, CommitmentReason.SWITCHED_LOST)

        # Check max commitment — force switch after too long
        if commitment_age_ms > self.max_commitment_ms:
            self._commit(best_candidate, CommitmentReason.SWITCHED_BETTER, now)
            return (best_candidate, CommitmentReason.SWITCHED_BETTER)

        # Hysteresis: new target must be significantly better
        score_improvement = best_score - current_score
        if score_improvement > self.hysteresis_margin:
            logger.debug("[STICKY_TARGET] Switching: %.2f > %.2f (margin=%.2f)",
                         best_score, current_score, score_improvement)
            self._commit(best_candidate, CommitmentReason.SWITCHED_BETTER, now)
            return (best_candidate, CommitmentReason.SWITCHED_BETTER)

        # Stay with current target
        if current_visible:
            self._state.target_info = current_visible[0]
            self._state.focus_time = now - self._state.committed_at
            return (current_visible[0], CommitmentReason.MAINTAINED)

        # Fallback
        return (best_candidate, CommitmentReason.MAINTAINED)

    def record_shot(self, hit: bool = False):
        """Record that we shot at the committed target."""
        if self._state.target_id is not None:
            self._state.shots_at_target += 1
            if hit:
                self._state.hits_on_target += 1
            if self._state.shots_at_target > 0:
                self._state.effectiveness = (
                    self._state.hits_on_target / self._state.shots_at_target
                )

    def force_switch(self, new_target: Optional[TargetInfo] = None):
        """Force a target switch (e.g., when using super on a specific enemy)."""
        if new_target:
            self._commit(new_target, CommitmentReason.OVERRIDE, time.time())
        else:
            self._release_commitment(CommitmentReason.OVERRIDE)

    def get_smoothed_direction(self, target: TargetInfo,
                                player_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Get smoothed movement direction toward target.

        Prevents direction jitter by blending with previous direction.
        """
        dx = target.x - player_pos[0]
        dy = target.y - player_pos[1]
        length = math.sqrt(dx * dx + dy * dy)

        if length < 1.0:
            return (0.0, 0.0)

        new_dir = (dx / length, dy / length)

        if self._current_direction is None:
            self._current_direction = new_dir
            return new_dir

        # Smooth: blend old and new directions
        alpha = self._direction_smoothing
        smoothed = (
            self._current_direction[0] * (1 - alpha) + new_dir[0] * alpha,
            self._current_direction[1] * (1 - alpha) + new_dir[1] * alpha,
        )

        # Re-normalize
        length = math.sqrt(smoothed[0] ** 2 + smoothed[1] ** 2)
        if length > 0.001:
            smoothed = (smoothed[0] / length, smoothed[1] / length)

        self._current_direction = smoothed
        return smoothed

    def get_commitment_info(self) -> Dict:
        """Get current commitment state info."""
        if self._state.target_id is None:
            return {"committed": False}

        now = time.time()
        return {
            "committed": True,
            "target_id": self._state.target_id,
            "commit_age_ms": round((now - self._state.committed_at) * 1000),
            "focus_time_s": round(self._state.focus_time, 1),
            "shots": self._state.shots_at_target,
            "hits": self._state.hits_on_target,
            "effectiveness": round(self._state.effectiveness, 2),
            "reason": self._state.reason.value,
        }

    def get_stats(self) -> Dict:
        """Get sticky target statistics."""
        return {
            "total_commits": self._total_commits,
            "total_switches": self._total_switches,
            "current_commitment": self.get_commitment_info(),
        }

    # --- Internal ---

    def _commit(self, target: TargetInfo, reason: CommitmentReason, now: float):
        """Commit to a new target."""
        if self._state.target_id is not None and target.track_id != self._state.target_id:
            self._total_switches += 1

        self._state = CommitmentState(
            target_id=target.track_id,
            target_info=target,
            committed_at=now,
            reason=reason,
            focus_time=0.0,
            shots_at_target=0,
            hits_on_target=0,
            effectiveness=0.0,
        )
        self._total_commits += 1

        # Store in history
        self._previous_targets[target.track_id] = target

    def _release_commitment(self, reason: CommitmentReason):
        """Release current commitment."""
        if self._state.target_id is not None:
            self._total_switches += 1
        self._state = CommitmentState(reason=reason)

    def _default_scorer(self, target: TargetInfo) -> float:
        """
        Default target scoring function.

        Prioritizes:
        1. Low-health enemies (easy kills)
        2. Close enemies (can actually hit)
        3. High-threat enemies (dangerous if ignored)
        4. Stationary enemies (easier to hit)
        """
        score = 0.0

        # Health: prefer low-health (kill potential)
        if target.health < 0.3:
            score += 0.4
        elif target.health < 0.6:
            score += 0.2
        else:
            score += 0.1

        # Distance: prefer medium range
        # (too close = risky, too far = can't hit)
        dist = math.sqrt(target.x ** 2 + target.y ** 2)  # Approximate
        if dist < 200:
            score += 0.2
        elif dist < 400:
            score += 0.3
        elif dist < 600:
            score += 0.15
        else:
            score += 0.05

        # Threat: prefer high-threat enemies
        score += target.threat_score * 0.3

        # Velocity: stationary enemies are easier to hit
        speed = math.sqrt(target.velocity_x ** 2 + target.velocity_y ** 2)
        if speed < 2.0:
            score += 0.15  # Nearly stationary
        elif speed > 10.0:
            score -= 0.1  # Very fast, hard to hit

        # Confidence: higher confidence detections are more reliable
        score += target.confidence * 0.1

        return score
