"""
decision/enemy_intention.py

Enemy Intention Predictor for Brawl Stars bot.

Solves the "no enemy intention prediction" problem by classifying
enemy behavior patterns into tactical intentions:

- RUSH: Enemy is charging directly at us (assassin dive)
- RETREAT: Enemy is running away (low health, disengaging)
- BAIT: Enemy is pretending to retreat but waiting to ambush
- FLANK: Enemy is moving around to attack from the side
- POKE: Enemy is attacking from range then backing off
- HOLD: Enemy is holding position (zone control)
- FARM: Enemy is collecting power cubes
- CHASE: Enemy is pursuing a teammate

This information feeds into:
- UtilityAI: adjust action scores based on enemy intent
- Kiting engine: different kiting for rush vs poke
- Cover engine: anticipate flank routes
- Target priority: bait enemies are dangerous
"""

import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EnemyIntention(Enum):
    """Predicted enemy tactical intention."""
    RUSH = "rush"           # Charging directly at us
    RETREAT = "retreat"     # Running away
    BAIT = "bait"           # Pretending to retreat, will ambush
    FLANK = "flank"         # Moving around to attack from side
    POKE = "poke"           # Attack from range, then back off
    HOLD = "hold"           # Holding position
    FARM = "farm"           # Collecting power cubes
    CHASE = "chase"         # Pursuing a teammate
    UNKNOWN = "unknown"     # Not enough data


@dataclass
class EnemyTrack:
    """Tracking data for a single enemy over time."""
    track_id: int
    positions: list[tuple[float, float, float]] = None  # (x, y, timestamp)
    velocities: list[tuple[float, float]] = None  # (vx, vy) in pixels/s
    health_history: list[tuple[float, float]] = None  # (health, timestamp)
    class_name: str = ""
    current_intention: EnemyIntention = EnemyIntention.UNKNOWN
    intention_confidence: float = 0.0
    last_seen: float = 0.0

    def __post_init__(self):
        if self.positions is None:
            self.positions = []
        if self.velocities is None:
            self.velocities = []
        if self.health_history is None:
            self.health_history = []


class EnemyIntentionPredictor:
    """
    Predicts enemy tactical intentions from movement patterns.

    Uses velocity analysis, position history, and game context to
    classify what each enemy is trying to do.

    Key signals:
    - Velocity direction relative to player (toward = rush, away = retreat)
    - Velocity magnitude (fast = committed action, slow = holding)
    - Health trend (dropping = retreat likely, stable = aggressive)
    - Position relative to map features (near bush = ambush potential)
    - Lateral movement (perpendicular to player = flanking)
    """

    # Time window for velocity calculation (seconds)
    VELOCITY_WINDOW = 0.5

    # Maximum position history length
    MAX_HISTORY = 30

    # Minimum observations before predicting
    MIN_OBSERVATIONS = 3

    # Speed thresholds (pixels/second)
    SPEED_STATIONARY = 20.0
    SPEED_WALKING = 80.0
    SPEED_RUNNING = 200.0

    # Angle thresholds (degrees)
    ANGLE_DIRECT = 30.0      # Within 30° of direct line = "toward/away"
    ANGLE_FLANK = 60.0       # 30-60° from direct = "flanking"

    # Bait detection: if enemy was retreating then suddenly stops/reverses
    BAIT_REVERSAL_WINDOW = 2.0  # seconds to look back for reversal

    def __init__(self):
        self._tracks: dict[int, EnemyTrack] = {}
        self._player_pos: tuple[float, float] | None = None
        self._lock = threading.RLock()

        logger.info("[ENEMY_INTENT] Initialized")

    def update(self, enemy_detections: list[dict],
               player_pos: tuple[float, float],
               ally_positions: list[tuple[float, float]] | None = None):
        """
        Update enemy tracking and predict intentions.

        Args:
            enemy_detections: List of enemy dicts with track_id, x, y, width, height,
                              class_name, health (optional)
            player_pos: Player position in pixels
            ally_positions: Optional ally positions for chase detection
        """
        now = time.time()
        self._player_pos = player_pos

        with self._lock:
            # Update existing tracks and create new ones
            active_ids = set()

            for det in enemy_detections:
                tid = det.get("track_id", id(det))  # Use object id if no track_id
                active_ids.add(tid)

                if tid not in self._tracks:
                    self._tracks[tid] = EnemyTrack(
                        track_id=tid,
                        class_name=det.get("class_name", ""),
                    )

                track = self._tracks[tid]
                x, y = det.get("x", 0), det.get("y", 0)
                health = det.get("health", None)

                # Add position
                track.positions.append((x, y, now))
                if len(track.positions) > self.MAX_HISTORY:
                    track.positions = track.positions[-self.MAX_HISTORY:]

                # Add health
                if health is not None:
                    track.health_history.append((health, now))
                    if len(track.health_history) > self.MAX_HISTORY:
                        track.health_history = track.health_history[-self.MAX_HISTORY:]

                # Calculate velocity
                self._update_velocity(track)

                # Predict intention
                self._predict_intention(track, player_pos, ally_positions)

                track.last_seen = now

            # Mark tracks as stale if not seen recently
            stale_ids = [tid for tid, track in self._tracks.items()
                         if now - track.last_seen > 2.0]
            for tid in stale_ids:
                del self._tracks[tid]

    def get_enemy_intention(self, track_id: int) -> tuple[EnemyIntention, float]:
        """
        Get the predicted intention for a specific enemy.

        Returns:
            (intention, confidence) tuple
        """
        with self._lock:
            track = self._tracks.get(track_id)
            if track:
                return (track.current_intention, track.intention_confidence)
            return (EnemyIntention.UNKNOWN, 0.0)

    def get_all_intentions(self) -> dict[int, tuple[EnemyIntention, float]]:
        """Get intentions for all tracked enemies."""
        with self._lock:
            return {
                tid: (track.current_intention, track.intention_confidence)
                for tid, track in self._tracks.items()
            }

    def get_rushers(self) -> list[int]:
        """Get track IDs of enemies currently rushing toward us."""
        with self._lock:
            return [tid for tid, t in self._tracks.items()
                    if t.current_intention == EnemyIntention.RUSH
                    and t.intention_confidence > 0.5]

    def get_flankers(self) -> list[int]:
        """Get track IDs of enemies attempting to flank."""
        with self._lock:
            return [tid for tid, t in self._tracks.items()
                    if t.current_intention == EnemyIntention.FLANK
                    and t.intention_confidence > 0.4]

    def get_baiters(self) -> list[int]:
        """Get track IDs of enemies that might be baiting."""
        with self._lock:
            return [tid for tid, t in self._tracks.items()
                    if t.current_intention == EnemyIntention.BAIT
                    and t.intention_confidence > 0.3]

    def get_stats(self) -> dict:
        """Get predictor statistics."""
        with self._lock:
            intent_counts = {}
            for track in self._tracks.values():
                key = track.current_intention.value
                intent_counts[key] = intent_counts.get(key, 0) + 1
            return {
                "tracked_enemies": len(self._tracks),
                "intent_distribution": intent_counts,
            }

    # --- Internal ---

    def _update_velocity(self, track: EnemyTrack):
        """Calculate velocity from position history."""
        if len(track.positions) < 2:
            track.velocities.append((0.0, 0.0))
            return

        # Use recent positions within the velocity window
        now = time.time()
        recent = [(x, y, t) for x, y, t in track.positions
                  if now - t <= self.VELOCITY_WINDOW]

        if len(recent) < 2:
            track.velocities.append((0.0, 0.0))
            return

        # Average velocity over recent window
        vx_sum, vy_sum = 0.0, 0.0
        count = 0
        for i in range(1, len(recent)):
            dt = recent[i][2] - recent[i - 1][2]
            if dt > 0.01:
                vx = (recent[i][0] - recent[i - 1][0]) / dt
                vy = (recent[i][1] - recent[i - 1][1]) / dt
                vx_sum += vx
                vy_sum += vy
                count += 1

        if count > 0:
            track.velocities.append((vx_sum / count, vy_sum / count))
        else:
            track.velocities.append((0.0, 0.0))

        if len(track.velocities) > self.MAX_HISTORY:
            track.velocities = track.velocities[-self.MAX_HISTORY:]

    def _predict_intention(self, track: EnemyTrack,
                           player_pos: tuple[float, float],
                           ally_positions: list[tuple[float, float]] | None):
        """Predict the intention of a single enemy."""
        if len(track.positions) < self.MIN_OBSERVATIONS:
            track.current_intention = EnemyIntention.UNKNOWN
            track.intention_confidence = 0.0
            return

        # Current position and velocity
        cx, cy, _ = track.positions[-1]
        vx, vy = track.velocities[-1] if track.velocities else (0.0, 0.0)
        speed = math.sqrt(vx * vx + vy * vy)

        # Vector from enemy to player
        dx = player_pos[0] - cx
        dy = player_pos[1] - cy
        dist_to_player = math.sqrt(dx * dx + dy * dy)

        if dist_to_player < 1.0:
            dist_to_player = 1.0

        # Angle between velocity and direction to player
        if speed > self.SPEED_STATIONARY:
            # Normalize vectors
            vel_norm = (vx / speed, vy / speed)
            dir_norm = (dx / dist_to_player, dy / dist_to_player)

            # Dot product = cosine of angle
            dot = vel_norm[0] * dir_norm[0] + vel_norm[1] * dir_norm[1]
            angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
        else:
            angle_deg = 180.0  # Stationary = not moving toward player

        # Score each intention
        scores = {}

        # RUSH: Moving fast toward player
        rush_score = 0.0
        if speed > self.SPEED_RUNNING and angle_deg < self.ANGLE_DIRECT:
            rush_score = 0.8
        elif speed > self.SPEED_WALKING and angle_deg < self.ANGLE_DIRECT:
            rush_score = 0.5
        # Assassin brawlers are more likely to rush
        if track.class_name.lower() in ("edgar", "leon", "mortis", "fang", "el_primo"):
            rush_score += 0.2
        scores[EnemyIntention.RUSH] = rush_score

        # RETREAT: Moving away from player
        retreat_score = 0.0
        if speed > self.SPEED_WALKING and angle_deg > (180 - self.ANGLE_DIRECT):
            retreat_score = 0.6
        # Low health enemies are more likely retreating
        if track.health_history and track.health_history[-1][0] < 0.4:
            retreat_score += 0.3
        scores[EnemyIntention.RETREAT] = retreat_score

        # BAIT: Was retreating, then stopped or reversed
        bait_score = self._detect_bait(track, player_pos)
        scores[EnemyIntention.BAIT] = bait_score

        # FLANK: Moving perpendicular to direct line
        flank_score = 0.0
        if speed > self.SPEED_WALKING:
            if self.ANGLE_DIRECT < angle_deg < (90 - self.ANGLE_FLANK / 2):
                flank_score = 0.4
            elif (90 - self.ANGLE_FLANK / 2) <= angle_deg <= (90 + self.ANGLE_FLANK / 2):
                flank_score = 0.6
        # Assassins are more likely to flank
        if track.class_name.lower() in ("leon", "mortis", "fang"):
            flank_score += 0.2
        scores[EnemyIntention.FLANK] = flank_score

        # POKE: Moderate speed, slight approach then retreat pattern
        poke_score = 0.0
        if self.SPEED_STATIONARY < speed < self.SPEED_RUNNING:
            # Check for oscillating movement (approach-retreat)
            if len(track.velocities) >= 4:
                recent_vx = [v[0] for v in track.velocities[-4:]]
                # If velocity direction changes sign, it's poking
                sign_changes = sum(1 for i in range(1, len(recent_vx))
                                   if recent_vx[i] * recent_vx[i - 1] < 0)
                if sign_changes >= 2:
                    poke_score = 0.5
            # Ranged brawlers are more likely to poke
            if track.class_name.lower() in ("piper", "colt", "brock", "penny", "sprout"):
                poke_score += 0.2
        scores[EnemyIntention.POKE] = poke_score

        # HOLD: Stationary or very slow
        hold_score = 0.0
        if speed < self.SPEED_STATIONARY:
            hold_score = 0.5
        elif speed < self.SPEED_WALKING:
            hold_score = 0.2
        scores[EnemyIntention.HOLD] = hold_score

        # FARM: Moving toward power cube locations (heuristic: moving away from center)
        farm_score = 0.0
        if speed > self.SPEED_WALKING:
            # Moving away from center of map
            center_x, center_y = 640, 360
            dist_from_center = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
            vel_from_center = (vx * (cx - center_x) + vy * (cy - center_y)) / max(1, dist_from_center)
            if vel_from_center > 50:  # Moving outward
                farm_score = 0.3
        scores[EnemyIntention.FARM] = farm_score

        # CHASE: Moving toward an ally (not player)
        chase_score = 0.0
        if ally_positions and speed > self.SPEED_WALKING:
            for ax, ay in ally_positions:
                adx = ax - cx
                ady = ay - cy
                adist = math.sqrt(adx * adx + ady * ady)
                if adist > 1.0 and speed > self.SPEED_WALKING:
                    a_dot = (vx * adx + vy * ady) / (speed * adist)
                    if a_dot > 0.7:  # Moving toward ally
                        chase_score = max(chase_score, 0.4)
        scores[EnemyIntention.CHASE] = chase_score

        # Select best intention
        if not scores or max(scores.values()) < 0.1:
            track.current_intention = EnemyIntention.UNKNOWN
            track.intention_confidence = 0.0
            return

        best_intention = max(scores, key=scores.get)
        best_score = scores[best_intention]

        # Apply persistence: slight bonus for keeping same intention
        if track.current_intention == best_intention:
            best_score += 0.1

        track.current_intention = best_intention
        track.intention_confidence = min(1.0, best_score)

    def _detect_bait(self, track: EnemyTrack,
                     player_pos: tuple[float, float]) -> float:
        """
        Detect bait pattern: enemy was retreating then stopped/reversed.

        This is the most dangerous pattern because the bot might chase
        a "retreating" enemy into an ambush.
        """
        if len(track.velocities) < 4:
            return 0.0

        time.time()

        # Check if enemy was moving away then stopped or reversed
        was_retreating = False
        reversed_or_stopped = False

        for i, (vx, vy) in enumerate(track.velocities):
            speed = math.sqrt(vx * vx + vy * vy)
            if speed < self.SPEED_STATIONARY:
                continue

            # Get position at this time
            if i < len(track.positions):
                px, py, pt = track.positions[i]
                dx = player_pos[0] - px
                dy = player_pos[1] - py
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 1.0:
                    dot = (vx * dx + vy * dy) / (speed * dist)
                    if dot < -0.5:  # Moving away from player
                        was_retreating = True
                    elif dot > 0.3 and was_retreating:
                        reversed_or_stopped = True

        if was_retreating and reversed_or_stopped:
            return 0.6

        # Near a bush while "retreating" = potential bait
        if was_retreating and len(track.positions) > 0:
            # Heuristic: if enemy is near edge of map (where bushes often are)
            cx, cy, _ = track.positions[-1]
            if cx < 150 or cx > 1130 or cy < 100 or cy > 620:
                return 0.3

        return 0.0
