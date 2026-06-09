"""
core/behavioral_profile.py

Behavioral Profile System for human-like play styles.

Solves the "superficial humanization" problem. Current humanization only
adds random delays and jitter — it doesn't model actual human behavior
patterns. Real players have distinct play styles that persist over time.

Profiles:
- AGGRESSIVE: Rushes in, takes fights, high APM
- PASSIVE: Stays back, avoids fights, low APM
- SNIPER: Stays at range, precise shots, low movement
- NERVOUS: Over-reacts to threats, erratic movement
- BALANCED: Moderate everything, most common human style

Each profile modifies:
- Reaction time distribution
- APM (actions per minute)
- Movement patterns (aggressive push vs cautious retreat)
- Target selection preferences
- Willingness to take fights
- Super usage patterns
- Position preferences (front line vs back line)

Profiles can:
- Be selected randomly at match start
- Blend between profiles for more variety
- Shift during a match (e.g., aggressive early, passive late)
- Be influenced by brawler type (tank → more aggressive)
"""

import json
import logging
import math
import random
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ProfileType(Enum):
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    SNIPER = "sniper"
    NERVOUS = "nervous"
    BALANCED = "balanced"


@dataclass
class ProfileParams:
    """Parameters that define a behavioral profile."""
    # Reaction time (seconds)
    reaction_time_min: float = 0.12
    reaction_time_max: float = 0.40
    reaction_time_mean: float = 0.22

    # APM (actions per minute)
    apm_target: float = 35.0
    apm_variance: float = 10.0

    # Movement
    movement_aggression: float = 0.5    # 0=retreat-first, 1=push-first
    movement_smoothness: float = 0.5    # 0=erratic, 1=smooth
    strafe_frequency: float = 0.2       # How often to strafe while fighting
    retreat_threshold: float = 0.4      # Health % to start retreating

    # Combat
    fight_willingness: float = 0.5      # 0=avoid fights, 1=always fight
    target_switch_tolerance: float = 0.5 # How easily to switch targets
    super_usage_eagerness: float = 0.5  # 0=save super, 1=use immediately
    chase_willingness: float = 0.5      # Willingness to chase low-HP enemies

    # Positioning
    preferred_range: float = 0.5        # 0=melee, 1=max range
    zone_commitment: float = 0.5        # How strongly to hold position
    bush_usage: float = 0.5             # How often to use bushes

    # Timing patterns
    burst_pattern: float = 0.5          # 0=steady, 1=burst then pause
    pause_frequency: float = 0.1        # How often to pause (think time)
    pause_duration_min: float = 0.3
    pause_duration_max: float = 1.5

    # UtilityAI weight overrides
    utility_weights: Dict = field(default_factory=dict)


# Predefined profiles
PROFILES: Dict[ProfileType, ProfileParams] = {
    ProfileType.AGGRESSIVE: ProfileParams(
        reaction_time_min=0.08, reaction_time_max=0.25, reaction_time_mean=0.15,
        apm_target=45.0, apm_variance=8.0,
        movement_aggression=0.8, movement_smoothness=0.6,
        strafe_frequency=0.3, retreat_threshold=0.2,
        fight_willingness=0.9, target_switch_tolerance=0.6,
        super_usage_eagerness=0.7, chase_willingness=0.8,
        preferred_range=0.2, zone_commitment=0.3,
        bush_usage=0.2, burst_pattern=0.7,
        pause_frequency=0.05, pause_duration_min=0.2, pause_duration_max=0.8,
        utility_weights={"aggression": 1.0, "survival": 0.3, "opportunity": 0.8},
    ),
    ProfileType.PASSIVE: ProfileParams(
        reaction_time_min=0.20, reaction_time_max=0.60, reaction_time_mean=0.35,
        apm_target=25.0, apm_variance=8.0,
        movement_aggression=0.2, movement_smoothness=0.7,
        strafe_frequency=0.1, retreat_threshold=0.6,
        fight_willingness=0.2, target_switch_tolerance=0.3,
        super_usage_eagerness=0.3, chase_willingness=0.1,
        preferred_range=0.7, zone_commitment=0.7,
        bush_usage=0.7, burst_pattern=0.2,
        pause_frequency=0.2, pause_duration_min=0.5, pause_duration_max=2.0,
        utility_weights={"aggression": 0.2, "survival": 1.0, "farm": 0.8},
    ),
    ProfileType.SNIPER: ProfileParams(
        reaction_time_min=0.10, reaction_time_max=0.30, reaction_time_mean=0.18,
        apm_target=28.0, apm_variance=6.0,
        movement_aggression=0.3, movement_smoothness=0.8,
        strafe_frequency=0.15, retreat_threshold=0.5,
        fight_willingness=0.5, target_switch_tolerance=0.4,
        super_usage_eagerness=0.4, chase_willingness=0.1,
        preferred_range=0.9, zone_commitment=0.6,
        bush_usage=0.6, burst_pattern=0.3,
        pause_frequency=0.15, pause_duration_min=0.3, pause_duration_max=1.2,
        utility_weights={"aggression": 0.4, "survival": 0.7, "control": 0.6},
    ),
    ProfileType.NERVOUS: ProfileParams(
        reaction_time_min=0.06, reaction_time_max=0.20, reaction_time_mean=0.10,
        apm_target=50.0, apm_variance=15.0,
        movement_aggression=0.3, movement_smoothness=0.2,
        strafe_frequency=0.5, retreat_threshold=0.6,
        fight_willingness=0.3, target_switch_tolerance=0.8,
        super_usage_eagerness=0.6, chase_willingness=0.2,
        preferred_range=0.5, zone_commitment=0.2,
        bush_usage=0.8, burst_pattern=0.8,
        pause_frequency=0.05, pause_duration_min=0.1, pause_duration_max=0.5,
        utility_weights={"aggression": 0.3, "survival": 1.0, "opportunity": 0.3},
    ),
    ProfileType.BALANCED: ProfileParams(
        reaction_time_min=0.12, reaction_time_max=0.40, reaction_time_mean=0.22,
        apm_target=35.0, apm_variance=10.0,
        movement_aggression=0.5, movement_smoothness=0.5,
        strafe_frequency=0.2, retreat_threshold=0.4,
        fight_willingness=0.5, target_switch_tolerance=0.5,
        super_usage_eagerness=0.5, chase_willingness=0.5,
        preferred_range=0.5, zone_commitment=0.5,
        bush_usage=0.5, burst_pattern=0.5,
        pause_frequency=0.1, pause_duration_min=0.3, pause_duration_max=1.5,
        utility_weights={},
    ),
}

# Brawler role → profile bias
BRAWLER_PROFILE_BIAS = {
    "tank": {"aggressive": 0.4, "balanced": 0.3, "nervous": 0.2, "passive": 0.1},
    "assassin": {"aggressive": 0.5, "balanced": 0.2, "nervous": 0.2, "sniper": 0.1},
    "damage": {"balanced": 0.3, "sniper": 0.3, "aggressive": 0.2, "passive": 0.2},
    "support": {"passive": 0.3, "balanced": 0.4, "nervous": 0.2, "sniper": 0.1},
    "control": {"balanced": 0.3, "sniper": 0.3, "passive": 0.2, "aggressive": 0.2},
}


class BehavioralProfile:
    """
    Manages the bot's behavioral profile for human-like play.

    Features:
    - Random profile selection at match start (weighted by brawler role)
    - Profile blending for unique combinations
    - In-match profile shifts (e.g., more aggressive when winning)
    - All parameters accessible for downstream systems
    """

    def __init__(self, brawler_role: str = "damage"):
        self.brawler_role = brawler_role
        self._current_type: ProfileType = ProfileType.BALANCED
        self._params: ProfileParams = PROFILES[ProfileType.BALANCED]
        self._blend_ratio: float = 0.0  # 0 = pure profile, 1 = fully blended
        self._blend_target: Optional[ProfileType] = None
        self._match_start_time: float = 0.0
        self._shift_schedule: List[Tuple[float, ProfileType, float]] = []  # (time_s, type, blend)
        self._lock = threading.RLock()

        logger.info("[BEHAVIORAL_PROFILE] Initialized with role=%s", brawler_role)

    def select_for_match(self, brawler_role: Optional[str] = None):
        """
        Select a profile for a new match.

        Weighted random selection based on brawler role.
        """
        if brawler_role:
            self.brawler_role = brawler_role

        # Get weights for this role
        weights = BRAWLER_PROFILE_BIAS.get(self.brawler_role, {
            "balanced": 0.4, "aggressive": 0.2, "passive": 0.2,
            "sniper": 0.1, "nervous": 0.1,
        })

        # Weighted random selection
        profiles = list(weights.keys())
        probs = list(weights.values())
        total = sum(probs)
        probs = [p / total for p in probs]

        chosen = random.choices(profiles, weights=probs, k=1)[0]
        self._current_type = ProfileType(chosen)
        self._params = PROFILES[self._current_type]
        self._match_start_time = time.time()
        self._blend_ratio = 0.0
        self._blend_target = None

        # Schedule mid-match profile shifts
        self._schedule_shifts()

        logger.info("[BEHAVIORAL_PROFILE] Selected %s for %s role",
                     self._current_type.value, self.brawler_role)

    def get_params(self) -> ProfileParams:
        """Get current profile parameters (possibly blended)."""
        with self._lock:
            if self._blend_target is not None and self._blend_ratio > 0:
                return self._blend_params()
            return self._params

    def get_type(self) -> ProfileType:
        """Get current profile type."""
        return self._current_type

    def get_reaction_time(self) -> float:
        """Get a random reaction time from current profile distribution."""
        p = self.get_params()
        # Triangular distribution for more natural feel
        return random.triangular(p.reaction_time_min, p.reaction_time_max,
                                 p.reaction_time_mean)

    def get_apm_target(self) -> float:
        """Get current APM target with variance."""
        p = self.get_params()
        return max(10.0, random.gauss(p.apm_target, p.apm_variance))

    def should_pause(self) -> Tuple[bool, float]:
        """
        Check if the bot should take a thinking pause.

        Returns (should_pause, duration_seconds).
        """
        p = self.get_params()
        if random.random() < p.pause_frequency:
            duration = random.uniform(p.pause_duration_min, p.pause_duration_max)
            return (True, duration)
        return (False, 0.0)

    def should_strafe(self) -> bool:
        """Check if the bot should strafe during combat."""
        p = self.get_params()
        return random.random() < p.strafe_frequency

    def get_retreat_threshold(self) -> float:
        """Get health threshold for retreating."""
        p = self.get_params()
        # Add some randomness per check
        return p.retreat_threshold + random.uniform(-0.05, 0.05)

    def get_fight_willingness(self) -> float:
        """Get current willingness to fight (0-1)."""
        p = self.get_params()
        return min(1.0, max(0.0, p.fight_willingness + random.uniform(-0.1, 0.1)))

    def get_utility_weights(self) -> Dict:
        """Get UtilityAI weight overrides from profile."""
        p = self.get_params()
        return p.utility_weights.copy()

    def update_match_phase(self, phase: str):
        """
        Adjust profile based on match phase.

        Early: more aggressive/farming
        Mid: balanced
        Late: more cautious (in showdown) or more aggressive (in gem grab)
        """
        with self._lock:
            if phase == "late":
                # In late game showdown, become more cautious
                if self.brawler_role in ("damage", "support"):
                    self._blend_target = ProfileType.PASSIVE
                    self._blend_ratio = 0.3
            elif phase == "early":
                # Early game, slightly more aggressive
                self._blend_target = ProfileType.AGGRESSIVE
                self._blend_ratio = 0.2
            else:
                # Mid game: return to base profile
                self._blend_ratio = 0.0
                self._blend_target = None

    def get_profile_info(self) -> Dict:
        """Get profile information for logging/display."""
        with self._lock:
            p = self.get_params()
            return {
                "type": self._current_type.value,
                "role": self.brawler_role,
                "reaction_time_ms": round(p.reaction_time_mean * 1000),
                "apm_target": round(p.apm_target),
                "aggression": round(p.movement_aggression, 2),
                "fight_willingness": round(p.fight_willingness, 2),
                "retreat_threshold": round(p.retreat_threshold, 2),
                "preferred_range": round(p.preferred_range, 2),
                "blend_target": self._blend_target.value if self._blend_target else None,
                "blend_ratio": round(self._blend_ratio, 2),
            }

    def save(self, filepath: Optional[str] = None) -> bool:
        """Save behavioral profile data to file."""
        if filepath is None:
            filepath = Path("data/behavioral_profiles.json")

        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            data = {
                "current_type": self._current_type.value,
                "brawler_role": self.brawler_role,
                "params": {
                    "reaction_time_min": self._params.reaction_time_min,
                    "reaction_time_max": self._params.reaction_time_max,
                    "reaction_time_mean": self._params.reaction_time_mean,
                    "apm_target": self._params.apm_target,
                    "apm_variance": self._params.apm_variance,
                    "movement_aggression": self._params.movement_aggression,
                    "movement_smoothness": self._params.movement_smoothness,
                    "strafe_frequency": self._params.strafe_frequency,
                    "retreat_threshold": self._params.retreat_threshold,
                    "fight_willingness": self._params.fight_willingness,
                    "target_switch_tolerance": self._params.target_switch_tolerance,
                    "super_usage_eagerness": self._params.super_usage_eagerness,
                    "chase_willingness": self._params.chase_willingness,
                    "preferred_range": self._params.preferred_range,
                    "zone_commitment": self._params.zone_commitment,
                    "bush_usage": self._params.bush_usage,
                    "burst_pattern": self._params.burst_pattern,
                    "pause_frequency": self._params.pause_frequency,
                    "pause_duration_min": self._params.pause_duration_min,
                    "pause_duration_max": self._params.pause_duration_max,
                },
                "blend_ratio": self._blend_ratio,
                "blend_target": self._blend_target.value if self._blend_target else None,
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("[BEHAVIORAL_PROFILE] Saved to %s", filepath)
            return True
        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
            logger.error("[BEHAVIORAL_PROFILE] Failed to save: %s", e)
            return False

    def load(self, filepath: Optional[str] = None) -> bool:
        """Load behavioral profile data from file."""
        if filepath is None:
            filepath = Path("data/behavioral_profiles.json")

        if not Path(filepath).exists():
            logger.debug("[BEHAVIORAL_PROFILE] No save file found at %s", filepath)
            return False

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self._current_type = ProfileType(data["current_type"])
            self.brawler_role = data["brawler_role"]
            self._blend_ratio = data.get("blend_ratio", 0.0)
            blend_target = data.get("blend_target")
            self._blend_target = ProfileType(blend_target) if blend_target else None

            params_data = data["params"]
            self._params = ProfileParams(**params_data)

            logger.info("[BEHAVIORAL_PROFILE] Loaded from %s", filepath)
            return True
        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
            logger.error("[BEHAVIORAL_PROFILE] Failed to load: %s", e)
            return False

    # --- Internal ---

    def _schedule_shifts(self):
        """Schedule mid-match profile shifts for variety."""
        self._shift_schedule = []

        # 30% chance of a shift at ~60s into the match
        if random.random() < 0.3:
            shift_type = random.choice([ProfileType.AGGRESSIVE, ProfileType.PASSIVE,
                                         ProfileType.BALANCED])
            self._shift_schedule.append((60.0, shift_type, 0.3))

        # 20% chance of a shift at ~120s
        if random.random() < 0.2:
            shift_type = random.choice([ProfileType.NERVOUS, ProfileType.SNIPER])
            self._shift_schedule.append((120.0, shift_type, 0.2))

    def _blend_params(self) -> ProfileParams:
        """Blend current params with target profile."""
        if self._blend_target is None:
            return self._params

        target = PROFILES[self._blend_target]
        ratio = self._blend_ratio

        # Linear interpolation of all numeric fields
        blended = ProfileParams()
        for attr in vars(self._params):
            if attr.startswith('_'):
                continue
            current_val = getattr(self._params, attr)
            target_val = getattr(target, attr)

            if isinstance(current_val, (int, float)):
                setattr(blended, attr, current_val * (1 - ratio) + target_val * ratio)
            elif isinstance(current_val, dict):
                # Blend dicts (utility_weights)
                merged = current_val.copy()
                for k, v in target_val.items():
                    if k in merged:
                        merged[k] = merged[k] * (1 - ratio) + v * ratio
                    else:
                        merged[k] = v * ratio
                setattr(blended, attr, merged)

        return blended
