"""
decision/utility_ai.py

Utility AI System for Brawl Stars bot.

Replaces the reactive "frame → detection → action" pattern with
scored action selection. Each possible action gets a utility score
based on the current world state, and the highest-scoring action wins.

This is MUCH better than Q-Learning for this type of game because:
- Decisions are transparent and debuggable
- No training needed — works immediately
- Easy to add new actions or modify behavior
- Naturally handles context (health, ammo, enemies, map, phase)
- Can be weighted by behavioral profiles

Actions (UnifiedAction from core.class_registry):
- ATTACK: Shoot at a target
- RETREAT: Move away from threats
- COLLECT_CUBE: Pick up a power cube
- TAKE_COVER: Move to bush/wall for cover
- HOLD_POSITION: Stay in current position (zone control)
- HEAL_UP: Stay in safe area to regenerate
- AMBUSH: Wait in bush for enemy to approach
- CHASE: Pursue a low-health enemy
- KITE: Attack while moving away (shoot-and-scoot)
- USE_SUPER: Activate super ability

Migration:
    This module now uses UnifiedAction from core.class_registry for
    consistency with rl_engine.py and future neural policy.
    The old Action enum is deprecated; use UnifiedAction directly.
"""

import logging
import time
from dataclasses import dataclass

from core.class_registry import UnifiedAction

# Import unified action space
from core.class_registry import UnifiedAction as Action

logger = logging.getLogger(__name__)

# Re-export Action as UnifiedAction for backward compatibility
__all__ = ["UtilityAI", "Action", "ActionScore", "UnifiedAction"]


@dataclass
class ActionScore:
    """Scored action with reasoning."""
    action: Action
    score: float
    target_position: tuple[float, float] | None = None
    target_enemy_id: int | None = None
    reasoning: str = ""
    urgency: float = 0.0  # 0=can_wait, 1=must_act_now


class UtilityAI:
    """
    Utility-based decision making system.

    Each frame, evaluates all possible actions and selects the one
    with the highest utility score. Scores are computed from world
    state factors like health, ammo, enemy proximity, match phase, etc.

    Integrates with:
    - WorldModel (for enemy memory, pressure, danger zones)
    - OccupancyGrid (for cover, pathfinding)
    - IntentSystem (for persistent goals)
    - BehavioralProfile (for play style weighting)
    """

    # Default weight configuration
    DEFAULT_WEIGHTS = {
        "survival": 1.0,      # Staying alive
        "aggression": 0.5,    # Dealing damage / killing
        "farm": 0.7,          # Collecting power cubes
        "control": 0.3,       # Zone control
        "opportunity": 0.6,   # Taking advantage of enemy mistakes
    }

    # Action-specific weight multipliers per behavioral profile
    PROFILE_WEIGHTS = {
        "aggressive": {
            Action.ATTACK: 1.5, Action.CHASE: 1.3, Action.KITE: 1.2,
            Action.RETREAT: 0.5, Action.HEAL_UP: 0.4, Action.TAKE_COVER: 0.3,
        },
        "passive": {
            Action.ATTACK: 0.6, Action.RETREAT: 1.4, Action.TAKE_COVER: 1.3,
            Action.HEAL_UP: 1.2, Action.CHASE: 0.3, Action.KITE: 0.8,
        },
        "sniper": {
            Action.ATTACK: 1.2, Action.KITE: 1.5, Action.HOLD_POSITION: 1.3,
            Action.AMBUSH: 1.2, Action.CHASE: 0.2, Action.RETREAT: 0.9,
        },
        "nervous": {
            Action.RETREAT: 1.5, Action.TAKE_COVER: 1.4, Action.HEAL_UP: 1.3,
            Action.ATTACK: 0.5, Action.CHASE: 0.2, Action.KITE: 0.7,
        },
        "balanced": {
            # All 1.0 — no modification
        },
    }

    def __init__(self, profile: str = "balanced",
                 weights: dict | None = None):
        self.profile = profile
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._last_action: Action | None = None
        self._last_action_time: float = 0.0
        self._action_history: list[ActionScore] = []

    def evaluate(self, context: dict) -> ActionScore:
        """
        Evaluate all actions and return the best one.

        Args:
            context: Dict containing world state:
                - health: float (0-1)
                - ammo: int (0-3)
                - max_ammo: int
                - super_charged: bool
                - enemies_nearby: int
                - nearest_enemy_dist: float
                - nearest_enemy_health: float
                - player_position: (x, y)
                - pressure: float (from WorldModel)
                - danger: float (from WorldModel)
                - has_cover_nearby: bool
                - cover_position: (x, y) or None
                - nearest_cube_dist: float or None
                - cube_position: (x, y) or None
                - match_phase: str ("early", "mid", "late")
                - brawler_role: str ("tank", "assassin", "support", "damage", "control")
                - intent: str (from IntentSystem, optional)

        Returns:
            ActionScore with the highest-scoring action
        """
        scores = []

        health = context.get("health", 1.0)
        ammo = context.get("ammo", 3)
        max_ammo = context.get("max_ammo", 3)
        super_charged = context.get("super_charged", False)
        enemies_nearby = context.get("enemies_nearby", 0)
        nearest_dist = context.get("nearest_enemy_dist", 999.0)
        nearest_health = context.get("nearest_enemy_health", 1.0)
        pressure = context.get("pressure", 0.0)
        danger = context.get("danger", 0.0)
        has_cover = context.get("has_cover_nearby", False)
        context.get("cover_position")
        cube_dist = context.get("nearest_cube_dist")
        context.get("cube_position")
        phase = context.get("match_phase", "early")
        role = context.get("brawler_role", "damage")
        intent = context.get("intent", None)
        game_mode = context.get("game_mode", "showdown")

        # --- Score each action ---

        # ATTACK: Want to shoot at enemies
        attack_score = self._score_attack(
            health, ammo, max_ammo, enemies_nearby, nearest_dist,
            nearest_health, pressure, role
        )
        scores.append(attack_score)

        # RETREAT: Move away from threats
        retreat_score = self._score_retreat(
            health, enemies_nearby, pressure, danger, nearest_dist, role
        )
        scores.append(retreat_score)

        # COLLECT_CUBE: Pick up power cubes
        cube_score = self._score_collect_cube(
            health, cube_dist, enemies_nearby, phase, nearest_dist
        )
        scores.append(cube_score)

        # TAKE_COVER: Move to bush/wall
        cover_score = self._score_take_cover(
            health, has_cover, pressure, enemies_nearby, danger, role
        )
        scores.append(cover_score)

        # HOLD_POSITION: Stay put (zone control)
        hold_score = self._score_hold_position(
            health, pressure, enemies_nearby, phase, role
        )
        scores.append(hold_score)

        # HEAL_UP: Regenerate health
        heal_score = self._score_heal_up(
            health, enemies_nearby, pressure, nearest_dist
        )
        scores.append(heal_score)

        # AMBUSH: Wait in bush for enemy
        ambush_score = self._score_ambush(
            health, has_cover, enemies_nearby, nearest_dist, role
        )
        scores.append(ambush_score)

        # CHASE: Pursue low-health enemy
        chase_score = self._score_chase(
            health, nearest_health, nearest_dist, enemies_nearby, role
        )
        scores.append(chase_score)

        # KITE: Attack while retreating
        kite_score = self._score_kite(
            health, ammo, enemies_nearby, nearest_dist, pressure, role
        )
        scores.append(kite_score)

        # USE_SUPER: Activate super ability
        super_score = self._score_use_super(
            health, super_charged, enemies_nearby, nearest_dist, role
        )
        scores.append(super_score)

        # Apply profile weights
        profile_mods = self.PROFILE_WEIGHTS.get(self.profile, {})
        for s in scores:
            mod = profile_mods.get(s.action, 1.0)
            s.score *= mod

        # Apply intent bonus (if IntentSystem says we should be doing something)
        if intent:
            intent_bonus = {
                "farm": {Action.COLLECT_CUBE: 1.5},
                "survive": {Action.RETREAT: 1.5, Action.TAKE_COVER: 1.3, Action.HEAL_UP: 1.3},
                "aggressive": {Action.ATTACK: 1.5, Action.CHASE: 1.3, Action.USE_SUPER: 1.2},
                "control": {Action.HOLD_POSITION: 1.5, Action.KITE: 1.2},
                "ambush": {Action.AMBUSH: 1.5, Action.TAKE_COVER: 1.3},
            }.get(intent, {})
            for s in scores:
                s.score *= intent_bonus.get(s.action, 1.0)

        # Apply game mode bonuses (objective-aware tactical bias)
        mode_bonus = self._get_mode_bonus(game_mode)
        if mode_bonus:
            for s in scores:
                s.score *= mode_bonus.get(s.action, 1.0)

        # Select best action
        scores.sort(key=lambda s: s.score, reverse=True)
        best = scores[0]

        # Log decision
        logger.debug(
            "[UTILITY_AI] Decision: %s (%.2f) | Top 3: %s",
            best.action.value, best.score,
            [(s.action.value, f"{s.score:.2f}") for s in scores[:3]]
        )

        self._last_action = best.action
        self._last_action_time = time.time()
        self._action_history.append(best)
        if len(self._action_history) > 100:
            self._action_history = self._action_history[-100:]

        return best

    def _get_mode_bonus(self, game_mode: str) -> dict[Action, float]:
        """Return lightweight action multipliers for the current game mode."""
        mode = (game_mode or "showdown").lower().replace(" ", "_")
        bonuses: dict[str, dict[Action, float]] = {
            "showdown": {
                Action.COLLECT_CUBE: 1.30,
                Action.RETREAT: 1.15,
                Action.TAKE_COVER: 1.10,
                Action.HEAL_UP: 1.10,
            },
            "gem_grab": {
                Action.HOLD_POSITION: 1.45,
                Action.TAKE_COVER: 1.20,
                Action.RETREAT: 1.10,
                Action.AMBUSH: 1.10,
            },
            "brawl_ball": {
                Action.ATTACK: 1.20,
                Action.CHASE: 1.15,
                Action.USE_SUPER: 1.15,
                Action.HOLD_POSITION: 1.05,
            },
            "heist": {
                Action.ATTACK: 1.25,
                Action.CHASE: 1.15,
                Action.USE_SUPER: 1.20,
            },
            "bounty": {
                Action.RETREAT: 1.20,
                Action.TAKE_COVER: 1.20,
                Action.HEAL_UP: 1.15,
                Action.HOLD_POSITION: 1.10,
            },
            "knockout": {
                Action.RETREAT: 1.20,
                Action.TAKE_COVER: 1.15,
                Action.HEAL_UP: 1.15,
                Action.HOLD_POSITION: 1.10,
            },
            "hot_zone": {
                Action.HOLD_POSITION: 1.45,
                Action.TAKE_COVER: 1.15,
                Action.AMBUSH: 1.10,
            },
        }
        return bonuses.get(mode, {})

    def get_last_action(self) -> Action | None:
        return self._last_action

    def get_action_history(self, limit: int = 20) -> list[dict]:
        return [
            {"action": s.action.value, "score": round(s.score, 2), "reasoning": s.reasoning}
            for s in self._action_history[-limit:]
        ]

    # --- Individual action scorers ---

    def _score_attack(self, health, ammo, max_ammo, enemies, dist, enemy_hp, pressure, role) -> ActionScore:
        """Attack is good when: healthy, have ammo, enemy is close, low pressure."""
        score = 0.0
        reasons = []

        # Must have enemies to attack
        if enemies == 0:
            return ActionScore(Action.ATTACK, 0.0, reasoning="No enemies")

        # Ammo factor
        ammo_ratio = ammo / max(1, max_ammo)
        if ammo == 0:
            return ActionScore(Action.ATTACK, 0.05, reasoning="No ammo")
        score += ammo_ratio * 0.3
        reasons.append(f"ammo={ammo_ratio:.0%}")

        # Health factor: attack more when healthy
        score += health * 0.2
        reasons.append(f"hp={health:.0%}")

        # Distance factor: prefer medium range
        if dist < 200:
            score += 0.3 if role in ("tank", "assassin") else 0.1
        elif dist < 400:
            score += 0.3 if role in ("damage", "control") else 0.2
        else:
            score += 0.1 if role == "sniper" else 0.05

        # Enemy health: prefer low-health targets
        if enemy_hp < 0.3:
            score += 0.3
            reasons.append("low_hp_target")
        elif enemy_hp < 0.6:
            score += 0.15

        # Pressure penalty: don't attack when under heavy pressure
        if pressure > 3.0:
            score *= 0.5
            reasons.append("high_pressure")

        score *= self.weights.get("aggression", 0.5)

        return ActionScore(
            Action.ATTACK, score,
            reasoning=" | ".join(reasons) if reasons else "standard_attack",
            urgency=0.6 if enemy_hp < 0.3 else 0.3,
        )

    def _score_retreat(self, health, enemies, pressure, danger, dist, role) -> ActionScore:
        """Retreat is good when: low health, high pressure, close enemies."""
        score = 0.0
        reasons = []

        # Health urgency
        if health < 0.2:
            score += 0.8
            reasons.append("CRITICAL_HP")
        elif health < 0.4:
            score += 0.5
            reasons.append("low_hp")
        elif health < 0.6:
            score += 0.2
        else:
            score += 0.05

        # Pressure
        if pressure > 5.0:
            score += 0.4
            reasons.append("overwhelmed")
        elif pressure > 2.0:
            score += 0.2

        # Danger zones
        if danger > 2.0:
            score += 0.3
            reasons.append("in_danger_zone")

        # Close enemies (especially for non-tanks)
        if dist < 150 and role not in ("tank",):
            score += 0.3
            reasons.append("too_close")

        # Role modifier
        if role == "assassin":
            score *= 0.7  # Assassins retreat less

        score *= self.weights.get("survival", 1.0)

        return ActionScore(
            Action.RETREAT, score,
            reasoning=" | ".join(reasons) if reasons else "safe_retreat",
            urgency=0.9 if health < 0.2 else 0.5,
        )

    def _score_collect_cube(self, health, cube_dist, enemies, phase, nearest_dist) -> ActionScore:
        """Collect cubes when: safe, early game, cube is close."""
        score = 0.0
        reasons = []

        if cube_dist is None or cube_dist > 500:
            return ActionScore(Action.COLLECT_CUBE, 0.0, reasoning="No cube visible")

        # Phase bonus
        if phase == "early":
            score += 0.5
            reasons.append("early_game")
        elif phase == "mid":
            score += 0.2
        else:
            score += 0.05  # Late game cubes less important

        # Distance bonus (closer = better)
        if cube_dist < 100:
            score += 0.4
        elif cube_dist < 200:
            score += 0.3
        elif cube_dist < 400:
            score += 0.1

        # Safety: don't collect if enemies are very close
        if nearest_dist < 150:
            score *= 0.3
            reasons.append("enemy_too_close")
        elif enemies > 2:
            score *= 0.5
            reasons.append("many_enemies")

        # Health: don't collect when critically low
        if health < 0.3:
            score *= 0.4

        score *= self.weights.get("farm", 0.7)

        return ActionScore(
            Action.COLLECT_CUBE, score,
            target_position=None,
            reasoning=" | ".join(reasons) if reasons else "cube_available",
            urgency=0.3,
        )

    def _score_take_cover(self, health, has_cover, pressure, enemies, danger, role) -> ActionScore:
        """Take cover when: low health, cover available, under pressure."""
        if not has_cover:
            return ActionScore(Action.TAKE_COVER, 0.0, reasoning="No cover available")

        score = 0.0
        reasons = []

        # Health factor
        if health < 0.4:
            score += 0.5
            reasons.append("low_hp")
        elif health < 0.6:
            score += 0.3

        # Pressure
        if pressure > 3.0:
            score += 0.4
            reasons.append("high_pressure")

        # Danger zone
        if danger > 1.0:
            score += 0.2

        # Role: supports and snipers prefer cover
        if role in ("support", "damage"):
            score += 0.1

        score *= self.weights.get("survival", 1.0)

        return ActionScore(
            Action.TAKE_COVER, score,
            reasoning=" | ".join(reasons) if reasons else "cover_available",
            urgency=0.6 if health < 0.3 else 0.3,
        )

    def _score_hold_position(self, health, pressure, enemies, phase, role) -> ActionScore:
        """Hold position when: healthy, low pressure, zone control matters."""
        score = 0.0

        if pressure > 2.0:
            return ActionScore(Action.HOLD_POSITION, 0.05, reasoning="Under pressure")

        if health > 0.6:
            score += 0.2

        if phase == "mid":
            score += 0.2  # Zone control important mid-game

        if role in ("support", "control"):
            score += 0.15

        score *= self.weights.get("control", 0.3)

        return ActionScore(Action.HOLD_POSITION, score, reasoning="stable_position", urgency=0.1)

    def _score_heal_up(self, health, enemies, pressure, nearest_dist) -> ActionScore:
        """Heal up when: very low health, safe area, no immediate threats."""
        score = 0.0
        reasons = []

        if health > 0.7:
            return ActionScore(Action.HEAL_UP, 0.0, reasoning="Healthy enough")

        if health < 0.3:
            score += 0.6
            reasons.append("critical_hp")
        elif health < 0.5:
            score += 0.3
            reasons.append("low_hp")

        # Only heal if relatively safe
        if enemies == 0:
            score += 0.3
            reasons.append("no_enemies")
        elif nearest_dist > 300:
            score += 0.15
        else:
            score *= 0.3  # Don't try to heal with enemies close
            reasons.append("enemies_close")

        if pressure < 1.0:
            score += 0.1

        score *= self.weights.get("survival", 1.0)

        return ActionScore(
            Action.HEAL_UP, score,
            reasoning=" | ".join(reasons) if reasons else "need_healing",
            urgency=0.8 if health < 0.2 else 0.4,
        )

    def _score_ambush(self, health, has_cover, enemies, nearest_dist, role) -> ActionScore:
        """Ambush when: in cover, enemy approaching, healthy enough."""
        if not has_cover:
            return ActionScore(Action.AMBUSH, 0.0, reasoning="No cover for ambush")

        score = 0.0
        reasons = []

        if health < 0.5:
            return ActionScore(Action.AMBUSH, 0.05, reasoning="Too risky at low HP")

        # Enemy approaching (medium distance is ideal for ambush)
        if 150 < nearest_dist < 350:
            score += 0.4
            reasons.append("enemy_approaching")
        elif nearest_dist < 150:
            score += 0.2  # Already close, ambush less useful
        else:
            score += 0.1

        # Role bonus
        if role in ("assassin", "tank"):
            score += 0.2
            reasons.append(f"role={role}")

        return ActionScore(
            Action.AMBUSH, score,
            reasoning=" | ".join(reasons) if reasons else "ambush_opportunity",
            urgency=0.2,
        )

    def _score_chase(self, health, enemy_hp, nearest_dist, enemies, role) -> ActionScore:
        """Chase when: enemy is low health, we're healthy, not too many enemies."""
        score = 0.0
        reasons = []

        if enemies == 0:
            return ActionScore(Action.CHASE, 0.0, reasoning="No enemies")

        # Only chase low-health enemies
        if enemy_hp > 0.4:
            return ActionScore(Action.CHASE, 0.05, reasoning="Enemy not low HP")

        score += 0.3
        reasons.append("low_hp_target")

        # Must be healthy enough to chase
        if health < 0.5:
            score *= 0.3
            reasons.append("risky_hp")
        elif health > 0.7:
            score += 0.2

        # Don't chase into multiple enemies
        if enemies > 2:
            score *= 0.4
            reasons.append("multiple_enemies")

        # Distance: don't chase if too far
        if nearest_dist > 400:
            score *= 0.5

        # Role bonus
        if role == "assassin":
            score += 0.2
        elif role == "tank":
            score += 0.1

        score *= self.weights.get("opportunity", 0.6)

        return ActionScore(
            Action.CHASE, score,
            reasoning=" | ".join(reasons) if reasons else "kill_opportunity",
            urgency=0.7,
        )

    def _score_kite(self, health, ammo, enemies, nearest_dist, pressure, role) -> ActionScore:
        """Kite (attack while retreating) when: moderate health, enemies close, have ammo."""
        score = 0.0
        reasons = []

        if enemies == 0:
            return ActionScore(Action.KITE, 0.0, reasoning="No enemies")
        if ammo == 0:
            return ActionScore(Action.KITE, 0.05, reasoning="No ammo for kiting")

        # Kite is best at moderate health
        if 0.3 < health < 0.7:
            score += 0.3
            reasons.append("moderate_hp")
        elif health < 0.3:
            score += 0.1  # Still kite but risky
        else:
            score += 0.15  # Healthy enough to just attack instead

        # Pressure: kite when under some pressure
        if 1.0 < pressure < 4.0:
            score += 0.3
            reasons.append("moderate_pressure")
        elif pressure >= 4.0:
            score += 0.1  # Too much pressure, just retreat

        # Distance: kite at medium range
        if 150 < nearest_dist < 350:
            score += 0.2

        # Role bonus
        if role in ("damage", "control"):
            score += 0.2
            reasons.append(f"kite_role={role}")

        return ActionScore(
            Action.KITE, score,
            reasoning=" | ".join(reasons) if reasons else "kite_opportunity",
            urgency=0.5,
        )

    def _score_use_super(self, health, super_charged, enemies, nearest_dist, role) -> ActionScore:
        """Use super when: charged, enemies nearby, tactical opportunity."""
        if not super_charged:
            return ActionScore(Action.USE_SUPER, 0.0, reasoning="Super not charged")

        score = 0.0
        reasons = []

        if enemies == 0:
            return ActionScore(Action.USE_SUPER, 0.05, reasoning="No targets")

        # Multiple enemies = great super opportunity
        if enemies >= 2:
            score += 0.5
            reasons.append("multiple_targets")
        else:
            score += 0.2

        # Close range for most supers
        if nearest_dist < 250:
            score += 0.3
            reasons.append("in_range")

        # Don't waste super at full health unless multiple enemies
        if health > 0.7 and enemies < 2:
            score *= 0.5
            reasons.append("save_super")

        # Role: tanks and assassins benefit most from super
        if role in ("tank", "assassin"):
            score += 0.15

        return ActionScore(
            Action.USE_SUPER, score,
            reasoning=" | ".join(reasons) if reasons else "super_available",
            urgency=0.8 if enemies >= 2 else 0.4,
        )
