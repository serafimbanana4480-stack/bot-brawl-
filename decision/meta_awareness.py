"""
decision/meta_awareness.py

Meta-Awareness Combat System for Brawl Stars bot.

Solves the "meta-awareness only in selection" problem. Counter-picks
exist in brawler_selector.py but aren't used during actual combat.
This module provides matchup-aware combat behavior:

- Knows which brawlers counter which
- Adjusts aggression/retreat based on matchup advantage
- Knows enemy super status and adjusts behavior
- Tracks team composition for synergy awareness
- Provides matchup-specific combat tips

Usage:
    meta = MetaAwareness()

    # Before combat:
    matchup = meta.evaluate_matchup("colt", "mortis")
    # matchup.advantage = -0.3 (colt loses to mortis close range)
    # matchup.tips = ["keep_distance", "kite_away"]

    # During combat:
    adjustment = meta.get_combat_adjustment(our_brawler, enemy_brawler, context)
    # adjustment.aggression_modifier = -0.3 (be less aggressive)
    # adjustment.preferred_distance = 400 (stay far)
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MatchupAdvantage(Enum):
    STRONG_ADVANTAGE = 2     # We hard-counter them
    ADVANTAGE = 1            # We have the edge
    EVEN = 0                 # 50/50
    DISADVANTAGE = -1        # They have the edge
    STRONG_DISADVANTAGE = -2 # We hard-lose


@dataclass
class MatchupResult:
    """Result of a matchup evaluation."""
    our_brawler: str
    enemy_brawler: str
    advantage: float = 0.0           # -1.0 to 1.0
    advantage_level: MatchupAdvantage = MatchupAdvantage.EVEN
    preferred_distance: float = 300.0  # Optimal engagement distance
    tips: list[str] = None           # Combat tips
    danger_level: float = 0.5       # How dangerous this enemy is to us

    def __post_init__(self):
        if self.tips is None:
            self.tips = []


@dataclass
class CombatAdjustment:
    """Adjustments to combat behavior based on matchup."""
    aggression_modifier: float = 0.0    # Add to aggression (-1 to 1)
    retreat_threshold_modifier: float = 0.0  # Add to retreat threshold
    preferred_distance: float = 300.0   # Optimal distance to maintain
    should_kite: bool = False           # Should we kite this enemy
    should_bait: bool = False           # Should we bait this enemy
    should_rush: bool = False           # Should we rush this enemy
    super_priority: float = 0.5         # How important to save/use super
    focus_priority: float = 0.5         # How important to focus this target
    tips: list[str] = None

    def __post_init__(self):
        if self.tips is None:
            self.tips = []


# Matchup table: (our_brawler, enemy_brawler) → advantage
# Positive = we win, negative = we lose
# This is a simplified version — real matchup data would be more detailed
MATCHUP_TABLE = {
    # Assassins vs ranged
    ("mortis", "colt"): 0.6,
    ("mortis", "piper"): 0.7,
    ("mortis", "brock"): 0.5,
    ("mortis", "tick"): 0.8,
    ("edgar", "colt"): 0.5,
    ("edgar", "piper"): 0.6,
    ("edgar", "brock"): 0.4,
    ("leon", "piper"): 0.5,
    ("leon", "colt"): 0.4,
    ("fang", "colt"): 0.3,

    # Ranged vs assassins (at range)
    ("colt", "mortis"): -0.4,
    ("piper", "mortis"): -0.5,
    ("piper", "edgar"): -0.4,
    ("colt", "edgar"): -0.3,
    ("brock", "mortis"): -0.3,

    # Tanks vs assassins
    ("el_primo", "mortis"): 0.3,
    ("bull", "mortis"): 0.2,
    ("rosa", "mortis"): 0.4,
    ("frank", "mortis"): -0.2,

    # Tanks vs ranged
    ("el_primo", "colt"): -0.3,
    ("el_primo", "piper"): -0.4,
    ("bull", "piper"): -0.3,
    ("rosa", "piper"): -0.1,

    # Snipers vs tanks
    ("piper", "el_primo"): 0.4,
    ("piper", "bull"): 0.3,
    ("brock", "el_primo"): 0.3,
    ("colt", "el_primo"): 0.3,

    # Support vs assassins
    ("poco", "mortis"): -0.5,
    ("poco", "edgar"): -0.4,
    ("byron", "mortis"): -0.3,

    # Control vs tanks
    ("sprout", "el_primo"): 0.3,
    ("sprout", "bull"): 0.2,
    ("lou", "el_primo"): 0.2,

    # Shelly (shotgun) — strong close range
    ("shelly", "el_primo"): 0.4,
    ("shelly", "mortis"): 0.3,
    ("shelly", "colt"): -0.2,
    ("shelly", "piper"): -0.3,

    # Dynamike — area denial
    ("dynamike", "el_primo"): 0.3,
    ("dynamike", "bull"): 0.2,
    ("dynamike", "mortis"): -0.2,

    # Tick — long range poke
    ("tick", "el_primo"): 0.4,
    ("tick", "bull"): 0.3,
    ("tick", "mortis"): -0.6,
    ("tick", "edgar"): -0.7,
}

# Brawler preferred engagement distances (pixels, approximate)
PREFERRED_DISTANCES = {
    "el_primo": 100, "bull": 120, "shelly": 130, "rosa": 140,
    "mortis": 120, "edgar": 150, "fang": 180, "leon": 200,
    "frank": 200, "jacky": 130,
    "colt": 350, "piper": 400, "brock": 380, "nita": 300,
    "dynamike": 300, "tick": 450, "penny": 350, "jessie": 320,
    "sprout": 380, "lou": 300, "byron": 350, "poco": 250,
}

# Brawler danger levels when they have super
SUPER_DANGER_BRAWLERS = {
    "el_primo": 0.9, "shelly": 0.9, "bull": 0.8, "frank": 0.9,
    "mortis": 0.7, "edgar": 0.8, "fang": 0.8, "leon": 0.6,
    "piper": 0.7, "colt": 0.6, "brock": 0.5, "tick": 0.4,
    "penny": 0.6, "sprout": 0.5, "nita": 0.6,
}

# Brawler roles
BRAWLER_ROLES = {
    "el_primo": "tank", "bull": "tank", "rosa": "tank", "frank": "tank",
    "jacky": "tank",
    "mortis": "assassin", "edgar": "assassin", "leon": "assassin",
    "fang": "assassin",
    "colt": "damage", "piper": "damage", "brock": "damage", "nita": "damage",
    "shelly": "damage", "dynamike": "damage", "tick": "damage",
    "penny": "control", "jessie": "control", "sprout": "control", "lou": "control",
    "poco": "support", "byron": "support",
}


class MetaAwareness:
    """
    Meta-awareness combat system.

    Uses matchup knowledge to adjust combat behavior in real-time.
    Goes beyond brawler selection to actually fight differently
    based on who we're fighting.
    """

    def __init__(self):
        self._matchup_cache: dict[tuple[str, str], MatchupResult] = {}
        logger.info("[META_AWARENESS] Initialized with %d matchups",
                     len(MATCHUP_TABLE))

    def evaluate_matchup(self, our_brawler: str, enemy_brawler: str) -> MatchupResult:
        """
        Evaluate the matchup between our brawler and an enemy brawler.

        Returns a MatchupResult with advantage, tips, and preferred distance.
        """
        our = our_brawler.lower().replace(" ", "_")
        enemy = enemy_brawler.lower().replace(" ", "_")

        # Check cache
        key = (our, enemy)
        if key in self._matchup_cache:
            return self._matchup_cache[key]

        # Look up matchup
        advantage = MATCHUP_TABLE.get(key, 0.0)

        # Also check reverse matchup
        reverse_key = (enemy, our)
        reverse = MATCHUP_TABLE.get(reverse_key, 0.0)
        if reverse != 0.0 and advantage == 0.0:
            advantage = -reverse  # If they beat us, we have negative advantage

        # Determine advantage level
        if advantage >= 0.5:
            level = MatchupAdvantage.STRONG_ADVANTAGE
        elif advantage >= 0.2:
            level = MatchupAdvantage.ADVANTAGE
        elif advantage <= -0.5:
            level = MatchupAdvantage.STRONG_DISADVANTAGE
        elif advantage <= -0.2:
            level = MatchupAdvantage.DISADVANTAGE
        else:
            level = MatchupAdvantage.EVEN

        # Get preferred distance
        our_dist = PREFERRED_DISTANCES.get(our, 300)
        enemy_dist = PREFERRED_DISTANCES.get(enemy, 300)

        # If we have advantage, fight at our preferred distance
        # If disadvantage, fight at distance that denies their advantage
        if advantage > 0:
            preferred = our_dist
        elif advantage < 0:
            # Try to fight at distance that's bad for them
            if enemy_dist < 200:
                preferred = 400  # Keep them at range
            else:
                preferred = 150  # Get in their face
        else:
            preferred = our_dist

        # Generate tips
        tips = self._generate_tips(our, enemy, advantage, level)

        # Danger level
        danger = 0.5 - advantage * 0.3
        danger = max(0.1, min(1.0, danger))

        result = MatchupResult(
            our_brawler=our,
            enemy_brawler=enemy,
            advantage=advantage,
            advantage_level=level,
            preferred_distance=preferred,
            tips=tips,
            danger_level=danger,
        )

        self._matchup_cache[key] = result
        return result

    def get_combat_adjustment(self, our_brawler: str, enemy_brawler: str,
                               context: dict = None) -> CombatAdjustment:
        """
        Get combat behavior adjustments for a specific matchup.

        Args:
            our_brawler: Our brawler name
            enemy_brawler: Enemy brawler name
            context: Optional dict with enemy_has_super, our_health, distance

        Returns:
            CombatAdjustment with modified behavior parameters
        """
        matchup = self.evaluate_matchup(our_brawler, enemy_brawler)
        ctx = context or {}

        adj = CombatAdjustment(
            preferred_distance=matchup.preferred_distance,
            tips=matchup.tips.copy(),
        )

        # Aggression modifier based on matchup
        adj.aggression_modifier = matchup.advantage * 0.3

        # Retreat threshold: retreat earlier against bad matchups
        if matchup.advantage < -0.3:
            adj.retreat_threshold_modifier = 0.15
        elif matchup.advantage > 0.3:
            adj.retreat_threshold_modifier = -0.1

        # Should we kite?
        our_role = BRAWLER_ROLES.get(our_brawler.lower().replace(" ", "_"), "damage")
        enemy_role = BRAWLER_ROLES.get(enemy_brawler.lower().replace(" ", "_"), "damage")

        if enemy_role == "assassin" and our_role in ("damage", "control", "support"):
            adj.should_kite = True
            adj.preferred_distance = max(adj.preferred_distance, 350)

        if enemy_role == "tank" and our_role in ("damage", "control"):
            adj.should_kite = True
            adj.kite_away = True

        # Should we rush?
        if our_role == "assassin" and enemy_role in ("damage", "control", "support"):
            adj.should_rush = True
        if our_role == "tank" and enemy_role in ("damage", "control"):
            adj.should_rush = True

        # Should we bait?
        if matchup.advantage < -0.2 and our_role in ("damage", "control"):
            adj.should_bait = True

        # Super priority
        enemy_has_super = ctx.get("enemy_has_super", False)
        if enemy_has_super:
            super_danger = SUPER_DANGER_BRAWLERS.get(
                enemy_brawler.lower().replace(" ", "_"), 0.5
            )
            if super_danger > 0.7:
                adj.retreat_threshold_modifier += 0.2
                adj.super_priority = 0.8  # Use our super defensively

        # Focus priority: focus enemies we counter
        adj.focus_priority = 0.5 + matchup.advantage * 0.3

        return adj

    def get_team_analysis(self, our_team: list[str], enemy_team: list[str]) -> dict:
        """
        Analyze team compositions for synergy and counter potential.
        """
        our_roles = [BRAWLER_ROLES.get(b.lower().replace(" ", "_"), "damage") for b in our_team]
        enemy_roles = [BRAWLER_ROLES.get(b.lower().replace(" ", "_"), "damage") for b in enemy_team]

        # Team balance
        our_balance = {
            "tanks": our_roles.count("tank"),
            "assassins": our_roles.count("assassin"),
            "damage": our_roles.count("damage"),
            "control": our_roles.count("control"),
            "support": our_roles.count("support"),
        }

        enemy_balance = {
            "tanks": enemy_roles.count("tank"),
            "assassins": enemy_roles.count("assassin"),
            "damage": enemy_roles.count("damage"),
            "control": enemy_roles.count("control"),
            "support": enemy_roles.count("support"),
        }

        # Overall matchup score
        total_advantage = 0.0
        matchups = 0
        for our in our_team:
            for enemy in enemy_team:
                m = self.evaluate_matchup(our, enemy)
                total_advantage += m.advantage
                matchups += 1

        avg_advantage = total_advantage / max(1, matchups)

        return {
            "our_composition": our_balance,
            "enemy_composition": enemy_balance,
            "overall_advantage": round(avg_advantage, 2),
            "team_size": len(our_team),
        }

    def _generate_tips(self, our: str, enemy: str, advantage: float,
                        level: MatchupAdvantage) -> list[str]:
        """Generate matchup-specific combat tips."""
        tips = []
        our_role = BRAWLER_ROLES.get(our, "damage")
        enemy_role = BRAWLER_ROLES.get(enemy, "damage")

        if level == MatchupAdvantage.STRONG_ADVANTAGE:
            tips.append("push_advantage")
            tips.append("fight_at_our_range")
        elif level == MatchupAdvantage.ADVANTAGE:
            tips.append("slight_edge")
            tips.append("play_standard")
        elif level == MatchupAdvantage.DISADVANTAGE:
            tips.append("play_careful")
            if enemy_role == "assassin":
                tips.append("keep_distance")
                tips.append("kite_away")
            elif enemy_role == "tank":
                tips.append("maintain_range")
            tips.append("wait_for_mistake")
        elif level == MatchupAdvantage.STRONG_DISADVANTAGE:
            tips.append("avoid_fight")
            tips.append("retreat_early")
            if enemy_role == "assassin":
                tips.append("never_alone")
                tips.append("stay_near_team")
            elif enemy_role == "tank":
                tips.append("keep_max_distance")

        # Role-specific tips
        if our_role == "assassin" and enemy_role in ("damage", "control"):
            tips.append("close_gap_quickly")
        elif our_role == "damage" and enemy_role == "assassin":
            tips.append("shoot_while_retreating")
        elif our_role == "tank" and enemy_role == "damage":
            tips.append("use_cover_to_approach")
        elif our_role == "support" and enemy_role == "assassin":
            tips.append("stay_behind_team")

        return tips
