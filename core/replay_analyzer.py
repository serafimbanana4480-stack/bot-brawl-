"""
core/replay_analyzer.py

Replay Failure Analyzer for Brawl Stars bot.

Solves the "no replay failure analysis" problem. When the bot dies
or loses a match, this module automatically analyzes what went wrong:

- Death cause: rushed, flanked, overwhelmed, outplayed, positioning error
- Pattern detection: repeated mistakes across matches
- Suggestion generation: what to do differently
- Integration with WorldModel for contextual analysis

Usage:
    analyzer = ReplayFailureAnalyzer()
    
    # When bot dies:
    analyzer.record_death(context)
    
    # When match ends:
    analyzer.record_match_result(result, context)
    
    # Get analysis:
    report = analyzer.get_death_report()
    suggestions = analyzer.get_suggestions()
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter

logger = logging.getLogger(__name__)


class DeathCause(Enum):
    RUSHED = "rushed"               # Enemy rushed and killed us
    FLANKED = "flanked"             # Enemy attacked from side/behind
    OVERWHELMED = "overwhelmed"     # Too many enemies at once
    OUTPLAYED = "outplayed"         # Enemy dodged our shots and killed us
    POSITIONING = "positioning"     # Bad position (no cover, exposed)
    LOW_HP_FIGHT = "low_hp_fight"   # Took fight at low health
    SUPER_KILLED = "super_killed"   # Killed by enemy super
    POISON = "poison"               # Died to shrinking zone
    AFK = "afk"                     # Stood still too long
    UNKNOWN = "unknown"


class MatchResult(Enum):
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


@dataclass
class DeathRecord:
    """Record of a single death event."""
    cause: DeathCause = DeathCause.UNKNOWN
    timestamp: float = 0.0
    health_before: float = 1.0
    enemies_nearby: int = 0
    pressure: float = 0.0
    position: Tuple[float, float] = (0.0, 0.0)
    had_cover: bool = False
    intent: str = ""
    action: str = ""
    match_time: float = 0.0
    brawler: str = ""
    game_mode: str = ""


@dataclass
class MatchRecord:
    """Record of a complete match."""
    result: MatchResult = MatchResult.LOSS
    duration: float = 0.0
    brawler: str = ""
    game_mode: str = ""
    deaths: int = 0
    kills: int = 0
    death_causes: List[DeathCause] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class Suggestion:
    """A suggestion for improvement."""
    category: str           # "positioning", "combat", "awareness", "timing"
    description: str
    priority: float         # 0-1, how important
    evidence: str           # What data supports this
    action: str             # What to change


class ReplayFailureAnalyzer:
    """
    Analyzes deaths and match losses to identify patterns and suggest improvements.
    
    Features:
    - Automatic death cause classification
    - Pattern detection across matches
    - Suggestion generation
    - Statistics tracking
    - Integration with WorldModel, PressureMap, IntentSystem
    """

    MAX_DEATH_HISTORY = 100
    MAX_MATCH_HISTORY = 50

    def __init__(self):
        self._death_history: List[DeathRecord] = []
        self._match_history: List[MatchRecord] = []
        self._suggestions: List[Suggestion] = []
        self._lock = threading.RLock()
        
        # References
        self._world_model = None
        self._pressure_map = None
        self._intent_system = None
        
        logger.info("[REPLAY_ANALYZER] Initialized")

    def set_references(self, world_model=None, pressure_map=None, intent_system=None):
        """Set references to core systems for better analysis."""
        self._world_model = world_model
        self._pressure_map = pressure_map
        self._intent_system = intent_system

    def record_death(self, context: Dict) -> DeathCause:
        """
        Record and analyze a death event.
        
        Args:
            context: Dict with health, enemies_nearby, pressure, position,
                     had_cover, intent, action, match_time, brawler, game_mode
        
        Returns:
            Classified DeathCause
        """
        cause = self._classify_death(context)
        
        record = DeathRecord(
            cause=cause,
            timestamp=time.time(),
            health_before=context.get("health_before", 1.0),
            enemies_nearby=context.get("enemies_nearby", 0),
            pressure=context.get("pressure", 0.0),
            position=context.get("position", (0.0, 0.0)),
            had_cover=context.get("had_cover", False),
            intent=context.get("intent", ""),
            action=context.get("action", ""),
            match_time=context.get("match_time", 0.0),
            brawler=context.get("brawler", ""),
            game_mode=context.get("game_mode", ""),
        )
        
        with self._lock:
            self._death_history.append(record)
            if len(self._death_history) > self.MAX_DEATH_HISTORY:
                self._death_history = self._death_history[-self.MAX_DEATH_HISTORY:]
        
        # Update suggestions based on new death
        self._update_suggestions(record)
        
        logger.info("[REPLAY_ANALYZER] Death classified: %s (enemies=%d, pressure=%.1f, cover=%s)",
                     cause.value, record.enemies_nearby, record.pressure, record.had_cover)
        
        return cause

    def record_match_result(self, result: str, context: Dict):
        """Record a match result."""
        match_result = MatchResult(result) if result in [r.value for r in MatchResult] else MatchResult.LOSS
        
        record = MatchRecord(
            result=match_result,
            duration=context.get("duration", 0.0),
            brawler=context.get("brawler", ""),
            game_mode=context.get("game_mode", ""),
            deaths=context.get("deaths", 0),
            kills=context.get("kills", 0),
            death_causes=context.get("death_causes", []),
            timestamp=time.time(),
        )
        
        with self._lock:
            self._match_history.append(record)
            if len(self._match_history) > self.MAX_MATCH_HISTORY:
                self._match_history = self._match_history[-self.MAX_MATCH_HISTORY:]
        
        # Update suggestions based on match result
        self._update_match_suggestions(record)
        
        logger.info("[REPLAY_ANALYZER] Match result: %s (%s, %d deaths, %d kills)",
                     match_result.value, record.brawler, record.deaths, record.kills)

    def get_death_report(self, limit: int = 10) -> List[Dict]:
        """Get recent death reports."""
        with self._lock:
            return [
                {
                    "cause": d.cause.value,
                    "enemies": d.enemies_nearby,
                    "pressure": round(d.pressure, 1),
                    "cover": d.had_cover,
                    "intent": d.intent,
                    "action": d.action,
                    "match_time": round(d.match_time, 1),
                }
                for d in self._death_history[-limit:]
            ]

    def get_suggestions(self, min_priority: float = 0.3) -> List[Dict]:
        """Get current improvement suggestions."""
        with self._lock:
            return [
                {
                    "category": s.category,
                    "description": s.description,
                    "priority": round(s.priority, 2),
                    "evidence": s.evidence,
                    "action": s.action,
                }
                for s in sorted(self._suggestions, key=lambda x: x.priority, reverse=True)
                if s.priority >= min_priority
            ]

    def get_death_stats(self) -> Dict:
        """Get death statistics."""
        with self._lock:
            if not self._death_history:
                return {"total_deaths": 0}
            
            cause_counts = Counter(d.cause.value for d in self._death_history)
            avg_enemies = sum(d.enemies_nearby for d in self._death_history) / len(self._death_history)
            cover_pct = sum(1 for d in self._death_history if d.had_cover) / len(self._death_history) * 100
            
            return {
                "total_deaths": len(self._death_history),
                "cause_distribution": dict(cause_counts.most_common()),
                "avg_enemies_at_death": round(avg_enemies, 1),
                "had_cover_pct": round(cover_pct, 1),
                "most_common_cause": cause_counts.most_common(1)[0][0] if cause_counts else "none",
            }

    def get_match_stats(self) -> Dict:
        """Get match statistics."""
        with self._lock:
            if not self._match_history:
                return {"total_matches": 0}
            
            wins = sum(1 for m in self._match_history if m.result == MatchResult.WIN)
            total = len(self._match_history)
            
            return {
                "total_matches": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": round(wins / total * 100, 1),
                "avg_deaths": round(sum(m.deaths for m in self._match_history) / total, 1),
                "avg_kills": round(sum(m.kills for m in self._match_history) / total, 1),
            }

    # --- Internal ---

    def _classify_death(self, ctx: Dict) -> DeathCause:
        """Classify the cause of a death from context."""
        health = ctx.get("health_before", 1.0)
        enemies = ctx.get("enemies_nearby", 0)
        pressure = ctx.get("pressure", 0.0)
        had_cover = ctx.get("had_cover", False)
        intent = ctx.get("intent", "")
        action = ctx.get("action", "")
        match_time = ctx.get("match_time", 0.0)
        
        # Poison zone death (Showdown late game)
        if match_time > 120 and pressure > 5.0:
            return DeathCause.POISON
        
        # AFK death (stood still too long)
        if action in ("idle", "hold_position") and enemies > 0:
            return DeathCause.AFK
        
        # Low HP fight: took fight when already low
        if health < 0.3 and action in ("attack", "chase"):
            return DeathCause.LOW_HP_FIGHT
        
        # Rushed: enemy was rushing and killed us
        if intent == "aggressive" and enemies >= 1 and pressure > 3.0:
            return DeathCause.RUSHED
        
        # Overwhelmed: too many enemies
        if enemies >= 3:
            return DeathCause.OVERWHELMED
        
        # Flanked: enemies nearby but no cover
        if enemies >= 1 and not had_cover and pressure > 2.0:
            return DeathCause.FLANKED
        
        # Positioning: no cover, moderate pressure
        if not had_cover and pressure > 1.0:
            return DeathCause.POSITIONING
        
        # Super killed
        if pressure > 4.0 and enemies <= 1:
            return DeathCause.SUPER_KILLED
        
        # Outplayed: few enemies, had cover, still died
        if had_cover and enemies <= 2:
            return DeathCause.OUTPLAYED
        
        return DeathCause.UNKNOWN

    def _update_suggestions(self, death: DeathRecord):
        """Update suggestions based on a new death."""
        # Check for repeated patterns
        recent = self._death_history[-10:]
        
        # Repeated positioning deaths
        positioning_deaths = sum(1 for d in recent if d.cause in 
                                 (DeathCause.POSITIONING, DeathCause.FLANKED))
        if positioning_deaths >= 3:
            self._add_or_update_suggestion(
                "positioning", 
                "Frequent positioning deaths — seek cover more aggressively",
                0.8,
                f"{positioning_deaths}/10 recent deaths from positioning",
                "increase_retreat_threshold",
            )
        
        # Repeated low-HP fights
        low_hp_deaths = sum(1 for d in recent if d.cause == DeathCause.LOW_HP_FIGHT)
        if low_hp_deaths >= 2:
            self._add_or_update_suggestion(
                "combat",
                "Taking fights at low health — retreat earlier",
                0.7,
                f"{low_hp_deaths}/10 deaths from low-HP fights",
                "increase_retreat_threshold_by_0.15",
            )
        
        # Repeated rushed deaths
        rushed_deaths = sum(1 for d in recent if d.cause == DeathCause.RUSHED)
        if rushed_deaths >= 2:
            self._add_or_update_suggestion(
                "awareness",
                "Getting rushed frequently — watch for enemy approach",
                0.6,
                f"{rushed_deaths}/10 deaths from rushes",
                "increase_pressure_sensitivity",
            )

    def _update_match_suggestions(self, match: MatchRecord):
        """Update suggestions based on match result."""
        if match.result == MatchResult.LOSS and match.deaths >= 3:
            self._add_or_update_suggestion(
                "survival",
                f"High death count ({match.deaths}) in losses — prioritize survival",
                0.7,
                f"{match.deaths} deaths in {match.game_mode}",
                "prioritize_survival_in_{mode}".format(mode=match.game_mode),
            )

    def _add_or_update_suggestion(self, category: str, description: str,
                                    priority: float, evidence: str, action: str):
        """Add or update a suggestion."""
        with self._lock:
            # Check if similar suggestion exists
            for s in self._suggestions:
                if s.category == category and s.action == action:
                    s.priority = max(s.priority, priority)
                    s.evidence = evidence
                    return
            
            self._suggestions.append(Suggestion(
                category=category,
                description=description,
                priority=priority,
                evidence=evidence,
                action=action,
            ))
            
            # Keep top 20 suggestions
            self._suggestions.sort(key=lambda x: x.priority, reverse=True)
            if len(self._suggestions) > 20:
                self._suggestions = self._suggestions[:20]
