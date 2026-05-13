"""
performance_analyzer.py

Performance analyzer for gameplay replays.

Analyzes performance metrics from parsed replays including
K/D ratio, damage efficiency, decision quality, and improvement trends.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from .replay_parser import ParsedReplay, GameEvent

logger = logging.getLogger(__name__)


@dataclass
class CombatMetrics:
    """Combat performance metrics."""
    total_kills: int = 0
    total_deaths: int = 0
    total_damage_dealt: float = 0.0
    total_damage_taken: float = 0.0
    total_attacks: int = 0
    total_hits: int = 0  # Successful attacks
    total_supers: int = 0
    total_super_hits: int = 0
    
    @property
    def kda_ratio(self) -> float:
        """Kill/Death/Assist ratio."""
        if self.total_deaths == 0:
            return float(self.total_kills)
        return self.total_kills / max(1, self.total_deaths)
    
    @property
    def accuracy(self) -> float:
        """Attack accuracy (hits / total attacks)."""
        if self.total_attacks == 0:
            return 0.0
        return self.total_hits / self.total_attacks
    
    @property
    def damage_efficiency(self) -> float:
        """Damage dealt vs damage taken ratio."""
        if self.total_damage_taken == 0:
            return float(self.total_damage_dealt)
        return self.total_damage_dealt / max(1, self.total_damage_taken)


@dataclass
class MovementMetrics:
    """Movement performance metrics."""
    total_distance: float = 0.0  # Total pixels moved
    average_speed: float = 0.0  # Average pixels per second
    max_speed: float = 0.0
    idle_time: float = 0.0  # Time spent not moving
    bush_time: float = 0.0  # Time spent in bushes


@dataclass
class DecisionMetrics:
    """Decision quality metrics."""
    total_decisions: int = 0
    good_decisions: int = 0
    bad_decisions: int = 0
    reaction_time_avg: float = 0.0  # Average reaction time in seconds
    positioning_score: float = 0.0  # 0.0 to 1.0
    
    @property
    def decision_quality(self) -> float:
        """Ratio of good decisions."""
        if self.total_decisions == 0:
            return 0.0
        return self.good_decisions / self.total_decisions


@dataclass
class PerformanceReport:
    """Complete performance report."""
    replay_id: str
    combat: CombatMetrics
    movement: MovementMetrics
    decision: DecisionMetrics
    overall_score: float  # 0.0 to 1.0
    improvement_areas: List[str]
    strengths: List[str]


class PerformanceAnalyzer:
    """
    Analyzes performance from parsed replays.
    
    Computes various metrics to evaluate bot performance
    and identify areas for improvement.
    """
    
    def __init__(self):
        self.reports: List[PerformanceReport] = []
    
    def analyze(self, replay: ParsedReplay, replay_id: str = "unknown") -> PerformanceReport:
        """
        Analyze a parsed replay.
        
        Args:
            replay: ParsedReplay object
            replay_id: Identifier for the replay
            
        Returns:
            PerformanceReport object
        """
        # Analyze combat
        combat = self._analyze_combat(replay)
        
        # Analyze movement
        movement = self._analyze_movement(replay)
        
        # Analyze decisions
        decision = self._analyze_decisions(replay)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(combat, movement, decision)
        
        # Identify improvement areas
        improvement_areas = self._identify_improvement_areas(combat, movement, decision)
        
        # Identify strengths
        strengths = self._identify_strengths(combat, movement, decision)
        
        report = PerformanceReport(
            replay_id=replay_id,
            combat=combat,
            movement=movement,
            decision=decision,
            overall_score=overall_score,
            improvement_areas=improvement_areas,
            strengths=strengths
        )
        
        self.reports.append(report)
        logger.info(f"Analyzed replay {replay_id}: overall_score={overall_score:.2f}")
        
        return report
    
    def _analyze_combat(self, replay: ParsedReplay) -> CombatMetrics:
        """Analyze combat metrics."""
        metrics = CombatMetrics()
        
        for event in replay.events:
            if event.event_type == 'attack':
                metrics.total_attacks += 1
                if event.data.get('hit', False):
                    metrics.total_hits += 1
                if 'damage' in event.data:
                    metrics.total_damage_dealt += event.data['damage']
            
            elif event.event_type == 'kill':
                metrics.total_kills += 1
            
            elif event.event_type == 'death':
                metrics.total_deaths += 1
                if 'damage_taken' in event.data:
                    metrics.total_damage_taken += event.data['damage_taken']
            
            elif event.event_type == 'super':
                metrics.total_supers += 1
                if event.data.get('hit', False):
                    metrics.total_super_hits += 1
            
            elif event.event_type == 'damage_taken':
                metrics.total_damage_taken += event.data.get('damage', 0)
        
        return metrics
    
    def _analyze_movement(self, replay: ParsedReplay) -> MovementMetrics:
        """Analyze movement metrics."""
        metrics = MovementMetrics()
        
        # Extract movement events
        move_events = [e for e in replay.events if e.event_type == 'move']
        
        if not move_events:
            return metrics
        
        # Calculate total distance
        total_distance = 0.0
        prev_pos = move_events[0].position
        prev_time = move_events[0].timestamp
        
        speeds = []
        
        for event in move_events[1:]:
            curr_pos = event.position
            curr_time = event.timestamp
            
            # Calculate distance
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            distance = (dx**2 + dy**2)**0.5
            total_distance += distance
            
            # Calculate speed
            dt = curr_time - prev_time
            if dt > 0:
                speed = distance / dt
                speeds.append(speed)
            
            prev_pos = curr_pos
            prev_time = curr_time
        
        metrics.total_distance = total_distance
        
        if speeds:
            metrics.average_speed = np.mean(speeds)
            metrics.max_speed = np.max(speeds)
        
        # Calculate idle time (time between move events)
        if len(move_events) > 1:
            idle_times = []
            for i in range(1, len(move_events)):
                dt = move_events[i].timestamp - move_events[i-1].timestamp
                if dt > 1.0:  # Consider > 1s as idle
                    idle_times.append(dt)
            
            metrics.idle_time = sum(idle_times)
        
        return metrics
    
    def _analyze_decisions(self, replay: ParsedReplay) -> DecisionMetrics:
        """Analyze decision metrics."""
        metrics = DecisionMetrics()
        
        # Count decision events
        decision_events = [e for e in replay.events 
                          if e.event_type in ['attack', 'retreat', 'advance', 'collect']]
        
        metrics.total_decisions = len(decision_events)
        
        # Estimate decision quality based on outcomes
        # This is a simplified heuristic - in production, use actual outcome data
        for event in decision_events:
            # Simple heuristic: good if followed by positive outcome
            # In production, this should use actual game outcome data
            metrics.good_decisions += 1  # Placeholder
        
        # Calculate reaction time (time between enemy appearance and response)
        enemy_events = [e for e in replay.events if e.event_type == 'enemy_detected']
        response_events = [e for e in replay.events if e.event_type == 'attack']
        
        reaction_times = []
        for enemy_event in enemy_events:
            # Find next attack after enemy detection
            for response_event in response_events:
                if response_event.timestamp > enemy_event.timestamp:
                    reaction_time = response_event.timestamp - enemy_event.timestamp
                    if 0 < reaction_time < 2.0:  # Reasonable reaction time
                        reaction_times.append(reaction_time)
                    break
        
        if reaction_times:
            metrics.reaction_time_avg = np.mean(reaction_times)
        
        # Positioning score (placeholder - would need map analysis)
        metrics.positioning_score = 0.7  # Placeholder
        
        return metrics
    
    def _calculate_overall_score(self, combat: CombatMetrics, 
                                  movement: MovementMetrics,
                                  decision: DecisionMetrics) -> float:
        """Calculate overall performance score (0.0 to 1.0)."""
        # Weighted average of different metrics
        combat_score = min(1.0, combat.kda_ratio / 3.0)  # Normalize KDA
        accuracy_score = combat.accuracy
        decision_score = decision.decision_quality
        
        # Weights
        weights = {
            'combat': 0.4,
            'accuracy': 0.3,
            'decision': 0.3
        }
        
        overall = (
            weights['combat'] * combat_score +
            weights['accuracy'] * accuracy_score +
            weights['decision'] * decision_score
        )
        
        return overall
    
    def _identify_improvement_areas(self, combat: CombatMetrics,
                                     movement: MovementMetrics,
                                     decision: DecisionMetrics) -> List[str]:
        """Identify areas needing improvement."""
        areas = []
        
        # Combat issues
        if combat.accuracy < 0.5:
            areas.append("Low attack accuracy - consider aim assist tuning")
        if combat.kda_ratio < 1.0:
            areas.append("Low K/D ratio - improve positioning and target selection")
        if combat.damage_efficiency < 1.0:
            areas.append("Taking too much damage - improve dodging and cover usage")
        
        # Movement issues
        if movement.idle_time > 10.0:
            areas.append("Excessive idle time - improve movement efficiency")
        if movement.average_speed < 100:
            areas.append("Slow movement - consider pathfinding optimization")
        
        # Decision issues
        if decision.decision_quality < 0.6:
            areas.append("Poor decision quality - review neural policy training")
        if decision.reaction_time_avg > 0.5:
            areas.append("Slow reaction time - optimize detection pipeline")
        
        if not areas:
            areas.append("Performance is good - continue current strategy")
        
        return areas
    
    def _identify_strengths(self, combat: CombatMetrics,
                           movement: MovementMetrics,
                           decision: DecisionMetrics) -> List[str]:
        """Identify strengths."""
        strengths = []
        
        # Combat strengths
        if combat.accuracy > 0.7:
            strengths.append("High attack accuracy")
        if combat.kda_ratio > 2.0:
            strengths.append("Excellent K/D ratio")
        if combat.damage_efficiency > 2.0:
            strengths.append("Great damage efficiency")
        
        # Movement strengths
        if movement.idle_time < 5.0:
            strengths.append("Efficient movement")
        if movement.average_speed > 200:
            strengths.append("Fast movement")
        
        # Decision strengths
        if decision.decision_quality > 0.8:
            strengths.append("Excellent decision quality")
        if decision.reaction_time_avg < 0.2:
            strengths.append("Fast reaction time")
        
        if not strengths:
            strengths.append("No significant strengths identified yet")
        
        return strengths
    
    def get_trends(self, n_recent: int = 10) -> Dict[str, List[float]]:
        """
        Get performance trends over recent replays.
        
        Args:
            n_recent: Number of recent replays to analyze
            
        Returns:
            Dictionary of metric trends
        """
        if not self.reports:
            return {}
        
        recent = self.reports[-n_recent:]
        
        trends = {
            'overall_scores': [r.overall_score for r in recent],
            'kda_ratios': [r.combat.kda_ratio for r in recent],
            'accuracies': [r.combat.accuracy for r in recent],
            'decision_qualities': [r.decision.decision_quality for r in recent],
        }
        
        return trends
    
    def get_average_performance(self, n_recent: int = 10) -> Dict[str, float]:
        """
        Get average performance metrics.
        
        Args:
            n_recent: Number of recent replays to average
            
        Returns:
            Dictionary of average metrics
        """
        if not self.reports:
            return {}
        
        recent = self.reports[-n_recent:]
        
        avg = {
            'overall_score': np.mean([r.overall_score for r in recent]),
            'kda_ratio': np.mean([r.combat.kda_ratio for r in recent]),
            'accuracy': np.mean([r.combat.accuracy for r in recent]),
            'damage_efficiency': np.mean([r.combat.damage_efficiency for r in recent]),
            'decision_quality': np.mean([r.decision.decision_quality for r in recent]),
        }
        
        return avg
