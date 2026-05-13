"""
Scoring system for target prioritization and action evaluation.
Implements various scoring algorithms for optimal decision making.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import math
import random


@dataclass
class TargetScore:
    """Score breakdown for a target."""
    target_id: int
    total_score: float
    
    # Component scores
    health_score: float  # 0-1, lower health = higher score
    distance_score: float  # 0-1, optimal distance = higher score
    threat_score: float  # 0-1, lower threat = higher score
    vulnerability_score: float  # 0-1, exposed targets = higher score
    
    # Modifiers
    kill_pressure: float  # Bonus for enemies that can be killed quickly
    positioning_score: float  # Score for favorable positioning
    
    reasoning: str


class TargetScorer:
    """
    Scores potential targets based on multiple factors.
    Used to prioritize which enemy to attack first.
    """
    
    def __init__(
        self,
        optimal_range: float = 150.0,
        max_range: float = 500.0,
        low_health_threshold: float = 0.4,
        danger_distance: float = 300.0
    ):
        self.optimal_range = optimal_range
        self.max_range = max_range
        self.low_health_threshold = low_health_threshold
        self.danger_distance = danger_distance
        
    def score_target(
        self,
        target,
        player_position: Tuple[float, float],
        player_health: float,
        all_enemies: List,
        walls: List
    ) -> TargetScore:
        """
        Calculate comprehensive score for a target.
        
        Args:
            target: EnemyInfo object
            player_position: Current player position
            player_health: Current player health (0-1)
            all_enemies: List of all visible enemies
            walls: List of walls for cover calculation
            
        Returns:
            TargetScore with breakdown
        """
        distance = target.distance
        
        # Health score (inverse of health, with bonus for low health)
        if target.health_estimate <= self.low_health_threshold:
            health_score = 1.0 + (self.low_health_threshold - target.health_estimate) * 2
        else:
            health_score = 1.0 - target.health_estimate
        
        # Distance score (peaks at optimal range, drops off)
        if distance <= self.optimal_range:
            distance_score = 0.7 + 0.3 * (1 - distance / self.optimal_range)
        elif distance <= self.max_range:
            # Linear falloff from optimal to max
            t = (distance - self.optimal_range) / (self.max_range - self.optimal_range)
            distance_score = 0.7 * (1 - t)
        else:
            distance_score = 0.0
        
        # Threat score (inverse of threat level)
        threat_score = 1.0 - target.threat_level
        
        # Vulnerability (isolated enemies are more vulnerable)
        nearby_allies = sum(
            1 for e in all_enemies
            if e.track_id != target.track_id and
            self._distance(target.position, e.position) < 200
        )
        vulnerability_score = 1.0 / (1 + nearby_allies)
        
        # Kill pressure (can we kill them quickly?)
        time_to_kill = target.health_estimate / self._estimate_dps(distance)
        if time_to_kill < 2.0:  # Can kill in under 2 seconds
            kill_pressure = 1.5
        elif time_to_kill < 4.0:
            kill_pressure = 1.2
        else:
            kill_pressure = 0.8
        
        # Positioning score (do we have advantage?)
        positioning_score = self._calculate_positioning_score(
            player_position, target, walls
        )
        
        # Combined score with weights
        total_score = (
            health_score * 0.30 +
            distance_score * 0.25 +
            threat_score * 0.20 +
            vulnerability_score * 0.10 +
            positioning_score * 0.15
        ) * kill_pressure
        
        # Add randomization (non-deterministic behavior)
        total_score *= random.uniform(0.95, 1.05)
        
        reasoning = self._generate_reasoning(
            target, health_score, distance_score, threat_score, kill_pressure
        )
        
        return TargetScore(
            target_id=target.track_id,
            total_score=total_score,
            health_score=health_score,
            distance_score=distance_score,
            threat_score=threat_score,
            vulnerability_score=vulnerability_score,
            kill_pressure=kill_pressure,
            positioning_score=positioning_score,
            reasoning=reasoning
        )
    
    def rank_targets(
        self,
        enemies: List,
        player_position: Tuple[float, float],
        player_health: float,
        walls: List
    ) -> List[TargetScore]:
        """
        Rank all potential targets by score.
        
        Returns:
            List of TargetScore sorted by total_score (highest first)
        """
        scores = []
        for enemy in enemies:
            score = self.score_target(
                enemy, player_position, player_health, enemies, walls
            )
            scores.append(score)
        
        return sorted(scores, key=lambda s: -s.total_score)
    
    def _distance(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _estimate_dps(self, distance: float) -> float:
        """Estimate damage per second at given distance."""
        # Simplified: closer = more DPS
        if distance < 100:
            return 500  # Close range burst
        elif distance < 300:
            return 300  # Mid range
        else:
            return 100  # Long range
    
    def _calculate_positioning_score(
        self,
        player_pos: Tuple[float, float],
        target,
        walls: List
    ) -> float:
        """Calculate positioning advantage score."""
        score = 0.5  # Base score
        
        # Do we have cover?
        for wall in walls:
            if self._is_between(player_pos, target.position, wall.center, 50):
                score += 0.3  # We have cover
                break
        
        # Are we flanking?
        # Simplified: check if we're at an angle to enemy
        # (In full implementation, would check enemy facing direction)
        score += 0.2  # Assume good positioning
        
        return min(1.0, score)
    
    def _is_between(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        check: Tuple[float, float],
        tolerance: float
    ) -> bool:
        """Check if check point is approximately between p1 and p2."""
        # Check if check is colinear with p1-p2
        d1 = self._distance(p1, check)
        d2 = self._distance(check, p2)
        d_total = self._distance(p1, p2)
        
        return abs(d1 + d2 - d_total) < tolerance
    
    def _generate_reasoning(
        self,
        target,
        health_score: float,
        distance_score: float,
        threat_score: float,
        kill_pressure: float
    ) -> str:
        """Generate human-readable reasoning for the score."""
        reasons = []
        
        if health_score > 0.7:
            reasons.append("low health")
        if distance_score > 0.7:
            reasons.append("optimal range")
        if threat_score > 0.7:
            reasons.append("low threat")
        if kill_pressure > 1.2:
            reasons.append("kill pressure")
        
        if not reasons:
            return "average target"
        
        return ", ".join(reasons)


class ActionScorer:
    """
    Scores potential actions based on predicted outcomes.
    """
    
    def __init__(self):
        self.weights = {
            "damage_dealt": 1.0,
            "damage_taken": -1.5,
            "kill_potential": 2.0,
            "death_risk": -3.0,
            "position_improvement": 0.5,
            "resource_gain": 0.3,
        }
    
    def score_action(
        self,
        action_name: str,
        predicted_outcome: dict
    ) -> float:
        """
        Score an action based on predicted outcome.
        
        Args:
            action_name: Name of the action
            predicted_outcome: Dict with predicted metrics
            
        Returns:
            Score value (higher = better)
        """
        score = 0.0
        
        for metric, value in predicted_outcome.items():
            weight = self.weights.get(metric, 0.0)
            score += value * weight
        
        # Add small randomization
        score *= random.uniform(0.98, 1.02)
        
        return score
    
    def compare_actions(
        self,
        actions: List[Tuple[str, dict]]
    ) -> List[Tuple[str, float, str]]:
        """
        Compare multiple actions and return ranked list.
        
        Args:
            actions: List of (action_name, predicted_outcome) tuples
            
        Returns:
            List of (action_name, score, reasoning) sorted by score
        """
        results = []
        
        for action_name, outcome in actions:
            score = self.score_action(action_name, outcome)
            reasoning = self._generate_action_reasoning(action_name, outcome)
            results.append((action_name, score, reasoning))
        
        return sorted(results, key=lambda x: -x[1])
    
    def _generate_action_reasoning(
        self,
        action_name: str,
        outcome: dict
    ) -> str:
        """Generate reasoning for action score."""
        reasons = []
        
        if outcome.get("damage_dealt", 0) > 0:
            reasons.append(f"deal {outcome['damage_dealt']:.0f} dmg")
        if outcome.get("kill_potential", 0) > 0.5:
            reasons.append("kill potential")
        if outcome.get("death_risk", 0) > 0.3:
            reasons.append("HIGH RISK")
        elif outcome.get("damage_taken", 0) > 0:
            reasons.append(f"take {outcome['damage_taken']:.0f} dmg")
        
        return ", ".join(reasons) if reasons else "neutral action"


class SituationScorer:
    """
    Scores overall game situation.
    Used to determine if we should play aggressive or defensive.
    """
    
    def __init__(self):
        self.aggression_threshold = 0.6
        self.defense_threshold = 0.4
    
    def score_situation(self, game_state) -> dict:
        """
        Score the current game situation.
        
        Returns:
            Dict with situation analysis
        """
        scores = {
            "health_advantage": self._health_advantage(game_state),
            "number_advantage": self._number_advantage(game_state),
            "positioning": self._positioning_score(game_state),
            "objective_control": self._objective_score(game_state),
        }
        
        # Overall situation score
        overall = sum(scores.values()) / len(scores)
        
        recommendation = "neutral"
        if overall > self.aggression_threshold:
            recommendation = "aggressive"
        elif overall < self.defense_threshold:
            recommendation = "defensive"
        
        return {
            "scores": scores,
            "overall": overall,
            "recommendation": recommendation,
            "confidence": min(1.0, abs(overall - 0.5) * 2)
        }
    
    def _health_advantage(self, game_state) -> float:
        """Calculate health advantage score."""
        if not game_state.enemies:
            return 1.0
        
        player_health = game_state.player_health
        avg_enemy_health = sum(e.health_estimate for e in game_state.enemies) / len(game_state.enemies)
        
        # Score based on health differential
        diff = player_health - avg_enemy_health
        return 0.5 + diff * 0.5  # Center at 0.5
    
    def _number_advantage(self, game_state) -> float:
        """Calculate number advantage score."""
        num_enemies = len(game_state.enemies)
        
        if num_enemies == 0:
            return 1.0
        elif num_enemies == 1:
            return 0.8
        elif num_enemies == 2:
            return 0.5
        else:
            return 0.2
    
    def _positioning_score(self, game_state) -> float:
        """Calculate positioning score."""
        # Simplified: based on danger score
        return 1.0 - game_state.danger_score
    
    def _objective_score(self, game_state) -> float:
        """Calculate objective control score."""
        # Placeholder for objective-based modes
        # In Showdown: based on safe position
        # In Gem Grab: based on gem count
        # etc.
        return 0.5  # Neutral default


def create_default_scorers() -> Tuple[TargetScorer, ActionScorer, SituationScorer]:
    """Factory to create all default scorers."""
    return (
        TargetScorer(),
        ActionScorer(),
        SituationScorer()
    )
