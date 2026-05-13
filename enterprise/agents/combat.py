"""Combat Agent - Real-time combat decisions and threat assessment"""

import asyncio
import time
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType, ConfidenceScore
from ..orchestration.event_bus import EventBus, EventType


@dataclass
class Threat:
    source_id: str
    position: Tuple[float, float]
    threat_type: str
    severity: float
    distance: float
    attack_timing: float


@dataclass
class CombatAction:
    action_type: str
    target_id: Optional[str]
    position: Optional[Tuple[float, float]]
    timing: float
    parameters: Dict[str, Any]
    confidence: float


class CombatAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        self.active_threats: List[Threat] = []
        self.combat_history: List[Dict[str, Any]] = []
        self.target_priorities: Dict[str, float] = {}
        self.last_engagement_time: float = 0
        self.combat_cooldown: float = 2.0
        
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        action = message.content.get("action", "evaluate")
        
        try:
            if action == "evaluate":
                result = await self._evaluate_combat_situation(message.content)
            elif action == "engage":
                result = await self._engage_target(message.content)
            elif action == "retreat":
                result = await self._execute_retreat(message.content)
            elif action == "defend":
                result = await self._execute_defense(message.content)
            elif action == "predict":
                result = await self._predict_combat_outcome(message.content)
            else:
                result = {"error": f"Unknown action: {action}"}
            
            return AgentResponse(
                success=True,
                message=message,
                data=result,
                confidence=0.85,
                processing_time=time.time() - start_time,
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                message=message,
                error=str(e),
                processing_time=time.time() - start_time,
            )
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        threats = context.get("threats", [])
        opportunities = context.get("opportunities", [])
        my_state = context.get("my_state", {})
        
        immediate_threat = None
        highest_severity = 0.0
        
        for threat in threats:
            if threat.get("severity", 0) > highest_severity:
                highest_severity = threat.get("severity", 0)
                immediate_threat = threat
        
        recommended_tactics = []
        
        if immediate_threat and immediate_threat.get("severity", 0) > 0.7:
            recommended_tactics.append("immediate_retreat")
        elif opportunities and len(opportunities) > 2:
            recommended_tactics.append("aggressive_push")
        
        return {
            "immediate_threat": immediate_threat,
            "recommended_tactics": recommended_tactics,
            "threat_count": len(threats),
            "opportunity_count": len(opportunities),
            "confidence": 0.8,
        }
    
    async def _evaluate_combat_situation(self, content: Dict[str, Any]) -> Dict[str, Any]:
        game_state = content.get("game_state", {})
        enemies = content.get("enemies", [])
        allies = content.get("allies", [])
        my_position = content.get("my_position", (0, 0))
        my_health = game_state.get("health", 100)
        
        threats = self._identify_threats(enemies, my_position)
        self.active_threats = threats
        
        opportunities = self._identify_opportunities(enemies, allies, my_position)
        
        engagement_score = self._calculate_engagement_score(
            threats, opportunities, my_health, len(allies), len(enemies)
        )
        
        best_targets = self._prioritize_targets(enemies, my_position, my_health)
        
        recommended_action = self._determine_combat_action(
            engagement_score, threats, opportunities, my_health
        )
        
        return {
            "engagement_score": engagement_score,
            "threat_count": len(threats),
            "opportunity_count": len(opportunities),
            "best_targets": best_targets[:3],
            "recommended_action": recommended_action,
            "threats": [
                {
                    "source_id": t.source_id,
                    "severity": t.severity,
                    "distance": t.distance,
                    "threat_type": t.threat_type,
                }
                for t in threats
            ],
        }
    
    async def _engage_target(self, content: Dict[str, Any]) -> Dict[str, Any]:
        target_id = content.get("target_id")
        engagement_type = content.get("type", "normal")
        
        if time.time() - self.last_engagement_time < self.combat_cooldown:
            return {
                "status": "cooldown",
                "remaining_time": self.combat_cooldown - (time.time() - self.last_engagement_time),
            }
        
        combat_action = CombatAction(
            action_type="engage",
            target_id=target_id,
            position=None,
            timing=time.time(),
            parameters={"engagement_type": engagement_type},
            confidence=0.8,
        )
        
        self.last_engagement_time = time.time()
        
        self.combat_history.append({
            "action": "engage",
            "target_id": target_id,
            "type": engagement_type,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        await self.event_bus.publish(Event(
            source=self.id,
            type=EventType.DECISION_PROPOSED,
            data={
                "combat_action": combat_action.action_type,
                "target": target_id,
                "confidence": combat_action.confidence,
            },
        ))
        
        return {
            "status": "engaged",
            "target_id": target_id,
            "action": combat_action,
            "expected_damage": self._estimate_damage(target_id),
        }
    
    async def _execute_retreat(self, content: Dict[str, Any]) -> Dict[str, Any]:
        threat_direction = content.get("threat_direction", (0, 0))
        safe_positions = content.get("safe_positions", [])
        
        if not safe_positions:
            safe_positions = self._find_safe_positions(threat_direction)
        
        best_retreat = self._select_best_retreat_position(safe_positions, threat_direction)
        
        retreat_action = CombatAction(
            action_type="retreat",
            target_id=None,
            position=best_retreat,
            timing=time.time(),
            parameters={"reason": "threat_avoidance"},
            confidence=0.9,
        )
        
        self.combat_history.append({
            "action": "retreat",
            "destination": best_retreat,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return {
            "status": "retreating",
            "destination": best_retreat,
            "action": retreat_action,
            "estimated_safety": 0.85,
        }
    
    async def _execute_defense(self, content: Dict[str, Any]) -> Dict[str, Any]:
        defend_position = content.get("position", (0, 0))
        expected_threats = content.get("expected_threats", [])
        
        defense_action = CombatAction(
            action_type="defend",
            target_id=None,
            position=defend_position,
            timing=time.time(),
            parameters={
                "position_held": True,
                "expected_threats": len(expected_threats),
            },
            confidence=0.75,
        )
        
        return {
            "status": "defending",
            "position": defend_position,
            "action": defense_action,
            "block_strength": 0.8,
        }
    
    async def _predict_combat_outcome(self, content: Dict[str, Any]) -> Dict[str, Any]:
        my_state = content.get("my_state", {})
        enemy_states = content.get("enemy_states", [])
        
        my_health = my_state.get("health", 100)
        my_damage = my_state.get("damage_output", 10)
        
        total_enemy_health = sum(e.get("health", 50) for e in enemy_states)
        total_enemy_damage = sum(e.get("damage_output", 10) for e in enemy_states)
        
        my_time_to_kill = total_enemy_health / my_damage if my_damage > 0 else 999
        enemy_time_to_kill = my_health / total_enemy_damage if total_enemy_damage > 0 else 999
        
        win_probability = 1.0 / (1.0 + math.exp(my_time_to_kill - enemy_time_to_kill))
        
        predicted_outcome = "uncertain"
        if win_probability > 0.7:
            predicted_outcome = "favorable"
        elif win_probability < 0.3:
            predicted_outcome = "unfavorable"
        
        return {
            "win_probability": win_probability,
            "predicted_outcome": predicted_outcome,
            "time_to_kill_enemies": my_time_to_kill,
            "time_to_be_killed": enemy_time_to_kill,
            "recommendation": "engage" if win_probability > 0.6 else "retreat",
        }
    
    def _identify_threats(self, enemies: List[Dict[str, Any]], 
                         my_position: Tuple[float, float]) -> List[Threat]:
        threats = []
        
        for enemy in enemies:
            enemy_pos = tuple(enemy.get("position", (0, 0)))
            distance = self._calculate_distance(my_position, enemy_pos)
            
            severity = self._calculate_threat_severity(enemy, distance)
            
            threat = Threat(
                source_id=enemy.get("id", "unknown"),
                position=enemy_pos,
                threat_type=enemy.get("type", "normal"),
                severity=severity,
                distance=distance,
                attack_timing=enemy.get("attack_cooldown", 1.0),
            )
            threats.append(threat)
        
        threats.sort(key=lambda t: t.severity, reverse=True)
        return threats
    
    def _identify_opportunities(self, enemies: List[Dict[str, Any]],
                               allies: List[Dict[str, Any]],
                               my_position: Tuple[float, float]) -> List[Dict[str, Any]]:
        opportunities = []
        
        isolated_enemies = [e for e in enemies if self._is_isolated(e, enemies)]
        for enemy in isolated_enemies:
            opportunities.append({
                "type": "isolated_target",
                "target_id": enemy.get("id"),
                "advantage": 0.3,
            })
        
        low_health_enemies = [e for e in enemies if e.get("health", 100) < 30]
        for enemy in low_health_enemies:
            opportunities.append({
                "type": "low_health_target",
                "target_id": enemy.get("id"),
                "advantage": 0.4,
            })
        
        return opportunities
    
    def _calculate_engagement_score(self, threats: List[Threat],
                                   opportunities: List[Dict[str, Any]],
                                   my_health: float,
                                   ally_count: int,
                                   enemy_count: int) -> float:
        threat_factor = sum(t.severity for t in threats) / max(1, len(threats))
        opportunity_factor = sum(o.get("advantage", 0) for o in opportunities) / max(1, len(opportunities))
        
        health_factor = my_health / 100.0
        
        numbers_factor = (ally_count + 1) / (enemy_count + 1)
        
        engagement_score = (
            opportunity_factor * 0.4 +
            health_factor * 0.3 +
            numbers_factor * 0.2 -
            threat_factor * 0.3
        )
        
        return max(0.0, min(1.0, engagement_score))
    
    def _prioritize_targets(self, enemies: List[Dict[str, Any]],
                          my_position: Tuple[float, float],
                          my_health: float) -> List[Dict[str, Any]]:
        targets = []
        
        for enemy in enemies:
            enemy_pos = tuple(enemy.get("position", (0, 0)))
            distance = self._calculate_distance(my_position, enemy_pos)
            health = enemy.get("health", 100)
            
            priority = (
                (100 - health) * 0.3 +
                (1 / (distance + 1)) * 0.3 +
                (1 if enemy.get("isolated", False) else 0) * 0.2 +
                (1 if my_health > 50 else 0.2)
            )
            
            targets.append({
                "id": enemy.get("id"),
                "priority": priority,
                "distance": distance,
                "health": health,
            })
        
        targets.sort(key=lambda t: t["priority"], reverse=True)
        return targets
    
    def _determine_combat_action(self, engagement_score: float,
                                threats: List[Threat],
                                opportunities: List[Dict[str, Any]],
                                my_health: float) -> str:
        if engagement_score > 0.7 and my_health > 50:
            return "aggressive"
        elif engagement_score < 0.3 or my_health < 30:
            return "retreat"
        elif engagement_score > 0.4:
            return "opportunistic"
        else:
            return "defensive"
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                           pos2: Tuple[float, float]) -> float:
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _calculate_threat_severity(self, enemy: Dict[str, Any], distance: float) -> float:
        base_severity = enemy.get("damage_output", 10) / 100.0
        
        distance_factor = 1.0 / (1.0 + distance / 500.0)
        
        health_factor = enemy.get("health", 100) / 100.0
        
        return (base_severity * 0.5 + distance_factor * 0.3 + (1 - health_factor) * 0.2)
    
    def _is_isolated(self, enemy: Dict[str, Any], all_enemies: List[Dict[str, Any]]) -> bool:
        enemy_pos = tuple(enemy.get("position", (0, 0)))
        isolation_threshold = 300
        
        for other in all_enemies:
            if other.get("id") == enemy.get("id"):
                continue
            other_pos = tuple(other.get("position", (0, 0)))
            if self._calculate_distance(enemy_pos, other_pos) < isolation_threshold:
                return False
        
        return True
    
    def _estimate_damage(self, target_id: str) -> float:
        return 25.0
    
    def _find_safe_positions(self, threat_direction: Tuple[float, float]) -> List[Tuple[float, float]]:
        away_x = -threat_direction[0] * 200
        away_y = -threat_direction[1] * 200
        return [(away_x, away_y)]
    
    def _select_best_retreat_position(self, safe_positions: List[Tuple[float, float]],
                                    threat_direction: Tuple[float, float]) -> Tuple[float, float]:
        if not safe_positions:
            return (0, 0)
        return safe_positions[0]
