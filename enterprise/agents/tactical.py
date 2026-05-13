"""Tactical Planner Agent - Short-term tactical planning and decision making"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import random

from .base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType, ConfidenceScore
from ..orchestration.event_bus import EventBus, EventType


@dataclass
class Tactic:
    name: str
    description: str
    success_probability: float
    risk_level: float
    required_conditions: List[str]
    parameters: Dict[str, Any]


class TacticalPlannerAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        self.active_tactics: List[Tactic] = []
        self.tactical_history: List[Dict[str, Any]] = []
        self.current_tactical_situation: Dict[str, Any] = {}
        
        self._initialize_tactics()
        
    def _initialize_tactics(self):
        self.available_tactics = {
            "aggressive_push": Tactic(
                name="aggressive_push",
                description="Direct assault on enemy position",
                success_probability=0.6,
                risk_level=0.7,
                required_conditions=["super_available", "health_above_50"],
                parameters={"engage_distance": 200, "use_super": True},
            ),
            "hit_and_run": Tactic(
                name="hit_and_run",
                description="Quick attack then retreat",
                success_probability=0.75,
                risk_level=0.4,
                required_conditions=["has_movement_ability"],
                parameters={"attack_duration": 2.0, "retreat_distance": 150},
            ),
            "flanking": Tactic(
                name="flanking",
                description="Attack from侧面",
                success_probability=0.7,
                risk_level=0.5,
                required_conditions=["enemy_focused_elsewhere"],
                parameters={"approach_angle": 90},
            ),
            "zone_control": Tactic(
                name="zone_control",
                description="Control area and wait",
                success_probability=0.8,
                risk_level=0.2,
                required_conditions=["has_ranged_attack"],
                parameters={"control_radius": 150, "wait_time": 5.0},
            ),
            "burst_damage": Tactic(
                name="burst_damage",
                description="Maximum damage output",
                success_probability=0.55,
                risk_level=0.8,
                required_conditions=["super_available", "low_health_target"],
                parameters={"combo_sequence": ["attack", "super", "attack"]},
            ),
            "defensive_retreat": Tactic(
                name="defensive_retreat",
                description="Safe retreat to recover",
                success_probability=0.9,
                risk_level=0.1,
                required_conditions=["has_escape_ability"],
                parameters={"retreat_path": "shortest", "heal_at_destination": True},
            ),
        }
        
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        action = message.content.get("action", "plan")
        
        try:
            if action == "plan":
                result = await self._create_tactical_plan(message.content)
            elif action == "evaluate":
                result = await self._evaluate_tactical_situation(message.content)
            elif action == "adapt":
                result = await self._adapt_tactic(message.content)
            elif action == "select":
                result = await self._select_best_tactic(message.content)
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
        game_state = context.get("game_state", {})
        enemies = context.get("enemies", [])
        allies = context.get("allies", [])
        
        situation = self._analyze_tactical_situation(game_state, enemies, allies)
        self.current_tactical_situation = situation
        
        recommended_tactics = self._get_recommended_tactics(situation)
        
        return {
            "situation": situation,
            "recommended_tactics": recommended_tactics,
            "urgency": situation.get("urgency", "normal"),
            "confidence": 0.8,
        }
    
    async def _create_tactical_plan(self, content: Dict[str, Any]) -> Dict[str, Any]:
        objectives = content.get("objectives", [])
        constraints = content.get("constraints", {})
        game_state = content.get("game_state", {})
        enemies = content.get("enemies", [])
        
        situation = self._analyze_tactical_situation(game_state, enemies, [])
        
        selected_tactics = []
        for objective in objectives:
            best_tactic = self._select_tactic_for_objective(objective, situation)
            if best_tactic:
                selected_tactics.append(best_tactic)
        
        execution_order = self._order_tactics(selected_tactics)
        
        plan = {
            "situation_analysis": situation,
            "tactics": [t.name for t in execution_order],
            "execution_sequence": [
                {
                    "tactic": t.name,
                    "timing": self._estimate_timing(t),
                    "fallback": self._get_fallback_tactic(t),
                }
                for t in execution_order
            ],
            "risk_assessment": self._assess_plan_risk(execution_order),
        }
        
        await self.event_bus.publish(Event(
            source=self.id,
            type=EventType.DECISION_PROPOSED,
            data={"plan": plan, "tactics_count": len(execution_order)},
        ))
        
        return plan
    
    async def _evaluate_tactical_situation(self, content: Dict[str, Any]) -> Dict[str, Any]:
        game_state = content.get("game_state", {})
        enemies = content.get("enemies", [])
        allies = content.get("allies", [])
        
        situation = self._analyze_tactical_situation(game_state, enemies, allies)
        
        available_actions = self._get_available_actions(situation)
        
        return {
            "situation": situation,
            "available_actions": available_actions,
            "threat_level": situation.get("threat_level", 0.5),
            "opportunity_level": situation.get("opportunity_level", 0.5),
        }
    
    async def _adapt_tactic(self, content: Dict[str, Any]) -> Dict[str, Any]:
        current_tactic_name = content.get("current_tactic")
        new_context = content.get("context", {})
        
        if current_tactic_name not in self.available_tactics:
            return {"error": "Unknown tactic"}
        
        original_tactic = self.available_tactics[current_tactic_name]
        adapted_parameters = self._adapt_parameters(original_tactic, new_context)
        
        adapted_tactic = Tactic(
            name=f"{original_tactic.name}_adapted",
            description=original_tactic.description,
            success_probability=original_tactic.success_probability * 0.9,
            risk_level=original_tactic.risk_level,
            required_conditions=original_tactic.required_conditions,
            parameters=adapted_parameters,
        )
        
        return {
            "original_tactic": current_tactic_name,
            "adapted_tactic": adapted_tactic.name,
            "changes": adapted_parameters,
        }
    
    async def _select_best_tactic(self, content: Dict[str, Any]) -> Dict[str, Any]:
        game_state = content.get("game_state", {})
        enemies = content.get("enemies", [])
        
        situation = self._analyze_tactical_situation(game_state, enemies, [])
        
        situation_urgency = situation.get("urgency", "normal")
        
        if situation_urgency == "critical":
            best_tactic = self.available_tactics.get("defensive_retreat")
        elif situation_urgency == "high":
            best_tactic = self.available_tactics.get("hit_and_run")
        else:
            best_tactic = self._select_highest_value_tactic(situation)
        
        if best_tactic:
            return {
                "selected_tactic": best_tactic.name,
                "description": best_tactic.description,
                "success_probability": best_tactic.success_probability,
                "risk_level": best_tactic.risk_level,
                "parameters": best_tactic.parameters,
            }
        
        return {"error": "No suitable tactic found"}
    
    def _analyze_tactical_situation(self, game_state: Dict[str, Any],
                                    enemies: List[Dict[str, Any]],
                                    allies: List[Dict[str, Any]]) -> Dict[str, Any]:
        my_health = game_state.get("health", 100)
        my_super = game_state.get("super_charge", 0)
        enemy_count = len(enemies)
        ally_count = len(allies)
        
        threat_level = 0.0
        if enemy_count > 2:
            threat_level += 0.3
        if any(e.get("has_super", False) for e in enemies):
            threat_level += 0.2
        
        opportunity_level = 0.0
        low_health_enemies = [e for e in enemies if e.get("health", 100) < 30]
        if low_health_enemies:
            opportunity_level += 0.4
        if my_super >= 100:
            opportunity_level += 0.3
        
        urgency = "normal"
        if my_health < 20:
            urgency = "critical"
        elif my_health < 40 or threat_level > 0.6:
            urgency = "high"
        
        return {
            "threat_level": min(1.0, threat_level),
            "opportunity_level": min(1.0, opportunity_level),
            "urgency": urgency,
            "my_health": my_health,
            "my_super": my_super,
            "enemy_count": enemy_count,
            "ally_count": ally_count,
            "numerical_advantage": ally_count > enemy_count,
        }
    
    def _get_recommended_tactics(self, situation: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        if situation["urgency"] == "critical":
            recommendations.append("defensive_retreat")
        elif situation["numerical_advantage"]:
            recommendations.append("aggressive_push")
        elif situation["opportunity_level"] > 0.5:
            recommendations.append("burst_damage")
        else:
            recommendations.append("zone_control")
        
        return recommendations
    
    def _select_tactic_for_objective(self, objective: Dict[str, Any],
                                    situation: Dict[str, Any]) -> Optional[Tactic]:
        objective_type = objective.get("type", "generic")
        
        if objective_type == "eliminate":
            if situation["my_health"] > 50:
                return self.available_tactics.get("aggressive_push")
            return self.available_tactics.get("burst_damage")
        elif objective_type == "survive":
            return self.available_tactics.get("defensive_retreat")
        elif objective_type == "control":
            return self.available_tactics.get("zone_control")
        
        return self.available_tactics.get("hit_and_run")
    
    def _order_tactics(self, tactics: List[Tactic]) -> List[Tactic]:
        priority_order = {
            "defensive_retreat": 0,
            "hit_and_run": 1,
            "zone_control": 2,
            "flanking": 3,
            "aggressive_push": 4,
            "burst_damage": 5,
        }
        
        return sorted(tactics, key=lambda t: priority_order.get(t.name, 99))
    
    def _estimate_timing(self, tactic: Tactic) -> float:
        base_times = {
            "defensive_retreat": 2.0,
            "hit_and_run": 4.0,
            "zone_control": 5.0,
            "flanking": 3.0,
            "aggressive_push": 2.5,
            "burst_damage": 1.5,
        }
        return base_times.get(tactic.name, 3.0)
    
    def _get_fallback_tactic(self, tactic: Tactic) -> str:
        fallbacks = {
            "aggressive_push": "hit_and_run",
            "burst_damage": "hit_and_run",
            "flanking": "zone_control",
            "zone_control": "defensive_retreat",
            "hit_and_run": "defensive_retreat",
            "defensive_retreat": "zone_control",
        }
        return fallbacks.get(tactic.name, "zone_control")
    
    def _assess_plan_risk(self, tactics: List[Tactic]) -> Dict[str, Any]:
        if not tactics:
            return {"overall_risk": 0.0, "riskiest_tactic": None}
        
        max_risk = max(t.risk_level for t in tactics)
        avg_risk = sum(t.risk_level for t in tactics) / len(tactics)
        
        return {
            "overall_risk": avg_risk,
            "max_risk": max_risk,
            "riskiest_tactic": max(tactics, key=lambda t: t.risk_level).name,
        }
    
    def _get_available_actions(self, situation: Dict[str, Any]) -> List[str]:
        available = []
        
        if situation["my_super"] >= 100:
            available.append("use_super")
        if situation["my_health"] > 30:
            available.append("engage")
        available.append("retreat")
        available.append("reposition")
        
        return available
    
    def _adapt_parameters(self, tactic: Tactic, new_context: Dict[str, Any]) -> Dict[str, Any]:
        adapted = tactic.parameters.copy()
        
        if "distance_override" in new_context:
            adapted["engage_distance"] = new_context["distance_override"]
        
        if "use_super" in new_context:
            adapted["use_super"] = new_context["use_super"]
        
        return adapted
    
    def _select_highest_value_tactic(self, situation: Dict[str, Any]) -> Optional[Tactic]:
        best_tactic = None
        best_value = -1
        
        for tactic in self.available_tactics.values():
            conditions_met = all(
                self._check_condition(c, situation) 
                for c in tactic.required_conditions
            )
            
            if not conditions_met:
                continue
            
            value = tactic.success_probability - tactic.risk_level * 0.5
            
            if situation["numerical_advantage"]:
                if "aggressive" in tactic.name or "push" in tactic.name:
                    value += 0.2
            
            if value > best_value:
                best_value = value
                best_tactic = tactic
        
        return best_tactic
    
    def _check_condition(self, condition: str, situation: Dict[str, Any]) -> bool:
        condition_checks = {
            "super_available": situation.get("my_super", 0) >= 100,
            "health_above_50": situation.get("my_health", 0) > 50,
            "has_movement_ability": True,
            "has_ranged_attack": True,
            "low_health_target": True,
            "enemy_focused_elsewhere": situation.get("enemy_count", 0) > 1,
        }
        
        return condition_checks.get(condition, True)
