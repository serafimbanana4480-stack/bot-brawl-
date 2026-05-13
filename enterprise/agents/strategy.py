"""Strategy Agent - Long-term strategic planning and goal setting"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from .base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType, ConfidenceScore
from ..orchestration.event_bus import EventBus, EventType


@dataclass
class StrategicGoal:
    id: str
    name: str
    description: str
    priority: int
    target_metrics: Dict[str, float]
    current_progress: float = 0.0
    status: str = "active"
    created_at: datetime = None
    completed_at: Optional[datetime] = None


@dataclass  
class Strategy:
    id: str
    name: str
    goals: List[StrategicGoal]
    actions: List[Dict[str, Any]]
    confidence: float
    risk_assessment: Dict[str, float]
    created_at: datetime = None


class StrategyAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        self.current_strategy: Optional[Strategy] = None
        self.goals: List[StrategicGoal] = []
        self.strategy_history: List[Strategy] = []
        self.strategic_context: Dict[str, Any] = {}
        
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        action = message.content.get("action", "plan")
        
        try:
            if action == "plan":
                result = await self._create_strategy(message.content)
            elif action == "evaluate":
                result = await self._evaluate_strategy(message.content)
            elif action == "adapt":
                result = await self._adapt_strategy(message.content)
            elif action == "set_goal":
                result = await self._set_goal(message.content)
            elif action == "get_status":
                result = await self._get_strategy_status()
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
        opponent_info = context.get("opponent_info", {})
        resources = context.get("resources", {})
        
        strategic_insights = []
        recommended_actions = []
        
        if game_state.get("health", 100) < 30:
            strategic_insights.append("Critical health - survival is priority")
            recommended_actions.append("retreat_to_safety")
        
        if opponent_info.get("count", 0) > 3:
            strategic_insights.append("Outnumbered - need tactical advantage")
            recommended_actions.append("seek_enclosure")
        
        win_probability = self._calculate_win_probability(context)
        
        return {
            "insights": strategic_insights,
            "recommended_actions": recommended_actions,
            "win_probability": win_probability,
            "confidence": 0.8,
        }
    
    async def _create_strategy(self, content: Dict[str, Any]) -> Dict[str, Any]:
        objectives = content.get("objectives", [])
        constraints = content.get("constraints", {})
        game_state = content.get("game_state", {})
        
        strategy_goals = []
        for obj in objectives:
            goal = StrategicGoal(
                id=f"goal_{len(self.goals)}",
                name=obj.get("name", "unnamed"),
                description=obj.get("description", ""),
                priority=obj.get("priority", 1),
                target_metrics=obj.get("metrics", {}),
                created_at=datetime.utcnow(),
            )
            strategy_goals.append(goal)
            self.goals.append(goal)
        
        actions = self._generate_strategic_actions(strategy_goals, game_state)
        
        risk_assessment = self._assess_risks(actions, constraints)
        
        self.current_strategy = Strategy(
            id=f"strategy_{int(time.time())}",
            name=content.get("name", "Strategic Plan"),
            goals=strategy_goals,
            actions=actions,
            confidence=0.75,
            risk_assessment=risk_assessment,
            created_at=datetime.utcnow(),
        )
        
        self.strategy_history.append(self.current_strategy)
        
        await self.event_bus.publish(Event(
            source=self.id,
            type=EventType.DECISION_PROPOSED,
            data={
                "strategy": {
                    "id": self.current_strategy.id,
                    "name": self.current_strategy.name,
                    "actions_count": len(actions),
                }
            },
        ))
        
        return {
            "strategy_id": self.current_strategy.id,
            "goals": [g.id for g in strategy_goals],
            "actions": actions,
            "risk_assessment": risk_assessment,
        }
    
    async def _evaluate_strategy(self, content: Dict[str, Any]) -> Dict[str, Any]:
        strategy_id = content.get("strategy_id")
        current_state = content.get("current_state", {})
        
        if not self.current_strategy or self.current_strategy.id != strategy_id:
            return {"error": "Strategy not found", "valid": False}
        
        goal_progress = []
        for goal in self.current_strategy.goals:
            progress = self._calculate_goal_progress(goal, current_state)
            goal_progress.append({
                "goal_id": goal.id,
                "progress": progress,
                "status": "completed" if progress >= 1.0 else "in_progress",
            })
        
        overall_score = sum(p["progress"] for p in goal_progress) / len(goal_progress) if goal_progress else 0
        
        return {
            "strategy_id": strategy_id,
            "overall_score": overall_score,
            "goal_progress": goal_progress,
            "recommendations": self._get_strategy_recommendations(goal_progress),
        }
    
    async def _adapt_strategy(self, content: Dict[str, Any]) -> Dict[str, Any]:
        trigger = content.get("trigger", "unknown")
        new_context = content.get("context", {})
        
        self.strategic_context.update(new_context)
        
        if self.current_strategy:
            old_strategy_id = self.current_strategy.id
            self.strategy_history.append(self.current_strategy)
        else:
            old_strategy_id = None
        
        adaptation_type = self._determine_adaptation_type(trigger)
        
        if adaptation_type == "minor":
            adapted_actions = self._minor_adjustment()
            confidence = 0.8
        elif adaptation_type == "major":
            adapted_actions = await self._major_replanning(new_context)
            confidence = 0.65
        else:
            adapted_actions = []
            confidence = 0.5
        
        new_strategy = Strategy(
            id=f"strategy_{int(time.time())}",
            name=f"Adapted from {old_strategy_id}",
            goals=self.goals[-5:] if self.goals else [],
            actions=adapted_actions,
            confidence=confidence,
            risk_assessment={},
            created_at=datetime.utcnow(),
        )
        
        self.current_strategy = new_strategy
        
        return {
            "adaptation_type": adaptation_type,
            "new_strategy_id": new_strategy.id,
            "changes": adapted_actions,
            "confidence": confidence,
        }
    
    async def _set_goal(self, content: Dict[str, Any]) -> Dict[str, Any]:
        goal = StrategicGoal(
            id=f"goal_{len(self.goals)}",
            name=content.get("name", "New Goal"),
            description=content.get("description", ""),
            priority=content.get("priority", 1),
            target_metrics=content.get("metrics", {}),
            created_at=datetime.utcnow(),
        )
        
        self.goals.append(goal)
        
        return {
            "goal_id": goal.id,
            "status": "created",
            "total_goals": len(self.goals),
        }
    
    async def _get_strategy_status(self) -> Dict[str, Any]:
        return {
            "current_strategy": {
                "id": self.current_strategy.id if self.current_strategy else None,
                "name": self.current_strategy.name if self.current_strategy else None,
                "confidence": self.current_strategy.confidence if self.current_strategy else 0.0,
            },
            "active_goals": len([g for g in self.goals if g.status == "active"]),
            "completed_goals": len([g for g in self.goals if g.status == "completed"]),
            "strategy_history_count": len(self.strategy_history),
        }
    
    def _generate_strategic_actions(self, goals: List[StrategicGoal], 
                                   game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        actions = []
        
        for goal in goals:
            if "defeat" in goal.name.lower():
                actions.append({
                    "type": "aggressive",
                    "tactic": "engage_nearest",
                    "priority": goal.priority,
                })
            elif "survive" in goal.name.lower() or "health" in goal.name.lower():
                actions.append({
                    "type": "defensive",
                    "tactic": "conserve_resources",
                    "priority": goal.priority,
                })
            elif "capture" in goal.name.lower():
                actions.append({
                    "type": "objective",
                    "tactic": "move_to_objective",
                    "priority": goal.priority,
                })
        
        return actions
    
    def _assess_risks(self, actions: List[Dict[str, Any]], 
                    constraints: Dict[str, Any]) -> Dict[str, float]:
        risks = {
            "overall": 0.5,
            "resource_depletion": 0.3,
            "exposure": 0.4,
            "coordination_failure": 0.2,
        }
        
        aggressive_actions = sum(1 for a in actions if a.get("type") == "aggressive")
        if aggressive_actions > 3:
            risks["overall"] += 0.2
            risks["exposure"] += 0.3
        
        return risks
    
    def _calculate_win_probability(self, context: Dict[str, Any]) -> float:
        player_health = context.get("game_state", {}).get("health", 50)
        opponent_health = context.get("opponent_info", {}).get("average_health", 50)
        player_advantage = (player_health - opponent_health) / 100.0
        
        return max(0.0, min(1.0, 0.5 + player_advantage))
    
    def _calculate_goal_progress(self, goal: StrategicGoal, 
                                current_state: Dict[str, Any]) -> float:
        progress = 0.0
        
        for metric, target in goal.target_metrics.items():
            current = current_state.get(metric, 0)
            if target > 0:
                progress += min(1.0, current / target)
        
        return progress / len(goal.target_metrics) if goal.target_metrics else 0.0
    
    def _determine_adaptation_type(self, trigger: str) -> str:
        critical_triggers = ["critical_health", "team_wipe", "objective_lost"]
        major_triggers = ["significant_health_loss", "opponent_advantage", "resource_shortage"]
        
        if trigger in critical_triggers:
            return "major"
        elif trigger in major_triggers:
            return "minor"
        return "none"
    
    def _minor_adjustment(self) -> List[Dict[str, Any]]:
        if not self.current_strategy:
            return []
        
        adjusted = []
        for action in self.current_strategy.actions:
            adjusted_action = action.copy()
            if action.get("type") == "aggressive":
                adjusted_action["tactic"] = "cautious_engage"
            adjusted.append(adjusted_action)
        
        return adjusted
    
    async def _major_replanning(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"type": "reassess", "priority": 1, "reason": "major_context_change"},
            {"type": "conservative", "priority": 2, "tactic": "play_safe"},
        ]
    
    def _get_strategy_recommendations(self, goal_progress: List[Dict[str, Any]]) -> List[str]:
        recommendations = []
        
        low_progress = [g for g in goal_progress if g["progress"] < 0.3]
        if low_progress:
            recommendations.append("Focus resources on lagging goals")
        
        return recommendations
