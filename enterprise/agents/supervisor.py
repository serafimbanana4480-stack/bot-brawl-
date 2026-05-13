"""Supervisor Agent - Orchestrates all other agents and manages workflows"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType, AgentStatus, ConfidenceScore
from ..orchestration.event_bus import EventBus, Event, EventType
from ..orchestration.engine import OrchestrationEngine, TaskPriority


class SupervisorAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus, engine: OrchestrationEngine):
        super().__init__(config)
        self.event_bus = event_bus
        self.engine = engine
        self.active_workflows: Dict[str, Any] = {}
        self.agent_health: Dict[str, float] = {}
        self.decision_history: List[Dict[str, Any]] = []
        
    async def initialize(self) -> bool:
        await super().initialize()
        await self.event_bus.subscribe(
            self.id,
            self._handle_event,
            [EventType.ERROR, EventType.TASK_FAILED, EventType.LEARNING_UPDATE]
        )
        self.status = AgentStatus.IDLE
        return True
    
    async def _handle_event(self, event: Event):
        if event.type == EventType.ERROR:
            await self._handle_error(event)
        elif event.type == EventType.TASK_FAILED:
            await self._handle_task_failure(event)
        elif event.type == EventType.LEARNING_UPDATE:
            await self._handle_learning_update(event)
    
    async def _handle_error(self, event: Event):
        self.logger.error(f"Supervisor received error event: {event.data}")
        await self._create_alert("ERROR", event.data)
    
    async def _handle_task_failure(self, event: Event):
        task_data = event.data
        self.logger.warning(f"Task failed: {task_data.get('name', 'unknown')}")
        
    async def _handle_learning_update(self, event: Event):
        self.logger.info(f"Learning update: {event.data}")
    
    async def _create_alert(self, alert_type: str, data: Dict[str, Any]):
        alert_event = Event(
            source=self.id,
            type=EventType.ERROR,
            priority=EventPriority.CRITICAL,
            data={"alert_type": alert_type, "details": data, "timestamp": datetime.utcnow().isoformat()},
        )
        await self.event_bus.publish(alert_event)
    
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        action = message.content.get("action", "unknown")
        
        try:
            if action == "supervise":
                result = await self._supervise(message.content)
            elif action == "create_workflow":
                result = await self._create_workflow(message.content)
            elif action == "check_health":
                result = await self._check_system_health()
            elif action == "coordinate":
                result = await self._coordinate_agents(message.content)
            elif action == "approve_decision":
                result = await self._approve_decision(message.content)
            else:
                result = {"error": f"Unknown action: {action}"}
            
            confidence = self._calculate_confidence(result)
            
            return AgentResponse(
                success=True,
                message=message,
                data=result,
                confidence=confidence,
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
        system_state = context.get("system_state", {})
        active_agents = system_state.get("active_agents", [])
        pending_tasks = system_state.get("pending_tasks", [])
        
        insights = []
        recommendations = []
        
        if len(pending_tasks) > 10:
            insights.append("High task backlog detected")
            recommendations.append("Consider scaling up agent resources")
        
        if any(self.agent_health.get(a, 1.0) < 0.5 for a in active_agents):
            insights.append("Some agents are unhealthy")
            recommendations.append("Restart or redistribute tasks from unhealthy agents")
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "confidence": 0.85,
        }
    
    async def _supervise(self, content: Dict[str, Any]) -> Dict[str, Any]:
        system_state = content.get("system_state", {})
        decisions = []
        
        metrics = self.engine.get_metrics()
        task_graph = self.engine.get_task_graph()
        
        return {
            "status": "supervised",
            "metrics": metrics,
            "task_graph": task_graph,
            "agent_health": self.agent_health,
            "decisions": decisions,
        }
    
    async def _create_workflow(self, content: Dict[str, Any]) -> Dict[str, Any]:
        workflow_name = content.get("name", "unnamed")
        steps = content.get("steps", [])
        
        workflow_id = f"workflow_{workflow_name}_{int(time.time())}"
        self.active_workflows[workflow_id] = {
            "name": workflow_name,
            "steps": steps,
            "status": "created",
            "created_at": datetime.utcnow().isoformat(),
        }
        
        return {
            "workflow_id": workflow_id,
            "status": "created",
        }
    
    async def _check_system_health(self) -> Dict[str, Any]:
        health_data = {
            "overall": "healthy",
            "agents": {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        for agent_id, agent in self.engine.agents.items():
            agent_metrics = agent.get_metrics()
            health_data["agents"][agent.name] = {
                "status": agent.status.value,
                "health_score": self.agent_health.get(agent_id, 1.0),
                "metrics": agent_metrics,
            }
            
            if agent.status == AgentStatus.ERROR:
                health_data["overall"] = "degraded"
        
        return health_data
    
    async def _coordinate_agents(self, content: Dict[str, Any]) -> Dict[str, Any]:
        target_agents = content.get("agents", [])
        action = content.get("action", "")
        data = content.get("data", {})
        
        results = {}
        for agent_id in target_agents:
            if agent_id in self.engine.agents:
                message = AgentMessage(
                    sender=self.id,
                    recipient=agent_id,
                    content={"action": action, **data},
                )
                response = await self.engine.agents[agent_id].process(message)
                results[agent_id] = response.to_dict()
        
        return {"coordination_results": results}
    
    async def _approve_decision(self, content: Dict[str, Any]) -> Dict[str, Any]:
        decision_id = content.get("decision_id")
        approved = content.get("approved", False)
        reasoning = content.get("reasoning", "")
        
        self.decision_history.append({
            "decision_id": decision_id,
            "approved": approved,
            "reasoning": reasoning,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return {
            "decision_id": decision_id,
            "approved": approved,
            "status": "processed",
        }
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        factors = {
            "data_completeness": len(result) / 10.0 if result else 0.0,
            "error_absence": 1.0 if "error" not in result else 0.0,
        }
        return sum(factors.values()) / len(factors)
    
    async def monitor_agent_health(self, agent_id: str, health_score: float):
        self.agent_health[agent_id] = health_score
        
        if health_score < 0.3:
            await self.event_bus.publish(Event(
                source=self.id,
                type=EventType.ERROR,
                priority=EventPriority.HIGH,
                data={"agent_id": agent_id, "health_score": health_score, "type": "agent_unhealthy"},
            ))
