"""Coordination Agent - Multi-agent coordination and consensus building"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import heapq

from .base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType, ConfidenceScore
from ..orchestration.event_bus import EventBus, EventType, EventPriority, Event


@dataclass
class CoordinationTask:
    id: str
    description: str
    assigned_agents: List[str]
    status: str
    deadline: Optional[float]
    priority: int
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None


@dataclass
class ConsensusVote:
    agent_id: str
    decision: str
    confidence: float
    reasoning: str
    timestamp: float


class CoordinationAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        
        self.active_tasks: Dict[str, CoordinationTask] = {}
        self.completed_tasks: Dict[str, CoordinationTask] = {}
        self.pending_votes: Dict[str, List[ConsensusVote]] = {}
        self.agent_loads: Dict[str, float] = {}
        
        self.coordination_history: List[Dict[str, Any]] = []
        
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        action = message.content.get("action", "coordinate")
        
        try:
            if action == "coordinate":
                result = await self._coordinate_agents(message.content)
            elif action == "delegate":
                result = await self._delegate_task(message.content)
            elif action == "consensus":
                result = await self._build_consensus(message.content)
            elif action == "vote":
                result = await self._process_vote(message.content)
            elif action == "sync":
                result = await self._synchronize_agents(message.content)
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
        active_count = len(self.active_tasks)
        pending_votes_count = len(self.pending_votes)
        agent_loads = context.get("agent_loads", {})
        
        overloaded_agents = [a for a, load in agent_loads.items() if load > 0.9]
        
        return {
            "active_tasks": active_count,
            "pending_votes": pending_votes_count,
            "overloaded_agents": overloaded_agents,
            "coordination_load": self._calculate_coordination_load(),
            "confidence": 0.85,
        }
    
    async def _coordinate_agents(self, content: Dict[str, Any]) -> Dict[str, Any]:
        task_description = content.get("task", "general coordination")
        required_agents = content.get("required_agents", [])
        parallel = content.get("parallel", True)
        
        if parallel:
            results = await self._coordinate_parallel(required_agents, task_description)
        else:
            results = await self._coordinate_sequential(required_agents, task_description)
        
        self.coordination_history.append({
            "timestamp": time.time(),
            "task": task_description,
            "agents": required_agents,
            "results": results,
        })
        
        return {
            "status": "coordinated",
            "results": results,
            "participating_agents": len(required_agents),
        }
    
    async def _delegate_task(self, content: Dict[str, Any]) -> Dict[str, Any]:
        task_description = content.get("description", "")
        target_agent = content.get("agent_id")
        priority = content.get("priority", 1)
        deadline = content.get("deadline")
        
        if not target_agent:
            target_agent = self._select_least_loaded_agent()
        
        task_id = f"task_{int(time.time())}"
        
        task = CoordinationTask(
            id=task_id,
            description=task_description,
            assigned_agents=[target_agent],
            status="pending",
            deadline=deadline,
            priority=priority,
        )
        
        self.active_tasks[task_id] = task
        
        await self.event_bus.publish(Event(
            source=self.id,
            type=EventType.TASK_CREATED,
            data={"task_id": task_id, "agent": target_agent},
        ))
        
        return {
            "task_id": task_id,
            "assigned_agent": target_agent,
            "status": "delegated",
        }
    
    async def _build_consensus(self, content: Dict[str, Any]) -> Dict[str, Any]:
        decision_topic = content.get("topic", "")
        participating_agents = content.get("agents", [])
        required_confidence = content.get("required_confidence", 0.7)
        
        vote_id = f"vote_{int(time.time())}"
        self.pending_votes[vote_id] = []
        
        for agent_id in participating_agents:
            await self.event_bus.publish(Event(
                source=self.id,
                type=EventType.DECISION_PROPOSED,
                target=agent_id,
                data={
                    "vote_id": vote_id,
                    "topic": decision_topic,
                    "required_confidence": required_confidence,
                },
            ))
        
        await asyncio.sleep(0.5)
        
        votes = self.pending_votes.get(vote_id, [])
        
        consensus_reached, final_decision = self._tally_votes(votes, required_confidence)
        
        if consensus_reached:
            await self.event_bus.publish(Event(
                source=self.id,
                type=EventType.DECISION_APPROVED,
                data={
                    "vote_id": vote_id,
                    "decision": final_decision,
                    "votes": len(votes),
                },
            ))
        
        return {
            "vote_id": vote_id,
            "consensus_reached": consensus_reached,
            "final_decision": final_decision,
            "votes": [{"agent": v.agent_id, "decision": v.decision} for v in votes],
            "average_confidence": sum(v.confidence for v in votes) / len(votes) if votes else 0.0,
        }
    
    async def _process_vote(self, content: Dict[str, Any]) -> Dict[str, Any]:
        vote_id = content.get("vote_id")
        agent_id = content.get("agent_id")
        decision = content.get("decision")
        confidence = content.get("confidence", 0.5)
        reasoning = content.get("reasoning", "")
        
        if not vote_id or not agent_id or decision is None:
            return {"error": "Missing required fields"}
        
        vote = ConsensusVote(
            agent_id=agent_id,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=time.time(),
        )
        
        if vote_id not in self.pending_votes:
            self.pending_votes[vote_id] = []
        
        existing_idx = next(
            (i for i, v in enumerate(self.pending_votes[vote_id]) if v.agent_id == agent_id),
            -1
        )
        if existing_idx >= 0:
            self.pending_votes[vote_id][existing_idx] = vote
        else:
            self.pending_votes[vote_id].append(vote)
        
        return {
            "vote_id": vote_id,
            "recorded": True,
            "total_votes": len(self.pending_votes[vote_id]),
        }
    
    async def _synchronize_agents(self, content: Dict[str, Any]) -> Dict[str, Any]:
        agents_to_sync = content.get("agents", [])
        sync_data = content.get("data", {})
        
        sync_id = f"sync_{int(time.time())}"
        
        await self.event_bus.publish(Event(
            source=self.id,
            type=EventType.SYNC,
            data={
                "sync_id": sync_id,
                "data": sync_data,
                "broadcast": True,
            },
        ))
        
        await asyncio.sleep(0.1)
        
        return {
            "sync_id": sync_id,
            "synced_agents": agents_to_sync,
            "status": "synchronized",
        }
    
    async def _coordinate_parallel(self, agent_ids: List[str], task: str) -> Dict[str, Any]:
        tasks = []
        for agent_id in agent_ids:
            task_msg = AgentMessage(
                sender=self.id,
                recipient=agent_id,
                content={"action": "execute", "task": task},
            )
            tasks.append(task_msg)
        
        results = []
        for i, task_msg in enumerate(tasks):
            results.append({
                "agent_id": agent_ids[i],
                "status": "delegated",
                "task": task,
            })
        
        return {"type": "parallel", "results": results}
    
    async def _coordinate_sequential(self, agent_ids: List[str], task: str) -> Dict[str, Any]:
        results = []
        for agent_id in agent_ids:
            results.append({
                "agent_id": agent_id,
                "status": "delegated",
                "task": task,
            })
        
        return {"type": "sequential", "results": results}
    
    def _select_least_loaded_agent(self) -> Optional[str]:
        if not self.agent_loads:
            return None
        
        return min(self.agent_loads.items(), key=lambda x: x[1])[0]
    
    def _tally_votes(self, votes: List[ConsensusVote], required_confidence: float) -> tuple:
        if not votes:
            return False, None
        
        decision_votes: Dict[str, List[ConsensusVote]] = {}
        for vote in votes:
            if vote.decision not in decision_votes:
                decision_votes[vote.decision] = []
            decision_votes[vote.decision].append(vote)
        
        best_decision = None
        best_vote_count = 0
        best_avg_confidence = 0.0
        
        for decision, decision_votes_list in decision_votes.items():
            vote_count = len(decision_votes_list)
            avg_confidence = sum(v.confidence for v in decision_votes_list) / vote_count
            
            if vote_count > best_vote_count:
                best_vote_count = vote_count
                best_decision = decision
                best_avg_confidence = avg_confidence
            elif vote_count == best_vote_count and avg_confidence > best_avg_confidence:
                best_decision = decision
                best_avg_confidence = avg_confidence
        
        total_votes = len(votes)
        consensus_threshold = 0.5
        
        consensus_reached = (
            best_vote_count / total_votes >= consensus_threshold and
            best_avg_confidence >= required_confidence
        )
        
        return consensus_reached, best_decision
    
    def _calculate_coordination_load(self) -> float:
        active_load = len(self.active_tasks) / 100.0
        vote_load = len(self.pending_votes) / 50.0
        
        return min(1.0, active_load + vote_load)
    
    async def update_agent_load(self, agent_id: str, load: float):
        self.agent_loads[agent_id] = load
        
        if load > 0.95:
            await self.event_bus.publish(Event(
                source=self.id,
                type=EventType.WARNING,
                data={"agent_id": agent_id, "load": load},
            ))
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = "completed"
            task.result = result
            
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]
            
            await self.event_bus.publish(Event(
                source=self.id,
                type=EventType.TASK_COMPLETED,
                data={"task_id": task_id, "result": result},
            ))
