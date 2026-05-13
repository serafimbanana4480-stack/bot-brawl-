"""Orchestration Engine - LangGraph-based workflow orchestration with parallel execution"""

import asyncio
import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime
from enum import Enum
import logging

from ..agents.base import AgentMessage, AgentResponse, AgentType, BaseAgent, AgentConfig
from .event_bus import EventBus, Event, EventType, EventPriority


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    task_type: str = "generic"
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "status": self.status.value,
            "priority": self.priority.value,
            "assigned_agent": self.assigned_agent,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "parent_id": self.parent_id,
            "children": self.children_ids,
            "retry_count": self.retry_count,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class WorkflowStep:
    def __init__(self, name: str, agent_type: AgentType, 
                 task_template: Dict[str, Any],
                 condition: Optional[Callable] = None,
                 fallback: Optional[str] = None):
        self.name = name
        self.agent_type = agent_type
        self.task_template = task_template
        self.condition = condition
        self.fallback = fallback


class Workflow:
    def __init__(self, name: str, description: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.steps: List[WorkflowStep] = []
        self.parallel_groups: List[List[WorkflowStep]] = []
        self.current_step = 0
        self.status = TaskStatus.PENDING
        self.created_at = datetime.utcnow()
        
    def add_step(self, step: WorkflowStep):
        self.steps.append(step)
        
    def add_parallel_steps(self, steps: List[WorkflowStep]):
        self.parallel_groups.append(steps)


@dataclass
class ConsensusResult:
    reached: bool
    decision: Optional[Dict[str, Any]] = None
    votes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)


class OrchestrationEngine:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: Dict[str, Task] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.logger = logging.getLogger("orchestration")
        
        self._running_tasks: Set[str] = set()
        self._task_results: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._execution_lock = asyncio.Lock()
        
    def register_agent(self, agent: BaseAgent):
        self.agents[agent.id] = agent
        self.logger.info(f"Registered agent: {agent.name} ({agent.id})")
        
    async def unregister_agent(self, agent_id: str):
        async with self._lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                await agent.shutdown()
                del self.agents[agent_id]
                
    async def create_task(self, name: str, agent_type: AgentType,
                         input_data: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.NORMAL,
                         parent_id: Optional[str] = None) -> Task:
        task = Task(
            name=name,
            task_type=agent_type.value,
            priority=priority,
            input_data=input_data,
            parent_id=parent_id,
        )
        
        async with self._lock:
            self.tasks[task.id] = task
            
        if parent_id and parent_id in self.tasks:
            self.tasks[parent_id].children_ids.append(task.id)
            
        await self.event_bus.publish(Event(
            source="orchestration",
            type=EventType.TASK_CREATED,
            data=task.to_dict(),
        ))
        
        return task
    
    async def assign_task(self, task_id: str, agent_id: str) -> bool:
        async with self._lock:
            if task_id not in self.tasks:
                return False
            if agent_id not in self.agents:
                return False
                
            task = self.tasks[task_id]
            task.assigned_agent = agent_id
            return True
    
    async def execute_task(self, task_id: str) -> AgentResponse:
        async with self._execution_lock:
            if task_id not in self.tasks:
                return AgentResponse(
                    success=False,
                    message=AgentMessage(),
                    error="Task not found",
                )
            
            task = self.tasks[task_id]
            if task.status == TaskStatus.RUNNING:
                return AgentResponse(
                    success=False,
                    message=AgentMessage(),
                    error="Task already running",
                )
            
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            self._running_tasks.add(task_id)
            
            await self.event_bus.publish(Event(
                source="orchestration",
                type=EventType.TASK_CREATED,
                data={"task_id": task_id, "status": "running"},
            ))
        
        start_time = time.time()
        agent = None
        
        try:
            agent = self._find_agent_for_task(task)
            if not agent:
                raise Exception(f"No agent available for task type: {task.task_type}")
            
            message = AgentMessage(
                sender="orchestration",
                recipient=agent.id,
                content=task.input_data,
                metadata={"task_id": task_id},
            )
            
            response = await agent.process(message)
            
            task.output_data = response.data
            task.confidence = response.confidence
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            self._task_results[task_id] = response
            
            await self.event_bus.publish(Event(
                source="orchestration",
                type=EventType.TASK_COMPLETED,
                data=task.to_dict(),
            ))
            
            return response
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                self.logger.warning(f"Task {task_id} failed, retry {task.retry_count}/{task.max_retries}")
            else:
                await self.event_bus.publish(Event(
                    source="orchestration",
                    type=EventType.TASK_FAILED,
                    data=task.to_dict(),
                ))
            
            return AgentResponse(
                success=False,
                message=AgentMessage(),
                error=str(e),
                processing_time=time.time() - start_time,
            )
        finally:
            self._running_tasks.discard(task_id)
    
    def _find_agent_for_task(self, task: Task) -> Optional[BaseAgent]:
        for agent in self.agents.values():
            if agent.agent_type.value == task.task_type and agent.status.value != "offline":
                return agent
        return None
    
    async def execute_parallel(self, task_ids: List[str]) -> Dict[str, AgentResponse]:
        results = await asyncio.gather(
            *[self.execute_task(tid) for tid in task_ids],
            return_exceptions=True
        )
        
        return {
            task_ids[i]: results[i] if not isinstance(results[i], Exception) 
            else AgentResponse(success=False, message=AgentMessage(), error=str(results[i]))
            for i in range(len(task_ids))
        }
    
    async def decompose_task(self, task: Task, 
                            decomposition_fn: Callable[[Task], List[Task]]) -> List[Task]:
        subtasks = decomposition_fn(task)
        
        async with self._lock:
            for subtask in subtasks:
                subtask.parent_id = task.id
                self.tasks[subtask.id] = subtask
                task.children_ids.append(subtask.id)
        
        return subtasks
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        if workflow_id not in self.workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow = self.workflows[workflow_id]
        results = []
        
        await self.event_bus.publish(Event(
            source="orchestration",
            type=EventType.WORKFLOW_STARTED,
            data={"workflow_id": workflow_id, "name": workflow.name},
        ))
        
        for step in workflow.steps:
            if step.condition and not step.condition(step):
                if step.fallback:
                    fallback_task = await self.create_task(
                        f"{step.name}_fallback",
                        step.agent_type,
                        {"fallback": True}
                    )
                    await self.execute_task(fallback_task.id)
                continue
            
            task = await self.create_task(
                step.name,
                step.agent_type,
                step.task_template,
            )
            
            response = await self.execute_task(task.id)
            results.append({"step": step.name, "result": response.to_dict()})
            
            await self.event_bus.publish(Event(
                source="orchestration",
                type=EventType.WORKFLOW_STEP,
                data={"workflow_id": workflow_id, "step": step.name},
            ))
        
        await self.event_bus.publish(Event(
            source="orchestration",
            type=EventType.WORKFLOW_COMPLETED,
            data={"workflow_id": workflow_id, "results": results},
        ))
        
        return {"success": True, "workflow_id": workflow_id, "results": results}
    
    async def request_consensus(self, task_id: str,
                               participating_agents: List[str],
                               decision_data: Dict[str, Any]) -> ConsensusResult:
        votes = {}
        
        for agent_id in participating_agents:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                message = AgentMessage(
                    sender="orchestration",
                    recipient=agent_id,
                    content={
                        "action": "vote",
                        "decision_data": decision_data,
                        "task_id": task_id,
                    },
                )
                
                try:
                    response = await asyncio.wait_for(
                        agent.process(message),
                        timeout=10.0
                    )
                    votes[agent_id] = response.data or {}
                except Exception as e:
                    votes[agent_id] = {"error": str(e)}
        
        decision_values = [v.get("decision") for v in votes.values() if "decision" in v]
        if decision_values and len(set(decision_values)) == 1:
            return ConsensusResult(
                reached=True,
                decision=decision_values[0],
                votes=votes,
                confidence=1.0,
                reasoning=["Unanimous agreement reached"],
            )
        
        return ConsensusResult(
            reached=False,
            votes=votes,
            confidence=0.5,
            reasoning=["No consensus reached - conflicting decisions"],
        )
    
    async def dynamic_route(self, task: Task, 
                            routes: Dict[str, Callable]) -> str:
        for route_name, condition in routes.items():
            try:
                if asyncio.iscoroutinefunction(condition):
                    result = await condition(task)
                else:
                    result = condition(task)
                if result:
                    return route_name
            except Exception as e:
                self.logger.error(f"Route condition error for {route_name}: {e}")
        
        return "default"
    
    def get_task_graph(self) -> Dict[str, Any]:
        nodes = []
        edges = []
        
        for task in self.tasks.values():
            nodes.append({
                "id": task.id,
                "label": task.name,
                "status": task.status.value,
                "type": task.task_type,
            })
            
            for child_id in task.children_ids:
                edges.append({"from": task.id, "to": child_id})
        
        return {"nodes": nodes, "edges": edges}
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_agents": len(self.agents),
            "total_tasks": len(self.tasks),
            "running_tasks": len(self._running_tasks),
            "completed_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
            "pending_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
        }
