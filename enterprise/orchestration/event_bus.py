"""Event Bus - Real-time communication system between agents"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Optional, Set
from datetime import datetime
from enum import Enum
import logging


class EventType(Enum):
    AGENT_MESSAGE = "agent_message"
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_STEP = "workflow_step"
    WORKFLOW_COMPLETED = "workflow_completed"
    DECISION_PROPOSED = "decision_proposed"
    DECISION_APPROVED = "decision_approved"
    DECISION_REJECTED = "decision_rejected"
    VISION_UPDATE = "vision_update"
    MEMORY_RETRIEVED = "memory_retrieved"
    MEMORY_STORED = "memory_stored"
    LEARNING_UPDATE = "learning_update"
    METRICS_UPDATE = "metrics_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    SYNC = "sync"


class EventPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.AGENT_MESSAGE
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    target: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    broadcast: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "target": self.target,
            "priority": self.priority.value,
            "data": self.data,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "broadcast": self.broadcast,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=EventType(data.get("type", "agent_message")),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            source=data.get("source", ""),
            target=data.get("target"),
            priority=EventPriority(data.get("priority", 1)),
            data=data.get("data", {}),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            broadcast=data.get("broadcast", False),
        )


class Subscriber:
    def __init__(self, agent_id: str, callback: Callable, event_types: Set[EventType]):
        self.agent_id = agent_id
        self.callback = callback
        self.event_types = event_types
        self.active = True


class EventBus:
    def __init__(self, enable_logging: bool = True):
        self._subscribers: Dict[str, List[Subscriber]] = {}
        self._event_queues: Dict[str, asyncio.Queue] = {}
        self._history: List[Event] = []
        self._max_history = 10000
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger("event_bus")
        self._enable_logging = enable_logging
        self._event_counter = 0
        
    async def subscribe(self, agent_id: str, callback: Callable, 
                       event_types: List[EventType]):
        async with self._lock:
            if agent_id not in self._subscribers:
                self._subscribers[agent_id] = []
            self._subscribers[agent_id].append(
                Subscriber(agent_id, callback, set(event_types))
            )
            if self._enable_logging:
                self._logger.info(f"Agent {agent_id} subscribed to {[e.value for e in event_types]}")
    
    async def unsubscribe(self, agent_id: str):
        async with self._lock:
            if agent_id in self._subscribers:
                for sub in self._subscribers[agent_id]:
                    sub.active = False
                del self._subscribers[agent_id]
    
    async def publish(self, event: Event):
        async with self._lock:
            self._event_counter += 1
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        
        if self._enable_logging and event.type in [EventType.ERROR, EventType.DECISION_PROPOSED]:
            self._logger.debug(f"Event published: {event.type.value} from {event.source}")
        
        tasks = []
        async with self._lock:
            for agent_id, subs in self._subscribers.items():
                if agent_id == event.source and not event.broadcast:
                    continue
                    
                for sub in subs:
                    if not sub.active:
                        continue
                    if event.target and agent_id != event.target:
                        continue
                    if event.type not in sub.event_types:
                        continue
                    
                    if agent_id not in self._event_queues:
                        self._event_queues[agent_id] = asyncio.Queue()
                    
                    tasks.append(self._deliver_event(sub, event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _deliver_event(self, sub: Subscriber, event: Event):
        try:
            if asyncio.iscoroutinefunction(sub.callback):
                await sub.callback(event)
            else:
                sub.callback(event)
        except Exception as e:
            self._logger.error(f"Error delivering event to {sub.agent_id}: {e}")
    
    async def request(self, source: str, target: str, 
                     action: str, data: Dict[str, Any],
                     timeout: float = 30.0) -> Optional[Event]:
        correlation_id = str(uuid.uuid4())
        event = Event(
            source=source,
            target=target,
            type=EventType.AGENT_MESSAGE,
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
            data={"action": action, "payload": data},
        )
        
        future = asyncio.get_event_loop().create_future()
        responses = {}
        
        async def response_handler(resp: Event):
            if resp.correlation_id == correlation_id:
                responses[resp.source] = resp
                if not future.done():
                    future.set_result(resp)
        
        await self.subscribe(f"{source}_response_{correlation_id}", 
                           response_handler, [EventType.AGENT_MESSAGE])
        
        try:
            await self.publish(event)
            try:
                return await asyncio.wait_for(future, timeout)
            except asyncio.TimeoutError:
                return None
        finally:
            await self.unsubscribe(f"{source}_response_{correlation_id}")
    
    async def broadcast(self, source: str, event_type: EventType,
                      data: Dict[str, Any], priority: EventPriority = EventPriority.NORMAL):
        event = Event(
            source=source,
            type=event_type,
            priority=priority,
            data=data,
            broadcast=True,
        )
        await self.publish(event)
    
    def get_history(self, event_type: Optional[EventType] = None,
                   source: Optional[str] = None,
                   limit: int = 100) -> List[Event]:
        filtered = self._history
        if event_type:
            filtered = [e for e in filtered if e.type == event_type]
        if source:
            filtered = [e for e in filtered if e.source == source]
        return filtered[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_events": self._event_counter,
            "history_size": len(self._history),
            "subscribers": sum(len(subs) for subs in self._subscribers.values()),
            "agents": len(self._subscribers),
        }
    
    async def create_queue(self, agent_id: str) -> asyncio.Queue:
        async with self._lock:
            if agent_id not in self._event_queues:
                self._event_queues[agent_id] = asyncio.Queue()
            return self._event_queues[agent_id]
    
    async def get_queue_events(self, agent_id: str, timeout: float = 1.0) -> List[Event]:
        queue = await self.create_queue(agent_id)
        events = []
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout)
                    events.append(event)
                except asyncio.TimeoutError:
                    break
        except asyncio.CancelledError:
            pass
        return events
