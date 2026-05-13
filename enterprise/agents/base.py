"""Base Agent Protocol - Common interface for all agents in the enterprise platform"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Callable
from enum import Enum
from datetime import datetime
import uuid
import asyncio


class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"


class AgentType(Enum):
    SUPERVISOR = "supervisor"
    STRATEGY = "strategy"
    COMBAT = "combat"
    VISION = "vision"
    NAVIGATION = "navigation"
    TACTICAL = "tactical"
    REPLAY = "replay"
    LEARNING = "learning"
    MEMORY = "memory"
    REFLECTION = "reflection"
    COORDINATION = "coordination"


@dataclass
class AgentMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sender: str = ""
    recipient: Optional[str] = None
    agent_type: Optional[AgentType] = None
    content: Dict[str, Any] = field(default_factory=dict)
    message_type: str = "request"
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "recipient": self.recipient,
            "agent_type": self.agent_type.value if self.agent_type else None,
            "content": self.content,
            "message_type": self.message_type,
            "reply_to": self.reply_to,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


@dataclass
class AgentResponse:
    success: bool
    message: AgentMessage
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    confidence: float = 1.0
    processing_time: float = 0.0
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message.to_dict(),
            "data": self.data,
            "error": self.error,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "alternatives": self.alternatives,
        }


@dataclass
class AgentConfig:
    name: str
    agent_type: AgentType
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 30.0
    retry_attempts: int = 3
    memory_enabled: bool = True
    learning_enabled: bool = True
    streaming: bool = True
    callbacks: Dict[str, Callable] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    

@dataclass
class ConfidenceScore:
    value: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.value = max(0.0, min(1.0, self.value))
    
    def add_factor(self, name: str, value: float, reasoning: str = ""):
        self.factors[name] = max(0.0, min(1.0, value))
        if reasoning:
            self.reasoning.append(reasoning)
        self._recalculate()
    
    def _recalculate(self):
        if self.factors:
            self.value = sum(self.factors.values()) / len(self.factors)


class BaseAgent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.id = str(uuid.uuid4())
        self.name = config.name
        self.agent_type = config.agent_type
        self.status = AgentStatus.IDLE
        
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._processing = False
        self._lock = asyncio.Lock()
        
        self._metrics = {
            "messages_received": 0,
            "messages_sent": 0,
            "errors": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
        }
        
    @abstractmethod
    async def process(self, message: AgentMessage) -> AgentResponse:
        pass
    
    @abstractmethod
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    async def initialize(self) -> bool:
        return True
    
    async def shutdown(self):
        self.status = AgentStatus.OFFLINE
        
    async def receive(self, message: AgentMessage):
        await self._message_queue.put(message)
        self._metrics["messages_received"] += 1
        
    async def send(self, recipient: str, content: Dict[str, Any], 
                   message_type: str = "request") -> AgentMessage:
        message = AgentMessage(
            sender=self.id,
            recipient=recipient,
            content=content,
            message_type=message_type,
            agent_type=self.agent_type,
        )
        self._metrics["messages_sent"] += 1
        return message
    
    def on_event(self, event_type: str, handler: Callable):
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                await handler(data)
    
    def calculate_confidence(self, factors: Dict[str, float]) -> ConfidenceScore:
        score = ConfidenceScore()
        for name, value in factors.items():
            score.add_factor(name, value)
        return score
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self._metrics,
            "status": self.status.value,
            "queue_size": self._message_queue.qsize(),
            "id": self.id,
            "type": self.agent_type.value,
        }
    
    async def process_queue(self):
        if self._processing:
            return
        self._processing = True
        
        try:
            while not self._message_queue.empty():
                message = await self._message_queue.get()
                self.status = AgentStatus.PROCESSING
                await self.process(message)
        finally:
            self._processing = False
            self.status = AgentStatus.IDLE
    
    def get_system_prompt(self) -> str:
        if self.config.system_prompt:
            return self.config.system_prompt
        return f"""You are {self.name}, a {self.agent_type.value} agent in a multi-agent AI system.
Your role is to process requests, make decisions, and coordinate with other agents.
Always maintain high confidence standards and provide reasoning for your decisions."""
