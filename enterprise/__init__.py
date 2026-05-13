"""Enterprise AI Operating System - Multi-Agent Strategic Platform
Inspired by: LangGraph, CrewAI, AutoGen, CopilotKit AG-UI, FlowiseAI

Complete implementation for Brawl Stars AI with:
- Multi-agent orchestration (11 specialized agents)
- Vision pipeline (YOLOv8, ByteTrack, Minimap)
- Reinforcement Learning (PPO, SAC, Imitation Learning)
- Hybrid Memory System (Vector, Episodic, Semantic)
- Enterprise observability (Tracing, Metrics, Logging)
- FastAPI backend with WebSocket support
- Next.js 15 dashboard
"""

__version__ = "2.0.0"
__enterprise__ = True

from .agents.base import (
    BaseAgent,
    AgentConfig,
    AgentMessage,
    AgentResponse,
    AgentStatus,
    AgentType,
    ConfidenceScore,
)

from .agents.supervisor import SupervisorAgent
from .agents.strategy import StrategyAgent
from .agents.combat import CombatAgent
from .agents.vision_agent import VisionAgent
from .agents.navigation import NavigationAgent
from .agents.tactical import TacticalPlannerAgent
from .agents.replay import ReplayAnalystAgent
from .agents.learning import LearningAgent
from .agents.memory_agent import MemoryAgent
from .agents.reflection import ReflectionAgent
from .agents.coordination import CoordinationAgent

from .orchestration.event_bus import EventBus, Event, EventType, EventPriority
from .orchestration.engine import OrchestrationEngine, Task, TaskStatus, TaskPriority

from .memory.hybrid import HybridMemorySystem
from .memory.vector import VectorMemory
from .memory.episodic import EpisodicMemory
from .memory.semantic import SemanticMemory

from .observability.tracing import TracingService
from .observability.metrics import MetricsCollector
from .observability.logging_service import StructuredLogging

from .vision.pipeline import VisionPipeline
from .vision.yolo_detector import YOLOv8Detector
from .vision.tracker_integration import TrackerIntegration
from .vision.minimap import MinimapUnderstanding

from .learning.rl import RLFramework, ReplayBuffer
from .learning.imitation import ImitationLearning
from .learning.curriculum import CurriculumLearning

from .simulation.benchmarks import SimulationEnvironment, BenchmarkSuite

from .research.git_research import GitResearchAgent
from .research.rl_research import RLResearchAgent
from .research.vision_research import VisionPipelineResearchAgent
from .research.integration_planner import IntegrationPlannerAgent
from .research.benchmark_agent import BenchmarkAgent

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "AgentMessage",
    "AgentResponse",
    "AgentStatus",
    "AgentType",
    "ConfidenceScore",
    "SupervisorAgent",
    "StrategyAgent",
    "CombatAgent",
    "VisionAgent",
    "NavigationAgent",
    "TacticalPlannerAgent",
    "ReplayAnalystAgent",
    "LearningAgent",
    "MemoryAgent",
    "ReflectionAgent",
    "CoordinationAgent",
    "EventBus",
    "Event",
    "EventType",
    "EventPriority",
    "OrchestrationEngine",
    "Task",
    "TaskStatus",
    "TaskPriority",
    "HybridMemorySystem",
    "VectorMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "TracingService",
    "MetricsCollector",
    "StructuredLogging",
    "VisionPipeline",
    "YOLOv8Detector",
    "TrackerIntegration",
    "MinimapUnderstanding",
    "RLFramework",
    "ReplayBuffer",
    "ImitationLearning",
    "CurriculumLearning",
    "SimulationEnvironment",
    "BenchmarkSuite",
    "GitResearchAgent",
    "RLResearchAgent",
    "VisionPipelineResearchAgent",
    "IntegrationPlannerAgent",
    "BenchmarkAgent",
]
