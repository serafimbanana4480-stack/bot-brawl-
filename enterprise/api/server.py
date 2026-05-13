"""FastAPI Backend - Enterprise AI Multi-Agent Platform API"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json
import time

from ..orchestration.event_bus import EventBus, EventType
from ..orchestration.engine import OrchestrationEngine, TaskPriority
from ..agents.base import AgentConfig, AgentType
from ..agents.supervisor import SupervisorAgent
from ..agents.strategy import StrategyAgent
from ..agents.combat import CombatAgent
from ..agents.vision_agent import VisionAgent
from ..agents.navigation import NavigationAgent
from ..agents.tactical import TacticalPlannerAgent
from ..agents.replay import ReplayAnalystAgent
from ..agents.learning import LearningAgent
from ..agents.memory_agent import MemoryAgent
from ..agents.reflection import ReflectionAgent
from ..agents.coordination import CoordinationAgent
from ..observability.tracing import TracingService
from ..observability.metrics import MetricsCollector
from ..observability.logging_service import StructuredLogging
from ..memory.hybrid import HybridMemorySystem


class AgentCreateRequest(BaseModel):
    name: str
    agent_type: str
    config: Optional[Dict[str, Any]] = None


class TaskCreateRequest(BaseModel):
    name: str
    agent_type: str
    input_data: Dict[str, Any]
    priority: Optional[str] = "normal"


class MessageRequest(BaseModel):
    recipient: str
    content: Dict[str, Any]
    message_type: Optional[str] = "request"


class SystemStatusResponse(BaseModel):
    status: str
    uptime: float
    agents_count: int
    active_tasks: int
    metrics: Dict[str, Any]


app = FastAPI(title="Enterprise AI Multi-Agent Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

event_bus = EventBus()
engine = OrchestrationEngine(event_bus)
tracing_service = TracingService()
metrics = MetricsCollector()
logging_service = StructuredLogging()
memory_system = HybridMemorySystem()

agents: Dict[str, Any] = {}
start_time = time.time()

websocket_connections: List[WebSocket] = []


async def initialize_agents():
    agent_configs = [
        ("supervisor", SupervisorAgent),
        ("strategy", StrategyAgent),
        ("combat", CombatAgent),
        ("vision", VisionAgent),
        ("navigation", NavigationAgent),
        ("tactical", TacticalPlannerAgent),
        ("replay", ReplayAnalystAgent),
        ("learning", LearningAgent),
        ("memory", MemoryAgent),
        ("reflection", ReflectionAgent),
        ("coordination", CoordinationAgent),
    ]
    
    for agent_name, agent_class in agent_configs:
        config = AgentConfig(
            name=agent_name,
            agent_type=AgentType(agent_name),
        )
        
        if agent_name == "supervisor":
            agent = agent_class(config, event_bus, engine)
        elif agent_name == "coordination":
            agent = agent_class(config, event_bus)
        else:
            agent = agent_class(config, event_bus)
        
        await agent.initialize()
        agents[agent.id] = agent
        engine.register_agent(agent)
        
        await metrics.increment(f"agent_initialized", labels={"agent": agent_name})


@app.on_event("startup")
async def startup_event():
    await initialize_agents()
    logging_service.info("Enterprise AI Platform started", agents_count=len(agents))


@app.on_event("shutdown")
async def shutdown_event():
    for agent in agents.values():
        await agent.shutdown()
    logging_service.info("Enterprise AI Platform shutdown")


@app.get("/")
async def root():
    return {
        "name": "Enterprise AI Multi-Agent Platform",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/status", response_model=SystemStatusResponse)
async def get_status():
    uptime = time.time() - start_time
    
    return SystemStatusResponse(
        status="healthy",
        uptime=uptime,
        agents_count=len(agents),
        active_tasks=len(engine.tasks),
        metrics={
            "event_bus_stats": event_bus.get_stats(),
            "orchestration_metrics": engine.get_metrics(),
            "system_metrics": await metrics.get_system_metrics(),
        },
    )


@app.get("/agents")
async def list_agents():
    return {
        "agents": [
            {
                "id": agent.id,
                "name": agent.name,
                "type": agent.agent_type.value,
                "status": agent.status.value,
                "metrics": agent.get_metrics(),
            }
            for agent in agents.values()
        ]
    }


@app.post("/agents")
async def create_agent(request: AgentCreateRequest):
    agent_type = AgentType(request.agent_type)
    
    config = AgentConfig(
        name=request.name,
        agent_type=agent_type,
        **(request.config or {}),
    )
    
    if request.agent_type == "supervisor":
        agent = SupervisorAgent(config, event_bus, engine)
    elif request.agent_type == "coordination":
        agent = CoordinationAgent(config, event_bus)
    else:
        agent_class = globals().get(f"{request.agent_type.title()}Agent")
        if agent_class:
            agent = agent_class(config, event_bus)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown agent type: {request.agent_type}")
    
    await agent.initialize()
    agents[agent.id] = agent
    engine.register_agent(agent)
    
    await metrics.increment("agent_created", labels={"type": request.agent_type})
    
    return {"id": agent.id, "name": agent.name, "type": agent.agent_type.value}


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    return {
        "id": agent.id,
        "name": agent.name,
        "type": agent.agent_type.value,
        "status": agent.status.value,
        "metrics": agent.get_metrics(),
    }


@app.post("/agents/{agent_id}/message")
async def send_message(agent_id: str, request: MessageRequest):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    
    from ..agents.base import AgentMessage
    message = AgentMessage(
        sender="api",
        recipient=agent_id,
        content=request.content,
        message_type=request.message_type,
    )
    
    response = await agent.process(message)
    
    await metrics.increment("message_sent", labels={"agent": agent.name})
    
    return response.to_dict()


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [task.to_dict() for task in engine.tasks.values()]
    }


@app.post("/tasks")
async def create_task(request: TaskCreateRequest):
    agent_type = AgentType(request.agent_type)
    
    task = await engine.create_task(
        name=request.name,
        agent_type=agent_type,
        input_data=request.input_data,
        priority=TaskPriority[request.priority.upper()],
    )
    
    await metrics.increment("task_created")
    
    return {"id": task.id, "name": task.name, "status": task.status.value}


@app.post("/tasks/{task_id}/execute")
async def execute_task(task_id: str):
    response = await engine.execute_task(task_id)
    return response.to_dict()


@app.get("/workflow/graph")
async def get_workflow_graph():
    return engine.get_task_graph()


@app.get("/events")
async def get_events(limit: int = 100):
    return {
        "events": [e.to_dict() for e in event_bus.get_history(limit=limit)]
    }


@app.get("/logs")
async def get_logs(level: Optional[str] = None, limit: int = 100):
    return {
        "logs": logging_service.get_logs(level=level, limit=limit)
    }


@app.get("/metrics")
async def get_metrics():
    return await metrics.get_all_metrics_summary()


@app.get("/tracing/{trace_id}")
async def get_trace(trace_id: str):
    return {
        "trace": tracing_service.get_trace(trace_id)
    }


@app.get("/memory/stats")
async def get_memory_stats():
    return memory_system.get_stats()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                event_types = [EventType(t) for t in message.get("events", [])]
                await event_bus.subscribe(
                    f"ws_{id(websocket)}",
                    lambda e: asyncio.create_task(websocket.send_json(e.to_dict())),
                    event_types,
                )
            
            elif message.get("type") == "message":
                if "recipient" in message and message["recipient"] in agents:
                    agent = agents[message["recipient"]]
                    from ..agents.base import AgentMessage
                    agent_msg = AgentMessage(
                        sender="websocket",
                        recipient=message["recipient"],
                        content=message.get("content", {}),
                    )
                    response = await agent.process(agent_msg)
                    await websocket.send_json({
                        "type": "response",
                        "data": response.to_dict(),
                    })
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        await event_bus.unsubscribe(f"ws_{id(websocket)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
