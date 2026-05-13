"""Enterprise AI Multi-Agent Platform - Quick Start"""

import asyncio
from enterprise.orchestration.event_bus import EventBus, EventType
from enterprise.orchestration.engine import OrchestrationEngine
from enterprise.agents.base import AgentConfig, AgentType
from enterprise.agents.supervisor import SupervisorAgent
from enterprise.agents.strategy import StrategyAgent
from enterprise.agents.combat import CombatAgent
from enterprise.observability.logging_service import StructuredLogging


async def main():
    logging = StructuredLogging("enterprise-ai")
    logging.info("Starting Enterprise AI Multi-Agent Platform")
    
    event_bus = EventBus()
    engine = OrchestrationEngine(event_bus)
    
    agents = {}
    
    supervisor_config = AgentConfig(
        name="supervisor",
        agent_type=AgentType.SUPERVISOR,
    )
    supervisor = SupervisorAgent(supervisor_config, event_bus, engine)
    await supervisor.initialize()
    agents[supervisor.id] = supervisor
    engine.register_agent(supervisor)
    logging.info(f"Initialized Supervisor Agent: {supervisor.id}")
    
    strategy_config = AgentConfig(
        name="strategy",
        agent_type=AgentType.STRATEGY,
    )
    strategy = StrategyAgent(strategy_config, event_bus)
    await strategy.initialize()
    agents[strategy.id] = strategy
    engine.register_agent(strategy)
    logging.info(f"Initialized Strategy Agent: {strategy.id}")
    
    combat_config = AgentConfig(
        name="combat",
        agent_type=AgentType.COMBAT,
    )
    combat = CombatAgent(combat_config, event_bus)
    await combat.initialize()
    agents[combat.id] = combat
    engine.register_agent(combat)
    logging.info(f"Initialized Combat Agent: {combat.id}")
    
    from enterprise.agents.base import AgentMessage
    
    logging.info("Sending test message to Strategy Agent")
    message = AgentMessage(
        sender="test",
        recipient=strategy.id,
        content={
            "action": "plan",
            "objectives": [
                {"name": "defeat_enemies", "priority": 1, "metrics": {"kills": 5}},
                {"name": "survive", "priority": 2, "metrics": {"health": 50}},
            ],
            "game_state": {"health": 80, "position": (100, 100)},
        },
    )
    
    response = await strategy.process(message)
    logging.info(f"Strategy Agent response: success={response.success}, confidence={response.confidence}")
    
    logging.info("Sending combat evaluation to Combat Agent")
    combat_message = AgentMessage(
        sender="test",
        recipient=combat.id,
        content={
            "action": "evaluate",
            "game_state": {"health": 100},
            "enemies": [
                {"id": "e1", "position": (200, 200), "type": "normal", "health": 50, "damage_output": 15},
                {"id": "e2", "position": (300, 150), "type": "ranged", "health": 30, "damage_output": 20},
            ],
            "allies": [{"id": "a1", "position": (150, 150)}],
            "my_position": (100, 100),
        },
    )
    
    combat_response = await combat.process(combat_message)
    logging.info(f"Combat Agent response: success={combat_response.success}")
    logging.info(f"Recommended action: {combat_response.data.get('recommended_action')}")
    
    logging.info("Enterprise AI Platform demo completed successfully!")
    
    for agent in agents.values():
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
