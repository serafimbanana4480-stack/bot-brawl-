"""RL Research Agent - Reinforcement Learning repository research and analysis"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass

from ..agents.base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType
from ..orchestration.event_bus import EventBus, EventType


@dataclass
class RLFramework:
    name: str
    repo_url: str
    type: str
    algorithms: List[str]
    features: List[str]
    compatibility: float
    performance_score: float
    documentation_quality: float


@dataclass
class RLResearchResult:
    frameworks: List[RLFramework]
    algorithms_found: Set[str]
    best_practices: List[str]
    integration_difficulty: str
    recommendations: List[str]


class RLResearchAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        
        self.rl_frameworks = {
            "rllib": {
                "name": "RLlib",
                "url": "ray-project/ray",
                "type": "full_platform",
                "algorithms": ["PPO", "SAC", "DQN", "A2C", "A3C", "TD3", "ES", "APEX"],
                "features": ["distributed_training", "multi_agent", "population_based"],
            },
            "stable_baselines3": {
                "name": "Stable Baselines3",
                "url": "DLR-RM/stable-baselines3",
                "type": "beginner_friendly",
                "algorithms": ["PPO", "SAC", "TD3", "DQN", "A2C"],
                "features": ["documented", "easy_to_use", "benchmarked"],
            },
            "torch_reinforce": {
                "name": "PyTorch Reinforcement Learning",
                "url": "pytorch/examples",
                "type": "educational",
                "algorithms": ["REINFORCE", "Actor_Critic"],
                "features": ["simple", "educational", "customizable"],
            },
            "tianshou": {
                "name": "Tianshou",
                "url": "thudm/tianshou",
                "type": "high_performance",
                "algorithms": ["DQN", "DoubleDQN", "DuelingDQN", "PPO", "A2C", "SAC"],
                "features": ["fast", "parallel", "modular"],
            },
            "skrl": {
                "name": "skrl",
                "url": "tnakae/skrl",
                "type": "modern",
                "algorithms": ["PPO", "PPO1", "PPO2", "A2C", "ACER", "ACKTR", "SAC", "TD3"],
                "features": ["multi_agent", "shared_critic", "visualization"],
            },
        }
        
        self.algorithm_keywords = {
            "PPO": ["ppo", "proximal_policy", "clipped_objective"],
            "SAC": ["sac", "soft_actor", "soft_q"],
            "TD3": ["td3", "twin_delayed", "deterministic_policy"],
            "DQN": ["dqn", "deep_q", "q_network", "replay_buffer"],
            "A2C": ["a2c", "advantage_actor", "synchronous"],
            "A3C": ["a3c", "asynchronous", "actor_critic"],
            "DDPG": ["ddpg", "deep_deterministic", "deterministic_policy_gradient"],
        }
        
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = asyncio.get_event_loop().time()
        action = message.content.get("action", "research")
        
        try:
            if action == "research":
                result = await self._research_rl_frameworks(message.content)
            elif action == "compare":
                result = await self._compare_rl_frameworks(message.content)
            elif action == "recommend":
                result = await self._recommend_rl_setup(message.content)
            else:
                result = {"error": f"Unknown action: {action}"}
            
            return AgentResponse(
                success=True,
                message=message,
                data=result,
                confidence=0.85,
                processing_time=asyncio.get_event_loop().time() - start_time,
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                message=message,
                error=str(e),
                processing_time=asyncio.get_event_loop().time() - start_time,
            )
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "frameworks_available": len(self.rl_frameworks),
            "ready": True,
            "confidence": 0.9,
        }
    
    async def _research_rl_frameworks(self, content: Dict[str, Any]) -> Dict[str, Any]:
        focus_algorithms = content.get("algorithms", ["PPO", "SAC", "DQN"])
        requirements = content.get("requirements", {})
        
        suitable_frameworks = []
        algorithms_found = set()
        
        for fw_key, fw_info in self.rl_frameworks.items():
            fw_algorithms = set(fw_info["algorithms"])
            target_algorithms = set(focus_algorithms)
            
            match_score = len(fw_algorithms & target_algorithms) / len(target_algorithms)
            
            if match_score > 0:
                framework = RLFramework(
                    name=fw_info["name"],
                    repo_url=f"https://github.com/{fw_info['url']}",
                    type=fw_info["type"],
                    algorithms=list(fw_algorithms & target_algorithms),
                    features=fw_info["features"],
                    compatibility=match_score,
                    performance_score=self._estimate_performance(fw_info),
                    documentation_quality=self._estimate_docs(fw_info),
                )
                suitable_frameworks.append(framework)
                algorithms_found.update(framework.algorithms)
        
        suitable_frameworks.sort(key=lambda x: x.compatibility * x.performance_score, reverse=True)
        
        best_practices = self._extract_best_practices(suitable_frameworks)
        
        await self.event_bus.publish(Event(
            source=self.id,
            type=EventType.DECISION_PROPOSED,
            data={
                "action": "rl_research_completed",
                "frameworks_found": len(suitable_frameworks),
                "algorithms": list(algorithms_found),
            },
        ))
        
        return {
            "suitable_frameworks": [
                {
                    "name": fw.name,
                    "repo_url": fw.repo_url,
                    "type": fw.type,
                    "algorithms": fw.algorithms,
                    "features": fw.features,
                    "compatibility": fw.compatibility,
                    "performance_score": fw.performance_score,
                }
                for fw in suitable_frameworks
            ],
            "algorithms_found": list(algorithms_found),
            "best_practices": best_practices,
            "integration_difficulty": self._assess_integration_difficulty(suitable_frameworks),
            "recommendations": self._generate_recommendations(suitable_frameworks, focus_algorithms),
        }
    
    async def _compare_rl_frameworks(self, content: Dict[str, Any]) -> Dict[str, Any]:
        frameworks_to_compare = content.get("frameworks", list(self.rl_frameworks.keys()))
        
        comparisons = []
        for fw_key in frameworks_to_compare:
            if fw_key in self.rl_frameworks:
                fw_info = self.rl_frameworks[fw_key]
                comparisons.append({
                    "name": fw_info["name"],
                    "type": fw_info["type"],
                    "algorithm_count": len(fw_info["algorithms"]),
                    "features": fw_info["features"],
                    "complexity": "high" if fw_info["type"] == "full_platform" else "medium",
                    "learning_curve": "steep" if fw_info["type"] == "full_platform" else "moderate",
                })
        
        return {
            "frameworks": comparisons,
            "best_for_beginners": "stable_baselines3",
            "best_for_research": "rllib",
            "best_for_performance": "tianshou",
            "best_for_modern_stack": "skrl",
        }
    
    async def _recommend_rl_setup(self, content: Dict[str, Any]) -> Dict[str, Any]:
        use_case = content.get("use_case", "game_bot")
        constraints = content.get("constraints", {})
        
        recommendations = {
            "game_bot": {
                "primary": "stable_baselines3",
                "secondary": "rllib",
                "algorithm": "PPO",
                "reasoning": "PPO is robust for game bots with discrete/continuous action spaces",
                "alternative": "SAC for continuous control tasks",
            },
            "high_performance": {
                "primary": "tianshou",
                "secondary": "rllib",
                "algorithm": "PPO",
                "reasoning": "Tianshou offers high performance with parallel training",
            },
            "multi_agent": {
                "primary": "rllib",
                "secondary": "skrl",
                "algorithm": "PPO",
                "reasoning": "RLlib has native multi-agent support",
            },
        }
        
        rec = recommendations.get(use_case, recommendations["game_bot"])
        
        return {
            "recommendation": rec,
            "setup_instructions": self._generate_setup_instructions(rec),
            "expected_performance": self._estimate_expected_performance(rec),
        }
    
    def _estimate_performance(self, fw_info: Dict) -> float:
        if fw_info["type"] == "high_performance":
            return 0.9
        elif fw_info["type"] == "full_platform":
            return 0.85
        elif fw_info["type"] == "beginner_friendly":
            return 0.75
        return 0.7
    
    def _estimate_docs(self, fw_info: Dict) -> float:
        if fw_info["type"] == "beginner_friendly":
            return 0.95
        elif fw_info["type"] == "full_platform":
            return 0.9
        return 0.7
    
    def _extract_best_practices(self, frameworks: List[RLFramework]) -> List[str]:
        practices = [
            "Use replay buffers for experience storage",
            "Implement reward shaping carefully",
            "Normalize observations when possible",
            "Use gradient clipping to prevent exploding gradients",
            "Implement early stopping based on evaluation metrics",
            "Use separate environments for training and evaluation",
            "Log both training and evaluation metrics",
            "Implement model checkpoints for resumable training",
        ]
        
        if any(f.type == "high_performance" for f in frameworks):
            practices.append("Use vectorized environments for parallel training")
        
        if any("multi_agent" in f.features for f in frameworks):
            practices.append("Consider centralized vs decentralized training for multi-agent")
        
        return practices
    
    def _assess_integration_difficulty(self, frameworks: List[RLFramework]) -> str:
        if not frameworks:
            return "unknown"
        
        has_beginner = any(f.type == "beginner_friendly" for f in frameworks)
        has_full_platform = any(f.type == "full_platform" for f in frameworks)
        
        if has_beginner and not has_full_platform:
            return "easy"
        elif has_full_platform:
            return "moderate"
        return "moderate"
    
    def _generate_recommendations(self, frameworks: List[RLFramework], 
                                 target_algorithms: List[str]) -> List[str]:
        recommendations = []
        
        if not frameworks:
            recommendations.append("No direct framework match - consider building custom implementation")
            return recommendations
        
        best = frameworks[0]
        recommendations.append(
            f"Primary recommendation: {best.name} - {best.compatibility * 100:.0f}% compatible"
        )
        
        if len(frameworks) > 1:
            second = frameworks[1]
            recommendations.append(
                f"Secondary option: {second.name} for different algorithm needs"
            )
        
        for algo in target_algorithms:
            if algo not in best.algorithms:
                recommendations.append(f"{algo} not in {best.name} - use custom implementation")
        
        return recommendations
    
    def _generate_setup_instructions(self, recommendation: Dict) -> List[str]:
        return [
            f"Install: pip install {recommendation['primary']}",
            "Define observation and action spaces",
            "Create custom environment inheriting from gymnasium.Env",
            "Initialize agent with appropriate hyperparameters",
            "Implement training loop with periodic evaluation",
            "Use model.save() and CustomSaveCallback for checkpoints",
        ]
    
    def _estimate_expected_performance(self, recommendation: Dict) -> Dict[str, str]:
        return {
            "training_speed": "moderate" if recommendation["primary"] == "stable_baselines3" else "fast",
            "sample_efficiency": "good",
            "convergence_stability": "excellent for PPO",
            "final_performance": "state_of_art for game tasks",
        }
