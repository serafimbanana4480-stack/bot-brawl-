"""Learning Agent - Reinforcement learning, imitation learning and continuous improvement"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np

from .base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType, ConfidenceScore
from ..orchestration.event_bus import EventBus, EventType


@dataclass
class Experience:
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    done: bool
    timestamp: float


@dataclass
class LearningUpdate:
    agent_id: str
    update_type: str
    data: Dict[str, Any]
    timestamp: float
    metrics: Dict[str, float]


class LearningAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        
        self.replay_buffer: deque = deque(maxlen=10000)
        self.learning_history: List[LearningUpdate] = []
        self.current_model_version: int = 0
        self.training_iteration: int = 0
        
        self.learning_config = {
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "learning_rate": 0.001,
            "batch_size": 32,
            "target_update_freq": 100,
        }
        
        self.performance_metrics = {
            "average_reward": 0.0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "training_steps": 0,
        }
        
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        action = message.content.get("action", "learn")
        
        try:
            if action == "learn":
                result = await self._learn_from_experience(message.content)
            elif action == "update":
                result = await self._update_model(message.content)
            elif action == "evaluate":
                result = await self._evaluate_policy(message.content)
            elif action == "improve":
                result = await self._self_improve(message.content)
            elif action == "get_status":
                result = await self._get_learning_status()
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
        recent_updates = context.get("recent_learning_updates", [])
        current_performance = context.get("performance", {})
        
        performance_trend = self._analyze_performance_trend(recent_updates)
        
        improvement_suggestions = []
        if performance_trend < -0.1:
            improvement_suggestions.append("Performance declining - consider policy rollback")
        elif performance_trend > 0.1:
            improvement_suggestions.append("Strong improvement - safe to increase exploration")
        
        return {
            "current_iteration": self.training_iteration,
            "model_version": self.current_model_version,
            "performance_trend": performance_trend,
            "suggestions": improvement_suggestions,
            "confidence": 0.85,
        }
    
    async def _learn_from_experience(self, content: Dict[str, Any]) -> Dict[str, Any]:
        experience_data = content.get("experience")
        
        if experience_data:
            experience = Experience(
                state=experience_data.get("state", {}),
                action=experience_data.get("action", ""),
                reward=experience_data.get("reward", 0.0),
                next_state=experience_data.get("next_state", {}),
                done=experience_data.get("done", False),
                timestamp=time.time(),
            )
            self.replay_buffer.append(experience)
        
        if len(self.replay_buffer) >= self.learning_config["batch_size"]:
            update = await self._perform_gradient_update()
            self.learning_history.append(update)
        
        self._update_epsilon()
        
        return {
            "buffer_size": len(self.replay_buffer),
            "training_iteration": self.training_iteration,
            "epsilon": self.learning_config["epsilon"],
            "metrics": self.performance_metrics,
        }
    
    async def _update_model(self, content: Dict[str, Any]) -> Dict[str, Any]:
        update_type = content.get("type", "incremental")
        training_data = content.get("training_data", [])
        
        if update_type == "incremental":
            result = await self._incremental_update(training_data)
        elif update_type == "full":
            result = await self._full_retrain(training_data)
        elif update_type == "transfer":
            result = await self._transfer_learning(content)
        else:
            return {"error": f"Unknown update type: {update_type}"}
        
        self.current_model_version += 1
        
        learning_update = LearningUpdate(
            agent_id=self.id,
            update_type=update_type,
            data=result,
            timestamp=time.time(),
            metrics=self.performance_metrics.copy(),
        )
        self.learning_history.append(learning_update)
        
        await self.event_bus.publish(Event(
            source=self.id,
            type=EventType.LEARNING_UPDATE,
            data={
                "model_version": self.current_model_version,
                "update_type": update_type,
                "metrics": self.performance_metrics,
            },
        ))
        
        return result
    
    async def _evaluate_policy(self, content: Dict[str, Any]) -> Dict[str, Any]:
        evaluation_episodes = content.get("episodes", 10)
        
        episode_results = []
        for _ in range(evaluation_episodes):
            episode_result = await self._run_evaluation_episode()
            episode_results.append(episode_result)
        
        avg_reward = np.mean([r["total_reward"] for r in episode_results])
        win_rate = np.mean([1 if r["won"] else 0 for r in episode_results])
        
        return {
            "evaluation_results": episode_results,
            "average_reward": avg_reward,
            "win_rate": win_rate,
            "model_version": self.current_model_version,
        }
    
    async def _self_improve(self, content: Dict[str, Any]) -> Dict[str, Any]:
        target_metrics = content.get("target_metrics", {})
        current_metrics = self.performance_metrics.copy()
        
        improvement_areas = []
        for metric, target_value in target_metrics.items():
            current_value = current_metrics.get(metric, 0)
            if current_value < target_value:
                improvement_areas.append({
                    "metric": metric,
                    "current": current_value,
                    "target": target_value,
                    "gap": target_value - current_value,
                })
        
        if not improvement_areas:
            return {
                "status": "target_met",
                "message": "All target metrics achieved",
                "current_metrics": current_metrics,
            }
        
        selected_area = improvement_areas[0]
        
        training_strategy = self._determine_training_strategy(selected_area)
        
        return {
            "status": "improving",
            "improvement_areas": improvement_areas,
            "selected_focus": selected_area,
            "training_strategy": training_strategy,
            "estimated_iterations": self._estimate_improvement_time(selected_area),
        }
    
    async def _get_learning_status(self) -> Dict[str, Any]:
        return {
            "model_version": self.current_model_version,
            "training_iteration": self.training_iteration,
            "replay_buffer_size": len(self.replay_buffer),
            "learning_history_size": len(self.learning_history),
            "performance_metrics": self.performance_metrics,
            "learning_config": self.learning_config,
            "epsilon": self.learning_config["epsilon"],
        }
    
    async def _perform_gradient_update(self) -> LearningUpdate:
        batch = self._sample_batch()
        
        td_errors = []
        for experience in batch:
            td_error = await self._calculate_td_error(experience)
            td_errors.append(td_error)
        
        gradients = self._compute_gradients(td_errors)
        
        self.training_iteration += 1
        
        self.performance_metrics["training_steps"] = self.training_iteration
        
        return LearningUpdate(
            agent_id=self.id,
            update_type="gradient_update",
            data={"batch_size": len(batch)},
            timestamp=time.time(),
            metrics=self.performance_metrics.copy(),
        )
    
    async def _incremental_update(self, training_data: List[Dict]) -> Dict[str, Any]:
        for data in training_data:
            experience = Experience(
                state=data.get("state", {}),
                action=data.get("action", ""),
                reward=data.get("reward", 0.0),
                next_state=data.get("next_state", {}),
                done=data.get("done", False),
                timestamp=time.time(),
            )
            self.replay_buffer.append(experience)
        
        if len(self.replay_buffer) >= self.learning_config["batch_size"]:
            await self._perform_gradient_update()
        
        return {
            "experiences_added": len(training_data),
            "total_experiences": len(self.replay_buffer),
            "training_iterations": self.training_iteration,
        }
    
    async def _full_retrain(self, training_data: List[Dict]) -> Dict[str, Any]:
        self.replay_buffer.clear()
        
        for data in training_data:
            experience = Experience(
                state=data.get("state", {}),
                action=data.get("action", ""),
                reward=data.get("reward", 0.0),
                next_state=data.get("next_state", {}),
                done=data.get("done", False),
                timestamp=time.time(),
            )
            self.replay_buffer.append(experience)
        
        iterations = len(training_data) // self.learning_config["batch_size"]
        
        for _ in range(min(iterations, 100)):
            await self._perform_gradient_update()
        
        return {
            "total_experiences": len(self.replay_buffer),
            "training_iterations": self.training_iteration,
            "model_version": self.current_model_version + 1,
        }
    
    async def _transfer_learning(self, content: Dict[str, Any]) -> Dict[str, Any]:
        source_model = content.get("source_model")
        
        if not source_model:
            return {"error": "No source model specified"}
        
        transferred_knowledge = await self._extract_knowledge_from_model(source_model)
        
        self._apply_transferred_knowledge(transferred_knowledge)
        
        return {
            "status": "transfer_complete",
            "knowledge_transferred": transferred_knowledge,
            "new_model_version": self.current_model_version + 1,
        }
    
    async def _run_evaluation_episode(self) -> Dict[str, Any]:
        total_reward = 0.0
        steps = 0
        won = False
        
        return {
            "total_reward": total_reward,
            "steps": steps,
            "won": won,
        }
    
    def _sample_batch(self) -> List[Experience]:
        batch_size = self.learning_config["batch_size"]
        return list(np.random.choice(list(self.replay_buffer), size=min(batch_size, len(self.replay_buffer)), replace=False))
    
    async def _calculate_td_error(self, experience: Experience) -> float:
        reward = experience.reward
        gamma = self.learning_config["gamma"]
        
        td_error = reward
        if not experience.done:
            td_error += gamma * 0.5 - 0.5
        
        return td_error
    
    def _compute_gradients(self, td_errors: List[float]) -> Dict[str, Any]:
        return {"gradient_magnitude": np.mean(np.abs(td_errors))}
    
    def _update_epsilon(self):
        if self.learning_config["epsilon"] > self.learning_config["epsilon_min"]:
            self.learning_config["epsilon"] *= self.learning_config["epsilon_decay"]
    
    def _analyze_performance_trend(self, recent_updates: List[Dict]) -> float:
        if len(recent_updates) < 2:
            return 0.0
        
        recent_rewards = [u.get("metrics", {}).get("average_reward", 0) for u in recent_updates[-5:]]
        if len(recent_rewards) < 2:
            return 0.0
        
        return (recent_rewards[-1] - recent_rewards[0]) / recent_rewards[0] if recent_rewards[0] != 0 else 0.0
    
    def _determine_training_strategy(self, improvement_area: Dict[str, Any]) -> str:
        metric = improvement_area.get("metric", "")
        
        if "win_rate" in metric or "kda" in metric:
            return "focus_on_combat"
        elif "efficiency" in metric or "damage" in metric:
            return "optimize_resource_usage"
        else:
            return "balanced_training"
    
    def _estimate_improvement_time(self, improvement_area: Dict[str, Any]) -> int:
        gap = improvement_area.get("gap", 0)
        return int(gap * 100)
    
    async def _extract_knowledge_from_model(self, model_id: str) -> Dict[str, Any]:
        return {
            "policies": [],
            "value_estimates": [],
        }
    
    def _apply_transferred_knowledge(self, knowledge: Dict[str, Any]):
        pass
