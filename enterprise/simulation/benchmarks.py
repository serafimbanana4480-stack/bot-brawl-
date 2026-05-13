"""Simulation Environment and Benchmark Suite"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
import random


@dataclass
class GameState:
    player_health: float
    player_position: Tuple[float, float]
    player_velocity: Tuple[float, float]
    enemies: List[Dict[str, Any]]
    objectives: List[Dict[str, Any]]
    powerups: List[Dict[str, Any]]
    timestamp: float


class SimulationEnvironment:
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        self.map_size = config.get("map_size", (1000, 1000))
        self.max_steps = config.get("max_steps", 1000)
        self.step_count = 0
        
        self.state = self._reset_state()
        
    def _reset_state(self) -> GameState:
        return GameState(
            player_health=100.0,
            player_position=(500.0, 500.0),
            player_velocity=(0.0, 0.0),
            enemies=[
                {"id": 0, "position": (200.0, 200.0), "health": 50.0, "velocity": (0.0, 0.0)},
                {"id": 1, "position": (800.0, 200.0), "health": 50.0, "velocity": (0.0, 0.0)},
            ],
            objectives=[
                {"id": 0, "position": (500.0, 500.0), "captured": False},
            ],
            powerups=[
                {"id": 0, "position": (300.0, 300.0), "type": "health", "collected": False},
            ],
            timestamp=0.0,
        )
    
    def reset(self) -> GameState:
        self.step_count = 0
        self.state = self._reset_state()
        return self.state
    
    def step(self, action: Dict[str, Any]) -> Tuple[GameState, float, bool]:
        self.step_count += 1
        
        action_type = action.get("type", "move")
        
        if action_type == "move":
            dx = action.get("dx", 0)
            dy = action.get("dy", 0)
            new_x = max(0, min(self.map_size[0], self.state.player_position[0] + dx))
            new_y = max(0, min(self.map_size[1], self.state.player_position[1] + dy))
            self.state.player_position = (new_x, new_y)
            
        elif action_type == "attack":
            target_id = action.get("target_id")
            for enemy in self.state.enemies:
                if enemy["id"] == target_id:
                    damage = action.get("damage", 25)
                    enemy["health"] = max(0, enemy["health"] - damage)
        
        reward = self._calculate_reward()
        done = self.step_count >= self.max_steps or self.state.player_health <= 0
        
        for enemy in self.state.enemies:
            self._update_enemy_behavior(enemy)
        
        self._check_collisions()
        self.state.timestamp += 1.0
        
        return self.state, reward, done
    
    def _calculate_reward(self) -> float:
        reward = -0.01
        
        for enemy in self.state.enemies:
            if enemy["health"] <= 0:
                reward += 10.0
        
        for powerup in self.state.powerups:
            if not powerup["collected"]:
                dist = self._distance(self.state.player_position, powerup["position"])
                if dist < 30:
                    reward += 5.0
        
        if self.state.player_health < 30:
            reward -= 5.0
        
        return reward
    
    def _update_enemy_behavior(self, enemy: Dict[str, Any]):
        dx = self.state.player_position[0] - enemy["position"][0]
        dy = self.state.player_position[1] - enemy["position"][1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 50:
            speed = 2.0
            enemy["position"] = (
                enemy["position"][0] + (dx / dist) * speed,
                enemy["position"][1] + (dy / dist) * speed,
            )
    
    def _check_collisions(self):
        for enemy in self.state.enemies:
            if enemy["health"] > 0:
                dist = self._distance(self.state.player_position, enemy["position"])
                if dist < 40:
                    self.state.player_health -= 5.0
        
        for powerup in self.state.powerups:
            if not powerup["collected"]:
                dist = self._distance(self.state.player_position, powerup["position"])
                if dist < 30:
                    powerup["collected"] = True
                    if powerup["type"] == "health":
                        self.state.player_health = min(100, self.state.player_health + 25)
    
    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


class BenchmarkSuite:
    def __init__(self):
        self.benchmarks = {}
        
    def register_benchmark(self, name: str, fn: Callable):
        self.benchmarks[name] = fn
    
    def run_benchmark(self, name: str, num_episodes: int = 100) -> Dict[str, float]:
        if name not in self.benchmarks:
            return {"error": "Benchmark not found"}
        
        results = []
        for _ in range(num_episodes):
            result = self.benchmarks[name]()
            results.append(result)
        
        return self._aggregate_results(results)
    
    def run_all(self, num_episodes: int = 100) -> Dict[str, Dict[str, float]]:
        return {
            name: self.run_benchmark(name, num_episodes)
            for name in self.benchmarks.keys()
        }
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        
        aggregated = {}
        for key in all_keys:
            values = [r.get(key, 0) for r in results if key in r]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)
        
        return aggregated


def accuracy_benchmark(agent_fn: Callable) -> Dict[str, float]:
    correct = 0
    total = 100
    
    for _ in range(total):
        state = np.random.rand(10)
        expected_action = int(np.argmax(state))
        predicted_action = agent_fn(state)
        
        if predicted_action == expected_action:
            correct += 1
    
    return {"accuracy": correct / total}


def reaction_time_benchmark(agent_fn: Callable) -> Dict[str, float]:
    import time
    
    reaction_times = []
    
    for _ in range(50):
        start = time.time()
        agent_fn(np.random.rand(10))
        reaction_times.append((time.time() - start) * 1000)
    
    return {
        "reaction_time_mean_ms": np.mean(reaction_times),
        "reaction_time_std_ms": np.std(reaction_times),
    }


def strategic_quality_benchmark(env: SimulationEnvironment, 
                               agent_fn: Callable) -> Dict[str, float]:
    wins = 0
    total = 20
    
    for _ in range(total):
        state = env.reset()
        done = False
        
        while not done:
            action = agent_fn(state)
            state, reward, done = env.step(action)
        
        if any(e["health"] <= 0 for e in state.enemies):
            wins += 1
    
    return {"win_rate": wins / total}
