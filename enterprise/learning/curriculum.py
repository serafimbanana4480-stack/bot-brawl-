"""Curriculum Learning - Progressive difficulty training"""

import numpy as np
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass


@dataclass
class DifficultyLevel:
    level: int
    name: str
    parameters: Dict[str, Any]
    success_threshold: float
    min_episodes: int


class CurriculumLearning:
    def __init__(self, task_fn: Callable):
        self.task_fn = task_fn
        self.current_level = 0
        self.difficulty_levels: List[DifficultyLevel] = []
        self.performance_history: Dict[int, List[float]] = {}
        self.episode_count = 0
        
        self._initialize_curriculum()
    
    def _initialize_curriculum(self):
        self.difficulty_levels = [
            DifficultyLevel(
                level=0,
                name="Easy",
                parameters={
                    "enemy_count": 1,
                    "enemy_aggression": 0.3,
                    "map_complexity": 0.2,
                    "time_limit": 180,
                },
                success_threshold=0.4,
                min_episodes=50,
            ),
            DifficultyLevel(
                level=1,
                name="Medium",
                parameters={
                    "enemy_count": 2,
                    "enemy_aggression": 0.5,
                    "map_complexity": 0.4,
                    "time_limit": 150,
                },
                success_threshold=0.5,
                min_episodes=100,
            ),
            DifficultyLevel(
                level=2,
                name="Hard",
                parameters={
                    "enemy_count": 3,
                    "enemy_aggression": 0.7,
                    "map_complexity": 0.6,
                    "time_limit": 120,
                },
                success_threshold=0.55,
                min_episodes=150,
            ),
            DifficultyLevel(
                level=3,
                name="Expert",
                parameters={
                    "enemy_count": 4,
                    "enemy_aggression": 0.85,
                    "map_complexity": 0.8,
                    "time_limit": 90,
                },
                success_threshold=0.6,
                min_episodes=200,
            ),
            DifficultyLevel(
                level=4,
                name="Master",
                parameters={
                    "enemy_count": 5,
                    "enemy_aggression": 1.0,
                    "map_complexity": 1.0,
                    "time_limit": 60,
                },
                success_threshold=0.65,
                min_episodes=300,
            ),
        ]
    
    def get_current_task(self) -> Dict[str, Any]:
        if self.current_level >= len(self.difficulty_levels):
            return self.difficulty_levels[-1].parameters
        
        return self.difficulty_levels[self.current_level].parameters
    
    def update_progress(self, performance: float):
        self.episode_count += 1
        
        if self.current_level not in self.performance_history:
            self.performance_history[self.current_level] = []
        
        self.performance_history[self.current_level].append(performance)
        
        current_difficulty = self.difficulty_levels[self.current_level]
        
        recent_performances = self.performance_history[self.current_level][-current_difficulty.min_episodes:]
        
        if len(recent_performances) >= current_difficulty.min_episodes:
            avg_performance = sum(recent_performances) / len(recent_performances)
            
            if avg_performance >= current_difficulty.success_threshold:
                if self.current_level < len(self.difficulty_levels) - 1:
                    self.current_level += 1
                    print(f"Advancing to level {self.current_level}: {self.difficulty_levels[self.current_level].name}")
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "current_level": self.current_level,
            "current_difficulty": self.difficulty_levels[self.current_level].name,
            "total_episodes": self.episode_count,
            "levels_completed": self.current_level,
            "performance_by_level": {},
        }
        
        for level, performances in self.performance_history.items():
            if performances:
                stats["performance_by_level"][level] = {
                    "mean": float(np.mean(performances)),
                    "std": float(np.std(performances)),
                    "max": float(np.max(performances)),
                    "min": float(np.min(performances)),
                    "episodes": len(performances),
                }
        
        return stats
    
    def should_reset(self) -> bool:
        if self.current_level > 0:
            recent = self.performance_history.get(self.current_level, [])[-20:]
            if len(recent) >= 10 and np.mean(recent) < 0.2:
                return True
        return False
    
    def regress_level(self):
        if self.current_level > 0:
            self.current_level -= 1
            print(f"Regressing to level {self.current_level}: {self.difficulty_levels[self.current_level].name}")
