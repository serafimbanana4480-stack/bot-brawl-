"""
core/learning_rl_metrics.py

Métricas unificadas de RL (Q-Learning / PPO) para a dashboard.
Consome dados de rl_engine.py e neural/rl_bridge.py e expõe
dict serializável para a dashboard web.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RLMetricsSnapshot:
    timestamp: float = field(default_factory=time.time)
    engine_type: str = "q_learning"  # ou "ppo"
    active: bool = False

    # Q-Learning
    q_table_size: int = 0
    epsilon: float = 0.0
    total_updates: int = 0
    visits_avg: float = 0.0

    # PPO
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    buffer_size: int = 0
    buffer_capacity: int = 0

    # Rewards
    last_reward: float = 0.0
    episode_reward: float = 0.0
    total_reward: float = 0.0
    reward_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # Actions
    last_action: str = ""
    action_counts: Dict[str, int] = field(default_factory=dict)
    exploration_rate: float = 0.0  # % acoes exploratorias

    # Collection
    images_collected: int = 0
    labels_generated: int = 0
    collection_fps: float = 0.0
    last_image_path: str = ""


class LearningRLMetricsCollector:
    """
    Agrega métricas de RL e recolha de dados.
    Chamado periodicamente pelo wrapper para alimentar a dashboard.
    """

    def __init__(self):
        self._snapshot = RLMetricsSnapshot()
        self._lock = False
        self._start_time: Optional[float] = None

    def start_session(self):
        self._start_time = time.time()
        self._snapshot = RLMetricsSnapshot()
        self._snapshot.active = True

    def stop_session(self):
        self._snapshot.active = False

    # ------------------------------------------------------------------
    # Q-Learning updates
    # ------------------------------------------------------------------

    def update_q_learning(self, rl_engine: Any):
        if rl_engine is None:
            return
        self._snapshot.engine_type = "q_learning"
        self._snapshot.q_table_size = len(getattr(rl_engine, 'q_table', {}))
        self._snapshot.epsilon = getattr(rl_engine, 'epsilon', 0.0)
        self._snapshot.total_updates = getattr(rl_engine, 'total_updates', 0)
        vc = getattr(rl_engine, 'visit_counts', {})
        self._snapshot.visits_avg = round(sum(vc.values()) / max(1, len(vc)), 1) if vc else 0.0

    # ------------------------------------------------------------------
    # PPO updates
    # ------------------------------------------------------------------

    def update_ppo(self, rl_bridge: Any):
        if rl_bridge is None:
            return
        self._snapshot.engine_type = "ppo"
        self._snapshot.policy_loss = round(getattr(rl_bridge, '_last_policy_loss', 0.0), 4)
        self._snapshot.value_loss = round(getattr(rl_bridge, '_last_value_loss', 0.0), 4)
        self._snapshot.entropy = round(getattr(rl_bridge, '_last_entropy', 0.0), 4)
        buf = getattr(rl_bridge, 'experience_buffer', None)
        if buf:
            self._snapshot.buffer_size = len(getattr(buf, 'buffer', []))
            self._snapshot.buffer_capacity = getattr(buf, 'capacity', 0)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def record_reward(self, reward: float, action: str = "", is_exploration: bool = False):
        self._snapshot.last_reward = round(reward, 3)
        self._snapshot.episode_reward += reward
        self._snapshot.total_reward += reward
        self._snapshot.reward_history.append({
            "t": time.time(),
            "r": reward,
            "action": action,
        })
        self._snapshot.last_action = action
        self._snapshot.action_counts[action] = self._snapshot.action_counts.get(action, 0) + 1
        if is_exploration:
            self._snapshot.exploration_rate += 1
        # Normalize exploration rate over time window
        total_actions = sum(self._snapshot.action_counts.values())
        if total_actions > 0:
            self._snapshot.exploration_rate = round(self._snapshot.exploration_rate / total_actions * 100, 1)

    def end_episode(self):
        self._snapshot.episode_reward = 0.0

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def update_collection(self, images: int = 0, labels: int = 0, fps: float = 0.0, last_path: str = ""):
        self._snapshot.images_collected = images
        self._snapshot.labels_generated = labels
        self._snapshot.collection_fps = round(fps, 1)
        self._snapshot.last_image_path = last_path

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict:
        s = self._snapshot
        return {
            "active": s.active,
            "engine_type": s.engine_type,
            "session_duration": round(time.time() - self._start_time, 1) if self._start_time else 0.0,
            "q_table_size": s.q_table_size,
            "epsilon": round(s.epsilon, 4),
            "total_updates": s.total_updates,
            "visits_avg": s.visits_avg,
            "policy_loss": s.policy_loss,
            "value_loss": s.value_loss,
            "entropy": s.entropy,
            "buffer_size": s.buffer_size,
            "buffer_capacity": s.buffer_capacity,
            "last_reward": s.last_reward,
            "episode_reward": round(s.episode_reward, 2),
            "total_reward": round(s.total_reward, 2),
            "reward_history": list(s.reward_history)[-30:],
            "last_action": s.last_action,
            "action_counts": s.action_counts,
            "exploration_rate": s.exploration_rate,
            "images_collected": s.images_collected,
            "labels_generated": s.labels_generated,
            "collection_fps": s.collection_fps,
            "last_image_path": s.last_image_path,
        }
