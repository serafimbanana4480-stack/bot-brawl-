"""
Experience Buffer for RL Training

Thread-safe circular buffer with numpy-backed storage.
"""

import threading
from collections import deque
from typing import Dict, Optional

import numpy as np


class ExperienceBuffer:
    """Thread-safe circular experience buffer."""

    def __init__(self, capacity: int = 10000, min_episode_length: int = 10):
        self.capacity = capacity
        self.min_episode_length = min_episode_length
        self._lock = threading.Lock()
        self._size = 0
        self._episode_start = 0
        self._buf: deque = deque(maxlen=capacity)

    def add(self, state=None, action=None, reward=None, next_state=None, done=None,
            value=0.0, log_prob=0.0, grid=None, next_grid=None):
        """Add transition (thread-safe). Supports duck-typed Experience object."""
        if state is not None and hasattr(state, "state_vector"):
            obj = state
            state = getattr(obj, "state_vector", None)
            action = getattr(obj, "action_idx", getattr(obj, "action", None))
            reward = getattr(obj, "reward", None)
            next_state = getattr(obj, "next_state_vector", getattr(obj, "next_state", None))
            done = getattr(obj, "done", None)
            value = getattr(obj, "value", 0.0)
            log_prob = getattr(obj, "log_prob", 0.0)
            grid = getattr(obj, "grid", None)
            next_grid = getattr(obj, "next_grid", None)
        with self._lock:
            self._buf.append((
                np.asarray(state, np.float32),
                int(action or 0),
                float(reward or 0.0),
                np.asarray(next_state, np.float32),
                bool(done or False),
                float(value),
                float(log_prob),
                np.asarray(grid, np.float32) if grid is not None else None,
                np.asarray(next_grid, np.float32) if next_grid is not None else None,
            ))
            self._size = len(self._buf)

    def start_episode(self):
        with self._lock:
            self._episode_start = self._size

    def end_episode(self):
        with self._lock:
            ep_len = self._size - self._episode_start
            if 0 < ep_len < self.min_episode_length:
                for _ in range(ep_len):
                    self._buf.pop()
                self._size = len(self._buf)
            self._episode_start = self._size

    def sample(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        with self._lock:
            if self._size < batch_size:
                return None
            idx = np.random.choice(self._size, batch_size, replace=False)
            batch = [self._buf[i] for i in idx]
            out = {
                "states": np.stack([b[0] for b in batch]),
                "actions": np.array([b[1] for b in batch], dtype=np.int64),
                "rewards": np.array([b[2] for b in batch], dtype=np.float32),
                "next_states": np.stack([b[3] for b in batch]),
                "dones": np.array([b[4] for b in batch], dtype=np.float32),
                "values": np.array([b[5] for b in batch], dtype=np.float32),
                "old_log_probs": np.array([b[6] for b in batch], dtype=np.float32),
            }
            grids = [b[7] for b in batch]
            if any(g is not None for g in grids):
                ref = next(g for g in grids if g is not None)
                out["grids"] = np.stack([g if g is not None else np.zeros_like(ref) for g in grids])
                ngrids = [b[8] for b in batch]
                out["next_grids"] = np.stack([g if g is not None else np.zeros_like(ref) for g in ngrids])
            return out

    def clear(self):
        with self._lock:
            self._buf.clear()
            self._size = 0
            self._episode_start = 0

    def __len__(self):
        with self._lock:
            return self._size
