"""
neural/online_learning_pipeline.py

Pipeline de aprendizado online (Online Learning) com CQL e BC.

Usa o enriched dataset (data/enriched/) para treinamento contínuo:
- Behavioral Cloning (BC): imita decisões do bot quando ganha
- Conservative Q-Learning (CQL): aprende Q-values sem overfitting
- Experience Replay: buffer de experiências recentes

O pipeline roda em background, processando batches do dataset
enriquecido e atualizando a Q-table / política neural.

Uso:
    pipeline = OnlineLearningPipeline()
    pipeline.load_enriched_dataset("data/enriched/")
    pipeline.train_epoch(batch_size=64, epochs=1)
    pipeline.save_checkpoint("models/rl_online.pt")
"""

import json
import logging
import time
import random
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Uma experiência para replay."""
    state: Tuple[int, ...]
    action: int
    reward: float
    next_state: Tuple[int, ...]
    done: bool
    priority: float = 1.0


class PrioritizedReplayBuffer:
    """Buffer de experiências com sampling prioritizado."""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)

    def add(self, exp: Experience):
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(exp)
        self.priorities.append(max_prio)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray]:
        if len(self.buffer) == 0:
            return [], np.array([])

        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, weights

    def update_priorities(self, indices: List[int], td_errors: List[float]):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6

    def __len__(self):
        return len(self.buffer)


class ConservativeQLearner:
    """
    Conservative Q-Learning (CQL) simplificado.

    Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
    CQL penalty: subtrai log-sum-exp de Q(s,a') para evitar overestimation
    """

    def __init__(
        self,
        state_size: int = 216,
        action_size: int = 6,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        cql_alpha: float = 0.1,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.cql_alpha = cql_alpha

        # Q-table como dict: state_tuple -> numpy array de Q-values
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    def _get_q(self, state: Tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[state]

    def update(self, exp: Experience, weight: float = 1.0):
        """Atualiza Q-value com CQL penalty."""
        q_s = self._get_q(exp.state)
        q_next = self._get_q(exp.next_state)

        # Target Q
        target = exp.reward + (0 if exp.done else self.gamma * q_next.max())

        # TD error
        td_error = target - q_s[exp.action]

        # CQL penalty: penaliza Q-values altos para ações não tomadas
        q_values = q_s.copy()
        cql_penalty = self.cql_alpha * (np.log(np.sum(np.exp(q_values))) - q_values[exp.action])

        # Update
        q_s[exp.action] += self.lr * weight * td_error - cql_penalty

        return td_error

    def get_action(self, state: Tuple[int, ...], epsilon: float = 0.1) -> int:
        q = self._get_q(state)
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        return int(q.argmax())

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({
                "q_table": self.q_table,
                "state_size": self.state_size,
                "action_size": self.action_size,
            }, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.q_table = data.get("q_table", {})


class BehavioralCloning:
    """
    Behavioral Cloning: imita decisões de partidas vencedoras.

    Simples contador de frequência de ações por estado.
    Mais robusto que neural net para espaços pequenos.
    """

    def __init__(self):
        self.action_counts: Dict[Tuple[int, ...], Dict[int, int]] = {}

    def train(self, state: Tuple[int, ...], action: int):
        if state not in self.action_counts:
            self.action_counts[state] = {}
        self.action_counts[state][action] = self.action_counts[state].get(action, 0) + 1

    def predict(self, state: Tuple[int, ...]) -> Optional[int]:
        if state not in self.action_counts:
            return None
        counts = self.action_counts[state]
        return max(counts, key=counts.get)

    def get_confidence(self, state: Tuple[int, ...], action: int) -> float:
        if state not in self.action_counts:
            return 0.0
        counts = self.action_counts[state]
        total = sum(counts.values())
        return counts.get(action, 0) / total if total > 0 else 0.0


class OnlineLearningPipeline:
    """
    Pipeline completo de aprendizado online.

    Processa dados enriquecidos e atualiza CQL + BC.
    """

    ACTION_MAP = {
        "attack": 0, "move_to_enemy": 1, "retreat": 2,
        "use_super": 3, "collect_cube": 4, "idle": 5,
    }

    def __init__(
        self,
        dataset_dir: Path = Path("data/enriched"),
        checkpoint_dir: Path = Path("models/online_checkpoints"),
        buffer_capacity: int = 10000,
        batch_size: int = 64,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        self.cql = ConservativeQLearner()
        self.bc = BehavioralCloning()
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

        self._processed_files: set = set()
        self._total_epochs = 0

    # ------------------------------------------------------------------
    # Carregamento de Dataset
    # ------------------------------------------------------------------

    def load_enriched_dataset(self, max_files: int = 100) -> int:
        """
        Carrega frames de partidas do diretório enriquecido.
        Retorna número de frames carregados.
        """
        jsonl_files = sorted(self.dataset_dir.glob("*_frames.jsonl"))[:max_files]
        total = 0

        for jsonl_path in jsonl_files:
            if str(jsonl_path) in self._processed_files:
                continue
            self._processed_files.add(str(jsonl_path))

            meta_path = jsonl_path.parent / jsonl_path.name.replace("_frames.jsonl", "_meta.json")
            result = "unknown"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    result = meta.get("result", "unknown")
                except Exception:
                    pass

            frames = 0
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        frame = json.loads(line)
                        self._process_frame(frame, result)
                        frames += 1
                    except Exception as e:
                        logger.debug("[ONLINE] Erro ao processar frame: %s", e)

            total += frames
            logger.info("[ONLINE] Carregado %s: %d frames (%s)", jsonl_path.name, frames, result)

        return total

    def _process_frame(self, frame: Dict, match_result: str):
        """Converte um frame enriquecido em experiência."""
        # Estado simplificado (pode ser expandido)
        game_state = frame.get("game_state", "unknown")
        action_str = frame.get("action_taken", "idle")
        action = self.ACTION_MAP.get(action_str, 5)

        # Criar estado discreto (exemplo simplificado)
        # Na prática, o state viria do frame (HP band, enemies, distance, etc.)
        state = self._frame_to_state(frame)
        next_state = state  # Simplificado; idealmente seria o próximo frame

        # Reward baseado no resultado da partida e no frame
        reward = self._compute_reward(frame, match_result)

        # Done se for último frame (simplificado)
        done = False

        exp = Experience(state=state, action=action, reward=reward,
                        next_state=next_state, done=done)
        self.replay_buffer.add(exp)

        # BC apenas de partidas vencedoras
        if match_result in ("victory", "win"):
            self.bc.train(state, action)

    def _frame_to_state(self, frame: Dict) -> Tuple[int, ...]:
        """Extrai estado discreto de um frame."""
        # Placeholder: na prática, extrair HP band, enemy count, etc.
        # do frame enriquecido
        hp = frame.get("player_hp", 1.0)
        hp_band = min(int(hp * 4), 3)
        enemies = len(frame.get("enemies", []))
        enemy_band = min(enemies, 3)
        return (hp_band, enemy_band, 0, 0, 0)

    def _compute_reward(self, frame: Dict, match_result: str) -> float:
        """Computa reward para um frame."""
        reward = 0.0
        if match_result in ("victory", "win"):
            reward += 1.0
        elif match_result in ("loss", "defeat"):
            reward -= 1.0

        # Bonus por ação com scores altos
        scores = frame.get("action_scores", {})
        if scores:
            reward += max(scores.values()) * 0.1

        return reward

    # ------------------------------------------------------------------
    # Treinamento
    # ------------------------------------------------------------------

    def train_epoch(self, epochs: int = 1) -> Dict[str, float]:
        """Roda um epoch de treinamento."""
        if len(self.replay_buffer) < self.batch_size:
            logger.warning("[ONLINE] Buffer insuficiente: %d < %d", len(self.replay_buffer), self.batch_size)
            return {"status": "insufficient_data"}

        total_td = 0.0
        total_updates = 0

        for _ in range(epochs):
            samples, weights = self.replay_buffer.sample(self.batch_size)
            td_errors = []

            for exp, w in zip(samples, weights):
                td = self.cql.update(exp, weight=w)
                td_errors.append(td)
                total_td += abs(td)
                total_updates += 1

            # Atualizar prioridades
            # (simplificado: não temos os índices reais aqui)

        self._total_epochs += epochs

        avg_td = total_td / total_updates if total_updates > 0 else 0
        logger.info("[ONLINE] Epoch %d: %d updates, avg TD error=%.4f",
                    self._total_epochs, total_updates, avg_td)

        return {
            "epochs": self._total_epochs,
            "updates": total_updates,
            "avg_td_error": round(avg_td, 4),
            "buffer_size": len(self.replay_buffer),
            "q_table_size": len(self.cql.q_table),
        }

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save_checkpoint(self, name: Optional[str] = None):
        """Salva checkpoint do modelo."""
        if name is None:
            name = f"online_cql_epoch{self._total_epochs}.pt"
        path = self.checkpoint_dir / name
        self.cql.save(path)

        # Salvar BC também
        bc_path = path.with_suffix(".bc.pkl")
        with open(bc_path, "wb") as f:
            pickle.dump(self.bc.action_counts, f)

        logger.info("[ONLINE] Checkpoint salvo: %s", path)
        return path

    def load_checkpoint(self, path: Path):
        """Carrega checkpoint."""
        self.cql.load(path)
        bc_path = path.with_suffix(".bc.pkl")
        if bc_path.exists():
            with open(bc_path, "rb") as f:
                self.bc.action_counts = pickle.load(f)
        logger.info("[ONLINE] Checkpoint carregado: %s", path)

    # ------------------------------------------------------------------
    # Inferência
    # ------------------------------------------------------------------

    def predict_action(self, state: Tuple[int, ...], epsilon: float = 0.1) -> Tuple[int, Dict]:
        """Prediz ação combinando CQL + BC."""
        # BC tem prioridade se confiança alta
        bc_action = self.bc.predict(state)
        if bc_action is not None:
            confidence = self.bc.get_confidence(state, bc_action)
            if confidence > 0.7:
                return bc_action, {"source": "bc", "confidence": confidence}

        # Fallback para CQL
        cql_action = self.cql.get_action(state, epsilon=epsilon)
        return cql_action, {"source": "cql", "epsilon": epsilon}

    def get_status(self) -> Dict[str, Any]:
        return {
            "epochs_trained": self._total_epochs,
            "buffer_size": len(self.replay_buffer),
            "q_table_size": len(self.cql.q_table),
            "bc_states": len(self.bc.action_counts),
            "processed_files": len(self._processed_files),
        }
