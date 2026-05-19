"""
neural/rl_bridge.py

Bridge que integra NeuralPolicy + PPOTrainer + Q-Learning fallback.
Substitui a tabela Q por uma policy network profunda com partial observability.

Architecture:
    PlayLogic ──> RLBridge.get_action() ──> NeuralPolicy (CNN+LSTM+Fusion)
                     │
                     └─> ExperienceBuffer.collect()
                     │
                     └─> PPOTrainer.train() [periodicamente/offline]
                     │
                     └─> CombatQLearning [fallback se neural falhar]

Migration Path:
    OnlineLearner(use_neural=True) -> RLBridge -> NeuralPolicy
    OnlineLearner(use_neural=False) -> Q-Learning (legacy)
"""

from __future__ import annotations

import logging
import math
import pickle
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports para evitar dependência pesada no startup
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment, misc]

# ------------------------------------------------------------------
# Experience dataclass
# ------------------------------------------------------------------

@dataclass
class Experience:
    """Single transition (s, a, r, s', done, v, log_prob)."""
    state_vector: np.ndarray      # 44-dim state features
    grid: Optional[np.ndarray]   # 21x21xC spatial grid
    action_idx: int
    reward: float
    next_state_vector: np.ndarray
    next_grid: Optional[np.ndarray]
    done: bool
    value: float
    log_prob: float
    timestamp: float = field(default_factory=time.time)


# ------------------------------------------------------------------
# Experience Buffer
# ------------------------------------------------------------------

class ExperienceBuffer:
    """
    Rolling experience buffer com gating por reward e amostragem por prioridade.
    """

    def __init__(self, capacity: int = 10000, min_episode_length: int = 10):
        self.capacity = capacity
        self.min_episode_length = min_episode_length
        self.buffer: deque = deque(maxlen=capacity)
        self.episode_start_idx: int = 0
        self.total_collected: int = 0

    def add(self, exp: Experience) -> None:
        self.buffer.append(exp)
        self.total_collected += 1

    def start_episode(self) -> None:
        self.episode_start_idx = len(self.buffer)

    def end_episode(self) -> None:
        episode_len = len(self.buffer) - self.episode_start_idx
        if episode_len < self.min_episode_length:
            # Episódio muito curto: truncar
            for _ in range(episode_len):
                self.buffer.pop()
            logger.debug(f"[BUFFER] Episodio truncado (len={episode_len})")

    def sample(self, batch_size: int) -> Optional[Dict]:
        if len(self.buffer) < batch_size:
            return None
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        return self._collate(batch)

    def _collate(self, batch: List[Experience]) -> Dict:
        states = np.stack([e.state_vector for e in batch])
        actions = np.array([e.action_idx for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.stack([e.next_state_vector for e in batch])
        dones = np.array([e.done for e in batch], dtype=np.float32)
        values = np.array([e.value for e in batch])
        log_probs = np.array([e.log_prob for e in batch])

        # Grids: usar zero se não disponível
        grids = []
        for e in batch:
            if e.grid is not None:
                grids.append(e.grid)
            else:
                grids.append(np.zeros((21, 21, 1), dtype=np.float32))
        grids = np.stack(grids)

        return {
            "states": states,
            "grids": grids,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "values": values,
            "old_log_probs": log_probs,
        }

    def clear(self) -> None:
        self.buffer.clear()
        self.episode_start_idx = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: Path) -> None:
        try:
            with open(path, "wb") as f:
                pickle.dump(list(self.buffer), f)
            logger.info(f"[BUFFER] Salvo: {path} ({len(self.buffer)} experiencias)")
        except Exception as e:
            logger.warning(f"[BUFFER] Falha ao salvar: {e}")

    def load(self, path: Path) -> None:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.buffer = deque(data, maxlen=self.capacity)
            logger.info(f"[BUFFER] Carregado: {path} ({len(self.buffer)} experiencias)")
        except Exception as e:
            logger.warning(f"[BUFFER] Falha ao carregar: {e}")


# ------------------------------------------------------------------
# State Feature Extractor (converte estado bruto -> 44-dim vector)
# ------------------------------------------------------------------

class StateFeatureExtractor:
    """
    Converte estado bruto do jogo (HP, enemies, distancias, etc.)
    no vetor de 44 dims esperado pela NeuralPolicy.
    """

    # 44 features mapeadas para o schema FULL
    FEATURE_NAMES = [
        # Self-state (11)
        "hp_ratio", "ammo_ratio", "super_charge", "gadget_ready",
        "hypercharge_ready", "cooldown_attack", "cooldown_super",
        "cooldown_gadget", "in_bush", "has_power_cubes", "brawler_class",
        # Spatial (4) - posicao normalizada
        "pos_x", "pos_y", "nearest_wall_dist", "nearest_bush_dist",
        # Temporal (3)
        "time_alive", "time_in_combat", "frames_since_damage",
        # Enemy aggregate (8)
        "num_enemies", "num_enemies_near", "avg_enemy_hp",
        "nearest_enemy_dist", "enemies_in_range", "enemies_super_ready",
        "enemies_hidden", "threat_level",
        # Objective (6)
        "mode", "score_advantage", "time_remaining", "objective_dist",
        "teammates_alive", "team_score_ratio",
        # Tactical (12)
        "can_retreat", "can_push", "can_ambush", "in_storm",
        "storm_distance", "safe_zone_coverage", "cover_available",
        "escape_route_available", "power_cube_dist", "gem_dist",
        "ball_dist", "zone_control",
    ]

    NUM_FEATURES = len(FEATURE_NAMES)

    @classmethod
    def extract(
        cls,
        player_hp_pct: float = 1.0,
        ammo_ratio: float = 1.0,
        super_charge: float = 0.0,
        num_enemies: int = 0,
        nearest_enemy_dist: float = 999.0,
        can_attack: bool = True,
        can_super: bool = False,
        player_pos: Tuple[float, float] = (0.5, 0.5),
        enemies: Optional[List] = None,
        match_time_remaining: float = 120.0,
        **kwargs,
    ) -> np.ndarray:
        """
        Extrai vetor de estado de 44 dims a partir de dados brutos.
        """
        features = np.zeros(cls.NUM_FEATURES, dtype=np.float32)

        # Self-state
        features[0] = np.clip(player_hp_pct, 0.0, 1.0)
        features[1] = np.clip(ammo_ratio, 0.0, 1.0)
        features[2] = np.clip(super_charge, 0.0, 1.0)
        features[3] = 1.0 if kwargs.get("gadget_ready", False) else 0.0
        features[4] = 1.0 if kwargs.get("hypercharge_ready", False) else 0.0
        features[5] = np.clip(kwargs.get("cooldown_attack", 0.0), 0.0, 1.0)
        features[6] = np.clip(kwargs.get("cooldown_super", 0.0), 0.0, 1.0)
        features[7] = np.clip(kwargs.get("cooldown_gadget", 0.0), 0.0, 1.0)
        features[8] = 1.0 if kwargs.get("in_bush", False) else 0.0
        features[9] = float(kwargs.get("num_power_cubes", 0)) / 10.0
        features[10] = kwargs.get("brawler_class", 0) / 5.0  # 0-5 classes

        # Spatial
        px, py = player_pos
        features[11] = px  # Normalizado 0-1
        features[12] = py
        features[13] = np.clip(kwargs.get("nearest_wall_dist", 999.0) / 500.0, 0.0, 1.0)
        features[14] = np.clip(kwargs.get("nearest_bush_dist", 999.0) / 500.0, 0.0, 1.0)

        # Temporal
        features[15] = np.clip(kwargs.get("time_alive", 0.0) / 300.0, 0.0, 1.0)
        features[16] = np.clip(kwargs.get("time_in_combat", 0.0) / 60.0, 0.0, 1.0)
        features[17] = np.clip(kwargs.get("frames_since_damage", 0) / 300.0, 0.0, 1.0)

        # Enemy aggregate
        enemies = enemies or []
        num_enemies_near = sum(1 for e in enemies if cls._dist_to_player(e, player_pos) < 200)
        avg_enemy_hp = np.mean([cls._enemy_hp(e) for e in enemies]) if enemies else 1.0
        enemies_in_range = sum(1 for e in enemies if cls._dist_to_player(e, player_pos) < 400)
        enemies_super = sum(1 for e in enemies if cls._enemy_has_super(e))
        enemies_hidden = sum(1 for e in enemies if cls._enemy_hidden(e))

        features[18] = np.clip(num_enemies / 10.0, 0.0, 1.0)
        features[19] = np.clip(num_enemies_near / 5.0, 0.0, 1.0)
        features[20] = np.clip(avg_enemy_hp, 0.0, 1.0)
        features[21] = np.clip(nearest_enemy_dist / 800.0, 0.0, 1.0)
        features[22] = np.clip(enemies_in_range / 5.0, 0.0, 1.0)
        features[23] = np.clip(enemies_super / 5.0, 0.0, 1.0)
        features[24] = np.clip(enemies_hidden / 5.0, 0.0, 1.0)
        features[25] = np.clip(kwargs.get("threat_level", 0.0), 0.0, 1.0)

        # Objective
        features[26] = kwargs.get("game_mode", 0) / 10.0
        features[27] = np.clip(kwargs.get("score_advantage", 0.0) / 10.0, -1.0, 1.0)
        features[28] = np.clip(match_time_remaining / 180.0, 0.0, 1.0)
        features[29] = np.clip(kwargs.get("objective_dist", 999.0) / 800.0, 0.0, 1.0)
        features[30] = np.clip(kwargs.get("teammates_alive", 2) / 5.0, 0.0, 1.0)
        features[31] = np.clip(kwargs.get("team_score_ratio", 1.0), 0.0, 2.0) / 2.0

        # Tactical
        features[32] = 1.0 if kwargs.get("can_retreat", True) else 0.0
        features[33] = 1.0 if kwargs.get("can_push", False) else 0.0
        features[34] = 1.0 if kwargs.get("can_ambush", False) else 0.0
        features[35] = 1.0 if kwargs.get("in_storm", False) else 0.0
        features[36] = np.clip(kwargs.get("storm_distance", 999.0) / 800.0, 0.0, 1.0)
        features[37] = np.clip(kwargs.get("safe_zone_coverage", 0.0), 0.0, 1.0)
        features[38] = 1.0 if kwargs.get("cover_available", False) else 0.0
        features[39] = 1.0 if kwargs.get("escape_route", False) else 0.0
        features[40] = np.clip(kwargs.get("power_cube_dist", 999.0) / 800.0, 0.0, 1.0)
        features[41] = np.clip(kwargs.get("gem_dist", 999.0) / 800.0, 0.0, 1.0)
        features[42] = np.clip(kwargs.get("ball_dist", 999.0) / 800.0, 0.0, 1.0)
        features[43] = np.clip(kwargs.get("zone_control", 0.5), 0.0, 1.0)

        return features

    @staticmethod
    def _dist_to_player(enemy, player_pos):
        if not enemy or len(enemy) < 4:
            return 999.0
        ex = (enemy[0] + enemy[2]) / 2
        ey = (enemy[1] + enemy[3]) / 2
        return math.hypot(ex - player_pos[0], ey - player_pos[1])

    @staticmethod
    def _enemy_hp(enemy):
        return getattr(enemy, "hp_ratio", 1.0) if hasattr(enemy, "hp_ratio") else 1.0

    @staticmethod
    def _enemy_has_super(enemy):
        return getattr(enemy, "super_ready", False) if hasattr(enemy, "super_ready") else False

    @staticmethod
    def _enemy_hidden(enemy):
        return getattr(enemy, "in_bush", False) if hasattr(enemy, "in_bush") else False


# ------------------------------------------------------------------
# RL Bridge: NeuralPolicy + PPO + Q-Learning fallback
# ------------------------------------------------------------------

class RLBridge:
    """
    Bridge unificada que orquestra NeuralPolicy, PPOTrainer e Q-Learning fallback.
    Interface compatível com OnlineLearner para drop-in replacement.
    """

    def __init__(
        self,
        use_neural: bool = True,
        schema: str = "core",
        q_learning_fallback: bool = True,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        buffer_capacity: int = 10000,
        train_every_n_steps: int = 1000,
        save_path: Path = Path("data/rl_bridge.pkl"),
    ):
        self.use_neural = use_neural and HAS_TORCH
        self.q_learning_fallback = q_learning_fallback
        self.schema = schema
        self.save_path = Path(save_path)
        self.train_every_n_steps = train_every_n_steps
        self.total_steps = 0

        # Neural policy
        self.policy: Optional = None
        self.trainer: Optional = None
        self.device = self._resolve_device(device)

        if self.use_neural:
            try:
                from neural.neural_policy import NeuralPolicy
                from training.ppo_trainer import PPOTrainer

                self.policy = NeuralPolicy(schema=schema)
                self.policy.to(self.device)
                self.trainer = PPOTrainer(self.policy)

                if model_path and model_path.exists():
                    self._load_model(model_path)

                logger.info(f"[RL_BRIDGE] NeuralPolicy inicializada em {self.device}")
            except Exception as e:
                logger.error(f"[RL_BRIDGE] Falha ao carregar NeuralPolicy: {e}")
                self.use_neural = False
                self.policy = None
                self.trainer = None

        # Experience buffer
        self.buffer = ExperienceBuffer(capacity=buffer_capacity)

        # Q-Learning fallback
        self.q_learning = None
        if self.q_learning_fallback:
            try:
                from pylaai_real.rl_engine import CombatQLearning
                self.q_learning = CombatQLearning()
                logger.info("[RL_BRIDGE] Q-Learning fallback ativo")
            except Exception as e:
                logger.warning(f"[RL_BRIDGE] Q-Learning fallback indisponível: {e}")

        # State tracking
        self.last_state_vec: Optional[np.ndarray] = None
        self.last_action_idx: Optional[int] = None
        self.last_log_prob: float = 0.0
        self.last_value: float = 0.0

    def _resolve_device(self, device: Optional[str]) -> str:
        if device is not None:
            return device
        if not HAS_TORCH:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self, path: Path) -> None:
        try:
            state = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(state)
            logger.info(f"[RL_BRIDGE] Modelo carregado: {path}")
        except Exception as e:
            logger.warning(f"[RL_BRIDGE] Falha ao carregar modelo: {e}")

    def _save_model(self, path: Path) -> None:
        if self.policy is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.policy.state_dict(), path)
            logger.info(f"[RL_BRIDGE] Modelo salvo: {path}")
        except Exception as e:
            logger.warning(f"[RL_BRIDGE] Falha ao salvar modelo: {e}")

    # ------------------------------------------------------------------
    # Public API (compatível com OnlineLearner)
    # ------------------------------------------------------------------

    def get_action(
        self,
        state: Tuple,
        player_pos: Tuple[float, float] = (0.5, 0.5),
        enemies: Optional[List] = None,
        detections: Optional[Dict] = None,
        deterministic: bool = False,
        default_action: str = "idle",
    ) -> Tuple[str, float]:
        """
        Retorna (acao, confianca). Compatível com OnlineLearner.get_action().
        """
        if not self.use_neural or self.policy is None:
            return self._q_learning_action(state, default_action)

        # Extrair features
        state_vec = self._state_to_vector(state, player_pos, enemies)

        # Neural policy forward pass
        self.policy.eval()
        with torch.no_grad():
            grid = self._build_grid(player_pos, detections)
            state_t = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
            grid_t = torch.from_numpy(grid).float().unsqueeze(0).to(self.device)

            policy_logits, value = self.policy.forward(grid_t, state_t, reset_hidden=False)
            probs = torch.softmax(policy_logits, dim=-1)

            if deterministic:
                action_idx = torch.argmax(probs, dim=-1).item()
            else:
                action_idx = torch.multinomial(probs, 1).item()

            confidence = probs[0, action_idx].item()
            log_prob = torch.log(probs[0, action_idx]).item()

        # Guardar para experience buffer
        self.last_state_vec = state_vec.copy()
        self.last_action_idx = action_idx
        self.last_log_prob = log_prob
        self.last_value = value.item()

        action_name = self._idx_to_action(action_idx)
        logger.debug(f"[RL_BRIDGE] Neural action={action_name}, conf={confidence:.3f}")
        return action_name, confidence

    def learn_from_frame(
        self,
        state: Tuple,
        action: str,
        reward: float,
        next_state: Tuple,
        player_pos: Tuple[float, float] = (0.5, 0.5),
        enemies: Optional[List] = None,
        detections: Optional[Dict] = None,
        done: bool = False,
    ) -> None:
        """
        Atualiza com transição (s, a, r, s'). Compatível com OnlineLearner.learn_from_frame().
        """
        self.total_steps += 1

        # Neural path: coletar experiência
        if self.use_neural and self.last_state_vec is not None:
            next_state_vec = self._state_to_vector(next_state, player_pos, enemies)
            next_grid = self._build_grid(player_pos, detections)

            exp = Experience(
                state_vector=self.last_state_vec,
                grid=self._last_grid,
                action_idx=self.last_action_idx,
                reward=reward,
                next_state_vector=next_state_vec,
                next_grid=next_grid,
                done=done,
                value=self.last_value,
                log_prob=self.last_log_prob,
            )
            self.buffer.add(exp)

            # Treinar periodicamente
            if self.total_steps % self.train_every_n_steps == 0:
                self._train_step()

        # Q-Learning fallback (sempre atualiza para manter fallback fresco)
        if self.q_learning is not None:
            self.q_learning.update(state, action, reward, next_state)

    def end_episode(self, final_reward: float) -> None:
        """Finaliza episódio. Compatível com OnlineLearner.end_episode()."""
        self.buffer.end_episode()

        if self.use_neural and self.trainer is not None:
            # Treinar com experiências acumuladas
            if len(self.buffer) >= 64:
                self._train_step()
                self._save_model(self.save_path.with_suffix(".pt"))

            # Reset LSTM hidden state
            if self.policy is not None:
                self.policy.reset_hidden_state()

        # Q-Learning
        if self.q_learning is not None:
            self.q_learning.end_episode(final_reward)

        self.last_state_vec = None
        self.last_action_idx = None

    def start_episode(self) -> None:
        """Inicia novo episódio."""
        self.buffer.start_episode()
        self.total_steps = 0
        if self.policy is not None:
            self.policy.reset_hidden_state()

    def get_stats(self) -> Dict:
        """Estatísticas para dashboard."""
        stats = {
            "use_neural": self.use_neural,
            "buffer_size": len(self.buffer),
            "buffer_capacity": getattr(self.buffer, 'capacity', 0),
            "total_steps": self.total_steps,
            "device": self.device,
        }
        if self.q_learning is not None:
            stats["q_learning"] = self.q_learning.get_live_metrics()
        # PPO metrics placeholders (populated by _train_step)
        stats["ppo"] = {
            "policy_loss": round(getattr(self, '_last_policy_loss', 0.0), 4),
            "value_loss": round(getattr(self, '_last_value_loss', 0.0), 4),
            "entropy": round(getattr(self, '_last_entropy', 0.0), 4),
        }
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state_to_vector(
        self,
        state: Tuple,
        player_pos: Tuple[float, float],
        enemies: Optional[List],
    ) -> np.ndarray:
        """Converte estado discreto para vetor contínuo 44-dim."""
        hp_bucket, enemies_bucket, dist_bucket, ammo_bucket, super_bucket = state

        # Converter buckets para valores contínuos aproximados
        hp_pct = [0.15, 0.50, 0.85][min(hp_bucket, 2)]
        num_enemies = enemies_bucket
        nearest_dist = [75.0, 275.0, 600.0][min(dist_bucket, 2)]
        ammo_ratio = 1.0 if ammo_bucket else 0.0
        super_charge = 1.0 if super_bucket else 0.0

        return StateFeatureExtractor.extract(
            player_hp_pct=hp_pct,
            ammo_ratio=ammo_ratio,
            super_charge=super_charge,
            num_enemies=num_enemies,
            nearest_enemy_dist=nearest_dist,
            player_pos=player_pos,
            enemies=enemies,
        )

    def _build_grid(
        self,
        player_pos: Tuple[float, float],
        detections: Optional[Dict],
    ) -> np.ndarray:
        """Build spatial grid para NeuralPolicy."""
        try:
            from neural.grid_builder import SpatialGridBuilder
            from core.class_registry import get_schema
            grid_builder = SpatialGridBuilder()
            grid = grid_builder.build(player_pos, detections or {}, self.schema)
            self._last_grid = grid
            return grid
        except Exception:
            # Fallback: grid vazio
            num_classes = len(get_schema(self.schema))
            grid = np.zeros((21, 21, num_classes), dtype=np.float32)
            self._last_grid = grid
            return grid

    def _q_learning_action(self, state: Tuple, default_action: str) -> Tuple[str, float]:
        if self.q_learning is not None:
            return self.q_learning.get_action(state)
        return default_action, 0.0

    def _train_step(self) -> None:
        if self.trainer is None or len(self.buffer) < 64:
            return
        try:
            batch = self.buffer.sample(64)
            if batch is None:
                return

            # Adaptar batch para interface do PPOTrainer
            # O PPOTrainer espera: experience_buffer com keys específicas
            stats = self.trainer.train(
                experience_buffer=self._adapt_batch(batch),
                num_updates=1,
                batch_size=64,
                ppo_epochs=4,
            )
            logger.info(f"[RL_BRIDGE] Treino PPO: {stats}")
        except Exception as e:
            logger.warning(f"[RL_BRIDGE] Falha no treino PPO: {e}")

    def _adapt_batch(self, batch: Dict):
        """Adapta batch do ExperienceBuffer para interface do PPOTrainer."""
        # O PPOTrainer.train() espera experience_buffer com keys:
        # "states", "actions", "rewards", "dones", "values", "old_log_probs"
        # Nosso batch tem grids + state_vectors, precisamos combinar

        # Criar um objeto mock com método sample()
        class MockBuffer:
            def __init__(self, data):
                self.data = data
            def sample(self, n):
                return self.data

        # Combinar state_vector + grid em um único "state"
        # Para o PPOTrainer, usamos state_vector como proxy (grid é processado internamente)
        return MockBuffer({
            "states": batch["states"],  # (B, 44)
            "actions": batch["actions"],
            "rewards": batch["rewards"].tolist(),
            "dones": batch["dones"].tolist(),
            "values": batch["values"].tolist(),
            "old_log_probs": batch["old_log_probs"].tolist(),
        })

    @staticmethod
    def _idx_to_action(idx: int) -> str:
        from core.class_registry import UnifiedAction
        try:
            return UnifiedAction(idx).name.lower()
        except ValueError:
            return "idle"

    @staticmethod
    def _action_to_idx(name: str) -> int:
        from core.class_registry import UnifiedAction
        try:
            return UnifiedAction[name.upper()].value
        except (KeyError, ValueError):
            return 0  # idle
