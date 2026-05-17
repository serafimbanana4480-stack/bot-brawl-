"""
tests/test_rl_bridge.py

Testes de integracao para o RLBridge (NeuralPolicy + PPO + Q-Learning fallback).
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from neural.rl_bridge import RLBridge, ExperienceBuffer, StateFeatureExtractor, Experience


class TestStateFeatureExtractor:
    """Testes para extracao de features de estado."""

    def test_extract_basic(self):
        vec = StateFeatureExtractor.extract(
            player_hp_pct=0.8,
            ammo_ratio=1.0,
            super_charge=0.5,
            num_enemies=2,
            nearest_enemy_dist=200.0,
        )
        assert vec.shape == (StateFeatureExtractor.NUM_FEATURES,)
        assert vec.dtype == np.float32
        assert 0.0 <= vec[0] <= 1.0  # hp_ratio
        assert 0.0 <= vec[1] <= 1.0  # ammo_ratio

    def test_extract_clipping(self):
        vec = StateFeatureExtractor.extract(player_hp_pct=-0.5, ammo_ratio=2.0)
        assert vec[0] == 0.0  # clipped to 0
        assert vec[1] == 1.0  # clipped to 1

    def test_extract_with_enemies(self):
        enemies = [[100, 100, 200, 200], [300, 300, 400, 400]]
        vec = StateFeatureExtractor.extract(
            num_enemies=2,
            player_pos=(0.5, 0.5),
            enemies=enemies,
        )
        assert vec[18] > 0  # num_enemies feature


class TestExperienceBuffer:
    """Testes para o buffer de experiencias."""

    def test_add_and_sample(self):
        buf = ExperienceBuffer(capacity=100)
        for i in range(10):
            buf.add(Experience(
                state_vector=np.random.randn(44).astype(np.float32),
                grid=None,
                action_idx=i % 4,
                reward=1.0,
                next_state_vector=np.random.randn(44).astype(np.float32),
                next_grid=None,
                done=False,
                value=0.5,
                log_prob=-0.5,
            ))
        batch = buf.sample(5)
        assert batch is not None
        assert batch["states"].shape == (5, 44)
        assert batch["actions"].shape == (5,)
        assert batch["rewards"].shape == (5,)

    def test_sample_insufficient(self):
        buf = ExperienceBuffer(capacity=100)
        buf.add(Experience(
            state_vector=np.zeros(44, dtype=np.float32),
            grid=None,
            action_idx=0,
            reward=0.0,
            next_state_vector=np.zeros(44, dtype=np.float32),
            next_grid=None,
            done=False,
            value=0.0,
            log_prob=0.0,
        ))
        assert buf.sample(5) is None

    def test_episode_truncation(self):
        buf = ExperienceBuffer(capacity=100, min_episode_length=5)
        buf.start_episode()
        for _ in range(3):
            buf.add(Experience(
                state_vector=np.zeros(44, dtype=np.float32),
                grid=None,
                action_idx=0,
                reward=0.0,
                next_state_vector=np.zeros(44, dtype=np.float32),
                next_grid=None,
                done=False,
                value=0.0,
                log_prob=0.0,
            ))
        buf.end_episode()
        assert len(buf) == 0  # Truncated


class TestRLBridge:
    """Testes para o RLBridge."""

    def test_init_without_torch(self):
        """Deve funcionar sem torch (fallback Q-Learning)."""
        with patch("neural.rl_bridge.HAS_TORCH", False):
            bridge = RLBridge(use_neural=True, q_learning_fallback=False)
            assert bridge.use_neural is False
            assert bridge.policy is None

    def test_init_q_learning_only(self):
        """Deve funcionar com Q-Learning apenas."""
        bridge = RLBridge(use_neural=False, q_learning_fallback=True)
        assert bridge.use_neural is False
        assert bridge.q_learning is not None

    def test_q_learning_fallback_action(self):
        bridge = RLBridge(use_neural=False, q_learning_fallback=True)
        state = (1, 1, 1, 1, 0)  # hp, enemies, dist, ammo, super
        action, conf = bridge.get_action(state)
        assert isinstance(action, str)
        assert 0.0 <= conf <= 1.0

    @pytest.mark.skipif(not __import__("torch", __import__("sys").modules[__name__].__dict__), reason="torch not available")
    def test_neural_policy_inference(self):
        """Testa inference com NeuralPolicy real."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        bridge = RLBridge(use_neural=True, q_learning_fallback=False, schema="core")
        if bridge.policy is None:
            pytest.skip("NeuralPolicy could not be loaded")

        state = (1, 1, 1, 1, 0)
        action, conf = bridge.get_action(
            state,
            player_pos=(0.5, 0.5),
            enemies=[[100, 100, 200, 200]],
            detections={"enemy": [[100, 100, 200, 200]]},
        )
        assert isinstance(action, str)
        assert 0.0 <= conf <= 1.0

    def test_state_to_vector(self):
        bridge = RLBridge(use_neural=False, q_learning_fallback=False)
        state = (2, 2, 0, 1, 1)  # high hp, 2+ enemies, close, ammo, super
        vec = bridge._state_to_vector(state, (0.5, 0.5), None)
        assert vec.shape == (44,)
        assert vec.dtype == np.float32
        # hp_bucket=2 -> ~0.85
        assert 0.7 <= vec[0] <= 1.0

    def test_learn_from_frame_q_only(self):
        bridge = RLBridge(use_neural=False, q_learning_fallback=True)
        state = (1, 1, 1, 1, 0)
        bridge.learn_from_frame(state, "attack", 1.0, (1, 1, 1, 0, 0))
        # Nao deve levantar excecao
        assert bridge.total_steps == 1

    def test_end_episode(self):
        bridge = RLBridge(use_neural=False, q_learning_fallback=False)
        bridge.start_episode()
        bridge.end_episode(5.0)
        assert bridge.last_state_vec is None
        assert bridge.last_action_idx is None

    def test_stats(self):
        bridge = RLBridge(use_neural=False, q_learning_fallback=False)
        stats = bridge.get_stats()
        assert "use_neural" in stats
        assert "buffer_size" in stats
        assert "total_steps" in stats


class TestRLBridgeIntegration:
    """Testes de integracao end-to-end."""

    def test_neural_then_q_fallback_on_error(self):
        """Se NeuralPolicy falhar, deve recair para Q-Learning."""
        bridge = RLBridge(use_neural=False, q_learning_fallback=True)

        # Simular falha forçando use_neural=True mas com policy=None
        bridge.use_neural = True
        bridge.policy = None

        state = (1, 0, 2, 1, 0)
        action, conf = bridge.get_action(state)
        assert isinstance(action, str)

    def test_experience_collection_and_training(self):
        """Coleta de experiencias deve funcionar sem erros."""
        bridge = RLBridge(use_neural=True, q_learning_fallback=False)
        if bridge.policy is None:
            pytest.skip("NeuralPolicy not available")
        bridge.start_episode()

        for i in range(100):
            state = (1, i % 3, i % 3, 1, 0)
            action = "attack"
            reward = 1.0
            next_state = (1, (i + 1) % 3, (i + 1) % 3, 1, 0)
            # Simular last_state_vec para que o buffer colete
            bridge.last_state_vec = np.zeros(44, dtype=np.float32)
            bridge.last_action_idx = 0
            bridge.last_log_prob = -0.5
            bridge.last_value = 0.5
            bridge._last_grid = np.zeros((21, 21, 1), dtype=np.float32)
            bridge.learn_from_frame(state, action, reward, next_state)

        bridge.end_episode(10.0)
        assert len(bridge.buffer) > 0
