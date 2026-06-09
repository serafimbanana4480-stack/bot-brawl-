"""
tests/test_learning_rl_metrics.py

Testes para o coletor de métricas RL.
"""

import sys, os

import pytest
from unittest.mock import MagicMock

from core.learning_rl_metrics import LearningRLMetricsCollector


class TestLearningRLMetricsCollector:
    def test_init(self):
        c = LearningRLMetricsCollector()
        m = c.get_metrics()
        assert m["active"] is False
        assert m["q_table_size"] == 0

    def test_start_stop_session(self):
        c = LearningRLMetricsCollector()
        c.start_session()
        assert c.get_metrics()["active"] is True
        c.stop_session()
        assert c.get_metrics()["active"] is False

    def test_record_reward(self):
        c = LearningRLMetricsCollector()
        c.start_session()
        c.record_reward(1.5, action="attack", is_exploration=True)
        m = c.get_metrics()
        assert m["last_reward"] == 1.5
        assert m["episode_reward"] == 1.5
        assert m["total_reward"] == 1.5
        assert m["last_action"] == "attack"
        assert m["action_counts"]["attack"] == 1

    def test_update_q_learning(self):
        c = LearningRLMetricsCollector()
        rl = MagicMock()
        rl.q_table = {"s1": {"a": 1}}
        rl.epsilon = 0.2
        rl.total_updates = 42
        rl.visit_counts = {"s1": 5}
        c.update_q_learning(rl)
        m = c.get_metrics()
        assert m["q_table_size"] == 1
        assert m["epsilon"] == 0.2
        assert m["total_updates"] == 42
        assert m["visits_avg"] == 5.0

    def test_end_episode(self):
        c = LearningRLMetricsCollector()
        c.start_session()
        c.record_reward(2.0)
        c.end_episode()
        assert c.get_metrics()["episode_reward"] == 0.0
