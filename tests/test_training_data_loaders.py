"""Tests for real dataset loading in the unified training system."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent

from training.unified_training_system import UnifiedTrainingSystem


def test_bc_loader_uses_real_dataset():
    system = UnifiedTrainingSystem(ROOT)
    dataset = system._load_bc_dataset(ROOT / "dataset" / "synthetic_massive" / "bc_massive.json")

    assert len(dataset) > 0
    state, label = dataset[0]
    assert state.shape[0] == 8
    assert 0 <= int(label) <= 4


def test_cql_loader_uses_real_dataset():
    system = UnifiedTrainingSystem(ROOT)
    dataset = system._load_replay_buffer(ROOT / "dataset" / "synthetic_massive" / "replay_buffer_massive.json")

    assert len(dataset) > 0
    state, action, reward, next_state, done = dataset[0]
    assert state.shape[0] == 8
    assert action.shape[0] == 5
    assert reward.shape[0] == 1
    assert next_state.shape[0] == 8
    assert done.shape[0] == 1
