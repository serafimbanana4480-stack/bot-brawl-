"""
tests/test_integration_bridges.py

Testes de integração para os bridges e adapters criados.

Cobertura:
- DetectEnsembleAdapter
- CombatDecisionBridge
- TelemetryBridge
- ModelRegistry
- GradientBoostingDecisionSystem
- PositioningHeatmap
- EnrichedDatasetCollector
"""

import pytest
import time
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- DetectEnsembleAdapter ---

def test_ensemble_adapter_api():
    """Testa se o adapter expõe API compatível com Detect."""
    from vision.detect_ensemble_adapter import DetectEnsembleAdapter
    # Criar mock do ensemble
    adapter = DetectEnsembleAdapter([], classes={0: "enemy"})
    assert adapter.detect_objects is not None
    assert adapter.detect_objects_async is not None
    assert adapter.get_async_result is not None

def test_ensemble_adapter_fallback():
    """Testa fallback quando ensemble não está disponível."""
    from vision.detect_ensemble_adapter import DetectEnsembleAdapter
    adapter = DetectEnsembleAdapter([], classes={0: "enemy"})
    result = adapter.detect_objects(None)
    assert result == {}  # Sem fallback model configurado

# --- CombatDecisionBridge ---

def test_combat_bridge_decide_fallback():
    """Testa decisão do bridge sem MOO."""
    from decision.combat_decision_bridge import CombatDecisionBridge
    bridge = CombatDecisionBridge()
    action = bridge.decide_combat_action(
        valid_actions=["attack", "retreat"],
        game_context={"player_hp": 0.5},
        play_logic_recommendation="attack",
    )
    assert action in ["attack", "retreat"]

def test_combat_bridge_get_scores():
    """Testa retorno de scores."""
    from decision.combat_decision_bridge import CombatDecisionBridge
    bridge = CombatDecisionBridge()
    scores = bridge.get_action_scores("attack", {"player_hp": 0.5})
    # Se MOO disponível, retorna dict com objetivos; senão dict vazio
    assert isinstance(scores, dict)
    if bridge._moo is not None:
        assert len(scores) > 0
    else:
        assert scores == {}

# --- TelemetryBridge ---

def test_telemetry_bridge_update_and_get():
    """Testa atualização e recuperação de snapshot."""
    from core.telemetry_bridge import TelemetryBridge
    bridge = TelemetryBridge()

    mock_v2 = MagicMock()
    mock_v2.config = MagicMock()
    mock_v2.config.enabled = True
    mock_v2.config.account_id = "acc1"
    mock_v2._cycle_count = 42

    # Mocks para subsistemas
    mock_v2._degradation_mgr = MagicMock()
    mock_v2._degradation_mgr.get_status.return_value = {
        "mode": "full", "target_fps": 30.0, "max_apm": 60,
    }
    mock_v2._rate_limiter = MagicMock()
    mock_v2._rate_limiter.get_account_status.return_value = {
        "should_play_now": True, "current_session_minutes": 15.0,
        "win_streak": 2, "loss_streak": 0,
    }
    mock_v2._brawler_ctrl = MagicMock()
    mock_v2._brawler_ctrl.get_status.return_value = {
        "current_brawler": "Shelly", "current_playstyle": "aggro",
    }
    mock_v2._checkpointer = MagicMock()
    mock_v2._checkpointer.get_stats.return_value = {
        "total_checkpoints": 5, "last_checkpoint_age_seconds": 30.0,
    }
    mock_v2._alert_system = MagicMock()
    mock_v2._alert_system.get_active_alerts.return_value = [
        {"severity": "info", "message": "test"},
    ]
    mock_v2._frame_skipper = MagicMock()
    mock_v2._frame_skipper.get_stats.return_value = {"processed_ratio": 0.85}
    mock_v2._tracer = MagicMock()
    mock_v2._tracer.get_slow_spans.return_value = [{"name": "inference", "duration_ms": 120}]

    bridge.update(mock_v2, cycle_duration=0.050)
    snap = bridge.get_current()

    assert snap.cycle_count == 42
    assert snap.degradation_mode == "full"
    assert snap.can_play is True
    assert snap.current_brawler == "Shelly"
    assert snap.checkpoint_count == 5
    assert snap.active_alerts == 1

    # Testar JSON serialização
    d = bridge.to_dict()
    assert "degradation" in d
    assert d["rate_limiter"]["win_streak"] == 2

# --- ModelRegistry ---

def test_model_registry_register_and_activate(tmp_path):
    """Testa registro e ativação de modelo."""
    from core.model_registry import ModelRegistry
    reg_dir = tmp_path / "registry_test"
    reg_dir.mkdir()

    registry = ModelRegistry(base_dir=reg_dir)
    assert "yolo" not in registry.list_models()

    # Criar arquivo dummy
    dummy_model = reg_dir / "dummy.pt"
    dummy_model.write_bytes(b"dummy_model_data")

    ver = registry.register("yolo", dummy_model, version="v1.0.0", metrics={"mAP": 0.75})
    assert ver.version == "v1.0.0"
    assert ver.metrics["mAP"] == 0.75

    assert registry.set_active("yolo", "v1.0.0") is True
    assert registry.get_active_version("yolo") == "v1.0.0"

    # Segundo registro deve auto-bump
    ver2 = registry.register("yolo", dummy_model, metrics={"mAP": 0.82})
    assert ver2.version == "v1.0.1"

def test_model_registry_compare_versions(tmp_path):
    """Testa comparação entre versões."""
    from core.model_registry import ModelRegistry
    reg_dir = tmp_path / "registry_test2"
    reg_dir.mkdir()
    registry = ModelRegistry(base_dir=reg_dir)

    dummy = reg_dir / "d.pt"
    dummy.write_bytes(b"a")
    registry.register("yolo", dummy, version="v1", metrics={"mAP": 0.7, "speed": 100})
    registry.register("yolo", dummy, version="v2", metrics={"mAP": 0.8, "speed": 90})

    comp = registry.compare_versions("yolo", "v1", "v2")
    assert "mAP" in comp["metrics_diff"]
    assert "mAP" in comp["v2_better_in"]
    assert "speed" in comp["v1_better_in"]

def test_model_registry_rollback(tmp_path):
    """Testa rollback."""
    import time

    from core.model_registry import ModelRegistry
    reg_dir = tmp_path / "registry_test3"
    reg_dir.mkdir()
    registry = ModelRegistry(base_dir=reg_dir)

    dummy = reg_dir / "d.pt"
    dummy.write_bytes(b"x")
    registry.register("rl", dummy, version="v1")
    time.sleep(0.05)  # ensure distinct created_at timestamps
    registry.register("rl", dummy, version="v2")
    registry.set_active("rl", "v2")

    assert registry.rollback("rl", steps=1) is True
    assert registry.get_active_version("rl") == "v1"

# --- GradientBoostingDecisionSystem ---

def test_gbd_basic_voting():
    """Testa votação básica."""
    from decision.gradient_boosting_decisions import GradientBoostingDecisionSystem
    gbd = GradientBoostingDecisionSystem()

    def mock_decide(ctx):
        return ("attack", 0.8)

    gbd.add_voter("mock1", 1.0, mock_decide)
    action = gbd.decide({"hp": 0.5})
    assert action == "attack"

def test_gbd_multiple_voters():
    """Testa múltiplos votantes."""
    from decision.gradient_boosting_decisions import GradientBoostingDecisionSystem
    gbd = GradientBoostingDecisionSystem()

    gbd.add_voter("aggro", 1.0, lambda ctx: ("attack", 0.9))
    gbd.add_voter("def", 0.5, lambda ctx: ("retreat", 0.9))

    # aggro tem mais peso e mesma confiança, deve vencer
    action = gbd.decide({})
    assert action == "attack"

def test_gbd_weight_update():
    """Testa atualização de pesos."""
    from decision.gradient_boosting_decisions import GradientBoostingDecisionSystem
    gbd = GradientBoostingDecisionSystem(learning_rate=0.1)

    gbd.add_voter("v1", 1.0, lambda ctx: ("a", 1.0))
    initial_weight = gbd.voters[0].weight

    gbd.update_weights_from_outcome("a", "a", reward=1.0)
    assert gbd.voters[0].weight > initial_weight

    gbd.update_weights_from_outcome("b", "a", reward=-1.0)
    assert gbd.voters[0].weight < initial_weight + 0.001  # penalizado

def test_gbd_status():
    """Testa status do GBD."""
    from decision.gradient_boosting_decisions import GradientBoostingDecisionSystem
    gbd = GradientBoostingDecisionSystem()
    gbd.add_voter("v1", 1.0, lambda ctx: ("a", 1.0))
    status = gbd.get_status()
    assert status["voter_count"] == 1
    assert "voters" in status

# --- PositioningHeatmap ---

def test_heatmap_basic_update():
    """Testa atualização e estatísticas do heatmap."""
    from core.positioning_heatmap import PositioningHeatmap
    hm = PositioningHeatmap(map_width=1000, map_height=1000, cell_size=100)

    hm.update_bot_position(500, 500, dt=1.0)
    hm.update_bot_position(500, 500, dt=2.0)

    stats = hm.get_stats()
    assert stats["total_bot_time"] > 0
    assert stats["grid_size"] == (10, 10)

def test_heatmap_danger_zones():
    """Testa detecção de zonas de risco."""
    from core.positioning_heatmap import PositioningHeatmap
    hm = PositioningHeatmap(map_width=1000, map_height=1000, cell_size=100)

    # Simular muitas mortes numa zona
    for _ in range(5):
        hm.record_death(500, 500)

    zones = hm.compute_danger_zones(threshold_ratio=0.5)
    assert len(zones) > 0
    assert zones[0]["danger_score"] > 0

def test_heatmap_escape():
    """Testa direção de escape."""
    from core.positioning_heatmap import PositioningHeatmap
    hm = PositioningHeatmap(map_width=1000, map_height=1000, cell_size=100)

    # Bot visitou (500,500) muito
    hm.update_bot_position(500, 500, dt=100.0)

    escape = hm.get_least_visited_escape(500, 500, max_distance=200)
    assert escape is not None
    assert escape != (500, 500)

# --- EnrichedDatasetCollector ---

def test_collector_lifecycle(tmp_path):
    """Testa ciclo completo de coleta."""
    from data.enriched_collector import EnrichedDatasetCollector
    collector = EnrichedDatasetCollector(output_dir=tmp_path, save_screenshots=False)

    collector.start_match("match_1", "Shelly", "Gem_Grab")
    assert collector._current_match == "match_1"

    frame = collector.collect_frame(
        screenshot=None,
        game_state="in_game",
        detections={"enemy": [[100, 100, 200, 200]]},
        decision={"action": "attack", "scores": {"attack": 0.9}},
        performance={"cycle_ms": 50.0, "inference_ms": 20.0},
    )
    assert frame is not None
    assert frame.action_taken == "attack"

    collector.end_match("victory", {"trophies": 8})
    assert collector._current_match is None

    # Verificar se arquivo meta foi criado
    meta_file = tmp_path / "match_1_meta.json"
    assert meta_file.exists()
    with open(meta_file) as f:
        meta = json.load(f)
    assert meta["result"] == "victory"
    assert meta["total_frames"] == 1

def test_collector_stats():
    """Testa estatísticas do collector."""
    from data.enriched_collector import EnrichedDatasetCollector
    collector = EnrichedDatasetCollector(output_dir=Path("/tmp/test"), save_screenshots=False)
    stats = collector.get_stats()
    assert "session_id" in stats
    assert "output_dir" in stats
