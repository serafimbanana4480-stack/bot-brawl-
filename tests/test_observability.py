import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import importlib.util

_spec = importlib.util.spec_from_file_location("observability", str(_repo_root / "core" / "observability.py"))
_obs_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_obs_mod)
ObservabilityCollector = _obs_mod.ObservabilityCollector
HealthChecker = _obs_mod.HealthChecker
MetricSnapshot = _obs_mod.MetricSnapshot

import tempfile


def test_observability_initialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        obs = ObservabilityCollector(max_events=100, metrics_dir=Path(tmpdir))
        assert obs.matches_total == 0
        assert obs.wins == 0
        assert obs.losses == 0


def test_record_match_result():
    obs = ObservabilityCollector(max_events=10)
    obs.record_match_result("win", brawler="Shelly", map_name="Island Invasion")
    assert obs.matches_total == 1
    assert obs.wins == 1
    assert obs.losses == 0
    assert obs.current_brawler == "Shelly"
    assert obs.current_map == "Island Invasion"

    obs.record_match_result("loss", brawler="Colt")
    assert obs.matches_total == 2
    assert obs.wins == 1
    assert obs.losses == 1


def test_get_snapshot():
    obs = ObservabilityCollector(max_events=10)
    obs.record_match_result("win")
    obs.record_reward(10.5)
    snapshot = obs.get_snapshot()
    assert isinstance(snapshot, MetricSnapshot)
    assert snapshot.matches_total == 1
    assert snapshot.wins == 1
    assert snapshot.avg_reward == 10.5


def test_health_checker():
    hc = HealthChecker()
    hc.register("test", lambda: "ok")
    results = hc.run()
    assert results["test"]["status"] == "ok"
    assert results["test"]["details"] == "ok"


def test_health_checker_error():
    hc = HealthChecker()
    hc.register("failing", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    results = hc.run()
    assert results["failing"]["status"] == "error"
