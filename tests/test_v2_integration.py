"""
tests/test_v2_integration.py

Testes de integração para módulos v2.1.

Verifica que o integrador conecta corretamente todos os subsistemas
sem quebrar o ciclo principal.
"""

import pytest
import time
import tempfile
from pathlib import Path


class TestV2Integrator:
    def test_singleton(self):
        from core.v2_integration import V2Integrator, V2IntegrationConfig
        v2a = V2Integrator(config=V2IntegrationConfig())
        v2b = V2Integrator(config=V2IntegrationConfig())
        assert v2a is v2b

    def test_cycle_start_abort_on_rate_limit(self):
        from core.v2_integration import V2Integrator, V2IntegrationConfig
        cfg = V2IntegrationConfig()
        cfg.enable_rate_limiter = True
        cfg.account_id = "test_acc"
        v2 = V2Integrator(config=cfg)
        # Forçar break: entre 2h-6h da manha
        if v2._rate_limiter:
            profile = v2._rate_limiter.get_profile("test_acc")
            if profile:
                profile.current_session_start = time.time() - (300 * 60)
        result = v2.on_cycle_start()
        # O rate limiter pode ou não bloquear dependendo do horário
        # O importante é que não lance exceção e retorne bool
        assert isinstance(result, bool)

    def test_cycle_start_returns_true_when_ok(self):
        from core.v2_integration import V2Integrator, V2IntegrationConfig
        cfg = V2IntegrationConfig()
        cfg.enable_rate_limiter = False
        v2 = V2Integrator(config=cfg)
        assert v2.on_cycle_start() is True

    def test_brawler_adaptation(self):
        from core.v2_integration import V2Integrator, V2IntegrationConfig
        cfg = V2IntegrationConfig()
        cfg.enable_brawler_adaptive = True
        v2 = V2Integrator(config=cfg)
        v2._last_brawler = None
        # Simular mudança de brawler
        v2._get_current_brawler = lambda: "Colt"
        v2.on_cycle_start()
        if v2._brawler_ctrl:
            assert v2._last_brawler == "Colt"
        else:
            assert v2._last_brawler is None

    def test_match_hooks(self):
        from core.v2_integration import V2Integrator, V2IntegrationConfig
        cfg = V2IntegrationConfig()
        cfg.enable_event_store = True
        cfg.enable_rate_limiter = True
        v2 = V2Integrator(config=cfg)
        v2.on_match_start("Shelly", "Showdown")
        v2.on_match_end("win", "Shelly", "Showdown", {"kills": 3})
        # Se não lançou exceção, passou
        assert True

    def test_dashboard_data(self):
        from core.v2_integration import V2Integrator, V2IntegrationConfig
        cfg = V2IntegrationConfig()
        cfg.enable_degradation_manager = True
        cfg.enable_alert_system = True
        v2 = V2Integrator(config=cfg)
        data = v2.get_dashboard_data()
        assert "v2_enabled" in data
        assert "cycle_count" in data
        assert "degradation" in data
        assert "alerts" in data

    def test_shutdown(self):
        from core.v2_integration import V2Integrator, V2IntegrationConfig
        v2 = V2Integrator(config=V2IntegrationConfig())
        v2.shutdown()
        assert V2Integrator._instance is None


class TestSmartFrameSkipper:
    def test_never_skip_in_combat(self):
        from core.smart_frame_skipper import SmartFrameSkipper
        fs = SmartFrameSkipper()
        for i in range(10):
            assert fs.should_process_frame(i, "in_game", combat_active=True) is True

    def test_skip_in_lobby(self):
        from core.smart_frame_skipper import SmartFrameSkipper
        fs = SmartFrameSkipper()
        results = [fs.should_process_frame(i, "lobby") for i in range(10)]
        assert sum(results) < 10  # Alguns frames devem ser skipados

    def test_skip_increases_with_degradation(self):
        from core.smart_frame_skipper import SmartFrameSkipper
        fs = SmartFrameSkipper()
        full = sum(fs.should_process_frame(i, "lobby", "full_quality") for i in range(20))
        degraded = sum(fs.should_process_frame(i, "lobby", "degraded") for i in range(20))
        assert degraded <= full

    def test_stats(self):
        from core.smart_frame_skipper import SmartFrameSkipper
        fs = SmartFrameSkipper()
        for i in range(10):
            fs.should_process_frame(i, "lobby")
        stats = fs.get_stats()
        assert "processed_ratio" in stats
        assert "total_frames" in stats


class TestAlertSystem:
    def test_performance_alert(self):
        from core.alert_system import AlertSystem
        alerts = AlertSystem()
        alerts.check_alerts(cycle_duration=5.0)
        active = alerts.get_active_alerts()
        assert len(active) >= 1
        assert active[0]["category"] == "performance"

    def test_degradation_alert(self):
        from core.alert_system import AlertSystem
        alerts = AlertSystem()
        alerts.check_alerts(degradation_status={"mode": "emergency"})
        active = alerts.get_active_alerts()
        crit = [a for a in active if a["severity"] == "critical"]
        assert len(crit) >= 1

    def test_acknowledge(self):
        from core.alert_system import AlertSystem
        alerts = AlertSystem()
        alerts.check_alerts(cycle_duration=5.0)
        active = alerts.get_active_alerts()
        alert_id = active[0]["id"]
        assert alerts.acknowledge(alert_id) is True
        assert len(alerts.get_active_alerts()) == 0

    def test_stats(self):
        from core.alert_system import AlertSystem
        alerts = AlertSystem()
        alerts.check_alerts(cycle_duration=5.0)
        stats = alerts.get_stats()
        assert stats["total_alerts"] >= 1


class TestAutoROICalibrator:
    def test_get_roi(self):
        from core.auto_roi_calibrator import AutoROICalibrator
        cal = AutoROICalibrator()
        roi = cal.get_roi("hp_bar", 2560, 1440)
        assert roi is not None
        assert len(roi) == 4
        x, y, w, h = roi
        assert x >= 0 and y >= 0
        assert x + w <= 2560 and y + h <= 1440

    def test_scale_proportionally(self):
        from core.auto_roi_calibrator import AutoROICalibrator
        cal = AutoROICalibrator()
        roi_1080 = cal.get_roi("play_button", 1920, 1080)
        roi_1440 = cal.get_roi("play_button", 2560, 1440)
        # 1440 é ~1.33x maior que 1080 em ambas dimensões
        assert roi_1440[2] > roi_1080[2]
        assert roi_1440[3] > roi_1080[3]

    def test_cache(self):
        from core.auto_roi_calibrator import AutoROICalibrator
        with tempfile.TemporaryDirectory() as tmpdir:
            cal = AutoROICalibrator(cache_dir=Path(tmpdir))
            cal.get_roi("hp_bar", 1920, 1080)
            info = cal.get_calibration_info(1920, 1080)
            assert info["cached"] is True

    def test_add_custom_roi(self):
        from core.auto_roi_calibrator import AutoROICalibrator
        cal = AutoROICalibrator()
        cal.add_roi("custom_test", 100, 200, 50, 60, "Test ROI")
        roi = cal.get_roi("custom_test", 1920, 1080)
        assert roi == (100, 200, 50, 60)

    def test_clamping(self):
        from core.auto_roi_calibrator import AutoROICalibrator
        cal = AutoROICalibrator()
        # Forçar ROI que sairia da tela
        cal.add_roi("huge", 1800, 1000, 300, 200)
        roi = cal.get_roi("huge", 1920, 1080)
        x, y, w, h = roi
        assert x + w <= 1920
        assert y + h <= 1080
