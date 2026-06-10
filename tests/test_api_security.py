"""
Tests for API security hardening in api_server.py.

Run with: pytest tests/test_api_security.py -v
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_FAKE_KEY = "test-secret-key-12345"
_FAKE_CONFIG = {
    "api": {
        "host": "127.0.0.1",
        "port": 8003,
        "cors_origins": ["http://localhost:3000"],
        "api_key": _FAKE_KEY,
    },
    "logging": {"level": "INFO"},
}


@pytest.fixture(scope="module")
def client():
    fake_logger = MagicMock()
    fake_logger.info = MagicMock()
    fake_logger.warning = MagicMock()
    fake_logger.error = MagicMock()
    fake_logger.debug = MagicMock()

    class FakeReport:
        overall = "healthy"

    fake_health_mod = MagicMock()
    fake_health_mod.health_shallow.return_value = FakeReport()
    fake_health_mod.health_deep.return_value = FakeReport()
    fake_health_mod.to_dict.return_value = {"status": "ok"}

    with patch.dict(
        "sys.modules",
        {
            "core.logging_config": MagicMock(
                setup_logging=MagicMock(), get_logger=lambda *a, **k: fake_logger
            ),
            "core.metrics": MagicMock(
                set_bot_state=MagicMock(),
                inc_matches_completed=MagicMock(),
                inc_errors=MagicMock(),
            ),
            "core.health_checks": fake_health_mod,
            "brawl_bot.realtime_logs": MagicMock(),
            "brawl_bot.emulator_detector": MagicMock(),
            "wrapper": MagicMock(),
            "match_controller": MagicMock(),
            "safety_system": MagicMock(),
            "pylaai_real.lobby_automator": MagicMock(),
        },
    ):
        import api_server as _api_server_module
        import importlib

        importlib.reload(_api_server_module)

        _api_server_module._FAKE_CONFIG = _FAKE_CONFIG
        _api_server_module._API_KEY = _FAKE_KEY
        _api_server_module._IS_DEBUG = False

        fake_bot = MagicMock()
        fake_bot.running = False
        fake_bot.get_status.return_value = {"current_state": "idle"}
        fake_bot.safety.config.max_trophies = 400
        fake_bot.safety.config.warning_trophies = 380
        fake_bot.safety.config.max_session_hours = 3.0
        fake_bot.safety.config.min_apm = 20
        fake_bot.safety.config.max_apm = 60
        fake_bot.safety.config.auto_stop_on_detection = True
        fake_bot.safety.config.break_duration_min = 300
        fake_bot.safety.config.break_duration_max = 900
        fake_bot.safety.config.suspicious_pattern_threshold = 5
        fake_bot.get_queue.return_value = []
        fake_bot.brawler_queue = MagicMock()
        fake_bot.brawler_queue.brawlers = []
        fake_bot.start.return_value = True
        fake_bot.stop.return_value = True
        fake_bot.setup.return_value = True
        fake_bot.add_brawler_to_queue.return_value = None

        with patch.object(_api_server_module, "get_bot", return_value=fake_bot):
            from fastapi.testclient import TestClient

            with TestClient(_api_server_module.app) as tc:
                yield tc


@pytest.fixture(autouse=True)
def clear_rate_limiter():
    import api_server

    # slowapi stores limits in app.state.limiter
    if hasattr(api_server.app.state, "limiter"):
        api_server.app.state.limiter.reset()
    yield


class TestPublicEndpoints:
    """Public endpoints must work without API key."""

    def test_health_no_auth(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_deep_no_auth(self, client):
        r = client.get("/health/deep")
        assert r.status_code == 200

    def test_ready_no_auth(self, client):
        r = client.get("/ready")
        assert r.status_code == 200
        assert "ready" in r.json()

    def test_metrics_no_auth(self, client):
        r = client.get("/metrics")
        assert r.status_code in (200, 503)

    def test_api_health_no_auth(self, client):
        r = client.get("/api/brawl-stars/health")
        assert r.status_code == 200

    def test_api_status_no_auth(self, client):
        r = client.get("/api/brawl-stars/status")
        assert r.status_code == 200

    def test_docs_no_auth(self, client):
        r = client.get("/docs")
        assert r.status_code == 200


class TestApiKeyAuth:
    """Control endpoints require a valid X-API-Key header."""

    def test_start_without_key_401(self, client):
        r = client.post("/api/brawl-stars/start")
        assert r.status_code == 401
        assert "Invalid or missing API key" in r.json()["detail"]

    def test_start_with_wrong_key_401(self, client):
        r = client.post(
            "/api/brawl-stars/start", headers={"X-API-Key": "wrong-key"}
        )
        assert r.status_code == 401

    def test_start_with_valid_key_200(self, client):
        r = client.post(
            "/api/brawl-stars/start", headers={"X-API-Key": _FAKE_KEY}
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_stop_requires_auth(self, client):
        r = client.post("/api/brawl-stars/stop")
        assert r.status_code == 401

    def test_setup_requires_auth(self, client):
        r = client.post("/api/brawl-stars/setup")
        assert r.status_code == 401

    def test_config_post_requires_auth(self, client):
        r = client.post("/api/brawl-stars/config", json={})
        assert r.status_code == 401

    def test_brawler_add_requires_auth(self, client):
        r = client.post("/api/brawl-stars/brawler/add", json={})
        assert r.status_code == 401

    def test_brawler_delete_requires_auth(self, client):
        r = client.delete("/api/brawl-stars/brawler/Colt")
        assert r.status_code == 401

    def test_auto_tuning_post_requires_auth(self, client):
        r = client.post("/api/brawl-stars/auto-tuning/tune")
        assert r.status_code == 401

    def test_auto_tuning_reset_requires_auth(self, client):
        r = client.post("/api/brawl-stars/auto-tuning/reset")
        assert r.status_code == 401


class TestRateLimiting:
    """slowapi rate limiting."""

    def test_read_rate_limit(self, client):
        codes = []
        for _ in range(15):
            r = client.get("/health")
            codes.append(r.status_code)
        assert 429 in codes, f"Expected 429 among codes, got {codes}"

    def test_write_rate_limit(self, client):
        codes = []
        for _ in range(5):
            r = client.post(
                "/api/brawl-stars/start",
                headers={"X-API-Key": _FAKE_KEY},
            )
            codes.append(r.status_code)
        assert 429 in codes, f"Expected 429 among codes, got {codes}"

    def test_rate_limit_resets_after_window(self, client):
        for _ in range(15):
            client.get("/health")
        time.sleep(1.1)
        r = client.get("/health")
        assert r.status_code == 200


class TestCorsWhitelist:
    """CORS must only allow validated origins."""

    def test_allowed_origin(self, client):
        r = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert "access-control-allow-origin" in r.headers

    def test_disallowed_origin(self, client):
        r = client.get(
            "/health",
            headers={"Origin": "http://evil.com"},
        )
        assert r.headers.get("access-control-allow-origin") != "http://evil.com"


class TestInputSanitization:
    """Brawler names and queue updates must be sanitized."""

    def test_add_brawler_invalid_name_rejected(self, client):
        payload = {
            "name": "../../../etc/passwd",
            "current_trophies": 0,
            "target_trophies": 350,
            "target_wins": 10,
            "priority": 1,
        }
        r = client.post(
            "/api/brawl-stars/brawler/add",
            json=payload,
            headers={"X-API-Key": _FAKE_KEY},
        )
        assert r.status_code == 422

    def test_config_update_invalid_queue_rejected(self, client):
        payload = {
            "trophy_limit": 400,
            "warning_trophies": 380,
            "max_session_hours": 3.0,
            "min_apm": 20,
            "max_apm": 60,
            "auto_pause": True,
            "auto_stop_on_detection": True,
            "brawler_queue": [
                {"name": "valid_name", "priority": 1},
                {"name": "<script>alert(1)</script>", "priority": 2},
            ],
            "safety_settings": {},
            "diagnostic_mode": False,
        }
        r = client.post(
            "/api/brawl-stars/config",
            json=payload,
            headers={"X-API-Key": _FAKE_KEY},
        )
        assert r.status_code == 422

    def test_valid_brawler_name_accepted(self, client):
        payload = {
            "name": "Colt",
            "current_trophies": 0,
            "target_trophies": 350,
            "target_wins": 10,
            "priority": 1,
        }
        r = client.post(
            "/api/brawl-stars/brawler/add",
            json=payload,
            headers={"X-API-Key": _FAKE_KEY},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True


class TestGenericErrorMessages:
    """In production (LOG_LEVEL != DEBUG) stack traces must not leak."""

    def test_production_error_is_generic(self, client):
        import api_server

        original_debug = api_server._IS_DEBUG
        api_server._IS_DEBUG = False
        try:
            with patch.object(
                api_server, "get_bot", side_effect=RuntimeError("secret internals")
            ):
                r = client.get("/api/brawl-stars/status")
                assert r.status_code == 500
                body = r.json()
                assert "detail" in body
                assert "secret internals" not in body["detail"]
                assert "Internal server error" in body["detail"]
        finally:
            api_server._IS_DEBUG = original_debug

    def test_debug_mode_shows_error(self, client):
        import api_server

        original_debug = api_server._IS_DEBUG
        api_server._IS_DEBUG = True
        try:
            with patch.object(
                api_server, "get_bot", side_effect=RuntimeError("debug details")
            ):
                with pytest.raises(Exception) as exc_info:
                    client.get("/api/brawl-stars/status")
                assert "debug details" in str(exc_info.value)
        finally:
            api_server._IS_DEBUG = original_debug


class TestWebsocketAuth:
    """WebSocket control endpoint must also enforce auth."""

    def test_websocket_without_key_rejected(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/brawl-stars") as ws:
                pass

    def test_websocket_with_valid_key_accepted(self, client):
        with client.websocket_connect(
            "/ws/brawl-stars", headers={"X-API-Key": _FAKE_KEY}
        ) as ws:
            ws.send_json({"type": "ping"})
            data = ws.receive_json()
            assert data["type"] == "pong"
