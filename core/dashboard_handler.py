"""
Dashboard HTTP request handler (extracted from dashboard_server.py).
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

try:
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
except ImportError:
    ThreadingHTTPServer = None
    HTTPServer = None
    BaseHTTPRequestHandler = None

try:
    import importlib.util as _ilu
    HAS_NUMPY = _ilu.find_spec("numpy") is not None
except Exception:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)

try:
    from core.log_buffer import LogBuffer, get_log_buffer, install_log_buffer
    HAS_LOGBUFFER = True
except ImportError:
    HAS_LOGBUFFER = False
    LogBuffer = None
    install_log_buffer = None
    get_log_buffer = None

try:
    from core.notifications import NotificationManager, get_notification_manager
    HAS_NOTIFICATIONS = True
except ImportError:
    HAS_NOTIFICATIONS = False
    NotificationManager = None
    get_notification_manager = None

from core.dashboard_logic import (  # noqa: E402
    ABTestManager,
    DashboardDataBridge,
    ReplayRecorder,
)
from core.dashboard_templates import _DASHBOARD_HTML  # noqa: E402

# Import for BrawlerConfig used in farm mode
try:
    from pylaai_real.lobby_automator import BrawlerConfig
except ImportError:
    BrawlerConfig = None


class DashboardHandler(BaseHTTPRequestHandler):
    """Handler que serve dashboard HTML e API JSON."""

    bridge: DashboardDataBridge | None = None
    recorder: ReplayRecorder | None = None
    ab_test: ABTestManager | None = None
    wrapper_ref: Any | None = None  # Reference to wrapper for bot control
    log_buffer: Any | None = None  # Phase 2: LogBuffer for real-time logs
    notification_manager: Any | None = None  # Phase 3: NotificationManager

    def log_message(self, format, *args):
        pass  # Silenciar logs de acesso

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]

        if path == "/" or path == "/dashboard":
            self._send_html(_DASHBOARD_HTML)
        elif path == "/api/health":
            self._send_json({
                "ok": True,
                "dashboard": True,
                "bot_connected": bool(self.wrapper_ref),
                "timestamp": time.time(),
            })
        elif path == "/api/live":
            data = self.bridge.get_snapshot() if self.bridge else {}
            # Merge wrapper_ref running state when bridge is stale
            w = self.wrapper_ref
            if w and not data.get("running"):
                data["running"] = getattr(w, 'running', False)
                sm = getattr(w, 'state_manager', None)
                if sm:
                    data["current_state"] = data.get("current_state") or sm.current_state
                    brawler = getattr(sm, 'current_brawler', None)
                    if brawler:
                        data["brawler"] = data.get("brawler") or brawler
            self._send_json(data)
        elif path == "/api/history":
            data = self.bridge.get_history() if self.bridge else []
            self._send_json({"history": data})
        elif path == "/api/rewards":
            data = self.bridge.get_rewards_history() if self.bridge else []
            self._send_json({"rewards": data})
        elif path == "/api/replays":
            replays = self.recorder.list_replays() if self.recorder else []
            self._send_json({"replays": replays})
        elif path == "/api/abtest":
            summary = self.ab_test.get_summary() if self.ab_test else {}
            self._send_json(summary)
        elif path == "/api/recovery":
            # Phase 9: Detailed recovery stats
            recovery_data = {}
            if self.bridge and hasattr(self, 'server') and self.server:
                w = getattr(self.server, '_wrapper_ref', None)
                if w:
                    er = getattr(w, 'error_recovery', None)
                    if er:
                        recovery_data["error_recovery"] = er.get_stats()
                    sr = getattr(w, 'state_recovery', None)
                    if sr:
                        recovery_data["state_recovery"] = sr.get_recovery_status()
                    ac = getattr(w, 'auto_calibrator', None)
                    if ac:
                        recovery_data["auto_calibrator"] = {
                            "cache_size": len(ac.get_all_cached_coords()),
                            "templates_loaded": len(ac.templates) if hasattr(ac, 'templates') else 0,
                        }
                    ocr = getattr(w, 'ocr_detector', None)
                    if ocr:
                        recovery_data["ocr"] = ocr.get_detection_stats()
            self._send_json(recovery_data)
        # Premium API endpoints
        elif path == "/api/brawlers":
            if self.bridge:
                data = self.bridge.get_snapshot()
                self._send_json({
                    "brawlers": data.get("brawler_stats", []),
                    "total_trophies": data.get("total_trophies", 0),
                    "unlocked_brawlers": data.get("unlocked_brawlers", 0),
                    "total_brawlers": data.get("total_brawlers", 80),
                })
            else:
                self._send_json({"brawlers": [], "total_trophies": 0})
        elif path == "/api/match-analysis":
            if self.bridge:
                data = self.bridge.get_snapshot()
                self._send_json({
                    "recent_matches": data.get("recent_matches", []),
                    "match_analysis": data.get("match_analysis"),
                    "coach_tips": data.get("coach_tips", []),
                })
            else:
                self._send_json({"recent_matches": [], "coach_tips": []})
        elif path == "/api/ai-pick":
            if self.bridge:
                data = self.bridge.get_snapshot()
                self._send_json({
                    "suggestion": data.get("ai_pick_suggestion"),
                    "win_prediction": data.get("win_prediction", 0.0),
                    "current_brawler": data.get("brawler"),
                    "map": data.get("map_name"),
                })
            else:
                self._send_json({"suggestion": None, "win_prediction": 0.0})
        elif path == "/api/trophy-history":
            if self.bridge:
                data = self.bridge.get_snapshot()
                self._send_json({
                    "history": data.get("trophy_history", []),
                    "daily_evolution": data.get("daily_evolution", []),
                })
            else:
                self._send_json({"history": [], "daily_evolution": []})
        elif path == "/api/weekly-progress":
            if self.bridge:
                data = self.bridge.get_snapshot()
                self._send_json(data.get("weekly_progress") or {
                    "trophies_change": 0, "matches": 0,
                    "best_brawler": None, "winrate_change": 0,
                })
            else:
                self._send_json({"trophies_change": 0, "matches": 0})
        elif path == "/api/bot/status":
            # Bot control: get current bot status
            w = self.wrapper_ref
            if w:
                status = {
                    "running": getattr(w, 'running', False),
                    "paused": getattr(w, '_paused', False),
                    "current_state": w.state_manager.current_state if w.state_manager else "unknown",
                    "current_brawler": getattr(w.state_manager, 'current_brawler', None) if w.state_manager else None,
                    "current_map": getattr(w.state_manager, '_current_map', None) if w.state_manager else None,
                    "matches_played": getattr(w, 'matches_played', 0),
                    "brawler_queue": [b.name for b in w.brawler_queue.brawlers] if hasattr(w, 'brawler_queue') and w.brawler_queue else [],
                    "session_duration": time.time() - w.session_start if hasattr(w, 'session_start') and w.session_start else 0,
                }
                try:
                    if hasattr(w, 'safety') and w.safety and hasattr(w.safety, 'get_status'):
                        status["safety"] = w.safety.get_status()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] State recovery stats unavailable: {e}")
                try:
                    if hasattr(w, 'error_recovery') and w.error_recovery and hasattr(w.error_recovery, 'get_stats'):
                        status["error_recovery"] = w.error_recovery.get_stats()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] State recovery stats unavailable: {e}")
                # Phase 1: System status
                try:
                    if hasattr(w, 'get_system_status'):
                        status["systems"] = w.get_system_status().get("systems", {})
                except Exception as e:
                    logger.debug(f"[DASHBOARD] State recovery stats unavailable: {e}")
                self._send_json(status)
            else:
                self._send_json({"running": False, "error": "Bot not connected to dashboard"})
        elif path == "/api/system/status":
            w = self.wrapper_ref
            if w and hasattr(w, 'get_system_status'):
                self._send_json(w.get_system_status())
            else:
                # Return default system status when bot is not connected
                self._send_json({
                    "paused": False,
                    "running": False,
                    "systems": {
                        "rl_engine": {"enabled": False, "available": False},
                        "humanization": {"enabled": False, "available": False},
                        "anti_ban": {"enabled": False, "available": False},
                        "error_recovery": {"enabled": False, "available": False},
                        "recording": {"enabled": False, "available": False},
                        "auto_tuner": {"enabled": False, "available": False},
                        "data_collector": {"enabled": False, "available": False},
                    }
                })
        elif path == "/api/bot/queue":
            w = self.wrapper_ref
            if w and w.brawler_queue:
                try:
                    queue = []
                    for i, b in enumerate(w.brawler_queue.brawlers):
                        queue.append({
                            "index": i,
                            "name": b.name,
                            "current_trophies": getattr(b, 'current_trophies', 0),
                            "target_trophies": getattr(b, 'target_trophies', 350),
                            "priority": getattr(b, 'priority', 1),
                            "enabled": getattr(b, 'enabled', True),
                            "current": i == w.brawler_queue.current_index,
                        })
                    self._send_json({"queue": queue, "current_index": w.brawler_queue.current_index})
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
            else:
                self._send_json({"queue": [], "error": "Queue not available"})
        elif path == "/api/notifications/config":
            if self.notification_manager:
                self._send_json(self.notification_manager.get_config())
            else:
                self._send_json({"error": "NotificationManager not available"}, 500)
        elif path == "/api/notifications/history":
            if self.notification_manager:
                self._send_json({"history": self.notification_manager.get_history()})
            else:
                self._send_json({"history": []})
        elif path == "/api/config":
            # Phase 4: Return current config.json
            config_path = Path("config.json")
            if config_path.exists():
                try:
                    with open(config_path, encoding="utf-8") as f:
                        data = json.load(f)
                    self._send_json(data)
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
            else:
                self._send_json({})
        elif path == "/api/antiban/status":
            # Phase 6: Anti-ban status from wrapper
            w = self.wrapper_ref
            if w and w.anti_ban:
                try:
                    status = w.anti_ban.get_status() if hasattr(w.anti_ban, 'get_status') else {}
                    self._send_json(status)
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
            else:
                self._send_json({"enabled": False, "error": "Anti-ban not available"})
        elif path == "/api/training/models":
            # Training & Models data: registry + dataset + last training report
            training_data = {"models": [], "dataset": {}, "last_training": None}
            try:
                # Model Registry
                from core.model_registry import ModelRegistry
                registry = ModelRegistry()
                models_list = []
                for name, versions in registry._models.items():
                    for ver_str, ver in versions.items():
                        models_list.append({
                            "name": name,
                            "version": ver_str,
                            "schema": ver.metadata.get("training_data", "—") if ver.metadata else "—",
                            "map50": ver.metrics.get("mAP50", 0.0),
                            "map50_95": ver.metrics.get("mAP50-95", 0.0),
                            "is_active": registry._active.get(name) == ver_str,
                            "created": ver.created_at,
                        })
                training_data["models"] = sorted(models_list, key=lambda x: x["created"], reverse=True)

                # Scan models/ directory if requested
                if "scan" in self.path:
                    models_dir = Path("models")
                    scanned = 0
                    if models_dir.exists():
                        for pt_file in models_dir.glob("*.pt"):
                            if pt_file.stem not in ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]:
                                try:
                                    registry.register("yolo_scanned", pt_file, version=None, metrics={})
                                    scanned += 1
                                except Exception:
                                    pass
                    training_data["scanned"] = scanned
            except Exception as e:
                logger.debug(f"[DASHBOARD] Training models error: {e}")

            # Dataset stats from last derived dataset
            try:
                datasets = sorted(Path("dataset").glob("roboflow_raw_v2_*"), key=lambda p: p.stat().st_mtime, reverse=True)
                if datasets:
                    ds = datasets[0]
                    total_imgs = 0
                    total_boxes = 0
                    class_dist = {}
                    for split in ("train", "val"):
                        lbl_dir = ds / split / "labels"
                        if lbl_dir.exists():
                            for lbl in lbl_dir.glob("*.txt"):
                                total_imgs += 1
                                for line in lbl.read_text(encoding="utf-8").splitlines():
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        try:
                                            cls_id = int(parts[0])
                                            total_boxes += 1
                                            class_dist[str(cls_id)] = class_dist.get(str(cls_id), 0) + 1
                                        except ValueError:
                                            pass
                    training_data["dataset"] = {
                        "total_images": total_imgs,
                        "total_boxes": total_boxes,
                        "num_classes": len(class_dist),
                        "class_distribution": dict(sorted(class_dist.items())),
                    }
            except Exception:
                pass

            # Last training report
            try:
                reports = sorted(Path("runs").glob("*/training_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if reports:
                    with open(reports[0], encoding="utf-8") as f:
                        import json as _json
                        report = _json.load(f)
                    training_data["last_training"] = {
                        "run_id": report.get("run_id"),
                        "timestamp": report.get("timestamp"),
                        "schema": report.get("config", {}).get("schema"),
                        "map50": report.get("validation_metrics", {}).get("mAP50"),
                        "map50_95": report.get("validation_metrics", {}).get("mAP50-95"),
                        "duration_seconds": report.get("duration_seconds"),
                    }
            except Exception:
                pass

            self._send_json(training_data)
        elif path == "/api/export/stats":
            # Export all stats as JSON
            w = self.wrapper_ref
            export_data = {}
            if self.bridge:
                export_data["live"] = self.bridge.get_snapshot()
                export_data["history"] = self.bridge.get_history()
            if w:
                export_data["bot_status"] = w.get_status() if hasattr(w, 'get_status') else {}
                if w.brawler_queue:
                    export_data["queue"] = w.get_queue() if hasattr(w, 'get_queue') else []
            self._send_json(export_data)
        elif path == '/api/v2/status':
            w = self.wrapper_ref
            v2_data = {}
            if w and hasattr(w, 'v2_integrator') and w.v2_integrator:
                v2_data = w.v2_integrator.get_dashboard_data()
            self._send_json(v2_data)
        elif path == '/api/v2/degradation':
            w = self.wrapper_ref
            deg = {}
            if w and hasattr(w, 'v2_integrator') and w.v2_integrator and w.v2_integrator._degradation_mgr:
                deg = w.v2_integrator._degradation_mgr.get_status()
            self._send_json(deg)
        elif path == '/api/v2/alerts':
            w = self.wrapper_ref
            alerts = []
            if w and hasattr(w, 'v2_integrator') and w.v2_integrator and w.v2_integrator._alert_system:
                alerts = w.v2_integrator._alert_system.get_active_alerts()
            self._send_json({'alerts': alerts})
        elif path == '/api/v2/rate-limiter':
            w = self.wrapper_ref
            rl = {}
            if w and hasattr(w, 'v2_integrator') and w.v2_integrator and w.v2_integrator._rate_limiter:
                rl = w.v2_integrator._rate_limiter.get_account_status(w.v2_integrator.config.account_id)
            self._send_json(rl)
        elif path == '/api/v2/checkpoints':
            w = self.wrapper_ref
            cp = {}
            if w and hasattr(w, 'v2_integrator') and w.v2_integrator and w.v2_integrator._checkpointer:
                cp = w.v2_integrator._checkpointer.get_stats()
            self._send_json(cp)
        # Learning Mode endpoints
        elif path == '/api/learning-mode/status':
            w = self.wrapper_ref
            data = {"active": False, "metrics": {}}
            if w and hasattr(w, 'learning_mode_controller') and w.learning_mode_controller:
                try:
                    data = w.learning_mode_controller.get_live_metrics()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Learning mode metrics error: {e}")
            self._send_json(data)
        elif path == '/api/learning-mode/history':
            w = self.wrapper_ref
            history = []
            if w and hasattr(w, 'learning_mode_controller') and w.learning_mode_controller and w.learning_mode_controller.metrics:
                try:
                    history = w.learning_mode_controller.metrics.get_session_history()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Learning mode history error: {e}")
            self._send_json({"matches": history})
        elif path == '/api/esp/frame':
            # Retorna screenshot anotado + detecoes JSON
            w = self.wrapper_ref
            data = {"detections": [], "vision_stats": {}, "screenshot_b64": ""}
            if w and hasattr(w, 'get_detection_snapshot'):
                try:
                    data = w.get_detection_snapshot()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] ESP frame error: {e}")
            self._send_json(data)
        elif path == '/api/vision/stats':
            w = self.wrapper_ref
            stats = {}
            if w and hasattr(w, 'get_detection_snapshot'):
                try:
                    stats = w.get_detection_snapshot().get("vision_stats", {})
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Vision stats error: {e}")
            self._send_json(stats)
        elif path == '/api/detections/live':
            w = self.wrapper_ref
            detections = []
            if w and hasattr(w, 'get_detection_snapshot'):
                try:
                    detections = w.get_detection_snapshot().get("detections", [])
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Detections live error: {e}")
            self._send_json({"detections": detections, "timestamp": time.time()})
        elif path == '/api/mode/status':
            w = self.wrapper_ref
            status = {"available": False}
            if w and hasattr(w, 'get_mode_status'):
                try:
                    status = w.get_mode_status()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Mode status error: {e}")
            self._send_json(status)
        elif path == '/api/rl/metrics':
            w = self.wrapper_ref
            metrics = {}
            if w and hasattr(w, 'get_rl_metrics'):
                try:
                    metrics = w.get_rl_metrics()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] RL metrics error: {e}")
            self._send_json(metrics)
        elif path == '/api/logs':
            # Phase 2: Log viewer endpoint
            if self.log_buffer:
                import urllib.parse
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                limit = int(query.get("limit", ["100"])[0])
                level = query.get("level", ["ALL"])[0]
                component = query.get("component", ["ALL"])[0]
                search = query.get("search", [""])[0]
                lines = self.log_buffer.get_lines(limit=limit, level=level if level != "ALL" else None,
                                                  component=component if component != "ALL" else None,
                                                  search=search if search else None)
                self._send_json({"lines": lines, "stats": self.log_buffer.get_stats()})
            else:
                self._send_json({"lines": [], "stats": {"total": 0}, "error": "LogBuffer not available"})
        elif path == "/api/logs/stream":
            # Phase 2: Server-Sent Events for real-time logs
            if self.log_buffer:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                import threading
                listener = threading.Event()
                self.log_buffer.add_listener(listener)
                len(self.log_buffer.get_lines(limit=1))
                try:
                    while True:
                        # FIX #12: Add safety timeout to prevent infinite loop
                        if listener.wait(timeout=5.0):
                            listener.clear()
                        else:
                            # Timeout - check if server is shutting down
                            if hasattr(self.server, '_shutdown') and self.server._shutdown:
                                break
                            continue
                        lines = self.log_buffer.get_lines(limit=50)
                        if lines:
                            data = json.dumps({"lines": lines}, ensure_ascii=False)
                            self.wfile.write(f"data: {data}\n\n".encode())
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    logger.debug('[DASHBOARD] SSE client disconnected')
                except Exception as e:
                    logger.debug(f"[DASHBOARD] SSE stream error: {e}")
                finally:
                    self.log_buffer.remove_listener(listener)
            else:
                self._send_json({"error": "LogBuffer not available"}, 500)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}

        if path == "/api/abtest/start":
            variants = payload.get("variants", {"control": {}, "test": {}})
            if self.ab_test:
                self.ab_test.define_variants(variants)
                self.ab_test.start_test()
            self._send_json({"ok": True, "status": "started"})
        elif path == "/api/abtest/stop":
            if self.ab_test:
                self.ab_test.stop_test()
            self._send_json({"ok": True, "status": "stopped"})
        elif path == "/api/replay/start":
            name = payload.get("name")
            if self.recorder:
                self.recorder.start(name)
            self._send_json({"ok": True})
        elif path == "/api/replay/stop":
            if self.recorder:
                self.recorder.stop()
            self._send_json({"ok": True})
        # Premium: Record match result
        elif path == "/api/match/record":
            if self.bridge:
                brawler = payload.get("brawler", "colt")
                map_name = payload.get("map", "unknown")
                result = payload.get("result", "loss")
                kills = payload.get("kills", 0)
                deaths = payload.get("deaths", 0)
                duration = payload.get("duration", 0.0)
                self.bridge.brawler_tracker.record_match(
                    brawler, map_name, result, kills, deaths, duration,
                    payload.get("gadget", ""), payload.get("star_power", "")
                )
                # Record trophy snapshot
                total = self.bridge.brawler_tracker.get_total_trophies()
                self.bridge.trophy_tracker.record(total, brawler, 0)
                # Run match analysis
                analysis = self.bridge.match_analyzer.analyze_match(
                    brawler, map_name, result, kills, deaths,
                    payload.get("enemies", []), duration
                )
                # Update live data
                self.bridge.update(
                    recent_matches=self.bridge.match_analyzer.analyze_match(
                        brawler, map_name, result, kills, deaths,
                        payload.get("enemies", []), duration
                    ),
                    match_analysis=analysis,
                )
                self._send_json({"ok": True, "analysis": analysis})
            else:
                self._send_json({"error": "no bridge"}, 500)
        # Bot control endpoints
        elif path == "/api/bot/start":
            w = self.wrapper_ref
            if w:
                try:
                    if not getattr(w, 'running', False):
                        setup_ok = True
                        if hasattr(w, 'setup'):
                            setup_ok = w.setup()
                        if not setup_ok:
                            self._send_json({"ok": False, "error": "Setup failed - verify emulator, window focus and config"})
                            return
                        started = w.start()
                        if not started:
                            self._send_json({"ok": False, "error": "Failed to start bot - safety, anti-ban or runtime block"})
                            return
                        self._send_json({"ok": True, "status": "started"})
                    else:
                        self._send_json({"ok": True, "status": "already_running"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/stop":
            w = self.wrapper_ref
            if w:
                try:
                    if getattr(w, 'running', False):
                        w.stop()
                        self._send_json({"ok": True, "status": "stopped"})
                    else:
                        self._send_json({"ok": True, "status": "already_stopped"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/restart":
            w = self.wrapper_ref
            if w:
                try:
                    if getattr(w, 'running', False):
                        w.stop()
                        time.sleep(2)
                    w.start()
                    self._send_json({"ok": True, "status": "restarted"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/farm/start":
            w = self.wrapper_ref
            if w and hasattr(w, '_run_farm_loop'):
                try:
                    farm_cfg = payload.get("config", {})
                    if farm_cfg:
                        for bcfg in farm_cfg.get("brawlers", []):
                            w.brawler_queue.add_brawler(BrawlerConfig(
                                name=bcfg.get("name", "colt"),
                                target_trophies=bcfg.get("target_trophies", 500),
                                priority=bcfg.get("priority", 1),
                                enabled=bcfg.get("enabled", True),
                            ))
                    if not getattr(w, 'running', False):
                        w.start()
                    self._send_json({"ok": True, "status": "farm_started"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected or farm mode unavailable"})
        # Phase 1: Bot control endpoints
        elif path == "/api/bot/pause":
            w = self.wrapper_ref
            if w:
                try:
                    result = w.pause()
                    self._send_json({"ok": result, "status": "paused" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/resume":
            w = self.wrapper_ref
            if w:
                try:
                    result = w.resume()
                    self._send_json({"ok": result, "status": "resumed" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/queue/update":
            w = self.wrapper_ref
            if w:
                try:
                    queue_data = payload.get("queue", [])
                    result = w.update_queue(queue_data)
                    self._send_json({"ok": result, "queue": [b.name for b in w.brawler_queue.brawlers] if w.brawler_queue else []})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/queue/set-brawler":
            w = self.wrapper_ref
            if w:
                try:
                    name = payload.get("name", "")
                    result = w.set_brawler(name)
                    self._send_json({"ok": result, "current_brawler": w.brawler_queue.get_current().name if w.brawler_queue and w.brawler_queue.get_current() else None})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/queue/add":
            w = self.wrapper_ref
            if w:
                try:
                    name = payload.get("name", "colt")
                    current = payload.get("current_trophies", 0)
                    target = payload.get("target_trophies", 350)
                    priority = payload.get("priority", 1)
                    w.add_brawler_to_queue(name, current, target, priority=priority)
                    self._send_json({"ok": True, "queue": [b.name for b in w.brawler_queue.brawlers] if w.brawler_queue else []})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/queue/remove":
            w = self.wrapper_ref
            if w and w.brawler_queue:
                try:
                    index = payload.get("index", -1)
                    if 0 <= index < len(w.brawler_queue.brawlers):
                        removed = w.brawler_queue.brawlers.pop(index)
                        self._send_json({"ok": True, "removed": removed.name, "queue": [b.name for b in w.brawler_queue.brawlers]})
                    else:
                        self._send_json({"ok": False, "error": "Invalid index"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/config/update":
            w = self.wrapper_ref
            if w:
                try:
                    key = payload.get("key", "")
                    value = payload.get("value")
                    if key:
                        result = w.update_config(key, value)
                        self._send_json({"ok": result})
                    else:
                        self._send_json({"ok": False, "error": "Missing key"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/action":
            w = self.wrapper_ref
            if w:
                try:
                    action = payload.get("action", "")
                    if action:
                        result = w.execute_action(action, **payload.get("params", {}))
                        self._send_json({"ok": result, "action": action})
                    else:
                        self._send_json({"ok": False, "error": "Missing action"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/combat/param":
            w = self.wrapper_ref
            if w and w.play_logic:
                try:
                    param = payload.get("param", "")
                    value = payload.get("value")
                    if param == "aggressiveness":
                        w.play_logic.aggressiveness = float(value)
                        self._send_json({"ok": True, "param": param, "value": w.play_logic.aggressiveness})
                    elif param == "shot_cooldown":
                        w.play_logic.shot_cooldown = float(value) / 1000.0
                        self._send_json({"ok": True, "param": param, "value": w.play_logic.shot_cooldown})
                    elif param == "attack_distance":
                        w.play_logic.attack_distance = int(value)
                        self._send_json({"ok": True, "param": param, "value": w.play_logic.attack_distance})
                    else:
                        self._send_json({"ok": False, "error": f"Unknown param: {param}"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "PlayLogic not available"})
        elif path == "/api/system/toggle":
            w = self.wrapper_ref
            if w:
                try:
                    system = payload.get("system", "")
                    enabled = payload.get("enabled", True)
                    if system:
                        result = w.toggle_system(system, enabled)
                        self._send_json({"ok": result, "system": system, "enabled": enabled})
                    else:
                        self._send_json({"ok": False, "error": "Missing system name"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/notifications/config":
            if self.notification_manager:
                try:
                    self.notification_manager.update_config(**payload)
                    self._send_json({"ok": True, "config": self.notification_manager.get_config()})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "NotificationManager not available"}, 500)
        elif path == "/api/notifications/test":
            if self.notification_manager:
                try:
                    title = payload.get("title", "Teste")
                    message = payload.get("message", "Notificacao de teste do Soberana Omega")
                    self.notification_manager.send(title, message, "info", "test")
                    self._send_json({"ok": True})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "NotificationManager not available"}, 500)
        elif path == "/api/learning-mode/start":
            w = self.wrapper_ref
            if w and hasattr(w, 'toggle_learning_mode'):
                try:
                    max_matches = payload.get("max_matches", 5)
                    result = w.toggle_learning_mode(enabled=True, max_matches=max_matches)
                    self._send_json({"ok": result, "status": "started" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Bot not connected or learning mode unavailable"}, 500)
        elif path == "/api/learning-mode/stop":
            w = self.wrapper_ref
            if w and hasattr(w, 'toggle_learning_mode'):
                try:
                    result = w.toggle_learning_mode(enabled=False)
                    self._send_json({"ok": result, "status": "stopped" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Bot not connected or learning mode unavailable"}, 500)
        elif path == "/api/esp/toggle":
            w = self.wrapper_ref
            if w and hasattr(w, 'toggle_esp'):
                try:
                    enabled = payload.get("enabled", None)
                    result = w.toggle_esp(enabled)
                    self._send_json({"ok": result, "status": "on" if result else "off"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "ESP not available"}, 500)
        elif path == "/api/mode/training/start":
            w = self.wrapper_ref
            if w and hasattr(w, 'start_mode'):
                try:
                    config = payload.get("config", {})
                    result = w.start_mode("training", config)
                    self._send_json({"ok": result, "status": "started" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Mode controller not available"}, 500)
        elif path == "/api/mode/training/stop":
            w = self.wrapper_ref
            if w and hasattr(w, 'stop_mode'):
                try:
                    result = w.stop_mode("training")
                    self._send_json({"ok": result, "status": "stopped" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Mode controller not available"}, 500)
        elif path == "/api/mode/farm/start":
            w = self.wrapper_ref
            if w and hasattr(w, 'start_mode'):
                try:
                    config = payload.get("config", {})
                    result = w.start_mode("farm", config)
                    self._send_json({"ok": result, "status": "started" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Mode controller not available"}, 500)
        elif path == "/api/mode/farm/stop":
            w = self.wrapper_ref
            if w and hasattr(w, 'stop_mode'):
                try:
                    result = w.stop_mode("farm")
                    self._send_json({"ok": result, "status": "stopped" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Mode controller not available"}, 500)
        elif path == "/api/mode/learn/start":
            w = self.wrapper_ref
            if w and hasattr(w, 'start_mode'):
                try:
                    config = payload.get("config", {})
                    result = w.start_mode("learn", config)
                    self._send_json({"ok": result, "status": "started" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Mode controller not available"}, 500)
        elif path == "/api/mode/learn/stop":
            w = self.wrapper_ref
            if w and hasattr(w, 'stop_mode'):
                try:
                    result = w.stop_mode("learn")
                    self._send_json({"ok": result, "status": "stopped" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Mode controller not available"}, 500)
        elif path == "/api/rl/config":
            # Atualiza hiperparametros RL (placeholder — engine ajusta dinamicamente)
            self._send_json({"ok": True, "message": "RL config update not yet implemented"})
        elif path == "/api/config":
            # Phase 4: Save config.json
            try:
                config_path = Path("config.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                # Also update wrapper central_config
                w = self.wrapper_ref
                if w:
                    w.central_config.update(payload)
                self._send_json({"ok": True})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
        else:
            self._send_json({"error": "not found"}, 404)


# ---------------------------------------------------------------------------
# DASHBOARD HTML (embedded — sem ficheiros externos)
# ---------------------------------------------------------------------------

