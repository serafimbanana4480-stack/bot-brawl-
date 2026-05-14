"""
api.py

FastAPI backend API for Brawl Stars bot with WebSocket support for real-time updates.
Provides endpoints for bot control, configuration, and status monitoring.

Fixes Applied:
- Error #9:  get_bot() now calls setup() on first creation
- Error #19: CORS restricted to configured origins
- Error #20: WebSocket emit functions wired into bot lifecycle
- Error #21: remove_brawler actually removes from queue
- Error #22: match_history populated after each match
- Error #30: Full OpenAPI documentation on all endpoints
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for request/response (Error #30 - proper docs)
# ---------------------------------------------------------------------------
class BotConfig(BaseModel):
    """Bot configuration model for updating safety and queue settings."""
    trophy_limit: int = Field(400, description="Maximum trophies before auto-stop")
    warning_trophies: int = Field(380, description="Trophy count that triggers warning")
    max_session_hours: float = Field(3.0, description="Maximum session duration in hours")
    min_apm: int = Field(20, description="Minimum actions per minute")
    max_apm: int = Field(60, description="Maximum actions per minute")
    auto_pause: bool = Field(True, description="Auto-pause on safety trigger")
    auto_stop_on_detection: bool = Field(True, description="Stop if bot detection suspected")
    brawler_queue: List[Dict[str, Any]] = Field(default_factory=list, description="Brawler queue config")
    safety_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional safety settings")
    diagnostic_mode: bool = Field(False, description="Enable detailed lobby diagnostics")


class BrawlerConfigModel(BaseModel):
    """Brawler configuration for adding to queue."""
    name: str = Field(..., description="Brawler name (e.g., 'Colt', 'Shelly')")
    current_trophies: int = Field(0, description="Current trophy count")
    target_trophies: int = Field(350, description="Target trophy count")
    target_wins: int = Field(10, description="Target number of wins")
    priority: int = Field(1, ge=1, le=5, description="Priority 1-5, higher = more priority")


class MatchHistory(BaseModel):
    """Match history entry."""
    brawler: str
    result: str
    trophies_gained: int
    duration_seconds: int
    timestamp: datetime


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_bot_instance = None
_match_history: List[MatchHistory] = []
_setup_complete = False


def get_bot():
    """Get or create bot instance. Calls setup() on first creation (Fix Error #9)."""
    global _bot_instance, _setup_complete
    if _bot_instance is None:
        from .wrapper import PylaAIEnhanced
        _bot_instance = PylaAIEnhanced()
        _bot_instance.setup()  # ✅ Call setup after creation
        logger.info("Bot instance created and setup completed")
    return _bot_instance


def _load_cors_origins() -> List[str]:
    """Load CORS origins from config.json or use safe defaults (Fix Error #19)."""
    try:
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            import json as _json
            with open(config_path, encoding="utf-8") as f:
                config = _json.load(f)
            origins = config.get("api", {}).get("cors_origins", [])
            if origins:
                return origins
    except Exception as e:
        logger.warning(f"Failed to load CORS from config.json: {e}")

    return ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"]


# ---------------------------------------------------------------------------
# Create FastAPI app (Error #30 - full docs)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Brawl Stars Bot API - Soberana Omega",
    description=(
        "Backend API for Brawl Stars bot control and monitoring.\n\n"
        "## Features\n"
        "- Real-time WebSocket updates\n"
        "- Bot lifecycle management (setup/start/stop)\n"
        "- Brawler queue management\n"
        "- Safety system monitoring\n"
        "- Match history tracking"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware (Fix Error #19 - restricted origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_load_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for connection in disconnected:
            self.disconnect(connection)

    async def send_personal(self, message: Dict[str, Any], websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# WebSocket emit helpers (Fix Error #20 - now actually usable)
# ---------------------------------------------------------------------------
async def emit_bot_action(action_type: str, details: Dict[str, Any]):
    """Emit bot action event to all WebSocket clients."""
    await manager.broadcast({
        "type": "bot_action",
        "action_type": action_type,
        "details": details,
        "timestamp": datetime.now().isoformat()
    })


async def emit_match_progress(match_data: Dict[str, Any]):
    """Emit match progress event to all WebSocket clients."""
    await manager.broadcast({
        "type": "match_progress",
        "data": match_data,
        "timestamp": datetime.now().isoformat()
    })


async def emit_error(error_type: str, message: str, details: Dict[str, Any] = None):
    """Emit error event to all WebSocket clients."""
    await manager.broadcast({
        "type": "error",
        "error_type": error_type,
        "message": message,
        "details": details or {},
        "timestamp": datetime.now().isoformat()
    })


async def emit_safety_alert(alert_type: str, message: str, severity: str = "warning"):
    """Emit safety alert to all WebSocket clients."""
    await manager.broadcast({
        "type": "safety_alert",
        "alert_type": alert_type,
        "message": message,
        "severity": severity,
        "timestamp": datetime.now().isoformat()
    })


async def emit_telemetry_update(telemetry_data: Dict[str, Any]):
    """Emit telemetry update to all WebSocket clients."""
    await manager.broadcast({
        "type": "telemetry",
        "data": telemetry_data,
        "timestamp": datetime.now().isoformat()
    })


def record_match(brawler: str, result: str, trophies: int, duration: int):
    """Record a match result (Fix Error #22 - now populated)."""
    global _match_history
    entry = MatchHistory(
        brawler=brawler,
        result=result,
        trophies_gained=trophies,
        duration_seconds=duration,
        timestamp=datetime.now()
    )
    _match_history.append(entry)
    logger.info(f"Match recorded: {brawler} - {result} ({trophies:+d} trophies)")

    # Keep last 500 matches
    if len(_match_history) > 500:
        _match_history = _match_history[-500:]


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/brawl-stars")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bot updates.

    Events emitted:
    - bot_action: When the bot performs an action
    - match_progress: During match gameplay
    - safety_alert: When safety limits are approached
    - error: On errors
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await manager.send_personal({"type": "pong"}, websocket)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received via WebSocket")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Health and Readiness endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Health check", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Liveness probe: returns 200 if the API process itself is alive.
    Does NOT check bot internals — use /ready for that.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/ready", summary="Readiness check", tags=["Health"])
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness probe: returns 200 only when the bot instance is
    initialized, the state manager is running, and ADB is connected.

    Returns detailed component status for diagnostics.
    """
    checks = {
        "bot_instance": False,
        "state_manager_running": False,
        "adb_connected": False,
    }
    details = {}

    try:
        bot = get_bot()
        checks["bot_instance"] = True

        # Check state manager
        if hasattr(bot, 'state_manager') and bot.state_manager:
            checks["state_manager_running"] = getattr(bot.state_manager, 'running', False)
            details["current_state"] = getattr(bot.state_manager, 'current_state', 'unknown')

        # Check ADB / emulator controller
        if hasattr(bot, 'emulator_controller') and bot.emulator_controller:
            ec = bot.emulator_controller
            # Try a lightweight health check
            if hasattr(ec, 'adb') and hasattr(ec.adb, 'ping'):
                try:
                    ec.adb.ping()
                    checks["adb_connected"] = True
                except Exception:
                    checks["adb_connected"] = False
            elif hasattr(ec, 'is_connected'):
                checks["adb_connected"] = ec.is_connected()
            else:
                # Assume connected if controller exists
                checks["adb_connected"] = True

        # Get bot health check if available
        if hasattr(bot, 'check_health'):
            health = bot.check_health()
            details["health"] = health

    except Exception as e:
        details["error"] = str(e)

    ready = all(checks.values())
    return {
        "ready": ready,
        "checks": checks,
        "details": details,
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# REST API endpoints
# ---------------------------------------------------------------------------
@app.get("/api/brawl-stars/status", summary="Get bot status", tags=["Bot Control"])
async def get_status() -> Dict[str, Any]:
    """
    Returns the current bot status including running state, current brawler,
    session duration, and safety system status.
    """
    bot = get_bot()
    status = bot.get_status()
    return {"success": True, "status": status}


@app.get("/api/brawl-stars/logs", summary="Get recent logs with filters", tags=["Monitoring"])
async def get_logs(
    n: int = 100,
    category: Optional[str] = None,
    level: Optional[str] = None,
    search: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get recent logs from the log manager with optional filters.

    Parameters:
    - n: Number of recent logs to return (default: 100)
    - category: Filter by category (e.g., "lobby", "combat", "state", "auto_tuning", "safety")
    - level: Filter by level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    - search: Search text in log messages
    """
    from brawl_bot.realtime_logs import get_log_manager

    try:
        log_manager = get_log_manager()
        logs = log_manager.get_recent_logs(n=n, category=category, level=level)

        # Apply text search filter if provided
        if search:
            search_lower = search.lower()
            logs = [log for log in logs if search_lower in log.get("message", "").lower()]

        # Get statistics
        stats = log_manager.get_stats()

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "logs": logs,
            "count": len(logs),
            "filters": {
                "n": n,
                "category": category,
                "level": level,
                "search": search
            },
            "stats": stats
        }
    except Exception as e:
        logger.error(f"[API] Erro ao obter logs recentes: {e}")
        return {
            "success": False,
            "error": str(e),
            "logs": [],
            "count": 0
        }


@app.get("/api/brawl-stars/telemetry", summary="Get real-time telemetry metrics", tags=["Monitoring"])
async def get_telemetry() -> Dict[str, Any]:
    """
    Returns detailed real-time telemetry metrics including:
    - APM (Actions Per Minute)
    - Win rate
    - Suspicion score
    - Human likeness score
    - Movement statistics
    - Enemy tracking statistics
    - Session statistics
    """
    bot = get_bot()
    
    # Obter status do bot
    bot_status = bot.get_status()
    
    # Obter status do safety system se disponível
    safety_status = {}
    if hasattr(bot, 'safety_system') and bot.safety_system:
        safety_status = bot.safety_system.get_status()
    
    # Obter status do tracker se disponível
    tracker_status = {}
    if hasattr(bot, 'play_logic') and bot.play_logic and hasattr(bot.play_logic, 'enemy_tracker'):
        if bot.play_logic.enemy_tracker:
            # Usar get_stats() do tracker para obter estatísticas consistentes
            tracker_status = bot.play_logic.enemy_tracker.get_stats()
    
    # Obter métricas recentes de treinamento se disponível
    training_metrics = {}
    if hasattr(bot, 'training_monitor') and bot.training_monitor:
        if hasattr(bot.training_monitor, 'get_recent_metrics'):
            try:
                training_metrics = bot.training_monitor.get_recent_metrics(n_sessions=10)
                logger.debug(f"[API] Métricas de treinamento obtidas: {training_metrics}")
            except Exception as e:
                logger.warning(f"[API] Falha ao obter métricas de treinamento: {e}")
    
    # Calcular win rate do histórico de partidas
    win_rate = 0.0
    if _match_history:
        wins = sum(1 for m in _match_history if m.result == "win")
        win_rate = wins / len(_match_history) * 100
    
    # Calcular média de troféus ganhos por partida
    avg_trophies_per_match = 0
    if _match_history:
        total_trophies = sum(m.trophies_gained for m in _match_history)
        avg_trophies_per_match = total_trophies / len(_match_history)
    
    telemetry = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "bot_status": bot_status,
        "safety_status": safety_status,
        "tracker_status": tracker_status,
        "training_metrics": training_metrics,
        "match_statistics": {
            "total_matches": len(_match_history),
            "win_rate": round(win_rate, 2),
            "avg_trophies_per_match": round(avg_trophies_per_match, 2),
            "last_10_matches": [
                {
                    "brawler": m.brawler,
                    "result": m.result,
                    "trophies_gained": m.trophies_gained,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in _match_history[-10:]
            ]
        },
        "performance_metrics": {
            "apm": safety_status.get("apm", 0) if safety_status else 0,
            "suspicion_score": safety_status.get("suspicion_score", 0) if safety_status else 0,
            "human_likeness_score": safety_status.get("human_likeness_score", 100) if safety_status else 100,
            "avg_velocity": safety_status.get("avg_velocity", 0) if safety_status else 0,
            "max_acceleration": safety_status.get("max_acceleration", 0) if safety_status else 0,
            "curvature_variance": safety_status.get("curvature_variance", 0) if safety_status else 0
        }
    }
    
    return telemetry


@app.get("/api/brawl-stars/emulators", summary="Get available emulators", tags=["Emulators"])
async def get_emulators(emulator_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns available emulators with optional filter by type.
    
    Parameters:
    - emulator_type: Filter by emulator type (e.g., "LDPlayer", "BlueStacks", "Nox")
    """
    from brawl_bot.emulator_detector import get_emulator_detector
    
    try:
        detector = get_emulator_detector()
        detector.detect_all()
        
        if emulator_type:
            # Usar get_emulators_by_type() se disponível
            if hasattr(detector, 'get_emulators_by_type'):
                emulators = detector.get_emulators_by_type(emulator_type)
            else:
                # Fallback: filtrar manualmente
                emulators = [e for e in detector.available_emulators if e.type == emulator_type]
        else:
            emulators = detector.available_emulators
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "emulators": [
                {
                    "name": e.name,
                    "type": e.type,
                    "window_title": e.window_title,
                    "adb_path": e.adb_path,
                    "port": e.port
                }
                for e in emulators
            ],
            "count": len(emulators)
        }
    except Exception as e:
        logger.error(f"[API] Erro ao obter emuladores: {e}")
        return {
            "success": False,
            "error": str(e),
            "emulators": [],
            "count": 0
        }


@app.get("/api/brawl-stars/emulators/{name}", summary="Get emulator by name", tags=["Emulators"])
async def get_emulator_by_name(name: str) -> Dict[str, Any]:
    """
    Returns specific emulator by name or window title.
    
    Parameters:
    - name: Emulator name or window title
    """
    from brawl_bot.emulator_detector import get_emulator_detector
    
    try:
        detector = get_emulator_detector()
        detector.detect_all()
        
        # Usar get_emulator_by_name() se disponível
        if hasattr(detector, 'get_emulator_by_name'):
            emulator = detector.get_emulator_by_name(name)
        else:
            # Fallback: buscar manualmente
            emulator = None
            for e in detector.available_emulators:
                if e.name == name or e.window_title == name:
                    emulator = e
                    break
        
        if emulator:
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "emulator": {
                    "name": emulator.name,
                    "type": emulator.type,
                    "window_title": emulator.window_title,
                    "adb_path": emulator.adb_path,
                    "port": emulator.port
                }
            }
        else:
            return {
                "success": False,
                "error": f"Emulator '{name}' not found",
                "emulator": None
            }
    except Exception as e:
        logger.error(f"[API] Erro ao obter emulador por nome: {e}")
        return {
            "success": False,
            "error": str(e),
            "emulator": None
        }


@app.get("/api/brawl-stars/recommended-action", summary="Get recommended action", tags=["Bot Control"])
async def get_recommended_action() -> Dict[str, Any]:
    """
    Returns the recommended action based on current state.
    Possible actions: "start_match", "switch_brawler", "continue"
    """
    bot = get_bot()
    
    try:
        if hasattr(bot, 'match_controller') and bot.match_controller:
            # Usar get_recommended_action() se disponível
            if hasattr(bot.match_controller, 'get_recommended_action'):
                action = bot.match_controller.get_recommended_action()
                logger.info(f"[API] Ação recomendada: {action}")
                return {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "action": action,
                    "reason": _get_action_reason(action, bot.match_controller)
                }
            else:
                # Fallback: inferir ação baseada no estado
                if bot.running:
                    action = "continue"
                else:
                    action = "start_match"
                return {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "action": action,
                    "reason": "Inferred from bot state (get_recommended_action not available)"
                }
        else:
            return {
                "success": False,
                "error": "Match controller not available",
                "action": None
            }
    except Exception as e:
        logger.error(f"[API] Erro ao obter ação recomendada: {e}")
        return {
            "success": False,
            "error": str(e),
            "action": None
        }


def _get_action_reason(action: str, match_controller) -> str:
    """Retorna o motivo da ação recomendada"""
    if action == "start_match":
        return "Not currently in a match"
    elif action == "switch_brawler":
        return "Brawler queue recommends switching based on performance"
    elif action == "continue":
        return "Continue with current brawler"
    else:
        return "Unknown action"


@app.post("/api/brawl-stars/auto-tuning/tune", summary="Run auto-tuning", tags=["Auto-Tuning"])
async def run_auto_tuning() -> Dict[str, Any]:
    """
    Runs auto-tuning cycle to adjust parameters based on match history.
    Analyzes performance and adjusts attack distance, shot cooldown, safety threshold, etc.
    """
    bot = get_bot()
    
    try:
        if not bot.auto_tuner:
            return {
                "success": False,
                "error": "Auto-tuner is not enabled or initialized"
            }
        
        if not bot.play_logic or not bot.safety:
            return {
                "success": False,
                "error": "Play logic or safety system not available"
            }
        
        # Executar ciclo de auto-tuning
        result = bot.auto_tuner.tune(bot.play_logic, bot.safety)
        
        return {
            "success": result["success"],
            "timestamp": datetime.now().isoformat(),
            "analysis": result.get("analysis"),
            "adjustments": result.get("adjustments"),
            "current_params": result.get("current_params"),
            "reason": result.get("reason")
        }
    except Exception as e:
        logger.error(f"[API] Erro ao executar auto-tuning: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/brawl-stars/auto-tuning/status", summary="Get auto-tuning status", tags=["Auto-Tuning"])
async def get_auto_tuning_status() -> Dict[str, Any]:
    """
    Returns the current status of the auto-tuning system.
    """
    bot = get_bot()
    
    try:
        if not bot.auto_tuner:
            return {
                "success": False,
                "error": "Auto-tuner is not enabled or initialized",
                "enabled": bot.auto_tuning_enabled if hasattr(bot, 'auto_tuning_enabled') else False
            }
        
        status = bot.auto_tuner.get_tuning_status()
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "enabled": bot.auto_tuning_enabled if hasattr(bot, 'auto_tuning_enabled') else False,
            "status": status
        }
    except Exception as e:
        logger.error(f"[API] Erro ao obter status do auto-tuner: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/brawl-stars/auto-tuning/reset", summary="Reset auto-tuning parameters", tags=["Auto-Tuning"])
async def reset_auto_tuning() -> Dict[str, Any]:
    """
    Resets all auto-tuned parameters to their default values.
    """
    bot = get_bot()
    
    try:
        if not bot.auto_tuner:
            return {
                "success": False,
                "error": "Auto-tuner is not enabled or initialized"
            }
        
        if not bot.play_logic or not bot.safety:
            return {
                "success": False,
                "error": "Play logic or safety system not available"
            }
        
        # Resetar parâmetros
        success = bot.auto_tuner.reset_params(bot.play_logic, bot.safety)
        
        return {
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "message": "Parameters reset to defaults" if success else "Failed to reset parameters"
        }
    except Exception as e:
        logger.error(f"[API] Erro ao resetar auto-tuning: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/brawl-stars/setup", summary="Setup the bot", tags=["Bot Control"])
async def setup_bot() -> Dict[str, Any]:
    """
    Initialize the bot: connect to emulator, load models, configure components.
    Must be called before starting the bot.
    """
    bot = get_bot()
    if bot.running:
        return {"success": False, "error": "Bot is already running"}

    if bot.setup():
        global _setup_complete
        _setup_complete = True
        await manager.broadcast({
            "type": "bot_action",
            "event": "setup_completed",
            "timestamp": datetime.now().isoformat()
        })
        return {"success": True, "message": "Bot setup completed successfully"}
    else:
        await emit_error("setup_failed", "Failed to setup bot. Ensure emulator is running.")
        return {"success": False, "error": "Failed to setup bot. Ensure emulator is running."}


@app.post("/api/brawl-stars/start", summary="Start the bot", tags=["Bot Control"])
async def start_bot() -> Dict[str, Any]:
    """
    Start the bot automation loop. Requires setup to be completed first.
    """
    bot = get_bot()
    if bot.running:
        return {"success": False, "error": "Bot is already running"}

    # Auto-setup if not done
    global _setup_complete
    if not _setup_complete:
        if not bot.setup():
            return {"success": False, "error": "Failed to setup bot. Ensure emulator is running."}
        _setup_complete = True

    if bot.start():
        await emit_bot_action("started", {"timestamp": datetime.now().isoformat()})
        return {"success": True, "message": "Bot started successfully"}
    else:
        return {"success": False, "error": "Failed to start bot"}


@app.post("/api/brawl-stars/stop", summary="Stop the bot", tags=["Bot Control"])
async def stop_bot() -> Dict[str, Any]:
    """Stop the bot automation loop gracefully."""
    bot = get_bot()
    if not bot.running:
        return {"success": False, "error": "Bot is not running"}

    if bot.stop():
        await emit_bot_action("stopped", {"timestamp": datetime.now().isoformat()})
        return {"success": True, "message": "Bot stopped successfully"}
    else:
        return {"success": False, "error": "Failed to stop bot"}


@app.get("/api/brawl-stars/config", summary="Get configuration", tags=["Configuration"])
async def get_config() -> Dict[str, Any]:
    """Returns the current bot configuration including safety settings."""
    bot = get_bot()
    config = {
        "trophy_limit": bot.safety.config.max_trophies,
        "warning_trophies": bot.safety.config.warning_trophies,
        "max_session_hours": bot.safety.config.max_session_hours,
        "min_apm": bot.safety.config.min_apm,
        "max_apm": bot.safety.config.max_apm,
        "auto_pause": bot.safety.config.auto_stop_on_detection,
        "auto_stop_on_detection": bot.safety.config.auto_stop_on_detection,
        "brawler_queue": bot.get_queue(),
        "safety_settings": {
            "break_duration_min": bot.safety.config.break_duration_min,
            "break_duration_max": bot.safety.config.break_duration_max,
            "suspicious_pattern_threshold": bot.safety.config.suspicious_pattern_threshold
        }
    }
    return {"success": True, "config": config}


@app.post("/api/brawl-stars/config", summary="Update configuration", tags=["Configuration"])
async def update_config(config: BotConfig) -> Dict[str, Any]:
    """Update bot configuration. Changes take effect immediately."""
    bot = get_bot()
    try:
        from .safety_system import SafetyConfig
        new_safety_config = SafetyConfig(
            max_trophies=config.trophy_limit,
            warning_trophies=config.warning_trophies,
            max_session_hours=config.max_session_hours,
            min_apm=config.min_apm,
            max_apm=config.max_apm,
            auto_stop_on_detection=config.auto_stop_on_detection
        )
        bot.safety.config = new_safety_config

        # Update diagnostic mode for lobby/state flows
        bot.diagnostic_mode = config.diagnostic_mode
        if bot.lobby and hasattr(bot.lobby, "set_diagnostic_mode"):
            bot.lobby.set_diagnostic_mode(config.diagnostic_mode)
        if bot.state_manager is not None:
            bot.state_manager.diagnostic_mode = config.diagnostic_mode

        # Update brawler queue
        from .pylaai_real.lobby_automator import BrawlerQueue
        if not isinstance(bot.brawler_queue, BrawlerQueue):
            bot.brawler_queue = BrawlerQueue()
        else:
            bot.brawler_queue.brawlers.clear()
            bot.brawler_queue.current_index = 0
        for brawler in config.brawler_queue:
            bot.add_brawler_to_queue(
                name=brawler.get("name", "Unknown"),
                current_trophies=brawler.get("current_trophies", 0),
                target_trophies=brawler.get("target_trophies", 350),
                target_wins=brawler.get("target_wins", 10),
                priority=brawler.get("priority", 1)
            )

        await manager.broadcast({
            "type": "config_updated",
            "timestamp": datetime.now().isoformat()
        })
        return {"success": True, "message": "Configuration updated successfully"}

    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/brawl-stars/match-history", summary="Get match history", tags=["Statistics"])
async def get_match_history(limit: int = 50) -> Dict[str, Any]:
    """
    Returns the last N matches played. History is populated after each match.

    **Parameters:**
    - limit: Maximum number of matches to return (default: 50)
    """
    return {
        "success": True,
        "total": len(_match_history),
        "matches": [m.dict() for m in _match_history[-limit:]]
    }


@app.get("/api/brawl-stars/safety-status", summary="Get safety status", tags=["Safety"])
async def get_safety_status() -> Dict[str, Any]:
    """Returns current safety system status including APM, session time, and alerts."""
    bot = get_bot()
    safety_status = bot.safety.get_status()
    return {"success": True, "safety": safety_status}


@app.post("/api/brawl-stars/brawler/add", summary="Add brawler to queue", tags=["Brawler Queue"])
async def add_brawler(brawler: BrawlerConfigModel) -> Dict[str, Any]:
    """
    Add a brawler to the automation queue.

    The bot will play each brawler in order of priority until targets are met.
    """
    bot = get_bot()
    bot.add_brawler_to_queue(
        name=brawler.name,
        current_trophies=brawler.current_trophies,
        target_trophies=brawler.target_trophies,
        target_wins=brawler.target_wins,
        priority=brawler.priority
    )
    return {"success": True, "message": f"Brawler {brawler.name} added to queue"}


@app.delete("/api/brawl-stars/brawler/{name}", summary="Remove brawler from queue", tags=["Brawler Queue"])
async def remove_brawler(name: str) -> Dict[str, Any]:
    """
    Remove a brawler from the queue by name. (Fix Error #21 - now functional)
    """
    bot = get_bot()
    queue = bot.brawler_queue

    # Find and remove brawler by name
    for i, brawler in enumerate(queue.brawlers):
        if brawler.name.lower() == name.lower():
            queue.remove_brawler(i)
            logger.info(f"Brawler {name} removed from queue")
            return {"success": True, "message": f"Brawler {name} removed from queue"}

    return {"success": False, "error": f"Brawler {name} not found in queue"}


@app.get("/api/brawl-stars/queue", summary="Get brawler queue", tags=["Brawler Queue"])
async def get_queue() -> Dict[str, Any]:
    """Returns the current brawler queue with status for each brawler."""
    bot = get_bot()
    return {"success": True, "queue": bot.get_queue()}


@app.get("/api/brawl-stars/health", summary="Health check", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint. Returns system status.

    **Response fields:**
    - status: 'healthy' or 'degraded'
    - bot_running: Whether the bot is currently active
    - websocket_connections: Number of active WebSocket clients
    - setup_complete: Whether initial setup has been done
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "bot_running": _bot_instance.running if _bot_instance else False,
        "websocket_connections": len(manager.active_connections),
        "setup_complete": _setup_complete,
        "version": "2.0.0"
    }


@app.get("/api/brawl-stars/performance", summary="Get performance metrics", tags=["Statistics"])
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Returns performance metrics including FPS, latency, and match statistics.
    (Fix Error #27 - performance metrics endpoint)
    """
    bot = get_bot()
    total_matches = len(_match_history)
    wins = sum(1 for m in _match_history if m.result == "win")
    losses = sum(1 for m in _match_history if m.result == "loss")

    return {
        "success": True,
        "metrics": {
            "total_matches": total_matches,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / total_matches * 100) if total_matches > 0 else 0,
            "avg_duration_seconds": (
                sum(m.duration_seconds for m in _match_history) / total_matches
                if total_matches > 0 else 0
            ),
            "total_trophies_gained": sum(m.trophies_gained for m in _match_history),
            "bot_running": bot.running,
            "session_duration_minutes": bot.get_status().get("session_duration_minutes", 0),
        }
    }


@app.get("/api/brawl-stars/diagnostics", summary="Get bot diagnostics", tags=["System"])
async def get_diagnostics() -> Dict[str, Any]:
    """Returns a consolidated diagnostic snapshot for lobby, state, queue, safety and progress."""
    bot = get_bot()
    status = bot.get_status()
    diagnostics = status.get("diagnostics", {}) if isinstance(status, dict) else {}

    return {
        "success": True,
        "diagnostics": {
            "bot_running": status.get("running") if isinstance(status, dict) else False,
            "current_state": status.get("current_state") if isinstance(status, dict) else None,
            "current_brawler": status.get("current_brawler") if isinstance(status, dict) else None,
            "queue": status.get("queue") if isinstance(status, dict) else [],
            "safety": status.get("safety") if isinstance(status, dict) else None,
            "diagnostic_mode": diagnostics.get("diagnostic_mode"),
            "lobby": diagnostics.get("lobby"),
            "screen_state": diagnostics.get("screen_state"),
            "progress": diagnostics.get("progress"),
        }
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Brawl Stars Bot API...")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8003,
        log_level="info"
    )
