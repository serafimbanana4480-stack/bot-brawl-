"""
brawl_stars_routes.py

Rotas de API para detecção e controlo de emuladores Brawl Stars.
Expõe funções assíncronas compatíveis com FastAPI e com os testes de integração.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Importações com fallback para ambientes sem todas as dependências
try:
    from emulator_detector import EmulatorInfo, get_emulator_detector
except ImportError:
    try:
        from brawl_bot.emulator_detector import EmulatorInfo, get_emulator_detector
    except ImportError:
        def get_emulator_detector():
            return None
        EmulatorInfo = None

try:
    from emulator_detector import get_adb_path
except ImportError:
    try:
        from brawl_bot.emulator_detector import get_adb_path
    except ImportError:
        def get_adb_path():
            return "adb"


def _emulator_to_dict(emulator) -> dict[str, Any]:
    """Converte um EmulatorInfo (ou objeto duck-type) para dicionário serializável."""
    return {
        "name": getattr(emulator, "name", "unknown"),
        "type": getattr(emulator, "type", "unknown"),
        "adb_id": getattr(emulator, "adb_id", None),
        "window_title": getattr(emulator, "window_title", None),
        "connected": getattr(emulator, "connected", False),
    }


def _get_diagnostics() -> dict[str, Any]:
    """Coleta informações de diagnóstico do ambiente."""
    adb_path = None
    try:
        adb_path = get_adb_path()
    except Exception:
        pass

    psutil_available = False
    try:
        import psutil  # noqa: F401
        psutil_available = True
    except ImportError:
        pass

    pywin32_available = False
    try:
        import win32gui  # noqa: F401
        pywin32_available = True
    except ImportError:
        pass

    # Tentar carregar último relatório de verificação
    last_report = None
    try:
        report_path = Path(__file__).parent.parent / "verify_installation_report.json"
        if report_path.exists():
            with open(report_path, encoding="utf-8") as f:
                last_report = json.load(f)
    except Exception:
        pass

    return {
        "adb_path": adb_path,
        "psutil_available": psutil_available,
        "pywin32_available": pywin32_available,
        "last_verify_installation_report": last_report,
    }


async def list_emulators() -> dict[str, Any]:
    """Lista todos os emuladores detectados com diagnóstico do ambiente."""
    detector = get_emulator_detector()
    emulators = []

    if detector is not None:
        try:
            detected = detector.detect_all()
            emulators = [_emulator_to_dict(e) for e in (detected or [])]
        except Exception as e:
            logger.error(f"Erro ao detectar emuladores: {e}")

    return {
        "emulators": emulators,
        "count": len(emulators),
        "diagnostics": _get_diagnostics(),
    }


async def connect_emulator(name: str) -> dict[str, Any]:
    """Conecta ao emulador com o nome especificado via ADB."""
    detector = get_emulator_detector()
    emulator = None

    if detector is not None:
        try:
            emulator = detector.get_emulator_by_name(name)
        except Exception as e:
            logger.error(f"Erro ao obter emulador '{name}': {e}")

    if emulator is None:
        return {
            "success": False,
            "error": f"Emulador '{name}' não encontrado",
            "message": f"Emulador '{name}' não encontrado",
            "adb": {
                "chosen_adb_path": None,
                "stdout": "",
                "stderr": "",
            },
        }

    adb_id = getattr(emulator, "adb_id", None)
    if not adb_id:
        return {
            "success": False,
            "error": "Emulador sem adb_id definido",
            "adb": {
                "chosen_adb_path": None,
                "stdout": "",
                "stderr": "",
            },
        }

    chosen_adb = get_adb_path()

    stdout_text = ""
    stderr_text = ""
    success = False

    try:
        result = subprocess.run(
            [chosen_adb, "connect", adb_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        stdout_text = result.stdout or ""
        stderr_text = result.stderr or ""
        success = result.returncode == 0 and "connected" in stdout_text.lower()
    except Exception as e:
        stderr_text = str(e)

    return {
        "success": success,
        "adb": {
            "chosen_adb_path": chosen_adb,
            "stdout": stdout_text,
            "stderr": stderr_text,
        },
    }


# ---------------------------------------------------------------------------
# Bot lifecycle management
# ---------------------------------------------------------------------------

# Module-level bot singleton (replaced by tests via monkeypatch)
_bot_instance = None


def _get_bot():
    """Return the current bot instance, creating a default one if needed."""
    global _bot_instance
    if _bot_instance is None:
        try:
            from wrapper import PylaAIEnhanced
            _bot_instance = PylaAIEnhanced()
        except Exception as exc:
            logger.error(f"Could not create PylaAIEnhanced: {exc}")
    return _bot_instance


async def setup_bot() -> dict[str, Any]:
    """Set up the bot (load models, detect emulator, etc.)."""
    bot = _get_bot()
    if bot is None:
        return {"success": False, "error": "Bot instance not available"}
    try:
        ok = bot.setup()
        return {"success": bool(ok)}
    except Exception as exc:
        logger.error(f"setup_bot error: {exc}")
        return {"success": False, "error": str(exc)}


async def start_bot() -> dict[str, Any]:
    """Start the bot's main loop."""
    bot = _get_bot()
    if bot is None:
        return {"success": False, "error": "Bot instance not available"}
    try:
        ok = bot.start()
        return {"success": bool(ok)}
    except Exception as exc:
        logger.error(f"start_bot error: {exc}")
        return {"success": False, "error": str(exc)}


async def stop_bot() -> dict[str, Any]:
    """Stop the bot's main loop."""
    bot = _get_bot()
    if bot is None:
        return {"success": False, "error": "Bot instance not available"}
    try:
        ok = bot.stop()
        return {"success": bool(ok)}
    except Exception as exc:
        logger.error(f"stop_bot error: {exc}")
        return {"success": False, "error": str(exc)}


async def get_status() -> dict[str, Any]:
    """Return the current bot status dict."""
    bot = _get_bot()
    if bot is None:
        return {"running": False, "error": "Bot instance not available"}
    try:
        return bot.get_status()
    except Exception as exc:
        logger.error(f"get_status error: {exc}")
        return {"running": False, "error": str(exc)}


async def get_diagnostics() -> dict[str, Any]:
    """Return a consolidated diagnostics snapshot."""
    bot = _get_bot()
    if bot is None:
        return {"success": False, "error": "Bot instance not available"}
    try:
        status = bot.get_status()
        diagnostics = status.get("diagnostics", {}) or {}
        return {
            "success": True,
            "diagnostics": {
                "bot_running": status.get("running", False),
                "current_state": status.get("current_state"),
                "current_brawler": status.get("current_brawler"),
                "diagnostic_mode": diagnostics.get("diagnostic_mode", False),
                "lobby": diagnostics.get("lobby"),
                "screen_state": diagnostics.get("screen_state"),
                "progress": diagnostics.get("progress"),
                "combat": diagnostics.get("combat"),
                "match": diagnostics.get("match"),
            },
        }
    except Exception as exc:
        logger.error(f"get_diagnostics error: {exc}")
        return {"success": False, "error": str(exc)}
