"""
core/health_checks.py

Health check utilities for the Brawl Stars Bot.
Provides shallow (/health) and deep (/health/deep) checks.
"""

import os
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
_DISK_WARN_THRESHOLD = float(os.getenv("HEALTH_DISK_THRESHOLD", "80"))
_MEM_WARN_THRESHOLD = float(os.getenv("HEALTH_MEM_THRESHOLD", "80"))

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class HealthResult:
    name: str
    status: str  # "pass", "fail", "warn"
    response_time_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class HealthReport:
    overall: str  # "healthy", "degraded", "unhealthy"
    checks: list[HealthResult]
    timestamp: str


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_adb() -> HealthResult:
    """Verify ADB binary is available and responds to version."""
    import time
    start = time.perf_counter()
    try:
        adb_path = shutil.which("adb") or os.getenv("ADB_PATH", "adb")
        result = subprocess.run(
            [adb_path, "version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        elapsed = (time.perf_counter() - start) * 1000
        return HealthResult(
            name="adb",
            status="pass",
            response_time_ms=elapsed,
            details={"version_output": result.stdout.strip()[:200]},
        )
    except (FileNotFoundError, PermissionError, ConnectionError, TimeoutError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
        elapsed = (time.perf_counter() - start) * 1000
        return HealthResult(
            name="adb",
            status="fail",
            response_time_ms=elapsed,
            error=str(e),
        )


def check_emulator() -> HealthResult:
    """Verify at least one Android emulator is connected via ADB."""
    import time
    start = time.perf_counter()
    try:
        adb_path = shutil.which("adb") or os.getenv("ADB_PATH", "adb")
        result = subprocess.run(
            [adb_path, "devices", "-l"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        lines = [ln for ln in result.stdout.splitlines() if ln.strip() and not ln.startswith("*")]
        # Header line + at least one device line
        devices = [ln for ln in lines[1:] if "device" in ln]
        elapsed = (time.perf_counter() - start) * 1000
        if devices:
            return HealthResult(
                name="emulator",
                status="pass",
                response_time_ms=elapsed,
                details={"device_count": len(devices), "devices": devices[:3]},
            )
        return HealthResult(
            name="emulator",
            status="fail",
            response_time_ms=elapsed,
            error="No emulator connected via ADB",
        )
    except (FileNotFoundError, PermissionError, ConnectionError, TimeoutError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
        elapsed = (time.perf_counter() - start) * 1000
        return HealthResult(
            name="emulator",
            status="fail",
            response_time_ms=elapsed,
            error=str(e),
        )


def check_yolo_model() -> HealthResult:
    """Verify the main YOLO model file exists and is loadable."""
    import time
    start = time.perf_counter()
    try:
        model_path = os.getenv("YOLO_MODEL_PATH", "models/yolo/brawlstars_yolov8.pt")
        p = Path(model_path)
        if not p.exists():
            return HealthResult(
                name="yolo_model",
                status="fail",
                error=f"Model file not found: {model_path}",
            )
        # Check that ultralytics is importable
        try:
            from ultralytics import YOLO  # noqa: F401
        except Exception as exc:
            return HealthResult(
                name="yolo_model",
                status="fail",
                error=f"ultralytics not importable: {exc}",
            )
        # Try a lightweight load check (Ultralytics loads lazily; we at least verify file)
        size_mb = p.stat().st_size / (1024 * 1024)
        elapsed = (time.perf_counter() - start) * 1000
        return HealthResult(
            name="yolo_model",
            status="pass",
            response_time_ms=elapsed,
            details={"path": str(p.resolve()), "size_mb": round(size_mb, 2)},
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return HealthResult(
            name="yolo_model",
            status="fail",
            response_time_ms=elapsed,
            error=str(exc),
        )


def check_screenshot() -> HealthResult:
    """Verify screenshot pipeline can capture a frame."""
    import time
    start = time.perf_counter()
    try:
        from pylaai_real.screenshot_taker import ScreenshotTaker
        taker = ScreenshotTaker()
        frame = taker.take()
        elapsed = (time.perf_counter() - start) * 1000
        if frame is not None:
            return HealthResult(
                name="screenshot",
                status="pass",
                response_time_ms=elapsed,
                details={"shape": getattr(frame, "shape", None)},
            )
        return HealthResult(
            name="screenshot",
            status="warn",
            response_time_ms=elapsed,
            error="Screenshot returned None (emulator may be off or window not found)",
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return HealthResult(
            name="screenshot",
            status="fail",
            response_time_ms=elapsed,
            error=str(exc),
        )


def check_memory() -> HealthResult:
    """Check system memory usage is below threshold."""
    import time
    start = time.perf_counter()
    try:
        import psutil
        mem = psutil.virtual_memory()
        elapsed = (time.perf_counter() - start) * 1000
        usage_pct = mem.percent
        status = "pass" if usage_pct < _MEM_WARN_THRESHOLD else "warn" if usage_pct < 95 else "fail"
        return HealthResult(
            name="memory",
            status=status,
            response_time_ms=elapsed,
            details={"percent": usage_pct, "available_mb": round(mem.available / (1024 * 1024), 2)},
            error=f"Memory usage {usage_pct}% exceeds {_MEM_WARN_THRESHOLD}%" if status != "pass" else None,
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return HealthResult(
            name="memory",
            status="warn",
            response_time_ms=elapsed,
            error=f"psutil not available: {exc}",
        )


def check_disk() -> HealthResult:
    """Check disk usage is below threshold."""
    import time
    start = time.perf_counter()
    try:
        import psutil
        disk = psutil.disk_usage("/")
        elapsed = (time.perf_counter() - start) * 1000
        usage_pct = disk.percent
        status = "pass" if usage_pct < _DISK_WARN_THRESHOLD else "warn" if usage_pct < 95 else "fail"
        return HealthResult(
            name="disk",
            status=status,
            response_time_ms=elapsed,
            details={"percent": usage_pct, "free_gb": round(disk.free / (1024 ** 3), 2)},
            error=f"Disk usage {usage_pct}% exceeds {_DISK_WARN_THRESHOLD}%" if status != "pass" else None,
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return HealthResult(
            name="disk",
            status="warn",
            response_time_ms=elapsed,
            error=f"psutil not available: {exc}",
        )


def check_api_up() -> HealthResult:
    """Check API process is running and responsive."""
    import time
    start = time.perf_counter()
    try:
        port = int(os.getenv("BRAWL_BOT_API_PORT", "8003"))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            result = sock.connect_ex(("127.0.0.1", port))
        elapsed = (time.perf_counter() - start) * 1000
        if result == 0:
            return HealthResult(
                name="api",
                status="pass",
                response_time_ms=elapsed,
                details={"port": port, "message": "API is up"},
            )
        return HealthResult(
            name="api",
            status="fail",
            response_time_ms=elapsed,
            error=f"API not listening on port {port}",
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return HealthResult(
            name="api",
            status="warn",
            response_time_ms=elapsed,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------
_SHALLOW_CHECKS = [check_api_up, check_memory, check_disk]
_DEEP_CHECKS = [check_adb, check_emulator, check_yolo_model, check_screenshot, check_memory, check_disk]


def _run_checks(checks: list[Any]) -> HealthReport:
    results: list[HealthResult] = []
    for fn in checks:
        try:
            results.append(fn())
        except Exception as exc:
            results.append(
                HealthResult(name=fn.__name__, status="fail", error=str(exc))
            )
    # Overall status logic
    statuses = {r.status for r in results}
    if "fail" in statuses:
        overall = "unhealthy"
    elif "warn" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"
    return HealthReport(
        overall=overall,
        checks=results,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def health_shallow() -> HealthReport:
    """Run lightweight health checks (no emulator required)."""
    return _run_checks(_SHALLOW_CHECKS)


def health_deep() -> HealthReport:
    """Run deep health checks (requires emulator + screenshot)."""
    return _run_checks(_DEEP_CHECKS)


# ---------------------------------------------------------------------------
# FastAPI integration helpers
# ---------------------------------------------------------------------------
def to_dict(report: HealthReport) -> dict[str, Any]:
    return {
        "status": report.overall,
        "timestamp": report.timestamp,
        "checks": [
            {
                "name": c.name,
                "status": c.status,
                "response_time_ms": round(c.response_time_ms, 2),
                "details": c.details,
                "error": c.error,
            }
            for c in report.checks
        ],
    }
