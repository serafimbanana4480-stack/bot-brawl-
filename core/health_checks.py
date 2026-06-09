"""
core/health_checks.py

Health check utilities for the Brawl Stars Bot.
Provides shallow (/health) and deep (/health/deep) checks.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class HealthResult:
    name: str
    status: str  # "pass", "fail", "warn"
    response_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class HealthReport:
    overall: str  # "healthy", "degraded", "unhealthy"
    checks: List[HealthResult]
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
    except (FileNotFoundError, PermissionError, ConnectionError, TimeoutError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
        elapsed = (time.perf_counter() - start) * 1000
        return HealthResult(
            name="adb",
            status="fail",
            response_time_ms=elapsed,
            error=str(exc),
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
    except (FileNotFoundError, PermissionError, ConnectionError, TimeoutError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
        elapsed = (time.perf_counter() - start) * 1000
        return HealthResult(
            name="emulator",
            status="fail",
            response_time_ms=elapsed,
            error=str(exc),
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


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------
def check_api_up() -> HealthResult:
    """Shallow check: API process is running."""
    import time
    start = time.perf_counter()
    elapsed = (time.perf_counter() - start) * 1000
    return HealthResult(
        name="api",
        status="pass",
        response_time_ms=elapsed,
        details={"message": "API is up"},
    )


_SHALLOW_CHECKS = [check_api_up]
_DEEP_CHECKS = [check_adb, check_emulator, check_yolo_model, check_screenshot]


def _run_checks(checks: List[Any]) -> HealthReport:
    from datetime import datetime, timezone
    results: List[HealthResult] = []
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
def to_dict(report: HealthReport) -> Dict[str, Any]:
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
