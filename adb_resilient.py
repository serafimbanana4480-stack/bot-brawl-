"""
adb_resilient.py

Resilient ADB wrapper with retry logic, circuit breaker, and health checks.
Replaces raw subprocess calls throughout the bot for all ADB operations.

Features:
- Exponential backoff retry on transient failures
- Circuit breaker after consecutive failures
- Health check before every operation
- Command audit logging
- Timeout enforcement
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ADBConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    command_timeout: float = 15.0
    health_check_interval: float = 5.0


@dataclass
class ADBState:
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    total_commands: int = 0
    total_failures: int = 0
    last_health_check: float = 0.0
    last_health_ok: bool = False


class ResilientADB:
    """
    Resilient ADB command executor.

    Usage:
        adb = ResilientADB(adb_path="adb.exe", adb_id="127.0.0.1:5555")
        result = adb.run(["shell", "wm", "size"])
    """

    def __init__(self, adb_path: str, adb_id: str, config: Optional[ADBConfig] = None):
        self.adb_path = adb_path
        self.adb_id = adb_id
        self.config = config or ADBConfig()
        self.state = ADBState()

    def _is_circuit_open(self) -> bool:
        return time.time() < self.state.circuit_open_until

    def _open_circuit(self) -> None:
        self.state.circuit_open_until = time.time() + self.config.circuit_breaker_timeout
        logger.warning(
            f"CIRCUIT BREAKER OPEN for {self.adb_id} "
            f"({self.config.circuit_breaker_timeout}s)"
        )

    def _close_circuit(self) -> None:
        self.state.consecutive_failures = 0
        self.state.circuit_open_until = 0.0

    def _health_check(self) -> bool:
        """Quick ping to verify ADB connectivity."""
        now = time.time()
        if now - self.state.last_health_check < self.config.health_check_interval:
            return self.state.last_health_ok

        self.state.last_health_check = now
        try:
            result = subprocess.run(
                [self.adb_path, "-s", self.adb_id, "shell", "echo", "ping"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            ok = result.returncode == 0 and "ping" in result.stdout.lower()
            self.state.last_health_ok = ok
            if ok:
                self._close_circuit()
            return ok
        except Exception:
            self.state.last_health_ok = False
            return False

    def run(self, args: List[str], timeout: Optional[float] = None) -> Optional[subprocess.CompletedProcess]:
        """
        Execute ADB command with retry and circuit breaker.

        Args:
            args: ADB subcommand arguments (e.g., ["shell", "wm", "size"])
            timeout: Override default timeout

        Returns:
            CompletedProcess on success, None on failure (after retries)
        """
        if self._is_circuit_open():
            logger.warning(f"Circuit breaker open, skipping ADB command: {args}")
            return None

        if not self._health_check():
            self.state.consecutive_failures += 1
            if self.state.consecutive_failures >= self.config.circuit_breaker_threshold:
                self._open_circuit()
            logger.warning(f"ADB health check failed for {self.adb_id}")
            return None

        full_cmd = [self.adb_path, "-s", self.adb_id] + args
        timeout_val = timeout or self.config.command_timeout

        self.state.total_commands += 1

        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(f"ADB run (attempt {attempt + 1}): {' '.join(full_cmd)}")
                result = subprocess.run(
                    full_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_val,
                )

                if result.returncode == 0:
                    self.state.consecutive_failures = 0
                    return result

                # Non-zero return code — may be transient
                logger.warning(
                    f"ADB command failed (rc={result.returncode}): "
                    f"stderr={result.stderr[:200]}"
                )

            except subprocess.TimeoutExpired:
                logger.warning(f"ADB command timeout (>{timeout_val}s)")
            except Exception as e:
                logger.error(f"ADB command exception: {e}")

            if attempt < self.config.max_retries:
                delay = min(
                    self.config.base_delay * (2 ** attempt),
                    self.config.max_delay,
                )
                logger.info(f"Retrying ADB command in {delay:.1f}s...")
                time.sleep(delay)

        # All retries exhausted
        self.state.total_failures += 1
        self.state.consecutive_failures += 1

        if self.state.consecutive_failures >= self.config.circuit_breaker_threshold:
            self._open_circuit()

        logger.error(f"ADB command failed after {self.config.max_retries + 1} attempts: {args}")
        return None

    def screenshot(self) -> Optional[bytes]:
        """Capture screenshot with resilience."""
        result = self.run(["exec-out", "screencap", "-p"], timeout=20)
        if result is None:
            return None
        if len(result.stdout.encode("utf-8", errors="ignore")) < 1024:
            logger.error("Screenshot too small, likely corrupted")
            return None
        # result.stdout from text mode may be wrong for binary — re-run raw
        try:
            raw_result = subprocess.run(
                [self.adb_path, "-s", self.adb_id, "exec-out", "screencap", "-p"],
                capture_output=True,
                timeout=20,
            )
            if raw_result.returncode == 0 and len(raw_result.stdout) > 1024:
                return raw_result.stdout
        except Exception as e:
            logger.error(f"Raw screenshot capture failed: {e}")
        return None

    def tap(self, x: int, y: int) -> bool:
        """Tap screen at coordinates. Returns True on success."""
        result = self.run(["shell", "input", "tap", str(x), str(y)])
        return result is not None

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> bool:
        """Swipe from (x1,y1) to (x2,y2)."""
        result = self.run(
            ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)]
        )
        return result is not None

    def get_stats(self) -> dict:
        return {
            "adb_id": self.adb_id,
            "total_commands": self.state.total_commands,
            "total_failures": self.state.total_failures,
            "consecutive_failures": self.state.consecutive_failures,
            "circuit_open": self._is_circuit_open(),
            "circuit_open_until": self.state.circuit_open_until,
            "health_ok": self.state.last_health_ok,
        }
