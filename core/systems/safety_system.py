"""
core/systems/safety_system.py

Encapsulates all safety & anti-detection subsystems:
- SafetySystem (trophies, APM, session limits)
- HumanizationEngine (delays, Bezier curves)
- AntiBanSystem / AdvancedAntiBanSystem
- ErrorRecoverySystem, StateRecoverySystem, AutoFixEngine

Interface: init(), start(), stop(), status(), health_check()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from safety_system import SafetySystem, SafetyConfig
from humanization import HumanizationEngine, HumanizationConfig

logger = logging.getLogger(__name__)


class SafetySystemWrapper:
    """Cohesive safety subsystem with graceful degradation."""

    def __init__(
        self,
        safety_config: Optional[SafetyConfig] = None,
        humanization_config: Optional[HumanizationConfig] = None,
        central_config: Optional[Dict[str, Any]] = None,
        enable_error_recovery: bool = True,
    ):
        self.central_config = central_config or {}
        self.enable_error_recovery = enable_error_recovery

        # Required components (fatal if missing)
        self.safety: Optional[SafetySystem] = None
        self.humanization: Optional[HumanizationEngine] = None

        # Optional components
        self.anti_ban: Optional[Any] = None
        self.error_recovery: Optional[Any] = None
        self.recovery_integration: Optional[Any] = None
        self.state_recovery: Optional[Any] = None
        self.auto_fix: Optional[Any] = None
        self._running = False

        # Initialize required systems immediately
        try:
            self.safety = SafetySystem(safety_config)
            logger.info("[SAFETY] SafetySystem initialized")
        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            logger.error("[SAFETY] SafetySystem FATAL: %s", e)
            raise RuntimeError("SafetySystem failed to initialize - unsafe to continue") from e

        try:
            self.humanization = HumanizationEngine(humanization_config)
            logger.info("[SAFETY] HumanizationEngine initialized")
        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            logger.error("[SAFETY] HumanizationEngine FATAL: %s", e)
            raise RuntimeError("HumanizationEngine failed to initialize") from e

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, emulator_controller: Optional[Any] = None) -> bool:
        """Initialize optional safety components."""
        # Anti-ban (canonical: core.anti_ban.AntiBanSystem)
        try:
            from core.anti_ban import AntiBanSystem
            self.anti_ban = AntiBanSystem()
            logger.info("[SAFETY] AntiBanSystem initialized")
        except (ImportError, ModuleNotFoundError, TypeError) as e:
            logger.warning("[SAFETY] Anti-ban unavailable: %s", e)

        # Error Recovery
        if self.enable_error_recovery:
            try:
                from core.error_recovery import ErrorRecoverySystem, ErrorRecoveryIntegration
                self.error_recovery = ErrorRecoverySystem(
                    enable_auto_recovery=True,
                    max_recovery_attempts=3,
                    global_circuit_breaker=True,
                )
                logger.info("[SAFETY] ErrorRecoverySystem initialized")
            except (ImportError, ModuleNotFoundError, TypeError) as e:
                logger.warning("[SAFETY] ErrorRecoverySystem unavailable: %s", e)

        # State Recovery
        try:
            from pylaai_real.state_recovery import StateRecoverySystem
            if emulator_controller:
                self.state_recovery = StateRecoverySystem(
                    emulator_controller=emulator_controller,
                    max_unknown_duration=30.0,
                    max_loop_duration=15.0,
                    enable_auto_restart=False,
                )
                logger.info("[SAFETY] StateRecoverySystem initialized")
        except (ImportError, ModuleNotFoundError, TypeError) as e:
            logger.debug("[SAFETY] StateRecoverySystem unavailable: %s", e)

        # AutoFixEngine
        try:
            from core.auto_fix_engine import AutoFixEngine
            self.auto_fix = AutoFixEngine()
            logger.info("[SAFETY] AutoFixEngine initialized")
        except (ImportError, ModuleNotFoundError, TypeError) as e:
            logger.debug("[SAFETY] AutoFixEngine unavailable: %s", e)

        return True

    def start(self) -> bool:
        self._running = True
        if self.safety:
            try:
                self.safety.start_session()
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning("[SAFETY] Failed to start safety session: %s", e)
            except Exception as e:
                logger.exception("[SAFETY] Unexpected safety session start error: %s", e)
                raise
        return True

    def stop(self) -> bool:
        self._running = False
        if self.anti_ban and hasattr(self.anti_ban, "stop"):
            try:
                self.anti_ban.stop()
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug("[SAFETY] Failed to stop anti_ban: %s", e)
        if self.error_recovery and hasattr(self.error_recovery, "stop"):
            try:
                self.error_recovery.stop()
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug("[SAFETY] Failed to stop error_recovery: %s", e)
        return True

    # ------------------------------------------------------------------
    # Status / Health
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "safety_ok": self.safety is not None,
            "humanization_ok": self.humanization is not None,
            "anti_ban_ok": self.anti_ban is not None,
            "error_recovery_ok": self.error_recovery is not None,
            "state_recovery_ok": self.state_recovery is not None,
            "auto_fix_ok": self.auto_fix is not None,
        }

    def health_check(self) -> Dict[str, Any]:
        issues = []
        if self.safety is None:
            issues.append("no_safety")
        if self.humanization is None:
            issues.append("no_humanization")
        return {"healthy": len(issues) == 0, "issues": issues}
