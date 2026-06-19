"""
core/subsystems/safety_subsystem.py

SafetySubsystem: safety system, anti-ban, humanization, error recovery,
state recovery, and health monitoring.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wrapper import PylaAIEnhanced

logger = logging.getLogger(__name__)


class SafetySubsystem:
    """Manages safety, anti-ban, humanization, error recovery, and health."""

    HEARTBEAT_TIMEOUT = 30.0
    MAX_UNKNOWN_STATE_DURATION = 60.0

    def __init__(
        self,
        wrapper: PylaAIEnhanced,
        safety_config: Any,
        humanization_config: Any,
        enable_error_recovery: bool,
    ):
        self.wrapper = wrapper
        self.safety_config = safety_config
        self.humanization_config = humanization_config
        self.enable_error_recovery = enable_error_recovery
        self.safety: Any | None = None
        self.humanization: Any | None = None
        self.anti_ban: Any | None = None
        self.error_recovery: Any | None = None
        self.recovery_integration: Any | None = None
        self.state_recovery: Any | None = None
        self._last_action_time = time.time()
        self._health_lock = threading.Lock()
        self._shutdown_hooks: list = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> bool:
        """Initialize safety, anti-ban, recovery, and health systems."""
        from humanization import HumanizationEngine
        from safety_system import SafetySystem

        logger.debug("[WRAPPER] Inicializando SafetySystem")
        try:
            self.safety = SafetySystem(self.safety_config)
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.error(f"[WRAPPER] SafetySystem FATAL: {e}")
            raise RuntimeError("SafetySystem failed to initialize - unsafe to continue") from e

        logger.debug("[WRAPPER] Inicializando HumanizationEngine")
        try:
            self.humanization = HumanizationEngine(self.humanization_config)
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.error(f"[WRAPPER] HumanizationEngine FATAL: {e}")
            raise RuntimeError("HumanizationEngine failed to initialize - anti-ban protection unavailable") from e

        # Anti-ban — prefer core.anti_ban (novo, melhorado), fallback para legacy
        try:
            from core.anti_ban import AntiBanSystem

            self.anti_ban = AntiBanSystem()
            logger.info("[WRAPPER] Anti-ban system (core) inicializado")
        except (ImportError, ModuleNotFoundError):
            try:
                from pylaai_real.anti_ban_advanced import AdvancedAntiBanSystem

                self.anti_ban = AdvancedAntiBanSystem({"enabled": True})
                logger.info("[WRAPPER] Advanced anti-ban system (legacy fallback) inicializado")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"[WRAPPER] AntiBanSystem indisponível (não instalado): {e}")
            except Exception as e:
                logger.error(f"[WRAPPER] AntiBanSystem ERRO: {e}")

        # Error Recovery
        if self.enable_error_recovery:
            try:
                from core.error_recovery import ErrorRecoverySystem

                self.error_recovery = ErrorRecoverySystem(
                    enable_auto_recovery=True,
                    max_recovery_attempts=3,
                    global_circuit_breaker=True,
                )
                logger.info("[WRAPPER] Error Recovery System inicializado")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"[WRAPPER] Error Recovery System indisponível: {e}")
                self.enable_error_recovery = False

        # State Recovery (deferred - needs emulator_controller)
        self.state_recovery = None

        # Sync back
        self.wrapper.safety = self.safety
        self.wrapper.humanization = self.humanization
        self.wrapper.anti_ban = self.anti_ban
        self.wrapper.error_recovery = self.error_recovery
        self.wrapper.recovery_integration = self.recovery_integration
        self.wrapper.state_recovery = self.state_recovery
        self.wrapper._last_action_time = self._last_action_time
        self.wrapper._health_lock = self._health_lock
        self.wrapper._shutdown_hooks = self._shutdown_hooks
        return True

    def post_emulator_setup(self) -> None:
        """Initialize state recovery after emulator is available."""
        emulator_controller = getattr(self.wrapper, "emulator_controller", None)
        if emulator_controller:
            try:
                from pylaai_real.state_recovery import StateRecoverySystem

                self.state_recovery = StateRecoverySystem(
                    emulator_controller=emulator_controller,
                    max_unknown_duration=30.0,
                    max_loop_duration=15.0,
                    enable_auto_restart=False,
                )
                logger.info("[WRAPPER] State Recovery System inicializado")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"[WRAPPER] State Recovery System indisponível: {e}")

        if self.enable_error_recovery and self.error_recovery:
            try:
                from core.error_recovery import ErrorRecoveryIntegration

                self.recovery_integration = ErrorRecoveryIntegration(self.wrapper)
                self.recovery_integration.enable()
                logger.info("[WRAPPER] Error Recovery Integration configurado")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"[WRAPPER] Error Recovery Integration indisponível: {e}")

        self.wrapper.state_recovery = self.state_recovery
        self.wrapper.recovery_integration = self.recovery_integration

    def start(self) -> None:
        pass

    def stop(self) -> None:
        if self.anti_ban:
            try:
                if hasattr(self.anti_ban, "stop"):
                    self.anti_ban.stop()
                logger.info("[CLEANUP] Anti-ban system parado")
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug(f"[CLEANUP] Falha ao parar anti_ban: {e}")
        if self.state_recovery:
            try:
                self.state_recovery.cancel_recovery()
                logger.info("[CLEANUP] State Recovery cancelado")
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
                logger.debug(f"[CLEANUP] Falha ao cancelar state recovery: {e}")

    def cleanup(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Monitor loop (extracted from wrapper.py facade)
    # ------------------------------------------------------------------

    def run_monitor_loop(self, wrapper: PylaAIEnhanced, stop_event: threading.Event) -> None:
        """Main safety / anti-ban / recovery / health monitor loop."""
        logger.info("[WRAPPER] Monitor loop started")
        _health_check_counter = 0
        _health_check_interval = 10  # a cada 10 ciclos (~10-15s)
        while not stop_event.is_set() and wrapper.running:
            cycle_start = time.time()
            try:
                # V2 cycle start
                if wrapper.v2_integrator:
                    try:
                        if not wrapper.v2_integrator.on_cycle_start():
                            time.sleep(5.0)
                            continue
                    except Exception as e:
                        logger.debug(f"[WRAPPER] V2 cycle start error: {e}")

                # Periodic health check
                _health_check_counter += 1
                if _health_check_counter >= _health_check_interval:
                    _health_check_counter = 0
                    try:
                        from core.health_checks import health_shallow
                        report = health_shallow()
                        if report.get("overall") != "healthy":
                            logger.warning(
                                f"[HEALTH] Sistema não saudável: {report.get('overall')} — "
                                f"falhas: {[c['name'] for c in report.get('checks', []) if c['status'] != 'pass']}"
                            )
                        else:
                            logger.debug("[HEALTH] Sistema saudável")
                    except Exception as e:
                        logger.debug(f"[HEALTH] Health check falhou: {e}")

                # State recovery
                if wrapper.state_recovery and wrapper.state_recovery.is_recovering():
                    wrapper.state_recovery.execute_recovery_step()

                # Debug visualizer
                if wrapper.debug_integration:
                    try:
                        wrapper.debug_integration.update()
                    except Exception as e:
                        logger.debug(f"[WRAPPER] Debug visualizer update failed: {e}")

                # Safety checks
                safety_check = wrapper.safety.record_action()
                if not safety_check["safe"]:
                    logger.warning("Problema de seguranca detetado!")
                    if wrapper.safety.emergency_stop_triggered:
                        logger.error("Emergency stop ativado!")
                        wrapper.stop()
                        break
                if safety_check.get("should_delay"):
                    time.sleep(safety_check["delay"])
                if wrapper.safety.should_take_break():
                    break_duration = wrapper.safety.get_break_duration()
                    logger.info(f"Pausa obrigatoria: {break_duration / 60:.1f} min")
                    self._take_break(wrapper, break_duration)
                    break

                # Window randomization
                if wrapper.emulator_controller:
                    wrapper.emulator_controller.randomize_window_periodically()

                # Window resize detection
                if hasattr(wrapper, "emulator_subsystem"):
                    wrapper.emulator_subsystem.detect_window_resize()

                # World model / pressure map / behavioral updates
                self._update_decision_modules(wrapper)

                # Watchdog
                self._run_watchdog(wrapper)

                # V2 cycle end
                if wrapper.v2_integrator:
                    try:
                        wrapper.v2_integrator.on_cycle_end(cycle_duration=time.time() - cycle_start)
                    except Exception as e:
                        logger.debug(f"[WRAPPER] V2 cycle end error: {e}")

                # Observability
                if wrapper.observability:
                    wrapper.observability.record_cycle_time(time.time() - cycle_start)
                    if wrapper.state_manager:
                        wrapper.observability.update_state(wrapper.state_manager.current_state)
                    # --- SOVERANA FIX 2026-06-19: anti-stuck watchdog ---
                    try:
                        current_state = getattr(wrapper.state_manager, "current_state", None) if wrapper.state_manager else None
                        if current_state != last_state:
                            last_state = current_state
                            last_state_change = time.time()
                        elif current_state and time.time() - last_state_change > 90.0:
                            # Stuck > 90s no mesmo estado
                            logger.warning(f"[WRAPPER][WATCHDOG] Preso em {current_state} ha {time.time()-last_state_change:.0f}s")
                            if hasattr(wrapper.state_manager, "_last_action_time"):
                                if time.time() - getattr(wrapper.state_manager, "_last_action_time", 0) > 60.0:
                                    logger.warning("[WRAPPER][WATCHDOG] Sem acao ha >60s, forcando recovery")
                                    if hasattr(wrapper.state_manager, "current_state"):
                                        wrapper.state_manager.current_state = "unknown"
                                    last_state = "unknown"
                                    last_state_change = time.time()
                    except Exception as _wd_err:
                        logger.debug(f"[WRAPPER][WATCHDOG] check failed: {_wd_err}")
                    # --- end fix ---

                    if wrapper.state_recovery and wrapper.state_manager:
                        try:
                            confidence = 0.8
                            if wrapper.unified_detector and hasattr(wrapper.unified_detector, "_last_confidence"):
                                confidence = wrapper.unified_detector._last_confidence
                            wrapper.state_recovery.update_state(wrapper.state_manager.current_state, confidence)
                        except Exception as e:
                            logger.debug(f"[WRAPPER] State recovery update failed: {e}")

                # Dashboard
                if hasattr(wrapper, "ui_subsystem"):
                    wrapper.ui_subsystem.update_dashboard(wrapper)

                # Anti-ban
                if wrapper.anti_ban:
                    if wrapper.anti_ban.check_pattern():
                        logger.warning("[WRAPPER] Anti-ban: padrão repetitivo detetado, pausando brevemente")
                        time.sleep(__import__("random").uniform(5, 15))
                    if wrapper.anti_ban.check_throttle():
                        logger.warning("[WRAPPER] Anti-ban: throttling ativado")
                        if hasattr(wrapper.anti_ban, "get_adaptive_pacing"):
                            time.sleep(wrapper.anti_ban.get_adaptive_pacing(0.5, "menu_nav"))
                            continue

                # Adaptive delay
                if wrapper.state_manager and wrapper.state_manager.current_state == "in_game":
                    delay = 0.3
                    if wrapper.anti_ban and hasattr(wrapper.anti_ban, "get_adaptive_pacing"):
                        delay = wrapper.anti_ban.get_adaptive_pacing(0.3, "attack")
                    time.sleep(delay)
                else:
                    delay = __import__("random").uniform(0.8, 1.2)
                    if wrapper.anti_ban and hasattr(wrapper.anti_ban, "get_adaptive_pacing"):
                        delay = wrapper.anti_ban.get_adaptive_pacing(delay, "menu_nav")
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"Erro no monitor: {e}")
                if wrapper.enable_error_recovery and wrapper.error_recovery:
                    try:
                        context = wrapper.error_recovery.classify_error(
                            e, component="wrapper", operation="monitor_loop"
                        )
                        recovered = wrapper.error_recovery.handle_error(context, wrapper)
                        if recovered:
                            logger.info("[WRAPPER] Monitor loop error recovered")
                    except Exception:
                        pass
                time.sleep(__import__("random").uniform(0.5, 1.0))

    def _update_decision_modules(self, wrapper: PylaAIEnhanced) -> None:
        if not (wrapper.state_manager and wrapper.state_manager.current_state == "in_game"):
            return
        if wrapper.world_model and wrapper.play_logic and hasattr(wrapper.play_logic, "last_combat_snapshot"):
            try:
                snap = wrapper.play_logic.last_combat_snapshot
                if snap and snap.get("enemies"):
                    player = snap.get("player")
                    if player:
                        wrapper.world_model.update_enemies(snap["enemies"], player)
            except Exception as e:
                logger.warning(f"[WRAPPER] WorldModel.update failed: {e}")
        if wrapper.pressure_map and wrapper.play_logic and hasattr(wrapper.play_logic, "last_combat_snapshot"):
            try:
                snap = wrapper.play_logic.last_combat_snapshot
                if snap and snap.get("enemies"):
                    for enemy in snap["enemies"]:
                        cx = (enemy[0] + enemy[2]) // 2
                        cy = (enemy[1] + enemy[3]) // 2
                        wrapper.pressure_map.add_pressure(cx, cy, intensity=1.0)
            except Exception as e:
                logger.warning(f"[WRAPPER] PressureMap.update failed: {e}")
        if wrapper.behavioral_profile and wrapper.state_manager:
            try:
                wrapper.behavioral_profile.record_state(wrapper.state_manager.current_state)
            except Exception as e:
                logger.warning(f"[WRAPPER] BehavioralProfile.record_state failed: {e}")

    def _run_watchdog(self, wrapper: PylaAIEnhanced) -> None:
        sm = wrapper.state_manager
        if not sm or not hasattr(sm, "state_start_time") or not sm.state_start_time:
            return
        try:
            state = sm.current_state
            elapsed = time.time() - sm.state_start_time
            if elapsed > 5:
                logger.debug(f"[WATCHDOG] Estado {state} há {elapsed:.0f}s")
            if state == "matchmaking" and elapsed > 10:
                logger.warning(f"[WATCHDOG] Matchmaking preso há {elapsed:.0f}s - forçando in_game")
                sm.current_state = "in_game"
                sm.state_start_time = time.time()
            elif state == "loading" and elapsed > 12:
                logger.warning(f"[WATCHDOG] Loading preso há {elapsed:.0f}s - forçando in_game")
                sm.current_state = "in_game"
                sm.state_start_time = time.time()
            elif state == "lobby" and elapsed > 25:
                logger.warning(f"[WATCHDOG] Lobby preso há {elapsed:.0f}s - tentando clicar Play")
                if hasattr(sm, "_force_click_play"):
                    sm._force_click_play()
                    sm.state_start_time = time.time()
            elif state == "end" and elapsed > 15:
                logger.warning(f"[WATCHDOG] End screen preso há {elapsed:.0f}s - forçando lobby")
                sm.current_state = "lobby"
                sm.state_start_time = time.time()
            elif state == "unknown" and elapsed > 8:
                logger.warning(f"[WATCHDOG] Unknown preso há {elapsed:.0f}s - forçando lobby")
                sm.current_state = "lobby"
                sm.state_start_time = time.time()
                sm.unknown_since = None
        except Exception as e:
            logger.debug(f"[WATCHDOG] Erro no recovery: {e}")

    def _take_break(self, wrapper: PylaAIEnhanced, duration: float) -> None:
        wrapper.stop()
        time.sleep(duration)
        logger.info("Retomando apos pausa...")
        wrapper.start()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def record_heartbeat(self) -> None:
        with self._health_lock:
            self._last_action_time = time.time()
        self.wrapper._last_action_time = self._last_action_time

    def check_health(self) -> dict:
        with self._health_lock:
            time_since_action = time.time() - self._last_action_time

        health = {
            "healthy": True,
            "time_since_last_action": time_since_action,
            "running": self.wrapper.running,
            "current_state": (
                self.wrapper.state_manager.current_state
                if getattr(self.wrapper, "state_manager", None)
                else "none"
            ),
            "issues": [],
        }

        if self.wrapper.running and time_since_action > self.HEARTBEAT_TIMEOUT:
            health["healthy"] = False
            health["issues"].append(f"DEADLOCK: No action for {time_since_action:.0f}s")
            logger.error(f"[HEALTH] Deadlock detected! No action for {time_since_action:.0f}s")

        state_manager = getattr(self.wrapper, "state_manager", None)
        if (
            state_manager
            and state_manager.current_state == "unknown"
            and getattr(state_manager, "unknown_since", None)
        ):
            unknown_elapsed = time.time() - state_manager.unknown_since
            if unknown_elapsed > self.MAX_UNKNOWN_STATE_DURATION:
                health["healthy"] = False
                health["issues"].append(f"STUCK_UNKNOWN: In unknown state for {unknown_elapsed:.0f}s")
                logger.error(f"[HEALTH] Stuck in unknown state for {unknown_elapsed:.0f}s, forcing reset")
                state_manager.current_state = "lobby"
                state_manager.unknown_since = None

        return health
