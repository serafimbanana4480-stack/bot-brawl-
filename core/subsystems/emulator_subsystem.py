"""
core/subsystems/emulator_subsystem.py

EmulatorSubsystem: emulator detection, connection, screenshot capture,
resolution management, and window lifecycle.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wrapper import PylaAIEnhanced

logger = logging.getLogger(__name__)


class EmulatorSubsystem:
    """Manages emulator connection, screenshots, and window state."""

    def __init__(
        self,
        wrapper: PylaAIEnhanced,
        central_config: dict,
        images_path: Path,
        models_path: Path,
    ):
        self.wrapper = wrapper
        self.central_config = central_config
        self.images_path = images_path
        self.models_path = models_path
        self.controller: Any | None = None
        self.screenshot: Any | None = None
        self.resolution_manager: Any | None = None
        self._last_window_w: int = 0
        self._last_window_h: int = 0

        # Threading: screenshot buffer + input queue
        self._frame_buffer = collections.deque(maxlen=3)
        self._frame_lock = threading.Lock()
        self._screenshot_stop = threading.Event()
        self._screenshot_thread: threading.Thread | None = None

        self._input_queue: queue.Queue = queue.Queue(maxsize=20)
        self._input_stop = threading.Event()
        self._input_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> bool:
        """Connect to emulator and initialize screenshot capture."""
        emulator_ok = self._try_init_emulator_controller()

        if not emulator_ok:
            # Fallback to ScreenshotTaker
            try:
                from emulator_detector import get_emulator_detector

                detector = get_emulator_detector()
                emulators = detector.detect_all()
                chosen_title = None
                if emulators:
                    preferred = ["bluestacks", "ldplayer", "nox", "memu"]
                    for et in preferred:
                        for e in emulators:
                            if e.type == et:
                                chosen_title = e.window_title or e.name
                                break
                        if chosen_title:
                            break
                    if not chosen_title:
                        chosen_title = emulators[0].window_title or emulators[0].name
                    logger.info(f"Selected emulator: {chosen_title}")
                else:
                    logger.info("No emulators detected; using default title")
            except (ImportError, ModuleNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
                logger.debug(f"Emulator detection failed: {e}")
                chosen_title = None

            window_title = chosen_title or self.central_config.get("emulator", {}).get(
                "window_title", "BlueStacks App Player"
            )
            from pylaai_real.screenshot_taker import ScreenshotTaker

            self.screenshot = ScreenshotTaker(window_title)
            if not self.screenshot.find_window():
                logger.error("[WRAPPER] Emulador nao encontrado! Abra o emulador primeiro.")
                return False

            # Test screenshot capture
            logger.debug("[WRAPPER] Testando captura de screenshot...")
            if self.controller:
                logger.info("[WRAPPER] Garantindo que a janela do emulador está ativa")
                self.controller.ensure_window_active()
                time.sleep(0.2)

            test_img = self.screenshot.take()
            if test_img is None:
                logger.error("[WRAPPER] Emulador encontrado mas screenshot falhou.")
                return False
            logger.info("[WRAPPER] Screenshot funcionando, emulador responsivo")
            logger.info("Emulador conectado via ScreenshotTaker")

        # Create ScreenshotTaker if missing and we have an emulator controller
        if not self.screenshot and self.controller:
            window_title = self.controller.config.window_title
            from pylaai_real.screenshot_taker import ScreenshotTaker

            self.screenshot = ScreenshotTaker(window_title)
            if not self.screenshot.find_window():
                logger.warning("[WRAPPER] ScreenshotTaker window not found, will use ADB screenshots")
                self.screenshot = None

        # Resolution management
        window_title = "auto"
        if self.controller and hasattr(self.controller, "config"):
            window_title = self.controller.config.window_title

        try:
            from core.resolution_manager import ResolutionManager

            self.resolution_manager = ResolutionManager(
                window_title=window_title,
                on_resolution_change=self._on_resolution_change,
            )
            self.resolution_manager.detect()
            window_w, window_h = self.resolution_manager.actual_resolution
            logger.info(
                f"[WRAPPER] ResolutionManager ativo: {window_w}x{window_h} "
                f"(source={self.resolution_manager.profile.source})"
            )
        except (ImportError, ModuleNotFoundError, FileNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.debug(f"[WRAPPER] ResolutionManager unavailable: {e}")
            self.resolution_manager = None
            window_w, window_h = 1920, 1080

        # Fallback via screenshot
        if self.resolution_manager is None or not self.resolution_manager.profile.is_reasonable():
            if self.screenshot:
                try:
                    test_img = self.screenshot.take()
                    if test_img is not None:
                        actual_h, actual_w = test_img.shape[:2]
                        window_w, window_h = actual_w, actual_h
                        logger.warning(
                            f"[WRAPPER] ResolutionManager invalido — usando screenshot size: {window_w}x{window_h}"
                        )
                except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                    logger.debug(f"[WRAPPER] Screenshot size fallback failed: {e}")

        logger.info(f"[WRAPPER] Final resolution for coordinates: {window_w}x{window_h}")
        self._last_window_w = window_w
        self._last_window_h = window_h

        # Attach to wrapper for backward compatibility
        self.wrapper.emulator_controller = self.controller
        self.wrapper.screenshot = self.screenshot
        self.wrapper.resolution_manager = self.resolution_manager
        return True

    def start(self) -> None:
        """Launch screenshot and input worker threads."""
        self._screenshot_stop.clear()
        self._screenshot_thread = threading.Thread(
            target=self._screenshot_loop, daemon=True, name="emu-screenshot"
        )
        self._screenshot_thread.start()

        self._input_stop.clear()
        self._input_thread = threading.Thread(
            target=self._input_loop, daemon=True, name="emu-input"
        )
        self._input_thread.start()

    def stop(self) -> None:
        """Signal threads to stop and wait for graceful shutdown."""
        self._screenshot_stop.set()
        self._input_stop.set()
        for thr, name in [
            (self._screenshot_thread, "screenshot"),
            (self._input_thread, "input"),
        ]:
            if thr and thr.is_alive():
                thr.join(timeout=5.0)
                if thr.is_alive():
                    logger.warning(f"[EMULATOR] Thread {name} não terminou em 5s")
        if self.controller:
            try:
                self.controller.randomize_window_periodically(interval=0)
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.debug(f"[CLEANUP] Falha ao randomizar janela: {e}")

    def cleanup(self) -> None:
        """Full cleanup."""
        self.stop()

    # ------------------------------------------------------------------
    # Helpers exposed on wrapper
    # ------------------------------------------------------------------

    def get_screenshot_source(self) -> Any:
        """Return a screenshot source with .take() method."""
        if self.controller:

            class EmulatorWrapper:
                def __init__(self, controller, win32_taker):
                    self.controller = controller
                    self.win32_taker = win32_taker

                def take(self):
                    if self.win32_taker:
                        img = self.win32_taker.take()
                        if img is not None:
                            return img
                    import io

                    import numpy as np
                    from PIL import Image

                    try:
                        data = self.controller.get_screenshot()
                        if data:
                            return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
                    except (FileNotFoundError, PermissionError, ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                        logger.debug(f"[WRAPPER] ADB screenshot fallback failed: {e}")
                    return None

            if not self.screenshot:
                from pylaai_real.screenshot_taker import ScreenshotTaker

                self.screenshot = ScreenshotTaker(self.controller.config.window_title)
            return EmulatorWrapper(self.controller, self.screenshot)
        return self.screenshot

    def get_safe_resolution(self) -> tuple[int, int]:
        """Return current resolution safely."""
        if self.resolution_manager is not None:
            try:
                return self.resolution_manager.actual_resolution
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass
        if self._last_window_w and self._last_window_h:
            return (self._last_window_w, self._last_window_h)
        return (1920, 1080)

    def detect_window_resize(self) -> None:
        """Detect window resize and update coordinates."""
        if self.screenshot and hasattr(self.screenshot, "window_handle") and self.screenshot.window_handle:
            try:
                import win32gui

                rect = win32gui.GetWindowRect(self.screenshot.window_handle)
                new_w = rect[2] - rect[0]
                new_h = rect[3] - rect[1]
                if new_w != self._last_window_w or new_h != self._last_window_h:
                    logger.info(f"[WRAPPER] Window resized: {new_w}x{new_h}")
                    self._update_all_coordinates(new_w, new_h)
                self._last_window_w = new_w
                self._last_window_h = new_h
            except (ImportError, ModuleNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
                logger.debug(f"[WRAPPER] Window resize detection failed: {e}")

    # ------------------------------------------------------------------
    # Threading: screenshot buffer + input queue
    # ------------------------------------------------------------------

    def _screenshot_loop(self) -> None:
        """Capture frames into a circular buffer at ~30 FPS."""
        target_interval = 1.0 / 30.0
        while not self._screenshot_stop.is_set():
            t0 = time.time()
            frame = self._take_screenshot()
            if frame is not None:
                with self._frame_lock:
                    self._frame_buffer.append(frame)
            elapsed = time.time() - t0
            sleep_time = max(0.0, target_interval - elapsed)
            self._screenshot_stop.wait(timeout=sleep_time)

    def _take_screenshot(self) -> Any | None:
        """Try screenshot_taker first, then ADB fallback."""
        if self.screenshot is not None:
            try:
                return self.screenshot.take()
            except Exception as e:
                logger.debug(f"[EMULATOR] ScreenshotTaker failed: {e}")
        if self.controller is not None:
            try:
                import io

                import numpy as np
                from PIL import Image

                data = self.controller.get_screenshot()
                if data:
                    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            except Exception as e:
                logger.debug(f"[EMULATOR] ADB screenshot failed: {e}")
        return None

    def get_latest_frame(self) -> Any | None:
        """Return the most recent frame (non-blocking)."""
        with self._frame_lock:
            return self._frame_buffer[-1] if self._frame_buffer else None

    def enqueue_input(self, action: dict) -> bool:
        """Queue an input action for the worker thread."""
        try:
            self._input_queue.put(action, block=False)
            return True
        except queue.Full:
            logger.warning("[EMULATOR] Input queue full, dropping action")
            return False

    def _input_loop(self) -> None:
        """Consume input actions and dispatch to the controller."""
        while not self._input_stop.is_set():
            try:
                action = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._execute_input(action)
            except Exception as e:
                logger.error(f"[EMULATOR] Input execution failed: {e}")

    def _execute_input(self, action: dict) -> None:
        if self.controller is None:
            return
        atype = action.get("type")
        if atype == "tap":
            x = int(action.get("x", 0))
            y = int(action.get("y", 0))
            if hasattr(self.controller, "tap_scaled"):
                self.controller.tap_scaled(x, y)
            else:
                self.controller.tap(x, y)
        elif atype == "swipe":
            x1 = int(action.get("x1", 0))
            y1 = int(action.get("y1", 0))
            x2 = int(action.get("x2", 0))
            y2 = int(action.get("y2", 0))
            duration = action.get("duration_ms", 300)
            if hasattr(self.controller, "swipe_scaled"):
                self.controller.swipe_scaled(x1, y1, x2, y2, duration)
            else:
                self.controller.swipe(x1, y1, x2, y2, duration)
        elif atype == "key":
            keycode = action.get("keycode", 4)
            self.controller.keyevent(keycode)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_init_emulator_controller(self) -> bool:
        logger.debug("[WRAPPER] Tentando inicializar EmulatorController")
        try:
            from emulator_controller import EmulatorConfig, EmulatorController
            from emulator_detector import get_emulator_detector

            detector = get_emulator_detector()
            emulators = detector.detect_all()
            logger.info(f"[WRAPPER] {len(emulators)} emulador(es) detectado(s)")

            best_emu = None
            if emulators:
                for e in emulators:
                    if e.type == "bluestacks" and e.connected:
                        best_emu = e
                        break
                if not best_emu:
                    for e in emulators:
                        if e.connected:
                            best_emu = e
                            break

            emu_cfg = self.central_config.get("emulator", {})
            emu_type = emu_cfg.get("type", "bluestacks").lower()

            if best_emu:
                logger.info(
                    f"Using automatically detected emulator: {best_emu.name} "
                    f"(ID: {best_emu.adb_id}, Type: {best_emu.type})"
                )
                adb_port = 5555
                if best_emu.adb_id:
                    if ":" in best_emu.adb_id:
                        try:
                            adb_port = int(best_emu.adb_id.split(":")[-1])
                        except (ValueError, IndexError) as e:
                            logger.debug(f"[WRAPPER] Falha ao extrair porta ADB de {best_emu.adb_id}: {e}")
                    elif "emulator-" in best_emu.adb_id:
                        try:
                            adb_port = int(best_emu.adb_id.split("-")[-1])
                        except (ValueError, IndexError) as e:
                            logger.debug(f"[WRAPPER] Falha ao extrair porta ADB de {best_emu.adb_id}: {e}")

                config = EmulatorConfig(
                    name=best_emu.type,
                    adb_port=adb_port,
                    window_title=best_emu.window_title or emu_cfg.get("window_title", "BlueStacks App Player"),
                    resolution=emu_cfg.get("resolution", (1920, 1080)),
                    adb_path=getattr(best_emu, "adb_path", None),
                )
            else:
                if emu_type == "bluestacks":
                    config = EmulatorConfig.for_bluestacks()
                elif emu_type == "ldplayer":
                    config = EmulatorConfig.for_ldplayer()
                else:
                    config = EmulatorConfig(
                        name=emu_type,
                        adb_port=emu_cfg.get("adb_port", 5555),
                        window_title=emu_cfg.get("window_title", "BlueStacks App Player"),
                    )

            if "window_title" in emu_cfg:
                config.window_title = emu_cfg["window_title"]

            logger.debug(f"[WRAPPER] Criando EmulatorController com config: type={config.name}, port={config.adb_port}")

            self.controller = EmulatorController(
                config,
                safety_system=getattr(self.wrapper, "safety", None),
                humanization_system=getattr(self.wrapper, "humanization", None),
            )
            if self.controller.connect():
                logger.info(
                    f"[WRAPPER] EmulatorController conectado via ADB "
                    f"(Port: {config.adb_port}, ID: {self.controller.adb.device_id})"
                )
                return True
            else:
                logger.warning("[WRAPPER] EmulatorController falhou ao conectar, usando fallback ScreenshotTaker")
                self.controller = None
                return False
        except ImportError as e:
            logger.warning(f"[WRAPPER] EmulatorController não disponível (missing win32gui?): {e}")
            return False
        except (ModuleNotFoundError, ConnectionError, TimeoutError, ValueError, TypeError, RuntimeError, OSError) as e:
            self.controller = None
            logger.error(f"[WRAPPER] EmulatorController init falhou (fatal): {e}", exc_info=True)
            raise RuntimeError(
                "EmulatorController failed to initialize; refusing to continue without a working control path. "
                "Fix ADB/emulator configuration or install missing platform dependencies."
            ) from e

    def _on_resolution_change(self, profile) -> None:
        logger.warning(
            f"[WRAPPER] Resolucao mudou: {profile.previous_actual} -> {profile.actual_resolution}. "
            f"Recalibrando coordenadas..."
        )
        if hasattr(self.wrapper, "movement") and self.wrapper.movement:
            try:
                self.wrapper.movement.update_window_size(profile.actual_width, profile.actual_height)
                logger.info(f"[WRAPPER] MovementEngine atualizado para {profile.actual_resolution}")
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.error(f"[WRAPPER] Falha ao atualizar MovementEngine: {e}")
        if hasattr(self.wrapper, "unified_detector") and self.wrapper.unified_detector:
            try:
                self.wrapper.unified_detector.update_resolution(profile.actual_width, profile.actual_height)
                logger.info(f"[WRAPPER] UnifiedStateDetector atualizado para {profile.actual_resolution}")
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.error(f"[WRAPPER] Falha ao atualizar UnifiedStateDetector: {e}")
        if hasattr(self.wrapper, "auto_calibrator") and self.wrapper.auto_calibrator:
            try:
                self.wrapper.auto_calibrator.invalidate_cache()
                logger.info("[WRAPPER] AutoCalibrator cache invalidado")
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.error(f"[WRAPPER] Falha ao invalidar cache: {e}")

    def _update_all_coordinates(self, new_w: int, new_h: int) -> None:
        logger.info(f"[WRAPPER] Atualizando coordenadas para {new_w}x{new_h}")
        if hasattr(self.wrapper, "unified_detector") and self.wrapper.unified_detector:
            self.wrapper.unified_detector.update_window_size(new_w, new_h)
        if hasattr(self.wrapper, "lobby") and hasattr(self.wrapper.lobby, "update_window_size"):
            self.wrapper.lobby.update_window_size(new_w, new_h)
        if (
            hasattr(self.wrapper, "play_logic")
            and self.wrapper.play_logic
            and self.wrapper.play_logic.movement
        ):
            self.wrapper.play_logic.movement.update_window_size(new_w, new_h)
        if (
            hasattr(self.wrapper, "state_manager")
            and self.wrapper.state_manager
            and self.wrapper.state_manager.screen_automation
        ):
            self.wrapper.state_manager.screen_automation.update_window(new_w, new_h)
