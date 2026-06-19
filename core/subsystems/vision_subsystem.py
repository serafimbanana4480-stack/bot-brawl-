"""
core/subsystems/vision_subsystem.py

VisionSubsystem: YOLO model loading, state detection, screenshot analysis,
AutoCalibrator, OCR, and debug visualizer setup.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wrapper import PylaAIEnhanced

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class VisionSubsystem:
    """Manages vision models, detectors, and visual analysis."""

    def __init__(
        self,
        wrapper: PylaAIEnhanced,
        models_path: Path,
        images_path: Path,
        diagnostic_mode: bool,
        central_config: dict,
    ):
        self.wrapper = wrapper
        self.models_path = models_path
        self.images_path = images_path
        self.diagnostic_mode = diagnostic_mode
        self.central_config = central_config
        self.detect_main: Any | None = None
        self.detect_enemies: Any | None = None
        self.unified_detector: Any | None = None
        self.ocr_detector: Any | None = None
        self.auto_calibrator: Any | None = None
        self.debug_visualizer: Any | None = None
        self.debug_integration: Any | None = None
        self._debug_mode_enabled = bool(
            central_config.get("debug_visualizer", False)
            or __import__("os").getenv("PYLAAI_DEBUG_VISUAL", "0") == "1"
        )

        # Threading: inference worker
        self._inference_stop = threading.Event()
        self._inference_thread: threading.Thread | None = None
        self._latest_snapshot: Any | None = None
        self._snapshot_lock = threading.Lock()
        self._inference_errors_in_a_row = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> bool:
        """Load vision models and initialize detectors."""
        self._load_trained_models()

        # Phase 9: AutoCalibrator
        try:
            from pylaai_real.auto_calibrator import AutoCalibrator

            self.auto_calibrator = AutoCalibrator(
                templates_dir=self.images_path / "templates"
            )
            logger.info("[WRAPPER] AutoCalibrator inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] AutoCalibrator indisponível: {e}")

        # Phase 9: OCR State Detector
        try:
            from pylaai_real.ocr_state_detector import OCRStateDetector

            self.ocr_detector = OCRStateDetector()
            logger.info("[WRAPPER] OCR State Detector inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] OCR State Detector indisponível: {e}")

        # Phase 9: Debug Visualizer
        if self._debug_mode_enabled:
            try:
                from pylaai_real.debug_visualizer import DebugMode, DebugVisualizer

                self.debug_visualizer = DebugVisualizer(mode=DebugMode.DETAILED)
                logger.info("[WRAPPER] Debug Visualizer inicializado")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"[WRAPPER] Debug Visualizer indisponível: {e}")

        # Initialize UnifiedStateDetector (needs window size from wrapper)
        window_w, window_h = self._get_resolution()
        try:
            from pylaai_real.unified_state_detector import UnifiedStateDetector

            self.unified_detector = UnifiedStateDetector(
                self.images_path,
                window_w=window_w,
                window_h=window_h,
                ocr_detector=self.ocr_detector,
            )
            logger.info(f"[WRAPPER] UnifiedStateDetector inicializado ({window_w}x{window_h})")
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(f"[WRAPPER] UnifiedStateDetector init falhou: {e}")
            return False

        # Sync back to wrapper for backward compatibility
        self.wrapper.detect_main = self.detect_main
        self.wrapper.detect_enemies = self.detect_enemies
        self.wrapper.unified_detector = self.unified_detector
        self.wrapper.ocr_detector = self.ocr_detector
        self.wrapper.auto_calibrator = self.auto_calibrator
        self.wrapper.debug_visualizer = self.debug_visualizer
        self.wrapper.debug_integration = self.debug_integration
        return True

    def start(self) -> None:
        """Launch the inference worker thread."""
        self._inference_stop.clear()
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="vision-inference"
        )
        self._inference_thread.start()

    def stop(self) -> None:
        self._inference_stop.set()
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=5.0)
            if self._inference_thread.is_alive():
                logger.warning("[VISION] Inference thread não terminou em 5s")
        if self.debug_visualizer:
            try:
                self.debug_visualizer.stop()
                logger.info("[CLEANUP] Debug Visualizer parado")
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug(f"[CLEANUP] Falha ao parar debug visualizer: {e}")

    def cleanup(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_trained_models(self) -> bool:
        if not HAS_YOLO:
            logger.error("ultralytics not installed. pip install ultralytics")
            return False

        trained_models = {
            "latest_trained": None,
            "brawlstars_yolov8_8class": self.models_path / "brawlstars_yolov8_8class.pt",
            "brawlstars_yolov8": self.models_path / "brawlstars_yolov8.pt",
            "main_info": self.models_path / "main_info.pt",
            "brawler_id": self.models_path / "brawler_id.pt",
        }

        try:
            yolo_runs = sorted(
                self.models_path.glob("yolo/*/weights/best.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if yolo_runs:
                trained_models["latest_trained"] = yolo_runs[0]
                logger.debug(f"[MODEL] Discovered latest trained model: {yolo_runs[0]}")
        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError):
            pass

        generic_models = {
            "yolov8n": self.models_path / "yolov8n.pt",
            "yolov8m": self.models_path / "yolov8m.pt",
        }

        model_loaded = False

        # Latest trained
        latest_path = trained_models.get("latest_trained")
        if latest_path and latest_path.exists():
            for attempt in range(1, 4):
                try:
                    real_model = YOLO(str(latest_path))
                    try:
                        import torch

                        if torch.cuda.is_available():
                            real_model.to("cuda")
                            logger.info(f"[MODEL] Model moved to GPU: {torch.cuda.get_device_name(0)}")
                    except ImportError:
                        logger.debug("[MODEL] PyTorch/CUDA not available - using CPU")
                    from core.class_registry import get_schema

                    schema = "core"
                    expected = get_schema(schema)
                    actual_classes = set(real_model.names.values()) if hasattr(real_model, "names") else set()
                    expected_classes = set(expected.values())
                    if expected_classes.issubset(actual_classes):
                        from pylaai_real.detect import Detect

                        self.detect_main = Detect(model=real_model, classes=expected, conf=0.40)
                        self.detect_enemies = self.detect_main
                        logger.info(
                            f"[MODEL] Loaded LATEST TRAINED model: {latest_path.name} "
                            f"(classes: {actual_classes}, attempt={attempt})"
                        )
                        model_loaded = True
                        break
                    else:
                        logger.warning(
                            f"[MODEL] Latest trained model missing classes {expected_classes - actual_classes}, "
                            f"got: {actual_classes}"
                        )
                except (ImportError, ModuleNotFoundError, FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
                    logger.error(f"[MODEL] Failed to load latest trained model {latest_path} (attempt {attempt}/3): {e}")
                    if attempt < 3:
                        time.sleep(0.5)

        # BrawlStarsBot trained model
        if not model_loaded:
            brawlstars_path = trained_models.get("brawlstars_yolov8_8class") or trained_models.get("brawlstars_yolov8")
            if brawlstars_path and brawlstars_path.exists():
                for attempt in range(1, 4):
                    try:
                        real_model = YOLO(str(brawlstars_path))
                        try:
                            import torch

                            if torch.cuda.is_available():
                                real_model.to("cuda")
                                logger.info(f"[MODEL] Model moved to GPU: {torch.cuda.get_device_name(0)}")
                        except ImportError:
                            logger.debug("[MODEL] PyTorch/CUDA not available - using CPU")
                        from core.class_registry import get_canonical, get_schema

                        schema = "extended" if "8class" in brawlstars_path.name else "core"
                        expected_classes_raw = get_schema(schema).values()
                        expected_classes = {get_canonical(name) for name in expected_classes_raw}
                        actual_classes = set(real_model.names.values()) if hasattr(real_model, "names") else set()
                        if expected_classes.issubset(actual_classes):
                            from pylaai_real.detect import Detect

                            self.detect_main = Detect(model=real_model, classes=get_schema(schema), conf=0.40)
                            self.detect_enemies = self.detect_main
                            logger.info(
                                f"[MODEL] Loaded BrawlStarsBot TRAINED model: {brawlstars_path.name} "
                                f"(classes: {actual_classes}, attempt={attempt})"
                            )
                            model_loaded = True
                            break
                        else:
                            missing = expected_classes - actual_classes
                            logger.warning(f"[MODEL] BrawlStarsBot model missing classes {missing}, got: {actual_classes}")
                            from pylaai_real.detect import Detect

                            self.detect_main = Detect(model=real_model, classes=get_schema(schema), conf=0.40)
                            self.detect_enemies = self.detect_main
                            model_loaded = True
                            break
                    except (ImportError, ModuleNotFoundError, FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
                        logger.error(
                            f"[MODEL] Failed to load BrawlStarsBot model {brawlstars_path} (attempt {attempt}/3): {e}"
                        )
                        if attempt < 3:
                            time.sleep(0.5)

        # PylaAI trained models
        if not model_loaded:
            main_model_path = trained_models.get("main_info")
            if main_model_path and main_model_path.exists():
                try:
                    real_model = YOLO(str(main_model_path))
                    from pylaai_real.detect import Detect

                    self.detect_main = Detect(
                        model=real_model,
                        classes={0: "enemy", 1: "player", 2: "teammate"},
                        conf=0.1,
                    )
                    self.detect_enemies = self.detect_main
                    logger.info(f"Loaded PylaAI TRAINED model: {main_model_path.name}")
                    model_loaded = True
                except (ImportError, ModuleNotFoundError, FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
                    logger.error(f"Failed to load PylaAI model {main_model_path}: {e}")

        # Fallback generic models
        if not model_loaded:
            for _name, path in generic_models.items():
                if path.exists():
                    try:
                        real_model = YOLO(str(path))
                        from pylaai_real.detect import Detect

                        self.detect_main = Detect(
                            model=real_model,
                            classes={0: "person", 1: "brawler", 2: "object"},
                            conf=0.1,
                        )
                        self.detect_enemies = self.detect_main
                        logger.warning(f"Loaded GENERIC model: {path.name} (COCO-80, not trained for Brawl Stars)")
                        model_loaded = True
                        break
                    except (ImportError, ModuleNotFoundError, FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
                        logger.error(f"Failed to load generic model {path}: {e}")

        # Last resort: download yolov8n
        if not model_loaded:
            try:
                from model_downloader import get_model_downloader

                downloader = get_model_downloader()
                model_path = downloader.get_model_path("yolov8n")
                if not model_path:
                    logger.info("Downloading YOLOv8n as fallback...")
                    res = downloader.download_model("yolov8n")
                    if res.get("success"):
                        model_path = downloader.get_model_path("yolov8n")
                if model_path:
                    real_model = YOLO(str(model_path))
                    from pylaai_real.detect import Detect

                    self.detect_main = Detect(
                        model=real_model,
                        classes={0: "person", 1: "brawler", 2: "object"},
                        conf=0.1,
                    )
                    self.detect_enemies = self.detect_main
                    logger.warning("Using downloaded YOLOv8n with confidence threshold 0.1")
                    model_loaded = True
            except (ImportError, ModuleNotFoundError, FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, OSError) as e:
                logger.error(f"Model download fallback failed: {e}")

        if not model_loaded:
            logger.error("NO models loaded. Vision will not work.")
        return model_loaded

    # ------------------------------------------------------------------
    # Threading: inference worker
    # ------------------------------------------------------------------

    def get_latest_snapshot(self):
        """Return the most recent GameStateSnapshot (non-blocking)."""
        with self._snapshot_lock:
            return self._latest_snapshot

    def _inference_loop(self) -> None:
        """Consume frames from the emulator buffer and run YOLO + state detection."""

        target_interval = 1.0 / 30.0
        while not self._inference_stop.is_set():
            frame = None
            if hasattr(self.wrapper, "emulator_subsystem") and self.wrapper.emulator_subsystem:
                frame = self.wrapper.emulator_subsystem.get_latest_frame()
            if frame is None:
                self._inference_stop.wait(timeout=0.01)
                continue

            t0 = time.time()
            snapshot = self._run_inference(frame)
            if snapshot is not None:
                with self._snapshot_lock:
                    self._latest_snapshot = snapshot
                self._inference_errors_in_a_row = 0
            else:
                self._inference_errors_in_a_row += 1

            elapsed = time.time() - t0
            sleep_time = max(0.0, target_interval - elapsed)
            self._inference_stop.wait(timeout=sleep_time)

    def _run_inference(self, frame) -> Any | None:
        """Run detection and state classification on a single frame."""

        from core.ports.vision_port import GameStateSnapshot

        if frame is None:
            return None
        h, w = frame.shape[:2]
        snapshot = GameStateSnapshot(
            screenshot=frame,
            resolution=(w, h),
            timestamp=time.time(),
            latency_ms=0.0,
        )

        # YOLO detection
        if self.detect_main is not None:
            try:
                detections = self.detect_main(frame)
                snapshot.detected_objects = self._convert_detections(detections, w, h)
            except Exception as e:
                logger.debug(f"[VISION] Detection error: {e}")

        # State detection
        if self.unified_detector is not None:
            try:
                result = self.unified_detector.detect(frame)
                snapshot.game_phase = result.state
                snapshot.metadata["state_confidence"] = result.confidence
                snapshot.metadata["state_method"] = result.method
            except Exception as e:
                logger.debug(f"[VISION] State detection error: {e}")

        snapshot.latency_ms = (time.time() - snapshot.timestamp) * 1000
        return snapshot

    def _convert_detections(self, detections, img_w, img_h):
        """Convert YOLO detections to DetectedObject list."""
        from core.ports.vision_port import DetectedObject
        objects = []
        if detections is None:
            return objects
        try:
            if hasattr(detections, "boxes"):
                for box in detections.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = f"class_{cls_id}"
                    objects.append(DetectedObject(
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h),
                        center=((x1 + x2) / (2 * img_w), (y1 + y2) / (2 * img_h)),
                    ))
            elif isinstance(detections, list):
                for d in detections:
                    if len(d) >= 6:
                        x1, y1, x2, y2, conf, cls_id = d[:6]
                        class_name = f"class_{int(cls_id)}"
                        objects.append(DetectedObject(
                            class_name=class_name,
                            confidence=float(conf),
                            bbox=(x1, y1, x2, y2),
                            center=((x1 + x2) / 2, (y1 + y2) / 2),
                        ))
        except Exception as e:
            logger.debug(f"[VISION] Detection conversion error: {e}")
        return objects

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_resolution(self):
        if hasattr(self.wrapper, "resolution_manager") and self.wrapper.resolution_manager is not None:
            try:
                return self.wrapper.resolution_manager.actual_resolution
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass
        if hasattr(self.wrapper, "emulator_subsystem"):
            return self.wrapper.emulator_subsystem.get_safe_resolution()
        return (1920, 1080)
