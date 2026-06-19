"""
core/adapters/vision_adapter.py

Adapter: existing vision subsystem -> VisionPort

Wraps:
    - ScreenshotTaker (frame capture)
    - Detect (YOLO inference)
    - UnifiedStateDetector (state classification)
    - Optional: MultimodalPipeline (OCR + heuristics)
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from core.ports.vision_port import (
    DetectedObject,
    GameStateSnapshot,
    VisionPort,
)

logger = logging.getLogger(__name__)


class VisionAdapter(VisionPort):
    """
    Concrete adapter for the existing vision pipeline.
    Bridges YOLO + UnifiedStateDetector + ScreenshotTaker to VisionPort.
    """

    def __init__(
        self,
        screenshot_taker: Any = None,
        detector: Any = None,
        state_detector: Any = None,
        images_path: Path | None = None,
        resolution: tuple[int, int] = (1920, 1080),
        snapshot_source: Any | None = None,
    ):
        self._screenshot = screenshot_taker
        self._detector = detector
        self._state_detector = state_detector
        self._images_path = images_path
        self._resolution = resolution
        self._last_snapshot: GameStateSnapshot | None = None
        self._errors_in_a_row = 0
        self._max_errors = 5
        self._lock = threading.RLock()
        self._snapshot_source = snapshot_source

    def set_snapshot_source(self, source: Any | None) -> None:
        """Wire a non-blocking snapshot source (e.g. VisionSubsystem thread)."""
        self._snapshot_source = source

    # ------------------------------------------------------------------
    # VisionPort implementation
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Lazy init — components are injected or created on demand."""
        with self._lock:
            if self._screenshot is None:
                try:
                    from pylaai_real.screenshot_taker import ScreenshotTaker
                    self._screenshot = ScreenshotTaker()
                    if not self._screenshot.find_window():
                        logger.error("[VISION_ADAPTER] ScreenshotTaker: no window found")
                        return False
                except (ImportError, ModuleNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
                    logger.error(f"[VISION_ADAPTER] Failed to create ScreenshotTaker: {e}")
                    return False

            if self._detector is None:
                try:
                    from pylaai_real.detect import Detect
                    self._detector = Detect(model_path="models/brawlstars_yolov8_8class.pt")
                except (ImportError, ModuleNotFoundError, FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
                    logger.warning(f"[VISION_ADAPTER] Detector not available: {e}")

            if self._state_detector is None and self._images_path:
                try:
                    from pylaai_real.unified_state_detector import UnifiedStateDetector
                    self._state_detector = UnifiedStateDetector(
                        images_path=self._images_path,
                        window_w=self._resolution[0],
                        window_h=self._resolution[1],
                    )
                except (ImportError, ModuleNotFoundError, TypeError) as e:
                    logger.warning(f"[VISION_ADAPTER] State detector not available: {e}")

            logger.info("[VISION_ADAPTER] Initialized")
            return True

    def capture_and_perceive(self) -> GameStateSnapshot | None:
        # Fast non-blocking path when wired to a threaded vision subsystem
        if self._snapshot_source is not None:
            snapshot = self._snapshot_source()
            if snapshot is not None:
                with self._lock:
                    self._last_snapshot = snapshot
            return snapshot

        t0 = time.time()

        with self._lock:
            if self._screenshot is None:
                return None

            # Capture
            frame = self._screenshot.take()
            if frame is None:
                self._errors_in_a_row += 1
                if self._errors_in_a_row >= self._max_errors:
                    logger.error("[VISION_ADAPTER] Screenshot failing repeatedly")
                return None
            self._errors_in_a_row = 0

            h, w = frame.shape[:2]
            snapshot = GameStateSnapshot(
                screenshot=frame,
                resolution=(w, h),
                timestamp=time.time(),
                latency_ms=0.0,
            )

            # Detect objects (YOLO)
            if self._detector is not None:
                try:
                    detections = self._detector(frame)
                    snapshot.detected_objects = self._convert_detections(detections, w, h)
                except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                    logger.debug(f"[VISION_ADAPTER] Detection error: {e}")

            # Detect game state (lobby, combat, etc.)
            if self._state_detector is not None:
                try:
                    result = self._state_detector.detect(frame)
                    snapshot.game_phase = result.state
                    snapshot.metadata["state_confidence"] = result.confidence
                    snapshot.metadata["state_method"] = result.method
                except (FileNotFoundError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                    logger.debug(f"[VISION_ADAPTER] State detection error: {e}")

            snapshot.latency_ms = (time.time() - t0) * 1000
            self._last_snapshot = snapshot
            return snapshot

    def get_detected_objects(self, class_filter: list[str] | None = None) -> list[DetectedObject]:
        with self._lock:
            if self._last_snapshot is None:
                return []
            objects = self._last_snapshot.detected_objects
        if class_filter:
            objects = [o for o in objects if o.class_name in class_filter]
        return objects

    def health_check(self) -> dict[str, Any]:
        with self._lock:
            return {
                "screenshot_ok": self._screenshot is not None or self._snapshot_source is not None,
                "detector_ok": self._detector is not None,
                "state_detector_ok": self._state_detector is not None,
                "last_latency_ms": self._last_snapshot.latency_ms if self._last_snapshot else 0.0,
                "errors_in_a_row": self._errors_in_a_row,
            }

    def shutdown(self) -> None:
        logger.info("[VISION_ADAPTER] Shutdown")
        # Let GC handle screenshot/detector; no explicit cleanup needed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_detections(detections: Any, img_w: int, img_h: int) -> list[DetectedObject]:
        """Convert YOLO detections to DetectedObject list."""
        objects: list[DetectedObject] = []
        if detections is None:
            return objects

        try:
            # Handle different detection formats
            if hasattr(detections, "boxes"):
                # Ultralytics Results
                for box in detections.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    # Map class id to name (simplified)
                    class_name = f"class_{cls_id}"
                    objects.append(DetectedObject(
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h),
                        center=((x1 + x2) / (2 * img_w), (y1 + y2) / (2 * img_h)),
                    ))
            elif isinstance(detections, list):
                # List of [x1, y1, x2, y2, conf, cls]
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
        except (FileNotFoundError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.debug(f"[VISION_ADAPTER] Detection conversion error: {e}")

        return objects
