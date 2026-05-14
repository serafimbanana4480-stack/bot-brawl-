"""
core/async_pipeline.py

Async Pipeline for parallel inference, tracking, and decision making.

Solves the "sequential pipeline" problem where YOLO inference → tracking →
decision → action happens sequentially, adding cumulative latency.

Architecture:
  ┌─────────────┐    ┌──────────┐    ┌──────────┐
  │  Inference   │───▶│ Tracking │───▶│ Decision │───▶ Action
  │  (YOLO)      │    │ (ByteTrack)│  │ (UtilityAI)│
  └─────────────┘    └──────────┘    └──────────┘
        │                  │                │
        ▼                  ▼                ▼
  [Shared State Buffer with double-buffering]

Inference and Tracking run in parallel threads, writing to a back buffer.
Decision reads from the front buffer. Buffer swap happens atomically.

This means:
- YOLO inference doesn't wait for tracking to finish
- Tracking doesn't wait for decision
- Decision always reads the latest complete state
- Latency = max(inference, tracking) instead of sum
"""

import logging
import threading
import time
import traceback
from typing import Any, Callable, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    INFERENCE = "inference"
    TRACKING = "tracking"
    DECISION = "decision"
    ACTION = "action"


@dataclass
class PipelineFrame:
    """A single frame's data flowing through the pipeline."""
    frame_id: int = 0
    timestamp: float = 0.0

    # Inference outputs
    raw_detections: Optional[list] = None
    inference_time_ms: float = 0.0

    # Tracking outputs
    tracked_objects: Optional[list] = None
    tracking_time_ms: float = 0.0

    # Decision outputs
    action: Optional[Any] = None
    action_params: Optional[Dict] = None
    decision_time_ms: float = 0.0

    # World state snapshot
    world_state: Optional[Dict] = None

    # Pipeline metrics
    total_latency_ms: float = 0.0
    dropped: bool = False


class DoubleBuffer:
    """
    Thread-safe double buffer for pipeline state.

    Producer threads write to the back buffer.
    Consumer threads read from the front buffer.
    Swap is atomic.
    """

    def __init__(self):
        self._front = PipelineFrame()
        self._back = PipelineFrame()
        self._lock = threading.Lock()
        self._frame_counter = 0

    def write_back(self, **kwargs):
        """Write data to the back buffer (producer side)."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._back, key):
                    setattr(self._back, key, value)

    def swap(self):
        """Swap front and back buffers atomically."""
        with self._lock:
            self._front, self._back = self._back, self._front
            self._frame_counter += 1
            self._back = PipelineFrame(
                frame_id=self._frame_counter,
                timestamp=time.time()
            )

    def read_front(self) -> PipelineFrame:
        """Read from the front buffer (consumer side)."""
        with self._lock:
            return self._front

    @property
    def frame_id(self):
        with self._lock:
            return self._frame_counter


class AsyncPipeline:
    """
    Parallel inference-tracking-decision pipeline.

    Runs YOLO inference and ByteTrack tracking in parallel threads,
    with double-buffered state sharing. Decision runs on the main
    thread reading from the front buffer.

    Usage:
        pipeline = AsyncPipeline(
            inference_fn=my_yolo_detect,
            tracking_fn=my_bytetrack_update,
            decision_fn=my_utility_ai_evaluate,
        )
        pipeline.start()

        # In main loop:
        action = pipeline.get_latest_action()
        # ... execute action ...

        pipeline.stop()
    """

    def __init__(
        self,
        inference_fn: Optional[Callable] = None,
        tracking_fn: Optional[Callable] = None,
        decision_fn: Optional[Callable] = None,
        max_inference_fps: float = 15.0,
        max_tracking_fps: float = 30.0,
    ):
        self.inference_fn = inference_fn
        self.tracking_fn = tracking_fn
        self.decision_fn = decision_fn

        self.max_inference_fps = max_inference_fps
        self.max_tracking_fps = max_tracking_fps

        self._buffer = DoubleBuffer()
        self._running = False
        self._threads: Dict[str, threading.Thread] = {}

        # Performance tracking
        self._stats = {
            "inference_fps": 0.0,
            "tracking_fps": 0.0,
            "decision_fps": 0.0,
            "avg_latency_ms": 0.0,
            "frames_processed": 0,
            "frames_dropped": 0,
            "inference_errors": 0,
            "tracking_errors": 0,
        }
        self._fps_counters = {
            "inference": 0,
            "tracking": 0,
            "decision": 0,
        }
        self._fps_timestamps = {
            "inference": time.time(),
            "tracking": time.time(),
            "decision": time.time(),
        }
        self._latency_samples: list = []

        # Latest screenshot for inference thread
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._new_frame_event = threading.Event()

        logger.info("[ASYNC_PIPELINE] Initialized (inference=%.0ffps, tracking=%.0ffps)",
                     max_inference_fps, max_tracking_fps)

    def start(self):
        """Start the pipeline threads."""
        if self._running:
            return

        self._running = True

        # Inference thread
        if self.inference_fn:
            t = threading.Thread(
                target=self._inference_loop,
                name="pipeline-inference",
                daemon=True,
            )
            self._threads["inference"] = t
            t.start()

        # Tracking thread
        if self.tracking_fn:
            t = threading.Thread(
                target=self._tracking_loop,
                name="pipeline-tracking",
                daemon=True,
            )
            self._threads["tracking"] = t
            t.start()

        logger.info("[ASYNC_PIPELINE] Started %d threads", len(self._threads))

    def stop(self):
        """Stop all pipeline threads."""
        self._running = False
        self._new_frame_event.set()  # Wake up inference thread

        for name, t in self._threads.items():
            t.join(timeout=3.0)
            if t.is_alive():
                logger.warning("[ASYNC_PIPELINE] Thread %s did not stop cleanly", name)

        self._threads.clear()
        logger.info("[ASYNC_PIPELINE] Stopped")

    def submit_frame(self, frame: np.ndarray):
        """
        Submit a new screenshot frame for processing.

        Called from the main screenshot capture loop.
        The inference thread will pick it up.
        """
        with self._frame_lock:
            self._latest_frame = frame
        self._new_frame_event.set()

    def get_latest_action(self) -> Optional[Tuple[Any, Optional[Dict]]]:
        """
        Get the latest action from the decision stage.

        Called from the main game loop. Runs decision_fn if available.
        Returns (action, action_params) or None.
        """
        if not self.decision_fn:
            return None

        front = self._buffer.read_front()

        if front.world_state is None:
            return None

        start = time.time()
        try:
            result = self.decision_fn(front.world_state)
            elapsed = (time.time() - start) * 1000

            # Update back buffer with decision result
            if isinstance(result, tuple) and len(result) == 2:
                action, params = result
            else:
                action, params = result, None

            self._buffer.write_back(
                action=action,
                action_params=params,
                decision_time_ms=elapsed,
            )

            self._fps_counters["decision"] += 1
            self._update_fps("decision")

            return (action, params)

        except Exception as e:
            logger.error("[ASYNC_PIPELINE] Decision error: %s", e)
            return None

    def get_stats(self) -> Dict:
        """Get pipeline performance statistics."""
        self._update_all_fps()
        return self._stats.copy()

    # --- Internal thread loops ---

    def _inference_loop(self):
        """Inference thread: runs YOLO on latest frame."""
        min_interval = 1.0 / self.max_inference_fps if self.max_inference_fps > 0 else 0

        while self._running:
            try:
                # Wait for a new frame
                self._new_frame_event.wait(timeout=1.0)
                self._new_frame_event.clear()

                if not self._running:
                    break

                # Get latest frame
                with self._frame_lock:
                    frame = self._latest_frame
                    self._latest_frame = None

                if frame is None:
                    continue

                start = time.time()

                # Run inference
                detections = self.inference_fn(frame)

                elapsed = (time.time() - start) * 1000

                # Write results to back buffer
                self._buffer.write_back(
                    raw_detections=detections,
                    inference_time_ms=elapsed,
                )

                self._fps_counters["inference"] += 1
                self._update_fps("inference")

                # Throttle to max FPS
                elapsed_s = (time.time() - start)
                sleep_time = min_interval - elapsed_s
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error("[ASYNC_PIPELINE] Inference error: %s\n%s",
                             e, traceback.format_exc())
                self._stats["inference_errors"] += 1
                time.sleep(0.5)

    def _tracking_loop(self):
        """Tracking thread: runs ByteTrack on latest detections."""
        min_interval = 1.0 / self.max_tracking_fps if self.max_tracking_fps > 0 else 0

        while self._running:
            try:
                # Read latest detections from front buffer
                front = self._buffer.read_front()

                if front.raw_detections is None:
                    time.sleep(0.01)
                    continue

                start = time.time()

                # Run tracking
                tracked = self.tracking_fn(front.raw_detections)

                elapsed = (time.time() - start) * 1000

                # Write results to back buffer
                self._buffer.write_back(
                    tracked_objects=tracked,
                    tracking_time_ms=elapsed,
                )

                # Build world state from tracked objects
                world_state = self._build_world_state(tracked)
                self._buffer.write_back(world_state=world_state)

                # Swap buffers so decision can read the new state
                self._buffer.swap()

                self._fps_counters["tracking"] += 1
                self._update_fps("tracking")

                # Track latency
                total = (time.time() - front.timestamp) * 1000 if front.timestamp > 0 else 0
                self._latency_samples.append(total)
                if len(self._latency_samples) > 60:
                    self._latency_samples = self._latency_samples[-60:]
                self._stats["avg_latency_ms"] = (
                    sum(self._latency_samples) / len(self._latency_samples)
                    if self._latency_samples else 0
                )
                self._stats["frames_processed"] += 1

                # Throttle
                elapsed_s = (time.time() - start)
                sleep_time = min_interval - elapsed_s
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error("[ASYNC_PIPELINE] Tracking error: %s\n%s",
                             e, traceback.format_exc())
                self._stats["tracking_errors"] += 1
                time.sleep(0.1)

    def _build_world_state(self, tracked_objects: list) -> Dict:
        """
        Build a world state dict from tracked objects for the decision stage.

        This integrates with WorldModel, OccupancyGrid, PressureMap, etc.
        """
        state = {
            "enemies": [],
            "allies": [],
            "power_cubes": [],
            "walls": [],
            "bushes": [],
            "enemy_count": 0,
            "nearest_enemy_dist": 999.0,
            "nearest_enemy_health": 1.0,
            "pressure": 0.0,
            "danger": 0.0,
            "timestamp": time.time(),
        }

        if not tracked_objects:
            return state

        for obj in tracked_objects:
            class_name = obj.get("class_name", "").lower()
            x, y = obj.get("x", 0), obj.get("y", 0)

            if class_name in ("enemy", "opponent"):
                state["enemies"].append(obj)
            elif class_name in ("ally", "teammate"):
                state["allies"].append(obj)
            elif class_name in ("powerup", "power_cube"):
                state["power_cubes"].append(obj)
            elif class_name == "wall":
                state["walls"].append(obj)
            elif class_name == "bush":
                state["bushes"].append(obj)

        state["enemy_count"] = len(state["enemies"])

        # Calculate nearest enemy distance
        if state["enemies"]:
            # Assume player is at center-bottom of screen
            player_x, player_y = 640, 600
            min_dist = float('inf')
            for e in state["enemies"]:
                dist = math.sqrt((e.get("x", 0) - player_x) ** 2 +
                                 (e.get("y", 0) - player_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    state["nearest_enemy_dist"] = dist
                    state["nearest_enemy_health"] = e.get("health", 1.0)

        return state

    def _update_fps(self, stage: str):
        """Update FPS counter for a stage."""
        now = time.time()
        elapsed = now - self._fps_timestamps[stage]
        if elapsed >= 1.0:
            fps = self._fps_counters[stage] / elapsed
            self._stats[f"{stage}_fps"] = round(fps, 1)
            self._fps_counters[stage] = 0
            self._fps_timestamps[stage] = now

    def _update_all_fps(self):
        """Update all FPS counters."""
        for stage in ("inference", "tracking", "decision"):
            self._update_fps(stage)


# Need math import for _build_world_state
import math
