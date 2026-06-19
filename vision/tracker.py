"""
ByteTrack/DeepSORT-based object tracker for Brawl Stars.
Provides persistent object IDs across frames for tracking enemies, player, etc.
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Optional movement predictor for advanced prediction
try:
    import importlib.util as _vt_ilu
    MOVEMENT_PREDICTOR_AVAILABLE = _vt_ilu.find_spec("vision.movement_predictor") is not None
except Exception:
    MOVEMENT_PREDICTOR_AVAILABLE = False


@dataclass
class TrackedObject:
    """Represents a tracked object with persistent ID."""
    id: int
    class_name: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: tuple[float, float]
    velocity: tuple[float, float]
    age: int  # frames since last detection
    hits: int  # total detections
    last_seen: float

    def is_stale(self, max_age: int = 30) -> bool:
        """Check if tracking data is stale."""
        return self.age > max_age

    def predict_position(self) -> tuple[float, float]:
        """Predict next position based on velocity."""
        return (
            self.center[0] + self.velocity[0],
            self.center[1] + self.velocity[1]
        )


class ByteTracker:
    """
    Simplified ByteTrack implementation for Brawl Stars.
    Uses IoU-based matching with Kalman filter prediction.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        track_buffer: int = 30
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.track_buffer = track_buffer

        self.next_id = 1
        self.tracks: dict[int, TrackedObject] = {}
        self.track_history: dict[int, deque] = {}  # For velocity calculation

    def _compute_iou(
        self,
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int]
    ) -> float:
        """Compute Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _get_center(self, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        """Get center point of bbox."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _calculate_velocity(
        self,
        track_id: int,
        new_center: tuple[float, float]
    ) -> tuple[float, float]:
        """Calculate velocity based on position history."""
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=5)

        history = self.track_history[track_id]
        if len(history) > 0:
            prev_center = history[-1]
            velocity = (
                new_center[0] - prev_center[0],
                new_center[1] - prev_center[1]
            )
        else:
            velocity = (0.0, 0.0)

        history.append(new_center)
        return velocity

    def update(
        self,
        detections: list[tuple[str, tuple[int, int, int, int], float]]
    ) -> dict[int, TrackedObject]:
        """
        Update tracker with new detections.

        Args:
            detections: List of (class_name, bbox, confidence) tuples

        Returns:
            List of active tracked objects
        """
        current_time = time.time()

        # Age all existing tracks
        for track in self.tracks.values():
            track.age += 1

        # Separate high and low confidence detections
        high_conf_dets = [(i, d) for i, d in enumerate(detections) if d[2] > 0.5]
        low_conf_dets = [(i, d) for i, d in enumerate(detections) if d[2] <= 0.5]

        # Match high confidence detections
        matched_tracks = set()
        matched_dets = set()

        # First association: IoU matching for high confidence
        for track_id, track in self.tracks.items():
            if track.age > self.max_age:
                continue

            best_iou = self.iou_threshold
            best_det_idx = None

            for det_idx, (class_name, bbox, _conf) in high_conf_dets:
                if det_idx in matched_dets:
                    continue
                if class_name != track.class_name:
                    continue

                iou = self._compute_iou(track.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx is not None:
                # Update existing track
                class_name, bbox, conf = high_conf_dets[best_det_idx][1]
                center = self._get_center(bbox)
                velocity = self._calculate_velocity(track_id, center)

                self.tracks[track_id] = TrackedObject(
                    id=track_id,
                    class_name=class_name,
                    bbox=bbox,
                    confidence=conf,
                    center=center,
                    velocity=velocity,
                    age=0,
                    hits=track.hits + 1,
                    last_seen=current_time
                )

                matched_tracks.add(track_id)
                matched_dets.add(best_det_idx)

        # Second association: Match low confidence to unmatched tracks
        for track_id, track in self.tracks.items():
            if track_id in matched_tracks or track.age > self.max_age:
                continue

            best_iou = self.iou_threshold
            best_det_idx = None

            for det_idx, (class_name, bbox, _conf) in low_conf_dets:
                if det_idx in matched_dets:
                    continue
                if class_name != track.class_name:
                    continue

                iou = self._compute_iou(track.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx is not None:
                class_name, bbox, conf = low_conf_dets[best_det_idx][1]
                center = self._get_center(bbox)
                velocity = self._calculate_velocity(track_id, center)

                self.tracks[track_id] = TrackedObject(
                    id=track_id,
                    class_name=class_name,
                    bbox=bbox,
                    confidence=conf,
                    center=center,
                    velocity=velocity,
                    age=0,
                    hits=track.hits + 1,
                    last_seen=current_time
                )

                matched_tracks.add(track_id)
                matched_dets.add(best_det_idx)

        # Create new tracks for unmatched high confidence detections
        for det_idx, (class_name, bbox, conf) in high_conf_dets:
            if det_idx in matched_dets:
                continue

            center = self._get_center(bbox)
            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = TrackedObject(
                id=track_id,
                class_name=class_name,
                bbox=bbox,
                confidence=conf,
                center=center,
                velocity=(0.0, 0.0),
                age=0,
                hits=1,
                last_seen=current_time
            )

            self.track_history[track_id] = deque(maxlen=self.track_buffer)
            self.track_history[track_id].append(center)

        # Create new tracks for unmatched low confidence detections
        for det_idx, (class_name, bbox, conf) in low_conf_dets:
            if det_idx in matched_dets:
                continue

            center = self._get_center(bbox)
            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = TrackedObject(
                id=track_id,
                class_name=class_name,
                bbox=bbox,
                confidence=conf,
                center=center,
                velocity=(0.0, 0.0),
                age=0,
                hits=1,
                last_seen=current_time
            )

            self.track_history[track_id] = deque(maxlen=self.track_buffer)
            self.track_history[track_id].append(center)

        # Remove stale tracks
        stale_ids = [tid for tid, t in self.tracks.items() if t.age > self.max_age]
        for tid in stale_ids:
            del self.tracks[tid]
            self.track_history.pop(tid, None)

        return {tid: t for tid, t in self.tracks.items() if t.age <= self.max_age}

    def get_predicted_position(
        self,
        track_id: int,
        frames_ahead: int = 5
    ) -> tuple[float, float] | None:
        """
        Get predicted position for a track using movement predictor if available.
        Falls back to linear velocity prediction if not.

        Args:
            track_id: Track ID to predict for
            frames_ahead: Number of frames to predict ahead

        Returns:
            Predicted (x, y) position or None if track not found
        """
        if track_id not in self.tracks:
            return None

        track = self.tracks[track_id]

        # Use advanced predictor if available
        if self.movement_predictor is not None and MOVEMENT_PREDICTOR_AVAILABLE:
            if track_id in self.track_history and len(self.track_history[track_id]) >= 3:
                history = list(self.track_history[track_id])
                try:
                    prediction = self.movement_predictor.predict_trajectory(
                        history,
                        frames_ahead=frames_ahead
                    )
                    if prediction is not None and len(prediction) > 0:
                        return prediction[-1]  # Return last predicted position
                except Exception as e:
                    logger.warning(f"Movement predictor failed: {e}")

        # Fallback to linear prediction
        pred_x = track.center[0] + track.velocity[0] * frames_ahead
        pred_y = track.center[1] + track.velocity[1] * frames_ahead
        return (pred_x, pred_y)

    def get_leading_shot_position(
        self,
        track_id: int,
        projectile_speed: float,
        frame_delay: int = 0
    ) -> tuple[float, float] | None:
        """
        Get leading shot position for a track.

        Args:
            track_id: Track ID to calculate leading shot for
            projectile_speed: Speed of projectile in pixels/frame
            frame_delay: Additional frame delay (e.g., reaction time)

        Returns:
            Predicted (x, y) position to aim at or None if track not found
        """
        if track_id not in self.tracks:
            return None

        if self.movement_predictor is not None and MOVEMENT_PREDICTOR_AVAILABLE:
            if track_id in self.track_history and len(self.track_history[track_id]) >= 3:
                history = list(self.track_history[track_id])
                try:
                    leading_pos = self.movement_predictor.calculate_leading_shot(
                        history,
                        projectile_speed=projectile_speed,
                        frame_delay=frame_delay
                    )
                    if leading_pos is not None:
                        return leading_pos
                except Exception as e:
                    logger.warning(f"Leading shot calculation failed: {e}")

        # Fallback to simple velocity-based leading
        track = self.tracks[track_id]
        speed = math.sqrt(track.velocity[0]**2 + track.velocity[1]**2)
        if speed > 0 and projectile_speed > 0:
            # Time to intercept
            intercept_time = 5.0  # Default: 5 frames ahead
            lead_x = track.center[0] + track.velocity[0] * intercept_time
            lead_y = track.center[1] + track.velocity[1] * intercept_time
            return (lead_x, lead_y)

        return track.center

    def get_tracks_by_class(self, class_name: str) -> list[TrackedObject]:
        """Get all tracks of a specific class."""
        return [t for t in self.tracks.values() if t.class_name == class_name]

    def reset(self):
        """Reset all tracks."""
        self.next_id = 1
        self.tracks.clear()
        self.track_history.clear()
