"""
data/enriched_collector.py

Dataset Collector Enriquecido com EventStore.

Coleta dados de gameplay com metadados completos para treinamento BC/CQL/DQN.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class EnrichedFrame:
    timestamp: float
    frame_id: int
    session_id: str
    match_id: str
    game_state: str
    brawler: Optional[str] = None
    map_name: Optional[str] = None
    player_hp: float = 1.0
    action_taken: str = "idle"
    action_scores: Dict[str, float] = field(default_factory=dict)
    enemies: list = field(default_factory=list)
    cubes_on_map: list = field(default_factory=list)
    cycle_duration_ms: float = 0.0
    inference_time_ms: float = 0.0
    recent_events: list = field(default_factory=list)
    screenshot_path: Optional[str] = None


class EnrichedDatasetCollector:
    def __init__(
        self,
        output_dir: Path = Path("data/enriched"),
        max_frames_per_match: int = 5000,
        save_screenshots: bool = True,
        jpeg_quality: int = 80,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames_per_match = max_frames_per_match
        self.save_screenshots = save_screenshots
        self.jpeg_quality = jpeg_quality
        self._current_match = None
        self._frame_counter = 0
        self._buffer = deque(maxlen=100)
        self._session_id = f"sess_{int(time.time())}"

    def start_match(self, match_id: str, brawler: str, map_name: str):
        self._current_match = match_id
        self._frame_counter = 0
        logger.info("[ENRICHED] Match: %s (%s / %s)", match_id, brawler, map_name)

    def collect_frame(self, screenshot, game_state: str, detections: Dict, decision: Dict, performance: Dict, v2_integrator=None):
        if not self._current_match:
            return None
        self._frame_counter += 1
        if self._frame_counter > self.max_frames_per_match:
            return None

        frame = EnrichedFrame(
            timestamp=time.time(),
            frame_id=self._frame_counter,
            session_id=self._session_id,
            match_id=self._current_match,
            game_state=game_state,
            action_taken=decision.get("action", "idle"),
            action_scores=decision.get("scores", {}),
            enemies=[{"bbox": e} for e in (detections.get("enemy", []) + detections.get("Enemy", []))[:5]],
            cubes_on_map=[{"bbox": c} for c in (detections.get("cubebox", []) + detections.get("Cubebox", []))[:10]],
            cycle_duration_ms=performance.get("cycle_ms", 0.0),
            inference_time_ms=performance.get("inference_ms", 0.0),
        )

        if self.save_screenshots and screenshot is not None:
            try:
                import cv2
                img_path = self.output_dir / f"{self._current_match}_f{self._frame_counter:05d}.jpg"
                cv2.imwrite(str(img_path), screenshot, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                frame.screenshot_path = str(img_path)
            except Exception as e:
                logger.debug("[ENRICHED] Screenshot error: %s", e)

        if v2_integrator and v2_integrator._event_store:
            try:
                from core.event_store import DomainEventType
                recent = v2_integrator._event_store.get_events_between(time.time() - 5.0, time.time(), [DomainEventType.PLAYER_DAMAGED, DomainEventType.ENEMY_HIT, DomainEventType.ACTION_TAKEN])
                frame.recent_events = [{"type": e.event_type.value, "timestamp": e.timestamp, "payload": e.payload} for e in recent[-5:]]
            except Exception:
                pass

        self._buffer.append(frame)
        if len(self._buffer) >= 100:
            self._flush_buffer()
        return frame

    def end_match(self, result: str, metrics=None):
        if not self._current_match:
            return
        self._flush_buffer()
        meta = {"match_id": self._current_match, "session_id": self._session_id, "result": result, "total_frames": self._frame_counter, "metrics": metrics or {}, "ended_at": time.time()}
        with open(self.output_dir / f"{self._current_match}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("[ENRICHED] Match ended: %s (%s, %d frames)", self._current_match, result, self._frame_counter)
        self._current_match = None

    def _flush_buffer(self):
        if not self._buffer or not self._current_match:
            return
        with open(self.output_dir / f"{self._current_match}_frames.jsonl", "a", encoding="utf-8") as f:
            for frame in self._buffer:
                f.write(json.dumps(asdict(frame), default=str) + "\n")
        self._buffer.clear()

    def get_stats(self):
        return {"session_id": self._session_id, "current_match": self._current_match, "frames_in_buffer": len(self._buffer), "total_frames": self._frame_counter, "output_dir": str(self.output_dir)}
