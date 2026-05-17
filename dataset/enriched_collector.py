"""
dataset/enriched_collector.py

Extensao do GameplayCollector que grava GameState multimodal em cada frame.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from vision.game_state import GameState
    from vision.multimodal_pipeline import MultimodalPipeline
    HAS_MULTIMODAL = True
except ImportError:
    HAS_MULTIMODAL = False
    GameState = None
    MultimodalPipeline = None


class EnrichedFrameRecord:
    """FrameRecord estendido com GameState multimodal."""

    def __init__(
        self,
        timestamp: float,
        frame_id: int,
        state: str,
        detections: Optional[Dict] = None,
        action: Optional[Dict] = None,
        reward: float = 0.0,
        screenshot_path: Optional[str] = None,
        game_state_multimodal: Optional[Dict] = None,
        player_state: Optional[Dict] = None,
        hud_values: Optional[Dict] = None,
        vision_latency_ms: float = 0.0,
    ):
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.state = state
        self.detections = detections
        self.action = action
        self.reward = reward
        self.screenshot_path = screenshot_path
        self.game_state_multimodal = game_state_multimodal
        self.player_state = player_state
        self.hud_values = hud_values
        self.vision_latency_ms = vision_latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "state": self.state,
            "detections": self.detections,
            "action": self.action,
            "reward": self.reward,
            "screenshot_path": self.screenshot_path,
            "game_state_multimodal": self.game_state_multimodal,
            "player_state": self.player_state,
            "hud_values": self.hud_values,
            "vision_latency_ms": self.vision_latency_ms,
        }


class EnrichedGameplayCollector:
    """Coletor de gameplay que grava GameState multimodal em cada frame."""

    def __init__(
        self,
        base_dir: Path = None,
        enable_multimodal: bool = True,
        resolution: Tuple[int, int] = (1920, 1080),
        pipeline: Optional[Any] = None,
    ):
        self.base_dir = Path(base_dir) if base_dir else Path("dataset/enriched")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.enable_multimodal = enable_multimodal and HAS_MULTIMODAL
        self.resolution = resolution

        self._pipeline: Optional[Any] = pipeline
        self._has_pipeline = False

        self.frames_collected = 0
        self.episodes_collected = 0
        self.total_vision_latency_ms = 0.0

        self._current_episode_frames: List[EnrichedFrameRecord] = []
        self._episode_start_time: Optional[float] = None

        logger.info(
            "[ENRICHED] Inicializado: dir=%s, multimodal=%s",
            self.base_dir,
            self.enable_multimodal,
        )

    def _ensure_pipeline(self) -> bool:
        if self._pipeline is not None:
            return True
        if not self.enable_multimodal or not HAS_MULTIMODAL:
            return False
        try:
            self._pipeline = MultimodalPipeline(
                resolution=self.resolution,
                enable_ocr=True,
                enable_player_state=True,
                enable_hud=True,
            )
            self._has_pipeline = True
            logger.info("[ENRICHED] MultimodalPipeline ligado")
            return True
        except Exception as exc:
            logger.warning("[ENRICHED] MultimodalPipeline indisponivel: %s", exc)
            self._has_pipeline = False
            return False

    def record_frame(
        self,
        screenshot: np.ndarray,
        yolo_detections: List[Dict],
        game_state_hint: str,
        action: Optional[Dict] = None,
        reward: float = 0.0,
        frame_id: int = 0,
        screenshot_path: Optional[str] = None,
    ) -> EnrichedFrameRecord:
        t0 = time.time()

        game_state_multimodal = None
        player_state = None
        hud_values = None
        vision_latency = 0.0

        if self.enable_multimodal and self._ensure_pipeline():
            try:
                game_state = self._pipeline.process(
                    screenshot=screenshot,
                    yolo_detections=yolo_detections,
                    game_state_hint=game_state_hint,
                    frame_id=frame_id,
                )
                game_state_multimodal = game_state.to_dict()
                player_state = game_state.player.to_dict()
                hud_values = game_state.hud.to_dict()
                vision_latency = game_state.latency_ms
                self.total_vision_latency_ms += vision_latency
            except Exception as exc:
                logger.warning("[ENRICHED] Pipeline falhou: %s", exc)

        record = EnrichedFrameRecord(
            timestamp=time.time(),
            frame_id=frame_id,
            state=game_state_hint,
            detections={"count": len(yolo_detections), "items": yolo_detections} if yolo_detections else None,
            action=action,
            reward=reward,
            screenshot_path=screenshot_path,
            game_state_multimodal=game_state_multimodal,
            player_state=player_state,
            hud_values=hud_values,
            vision_latency_ms=vision_latency,
        )

        self._current_episode_frames.append(record)
        self.frames_collected += 1

        elapsed = (time.time() - t0) * 1000
        logger.debug(
            "[ENRICHED] Frame %d gravado em %.1f ms (vision=%.1f ms)",
            frame_id,
            elapsed,
            vision_latency,
        )

        return record

    def start_episode(self, episode_id: Optional[str] = None) -> str:
        if episode_id is None:
            episode_id = f"ep_{int(time.time())}"
        self._current_episode_frames = []
        self._episode_start_time = time.time()
        logger.info("[ENRICHED] Episodio iniciado: %s", episode_id)
        return episode_id

    def end_episode(
        self,
        episode_id: str,
        result: str = "unknown",
        metrics: Optional[Dict] = None,
    ) -> Path:
        duration = 0.0
        if self._episode_start_time is not None:
            duration = time.time() - self._episode_start_time

        episode_data = {
            "episode_id": episode_id,
            "start_time": self._episode_start_time,
            "end_time": time.time(),
            "duration_seconds": duration,
            "result": result,
            "metrics": metrics or {},
            "frame_count": len(self._current_episode_frames),
            "frames": [f.to_dict() for f in self._current_episode_frames],
        }

        output_file = self.base_dir / f"{episode_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False, default=str)

        self.episodes_collected += 1
        self._current_episode_frames = []
        self._episode_start_time = None

        logger.info(
            "[ENRICHED] Episodio %s salvo: %s (%d frames, %.1fs)",
            episode_id,
            output_file,
            episode_data["frame_count"],
            duration,
        )

        return output_file

    def get_stats(self) -> Dict[str, Any]:
        avg_latency = (
            self.total_vision_latency_ms / self.frames_collected
            if self.frames_collected > 0 else 0.0
        )
        return {
            "frames_collected": self.frames_collected,
            "episodes_collected": self.episodes_collected,
            "avg_vision_latency_ms": avg_latency,
            "multimodal_enabled": self.enable_multimodal,
            "pipeline_available": self._has_pipeline,
        }

    def reset(self) -> None:
        self.frames_collected = 0
        self.episodes_collected = 0
       
