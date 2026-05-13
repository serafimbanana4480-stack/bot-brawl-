"""
replay_parser.py

Parser for gameplay recordings and replays.

Extracts frames, actions, and metadata from gameplay recordings
created by the gameplay recorder in Phase 1.
"""

import cv2
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GameEvent:
    """Represents a single game event."""
    timestamp: float  # Time in seconds from start
    event_type: str  # "attack", "move", "super", "death", "kill", etc.
    position: Tuple[float, float]  # (x, y) position
    data: Dict[str, Any]  # Additional event data


@dataclass
class ReplayMetadata:
    """Metadata extracted from a replay."""
    video_path: str
    duration: float  # Duration in seconds
    fps: int
    resolution: Tuple[int, int]
    total_frames: int
    brawler: Optional[str] = None
    map_name: Optional[str] = None
    result: Optional[str] = None  # "win", "loss", "draw"
    trophies_gained: int = 0
    kills: int = 0
    deaths: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0


@dataclass
class ParsedReplay:
    """Complete parsed replay data."""
    metadata: ReplayMetadata
    events: List[GameEvent]
    frames: Optional[np.ndarray] = None  # Optional: all frames (memory intensive)
    frame_indices: List[int] = None  # Frame indices for events


class ReplayParser:
    """
    Parser for gameplay recordings.
    
    Parses video files and associated action logs to extract
    structured replay data for analysis.
    """
    
    def __init__(self, video_path: Path, action_log_path: Optional[Path] = None):
        """
        Initialize parser.
        
        Args:
            video_path: Path to gameplay video file
            action_log_path: Optional path to action log JSON file
        """
        self.video_path = Path(video_path)
        self.action_log_path = Path(action_log_path) if action_log_path else None
        
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.metadata: Optional[ReplayMetadata] = None
        
    def open(self) -> bool:
        """Open video file and extract metadata."""
        try:
            self.video_capture = cv2.VideoCapture(str(self.video_path))
            if not self.video_capture.isOpened():
                logger.error(f"Failed to open video: {self.video_path}")
                return False
            
            # Extract video metadata
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            self.metadata = ReplayMetadata(
                video_path=str(self.video_path),
                duration=duration,
                fps=int(fps),
                resolution=(width, height),
                total_frames=frame_count
            )
            
            logger.info(f"Opened replay: {self.video_path.name}")
            logger.info(f"  Duration: {duration:.2f}s, FPS: {fps}, Frames: {frame_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening video: {e}")
            return False
    
    def close(self):
        """Close video file."""
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
    
    def load_action_log(self) -> List[Dict[str, Any]]:
        """Load action log from JSON file."""
        if not self.action_log_path or not self.action_log_path.exists():
            logger.warning(f"Action log not found: {self.action_log_path}")
            return []
        
        try:
            with open(self.action_log_path, 'r') as f:
                actions = json.load(f)
            logger.info(f"Loaded {len(actions)} actions from log")
            return actions
        except Exception as e:
            logger.error(f"Error loading action log: {e}")
            return []
    
    def extract_events(self, actions: List[Dict[str, Any]]) -> List[GameEvent]:
        """
        Extract game events from action log.
        
        Args:
            actions: List of action dictionaries from action log
            
        Returns:
            List of GameEvent objects
        """
        events = []
        
        for action in actions:
            try:
                event_type = action.get('type', 'unknown')
                timestamp = action.get('timestamp', 0.0)
                position = action.get('position', (0, 0))
                
                # Normalize position if it's a list
                if isinstance(position, list) and len(position) >= 2:
                    position = (float(position[0]), float(position[1]))
                
                # Extract additional data
                data = {k: v for k, v in action.items() 
                       if k not in ['type', 'timestamp', 'position']}
                
                event = GameEvent(
                    timestamp=timestamp,
                    event_type=event_type,
                    position=position,
                    data=data
                )
                events.append(event)
                
            except Exception as e:
                logger.warning(f"Error parsing action: {e}")
        
        logger.info(f"Extracted {len(events)} events")
        return events
    
    def get_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """
        Get frame at specific time.
        
        Args:
            time_seconds: Time in seconds from start
            
        Returns:
            Frame as numpy array or None if failed
        """
        if not self.video_capture or not self.metadata:
            return None
        
        try:
            frame_number = int(time_seconds * self.metadata.fps)
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.video_capture.read()
            
            if ret:
                return frame
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error getting frame at {time_seconds}s: {e}")
            return None
    
    def get_frame_at_index(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get frame at specific index.
        
        Args:
            frame_index: Frame index
            
        Returns:
            Frame as numpy array or None if failed
        """
        if not self.video_capture:
            return None
        
        try:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.video_capture.read()
            
            if ret:
                return frame
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error getting frame at index {frame_index}: {e}")
            return None
    
    def parse(self, load_frames: bool = False) -> ParsedReplay:
        """
        Parse complete replay.
        
        Args:
            load_frames: If True, load all frames (memory intensive)
            
        Returns:
            ParsedReplay object
        """
        if not self.open():
            raise RuntimeError("Failed to open video file")
        
        try:
            # Load action log
            actions = self.load_action_log()
            
            # Extract events
            events = self.extract_events(actions)
            
            # Optionally load all frames
            frames = None
            frame_indices = []
            
            if load_frames:
                logger.warning("Loading all frames - this may use a lot of memory")
                frames = []
                for i in range(self.metadata.total_frames):
                    frame = self.get_frame_at_index(i)
                    if frame is not None:
                        frames.append(frame)
                
                if frames:
                    frames = np.array(frames)
            
            # Calculate frame indices for events
            for event in events:
                frame_idx = int(event.timestamp * self.metadata.fps)
                frame_indices.append(frame_idx)
            
            parsed = ParsedReplay(
                metadata=self.metadata,
                events=events,
                frames=frames,
                frame_indices=frame_indices
            )
            
            logger.info(f"Successfully parsed replay: {len(events)} events")
            return parsed
            
        finally:
            self.close()
    
    def extract_key_frames(self, events: List[GameEvent], 
                          window_seconds: float = 2.0) -> List[Tuple[int, np.ndarray]]:
        """
        Extract key frames around important events.
        
        Args:
            events: List of game events
            window_seconds: Time window before/after event to extract
            
        Returns:
            List of (frame_index, frame) tuples
        """
        if not self.open():
            return []
        
        try:
            key_frames = []
            window_frames = int(window_seconds * self.metadata.fps)
            
            for event in events:
                # Only extract frames for important event types
                if event.event_type in ['attack', 'super', 'death', 'kill']:
                    center_frame = int(event.timestamp * self.metadata.fps)
                    
                    # Extract frames around the event
                    start_frame = max(0, center_frame - window_frames)
                    end_frame = min(self.metadata.total_frames, center_frame + window_frames + 1)
                    
                    for frame_idx in range(start_frame, end_frame):
                        frame = self.get_frame_at_index(frame_idx)
                        if frame is not None:
                            key_frames.append((frame_idx, frame))
            
            logger.info(f"Extracted {len(key_frames)} key frames around events")
            return key_frames
            
        finally:
            self.close()
    
    def save_parsed_replay(self, parsed: ParsedReplay, output_path: Path):
        """
        Save parsed replay to JSON file.
        
        Args:
            parsed: ParsedReplay object
            output_path: Path to save JSON file
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            data = {
                'metadata': asdict(parsed.metadata),
                'events': [asdict(event) for event in parsed.events],
                'frame_indices': parsed.frame_indices,
                # Note: frames are not saved (too large for JSON)
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved parsed replay to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving parsed replay: {e}")
