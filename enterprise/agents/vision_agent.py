"""Vision Agent - Computer vision processing and scene understanding"""

import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType, ConfidenceScore
from ..orchestration.event_bus import EventBus, EventType


@dataclass
class DetectedObject:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    track_id: Optional[int] = None


@dataclass
class VisionAnalysis:
    frame_id: str
    timestamp: float
    objects: List[DetectedObject]
    scene_context: Dict[str, Any]
    heatmap: Optional[np.ndarray] = None


class VisionAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        self.frame_buffer: List[np.ndarray] = []
        self.max_buffer_size = 30
        self.detection_history: List[VisionAnalysis] = []
        self.current_analysis: Optional[VisionAnalysis] = None
        self.minimap_state: Optional[Dict[str, Any]] = None
        
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        action = message.content.get("action", "analyze")
        
        try:
            if action == "analyze":
                result = await self._analyze_frame(message.content)
            elif action == "detect":
                result = await self._detect_objects(message.content)
            elif action == "track":
                result = await self._track_objects(message.content)
            elif action == "minimap":
                result = await self._analyze_minimap(message.content)
            elif action == "predict":
                result = await self._predict_movement(message.content)
            else:
                result = {"error": f"Unknown action: {action}"}
            
            return AgentResponse(
                success=True,
                message=message,
                data=result,
                confidence=0.9,
                processing_time=time.time() - start_time,
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                message=message,
                error=str(e),
                processing_time=time.time() - start_time,
            )
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        current_analysis = context.get("current_analysis")
        
        if not current_analysis:
            return {
                "scene_summary": "No active vision data",
                "threats_detected": 0,
                "opportunities_detected": 0,
                "confidence": 0.0,
            }
        
        threats = [o for o in current_analysis.objects if self._is_threat(o)]
        opportunities = [o for o in current_analysis.objects if self._is_opportunity(o)]
        
        return {
            "scene_summary": self._generate_scene_summary(current_analysis),
            "threats_detected": len(threats),
            "opportunities_detected": len(opportunities),
            "critical_objects": [o.class_name for o in current_analysis.objects if o.confidence > 0.8],
            "confidence": sum(o.confidence for o in current_analysis.objects) / max(1, len(current_analysis.objects)),
        }
    
    async def _analyze_frame(self, content: Dict[str, Any]) -> Dict[str, Any]:
        frame = content.get("frame")
        frame_id = content.get("frame_id", str(int(time.time())))
        
        if frame is not None:
            self._add_to_buffer(frame)
        
        detections = await self._perform_detection(content)
        
        scene_context = self._extract_scene_context(detections, content)
        
        analysis = VisionAnalysis(
            frame_id=frame_id,
            timestamp=time.time(),
            objects=[DetectedObject(**d) if isinstance(d, dict) else d for d in detections],
            scene_context=scene_context,
        )
        
        self.current_analysis = analysis
        self.detection_history.append(analysis)
        
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
        
        await self.event_bus.publish(Event(
            source=self.id,
            type=EventType.VISION_UPDATE,
            data={
                "frame_id": frame_id,
                "object_count": len(detections),
                "scene_context": scene_context,
            },
        ))
        
        return {
            "frame_id": frame_id,
            "objects": [
                {
                    "class_name": o.class_name,
                    "confidence": o.confidence,
                    "bbox": o.bbox,
                    "track_id": o.track_id,
                }
                for o in analysis.objects
            ],
            "scene_context": scene_context,
            "heatmap": analysis.heatmap.tolist() if analysis.heatmap is not None else None,
        }
    
    async def _detect_objects(self, content: Dict[str, Any]) -> Dict[str, Any]:
        frame = content.get("frame")
        classes = content.get("classes", [])
        confidence_threshold = content.get("confidence", 0.5)
        
        detections = await self._perform_detection(content)
        
        filtered = [d for d in detections if d.get("confidence", 0) >= confidence_threshold]
        
        if classes:
            filtered = [d for d in filtered if d.get("class_name") in classes]
        
        return {
            "detections": filtered,
            "total_detections": len(detections),
            "filtered_count": len(filtered),
        }
    
    async def _track_objects(self, content: Dict[str, Any]) -> Dict[str, Any]:
        detections = content.get("detections", [])
        frame = content.get("frame")
        
        tracked = []
        for det in detections:
            tracked_det = DetectedObject(
                class_id=det.get("class_id", 0),
                class_name=det.get("class_name", "unknown"),
                confidence=det.get("confidence", 0.5),
                bbox=det.get("bbox", (0, 0, 0, 0)),
                track_id=det.get("track_id", np.random.randint(0, 100)),
            )
            tracked.append(tracked_det)
        
        return {
            "tracked_objects": [
                {
                    "class_name": t.class_name,
                    "track_id": t.track_id,
                    "confidence": t.confidence,
                    "position_history": [(t.bbox[0], t.bbox[1])],
                }
                for t in tracked
            ],
            "tracking_quality": 0.85,
        }
    
    async def _analyze_minimap(self, content: Dict[str, Any]) -> Dict[str, Any]:
        minimap_frame = content.get("minimap")
        
        if minimap_frame is None:
            return {"error": "No minimap data", "detected": False}
        
        hero_position = self._detect_minimap_hero(minimap_frame)
        enemy_positions = self._detect_minimap_enemies(minimap_frame)
        objective_position = self._detect_minimap_objective(minimap_frame)
        
        self.minimap_state = {
            "hero": hero_position,
            "enemies": enemy_positions,
            "objective": objective_position,
            "timestamp": time.time(),
        }
        
        return {
            "detected": True,
            "hero_position": hero_position,
            "enemy_positions": enemy_positions,
            "objective_position": objective_position,
            "map_control": self._calculate_map_control(hero_position, enemy_positions),
        }
    
    async def _predict_movement(self, content: Dict[str, Any]) -> Dict[str, Any]:
        track_id = content.get("track_id")
        history = content.get("history", [])
        
        if len(history) < 3:
            return {
                "predicted_position": history[-1] if history else (0, 0),
                "confidence": 0.3,
            }
        
        predicted = self._linear_prediction(history)
        confidence = min(0.9, 0.5 + len(history) * 0.1)
        
        return {
            "track_id": track_id,
            "predicted_position": predicted,
            "confidence": confidence,
            "prediction_horizon": 1.0,
        }
    
    async def _perform_detection(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "class_id": 0,
                "class_name": "enemy",
                "confidence": 0.85,
                "bbox": (100, 100, 150, 180),
            },
            {
                "class_id": 1,
                "class_name": "projectile",
                "confidence": 0.72,
                "bbox": (200, 150, 210, 160),
            },
        ]
    
    def _extract_scene_context(self, detections: List[Dict], content: Dict[str, Any]) -> Dict[str, Any]:
        class_counts = {}
        for det in detections:
            name = det.get("class_name", "unknown")
            class_counts[name] = class_counts.get(name, 0) + 1
        
        return {
            "object_counts": class_counts,
            "total_objects": len(detections),
            "timestamp": time.time(),
            "risk_level": self._assess_risk_level(class_counts),
        }
    
    def _assess_risk_level(self, class_counts: Dict[str, int]) -> str:
        threat_count = class_counts.get("enemy", 0) + class_counts.get("danger", 0)
        if threat_count > 3:
            return "high"
        elif threat_count > 1:
            return "medium"
        return "low"
    
    def _add_to_buffer(self, frame: np.ndarray):
        if len(self.frame_buffer) >= self.max_buffer_size:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(frame)
    
    def _is_threat(self, obj: DetectedObject) -> bool:
        threat_classes = ["enemy", "projectile", "danger_zone", "trap"]
        return obj.class_name in threat_classes and obj.confidence > 0.6
    
    def _is_opportunity(self, obj: DetectedObject) -> bool:
        opportunity_classes = ["powerup", "health", "cover"]
        return obj.class_name in opportunity_classes and obj.confidence > 0.6
    
    def _generate_scene_summary(self, analysis: VisionAnalysis) -> str:
        enemies = [o for o in analysis.objects if o.class_name == "enemy"]
        allies = [o for o in analysis.objects if o.class_name == "ally"]
        powerups = [o for o in analysis.objects if o.class_name == "powerup"]
        
        summary = f"Scene: {len(enemies)} enemies, {len(allies)} allies"
        if powerups:
            summary += f", {len(powerups)} powerups nearby"
        
        return summary
    
    def _detect_minimap_hero(self, minimap: np.ndarray) -> Optional[Tuple[int, int]]:
        return (50, 50)
    
    def _detect_minimap_enemies(self, minimap: np.ndarray) -> List[Tuple[int, int]]:
        return [(100, 80), (120, 60)]
    
    def _detect_minimap_objective(self, minimap: np.ndarray) -> Optional[Tuple[int, int]]:
        return (200, 200)
    
    def _calculate_map_control(self, hero_pos: Tuple[int, int],
                              enemy_positions: List[Tuple[int, int]]) -> float:
        if not enemy_positions:
            return 0.8
        
        avg_enemy_x = sum(p[0] for p in enemy_positions) / len(enemy_positions)
        avg_enemy_y = sum(p[1] for p in enemy_positions) / len(enemy_positions)
        
        hero_distance = np.sqrt(hero_pos[0]**2 + hero_pos[1]**2)
        enemy_distance = np.sqrt(avg_enemy_x**2 + avg_enemy_y**2)
        
        if hero_distance < enemy_distance:
            return 0.7
        return 0.4
    
    def _linear_prediction(self, history: List[Tuple[float, float]]) -> Tuple[float, float]:
        if len(history) < 2:
            return history[-1] if history else (0, 0)
        
        dx = history[-1][0] - history[0][0]
        dy = history[-1][1] - history[0][1]
        
        return (history[-1][0] + dx, history[-1][1] + dy)
