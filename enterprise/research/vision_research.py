"""Vision Pipeline Research Agent - Computer Vision and detection research"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass

from ..agents.base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType
from ..orchestration.event_bus import EventBus, EventType


@dataclass
class VisionFramework:
    name: str
    repo_url: str
    type: str
    detection_models: List[str]
    tracking_algorithms: List[str]
    features: List[str]
    fps_performance: Dict[str, int]
    accuracy_score: float


@dataclass
class VisionResearchResult:
    frameworks: List[VisionFramework]
    models_available: Set[str]
    best_combinations: List[Dict]
    recommendations: List[str]


class VisionPipelineResearchAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        
        self.vision_frameworks = {
            "yolov8": {
                "name": "YOLOv8",
                "url": "ultralytics/ultralytics",
                "type": "detection",
                "detection_models": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                "tracking_algorithms": ["ByteTrack", "DeepSORT"],
                "features": ["segmentation", "classification", "pose", "oriented_bbox"],
                "fps_base": {"yolov8n": 300, "yolov8s": 200, "yolov8m": 100},
                "accuracy_base": {"yolov8n": 0.37, "yolov8s": 0.44, "yolov8m": 0.50},
            },
            "yolov5": {
                "name": "YOLOv5",
                "url": "ultralytics/yolov5",
                "type": "detection",
                "detection_models": ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov8x"],
                "tracking_algorithms": ["DeepSORT", "StrongSORT"],
                "features": ["detection", "classification"],
                "fps_base": {"yolov5n": 250, "yolov5s": 180, "yolov5m": 90},
                "accuracy_base": {"yolov5n": 0.35, "yolov5s": 0.42, "yolov5m": 0.48},
            },
            "yolov11": {
                "name": "YOLOv11",
                "url": "ultralytics/ultralytics",
                "type": "detection",
                "detection_models": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
                "tracking_algorithms": ["ByteTrack", "DeepSORT", "BoT-SORT"],
                "features": ["detection", "segmentation", "pose", "classification"],
                "fps_base": {"yolo11n": 350, "yolo11s": 250, "yolo11m": 120},
                "accuracy_base": {"yolo11n": 0.39, "yolo11s": 0.46, "yolo11m": 0.52},
            },
            "deepsort": {
                "name": "DeepSORT",
                "url": "nwojke/deep_sort",
                "type": "tracking",
                "detection_models": ["YOLOv3", "YOLOv4", "YOLOv5"],
                "tracking_algorithms": ["DeepSORT"],
                "features": ["re_id", "appearance_feature"],
                "fps_base": {"default": 30},
                "accuracy_base": {"default": 0.78},
            },
            "bytetrack": {
                "name": "ByteTrack",
                "url": "ifzhang/ByteTrack",
                "type": "tracking",
                "detection_models": ["YOLOX", "YOLOv5", "YOLOv6", "YOLOv8"],
                "tracking_algorithms": ["ByteTrack"],
                "features": ["high_speed", "association_by_detection"],
                "fps_base": {"default": 50},
                "accuracy_base": {"default": 0.82},
            },
            "sam2": {
                "name": "Segment Anything Model 2",
                "url": "facebookresearch/segment-anything-2",
                "type": "segmentation",
                "detection_models": ["sam2.1_b", "sam2.1_s", "sam2.1_m"],
                "tracking_algorithms": [],
                "features": ["image_seg", "video_seg", "prompt_based"],
                "fps_base": {"sam2.1_s": 40, "sam2.1_m": 25},
                "accuracy_base": {"default": 0.89},
            },
        }
        
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = asyncio.get_event_loop().time()
        action = message.content.get("action", "research")
        
        try:
            if action == "research":
                result = await self._research_vision_pipelines(message.content)
            elif action == "optimize":
                result = await self._optimize_vision_pipeline(message.content)
            elif action == "compare":
                result = await self._compare_vision_approaches(message.content)
            else:
                result = {"error": f"Unknown action: {action}"}
            
            return AgentResponse(
                success=True,
                message=message,
                data=result,
                confidence=0.85,
                processing_time=asyncio.get_event_loop().time() - start_time,
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                message=message,
                error=str(e),
                processing_time=asyncio.get_event_loop().time() - start_time,
            )
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "frameworks_available": len(self.vision_frameworks),
            "ready": True,
            "confidence": 0.9,
        }
    
    async def _research_vision_pipelines(self, content: Dict[str, Any]) -> Dict[str, Any]:
        requirements = content.get("requirements", {
            "min_fps": 30,
            "detection_needed": True,
            "tracking_needed": True,
        })
        
        suitable_frameworks = []
        all_models = set()
        
        for fw_key, fw_info in self.vision_frameworks.items():
            meets_fps = any(
                fps >= requirements["min_fps"] 
                for fps in fw_info.get("fps_base", {}).values()
            )
            
            if requirements.get("detection_needed") and not fw_info.get("detection_models"):
                continue
            
            if requirements.get("tracking_needed") and not fw_info.get("tracking_algorithms"):
                continue
            
            if meets_fps:
                framework = VisionFramework(
                    name=fw_info["name"],
                    repo_url=f"https://github.com/{fw_info['url']}",
                    type=fw_info["type"],
                    detection_models=fw_info.get("detection_models", []),
                    tracking_algorithms=fw_info.get("tracking_algorithms", []),
                    features=fw_info.get("features", []),
                    fps_performance=fw_info.get("fps_base", {}),
                    accuracy_score=fw_info.get("accuracy_base", {}).get("default", 0.75),
                )
                suitable_frameworks.append(framework)
                all_models.update(framework.detection_models)
                all_models.update(framework.tracking_algorithms)
        
        suitable_frameworks.sort(key=lambda x: x.accuracy_score, reverse=True)
        
        best_combinations = self._find_best_combinations(suitable_frameworks, requirements)
        
        await self.event_bus.publish(Event(
            source=self.id,
            type=EventType.DECISION_PROPOSED,
            data={
                "action": "vision_research_completed",
                "frameworks_found": len(suitable_frameworks),
                "models": list(all_models)[:10],
            },
        ))
        
        return {
            "suitable_frameworks": [
                {
                    "name": fw.name,
                    "repo_url": fw.repo_url,
                    "type": fw.type,
                    "detection_models": fw.detection_models,
                    "tracking_algorithms": fw.tracking_algorithms,
                    "features": fw.features,
                    "fps_performance": fw.fps_performance,
                    "accuracy_score": fw.accuracy_score,
                }
                for fw in suitable_frameworks
            ],
            "models_available": list(all_models),
            "best_combinations": best_combinations,
            "recommendations": self._generate_vision_recommendations(suitable_frameworks, requirements),
        }
    
    async def _optimize_vision_pipeline(self, content: Dict[str, Any]) -> Dict[str, Any]:
        current_pipeline = content.get("pipeline", {})
        target_fps = content.get("target_fps", 60)
        target_platform = content.get("platform", "desktop")
        
        optimizations = []
        
        if target_platform == "desktop":
            optimizations.append({
                "type": "model_size",
                "suggestion": "Use YOLOv8n or YOLOv8s for better FPS",
                "expected_gain": "20-40% FPS improvement",
            })
        elif target_platform == "mobile":
            optimizations.append({
                "type": "model_quantization",
                "suggestion": "Use INT8 quantization for mobile deployment",
                "expected_gain": "2-3x speedup with minimal accuracy loss",
            })
        
        optimizations.append({
            "type": "inference_optimization",
            "suggestion": "Use TensorRT for NVIDIA GPUs",
            "expected_gain": "2-5x speedup on supported hardware",
        })
        
        optimizations.append({
            "type": "batch_processing",
            "suggestion": "Batch frames when latency is acceptable",
            "expected_gain": "Better GPU utilization",
        })
        
        if target_fps > 30:
            optimizations.append({
                "type": "tracking_skip",
                "suggestion": f"Run tracking every {int(target_fps/25)} frames",
                "expected_gain": "Reduced computational load",
            })
        
        return {
            "current_pipeline": current_pipeline,
            "target_fps": target_fps,
            "optimizations": optimizations,
            "expected_final_fps": self._estimate_optimized_fps(current_pipeline, optimizations),
        }
    
    async def _compare_vision_approaches(self, content: Dict[str, Any]) -> Dict[str, Any]:
        approaches = content.get("approaches", ["yolov8", "yolov11", "bytetrack"])
        
        comparisons = []
        for approach_key in approaches:
            if approach_key in self.vision_frameworks:
                fw = self.vision_frameworks[approach_key]
                comparisons.append({
                    "name": fw["name"],
                    "type": fw["type"],
                    "speed": max(fw.get("fps_base", {}).values()) if fw.get("fps_base") else 0,
                    "accuracy": fw.get("accuracy_base", {}).get("default", 0),
                    "features": fw.get("features", []),
                    "suitable_for_realtime": max(fw.get("fps_base", {}).values(), default=0) > 30,
                })
        
        return {
            "approaches": comparisons,
            "best_for_speed": max(comparisons, key=lambda x: x["speed"])["name"] if comparisons else None,
            "best_for_accuracy": max(comparisons, key=lambda x: x["accuracy"])["name"] if comparisons else None,
            "recommended_combination": self._recommend_combination(comparisons),
        }
    
    def _find_best_combinations(self, frameworks: List[VisionFramework],
                               requirements: Dict) -> List[Dict]:
        combinations = []
        
        detection_fws = [f for f in frameworks if f.type == "detection"]
        tracking_fws = [f for f in frameworks if f.type == "tracking"]
        
        for det in detection_fws[:2]:
            for track in tracking_fws[:2]:
                combo_fps = min(
                    max(det.fps_performance.values()) if det.fps_performance else 30,
                    max(track.fps_performance.values()) if track.fps_performance else 30,
                )
                
                if combo_fps >= requirements.get("min_fps", 30):
                    combinations.append({
                        "detection": det.name,
                        "tracking": track.name,
                        "combined_fps": combo_fps,
                        "avg_accuracy": (det.accuracy_score + track.accuracy_score) / 2,
                        "compatibility": "high",
                    })
        
        combinations.sort(key=lambda x: x["combined_fps"] * x["avg_accuracy"], reverse=True)
        
        return combinations[:3]
    
    def _generate_vision_recommendations(self, frameworks: List[VisionFramework],
                                       requirements: Dict) -> List[str]:
        recommendations = []
        
        if not frameworks:
            recommendations.append("No suitable framework found - consider custom implementation")
            return recommendations
        
        best = frameworks[0]
        recommendations.append(
            f"Primary detection: {best.name} - {best.accuracy_score * 100:.0f}% accuracy"
        )
        
        tracking_fws = [f for f in frameworks if f.type == "tracking"]
        if tracking_fws:
            best_tracker = tracking_fws[0]
            recommendations.append(
                f"Primary tracking: {best_tracker.name} - {max(best_tracker.fps_performance.values())} FPS"
            )
        
        recommendations.append(
            "For real-time game bot: Use YOLOv8n + ByteTrack for best speed/accuracy trade-off"
        )
        
        if requirements.get("min_fps", 30) >= 60:
            recommendations.append(
                "For 60+ FPS: Consider model quantization or smaller model variants"
            )
        
        return recommendations
    
    def _recommend_combination(self, comparisons: List[Dict]) -> Dict:
        has_detection = any(c["type"] == "detection" for c in comparisons)
        has_tracking = any(c["type"] == "tracking" for c in comparisons)
        
        if has_detection and has_tracking:
            best_det = max([c for c in comparisons if c["type"] == "detection"],
                         key=lambda x: x["speed"])
            best_track = max([c for c in comparisons if c["type"] == "tracking"],
                           key=lambda x: x["speed"])
            return {
                "detection": best_det["name"],
                "tracking": best_track["name"],
                "estimated_fps": min(best_det.get("speed", 30), best_track.get("speed", 30)),
            }
        
        return {"single_model": comparisons[0]["name"] if comparisons else None}
    
    def _estimate_optimized_fps(self, pipeline: Dict, optimizations: List[Dict]) -> int:
        base_fps = pipeline.get("current_fps", 30)
        
        for opt in optimizations:
            opt_type = opt.get("type", "")
            if opt_type == "model_size":
                base_fps *= 1.3
            elif opt_type == "inference_optimization":
                base_fps *= 2.0
            elif opt_type == "batch_processing":
                base_fps *= 1.5
            elif opt_type == "tracking_skip":
                base_fps *= 1.2
        
        return int(base_fps)
