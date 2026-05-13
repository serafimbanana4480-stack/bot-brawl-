"""Tracker Integration - Multi-object tracking with ByteTrack/DeepSORT"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class TrackerIntegration:
    def __init__(self, tracker_type: str = "bytetrack",
                 max_age: int = 30,
                 iou_threshold: float = 0.3,
                 embedder: Optional[str] = None):
        self.tracker_type = tracker_type
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.embedder = embedder
        self.tracks = {}
        self.track_id_counter = 0
        self._tracker = None
        
        self._initialize_tracker()
    
    def _initialize_tracker(self):
        if self.tracker_type == "bytetrack":
            try:
                from bytetrack import BYTETracker
                self._tracker = BYTETracker(
                    track_thresh=0.5,
                    track_buffer=30,
                    match_thresh=0.8,
                    frame_rate=30,
                )
            except ImportError:
                print("ByteTrack not installed. Using simple tracker.")
                self._tracker = None
        elif self.tracker_type == "deepsort":
            try:
                from deep_sort_realtime.deepsort_tracker import DeepSort
                self._tracker = DeepSort(
                    max_age=self.max_age,
                    embedder=self.embedder or "mobilenet",
                )
            except ImportError:
                print("DeepSort not installed. Using simple tracker.")
                self._tracker = None
    
    def update(self, detections: List[Dict[str, Any]], 
               frame: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        if not detections:
            return []
        
        if self._tracker is None:
            return self._simple_update(detections)
        
        if self.tracker_type == "bytetrack":
            return self._bytetrack_update(detections)
        elif self.tracker_type == "deepsort":
            return self._deepsort_update(detections, frame)
        
        return self._simple_update(detections)
    
    def _bytetrack_update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not hasattr(self, '_byte_tracks'):
            self._byte_tracks = []
        
        dets = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w, h = x2 - x1, y2 - y1
            dets.append([x1, y1, w, h, det["confidence"]])
        
        if dets:
            try:
                online_targets = self._tracker.update(
                    np.array(dets),
                    None,
                    None,
                )
                
                tracked = []
                for track in online_targets:
                    x1, y1, w, h, track_id = track[:5]
                    tracked.append({
                        "bbox": (int(x1), int(y1), int(x1+w), int(y1+h)),
                        "track_id": int(track_id),
                        "confidence": det.get("confidence", 0.5),
                        "class_name": det.get("class_name", "unknown"),
                    })
                
                self._byte_tracks = tracked
                return tracked
            except Exception:
                pass
        
        return self._simple_update(detections)
    
    def _deepsort_update(self, detections: List[Dict[str, Any]],
                        frame: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        if frame is None:
            return self._simple_update(detections)
        
        bboxes = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bboxes.append([x1, y1, x2-x1, y2-y1])
        
        if bboxes and len(bboxes) > 0:
            try:
                deepsort_detections = self._tracker.update(
                    np.array(bboxes),
                    detections[0].get("confidence", 0.5) if detections else 0.5,
                    frame,
                )
                
                tracked = []
                for track in deepsort_detections:
                    x1, y1, w, h = track.to_tlbr()
                    tracked.append({
                        "bbox": (int(x1), int(y1), int(x2 := x1+w), int(y2 := y1+h)),
                        "track_id": track.track_id,
                        "confidence": track.confidence,
                        "class_name": detections[0].get("class_name", "unknown") if detections else "unknown",
                    })
                
                return tracked
            except Exception:
                pass
        
        return self._simple_update(detections)
    
    def _simple_update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tracked = []
        
        for det in detections:
            track_id = id(det) % 1000
            
            if not hasattr(self, '_last_assignments'):
                self._last_assignments = {}
            
            self._last_assignments[track_id] = det["bbox"]
            
            tracked.append({
                "bbox": det["bbox"],
                "track_id": track_id,
                "confidence": det["confidence"],
                "class_name": det.get("class_name", "unknown"),
            })
        
        return tracked
    
    def get_track_history(self, track_id: int, 
                         max_history: int = 30) -> List[Tuple[int, int]]:
        if not hasattr(self, '_track_histories'):
            self._track_histories = {}
        
        return self._track_histories.get(track_id, [])[-max_history:]
    
    def get_active_tracks(self) -> List[int]:
        if hasattr(self, '_byte_tracks'):
            return [t["track_id"] for t in self._byte_tracks]
        return list(self._last_assignments.keys()) if hasattr(self, '_last_assignments') else []
    
    def reset(self):
        self.tracks = {}
        self.track_id_counter = 0
        if hasattr(self, '_byte_tracks'):
            self._byte_tracks = []
        if hasattr(self, '_track_histories'):
            self._track_histories = {}
