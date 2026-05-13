"""
model_switcher.py

Intelligent model switching logic for YOLO11 and YOLO-World.

Provides automatic fallback and model selection based on detection confidence
and context.
"""

import logging
from typing import Optional, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Container for detection results with metadata."""
    detections: List
    model_name: str
    confidence_score: float
    fallback_used: bool = False


class ModelSwitcher:
    """
    Switch between YOLO11 and YOLO-World based on context.
    
    Uses YOLO11 as primary model with automatic fallback to YOLO-World
    when detection confidence is low or when detecting unknown objects.
    """
    
    def __init__(self, primary_model, fallback_model, confidence_threshold: float = 0.3):
        """
        Initialize model switcher.
        
        Args:
            primary_model: Primary detection model (YOLO11)
            fallback_model: Fallback model (YOLO-World)
            confidence_threshold: Threshold below which to use fallback
        """
        self.primary = primary_model
        self.fallback = fallback_model
        self.confidence_threshold = confidence_threshold
        self.primary_model_name = getattr(primary_model, '__class__', type(primary_model)).__name__
        self.fallback_model_name = getattr(fallback_model, '__class__', type(fallback_model)).__name__
        
        # Statistics
        self.primary_usage_count = 0
        self.fallback_usage_count = 0
        self.total_detections = 0
        
    def detect(self, image, use_fallback: bool = False, conf_threshold: Optional[float] = None):
        """
        Detect with primary model, fallback to YOLO-World if needed.
        
        Args:
            image: Input image
            use_fallback: Force use of fallback model
            conf_threshold: Override default confidence threshold
            
        Returns:
            DetectionResult with detections and metadata
        """
        conf_threshold = conf_threshold or self.confidence_threshold
        
        # Force fallback if requested
        if use_fallback:
            logger.debug("Using fallback model (forced)")
            results = self._run_fallback(image, conf_threshold)
            self.fallback_usage_count += 1
            return DetectionResult(
                detections=results,
                model_name=self.fallback_model_name,
                confidence_score=0.0,
                fallback_used=True
            )
        
        # Try primary model first
        try:
            results = self._run_primary(image, conf_threshold)
            
            # Check if detection confidence is too low
            if results and len(results) > 0 and hasattr(results[0], 'boxes'):
                confidences = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else []
                max_conf = confidences.max() if len(confidences) > 0 else 0.0
                
                if max_conf < self.confidence_threshold:
                    logger.debug(f"Primary model confidence too low ({max_conf:.3f}), falling back")
                    results = self._run_fallback(image, conf_threshold)
                    self.fallback_usage_count += 1
                    return DetectionResult(
                        detections=results,
                        model_name=self.fallback_model_name,
                        confidence_score=max_conf,
                        fallback_used=True
                    )
                else:
                    self.primary_usage_count += 1
                    self.total_detections += len(confidences)
                    return DetectionResult(
                        detections=results,
                        model_name=self.primary_model_name,
                        confidence_score=max_conf,
                        fallback_used=False
                    )
            else:
                # No detections, try fallback
                logger.debug("Primary model no detections, trying fallback")
                results = self._run_fallback(image, conf_threshold)
                self.fallback_usage_count += 1
                return DetectionResult(
                    detections=results,
                    model_name=self.fallback_model_name,
                    confidence_score=0.0,
                    fallback_used=True
                )
                
        except Exception as e:
            logger.warning(f"Primary model failed, falling back to YOLO-World: {e}")
            results = self._run_fallback(image, conf_threshold)
            self.fallback_usage_count += 1
            return DetectionResult(
                detections=results,
                model_name=self.fallback_model_name,
                confidence_score=0.0,
                fallback_used=True
            )
    
    def _run_primary(self, image, conf_threshold: float):
        """Run primary model detection."""
        if hasattr(self.primary, 'detect'):
            return self.primary.detect(image, conf=conf_threshold)
        elif hasattr(self.primary, '__call__'):
            return self.primary(image, conf=conf_threshold, verbose=False)
        else:
            logger.error("Primary model has no detect or __call__ method")
            return None
    
    def _run_fallback(self, image, conf_threshold: float):
        """Run fallback model detection."""
        if hasattr(self.fallback, 'detect'):
            return self.fallback.detect(image, conf_threshold=conf_threshold)
        elif hasattr(self.fallback, '__call__'):
            return self.fallback(image, conf=conf_threshold, verbose=False)
        else:
            logger.error("Fallback model has no detect or __call__ method")
            return None
    
    def get_statistics(self) -> Dict:
        """Get usage statistics."""
        total = self.primary_usage_count + self.fallback_usage_count
        return {
            "primary_usage": self.primary_usage_count,
            "fallback_usage": self.fallback_usage_count,
            "total_detections": self.total_detections,
            "fallback_rate": self.fallback_usage_count / total if total > 0 else 0.0,
            "primary_model": self.primary_model_name,
            "fallback_model": self.fallback_model_name
        }
    
    def reset_statistics(self):
        """Reset usage statistics."""
        self.primary_usage_count = 0
        self.fallback_usage_count = 0
        self.total_detections = 0
