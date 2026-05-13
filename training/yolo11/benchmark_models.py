"""
benchmark_models.py

Benchmark script to compare YOLO11 vs YOLOv8 performance.

Measures inference latency, throughput, and accuracy metrics.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)


def benchmark_model(model, test_images: List, model_name: str = "model") -> Dict:
    """
    Benchmark model performance.
    
    Args:
        model: YOLO model instance
        test_images: List of test images (paths or numpy arrays)
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary with benchmark metrics
    """
    logger.info(f"Benchmarking {model_name} on {len(test_images)} images...")
    
    latencies = []
    detection_counts = []
    
    # Warmup
    logger.info("Running warmup...")
    for i in range(min(3, len(test_images))):
        try:
            if hasattr(model, '__call__'):
                _ = model(test_images[i], verbose=False)
        except Exception as e:
            logger.warning(f"Warmup error: {e}")
    
    # Benchmark
    logger.info("Running benchmark...")
    for img in test_images:
        try:
            start = time.perf_counter()
            
            if hasattr(model, '__call__'):
                results = model(img, verbose=False)
            else:
                logger.error("Model has no __call__ method")
                continue
            
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            # Count detections
            if results and len(results) > 0 and hasattr(results[0], 'boxes'):
                detection_counts.append(len(results[0].boxes))
            else:
                detection_counts.append(0)
                
        except Exception as e:
            logger.error(f"Benchmark error on image: {e}")
            continue
    
    if not latencies:
        logger.error("No successful benchmarks")
        return {"error": "No successful benchmarks"}
    
    # Calculate metrics
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    std_latency = np.std(latencies)
    throughput_fps = 1000 / avg_latency if avg_latency > 0 else 0
    avg_detections = np.mean(detection_counts)
    
    results = {
        "model_name": model_name,
        "num_images": len(latencies),
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(min_latency, 2),
        "max_latency_ms": round(max_latency, 2),
        "std_latency_ms": round(std_latency, 2),
        "throughput_fps": round(throughput_fps, 2),
        "avg_detections": round(avg_detections, 2)
    }
    
    logger.info(f"Benchmark results for {model_name}:")
    logger.info(f"  Avg latency: {avg_latency:.2f}ms")
    logger.info(f"  Throughput: {throughput_fps:.2f} FPS")
    logger.info(f"  Avg detections: {avg_detections:.2f}")
    
    return results


def compare_models(model1, model2, test_images: List, name1: str = "Model 1", name2: str = "Model 2") -> Dict:
    """
    Compare two models and return comparison metrics.
    
    Args:
        model1: First model
        model2: Second model
        test_images: List of test images
        name1: Name of first model
        name2: Name of second model
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing {name1} vs {name2}...")
    
    results1 = benchmark_model(model1, test_images, name1)
    results2 = benchmark_model(model2, test_images, name2)
    
    if "error" in results1 or "error" in results2:
        return {"error": "One or both benchmarks failed"}
    
    # Calculate improvements
    latency_improvement = (results1["avg_latency_ms"] - results2["avg_latency_ms"]) / results1["avg_latency_ms"] * 100
    throughput_improvement = (results2["throughput_fps"] - results1["throughput_fps"]) / results1["throughput_fps"] * 100
    
    comparison = {
        name1: results1,
        name2: results2,
        "latency_improvement_percent": round(latency_improvement, 2),
        "throughput_improvement_percent": round(throughput_improvement, 2),
        "winner": name2 if latency_improvement > 0 else name1
    }
    
    logger.info(f"Comparison results:")
    logger.info(f"  Latency improvement: {latency_improvement:.2f}%")
    logger.info(f"  Throughput improvement: {throughput_improvement:.2f}%")
    logger.info(f"  Winner: {comparison['winner']}")
    
    return comparison


def generate_test_images(image_dir: Path, num_images: int = 10) -> List:
    """
    Generate test images from directory.
    
    Args:
        image_dir: Directory containing images
        num_images: Number of images to use
        
    Returns:
        List of image paths
    """
    image_extensions = ['.png', '.jpg', '.jpeg']
    images = []
    
    for ext in image_extensions:
        images.extend(image_dir.glob(f"*{ext}"))
    
    if len(images) == 0:
        logger.warning(f"No images found in {image_dir}")
        return []
    
    # Limit to num_images
    if len(images) > num_images:
        images = images[:num_images]
    
    logger.info(f"Using {len(images)} test images from {image_dir}")
    return [str(img) for img in images]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from ultralytics import YOLO
    
    # Load models
    print("Loading models...")
    yolo_v8 = YOLO("yolov8n.pt")
    yolo_11 = YOLO("yolo11n.pt")
    
    # Generate test images
    test_dir = Path("./test_images")
    test_images = generate_test_images(test_dir, num_images=10)
    
    if not test_images:
        print("No test images available, using dummy data")
        # Create dummy images
        test_images = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(10)]
    
    # Compare
    comparison = compare_models(yolo_v8, yolo_11, test_images, "YOLOv8n", "YOLO11n")
    
    print("\nFinal Results:")
    print(f"YOLOv8n: {comparison['YOLOv8n']['avg_latency_ms']}ms, {comparison['YOLOv8n']['throughput_fps']} FPS")
    print(f"YOLO11n: {comparison['YOLO11n']['avg_latency_ms']}ms, {comparison['YOLO11n']['throughput_fps']} FPS")
    print(f"Improvement: {comparison['latency_improvement_percent']}% latency, {comparison['throughput_improvement_percent']}% throughput")
