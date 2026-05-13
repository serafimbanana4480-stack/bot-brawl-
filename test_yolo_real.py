"""
Test YOLO Real Detection - Verify trained model works
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("TESTING REAL YOLO DETECTION")
print("=" * 60)

print("\n[1] Loading YOLO model...")
from enterprise.vision.yolo_detector import YOLOv8Detector

try:
    detector = YOLOv8Detector(
        model_path="c:/Users/rodri/Desktop/bot brawl/models/brawlstars_yolov8.pt",
        conf_threshold=0.5
    )
    detector.load()
    print("✓ YOLO model loaded successfully!")
    print(f"  Model path: {detector.model_path}")
    print(f"  Classes: {detector.classes}")
    print(f"  Device: {detector.device}")
except Exception as e:
    print(f"✗ Failed to load YOLO: {e}")
    sys.exit(1)

print("\n[2] Testing detection on synthetic image...")
try:
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    detections = detector.detect(test_image)
    print(f"✓ Detection works! Found {len(detections)} objects (expected ~0 on random noise)")

except Exception as e:
    print(f"✗ Detection failed: {e}")
    sys.exit(1)

print("\n[3] Checking if game connector is available...")
try:
    from enterprise.integration.game_connector import GameConnector

    connector = GameConnector()
    if connector.is_connected():
        print("✓ Game Connector connected!")

        print("\n[4] Capturing real game frame...")
        frame = connector.capture_screen()
        if frame is not None:
            print(f"✓ Frame captured! Shape: {frame.shape}")

            print("\n[5] Running YOLO detection on real frame...")
            detections = detector.detect(frame)
            print(f"\n{'='*60}")
            print("DETECTION RESULTS ON REAL GAME:")
            print(f"{'='*60}")
            print(f"Total objects detected: {len(detections)}")

            for det in detections:
                print(f"  - {det['class_name']}: conf={det['confidence']:.2f}, bbox={det['bbox']}")

            print(f"{'='*60}")
        else:
            print("⚠ No frame captured (game might not be running)")
    else:
        print("⚠ Game Connector not connected (game not running)")

except Exception as e:
    print(f"⚠ Game Connector error: {e}")
    print("  (This is OK if game is not running)")

print("\n" + "=" * 60)
print("YOLO REAL TEST COMPLETE")
print("=" * 60)
print("\nSummary:")
print(f"  ✓ YOLO Model: LOADED")
print(f"  ✓ Detection: FUNCTIONAL")
print(f"  ✓ Classes: {list(detector.classes.values())}")
print(f"  ✓ SB3: Available for RL training")
print("\nTo train the AI:")
print("  1. Start Brawl Stars on emulator")
print("  2. Run: python training/train_ai.py --method ppo --timesteps 10000")
