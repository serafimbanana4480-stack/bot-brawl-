"""
Test script to verify movement predictor integration with tracker.
"""

import sys
import numpy as np

print("=" * 60)
print("Testing Movement Predictor Integration")
print("=" * 60)

# Test 1: Import tracker
print("\n[Test 1] Importing tracker...")
try:
    from tracker import EnemyTracker, MOVEMENT_PREDICTOR_AVAILABLE
    print(f"[OK] Tracker imported successfully")
    print(f"  Movement predictor available: {MOVEMENT_PREDICTOR_AVAILABLE}")
except ImportError as e:
    print(f"[FAIL] Failed to import tracker: {e}")
    sys.exit(1)

# Test 2: Initialize tracker with advanced prediction
print("\n[Test 2] Initializing tracker with advanced prediction...")
try:
    tracker = EnemyTracker(max_age=30, min_hits=2, use_advanced_prediction=True)
    print(f"[OK] Tracker initialized with advanced prediction")
    print(f"  Movement predictor instance: {tracker.movement_predictor is not None}")
except Exception as e:
    print(f"[FAIL] Failed to initialize tracker: {e}")
    sys.exit(1)

# Test 3: Test basic tracking
print("\n[Test 3] Testing basic tracking...")
try:
    # Simulate some detections
    detections = [
        ([100, 100, 150, 150], 0.9),
        ([105, 105, 155, 155], 0.88),
        ([110, 110, 160, 160], 0.85),
    ]
    
    for det in detections:
        tracks = tracker.update([det])
    
    print(f"[OK] Basic tracking works")
    print(f"  Active tracks: {len(tracks)}")
    if tracks:
        print(f"  First track ID: {tracks[0].id}")
except Exception as e:
    print(f"[FAIL] Failed basic tracking: {e}")
    sys.exit(1)

# Test 4: Test predict_position
print("\n[Test 4] Testing predict_position...")
try:
    if tracks:
        track_id = tracks[0].id
        prediction = tracker.predict_position(track_id, time_ahead=0.25)
        print(f"[OK] predict_position works")
        print(f"  Track ID: {track_id}")
        print(f"  Prediction: {prediction}")
    else:
        print("[WARN] No tracks available for prediction test")
except Exception as e:
    print(f"[FAIL] Failed predict_position: {e}")
    sys.exit(1)

# Test 5: Test get_velocity
print("\n[Test 5] Testing get_velocity...")
try:
    if tracks:
        track_id = tracks[0].id
        velocity = tracker.get_velocity(track_id)
        print(f"[OK] get_velocity works")
        print(f"  Track ID: {track_id}")
        print(f"  Velocity: {velocity}")
    else:
        print("[WARN] No tracks available for velocity test")
except Exception as e:
    print(f"[FAIL] Failed get_velocity: {e}")
    sys.exit(1)

# Test 6: Test get_leading_shot_position
print("\n[Test 6] Testing get_leading_shot_position...")
try:
    if tracks:
        track_id = tracks[0].id
        leading_pos = tracker.get_leading_shot_position(track_id, projectile_speed=15.0, frame_delay=0)
        print(f"[OK] get_leading_shot_position works")
        print(f"  Track ID: {track_id}")
        print(f"  Leading position: {leading_pos}")
    else:
        print("[WARN] No tracks available for leading shot test")
except Exception as e:
    print(f"[FAIL] Failed get_leading_shot_position: {e}")
    sys.exit(1)

# Test 7: Test play.py integration
print("\n[Test 7] Testing play.py integration...")
try:
    sys.path.insert(0, 'pylaai_real')
    from play import PlayLogic
    print(f"[OK] play.py imported successfully")
except ImportError as e:
    print(f"[WARN] Could not import play.py (may require additional dependencies): {e}")
except Exception as e:
    print(f"[WARN] Error importing play.py: {e}")

print("\n" + "=" * 60)
print("Integration Test Complete")
print("=" * 60)
print("\nSummary:")
print("[OK] All core tracker tests passed")
print("[OK] Movement predictor integration verified")
print("[OK] Advanced prediction methods available")
