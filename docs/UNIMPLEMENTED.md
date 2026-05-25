# UNIMPLEMENTED COMPONENTS

This file lists components that have been architected (stubs exist) but
are NOT functional due to missing prerequisites.

> **Rule**: No simulation, no fake data, no COCO reuse. Every item below
> requires real Brawl Stars data or real emulator access to implement.

---

## 1. YOLO Model Training

**Status**: ❌ Not started  
**Blocker**: No labeled Brawl Stars dataset exists yet

**What's needed**:
- Minimum 500 annotated frames per class
- Classes: `enemy`, `teammate`, `player`, `wall`, `bush`, `powerup`, `box`, `bullet`
- Annotation format: YOLO txt (class cx cy w h normalized)
- Tool: Label Studio or Roboflow

**Pipeline**:
```
screenshot_recorder.py   →  raw PNG frames
      ↓
Label Studio / Roboflow  →  annotated dataset
      ↓
yolo train data=brawlstars.yaml model=yolov8n.pt epochs=100
      ↓
models/brawlstars_v1.pt  →  model_validator.py registers it
```

**Reference**: `ultralytics` YOLOv8 training docs

---

## 2. SAM2 Auto-Labeling

**Status**: ❌ Not started  
**Blocker**: No seed annotations to bootstrap SAM2 prompts

**What's needed**:
- Meta's Segment Anything 2 (`segment-anything-2`)
- ~20-30 manually labeled "seed" frames
- SAM2 propagation across video frames from recorder output

**Goal**: Reduce manual labeling from 500 → ~30 seed frames by
propagating annotations through video clips automatically.

---

## 3. Behavior Cloning Training

**Status**: STUB in `rl_stubs/behavior_cloning.py`  
**Blocker**: No labeled (observation, action) dataset

**What's needed**:
- Human plays Brawl Stars while recorder captures frames
- Simultaneous ADB input logging to record what actions were taken
- Label pairing: frame_i ↔ action_i

**Minimum dataset size**: ~10,000 labeled transitions

---

## 4. CQL Offline RL Training

**Status**: STUB in `rl_stubs/cql_trainer.py`  
**Blocker**: No offline replay buffer

**What's needed**:
- Replay buffer in format: `(state, action, reward, next_state, done)`
- Reward function definition (kill = +1, death = -1, damage = +0.1, etc.)
- Minimum 50,000 transitions for stable offline training

**Dependency**: BC policy must exist first (warm-start for CQL)

---

## 5. Online RL (PPO)

**Status**: Not stubbed  
**Blocker**: No Brawl Stars simulator / gym environment exists

**Why it's hard**:
- Brawl Stars has no official API or open simulator
- Running the actual game as a training env requires:
  - BlueStacks running 24/7
  - Stable ADB connection
  - ~100k+ real game matches for convergence
  - Custom reward shaping from game state

**Alternative**: Use BC + CQL offline policy, deploy, then collect online
corrections (DAGGER-style) for future fine-tuning.

---

## 6. TensorRT Optimization

**Status**: Stub file exists (`tensorrt_optimizer.py`)  
**Blocker**: No trained model to optimize + requires NVIDIA GPU + TensorRT SDK

**What's needed**:
- Trained `brawlstars_v1.pt` from step 1
- NVIDIA GPU (RTX series recommended)
- TensorRT 8.x or 10.x installed
- `torch2trt` or `ultralytics` built-in export

**Expected speedup**: 3-5x inference latency reduction (from ~50ms → ~15ms at 1080p)

---

## 7. Frontend WebSocket Unification

**Status**: ⚠️ Partial — backend endpoint is `/api/brawl-stars/ws/logs`
**Issue**: Legacy `brawl-stars-ui` frontend used `/ws/brawl-stars` path

**Fix already applied**: Backend route is canonical at `/api/brawl-stars/ws/logs`.
Any frontend component using `/ws/brawl-stars` must be updated to match.

---

## 8. Real-Time Telemetry Dashboard

**Status**: ✅ Backend implemented, frontend not connected
**Implementation**: `api.py` has GET `/api/brawl-stars/telemetry` endpoint and `emit_telemetry_update()` WebSocket function
**Blocker**: Frontend not connected to telemetry endpoint

**What's needed**:
- Frontend component to fetch telemetry from `/api/brawl-stars/telemetry`
- WebSocket client to receive real-time updates via `emit_telemetry_update()`
- Dashboard UI to display APM, win rate, suspicion score, human likeness

---

## 9. Behavioral Biometrics

**Status**: ✅ Implemented and integrated
**Implementation**: `safety_system.py` has MovementAnalyzer class for swipe/tap analysis
**Integration**: Integrated in `emulator_controller.py` via `tap_scaled()` and `swipe_scaled()`

---

## 10. Multi-Object Tracking

**Status**: ✅ Implemented and integrated
**Implementation**: `tracker.py` has EnemyTracker class based on SORT
**Integration**: Integrated in `play.py` for enemy tracking and prediction
**Reset**: Tracker is reset between matches via `reset_for_new_match()`

---

## 11. Map-Specific Strategies

**Status**: ✅ Implemented and integrated
**Implementation**: `lobby.toml` has map-specific strategies, `movement.py` loads and applies them
**Integration**: Integrated in `state_manager.py` via `_handle_matchmaking()` calling `set_current_map()`

---

## 12. Brawler-Specific Strategies

**Status**: ✅ Implemented and integrated
**Implementation**: `lobby.toml` has brawler-specific strategies, `play.py` loads and applies them
**Integration**: Integrated in `lobby_automator.py` via `select_brawler()` calling `set_current_brawler()`

---

## 13. Tracker Position Prediction

**Status**: ✅ Implemented, not fully utilized
**Implementation**: `tracker.py` has `predict_position()` method for enemy position prediction
**Usage**: Available but not extensively used in combat logic
**Potential improvement**: Use prediction for aim assist and dodging

---

## SYSTEM INTEGRITY SUMMARY

| Component           | Status        | Blocker                         |
|---------------------|---------------|---------------------------------|
| Model validation    | ✅ Done        | —                               |
| Humanization API    | ✅ Done        | —                               |
| Safety gate         | ✅ Done        | —                               |
| Screenshot recorder | ✅ Done        | —                               |
| Model registry + CI | ✅ Done        | —                               |
| Emulator detection  | ✅ Fixed       | —                               |
| 5-step verification | ✅ Done        | —                               |
| Behavioral biometrics| ✅ Done       | —                               |
| Multi-object tracking| ✅ Done       | —                               |
| Map-specific strategies| ✅ Done     | —                               |
| Brawler-specific strategies| ✅ Done  | —                               |
| Tracker reset       | ✅ Done        | —                               |
| Real-time telemetry | ⚠️ Backend only| Frontend not connected         |
| YOLO training       | ❌ Not started | No Brawl Stars dataset          |
| SAM2 auto-label     | ❌ Not started | No seed annotations             |
| Behavior cloning    | STUB          | No labeled gameplay data        |
| CQL trainer         | STUB          | No replay buffer                |
| Online RL (PPO)     | Not started   | No simulator                    |
| TensorRT export     | STUB          | No trained model + no NVIDIA    |

**Current verdict**: ⚠️ DEMO ONLY
**Path to COMMERCIAL READY**: Train YOLO on real Brawl Stars data (step 1).
