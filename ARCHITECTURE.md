# Brawl Stars Bot - Architecture Documentation

## System Overview

This is a professional-grade autonomous Brawl Stars bot featuring:
- **Vision**: YOLOv8 detection + ByteTrack tracking + State extraction
- **Decision**: State machine (IDLE→SEARCH→ENGAGE→RETREAT→RECOVER) + Rule engine
- **Execution**: Humanized inputs with Bezier curves, delays, miss-clicks
- **Learning**: Auto-labeling + Performance monitoring + Continuous retraining

## Module Structure

```
brawl_bot/
├── vision/                 # Computer vision stack
│   ├── __init__.py
│   ├── tracker.py         # ByteTrack implementation
│   ├── state.py           # State extraction (GameState, EnemyInfo, etc.)
│   └── vision_engine.py   # YOLOv8 inference (existing)
│
├── decision/              # Decision making system
│   ├── __init__.py
│   ├── state_machine.py   # FSM for bot states
│   ├── rules.py           # Tactical rule engine
│   └── scorer.py          # Target/action scoring
│
├── control/               # Input execution (future expansion)
│   └── (humanization.py is in root)
│
├── training/              # Auto-learning pipeline
│   ├── __init__.py
│   ├── auto_labeler.py   # Heuristic auto-labeling
│   └── retrain.py        # Performance monitoring & retraining
│
├── core/                  # Orchestration
│   ├── __init__.py
│   └── orchestrator.py   # Main bot loop integration
│
├── main_enhanced.py       # New CLI entry point
└── wrapper.py            # Original bot wrapper
```

## Vision Module

### ByteTracker
- **Purpose**: Persistent object tracking across frames
- **Features**:
  - IoU-based matching
  - High/low confidence association
  - Velocity prediction
  - Track aging (max_age=30)
  - Non-maximum suppression

### StateExtractor
- **Purpose**: Convert tracks to structured game state
- **Outputs**:
  - `GameState`: Complete game situation
  - `EnemyInfo`: Position, health, threat level
  - `WallInfo`: Obstacle positions
  - `BushInfo`: Cover locations
- **Calculates**:
  - Danger score (0.0-1.0)
  - Nearest enemy
  - Lowest HP target
  - Biggest threat
  - Safe retreat positions

## Decision Module

### State Machine

**States:**
1. **IDLE**: Waiting, minimal action
2. **SEARCH**: Patrolling for enemies
3. **ENGAGE**: Attacking targets
4. **RETREAT**: Escaping danger
5. **RECOVER**: Healing, repositioning

**Transitions:**
- IDLE→SEARCH: No enemies visible
- IDLE→ENGAGE: Enemy spotted, safe to attack
- IDLE→RETREAT: Immediate danger
- SEARCH→ENGAGE: Found enemy
- ENGAGE→RETREAT: Low health or outnumbered
- ENGAGE→SEARCH: Enemy lost/killed
- RETREAT→RECOVER: Safe, need healing
- RETREAT→SEARCH: Safe and healthy
- RECOVER→SEARCH: Healed and ready

### Rule Engine

**Tactics:**
- `ENGAGE_CLOSE`: Shotgun/rush style
- `ENGAGE_RANGED`: Sharpshooter style
- `HARASS`: Chip damage from safety
- `FLANK`: Attack from side/back
- `CIRCLE_STRAFE`: Dodge while attacking
- `RETREAT_DEFENSIVE`: Safe escape
- `RETREAT_AGGRESSIVE`: Trade kills
- `TAKE_COVER`: Use walls
- `HEAL_UP`: Bush camping
- `PATROL`: Search pattern
- `AMBUSH`: Wait in bush

### Scoring System

**Target Scoring Factors:**
- Health (30%): Lower = higher priority
- Distance (25%): Optimal range bonus
- Threat (20%): Inverse of danger
- Vulnerability (10%): Isolated enemies
- Kill pressure (15%): Time-to-kill bonus

**Randomization:**
- ±5% score variance to prevent patterns
- Weighted random tactic selection

## Training Module

### AutoLabeler

**Heuristic Detection:**
- Health bars: Color segmentation (green/red)
- Players: Circle detection
- Walls: Edge detection + morphology
- Bushes: Green color range

**Template Matching:**
- Ammo icons
- Super charge indicators
- Player/enemy indicators

**Non-Maximum Suppression:**
- IoU threshold: 0.5
- Confidence filtering per class

### Continuous Learning

**Performance Metrics:**
- Kills/Deaths/KDA
- Win rate
- Detection accuracy
- Decision quality
- Average survival time

**Retraining Triggers:**
- Win rate < 40%
- Detection accuracy < 70%
- Decision accuracy < 60%
- 7 days since last retrain
- 500+ new samples

**Pipeline:**
1. Capture new gameplay data
2. Auto-label (heuristics + review)
3. Validate dataset
4. Retrain YOLO model
5. Validate new model
6. A/B test against current
7. Deploy if better

## Core Orchestrator

**Main Loop:**
```
1. Capture frame
2. Run YOLO inference
3. Update ByteTrack
4. Extract GameState
5. Update FSM state
6. Execute state handler
7. Apply humanization
8. Send inputs
9. Log performance
10. Check for retrain trigger
```

**Safety Limits:**
- Max APM: 180 (configurable)
- Trophy limit: 500 (configurable)
- Frame skip: 1 (configurable)
- Reaction delay: 80-220ms (humanized)

**Humanization:**
- Gaussian reaction delays
- Bezier curve mouse movements
- 3-5% miss-click rate
- 1-3 second pauses

## Usage

### Quick Start

```bash
# Validate system
python main_enhanced.py check

# Auto-label dataset
python main_enhanced.py auto-label ./captures -o ./labels

# Train model
python main_enhanced.py train -d ./dataset -e 100

# Run bot
python main_enhanced.py run -m ./models --auto-learn
```

### Configuration

```python
from core.orchestrator import create_bot_orchestrator

bot = create_bot_orchestrator(
    models_dir="./models",
    dataset_dir="./dataset",
    confidence_threshold=0.5,
    enable_auto_learning=True,
    max_apm=180
)

bot.initialize()
bot.start()
```

## Integration Points

### Vision → Decision
```python
tracks = tracker.update(detections)
game_state = state_extractor.extract_state(tracks)
context = StateContext(game_state=game_state, bot_instance=self)
new_state = state_machine.update(context)
```

### Decision → Control
```python
ranked_targets = target_scorer.rank_targets(
    game_state.enemies,
    game_state.player_position,
    game_state.player_health,
    game_state.walls
)
best_target = ranked_targets[0]
execute_attack(best_target, context)
```

### Training Integration
```python
# Auto-label new captures
stats = auto_label_dataset(
    images_dir="./captures",
    output_dir="./dataset/labels"
)

# Monitor and retrain
learner = create_continuous_learner(
    log_dir="./logs",
    dataset_dir="./dataset",
    models_dir="./models"
)
learner.start()
```

## Performance Considerations

**Inference Speed:**
- YOLOv8s: ~30 FPS on GTX 1060
- YOLOv8n: ~60 FPS on GTX 1060
- Frame skip reduces load

**Tracking Overhead:**
- ByteTrack: <1ms per frame
- State extraction: ~2ms

**Decision Latency:**
- FSM update: <0.1ms
- Rule evaluation: ~1ms
- Scoring: ~0.5ms per target

**Total Pipeline:**
- ~50ms per frame (20 FPS) with all features

## Future Enhancements

1. **RL Integration**: Replace rule engine with PPO/DQN
2. **Multi-modal**: Add audio cues for abilities
3. **Map Learning**: Remember wall positions
4. **Team Coordination**: 3v3 mode support
5. **Cloud Training**: Distributed dataset collection

## Troubleshooting

**Low Detection Accuracy:**
- Check model is trained on Brawl Stars data
- Verify confidence threshold isn't too high
- Ensure adequate lighting on screen

**Poor Decision Making:**
- Review performance logs
- Adjust danger score thresholds
- Check state transitions are firing

**High APM (Detection Risk):**
- Reduce max_apm setting
- Increase reaction_delay_min
- Add more random pauses

**Auto-learning Not Triggering:**
- Check dataset_dir exists and is writable
- Verify min_matches_before_retrain threshold
- Review performance_monitor logs
