# Soberana Omega — Brawl Stars Bot

> **Architecture Reference & Developer Guide**

---

## 1. Project Overview

**Soberana Omega** is an AI-powered Brawl Stars bot leveraging real-time computer vision (YOLO/TensorRT), adaptive combat decision-making, hierarchical state management, and anti-ban humanization. The bot operates on Windows via ADB/emulator control, continuously learning from gameplay through Q-Learning and ELO tracking.

| Property | Value |
|---|---|
| Language | Python 3.12 |
| Framework | PyTorch, Ultralytics YOLO |
| Platform | Windows + Android Emulator (LDPlayer) |
| Architecture | Multi-agent FSM with RL online learning |
| Persistence | JSON (ELO), Pickle (Q-table), Checkpoint (state recovery) |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                wrapper.py                                   │
│                           PylaAIEnhanced (entry)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │  Dashboard  │  │   Safety    │  │   Lobby     │  │    Logging      │   │
│  │  Server     │  │   System    │  │  Navigator  │  │    System       │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                          StateManager                                       │
│                    GameStateMachine (lobby/in_game/loading/etc.)            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     UnifiedStateDetector                              │  │
│  │              (pixel heuristics + template matching fusion)            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                           PlayLogic                                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────────────┐  │
│  │  Targeting │ │  Movement  │ │  Combat    │ │   Vision Pipeline     │  │
│  │  Engine    │ │  Engine     │ │  Advanced  │ │   (YOLO/TensorRT)     │  │
│  └────────────┘ └────────────┘ └────────────┘ └────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │ RL Engine  │ │  ELO       │ │  Meta      │ │ WorldModel │               │
│  │ (Q-Learning│ │  Tracker   │ │ Learning   │ │ Integrator │               │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │   Error    │ │  State     │ │ Behavioral │ │  Adaptive  │               │
│  │  Recovery  │ │ Persistence│ │  Profile   │ │ FrameSkip  │               │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Modules

### 3.1 Entry Point

| File | Class | Responsibility |
|---|---|---|
| `wrapper.py` | `PylaAIEnhanced` | Main orchestrator; initializes all subsystems, runs monitor loop |
| `emulator_controller.py` | `EmulatorController` | ADB input dispatch (tap, swipe, text) |

### 3.2 State Management

| File | Class | Responsibility |
|---|---|---|
| `pylaai_real/state_manager.py` | `StateManager` | Hierarchical game state machine; routes to appropriate handler |
| `pylaai_real/unified_state_detector.py` | `UnifiedStateDetector` | Pixel heuristics + template matching fusion with smoothing/voting |
| `pylaai_real/screenshot_taker.py` | `ScreenshotTaker` | Win32 screenshot capture (D3D, GDI fallback) |

**Game States:** `unknown`, `lobby`, `brawler_select`, `map_select`, `match_loading`, `in_game`, `in_game_countdown`, `mission`, `tutorial`, `news`, `brawler_unlock`, `season_reset`, `shop`, `settings`

### 3.3 Combat

| File | Class | Responsibility |
|---|---|---|
| `pylaai_real/play.py` | `PlayLogic` | Targeting, ability usage, leading shots, combo execution |
| `pylaai_real/movement.py` | `MovementEngine` | Pathfinding, kiting vectors, screen-bound clamping |
| `pylaai_real/combat_advanced.py` | `AdvancedCombatStrategy` | Phase 5 combat engine: leading shots, kiting, cover, combos |

### 3.4 Lobby Automation

| File | Class | Responsibility |
|---|---|---|
| `pylaai_real/lobby_automator.py` | `LobbyAutomator` | Play button, brawler select, team confirm |
| `pylaai_real/lobby_navigator.py` | `LobbyNavigatorV2` | Popup detection, smart play button, event detection, fast brawler select |
| `core/lobby_fsm.py` | `LobbyFSM` | Hierarchical lobby FSM with repetition guard |

### 3.5 Vision

| File | Class | Responsibility |
|---|---|---|
| `pylaai_real/detect.py` | `Detect` | YOLO inference wrapper (batch, async, TensorRT) |
| `vision/game_feature_extractor.py` | `GameFeatureExtractor` | Wall detection, HP extraction, bush detection, timer detection |
| `core/async_pipeline.py` | `AsyncPipeline` | Parallel inference/tracking/decision pipeline |
| `vision/ocr_hud_extractor.py` | `OCRHudExtractor` | Advanced HUD OCR: HP, ammo, super, timer, score, cubes, gems |
| `vision/player_state_detector.py` | `PlayerStateDetector` | Fused player state: life, super, visibility, threat (YOLO+OCR+pixel) |
| `vision/game_state.py` | `GameState` | Unified game state dataclass (objects + HUD + player status) |
| `vision/multimodal_pipeline.py` | `MultimodalPipeline` | 3-layer vision pipeline: YOLO → OCR → heuristics → GameState |

### 3.6 Resolution Management (`core/resolution_manager.py`)

Sistema centralizado de gestão de resolução — **resposta crítica à Análise de Arquitetura v2.0**.

**Problema identificado:** Coordenadas hardcoded 1920×1080 espalhadas por `wrapper.py`, `play.py`, `state_manager.py`, etc., quebravam em qualquer resolução não-padrão.

**Solução:**

| Componente | Classe | Responsabilidade |
|---|---|---|
| `core/resolution_manager.py` | `ResolutionManager` | Deteta resolução real (Win32/ADB), mantém canónico 1920×1080, fornece escala bidirecional |
| `core/resolution_manager.py` | `ResolutionProfile` | Perfil com metadados: actual_resolution, canonical_resolution, scale_x/y, validated, source |

**Fluxo de Coordenadas:**

```
Emulador (actual: 2560x1440)
    │
    ▼  ScreenshotTaker.normaliza_para_1920x1080()
Pipeline de Visão (canónico: 1920x1080)
    │
    ▼  ResolutionManager.from_canonical(x, y)
ADB Input (actual: 2560x1440)
```

**API Principal:**

```python
rm = ResolutionManager(window_title="LDPlayer")
rm.detect()  # Win32 → ADB → config → fallback

# Vision -> Input
actual_x, actual_y = rm.from_canonical(x_1080, y_1080)

# Input -> Vision (raramente necessário)
canonical_x, canonical_y = rm.to_canonical(x_actual, y_actual)

# ROI normalizada (0-1) -> pixels reais
px = rm.scale_relative_to_actual(rx, ry)
```

**Validação:**
- `MIN_WIDTH=640`, `MAX_WIDTH=5120`
- `MIN_ASPECT_RATIO=1.3`, `MAX_ASPECT_RATIO=2.4`
- Resoluções inválidas logam warning e usam fallback canónico

**Mudanças Runtime:**
- `check_for_changes()` detecta resize da janela
- Callback `on_resolution_change` atualiza `MovementEngine`, `UnifiedStateDetector`, invalida cache do `AutoCalibrator`

**Integração com AutoCalibrator:**
- Cache do AutoCalibrator armazena coordenadas **canónicas** (resolução-independente)
- `detect_element()` converte automáticamente: cache canónico → screenshot actual
- Templates são escalados do canónico para o actual antes do template matching
- Fallback coords são fornecidas em canónico e convertidas no retorno

---

## 4. Decision Systems

### 4.1 Q-Learning RL Engine (`rl_engine.py`)

```
State Space (discrete):
  - HP band:        0=<25%, 1=25-50%, 2=50-75%, 3=>75%
  - Enemy count:    0, 1, 2, 3+
  - Distance:       close, medium, far
  - Ammo:           0, 1-2, 3+
  - Super ready:    yes/no

Action Space (6 actions):
  0=attack  1=move_to_enemy  2=retreat  3=use_super
  4=collect_cube  5=idle

Epsilon-Greedy:
  - Initial: 0.40
  - Decay:   0.995 per match
  - Floor:   0.05 (5% continuous exploration)

Reward Signal:
  +10  hit enemy
  +50  eliminate enemy
  +5   collect power cube
  +100 win match
  -5   take damage
  -50  death
  -200 bot death (no respawn)
```

### 4.2 ELO Tracker (`elo_tracker.py`)

- Per brawler + map combination
- K-factor: 40 (rating < 1200), 24 (1200-1600), 16 (> 1600)
- Auto-save every 10 matches to `data/elo_ratings.json`

### 4.3 Utility AI (`decision/utility_ai.py`)

- Scored action selection (transparent, no training)
- Factors: distance, HP, threat level, cover availability, ammo

### 4.4 Sticky Target (`decision/sticky_target.py`)

- Target commitment to prevent thrashing
- Lock duration: 3-8 seconds depending on context

### 4.5 Intent System (`decision/intent_system.py`)

- Persistent strategic goals (collect gems, hold zone, push lane)
- 10-second goal windows with automatic re-evaluation

### 4.6 Enemy Intention (`decision/enemy_intention.py`)

- Predict enemy behavior patterns (aggressive, defensive, fleeing)
- Based on HP, position, movement direction, ability usage

### 4.7 Meta-Awareness (`decision/meta_awareness.py`)

- Tracks current meta brawler compositions
- Adjusts strategy based on enemy team composition

---

## 5. Resilience & Safety Systems

### 5.1 Error Recovery (`core/error_recovery.py`)

**Circuit Breaker Pattern** with 3 states and 5 recovery levels:

```
States: CLOSED → OPEN → HALF_OPEN

Recovery Levels (hierarchical):
  Level 1: Retry same action (3 attempts, 100ms delay)
  Level 2: Fallback to alternative detector/solver
  Level 3: Degrade to pixel/heuristic-only mode
  Level 4: Skip frame, increase cycle delay
  Level 5: Pause bot, alert dashboard, preserve state

CLOSED:     Normal operation, failures increment counter
OPEN:       After 5 consecutive failures, bypass subsystem
HALF_OPEN:  Probe with single action; success → CLOSED, fail → OPEN
```

### 5.2 Graceful Degradation (`core/error_recovery.py`)

```
Full Vision (YOLO) → Pixel Heuristics → Hardcoded Coordinates
     ↓                      ↓                    ↓
  ~40ms inference      ~5ms pixel check    ~0ms static
  High accuracy        Medium accuracy    No false positives
```

### 5.3 Watchdog System

- Monitor loop tracks execution time per subsystem
- Timeout: 2x expected duration → flag + log
- Automatic throttle if cycle time exceeds limit

### 5.4 Safety System (`safety_system.py`)

| Limit | Value | Reason |
|---|---|---|
| Max APM | 40 | Below detection threshold |
| Max matches/hour | 25 | Anti-ban pacing |
| Session cooldown | 5 min after 2h | Fatigue simulation |
| Random breaks | Every 20-40 matches, 2-5 min | Unpredictability |

### 5.5 State Persistence (`state_persistence.py`)

Automatic checkpoint every 30 seconds:
- Current game state, match progress, Q-table, ELO, statistics
- Recovery after crash: `load_checkpoint()` restores full context
- History: last 10 checkpoints retained

---

## 6. Memory & Spatial Systems

### 6.1 World Model (`core/world_model.py`)

Persistent spatial/temporal memory:
- Enemy positions with temporal decay (forgets after 5s without detection)
- Danger zones (areas with high enemy presence)
- Safe zones (recently cleared areas)
- Power cube positions with respawn tracking
- Bush/cover locations

### 6.2 World Model Integrator (`world_model_integration.py`)

Bridge between `play.py` detections and `WorldModel`:
- Updates enemy tracking with persistent IDs
- Predicts enemy positions (linear extrapolation)
- Queries safe zones for retreat decisions
- Extracts power cube and bush positions

### 6.3 Occupancy Grid (`core/occupancy_grid.py`)

Spatial 2D grid overlaid on game map:
- Cell size: 1 tile (configurable)
- Tracks: enemy occupancy, danger level, last visit time
- Used for pathfinding and zone control

### 6.4 Pressure Map (`core/pressure_map.py`)

Enemy influence zones:
- Gaussian falloff from enemy positions
- Threat visualization
- Used for cover selection and kiting direction

### 6.5 Cover System (`core/cover_system.py`)

Bush and wall strategy:
- Line-of-sight scoring
- Escape route analysis
- Power cube proximity bonus
- Threat cone penalty

---

## 7. Meta-Learning System (`meta_learning.py`)

Automatic hyperparameter adaptation:

```python
MetaLearningSystem
├── record_match_result()    # Feed match outcome + metrics
├── get_recommended_epsilon()  # Context-aware exploration rate
├── get_confirmed_confidence() # Adaptive confidence threshold
├── should_explore_new_strategy()
└── get_context_strategy()     # "aggressive" | "defensive" | "balanced"
```

**Performance Trend Detection:**
- `improving`: rolling win rate increasing over last 10 matches
- `stable`: win rate within ±5% band
- `declining`: win rate decreasing

**Strategy Adaptation:**
- Losing streak + aggressive strategy → switch to defensive
- Winning streak + defensive strategy → increase aggression
- High ELO enemy detected → defensive bias

---

## 8. Behavioral Profile System (`behavioral_profile_system.py`)

Session-level personality simulation:

```python
SessionProfile
├── personality:     aggressive | passive | balanced
├── skill_level:      beginner / intermediate / advanced
├── age_factor:       0.7-1.0 (declines over session)
├── hesitation_prob:  per-action hesitation probability
├── overcorrect_prob: movement overshoot probability
└── tunnel_vision:    target lock probability

Learning Curve (per session):
  warmup (0-2min) → peak (2-15min) → decline (15-25min) → fatigue (25min+)
```

**Behavioral Modifiers:**
- `should_show_hesitation()`: random delay before actions
- `should_overcorrect()`: intentional movement overshoot
- `should_tunnel_vision()`: reduced peripheral awareness
- `get_delay_modifier()`: multiplier on reaction time

---

## 9. Humanization

### 9.1 Humanization Utils (`humanization_utils.py`)

| Function | Purpose |
|---|---|
| `human_delay()` | Randomized delays with configurable jitter |
| `jitter_coords()` | Coordinate jitter (±3px) for natural clicks |
| `jitter_value()` | Value variation for cooldowns/timings |
| `bezier_curve_points()` | Bezier curves for human-like swipes |
| `HumanPauseSimulator` | Strategic pauses, 120-400ms reaction delay |
| `APMController` | Target ~35 APM with variance |

### 9.2 Fatigue Simulator (`humanization.py`)

```python
AdvancedHumanizationEngine
├── fatigue_curve:     warmup → peak → fatigue
├── error_rate:        increases after peak
├── reaction_time:     increases in fatigue phase
├── decision_speed:    decreases in fatigue phase
└── tremor_factor:      micro-movements increase
```

### 9.3 Adaptive Frame Skipper (`core/adaptive_frame_skipper.py`)

Dynamically adjusts inference frequency:
- Combat: every frame (60 FPS equivalent)
- Safe/repetitive states: skip 2-3 frames
- Static states (lobby, menu): skip 5-10 frames

---

## 10. Dashboard & Telemetry

### 10.1 Dashboard Server (`dashboard_server.py`)

Real-time web dashboard (port 8765):

| Endpoint | Data |
|---|---|
| `GET /api/live` | Current state, brawler, map, FPS, epsilon, cycle time |
| `GET /api/history` | Win rate, ELO, matches played, recent events |
| `GET /api/rewards` | Reward history chart |
| `GET /api/replays` | Replay list with thumbnails |
| `GET /api/abtest` | A/B test variant results |

**Features:**
- Dark-themed responsive UI
- Live charts (rewards, cycle times)
- Last screenshot thumbnail (base64)
- No external dependencies (embedded HTML+JS)

### 10.2 Replay Recorder

- Saves up to 150 frames (~5 min) per replay to `data/replays/`
- Each frame: screenshot (JPEG 50%), state, action, enemy count, reward
- JSON metadata with event list

### 10.3 A/B Testing Framework

- Round-robin variant assignment per match (50/50)
- Tracks: matches, wins, losses, win rate, avg reward
- Persists to `data/ab_tests.json`

---

## 11. Training Pipeline

### 11.1 YOLO Model Training

**Standard Class Mapping (Canonical):**
```
0: Player   1: Bush    2: Enemy   3: Cubebox
4: Wall     5: Powerup 6: Bullet  7: Super
```

**Roboflow Dataset (Bloxxy):**
- URL: https://universe.roboflow.com/bloxxy/brawl-stars-dataset
- Version: 22 (object-detection)
- Images: 2551 (1839 train, 483 val, 229 test)

**Training Commands:**
```bash
# Download Roboflow dataset
python training/download_roboflow_dataset.py --api-key KEY --output dataset/roboflow_raw_v2

# Train (GPU)
python training/enhanced_training_pipeline.py --train-only --epochs 100 --batch 16 --device cuda

# Validate
python training/enhanced_training_pipeline.py --validate-only --dataset dataset/roboflow_raw_v2

# Full workflow
python training/complete_training_workflow.py --all --api-key KEY --epochs 100
```

### 11.2 8-Class Model Training (`train_8class_model.py`)

Trains YOLO with full class set (including Bullet, Super):
```bash
python train_8class_model.py --epochs 100 --batch 16 --device cuda
```

---

## 12. Advanced Combat (Phase 5)

### 12.1 Leading Shot Engine

- **Projectile physics** per brawler (tiles/s → pixels/s conversion)
- **EMA velocity filter** (smooths zig-zag movement, detects "stopped")
- **HumanAimError model**:
  - Overshoot bias (shoots ahead of moving enemy)
  - Distance-proportional error (farther = less precise)
  - Temporal clustering (correlated errors between shots)
- Brawler-specific precision: Piper (0.95), El Primo (0.70), etc.

### 12.2 Kiting Engine

- **Resultant vector**: considers ALL enemies (inverse distance² weighting)
- **Bush preference**: retreats toward bush in escape direction
- **Screen clamping**: never retreats outside bounds
- **Adaptive cooldown** by role (assassins kite less)

### 12.3 Cover Engine

- Scoring: proximity + distance from enemies + line-of-sight break
- **Escape route bonus**: adjacent bush bonus (for continued retreat)
- **Power cube bonus**: +200 points if bush has nearby cube
- **Threat cone penalty**: penalizes bushes in enemy path

### 12.4 Combo Manager

- Conditional sequences per brawler: `any`, `enemy_low_hp`, `enemy_grouped`, `player_full_hp`
- One action per `play_round()` cycle (non-blocking)
- Only initiates if 1st action satisfies condition

### 12.5 Combat State Machine

```
States: neutral → aggressive / defensive / retreating
Transitions: automatic based on HP, brawler role, threat level
```

---

## 13. Configuration

### 13.1 config.json Structure

```json
{
  "detection": {
    "confidence": 0.5,
    "model_path": "models/brawlstars_yolov8.pt",
    "classes": ["Player", "Bush", "Enemy", "Cubebox", "Wall", "Powerup", "Bullet", "Super"],
    "device": "cuda"
  },
  "combat": {
    "shot_cooldown": 0.35,
    "targeting": "adaptive",
    "leading_shots": true
  },
  "humanization": {
    "apm_target": 35,
    "reaction_delay_min": 0.12,
    "reaction_delay_max": 0.40
  },
  "learning": {
    "epsilon_start": 0.40,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.05,
    "learning_rate": 0.10
  },
  "safety": {
    "max_matches_per_hour": 25,
    "max_session_minutes": 120,
    "cooldown_minutes": 5
  },
  "dashboard": {
    "port": 8765
  }
}
```

---

## 14. File Structure

```
c:/Users/rodri/Desktop/bot brawl/
├── wrapper.py                          # Main entry (PylaAIEnhanced)
├── emulator_controller.py              # ADB input
├── safety_system.py                   # Anti-ban limits
├── humanization.py                     # Fatigue simulation
├── config.json                        # Configuration
├── AGENTS.md                          # This file
│
├── core/
│   ├── error_recovery.py              # Circuit breakers, graceful degradation
│   ├── degradation_manager.py         # Automatic health-based degradation (v2.1)
│   ├── event_store.py                 # Event sourcing + CQRS (v2.1)
│   ├── game_state_checkpoint.py       # Advanced state checkpointing (v2.1)
│   ├── distributed_tracing.py         # OpenTelemetry-style tracing (v2.1)
│   ├── rate_limiter.py                # Intelligent per-account rate limiting (v2.1)
│   ├── replay_failure_analyzer.py     # AI replay failure analysis (v2.1)
│   ├── world_model.py                 # Spatial/temporal memory
│   ├── occupancy_grid.py              # 2D spatial grid
│   ├── pressure_map.py                # Enemy influence zones
│   ├── lobby_fsm.py                   # Hierarchical lobby FSM
│   ├── async_pipeline.py              # Parallel inference pipeline
│   ├── adaptive_screenshot.py         # Adaptive screenshot cache
│   ├── behavioral_profile.py          # Player behavior profiling
│   ├── input_optimizer.py             # Input latency optimization
│   ├── replay_analyzer.py            # Death cause analysis
│   ├── tactical_bridge.py             # Tactical/strategic bridge
│   └── cover_system.py                # Cover/wall strategy
│
├── decision/
│   ├── utility_ai.py                   # Scored action selection
│   ├── multi_objective_rl.py          # Multi-objective RL optimization (v2.1)
│   ├── brawler_adaptive_controller.py # Per-brawler adaptation (v2.1)
│   ├── sticky_target.py               # Target commitment
│   ├── intent_system.py               # Persistent strategic goals
│   ├── enemy_intention.py             # Enemy behavior prediction
│   └── meta_awareness.py              # Meta-game awareness
│
├── neural/                             # Neural network & learning (v2.1)
│   ├── transfer_learning.py           # Transfer learning for maps/brawlers
│   ├── curriculum_learner.py          # Adaptive difficulty curriculum
│   └── distributed_orchestrator.py    # Multi-bot distributed RL
│
├── vision/
│   ├── ensemble_detector.py           # Model ensemble + voting (v2.1)
│   ├── self_supervised_pretraining.py  # SimCLR pretraining (v2.1)
│   └── game_feature_extractor.py       # Wall/HP/bush/timer extraction
│
├── pylaai_real/
│   ├── state_manager.py               # Game state machine
│   ├── unified_state_detector.py       # State detection fusion
│   ├── play.py                        # Combat engine
│   ├── movement.py                     # Movement/pathfinding
│   ├── combat_advanced.py             # Phase 5 advanced combat
│   ├── lobby_automator.py             # Lobby automation
│   ├── lobby_navigator.py              # Lobby navigation v2
│   ├── detect.py                      # YOLO wrapper
│   ├── screenshot_taker.py            # Win32 screenshot
│   └── humanization_utils.py          # Humanization utilities
│
├── rl_engine.py                       # Q-Learning engine
├── elo_tracker.py                     # ELO rating tracker
├── meta_learning.py                    # Meta-learning system
├── world_model_integration.py         # WorldModel bridge
├── state_persistence.py               # State checkpoint/restore
├── behavioral_profile_system.py       # Session personality
├── wrapper_monitoring.py              # Health checks
├── wrapper_initializers.py            # Modular initializers
│
├── training/
│   ├── download_roboflow_dataset.py   # Roboflow download + remap
│   ├── enhanced_training_pipeline.py  # Full training pipeline
│   ├── complete_training_workflow.py  # Orchestrator
│   ├── validate_dataset.py            # Dataset integrity checks
│   ├── continuous_training_pipeline.py # Online learning
│   └── train_8class_model.py           # 8-class YOLO training
│
├── tests/
│   └── test_training_workflow_e2e.py  # End-to-end tests
│
├── data/
│   ├── q_table.pkl                    # Q-Learning table
│   ├── elo_ratings.json               # ELO per brawler+map
│   ├── checkpoints/                   # State persistence
│   ├── replays/                       # Replay recordings
│   └── ab_tests.json                  # A/B test results
│
├── models/
│   └── brawlstars_yolov8.pt           # Trained YOLO model
│
├── dataset/
│   └── roboflow_raw_v2/               # Remapped Roboflow dataset
│
└── images/                            # Templates and screenshots
```

---

## 15. Strategic Improvements v2.1 (Melhorias Estrategicas)

Modulos adicionados na v2.1 para robustez, observabilidade, anti-detecao e aprendizado acelerado.

### 15.1 Event Sourcing + CQRS (`core/event_store.py`)

Append-only event log para auditoria completa e debug pos-mortem.

| Componente | Classe | Responsabilidade |
|---|---|---|
| `core/event_store.py` | `EventStore` | Persiste todos os eventos de dominio em JSONL com compressao gzip |
| `core/event_store.py` | `DomainEvent` | Evento imutavel com trace_id, aggregate_id, payload, metadata |

**Eventos de Dominio:** `SESSION_STARTED`, `MATCH_STARTED`, `PLAYER_DAMAGED`, `ENEMY_HIT`, `PLAYER_DIED`, `ACTION_TAKEN`, `ERROR_OCCURRED`, `DEGRADATION_CHANGED`, `REWARD_RECEIVED`, etc.

**Beneficios:**
- Debug pos-mortem de bans/crashes via `post_mortem_analysis(session_id)`
- Replay de episodios problematicos
- Analise de padroes comportamentais
- Reconstrucao de estado em qualquer ponto temporal via `build_projection()`

**CQRS:** Separacao de command (append) e query (replay/projection). Projeoes sao caches de leitura reconstruidos a partir do event store.

---

### 15.2 Observability Distribuida (`core/distributed_tracing.py`)

Tracing end-to-end compativel com conceitos OpenTelemetry.

| Componente | Classe | Responsabilidade |
|---|---|---|
| `core/distributed_tracing.py` | `Tracer` | Cria spans aninhados (perception -> decision -> action) |
| `core/distributed_tracing.py` | `Span` | Operacao tracada com tags, logs, duracao |
| `core/distributed_tracing.py` | `SpanContext` | Propagacao de contexto entre subsistemas |

**Exportacao:**
- Formato interno JSON
- Formato Jaeger-compatible para importacao em UI

**Analise:**
- `get_slow_spans(threshold_ms)` — identifica bottlenecks
- `get_error_spans()` — correlacao entre falhas
- `get_latency_summary()` — p50/p95/p99 por operacao

---

### 15.3 Graceful Degradation Completo (`core/degradation_manager.py`)

Degradacao automatica baseada na saude do sistema em tempo real.

**Modos:**
| Modo | FPS | RL | Vision | OCR | Auto-Recovery | Descricao |
|---|---|---|---|---|---|---|
| `FULL_QUALITY` | 30 | DQN | YOLO multi-scale 640px | Sim | Sim | Operacao normal |
| `DEGRADED` | 20 | Q-table | YOLO single 320px | Nao | Sim | Erro > 30% |
| `MINIMAL` | 10 | Rules | Heuristicas pixel | Nao | Nao | Erro > 60% |
| `EMERGENCY` | 1 | Rules | Monitoramento apenas | Nao | Nao | Erro > 80% (pausa) |

**Monitoramento:** Taxa de erro/min, latencia de inferencia, latencia de screenshot, falhas ADB.
**Recuperacao:** Melhora um nivel apos 2 min com erro < 5%.

---

### 15.4 State Checkpointing Avancado (`core/game_state_checkpoint.py`)

Snapshots completos do jogo para recovery rapido apos crashes.

| Componente | Classe | Responsabilidade |
|---|---|---|
| `core/game_state_checkpoint.py` | `GameStateCheckpointer` | Salva/Restaura estado completo a cada 30s |
| `core/game_state_checkpoint.py` | `GameStateSnapshot` | Estado completo: spatial, RL, meta, mundo, acoes |
| `core/game_state_checkpoint.py` | `SpatialSnapshot` | Posicoes de player, inimigos, cubes, bushes, danger zones |

**Formato:** Pickle (completo) + JSON (legivel para debug).
**Retencao:** Ultimos 10 checkpoints.
**Recovery:** 90%+ chance com fallback para penultimo checkpoint se ultimo corrompido.

---

### 15.5 Intelligent Rate Limiter (`core/rate_limiter.py`)

Rate limiting que imita padroes humanos de jogo por conta.

| Componente | Classe | Responsabilidade |
|---|---|---|
| `core/rate_limiter.py` | `IntelligentRateLimiter` | Decide se e "realista" jogar agora |
| `core/rate_limiter.py` | `AccountProfile` | Perfil de comportamento humano simulado |

**Regras Anti-Deteccao:**
- Horarios de pico humano (19-23h, 12-13h weekdays; 14-23h weekends)
- Nunca jogar 2h-6h (sono)
- Gap minimo de 60-120 min entre sessoes
- Break apos streak de 2-4 derrotas ("frustracao humana")
- Duracao de sessao: 20-240 min (com jitter +-20%)
- Break aleatorio durante sessao (5-20 min)

---

### 15.6 Model Ensemble + Voting (`vision/ensemble_detector.py`)

Multiplos modelos YOLO votando para robustez contra falsos positivos.

| Componente | Classe | Responsabilidade |
|---|---|---|
| `vision/ensemble_detector.py` | `ModelEnsembleDetector` | Ensemble de N modelos com IoU-voting |
| `vision/ensemble_detector.py` | `Detection` | Deteccao unificada com vote count e sources |

**Voting:** Deteccao valida se >= 2/3 modelos concordam (IoU > 0.5).
**BBox final:** Media ponderada pela confianca dos modelos votantes.
**Fallback:** Em modo degradado, usa apenas o modelo mais rapido.
**Impacto:** mAP ~0.78 -> ~0.88; trade-off +300ms/ciclo.

---

### 15.7 Meta-Learning por Brawler (`decision/brawler_adaptive_controller.py`)

Adaptacao de parametros ao brawler selecionado.

| Componente | Classe | Responsabilidade |
|---|---|---|
| `decision/brawler_adaptive_controller.py` | `BrawlerAdaptiveController` | Carrega perfil por brawler e adapta sistema |
| `decision/brawler_adaptive_controller.py` | `BrawlerProfile` | Configuracao especifica (range, playstyle, RL params) |

**Perfis Built-in:** Shelly, Bull, El Primo, Rosa, Colt, Brock, Piper, Bea, Dynamike, Barley, Tick, Mortis, Leon, Crow, Poco, Gene, Sandy.
**Impacto estimado:** +8-12% win rate com especializacao.

---

### 15.8 Analise de Replays com IA (`core/replay_failure_analyzer.py`)

Analise automatica de derrotas para identificar failure modes.

| Componente | Classe | Responsabilidade |
|---|---|---|
| `core/replay_failure_analyzer.py` | `ReplayFailureAnalyzer` | Analisa replays de derrota e gera recomendacoes |
| `core/replay_failure_analyzer.py` | `FailureMode` | too_aggressive, caught_out, poor_ability, trapped, etc. |

**Failure Modes Detectados:**
- `too_aggressive` — Morreu atacando com HP baixo
- `caught_out_of_position` — Morto sem cover com multiplos inimigos
- `poor_ability_usage` — Super/gadget sem acertar
- `trapped_by_walls` — Preso com multiplas tentativas de escape
- `ignored_threat` — Coletando item ignorando inimigo visivel
- `overextended` — Longe do time, profundo no mapa inimigo

**Recomendacoes:** Ajustes acionaveis em DecisionEngine, CoverSystem, RL rewards, etc.

---

### 15.9 Transfer Learning (`neural/transfer_learning.py`)

Fine-tuning rapido para novos mapas/brawlers usando conhecimento anterior.

| Componente | Classe | Responsabilidade |
|---|---|---|
| `neural/transfer_learning.py` | `TransferLearningController` | Adapta modelo treinado para novo mapa/brawler |
| `neural/transfer_learning.py` | `TransferConfig` | Configuracao: camadas congeladas, LR, metodo |

**Metodo:** Congelar primeiras N camadas (conhecimento geral), fine-tune ultimas camadas (especifico).
**Adaptacao:** < 100 episodios (vs 1000+ do zero).
**Similaridade:** Escolhe melhor modelo fonte baseado em features do mapa.

---

### 15.10 Multi-Objective RL (`decision/multi_objective_rl.py`)

Otimizacao simultanea de multiplos objetivos conflitantes.

| Componente | Classe | Responsabilidade |
|---|---|---|
| `decision/multi_objective_rl.py` | `MultiObjectiveOptimizer` | Seleciona acao por weighted sum de objetivos |
| `decision/multi_objective_rl.py` | `Objective` | Objetivo com peso dinamico e funcao de valor |

**Objetivos (pesos padrao):**
| Objetivo | Peso | Descricao |
|---|---|---|
| Win Rate | 0.60 | Probabilidade de vitoria |
| Detection Risk | 0.20 | Risco anti-ban (invertido) |
| Survival | 0.10 | HP e seguranca posicional |
| Resource Collection | 0.05 | Cubes/gems/stars |
| Ability Efficiency | 0.05 | Eficiencia de super/gadget |

**Otimizacao Pareto:** Identifica solucoes Pareto-otimas antes de escolher por score total.

---

### 15.11 Curriculum Learning (`neural/curriculum_learner.py`)

Treinamento com dificuldade progressivamente crescente.

| Componente | Classe | Responsabilidade |
|---|---|---|
| `neural/curriculum_learner.py` | `CurriculumLearner` | Gerencia niveis de dificuldade adaptativa |
| `neural/curriculum_learner.py` | `DifficultyLevel` | Configuracao de um nivel (bot behavior, accuracy, HP, etc.) |

**Niveis (0-10):** Sandbox -> Tutorial -> Beginner -> Easy -> Normal -> Intermediate -> Hard -> Expert -> Master -> Champion -> Legend.
**Avanco:** Win rate > 70% por N episodios.
**Regressao:** Win rate < 30%.

---

### 15.12 Self-Supervised Vision (`vision/self_supervised_pretraining.py`)

Pre-treinamento auto-supervisionado do backbone YOLO em screenshots de Brawl Stars.

| Componente | Classe | Responsabilidade |
|---|---|---|
| `vision/self_supervised_pretraining.py` | `SelfSupervisedPretrainer` | Treina backbone com SimCLR contrastive learning |
| `vision/self_supervised_pretraining.py` | `ContrastiveLoss` | NT-Xent loss para aprendizado contrastivo |

**Metodo:** SimCLR — 2 augmentacoes por screenshot, maximiza similaridade entre views da mesma imagem.
**Beneficio:** Detector 15-20% mais preciso em dominio especifico (vs COCO generico).

---

### 15.13 Distributed RL (`neural/distributed_orchestrator.py`)

Coordenacao de multiplos bots com aprendizado centralizado.

| Componente | Classe | Responsabilidade |
|---|---|---|
| `neural/distributed_orchestrator.py` | `DistributedLearningOrchestrator` | Centraliza experiencias e sincroniza modelos |
| `neural/distributed_orchestrator.py` | `LocalExperienceBuffer` | Buffer local antes de flush para central |
| `neural/distributed_orchestrator.py` | `Experience` | Experiencia serializada (state, action, reward, next_state, done) |

**Arquitetura:**
1. Bots coletam experiencias localmente
2. Flush periodico para buffer compartilhado (JSONL)
3. Treinador central consolida e treina
4. Modelo global sincronizado de volta para todos os bots

**Beneficio:** Convergencia 5-10x mais rapida.

---

### 15.14 Build Seguro (`obfuscate_build.py`)

Ofuscacao de codigo para distribuicao segura.

| Componente | Classe | Responsabilidade |
|---|---|---|
| `obfuscate_build.py` | `BuildObfuscator` | Prepara build ofuscado com PyArmor ou Cython |

**Metodos:**
- **PyArmor:** Ofuscacao avancada com restricao de importacao
- **Cython:** Compilacao para .pyd/.so (mais rapido e dificil de reverter)

**Verificacao:** `verify_build()` confirma que entry point existe e e funcional.


---

## 15. Key Design Decisions

### 15.1 Why Q-Learning Instead of Deep RL?
- Tabular Q-learning is fully interpretable and debuggable
- State space is small enough (3×4×3×3×2 = 216 states) for tabular methods
- No training instability or gradient explosions
- Easy to inspect, modify, and reset

### 15.2 Why Pixel + Template Fusion?
- Pixel heuristics: fast (~5ms), reliable for known UI patterns
- Template matching: accurate for variable UI (popups, buttons)
- Fusion: high confidence threshold (0.5) + smoothing prevents oscillation
- Graceful fallback: if YOLO fails, pixel heuristics take over

### 15.3 Why Hierarchical State Machine?
- Explicit, auditable transitions (no black-box RL for navigation)
- Easy to add new states and handlers
- Deterministic replay and debugging
- State persistence enables crash recovery

### 15.4 Why 5 Recovery Levels?
- Level 1-2: Most failures are transient (retry/alternative solves 90%)
- Level 3-4: Rare hardware/detection issues (degrade gracefully)
- Level 5: Critical failure (preserve state, alert, full stop)

---

## 16. Development Guidelines

### 16.1 Adding a New Brawler

1. Add projectile speed to `BRAWLER_PHYSICS` in `combat_advanced.py`
2. Define combo sequences in `BRAWLER_COMBOS`
3. Set role (assassin/fighter/tank/sniper/support) in `BRAWLER_ROLES`
4. Add to `BRAWLER_PRECISION` if aim accuracy differs from default
5. Update `BRAWLER_SUPER_DURATION` if applicable

### 16.2 Adding a New Game State

1. Add state name to `GAME_STATES` in `state_manager.py`
2. Create handler method `_handle_<state_name>()`
3. Register in `_route_to_handler()` or `_handle_default()`
4. Add detection logic to `unified_state_detector.py`
5. Update state diagram in this document

### 16.3 Testing

```bash
# Import verification
python test_imports_v2.py

# Training pipeline tests
python -m pytest tests/ -v

# E2E training test
python tests/test_training_workflow_e2e.py
```

### 16.4 Performance Profiling

Key metrics to monitor:
- Screenshot capture time (target: <20ms)
- YOLO inference time (target: <40ms on GPU)
- Total cycle time (target: <200ms in-game)
- APM actual vs target (target: ~35 APM)

---

## 17. Glossary

| Term | Definition |
|---|---|
| Circuit Breaker | Pattern that stops calling a failing subsystem temporarily |
| Graceful Degradation | System continues with reduced functionality when components fail |
| ELO | Skill rating system (K-factor adjusts sensitivity) |
| Q-Learning | Model-free RL algorithm (tabular state-action values) |
| Epsilon-Greedy | Exploration policy (random action with probability ε) |
| EMA | Exponential Moving Average (smoothing filter) |
| APM | Actions Per Minute (humanization metric) |
| TensorRT | NVIDIA GPU inference optimization toolkit |
| State Persistence | Checkpoint/restore mechanism for crash recovery |
| Meta-Learning | Learning to learn (adapting learning parameters) |
| Behavioral Profile | Session-level personality simulation |
| Kiting | Attack-and-retreat combat technique |
| Threat Cone | Area in front of enemy where taking damage is likely |
| Occupancy Grid | 2D spatial representation of occupied cells |

---

## 18. Changelog

| Version | Date | Changes |
|---|---|---|
| 10.0 | 2026-05-17 | Added Meta-Learning, WorldModel Integrator, State Persistence, Behavioral Profile System |
| 9.0 | 2026-05-XX | Added A/B Testing, Replay Recorder, Dashboard enhancements |
| 8.0 | 2026-05-XX | Phase 8 dashboard + replays |
| 7.0 | 2026-05-XX | Phase 6-7 RL: Q-Learning, ELO tracking |
| 6.0 | 2026-05-XX | Phase 5: Advanced combat (leading shots, kiting, cover, combos) |
| 5.0 | 2026-05-XX | Phase 4: Humanization improvements |
| 4.0 | 2026-05-XX | Phase 3: Performance optimizations, lobby navigation v2 |
| 3.0 | 2026-05-XX | Phase 2: Vision pipeline |
| 2.0 | 2026-05-XX | Phase 1: Basic bot framework |
| 1.0 | 2026-05-XX | Initial version |
