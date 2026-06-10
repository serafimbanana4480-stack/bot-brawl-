# 🌵 Brawl Stars Bot — Soberana Omega

AI-powered Brawl Stars automation bot with real-time computer vision (YOLO), adaptive combat decision-making, anti-ban humanization, and a full FastAPI backend with WebSocket support.

---

## 📐 Architecture

The project follows a **hexagonal architecture** with clear separation between core business logic, adapters, and subsystems.

```
┌─────────────────────────────────────────────────────────────┐
│  API Layer (FastAPI + WebSocket)                           │
│  ├── Rate limiting (slowapi)                               │
│  ├── API key auth + CORS whitelist                         │
│  └── Prometheus / OpenTelemetry metrics                    │
├─────────────────────────────────────────────────────────────┤
│  Core Subsystems (core/subsystems/)                        │
│  ├── EmulatorSubsystem    → ADB / Win32 capture            │
│  ├── VisionSubsystem      → YOLO inference + OCR fallback    │
│  ├── DecisionSubsystem    → RL-based combat decisions        │
│  ├── SafetySubsystem      → APM limits, break scheduling   │
│  ├── LearningSubsystem    → PPO + experience buffer          │
│  └── UISubsystem          → Dashboard + diagnostics        │
├─────────────────────────────────────────────────────────────┤
│  Legacy Facade (pylaai_real/)                              │
│  └── UnifiedStateDetector, LobbyAutomator, PlayEngine      │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure                                            │
│  ├── models/              → YOLO .pt files                 │
│  ├── dataset/             → Training data + Roboflow merge │
│  ├── images/templates/    → UI templates for state detect  │
│  └── config.json          → Centralized configuration      │
└─────────────────────────────────────────────────────────────┘
```

### Key directories

| Directory | Purpose |
|-----------|---------|
| `api/` | FastAPI routes (emulator detection, brawler management) |
| `core/` | Hexagonal core: subsystems, state detection, error recovery, config manager |
| `pylaai_real/` | Legacy facade preserving backward compatibility |
| `vision/` | YOLO wrappers, template matching, OCR fallback |
| `decision/` | RL agents, reward shaping, tactical bridge |
| `training/` | YOLO training pipelines, PPO trainer, curriculum learning |
| `dataset/` | Data collection, cleaning, synthetic generation |
| `tests/` | Pytest suite (80+ test modules) |
| `docs/` | Architecture decisions, API docs, troubleshooting guides |

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10+
- **Emulator**: BlueStacks or LDPlayer at **1920×1080**
- **ADB** enabled in the emulator (Android debugging)
- **Brawl Stars** open in the main lobby

### Installation

```bash
# 1. Clone / enter the project
cd "bot brawl"

# 2. Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install runtime dependencies
pip install -r requirements.txt
pip install -e .

# 4. (Optional) Install dev dependencies
pip install -r requirements-dev.txt
pre-commit install
```

### Running the bot

#### Mode 1 — API Server (recommended)

```bash
# Start the FastAPI server
python -m uvicorn api_server:app --host 127.0.0.1 --port 8003 --reload
# or
make run
```

Then control the bot via HTTP:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/brawl-stars/setup` | Setup | Connect to emulator & load models |
| `POST /api/brawl-stars/brawler/add` | Setup | Queue a brawler to play |
| `POST /api/brawl-stars/start` | Control | Start automation |
| `POST /api/brawl-stars/stop` | Control | Stop automation |
| `GET /api/brawl-stars/status` | Monitor | Current bot state |
| `GET /health` | Monitor | Health check + diagnostics |

#### Mode 2 — Data Collection (RL training data)

Set `rl.data_collection_mode: true` in `config.json`, then run:

```bash
python wrapper.py --mode collect
```

This captures screenshots, game-state grids, and reward signals without sending actions.

#### Mode 3 — Direct wrapper (legacy)

```bash
python wrapper.py --mode play --brawler colt
```

---

## 🧠 Training Models

### YOLO (vision)

```bash
# Core schema (4 classes) — quick iteration
python train.py --schema core --epochs 50 --batch 8 --device auto

# Extended schema (8 classes) — production
python train.py --schema extended --epochs 100 --batch 16 --device cuda

# Full schema (35 classes) — maximum accuracy
python train.py --schema full --epochs 150 --batch 8 --model-size m
```

Trained models are saved to `models/` and registered automatically in `model_registry.json`.

### PPO (reinforcement learning)

```bash
# Train the combat decision model
python -m training.ppo_trainer --timesteps 1000000 --batch-size 64
```

Check `training/curriculum_trainer.py` for curriculum-learning schedules.

---

## ⚙️ Configuration

All runtime settings live in **`config.json`**:

```json
{
  "game": {
    "mode": "gem_grab",
    "brawler": "colt",
    "resolution": "1920x1080"
  },
  "emulator": {
    "type": "bluestacks",
    "adb_port": 5554,
    "window_title": "BlueStacks App Player"
  },
  "safety": {
    "max_session_hours": 3.0,
    "max_apm": 60,
    "auto_stop_on_detection": true
  },
  "anti_ban": {
    "enabled": true,
    "max_win_rate": 0.75,
    "max_matches_per_hour": 8
  },
  "vision": {
    "main_model": "brawlstars_yolov8.pt",
    "confidence_threshold": 0.37,
    "nms_iou_threshold": 0.45
  },
  "api": {
    "host": "127.0.0.1",
    "port": 8003,
    "api_key": null,
    "cors_origins": ["http://localhost:3000"]
  },
  "rl": {
    "enabled": false,
    "data_collection_mode": true,
    "experience_buffer_size": 10000
  }
}
```

Copy `config.example.json` to `config.json` and edit to taste.

---

## 🐳 Docker

```bash
# Build image
make build
# or
docker build -t brawl-bot:latest .

# Run container
docker run -p 8000:8000 --rm brawl-bot:latest

# Compose (if using docker-compose.yml)
docker-compose up --build
```

The Dockerfile is multi-stage:
1. **Builder** — pre-compiles Python wheels with system CV libs
2. **Runtime** — slim image with ADB + wheels, runs as non-root user

---

## 🧪 Tests

```bash
# Run full suite with coverage
make test
# or
pytest --cov=. --cov-report=term-missing --cov-report=html -v

# Run specific test file
pytest tests/test_core_functionality.py -v

# Lint & format
make lint
make format

# Type check
make typecheck

# Security audit
make security
```

---

## 📦 Project Structure

```
.
├── api/                    # FastAPI routes
├── api_server.py           # Main FastAPI app (WebSocket, metrics, auth)
├── backend/                # Legacy backend entrypoint
├── brawl_bot/              # Legacy bot package
├── core/                   # Hexagonal core
│   ├── subsystems/         # Emulator, Vision, Decision, Safety, Learning, UI
│   ├── combat/             # Combat engine & ability manager
│   ├── movement/           # Pathfinding & positioning
│   ├── abilities/          # Brawler ability definitions
│   ├── events/             # Event bus
│   └── ports/              # Interface contracts
├── decision/               # RL agents & reward shaping
├── dataset/                # Data collection, cleaning, synthetic data
├── docs/                   # Architecture & troubleshooting docs
├── images/
│   └── templates/          # UI templates (play, attack, super, gadget, joystick…)
├── models/                 # YOLO .pt checkpoints + registry
├── neural/                 # Neural network utilities
├── plugins/                # Plugin system (orchestrator, learning mode)
├── pylaai_real/            # Legacy facade (backward-compatible)
├── scripts/                # Utility scripts
├── tests/                  # Pytest suite
├── training/               # YOLO & PPO training pipelines
├── utils/                  # Shared helpers
├── vision/                 # CV engine, OCR, template matching
├── wrapper.py              # Main entrypoint (PylaAIEnhanced facade)
├── config.json             # Runtime configuration
├── pyproject.toml          # Package metadata & dependencies
├── requirements.txt        # Pinned runtime deps
├── requirements-dev.txt    # Dev & test deps
├── Dockerfile              # Multi-stage container build
├── docker-compose.yml      # Compose stack
└── Makefile                # Common tasks
```

---

## 🔒 Security & Humanization

- **Curves de Bézier** — smooth mouse movements via `humanization.py`
- **Safety System** — APM caps, session limits, automatic breaks (`safety_system.py`)
- **Anti-Ban** — win-rate clamping, fingerprint rotation, action noise (`core/anti_ban.py`)
- **Rate Limiting** — `slowapi` on all API endpoints
- **CORS Whitelist** — only `localhost:3000/5173` by default
- **API Key** — optional `api.api_key` in `config.json`

---

## 🐞 Troubleshooting

| Symptom | Fix |
|---------|-----|
| Emulator not detected | Ensure ADB is in PATH and USB debugging is enabled |
| GPU / CUDA errors | Set `CUDA_VISIBLE_DEVICES=""` to force CPU inference |
| Template matching fails | Check `images/templates/` files are valid PNGs (not empty) |
| High memory usage | Reduce `vision.confidence_threshold` or use `yolov8n.pt` |
| API 429 errors | Increase `api.rate_limit.read_per_second` in `config.json` |

See `docs/TROUBLESHOOTING.md` and `docs/INSTALLATION_GUIDE.md` for detailed guides.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests & update docs
4. Open a Pull Request

See `CONTRIBUTING.md` for detailed guidelines.

---

**Developed as part of the Soberana Omega ecosystem.**
