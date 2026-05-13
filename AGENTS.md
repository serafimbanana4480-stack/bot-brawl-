# AGENTS.md - Soberana Omega Brawl Stars Bot

## Project Overview

AI-powered Brawl Stars bot with real-time computer vision (YOLO), adaptive combat, intelligent lobby navigation, and anti-ban humanization.

## Architecture

### Core Modules
- `wrapper.py` - Main entry point (`PylaAIEnhanced` class)
- `pylaai_real/state_manager.py` - Game state machine (lobby, in_game, loading, etc.)
- `pylaai_real/unified_state_detector.py` - Unified pixel + template matching state detection
- `pylaai_real/play.py` - Combat engine (targeting, abilities, leading shots)
- `pylaai_real/movement.py` - Movement and pathfinding
- `pylaai_real/lobby_automator.py` - Lobby automation (play button, brawler select)
- `pylaai_real/lobby_navigator.py` - Intelligent lobby navigation v2 (popup detection, smart play button, event detection, fast brawler select)
- `pylaai_real/humanization_utils.py` - Humanization utilities (jitter, Bezier curves, reaction delays, APM control)
- `pylaai_real/screenshot_taker.py` - Win32 screenshot capture
- `emulator_controller.py` - ADB input controller
- `safety_system.py` - Anti-ban safety limits
- `humanization.py` - Legacy humanization engine

### Package Structure
- `brawl_bot/` - Python package shim for imports
- `dataset/` - ML dataset collection (`collector.py`)
- `decision/` - Brawler selection logic
- `training/` - Professional training pipeline
- `tests/` - Unit tests

## Key Improvements (Phase 3)

### 1. Performance Optimizations
- **Screenshot caching** in `state_manager.py`: Reuses screenshots within 150ms to avoid duplicate captures
- **Adaptive cycle delays**: Faster cycles in-game (0.12-0.25s), slower in static states (0.3-0.6s)
- **Reduced monitor loop delay** in `wrapper.py`: 0.3s in-game, 0.8-1.2s otherwise
- **Reduced shot cooldown** in `play.py`: Default reduced from 0.45s to 0.35s with +/-15% jitter

### 2. Humanization (`humanization_utils.py`)
- `human_delay()` - Randomized delays with configurable jitter
- `jitter_coords()` - Coordinate jitter for natural-looking clicks
- `jitter_value()` - Value variation for cooldowns/timings
- `bezier_curve_points()` - Bezier curve generation for human-like swipes
- `HumanPauseSimulator` - Simulates strategic pauses and reaction delays (120-400ms)
- `APMController` - Actions Per Minute control (target ~35 APM)

### 3. State Detection Improvements (`unified_state_detector.py`)
- **5 new game states mapped**: tutorial, news, brawler_unlock, season_reset, shop
- **In-game detection enhanced**: HP bar (green) + match timer (white) verification
- **Smoothing/voting system**: Prevents rapid state oscillation (requires 3/5 votes to change state)
- **Pixel + template fusion**: Higher confidence threshold (0.5) for pixel matching, smoothing applied to all detections

### 4. New State Handlers (`state_manager.py`)
- `_handle_tutorial()` - Auto-clicks through tutorial arrows
- `_handle_news()` - Closes news/Brawl Talk popup (X button)
- `_handle_brawler_unlock()` - Clicks proceed on brawler unlock screen
- `_handle_season_reset()` - Clicks proceed on season reset screen
- `_safe_back_to_lobby()` - Helper for state recovery
- `_get_window_size()` - Safe window dimension retrieval

### 5. Lobby Navigation v2 (`lobby_navigator.py`)
- `PopupManager` - Auto-detects and closes popups using color heuristics + template matching
- `SmartPlayButtonDetector` - Searches Play button in 4 regions before falling back to hardcoded coords
- `EventDetector` - Detects special events and clicks event-specific play button
- `BrawlerSelectorFast` - Pre-checks if brawler is already selected, uses coordinate cache, avoids 65% resize that degraded OCR

### 6. Advanced Combat Phase 5 (`combat_advanced.py`) v2.0
- **Leading Shot Engine** — Física de projétil por brawler
  - Velocidade do projétil em pixels/s (conversão automática de tiles/s)
  - Filtro EMA de velocidade do inimigo (suaviza zig-zags, detecta "parado")
  - **HumanAimError** — Modelo de erro humano com:
    - Overshoot bias (atiram à frente do inimigo em movimento)
    - Erro proporcional à distância (mais longe = menos preciso)
    - Clustering temporal (erros correlacionados entre tiros seguidos)
  - Brawler-specific precision: Piper (0.95), El Primo (0.70), etc.
- **Kiting Engine** — Atirar e recuar
  - **Vetor resultante**: Considera TODOS os inimigos (peso inversamente proporcional à distância²)
  - **Preferência por bushes**: Recua para bush na direção de fuga (esconder-se enquanto recua)
  - **Clamp ao ecrã**: Nunca recua para fora dos limites
  - Cooldown adaptativo por role (assassins kita menos)
- **Cover Engine** — Bush strategy
  - Scoring: perto + longe dos inimigos + line-of-sight break
  - **Escape route**: Bonus por bush adjacente (para continuar fugindo)
  - **Power cube bonus**: +200 pontos se bush tem cube próximo
  - **Threat cone**: Penaliza bushes no caminho dos inimigos
- **Combo Manager** — Sequências condicionais
  - Combos por brawler com condições: `any`, `enemy_low_hp`, `enemy_grouped`, `player_full_hp`
  - Só inicia combo se 1ª ação satisfaz condição (ex: Darryl só super+attack se inimigos agrupados)
  - Executa uma ação por ciclo de `play_round()` (sem bloquear)
- **TargetSelector** — Escolha inteligente de alvo
  - Prioriza inimigos com bbox pequena (heurística de low HP)
  - Snipers preferem alvos mais distantes (mais fáceis de acertar)
  - Assassins/fighters priorizam alvos próximos
- **AdvancedCombatStrategy** — Combat State Machine
  - Estados: `neutral` → `aggressive` (combo/full HP) / `defensive` (kite/cover/low HP) / `retreating`
  - Transições automáticas baseadas em HP estimado e brawler role

### 7. Online Learning Phase 6 (`rl_engine.py`, `elo_tracker.py`)
- **Q-Learning tabular** with discrete state space (HP, enemies, distance, ammo, super)
  - 6 actions: `attack`, `move_to_enemy`, `retreat`, `use_super`, `collect_cube`, `idle`
  - Epsilon-greedy with decay: 40% -> 5% exploration over time
  - Frame-by-frame updates during gameplay; backward pass on episode end
  - Persistent Q-table saved to `data/q_table.pkl`
- **ELO tracker** per brawler+map combination
  - Standard ELO formula with dynamic K-factor (40 for low ratings, 16 for high)
  - Auto-saves to `data/elo_ratings.json` every 10 matches
  - `get_best_brawler_for_map()` suggests optimal picks with exploration bonus
- **Integration**:
  - `PlayLogic` queries RL for action when confidence > 0.6 (overrides hardcoded)
  - `StateManager` feeds `(state, action, reward, next_state)` every frame + `end_episode()` on match end
  - `wrapper.py` initializes `OnlineLearner`, saves on shutdown

### 8. Dashboard + Replays + A/B Testing Phase 8 (`dashboard_server.py`)
- **Real-time Web Dashboard** (embedded HTML+JS, no external dependencies)
  - Built-in `http.server` on port 8765 (configurable via `config.json`)
  - **Zero mock data**: all metrics come from live bot runtime
  - Polling every 2s via `/api/live`, `/api/history`, `/api/rewards`, `/api/replays`, `/api/abtest`
  - Displays: current state, brawler, map, matches, win rate, FPS, Q-learning epsilon, ELO ratings, recent events, last screenshot thumbnail
  - Dark-themed responsive UI with live charts (rewards over time, cycle times)
- **Replay Recorder** (`ReplayRecorder`)
  - Saves up to 150 frames (~5 min) per replay to `data/replays/`
  - Each frame stores: screenshot (JPEG 50% quality), state, action, enemy count, reward
  - JSON metadata (`replay.json`) with duration and event list
  - Start/stop controlled via dashboard UI or wrapper hooks
- **A/B Testing Framework** (`ABTestManager`)
  - Compares strategies (e.g., old combat vs Phase 5 advanced combat)
  - Round-robin variant assignment per match (50/50)
  - Tracks matches, wins, losses, win rate, average reward per variant
  - Persists results to `data/ab_tests.json`
  - REST API: `POST /api/abtest/start` and `POST /api/abtest/stop`
- **Integration in `wrapper.py`**
  - Dashboard thread starts on `start()` and stops on `stop()`
  - `_monitor_loop()` feeds live data via `dashboard.update_from_wrapper(self)` every cycle
  - In-game screenshots are thumbnailed (320x180, JPEG 50%) and base64-encoded for the UI
  - `PlayLogic._last_action` and `_last_enemies` expose real combat decisions to the dashboard

### 9. Critical Bug Fixes
- **Circular import fix**: Lazy imports in `safety_system.py` and `auto_tuner.py` for `brawl_bot.realtime_logs`
- **Wrapper aliases fix**: Removed eager `wrapper.py` import from `brawl_bot/__init__.py`, added missing aliases
- **Missing methods fix**: Added `record_frame()`, `flush()`, `start_session()`, `end_session()` to `dataset/collector.py`
- **Pause/Resume fix**: Added `pause()` and `resume()` methods to `StateManager` (required by wrapper retraining hooks)
- **screenshot.w/h fix**: New handlers used non-existent `self.screenshot.w` → replaced with `_get_window_size()` helper

## Important Notes

### ScreenshotTaker
- Does NOT have `w`/`h` attributes. Use `movement.window_w`/`window_h` or get dimensions from captured image shape.
- The `self.screenshot` in `StateManager` is a `ScreenshotTaker` instance.

### StateManager
- `_get_cached_screenshot(max_age=0.15)` reuses recent screenshots. Buffer initialized in `__init__`.
- `pause()` / `resume()` pause the main loop without killing the thread.
- All handlers for non-game states are called without arguments; `in_game` and `unknown` receive the screenshot image.

### PlayLogic Combat
- `_try_smart_attack()` now uses `jitter_value()` for cooldown variation and `HumanPauseSimulator.reaction_delay()` before attacking.
- Attack uses directional swipe (not just tap) for more human-like behavior.

### Imports
- Run `py test_imports_v2.py` to verify all modules load correctly.
- `brawl_bot.wrapper` should be used instead of `wrapper` for relative imports to resolve properly.

## Known Limitations

- No wall/obstacle detection yet (Phase 1 of PLANO_GIGANTE_MELHORIAS.md)
- YOLO inference is still sequential (Phase 2 for parallel pipeline)
- OCR for map names not implemented yet
- Q-Learning uses heuristic rewards (not real game rewards) since HP/damage extraction from screen is not yet implemented

## File Paths
- Project root: `c:/Users/rodri/Desktop/bot brawl/`
- Images/templates: `images/` (or path passed to `LobbyAutomator`)
- Planning docs: `.planning/`
