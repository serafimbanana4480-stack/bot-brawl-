# API Documentation - Soberana Omega Bot

## Overview
This document describes all available API endpoints for the Soberana Omega Brawl Stars bot.

Base URL: `http://localhost:8003`

---

## Bot Control

### POST /api/brawl-stars/setup
**Description:** Initialize the bot: connect to emulator, load models, configure components. Must be called before starting the bot.

**Response:**
```json
{
  "success": true,
  "message": "Bot setup completed successfully"
}
```

### POST /api/brawl-stars/start
**Description:** Start the bot. Bot must be set up first.

**Response:**
```json
{
  "success": true,
  "message": "Bot started successfully"
}
```

### POST /api/brawl-stars/stop
**Description:** Stop the bot.

**Response:**
```json
{
  "success": true,
  "message": "Bot stopped successfully"
}
```

### GET /api/brawl-stars/status
**Description:** Returns the current bot status including running state, current brawler, session duration, and safety system status.

**Response:**
```json
{
  "success": true,
  "status": {
    "running": true,
    "current_brawler": "Colt",
    "session_duration": 3600,
    "safety_status": {...}
  }
}
```

### GET /api/brawl-stars/recommended-action
**Description:** Returns the recommended action based on current state. Possible actions: "start_match", "switch_brawler", "continue"

**Query Parameters:** None

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-07T23:00:00",
  "action": "continue",
  "reason": "Continue with current brawler"
}
```

---

## Monitoring

### GET /api/brawl-stars/telemetry
**Description:** Returns detailed real-time telemetry metrics including APM, win rate, suspicion score, human likeness score, movement statistics, enemy tracking statistics, and session statistics.

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-07T23:00:00",
  "bot_status": {...},
  "safety_status": {...},
  "tracker_status": {
    "active_tracks": 3,
    "frame_count": 1000,
    "next_track_id": 4
  },
  "training_metrics": {...},
  "match_statistics": {
    "total_matches": 50,
    "win_rate": 65.5,
    "avg_trophies_per_match": 12.3,
    "last_10_matches": [...]
  },
  "performance_metrics": {
    "apm": 45,
    "suspicion_score": 15,
    "human_likeness_score": 92.5,
    "avg_velocity": 150.2,
    "max_acceleration": 300.5,
    "curvature_variance": 0.05
  }
}
```

### GET /api/brawl-stars/logs
**Description:** Returns recent logs with optional filters.

**Query Parameters:**
- `n` (int, optional): Number of recent logs to return (default: 100)
- `category` (str, optional): Filter by category (e.g., "COMBAT", "SAFETY")
- `level` (str, optional): Filter by level (e.g., "DEBUG", "INFO", "WARNING", "ERROR")

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-07T23:00:00",
  "logs": [
    {
      "timestamp": "2026-05-07T23:00:00",
      "level": "INFO",
      "category": "COMBAT",
      "message": "Inimigo detectado"
    }
  ],
  "count": 100
}
```

### GET /api/brawl-stars/health
**Description:** Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-05-07T23:00:00"
}
```

### GET /api/brawl-stars/performance
**Description:** Get performance metrics.

**Response:**
```json
{
  "success": true,
  "performance": {
    "cpu_usage": 45.2,
    "memory_usage": 512,
    "fps": 30
  }
}
```

### GET /api/brawl-stars/diagnostics
**Description:** Get bot diagnostics.

**Response:**
```json
{
  "success": true,
  "diagnostics": {...}
}
```

---

## Emulators

### GET /api/brawl-stars/emulators
**Description:** Returns available emulators with optional filter by type.

**Query Parameters:**
- `emulator_type` (str, optional): Filter by emulator type (e.g., "LDPlayer", "BlueStacks", "Nox")

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-07T23:00:00",
  "emulators": [
    {
      "name": "LDPlayer-1",
      "type": "LDPlayer",
      "window_title": "LDPlayer",
      "adb_path": "C:\\LDPlayer\\adb.exe",
      "port": 5555
    }
  ],
  "count": 1
}
```

### GET /api/brawl-stars/emulators/{name}
**Description:** Returns specific emulator by name or window title.

**Path Parameters:**
- `name` (str): Emulator name or window title

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-07T23:00:00",
  "emulator": {
    "name": "LDPlayer-1",
    "type": "LDPlayer",
    "window_title": "LDPlayer",
    "adb_path": "C:\\LDPlayer\\adb.exe",
    "port": 5555
  }
}
```

---

## Configuration

### GET /api/brawl-stars/config
**Description:** Returns the current bot configuration including safety settings.

**Response:**
```json
{
  "success": true,
  "config": {
    "trophy_limit": 400,
    "warning_trophies": 380,
    "max_session_hours": 3,
    "min_apm": 20,
    "max_apm": 60,
    "auto_pause": true,
    "brawler_queue": [...],
    "safety_settings": {...}
  }
}
```

### POST /api/brawl-stars/config
**Description:** Update bot configuration. Changes take effect immediately.

**Request Body:**
```json
{
  "trophy_limit": 400,
  "warning_trophies": 380,
  "max_session_hours": 3,
  "min_apm": 20,
  "max_apm": 60,
  "auto_stop_on_detection": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Configuration updated successfully"
}
```

---

## Statistics

### GET /api/brawl-stars/match-history
**Description:** Get match history.

**Query Parameters:**
- `limit` (int, optional): Number of matches to return (default: 50)

**Response:**
```json
{
  "success": true,
  "match_history": [...],
  "total": 50
}
```

---

## Safety

### GET /api/brawl-stars/safety-status
**Description:** Get safety system status.

**Response:**
```json
{
  "success": true,
  "safety_status": {
    "suspicion_score": 15,
    "human_likeness_score": 92.5,
    "next_break_in_minutes": 30,
    "break_duration": 300
  }
}
```

### POST /api/brawl-stars/humanization
**Description:** Update humanization settings.

**Request Body:**
```json
{
  "min_delay": 0.3,
  "max_delay": 1.5,
  "mistake_probability": 0.1,
  "tremor_amplitude": 2.0,
  "bezier_control_variation": 0.3
}
```

**Response:**
```json
{
  "success": true,
  "message": "Humanization settings updated"
}
```

### POST /api/brawl-stars/safety
**Description:** Update safety settings.

**Request Body:**
```json
{
  "max_trophies": 400,
  "warning_trophies": 380,
  "max_session_hours": 3,
  "min_apm": 20,
  "max_apm": 60
}
```

**Response:**
```json
{
  "success": true,
  "message": "Safety settings updated"
}
```

---

## Brawler Queue

### GET /api/brawl-stars/queue
**Description:** Get current brawler queue.

**Response:**
```json
{
  "success": true,
  "queue": [
    {
      "name": "Colt",
      "target_trophies": 350,
      "target_wins": 10
    }
  ]
}
```

### POST /api/brawl-stars/brawler/add
**Description:** Add brawler to queue.

**Request Body:**
```json
{
  "name": "Colt",
  "target_trophies": 350,
  "target_wins": 10
}
```

**Response:**
```json
{
  "success": true,
  "message": "Brawler added to queue"
}
```

### DELETE /api/brawl-stars/brawler/{name}
**Description:** Remove brawler from queue.

**Path Parameters:**
- `name` (str): Brawler name

**Response:**
```json
{
  "success": true,
  "message": "Brawler removed from queue"
}
```

---

## Auto-Tuning

### POST /api/brawl-stars/auto-tuning/tune
**Description:** Runs auto-tuning cycle to adjust parameters based on match history. Analyzes performance and adjusts attack distance, shot cooldown, safety threshold, etc.

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-08T00:00:00",
  "analysis": {
    "win_rate": 0.45,
    "avg_kills_per_match": 1.2,
    "avg_damage_per_match": 1500,
    "performance_rating": 0.4
  },
  "adjustments": {
    "attack_distance": 15,
    "shot_cooldown": -5,
    "safety_threshold": 5,
    "aggressiveness": -10
  },
  "current_params": {
    "attack_distance": 230,
    "shot_cooldown": 0.43,
    "safety_threshold": 0.52,
    "aggressiveness": 0.4
  }
}
```

### GET /api/brawl-stars/auto-tuning/status
**Description:** Returns the current status of the auto-tuning system.

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-08T00:00:00",
  "enabled": true,
  "status": {
    "last_tuning_time": 1715145600.0,
    "last_tuning_hours_ago": 2.5,
    "tuning_history_count": 5,
    "current_params": {
      "attack_distance": 230,
      "shot_cooldown": 0.43,
      "safety_threshold": 0.52,
      "aggressiveness": 0.4
    },
    "config": {
      "min_matches_for_tuning": 10,
      "tuning_interval_hours": 1,
      "win_rate_target": 0.6
    }
  }
}
```

### POST /api/brawl-stars/auto-tuning/reset
**Description:** Resets all auto-tuned parameters to their default values.

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-08T00:00:00",
  "message": "Parameters reset to defaults"
}
```

---

## WebSocket

### WS /ws/brawl-stars
**Description:** WebSocket endpoint for real-time logs.

**Message Format:**
```json
{
  "timestamp": "2026-05-07T23:00:00",
  "level": "INFO",
  "category": "COMBAT",
  "message": "Inimigo detectado"
}
```

---

## Error Handling

All endpoints return errors in the following format:

```json
{
  "success": false,
  "error": "Error message description"
}
```

Common error codes:
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error

---

## Rate Limiting
Rate limiting is enforced via `slowapi`:
- **Read endpoints (GET):** 10 requests per second
- **Write endpoints (POST/PUT/DELETE):** 2 requests per second

Exceeded limits return HTTP `429 Too Many Requests`.

---

## Authentication
Control endpoints (POST/PUT/DELETE and non-public GET) require an API key passed in the `X-API-Key` header.

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8003/api/brawl-stars/start
```

The API key is read from `config.json` under `api.api_key`. If `api_key` is `null`, control endpoints are restricted to localhost (`127.0.0.1`, `localhost`, `::1`) only.

### Public Endpoints (no API key required)
The following endpoints are public and do not require authentication:
- `GET /health`
- `GET /health/deep`
- `GET /ready`
- `GET /metrics`
- `GET /api/brawl-stars/health`
- `GET /api/brawl-stars/status`
- `GET /docs`
- `GET /redoc`
- `GET /openapi.json`

---

## CORS
CORS origins are validated against the `api.cors_origins` array in `config.json`. The wildcard `*` is explicitly rejected. If no valid origins are configured, safe defaults (`http://localhost:3000`, `http://localhost:5173`) are used.
