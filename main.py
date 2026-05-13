"""
main.py - Soberana Omega Entry Point

Unified entry point integrating PylaAI with Safety System, Humanization, and API.
Supports two modes:
  - Bot mode (default): Runs the bot automation loop
  - API mode (--api): Starts the FastAPI server for remote control

Usage:
  python -m brawl_bot.main          # Start bot
  python -m brawl_bot.main --api    # Start API server
  python -m brawl_bot.main --check  # Verify installation
"""

import sys
import time
import json
import logging
import argparse
import io
from pathlib import Path

# Fix Unicode encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger("brawl_bot.main")

# Bot module root
BOT_ROOT = Path(__file__).parent


def _load_config() -> dict:
    """Load centralized config.json."""
    config_path = BOT_ROOT / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")
    return {}


def run_bot(args):
    """Start the bot in automation mode."""
    # Add parent directory to path for imports
    sys.path.insert(0, str(BOT_ROOT.parent))
    from brawl_bot.wrapper import PylaAIEnhanced
    from brawl_bot.safety_system import SafetyConfig
    from brawl_bot.humanization import HumanizationConfig

    config = _load_config()
    safety_cfg = config.get("safety", {})
    human_cfg = config.get("humanization", {})
    recording_cfg = config.get("recording", {})

    logger.info("=" * 60)
    logger.info("  SOBERANA OMEGA - Brawl Stars Bot (PylaAI Enhanced)")
    logger.info("=" * 60)

    # Build configs from centralized file
    safety_config = SafetyConfig(
        max_trophies=safety_cfg.get("max_trophies", 400),
        warning_trophies=safety_cfg.get("warning_trophies", 380),
        max_session_hours=safety_cfg.get("max_session_hours", 3.0),
        min_apm=safety_cfg.get("min_apm", 20),
        max_apm=safety_cfg.get("max_apm", 60),
    )

    humanization_config = HumanizationConfig(
        enabled=bool(args.humanization or human_cfg.get("enabled", True)),
    )

    # Create bot
    bot = PylaAIEnhanced(
        safety_config=safety_config,
        humanization_config=humanization_config,
        diagnostic_mode=args.diagnostic or config.get("diagnostic_mode", False),
        enable_recording=args.record or recording_cfg.get("enabled", False)
    )

    # Setup
    logger.info("[1/3] Running setup...")
    if not bot.setup():
        logger.error("Setup failed. Ensure emulator is running.")
        sys.exit(1)
    logger.info("[1/3] Setup complete")

    # Start
    logger.info("[2/3] Starting bot...")
    if not bot.start():
        logger.error("Failed to start bot")
        sys.exit(1)
    logger.info("[2/3] Bot started")

    logger.info("[3/3] Bot is running. Press Ctrl+C to stop.")
    logger.info("-" * 60)

    # Keep alive
    try:
        while bot.running:
            status = bot.get_status()
            logger.info(
                f"State: {status['current_state']} | "
                f"Matches: {status['matches_played']} | "
                f"Session: {status['session_duration_minutes']:.1f}min | "
                f"Brawler: {status['current_brawler'] or 'None'}"
            )
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Stopping bot (Ctrl+C)...")
        bot.stop()
        logger.info("Bot stopped. Goodbye!")


def run_api():
    """Start the FastAPI server."""
    # Add parent directory to path for imports
    sys.path.insert(0, str(BOT_ROOT.parent))
    from brawl_bot.api import app
    import uvicorn

    logger.info("=" * 60)
    logger.info("  SOBERANA OMEGA - API Server")
    logger.info("=" * 60)
    logger.info(f"  Listening on http://127.0.0.1:8003")
    logger.info(f"  Swagger UI: http://127.0.0.1:8003/docs")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8003,
        reload=False,
        log_level="info"
    )


def run_check():
    """Verify installation and configuration."""
    logger.info("=" * 60)
    logger.info("  SOBERANA OMEGA - Installation Check")
    logger.info("=" * 60)

    checks_passed = 0
    checks_failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal checks_passed, checks_failed
        if condition:
            checks_passed += 1
            logger.info(f"  [PASS] {name}" + (f" ({detail})" if detail else ""))
        else:
            checks_failed += 1
            logger.error(f"  [FAIL] {name}" + (f" ({detail})" if detail else ""))

    # Config
    config_path = BOT_ROOT / "config.json"
    check("config.json exists", config_path.exists())

    lobby_path = BOT_ROOT / "lobby.toml"
    check("lobby.toml exists", lobby_path.exists())

    # Directories
    check("images/ directory", (BOT_ROOT / "images").is_dir())
    check("models/ directory", (BOT_ROOT / "models").is_dir())
    check("data/ directory", (BOT_ROOT / "data").is_dir())
    check("logs/ directory", (BOT_ROOT / "logs").is_dir())

    # Model validation (checks actual model validity, not just file existence)
    try:
        sys.path.insert(0, str(BOT_ROOT.parent))
        from brawl_bot.model_validator import validate_all_models
        val = validate_all_models(delete_fakes=False)
        valid_count = len(val.get("valid", []))
        fake_count = len(val.get("fake", []))
        check(f"YOLO models ({valid_count} valid, {fake_count} fake)", valid_count > 0,
              f"{valid_count} valid, {fake_count} fake models found")
    except Exception as e:
        check("YOLO model validation", False, str(e))

    # Templates
    templates = ["thumbs_down.png", "play_button.png", "brawler_select.png", "joystick.png"]
    for t in templates:
        check(f"Template: {t}", (BOT_ROOT / "images" / t).exists())

    # Dependencies
    deps = {
        "ultralytics": "YOLO model loading",
        "cv2": "Computer vision",
        "easyocr": "OCR text reading",
        "PIL": "Image processing",
        "numpy": "Array operations",
        "fastapi": "API server",
        "toml": "Config parsing",
        "psutil": "Process detection",
    }
    for mod, desc in deps.items():
        try:
            __import__(mod)
            check(f"Module: {mod}", True, desc)
        except ImportError:
            check(f"Module: {mod}", False, f"{desc} - pip install {mod}")

    # Emulator detection
    try:
        # Add parent directory to path for imports
        sys.path.insert(0, str(BOT_ROOT.parent))
        from brawl_bot.emulator_detector import get_emulator_detector
        detector = get_emulator_detector()
        emus = detector.detect_all()
        check("Emulator detection", len(emus) > 0, f"Found {len(emus)} emulator(s)")
    except Exception as e:
        check("Emulator detection", False, str(e))

    # Summary
    logger.info("-" * 60)
    total = checks_passed + checks_failed
    logger.info(f"  Results: {checks_passed}/{total} passed, {checks_failed} failed")

    if checks_failed == 0:
        logger.info("  STATUS: ALL CHECKS PASSED - Ready to run!")
    elif checks_failed <= 3:
        logger.warning("  STATUS: PARTIAL - Some features may not work")
    else:
        logger.error("  STATUS: CRITICAL - Multiple issues found")

    logger.info("=" * 60)
    return checks_failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Soberana Omega - Brawl Stars Bot (PylaAI Enhanced)"
    )
    parser.add_argument(
        '--humanization',
        action='store_true',
        help='Enable humanized delays and movements'
    )
    parser.add_argument(
        '--diagnostic',
        action='store_true',
        help='Enable detailed lobby diagnostics'
    )
    parser.add_argument(
        '--record',
        action='store_true',
        help='Enable gameplay recording for training data collection'
    )
    parser.add_argument("--api", action="store_true", help="Start API server instead of bot")
    parser.add_argument("--check", action="store_true", help="Verify installation")

    # Handle legacy positional arg
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        run_api()
        return

    args = parser.parse_args()

    if args.check:
        success = run_check()
        sys.exit(0 if success else 1)
    elif args.api:
        run_api()
    else:
        run_bot(args)


if __name__ == "__main__":
    main()
