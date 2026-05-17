"""
core/factory.py

Factory for constructing BotOrchestrator with concrete adapters.

This is the ONLY place that knows about concrete implementations.
The orchestrator itself is completely decoupled from:
    - YOLO/Ultralytics
    - ADB/Win32
    - Q-Learning/NeuralPolicy specifics
    - Any particular safety backend

Usage:
    from core.factory import create_orchestrator
    bot = create_orchestrator(install_path=Path("."))
    bot.initialize()
    bot.run()  # blocking

Migration from legacy wrapper.py:
    Instead of directly creating ScreenshotTaker, EmulatorController, etc.,
    wrapper.py should call create_orchestrator() and delegate to it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from core.orchestrator import BotOrchestrator
from core.adapters import (
    DecisionAdapter,
    InputAdapter,
    PersistenceAdapter,
    SafetyAdapter,
    TelemetryAdapter,
    VisionAdapter,
)

logger = logging.getLogger(__name__)


def create_orchestrator(
    install_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    safety_config=None,
    humanization_config=None,
) -> BotOrchestrator:
    """
    Create BotOrchestrator with all concrete adapters wired in.

    This factory handles all the complexity of:
    - Detecting and connecting to emulator
    - Loading YOLO models
    - Initializing safety and humanization systems
    - Setting up telemetry
    - Graceful degradation when components are missing
    """
    install_path = install_path or Path(".")
    config = config or {}
    images_path = install_path / "images"
    models_path = install_path / "models"

    # ------------------------------------------------------------------
    # 1. Safety (first — may veto everything else)
    # ------------------------------------------------------------------
    safety_system = None
    anti_ban = None
    try:
        from safety_system import SafetySystem
        safety_system = SafetySystem(safety_config)
    except Exception as e:
        logger.warning(f"[FACTORY] SafetySystem unavailable: {e}")

    try:
        from core.anti_ban import AntiBanSystem
        anti_ban = AntiBanSystem()
    except Exception as e:
        logger.debug(f"[FACTORY] AntiBanSystem unavailable: {e}")

    safety_adapter = SafetyAdapter(safety_system=safety_system, anti_ban=anti_ban)

    # ------------------------------------------------------------------
    # 2. Input (emulator controller)
    # ------------------------------------------------------------------
    emulator_controller = _try_create_emulator_controller(config)
    input_adapter = InputAdapter(emulator_controller=emulator_controller)

    # Optional: adversarial humanization wrapper
    adv_human_cfg = config.get("adversarial_humanization", {})
    if adv_human_cfg.get("enabled", False):
        try:
            from core.adversarial_humanization import (
                AdversarialHumanizationConfig,
                AdversarialHumanizer,
            )
            from core.adapters.adversarial_input_adapter import AdversarialInputAdapter

            ahc = AdversarialHumanizationConfig(
                enabled=True,
                tap_jitter_sigma=adv_human_cfg.get("tap_jitter_sigma", 1.5),
                miss_tap_probability=adv_human_cfg.get("miss_tap_probability", 0.02),
                delayed_reaction_probability=adv_human_cfg.get("delayed_reaction_probability", 0.05),
                fatigue_decay_per_hour=adv_human_cfg.get("fatigue_decay_per_hour", 0.10),
                rotation_interval_minutes=adv_human_cfg.get("rotation_interval_minutes", 30.0),
            )
            humanizer = AdversarialHumanizer(config=ahc)
            input_adapter = AdversarialInputAdapter(
                primary=input_adapter,
                humanizer=humanizer,
            )
            logger.info("[FACTORY] Adversarial humanization enabled")
        except Exception as e:
            logger.warning("[FACTORY] Adversarial humanization failed: %s", e)

    # ------------------------------------------------------------------
    # 3. Vision (screenshot + YOLO + state detector)
    # ------------------------------------------------------------------
    screenshot_taker = _try_create_screenshot_taker(config, emulator_controller)
    detector = _try_create_detector(models_path)
    state_detector = None
    if images_path.exists():
        try:
            from pylaai_real.unified_state_detector import UnifiedStateDetector
            state_detector = UnifiedStateDetector(images_path=images_path)
        except Exception as e:
            logger.warning(f"[FACTORY] UnifiedStateDetector unavailable: {e}")

    vision_adapter = VisionAdapter(
        screenshot_taker=screenshot_taker,
        detector=detector,
        state_detector=state_detector,
        images_path=images_path,
    )

    # Optional: wrap with VLM fallback for anti-UI-change resilience
    vlm_cfg = config.get("vlm_fallback", {})
    if vlm_cfg.get("enabled", False):
        try:
            from core.vlm_fallback import VLMFallback, VLMConfig
            from core.adapters.vlm_vision_adapter import VLMVisionAdapter

            vlm_config = VLMConfig(
                enabled=True,
                provider=vlm_cfg.get("provider", "openai"),
                model=vlm_cfg.get("model", "gpt-4o-mini"),
                api_key=vlm_cfg.get("api_key") or None,
                api_base=vlm_cfg.get("api_base") or None,
                timeout_seconds=vlm_cfg.get("timeout_seconds", 3.0),
                max_calls_per_minute=vlm_cfg.get("max_calls_per_minute", 10),
                max_calls_per_hour=vlm_cfg.get("max_calls_per_hour", 100),
                cache_ttl_seconds=vlm_cfg.get("cache_ttl_seconds", 30.0),
                fallback_threshold=vlm_cfg.get("fallback_threshold", 0.35),
            )
            vlm = VLMFallback(config=vlm_config)
            vision_adapter = VLMVisionAdapter(primary=vision_adapter, vlm=vlm)
            logger.info("[FACTORY] VLM fallback enabled (provider=%s)", vlm_config.provider)
        except Exception as e:
            logger.warning("[FACTORY] VLM fallback initialization failed: %s", e)

    # ------------------------------------------------------------------
    # 4. Decision (RLBridge)
    # ------------------------------------------------------------------
    decision_adapter = DecisionAdapter()

    # ------------------------------------------------------------------
    # 5. Telemetry
    # ------------------------------------------------------------------
    observability = None
    try:
        from core.observability import ObservabilityCollector
        observability = ObservabilityCollector(
            max_events=2000,
            metrics_dir=install_path / "observability"
        )
    except Exception as e:
        logger.debug(f"[FACTORY] Observability unavailable: {e}")

    telemetry_adapter = TelemetryAdapter(observability=observability)

    # ------------------------------------------------------------------
    # 6. Persistence
    # ------------------------------------------------------------------
    persistence = None
    try:
        from state_persistence import StatePersistence
        persistence = StatePersistence()
    except Exception as e:
        logger.debug(f"[FACTORY] StatePersistence unavailable: {e}")

    persistence_adapter = PersistenceAdapter(persistence=persistence)

    # ------------------------------------------------------------------
    # 7. Build orchestrator
    # ------------------------------------------------------------------
    orchestrator = BotOrchestrator(
        vision=vision_adapter,
        input_=input_adapter,
        decision=decision_adapter,
        safety=safety_adapter,
        telemetry=telemetry_adapter,
        persistence=persistence_adapter,
        config=config,
    )

    logger.info("[FACTORY] BotOrchestrator created with all adapters")
    return orchestrator


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _try_create_emulator_controller(config: Dict[str, Any]) -> Optional[Any]:
    """Attempt to create and connect EmulatorController."""
    try:
        from emulator_controller import EmulatorController, EmulatorConfig
        from emulator_detector import get_emulator_detector

        emu_cfg = config.get("emulator", {})
        emu_type = emu_cfg.get("type", "bluestacks").lower()

        detector = get_emulator_detector()
        emulators = detector.detect_all() if detector else []

        if emulators:
            best = emulators[0]
            cfg = EmulatorConfig(
                name=best.type,
                adb_port=best.adb_port or 5555,
                window_title=best.window_title or best.name,
            )
        else:
            cfg = EmulatorConfig.for_bluestacks() if emu_type == "bluestacks" else EmulatorConfig.for_ldplayer()

        controller = EmulatorController(cfg)
        if controller.connect():
            logger.info(f"[FACTORY] EmulatorController connected: {cfg.name}")
            return controller
        else:
            logger.warning("[FACTORY] EmulatorController connection failed")
    except Exception as e:
        logger.warning(f"[FACTORY] EmulatorController unavailable: {e}")
    return None


def _try_create_screenshot_taker(config: Dict[str, Any], emulator_controller) -> Optional[Any]:
    """Attempt to create ScreenshotTaker."""
    try:
        from pylaai_real.screenshot_taker import ScreenshotTaker

        window_title = None
        if emulator_controller and hasattr(emulator_controller, "config"):
            window_title = emulator_controller.config.window_title

        if not window_title:
            window_title = config.get("emulator", {}).get("window_title", "BlueStacks App Player")

        st = ScreenshotTaker(window_title=window_title)
        if st.find_window():
            logger.info(f"[FACTORY] ScreenshotTaker: window found ({window_title})")
            return st
        else:
            logger.warning(f"[FACTORY] ScreenshotTaker: window not found ({window_title})")
    except Exception as e:
        logger.warning(f"[FACTORY] ScreenshotTaker unavailable: {e}")
    return None


def _try_create_detector(models_path: Path) -> Optional[Any]:
    """Attempt to load YOLO detector."""
    try:
        from pylaai_real.detect import Detect
        model_file = models_path / "brawlstars_yolov8_8class.pt"
        if not model_file.exists():
            model_file = models_path / "brawlstars_yolov8.pt"
        if not model_file.exists():
            model_file = models_path / "yolov8n.pt"
        if model_file.exists():
            det = Detect(model_path=str(model_file))
            logger.info(f"[FACTORY] Detector loaded: {model_file.name}")
            return det
        else:
            logger.warning("[FACTORY] No YOLO model found")
    except Exception as e:
        logger.warning(f"[FACTORY] Detector unavailable: {e}")
    return None
