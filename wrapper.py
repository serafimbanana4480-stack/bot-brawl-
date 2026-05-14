"""
wrapper.py

Wrapper que integra PylaAI real com Safety System, Humanizacao e Error Recovery.
Este e o ponto de entrada principal para usar o PylaAI melhorado.

Fixes Applied:
- Error #10: EmulatorController now integrated (replaces raw ScreenshotTaker)
- Error #12: Model loading prefers trained models (main_info.pt, brawler_id.pt)
- Error #28: Reads from centralized config.json
- Phase 9: Error Recovery, State Recovery, AutoCalibrator, OCR, Debug Visualizer
"""

import time
import json
import threading
import os
import random
import signal
import base64
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

from .pylaai_real.state_finder import StateFinder
from .pylaai_real.state_manager import StateManager
from .pylaai_real.screenshot_taker import ScreenshotTaker
from .pylaai_real.lobby_automator import LobbyAutomator, BrawlerQueue, BrawlerConfig
from .pylaai_real.progress_observer import ProgressObserver
from .pylaai_real.play import PlayLogic
from .pylaai_real.detect import Detect
from .pylaai_real.movement import Movement
from .pylaai_real.screen_automation import ScreenAutomation
from .pylaai_real.unified_state_detector import UnifiedStateDetector
from .diagnostic_overlay import DiagnosticOverlay
from .match_controller import MatchController
from .model_downloader import get_model_downloader
from .emulator_detector import get_emulator_detector

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from .safety_system import SafetySystem, SafetyConfig
from .humanization import HumanizationEngine, HumanizationConfig
from .auto_tuner import AutoTuner, TuningConfig
from .decision.brawler_selector import BrawlerSelector

# Lazy imports para componentes de ML pipeline (evita circular imports)
try:
    from core.reward_bridge import RewardBridge
    HAS_REWARD_BRIDGE = True
except ImportError:
    HAS_REWARD_BRIDGE = False
    RewardBridge = None

try:
    from dataset.collector import GameplayCollector
    HAS_COLLECTOR = True
except ImportError:
    HAS_COLLECTOR = False
    GameplayCollector = None

try:
    from core.observability import ObservabilityCollector, HealthChecker
    HAS_OBSERVABILITY = True
except ImportError:
    HAS_OBSERVABILITY = False
    ObservabilityCollector = None
    HealthChecker = None

try:
    from core.anti_ban import AntiBanSystem, AntiBanConfig
    HAS_ANTIBAN = True
except ImportError:
    HAS_ANTIBAN = False
    AntiBanSystem = None
    AntiBanConfig = None

# Phase 9: Error Recovery, State Recovery, AutoCalibrator, OCR, Debug Visualizer
try:
    from core.error_recovery import ErrorRecoverySystem, ErrorRecoveryIntegration
    HAS_ERROR_RECOVERY = True
except ImportError:
    # Fallback: import direto (core/__init__.py faz chain import que falha)
    try:
        import importlib.util as _ilu
        _er_path = Path(__file__).parent / "core" / "error_recovery.py"
        _spec = _ilu.spec_from_file_location("error_recovery", _er_path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        ErrorRecoverySystem = _mod.ErrorRecoverySystem
        ErrorRecoveryIntegration = _mod.ErrorRecoveryIntegration
        HAS_ERROR_RECOVERY = True
    except Exception:
        HAS_ERROR_RECOVERY = False
        ErrorRecoverySystem = None
        ErrorRecoveryIntegration = None

try:
    from .pylaai_real.state_recovery import StateRecoverySystem
    HAS_STATE_RECOVERY = True
except ImportError:
    HAS_STATE_RECOVERY = False
    StateRecoverySystem = None

try:
    from .pylaai_real.auto_calibrator import AutoCalibrator
    HAS_AUTO_CALIBRATOR = True
except ImportError:
    HAS_AUTO_CALIBRATOR = False
    AutoCalibrator = None

try:
    from .pylaai_real.ocr_state_detector import OCRStateDetector
    HAS_OCR_DETECTOR = True
except ImportError:
    HAS_OCR_DETECTOR = False
    OCRStateDetector = None

try:
    from .pylaai_real.debug_visualizer import DebugVisualizer, DebugIntegration, DebugMode
    HAS_DEBUG_VISUALIZER = True
except ImportError:
    HAS_DEBUG_VISUALIZER = False
    DebugVisualizer = None
    DebugIntegration = None
    DebugMode = None

# Phase 10: Advanced Core Modules
try:
    from .core.central_coordinator import CentralCoordinator
    HAS_CENTRAL_COORDINATOR = True
except ImportError:
    HAS_CENTRAL_COORDINATOR = False
    CentralCoordinator = None

try:
    from .core.world_model import WorldModel
    HAS_WORLD_MODEL = True
except ImportError:
    HAS_WORLD_MODEL = False
    WorldModel = None

try:
    from .core.occupancy_grid import OccupancyGrid
    HAS_OCCUPANCY_GRID = True
except ImportError:
    HAS_OCCUPANCY_GRID = False
    OccupancyGrid = None

try:
    from .core.pressure_map import PressureMap
    HAS_PRESSURE_MAP = True
except ImportError:
    HAS_PRESSURE_MAP = False
    PressureMap = None

try:
    from .core.lobby_fsm import HierarchicalFSM as LobbyFSM
    HAS_LOBBY_FSM = True
except ImportError:
    HAS_LOBBY_FSM = False
    LobbyFSM = None

try:
    from .core.async_pipeline import AsyncPipeline
    HAS_ASYNC_PIPELINE = True
except ImportError:
    HAS_ASYNC_PIPELINE = False
    AsyncPipeline = None

try:
    from .core.adaptive_screenshot import AdaptiveScreenshotCache
    HAS_ADAPTIVE_SCREENSHOT = True
except ImportError:
    HAS_ADAPTIVE_SCREENSHOT = False
    AdaptiveScreenshotCache = None

try:
    from .core.behavioral_profile import BehavioralProfile
    HAS_BEHAVIORAL_PROFILE = True
except ImportError:
    HAS_BEHAVIORAL_PROFILE = False
    BehavioralProfile = None

try:
    from .core.input_optimizer import InputOptimizer
    HAS_INPUT_OPTIMIZER = True
except ImportError:
    HAS_INPUT_OPTIMIZER = False
    InputOptimizer = None

try:
    from .core.replay_analyzer import ReplayFailureAnalyzer as ReplayAnalyzer
    HAS_REPLAY_ANALYZER = True
except ImportError:
    HAS_REPLAY_ANALYZER = False
    ReplayAnalyzer = None

try:
    from .core.tactical_bridge import TacticalBridge
    HAS_TACTICAL_BRIDGE = True
except ImportError:
    HAS_TACTICAL_BRIDGE = False
    TacticalBridge = None

try:
    from .core.cover_system import CoverSystem
    HAS_COVER_SYSTEM = True
except ImportError:
    HAS_COVER_SYSTEM = False
    CoverSystem = None

# Phase 10: Advanced Decision Modules
try:
    from .decision.utility_ai import UtilityAI
    HAS_UTILITY_AI = True
except ImportError:
    HAS_UTILITY_AI = False
    UtilityAI = None

try:
    from .decision.sticky_target import StickyTarget
    HAS_STICKY_TARGET = True
except ImportError:
    HAS_STICKY_TARGET = False
    StickyTarget = None

try:
    from .decision.intent_system import IntentSystem
    HAS_INTENT_SYSTEM = True
except ImportError:
    HAS_INTENT_SYSTEM = False
    IntentSystem = None

try:
    from .decision.enemy_intention import EnemyIntentionPredictor
    HAS_ENEMY_INTENTION = True
except ImportError:
    HAS_ENEMY_INTENTION = False
    EnemyIntentionPredictor = None

try:
    from .decision.meta_awareness import MetaAwareness
    HAS_META_AWARENESS = True
except ImportError:
    HAS_META_AWARENESS = False
    MetaAwareness = None

logger = logging.getLogger(__name__)

# Default install path
_DEFAULT_INSTALL_PATH = Path(os.getenv(
    "PYLAAI_INSTALL_PATH",
    str(Path(__file__).parent / "pylaai_workspace")
))

# Path to bot module root (where config.json, lobby.toml, models/ live)
_BOT_ROOT = Path(__file__).parent


def _load_central_config() -> dict:
    """Load centralized config.json (Fix Error #28)."""
    config_path = _BOT_ROOT / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")
    return {}


class PylaAIEnhanced:
    """
    PylaAI com melhorias:
    - Safety System (limites trofeus, APM, pausas)
    - Humanizacao (delays, curvas Bezier)
    - Fila de brawlers com auto-switch
    - Integracao EmulatorController (ADB + Window control)
    - Modelos treinados para Brawl Stars
    - Logs estruturados
    - Health monitoring and deadlock detection
    - Graceful shutdown with signal handlers
    - Error Recovery System (Phase 9)
    - State Recovery System (Phase 9)
    - AutoCalibrator (Phase 9)
    - OCR State Detector (Phase 9)
    - Debug Visualizer (Phase 9)
    """

    # Health monitor: detect bot deadlocks
    HEARTBEAT_TIMEOUT = 30.0  # seconds without action = potential deadlock
    MAX_UNKNOWN_STATE_DURATION = 60.0  # force reset after 60s in unknown

    def __init__(
        self,
        install_path: Path = _DEFAULT_INSTALL_PATH,
        safety_config: Optional[SafetyConfig] = None,
        humanization_config: Optional[HumanizationConfig] = None,
        diagnostic_mode: Optional[bool] = None,
        enable_recording: bool = False,
        enable_error_recovery: bool = True
    ):
        logger.info("[WRAPPER] Inicializando PylaAIEnhanced")
        self.install_path = install_path
        self.images_path = _BOT_ROOT / "images"
        self.models_path = _BOT_ROOT / "models"
        self.central_config = _load_central_config()
        self.diagnostic_mode = diagnostic_mode if diagnostic_mode is not None else bool(
            self.central_config.get("diagnostic_mode", False) or os.getenv("PYLAAI_DIAGNOSTIC", "0") == "1"
        )
        logger.info(f"[WRAPPER] Diagnostic mode: {self.diagnostic_mode}")

        # Gameplay recording configuration
        self.recording_enabled = enable_recording or bool(self.central_config.get("enable_recording", False))
        logger.info(f"[WRAPPER] Gameplay recording: {'enabled' if self.recording_enabled else 'disabled'}")

        # Auto-retraining configuration
        self.auto_retrain_enabled = bool(self.central_config.get("auto_retrain_enabled", False))
        logger.info(f"[WRAPPER] Auto-retraining: {'enabled' if self.auto_retrain_enabled else 'disabled'}")

        # Safety and humanization systems
        logger.debug("[WRAPPER] Inicializando SafetySystem")
        self.safety = SafetySystem(safety_config)
        logger.debug("[WRAPPER] Inicializando HumanizationEngine")
        self.humanization = HumanizationEngine(humanization_config)

        # Auto-tuner system
        self.auto_tuner: Optional[AutoTuner] = None
        self.auto_tuning_enabled = bool(self.central_config.get("auto_tuning_enabled", False) or os.getenv("PYLAAI_AUTO_TUNING", "0") == "1")
        logger.info(f"[WRAPPER] Auto-tuning enabled: {self.auto_tuning_enabled}")

        # Brawler selector for intelligent brawler switching
        self.brawler_selector: Optional[BrawlerSelector] = None
        self.brawler_selection_enabled = bool(self.central_config.get("brawler_selection_enabled", True))
        if self.brawler_selection_enabled:
            self.brawler_selector = BrawlerSelector()
            logger.info("[WRAPPER] Brawler selector initialized")
        else:
            logger.info("[WRAPPER] Brawler selection disabled")

        # PylaAI real components
        self.screenshot: Optional[ScreenshotTaker] = None
        self.state_finder: Optional[StateFinder] = None
        self.state_manager: Optional[StateManager] = None
        self.lobby: Optional[LobbyAutomator] = None
        self.progress: Optional[ProgressObserver] = None
        self.play_logic: Optional[PlayLogic] = None
        self.detect_main = None
        self.detect_enemies = None
        self.match_controller: Optional[MatchController] = None
        self.diagnostic_overlay: Optional[DiagnosticOverlay] = None
        self.overlay_enabled = bool(self.diagnostic_mode or os.getenv("PYLAAI_OVERLAY", "1") == "1")

        # ML Pipeline components (data collection + rewards)
        self.data_collector = None
        self.reward_bridge = None
        if HAS_COLLECTOR:
            try:
                self.data_collector = GameplayCollector()
                logger.info("[WRAPPER] Data collector inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Data collector indisponível: {e}")
        if HAS_REWARD_BRIDGE:
            try:
                self.reward_bridge = RewardBridge(data_collector=self.data_collector)
                logger.info("[WRAPPER] Reward bridge inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Reward bridge indisponível: {e}")

        # Observability
        self.observability: Optional[ObservabilityCollector] = None
        if HAS_OBSERVABILITY:
            try:
                self.observability = ObservabilityCollector(
                    max_events=2000,
                    metrics_dir=self.install_path / "observability"
                )
                logger.info("[WRAPPER] Observability collector inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Observability indisponível: {e}")

        # Health monitoring
        self._last_action_time = time.time()
        self._health_lock = threading.Lock()
        self._shutdown_hooks: List[callable] = []

        # Anti-ban system
        self.anti_ban: Optional[AntiBanSystem] = None
        if HAS_ANTIBAN:
            try:
                self.anti_ban = AntiBanSystem()
                logger.info("[WRAPPER] Anti-ban system inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Anti-ban system indisponível: {e}")

        # Phase 9: Error Recovery System
        self.error_recovery: Optional[Any] = None
        self.recovery_integration: Optional[Any] = None
        self.enable_error_recovery = enable_error_recovery and HAS_ERROR_RECOVERY
        if self.enable_error_recovery:
            try:
                self.error_recovery = ErrorRecoverySystem(
                    enable_auto_recovery=True,
                    max_recovery_attempts=3,
                    global_circuit_breaker=True
                )
                logger.info("[WRAPPER] Error Recovery System inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Error Recovery System indisponível: {e}")
                self.enable_error_recovery = False

        # Phase 9: State Recovery System (deferred init - needs emulator_controller)
        self.state_recovery: Optional[Any] = None

        # Phase 9: AutoCalibrator
        self.auto_calibrator: Optional[Any] = None
        if HAS_AUTO_CALIBRATOR:
            try:
                self.auto_calibrator = AutoCalibrator(
                    templates_dir=_BOT_ROOT / "images" / "templates"
                )
                logger.info("[WRAPPER] AutoCalibrator inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] AutoCalibrator indisponível: {e}")

        # Phase 9: OCR State Detector
        self.ocr_detector: Optional[Any] = None
        if HAS_OCR_DETECTOR:
            try:
                self.ocr_detector = OCRStateDetector()
                logger.info("[WRAPPER] OCR State Detector inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] OCR State Detector indisponível: {e}")

        # Phase 9: Debug Visualizer
        self.debug_visualizer: Optional[Any] = None
        self.debug_integration: Optional[Any] = None
        self._debug_mode_enabled = bool(
            self.central_config.get("debug_visualizer", False) or
            os.getenv("PYLAAI_DEBUG_VISUAL", "0") == "1"
        )
        if HAS_DEBUG_VISUALIZER and self._debug_mode_enabled:
            try:
                self.debug_visualizer = DebugVisualizer(mode=DebugMode.DETAILED)
                logger.info("[WRAPPER] Debug Visualizer inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Debug Visualizer indisponível: {e}")

        # Phase 10: Advanced Core Modules
        self.central_coordinator: Optional[Any] = None
        if HAS_CENTRAL_COORDINATOR:
            try:
                self.central_coordinator = CentralCoordinator()
                logger.info("[WRAPPER] Central Coordinator inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Central Coordinator indisponível: {e}")

        self.world_model: Optional[Any] = None
        if HAS_WORLD_MODEL:
            try:
                self.world_model = WorldModel()
                logger.info("[WRAPPER] World Model inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] World Model indisponível: {e}")

        self.occupancy_grid: Optional[Any] = None
        if HAS_OCCUPANCY_GRID:
            try:
                self.occupancy_grid = OccupancyGrid()
                logger.info("[WRAPPER] Occupancy Grid inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Occupancy Grid indisponível: {e}")

        self.pressure_map: Optional[Any] = None
        if HAS_PRESSURE_MAP:
            try:
                self.pressure_map = PressureMap()
                logger.info("[WRAPPER] Pressure Map inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Pressure Map indisponível: {e}")

        self.lobby_fsm: Optional[Any] = None
        if HAS_LOBBY_FSM:
            try:
                self.lobby_fsm = LobbyFSM()
                logger.info("[WRAPPER] Lobby FSM inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Lobby FSM indisponível: {e}")

        self.async_pipeline: Optional[Any] = None
        if HAS_ASYNC_PIPELINE:
            try:
                self.async_pipeline = AsyncPipeline()
                logger.info("[WRAPPER] Async Pipeline inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Async Pipeline indisponível: {e}")

        self.adaptive_screenshot_cache: Optional[Any] = None
        if HAS_ADAPTIVE_SCREENSHOT:
            try:
                self.adaptive_screenshot_cache = AdaptiveScreenshotCache()
                logger.info("[WRAPPER] Adaptive Screenshot Cache inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Adaptive Screenshot Cache indisponível: {e}")

        self.behavioral_profile: Optional[Any] = None
        if HAS_BEHAVIORAL_PROFILE:
            try:
                self.behavioral_profile = BehavioralProfile()
                logger.info("[WRAPPER] Behavioral Profile inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Behavioral Profile indisponível: {e}")

        self.input_optimizer: Optional[Any] = None
        if HAS_INPUT_OPTIMIZER:
            try:
                self.input_optimizer = InputOptimizer()
                logger.info("[WRAPPER] Input Optimizer inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Input Optimizer indisponível: {e}")

        self.replay_analyzer: Optional[Any] = None
        if HAS_REPLAY_ANALYZER:
            try:
                self.replay_analyzer = ReplayAnalyzer()
                logger.info("[WRAPPER] Replay Analyzer inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Replay Analyzer indisponível: {e}")

        self.tactical_bridge: Optional[Any] = None
        if HAS_TACTICAL_BRIDGE:
            try:
                self.tactical_bridge = TacticalBridge()
                logger.info("[WRAPPER] Tactical Bridge inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Tactical Bridge indisponível: {e}")

        self.cover_system: Optional[Any] = None
        if HAS_COVER_SYSTEM:
            try:
                self.cover_system = CoverSystem()
                logger.info("[WRAPPER] Cover System inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Cover System indisponível: {e}")

        # Phase 10: Advanced Decision Modules
        self.utility_ai: Optional[Any] = None
        if HAS_UTILITY_AI:
            try:
                self.utility_ai = UtilityAI()
                logger.info("[WRAPPER] Utility AI inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Utility AI indisponível: {e}")

        self.sticky_target: Optional[Any] = None
        if HAS_STICKY_TARGET:
            try:
                self.sticky_target = StickyTarget()
                logger.info("[WRAPPER] Sticky Target inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Sticky Target indisponível: {e}")

        self.intent_system: Optional[Any] = None
        if HAS_INTENT_SYSTEM:
            try:
                self.intent_system = IntentSystem()
                logger.info("[WRAPPER] Intent System inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Intent System indisponível: {e}")

        self.enemy_intention: Optional[Any] = None
        if HAS_ENEMY_INTENTION:
            try:
                self.enemy_intention = EnemyIntentionPredictor()
                logger.info("[WRAPPER] Enemy Intention Predictor inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Enemy Intention Predictor indisponível: {e}")

        self.meta_awareness: Optional[Any] = None
        if HAS_META_AWARENESS:
            try:
                self.meta_awareness = MetaAwareness()
                logger.info("[WRAPPER] Meta Awareness inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Meta Awareness indisponível: {e}")

        # EmulatorController integration (Fix Error #10)
        self.emulator_controller = None

        # Brawler queue - load from config.json if available
        logger.debug("[WRAPPER] Inicializando BrawlerQueue")
        self.brawler_queue = BrawlerQueue()
        queue_config = self.central_config.get("brawler_queue", [])
        if queue_config:
            for bcfg in queue_config:
                self.brawler_queue.add_brawler(BrawlerConfig(
                    name=bcfg.get("name", "colt"),
                    current_trophies=bcfg.get("current_trophies", 0),
                    target_trophies=bcfg.get("target_trophies", 500),
                    target_wins=bcfg.get("target_wins", 10),
                    priority=bcfg.get("priority", 1),
                    enabled=bcfg.get("enabled", True)
                ))
            logger.info(f"[WRAPPER] {len(queue_config)} brawlers carregados do config.json: {[b.get('name') for b in queue_config]}")
        else:
            # Fallback: default brawler
            self.brawler_queue.add_brawler(BrawlerConfig(
                name="colt",
                current_trophies=0,
                target_trophies=400,
                target_wins=10,
                priority=1,
                enabled=True
            ))
            logger.info("[WRAPPER] Brawler 'colt' adicionado à fila (padrão)")

        # State
        self._running_lock = threading.Lock()
        self._running = False
        self.stop_event = threading.Event()
        self.monitor_thread: Optional[threading.Thread] = None
        self.state_thread: Optional[threading.Thread] = None

        # Statistics
        self.session_start = None
        self.matches_played = 0

        # Gameplay Recording (for training data collection)
        self.gameplay_recorder = None
        self.recording_dir = _BOT_ROOT / "recordings"

        # Dashboard server (real-time web dashboard + replay + A/B testing)
        self.dashboard: Optional[Any] = None
        self.dashboard_enabled = bool(self.central_config.get("dashboard_enabled", True))
        if self.dashboard_enabled:
            try:
                from .pylaai_real.dashboard_server import DashboardServer
                self.dashboard = DashboardServer(port=self.central_config.get("dashboard_port", 8765))
                logger.info("[WRAPPER] Dashboard server inicializado (porta %s)",
                             self.central_config.get("dashboard_port", 8765))
            except Exception as e:
                logger.warning(f"[WRAPPER] Dashboard server indisponível: {e}")

        # Auto-retraining system
        self.performance_monitor = None
        self.retrain_orchestrator = None

    @property
    def running(self) -> bool:
        with self._running_lock:
            return self._running

    @running.setter
    def running(self, value: bool):
        with self._running_lock:
            self._running = value

    def _try_init_emulator_controller(self) -> bool:
        """Try to initialize EmulatorController for ADB-based control (Fix Error #10)."""
        logger.debug("[WRAPPER] Tentando inicializar EmulatorController")
        try:
            from .emulator_controller import EmulatorController, EmulatorConfig
            from .emulator_detector import get_emulator_detector

            # First, try to detect emulators automatically
            logger.debug("[WRAPPER] Detectando emuladores automaticamente")
            detector = get_emulator_detector()
            emulators = detector.detect_all()
            logger.info(f"[WRAPPER] {len(emulators)} emulador(es) detectado(s)")
            
            # Find the best emulator (BlueStacks or first connected)
            best_emu = None
            if emulators:
                # Prefer BlueStacks as per user setup
                logger.debug("[WRAPPER] Procurando BlueStacks conectado")
                for e in emulators:
                    if e.type == "bluestacks" and e.connected:
                        best_emu = e
                        logger.debug(f"[WRAPPER] BlueStacks encontrado: {e.name}")
                        break
                
                # Fallback to any connected emulator
                if not best_emu:
                    logger.debug("[WRAPPER] BlueStacks não encontrado, procurando qualquer emulador conectado")
                    for e in emulators:
                        if e.connected:
                            best_emu = e
                            logger.debug(f"[WRAPPER] Emulador conectado encontrado: {e.name}")
                            break
            
            emu_cfg = self.central_config.get("emulator", {})
            emu_type = emu_cfg.get("type", "bluestacks").lower()

            if best_emu:
                logger.info(f"Using automatically detected emulator: {best_emu.name} (ID: {best_emu.adb_id}, Type: {best_emu.type})")
                
                # Extract port from adb_id if it's in localhost:PORT or emulator-PORT format
                adb_port = 5555
                if best_emu.adb_id:
                    if ":" in best_emu.adb_id:
                        try:
                            adb_port = int(best_emu.adb_id.split(":")[-1])
                        except (ValueError, IndexError) as e:
                            logger.debug(f"[WRAPPER] Falha ao extrair porta ADB de {best_emu.adb_id}: {e}")
                    elif "emulator-" in best_emu.adb_id:
                        try:
                            adb_port = int(best_emu.adb_id.split("-")[-1])
                        except (ValueError, IndexError) as e:
                            logger.debug(f"[WRAPPER] Falha ao extrair porta ADB de {best_emu.adb_id}: {e}")
                
                config = EmulatorConfig(
                    name=best_emu.type,
                    adb_port=adb_port,
                    window_title=best_emu.window_title or emu_cfg.get("window_title", "BlueStacks App Player"),
                    resolution=emu_cfg.get("resolution", (1920, 1080))
                )
            else:
                # Fallback to config-based initialization
                if emu_type == "bluestacks":
                    config = EmulatorConfig.for_bluestacks()
                elif emu_type == "ldplayer":
                    config = EmulatorConfig.for_ldplayer()
                else:
                    config = EmulatorConfig(
                        name=emu_type,
                        adb_port=emu_cfg.get("adb_port", 5555),
                        window_title=emu_cfg.get("window_title", "BlueStacks App Player"),
                    )

            # Override from config if present
            if "window_title" in emu_cfg:
                config.window_title = emu_cfg["window_title"]

            logger.debug(f"[WRAPPER] Criando EmulatorController com config: type={config.name}, port={config.adb_port}")
            self.emulator_controller = EmulatorController(config, safety_system=self.safety)
            if self.emulator_controller.connect():
                logger.info(f"[WRAPPER] EmulatorController conectado via ADB (Port: {config.adb_port}, ID: {self.emulator_controller.adb.device_id})")
                return True
            else:
                logger.warning("[WRAPPER] EmulatorController falhou ao conectar, usando fallback ScreenshotTaker")
                self.emulator_controller = None
                return False
        except ImportError as e:
            logger.warning(f"[WRAPPER] EmulatorController não disponível (missing win32gui?): {e}")
            return False
        except Exception as e:
            logger.warning(f"[WRAPPER] EmulatorController init falhou: {e}")
            self.emulator_controller = None
            return False

    def _load_trained_models(self) -> bool:
        """Load Brawl Stars trained models, preferring real trained models (BrawlStarsBot integration)."""
        if not HAS_YOLO:
            logger.error("ultralytics not installed. pip install ultralytics")
            return False

        # Priority: BrawlStarsBot trained model > PylaAI trained > generic models
        trained_models = {
            "brawlstars_yolov8": self.models_path / "brawlstars_yolov8.pt",
            "main_info": self.models_path / "main_info.pt",
            "brawler_id": self.models_path / "brawler_id.pt",
        }
        generic_models = {
            "yolov8n": self.models_path / "yolov8n.pt",
            "yolov8m": self.models_path / "yolov8m.pt",
        }

        model_loaded = False

        # Try BrawlStarsBot trained model first (Player, Bush, Enemy, Cubebox)
        brawlstars_path = trained_models.get("brawlstars_yolov8")
        if brawlstars_path and brawlstars_path.exists():
            for attempt in range(1, 4):
                try:
                    real_model = YOLO(str(brawlstars_path))
                    # Verify model has expected classes
                    expected_classes = {"Player", "Bush", "Enemy", "Cubebox"}
                    actual_classes = set(real_model.names.values()) if hasattr(real_model, "names") else set()
                    if expected_classes.issubset(actual_classes):
                        self.detect_main = Detect(
                            model=real_model,
                            classes={0: "Player", 1: "Bush", 2: "Enemy", 3: "Cubebox"},
                            conf=0.25
                        )
                        self.detect_enemies = self.detect_main
                        logger.info(f"[MODEL] Loaded BrawlStarsBot TRAINED model: {brawlstars_path.name} (classes: {actual_classes}, attempt={attempt})")
                        model_loaded = True
                        break
                    else:
                        missing = expected_classes - actual_classes
                        logger.warning(f"[MODEL] BrawlStarsBot model missing classes {missing}, got: {actual_classes}")
                        # Still use it but log warning
                        self.detect_main = Detect(
                            model=real_model,
                            classes={0: "Player", 1: "Bush", 2: "Enemy", 3: "Cubebox"},
                            conf=0.25
                        )
                        self.detect_enemies = self.detect_main
                        model_loaded = True
                        break
                except Exception as e:
                    logger.error(f"[MODEL] Failed to load BrawlStarsBot model {brawlstars_path} (attempt {attempt}/3): {e}")
                    if attempt < 3:
                        time.sleep(0.5)
                    else:
                        logger.error("[MODEL] BrawlStarsBot model failed after all retries")

        # Try PylaAI trained models if BrawlStarsBot failed
        if not model_loaded:
            main_model_path = trained_models.get("main_info")
            if main_model_path and main_model_path.exists():
                try:
                    real_model = YOLO(str(main_model_path))
                    self.detect_main = Detect(
                        model=real_model,
                        classes={0: "enemy", 1: "player", 2: "teammate"},
                        conf=0.1
                    )
                    self.detect_enemies = self.detect_main
                    logger.info(f"Loaded PylaAI TRAINED model: {main_model_path.name}")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load PylaAI model {main_model_path}: {e}")

        # Fallback to generic models (COCO)
        if not model_loaded:
            for name, path in generic_models.items():
                if path.exists():
                    try:
                        real_model = YOLO(str(path))
                        self.detect_main = Detect(
                            model=real_model,
                            classes={0: "person", 1: "brawler", 2: "object"},
                            conf=0.1
                        )
                        self.detect_enemies = self.detect_main
                        logger.warning(f"Loaded GENERIC model: {path.name} (COCO-80, not trained for Brawl Stars)")
                        model_loaded = True
                        break
                    except Exception as e:
                        logger.error(f"Failed to load generic model {path}: {e}")

        # Last resort: download yolov8n
        if not model_loaded:
            try:
                downloader = get_model_downloader()
                model_path = downloader.get_model_path("yolov8n")
                if not model_path:
                    logger.info("Downloading YOLOv8n as fallback...")
                    res = downloader.download_model("yolov8n")
                    if res.get("success"):
                        model_path = downloader.get_model_path("yolov8n")

                if model_path:
                    real_model = YOLO(str(model_path))
                    self.detect_main = Detect(
                        model=real_model,
                        classes={0: "person", 1: "brawler", 2: "object"},
                        conf=0.1
                    )
                    self.detect_enemies = self.detect_main
                    logger.warning("Using downloaded YOLOv8n with confidence threshold 0.1")
                    model_loaded = True
            except Exception as e:
                logger.error(f"Model download fallback failed: {e}")

        if not model_loaded:
            logger.error("NO models loaded. Vision will not work.")

        return model_loaded

    def _init_gameplay_recorder(self) -> bool:
        """Initialize gameplay recorder for training data collection."""
        if not self.recording_enabled:
            logger.debug("[WRAPPER] Recording disabled, skipping recorder init")
            return False

        try:
            from .automation.gameplay_recorder import GameplayRecorder

            # Get ADB ID from emulator controller
            adb_id = None
            if self.emulator_controller and self.emulator_controller.adb:
                adb_id = self.emulator_controller.adb.device_id

            if not adb_id:
                logger.warning("[WRAPPER] No ADB connection available, gameplay recording disabled")
                return False

            # Create recordings directory
            self.recording_dir.mkdir(parents=True, exist_ok=True)

            # Initialize recorder
            self.gameplay_recorder = GameplayRecorder(
                adb_id=adb_id,
                adb_path=self.emulator_controller.adb.adb_path if self.emulator_controller else None,
                output_dir=self.recording_dir,
                fps=10,  # Lower FPS for recording to save space
                compress=True
            )

            logger.info("[WRAPPER] Gameplay recorder initialized successfully")
            return True

        except ImportError as e:
            logger.warning(f"[WRAPPER] GameplayRecorder not available: {e}")
            return False
        except Exception as e:
            logger.error(f"[WRAPPER] Failed to initialize gameplay recorder: {e}")
            return False

    def _init_auto_retrain_system(self) -> bool:
        """Initialize auto-retraining system for continuous model improvement."""
        if not self.auto_retrain_enabled:
            logger.debug("[WRAPPER] Auto-retrain disabled, skipping init")
            return False

        try:
            from .training.retrain import PerformanceMonitor, RetrainOrchestrator, RetrainTrigger

            # Initialize performance monitor
            logs_dir = _BOT_ROOT / "logs" / "performance"
            self.performance_monitor = PerformanceMonitor(log_dir=logs_dir)

            # Configure triggers
            trigger_config = self.central_config.get("retrain_triggers", {})
            triggers = RetrainTrigger(
                min_matches_before_retrain=trigger_config.get("min_matches", 10),
                win_rate_threshold=trigger_config.get("win_rate_threshold", 0.4),
                min_detection_accuracy=trigger_config.get("min_detection_accuracy", 0.7),
                max_false_positive_rate=trigger_config.get("max_false_positive_rate", 0.2),
                decision_accuracy_threshold=trigger_config.get("decision_accuracy_threshold", 0.6),
                max_days_without_retrain=trigger_config.get("max_days", 7),
                min_new_samples=trigger_config.get("min_new_samples", 500)
            )

            # Initialize retrain orchestrator
            dataset_dir = _BOT_ROOT / "dataset" / "raw"
            models_dir = _BOT_ROOT / "models"
            self.retrain_orchestrator = RetrainOrchestrator(
                monitor=self.performance_monitor,
                trigger_conditions=triggers,
                dataset_dir=dataset_dir,
                models_dir=models_dir
            )

            # Setup callbacks
            self.retrain_orchestrator.on_retrain_start = self._on_retrain_start
            self.retrain_orchestrator.on_retrain_complete = self._on_retrain_complete
            self.retrain_orchestrator.on_retrain_failed = self._on_retrain_failed

            logger.info("[WRAPPER] Auto-retrain system initialized successfully")
            return True

        except ImportError as e:
            logger.warning(f"[WRAPPER] Auto-retrain system not available: {e}")
            return False
        except Exception as e:
            logger.error(f"[WRAPPER] Failed to initialize auto-retrain system: {e}")
            return False

    def setup(self) -> bool:
        """Setup inicial - conecta emulador, carrega modelos, inicializa componentes."""
        try:
            logger.info("Iniciando setup do PylaAI Enhanced...")
            logger.info(f"Diagnostic mode: {'enabled' if self.diagnostic_mode else 'disabled'}")

            # Create directories
            self.install_path.mkdir(parents=True, exist_ok=True)
            self.images_path.mkdir(parents=True, exist_ok=True)
            self.models_path.mkdir(parents=True, exist_ok=True)
            (_BOT_ROOT / "data").mkdir(parents=True, exist_ok=True)
            (_BOT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

            # --- Step 1: Connect to emulator ---
            # Try EmulatorController first (Fix Error #10)
            emulator_ok = self._try_init_emulator_controller()

            if not emulator_ok:
                # Fallback to ScreenshotTaker
                try:
                    detector = get_emulator_detector()
                    emulators = detector.detect_all()
                    chosen_title = None
                    if emulators:
                        preferred = ["bluestacks", "ldplayer", "nox", "memu"]
                        for et in preferred:
                            for e in emulators:
                                if e.type == et:
                                    chosen_title = e.window_title or e.name
                                    break
                            if chosen_title:
                                break
                        if not chosen_title:
                            chosen_title = emulators[0].window_title or emulators[0].name
                        logger.info(f"Selected emulator: {chosen_title}")
                    else:
                        logger.info("No emulators detected; using default title")
                except Exception as e:
                    logger.debug(f"Emulator detection failed: {e}")
                    chosen_title = None

                window_title = chosen_title or self.central_config.get("emulator", {}).get(
                    "window_title", "BlueStacks App Player"
                )
                self.screenshot = ScreenshotTaker(window_title)
                if not self.screenshot.find_window():
                    logger.error("[WRAPPER] Emulador nao encontrado! Abra o emulador primeiro.")
                    return False
                
                # Test screenshot capture to verify emulator is actually responsive
                logger.debug("[WRAPPER] Testando captura de screenshot...")
                
                # Garantir que a janela está ativa antes de capturar screenshot
                if self.emulator_controller:
                    logger.info("[WRAPPER] Garantindo que a janela do emulador está ativa")
                    self.emulator_controller.ensure_window_active()
                    time.sleep(0.2)  # Esperar janela restaurar (reduzido para melhorar FPS)
                
                test_img = self.screenshot.take()
                if test_img is None:
                    logger.error("[WRAPPER] Emulador encontrado mas screenshot falhou. Verifique se o emulador está respondendo.")
                    return False
                logger.info("[WRAPPER] Screenshot funcionando, emulador responsivo")
                logger.info("Emulador conectado via ScreenshotTaker")

            # --- Step 2: Initialize state finder (reads lobby.toml) ---
            self.state_finder = StateFinder(self.images_path)
            self.progress = ProgressObserver()

            # Create ScreenshotTaker if we don't have one yet (needed for Win32 capture)
            if not self.screenshot and self.emulator_controller:
                window_title = self.emulator_controller.config.window_title
                self.screenshot = ScreenshotTaker(window_title)
                if not self.screenshot.find_window():
                    logger.warning("[WRAPPER] ScreenshotTaker window not found, will use ADB screenshots")
                    self.screenshot = None

            # Get window dimensions for dynamic coordinates
            # IMPORTANT: The resolution must match what the screenshot actually captures,
            # not the ADB internal resolution. Win32 captures the window at its display size.
            window_w, window_h = 1920, 1080

            # Priority 1: Get resolution from actual screenshot capture
            if self.screenshot and self.screenshot.window_handle:
                try:
                    import win32gui
                    rect = win32gui.GetWindowRect(self.screenshot.window_handle)
                    window_w = rect[2] - rect[0]
                    window_h = rect[3] - rect[1]
                    logger.info(f"[WRAPPER] Window size (Win32): {window_w}x{window_h}")
                except Exception:
                    pass

            # Priority 2: If no screenshot taker, use ADB resolution
            if not self.screenshot and self.emulator_controller and hasattr(self.emulator_controller, 'config'):
                try:
                    window_w, window_h = self.emulator_controller.config.resolution
                    logger.info(f"[WRAPPER] Window size (ADB): {window_w}x{window_h}")
                except Exception:
                    pass

            # Verify by taking a test screenshot and checking its actual size
            if self.screenshot:
                try:
                    test_img = self.screenshot.take()
                    if test_img is not None:
                        actual_h, actual_w = test_img.shape[:2]
                        if actual_w != window_w or actual_h != window_h:
                            logger.warning(f"[WRAPPER] Screenshot size ({actual_w}x{actual_h}) differs from window size ({window_w}x{window_h}). Using screenshot size.")
                            window_w, window_h = actual_w, actual_h
                except Exception:
                    pass

            logger.info(f"[WRAPPER] Final resolution for coordinates: {window_w}x{window_h}")

            # Phase 9: Initialize State Recovery System (needs emulator_controller)
            if HAS_STATE_RECOVERY and self.emulator_controller:
                try:
                    self.state_recovery = StateRecoverySystem(
                        emulator_controller=self.emulator_controller,
                        max_unknown_duration=30.0,
                        max_loop_duration=15.0,
                        enable_auto_restart=False
                    )
                    logger.info("[WRAPPER] State Recovery System inicializado")
                except Exception as e:
                    logger.warning(f"[WRAPPER] State Recovery System indisponível: {e}")

            # Phase 9: Initialize Error Recovery Integration (needs self)
            if self.enable_error_recovery and self.error_recovery:
                try:
                    self.recovery_integration = ErrorRecoveryIntegration(self)
                    self.recovery_integration.enable()
                    logger.info("[WRAPPER] Error Recovery Integration configurado")
                except Exception as e:
                    logger.warning(f"[WRAPPER] Error Recovery Integration indisponível: {e}")

            # Phase 9: Initialize Debug Integration
            if self.debug_visualizer and HAS_DEBUG_VISUALIZER:
                try:
                    self.debug_integration = DebugIntegration(self)
                    logger.info("[WRAPPER] Debug Integration configurado")
                except Exception as e:
                    logger.warning(f"[WRAPPER] Debug Integration indisponível: {e}")

            # Initialize UnifiedStateDetector (replaces dual ScreenAutomation + StateFinder)
            self.unified_detector = UnifiedStateDetector(
                self.images_path, window_w=window_w, window_h=window_h
            )
            logger.info(f"[WRAPPER] UnifiedStateDetector inicializado ({window_w}x{window_h})")

            self.lobby = LobbyAutomator(
                self.brawler_queue, self.emulator_controller,
                diagnostic_mode=self.diagnostic_mode, play_logic=self.play_logic,
                window_w=window_w, window_h=window_h,
                images_path=str(self.images_path)
            )
            self.match_controller = MatchController(self.install_path)
            self.match_controller.brawler_queue = self.brawler_queue

            # --- Step 2.5: Initialize auto-tuner if enabled ---
            if self.auto_tuning_enabled:
                self.auto_tuner = AutoTuner(self.match_controller)
                logger.info("[WRAPPER] Auto-tuner inicializado")
            else:
                logger.info("[WRAPPER] Auto-tuning desabilitado")

            # --- Step 3: Load YOLO models (Fix Error #12) ---
            self._load_trained_models()

            # --- Step 3.5: Initialize gameplay recorder if enabled ---
            if self.recording_enabled:
                self._init_gameplay_recorder()

            # --- Step 3.6: Initialize auto-retrain system if enabled ---
            if self.auto_retrain_enabled:
                self._init_auto_retrain_system()

            # --- Step 4: Initialize movement, RL engine, and play logic ---
            movement = Movement(
                emulator_controller=self.emulator_controller,
                window_w=window_w,
                window_h=window_h,
            )

            # NOVO: Motor de aprendizagem online (Q-Learning + ELO)
            try:
                from .pylaai_real.rl_engine import OnlineLearner
                self.online_learner = OnlineLearner(
                    reward_bridge=self.reward_bridge,
                    enabled=True,
                )
                logger.info("[WRAPPER] OnlineLearner (RL + ELO) inicializado")
            except Exception as e:
                logger.warning(f"[WRAPPER] OnlineLearner nao disponivel: {e}")
                self.online_learner = None

            self.play_logic = PlayLogic(
                detect_main=self.detect_main,
                detect_enemies=self.detect_enemies,
                movement=movement,
                humanization=self.humanization,
                emulator_controller=self.emulator_controller,  # Pass controller
                rl_engine=self.online_learner.q_learning if self.online_learner else None,
            )

            # --- Step 5: Create state manager ---
            # Ensure we have a compatible screenshot taker (Fix Error #10 integration)
            screenshot_source = self.screenshot
            if self.emulator_controller:
                # Wrap emulator controller to provide .take() method
                # Use Win32 for speed, ADB as backup
                class EmulatorWrapper:
                    def __init__(self, controller, win32_taker):
                        self.controller = controller
                        self.win32_taker = win32_taker
                    def take(self):
                        # Try Win32 first (much faster)
                        if self.win32_taker:
                            img = self.win32_taker.take()
                            if img is not None:
                                return img
                        
                        # Fallback to ADB
                        import numpy as np
                        from PIL import Image
                        import io
                        try:
                            data = self.controller.get_screenshot()
                            if data:
                                return np.array(Image.open(io.BytesIO(data)).convert('RGB'))
                        except Exception:
                            pass
                        return None
                
                # If we don't have a win32_taker yet, create one
                if not self.screenshot:
                    self.screenshot = ScreenshotTaker(self.emulator_controller.config.window_title)
                
                screenshot_source = EmulatorWrapper(self.emulator_controller, self.screenshot)

            # Create screen automation (kept for hint compatibility, but NOT started as thread)
            screen_auto = None
            try:
                if self.screenshot and self.screenshot.window_handle:
                    import win32gui
                    rect = win32gui.GetWindowRect(self.screenshot.window_handle)
                    w = rect[2] - rect[0]
                    h = rect[3] - rect[1]
                    screen_auto = ScreenAutomation(w, h, rect[0], rect[1])
                elif self.emulator_controller:
                    cfg = self.emulator_controller.config
                    screen_auto = ScreenAutomation(
                        cfg.resolution[0], cfg.resolution[1], 0, 0
                    )
            except Exception as e:
                logger.warning(f"ScreenAutomation init failed: {e}")

            self.state_manager = StateManager(
                screenshot_taker=screenshot_source,
                state_finder=self.state_finder,
                lobby_automator=self.lobby,
                progress_observer=self.progress,
                play_logic=self.play_logic,
                match_controller=self.match_controller,
                emulator_controller=self.emulator_controller,
                screen_automation=screen_auto,  # Kept for hints only
                movement=movement,
                diagnostic_mode=self.diagnostic_mode,
                reward_bridge=self.reward_bridge,
                data_collector=self.data_collector,
                brawler_selector=self.brawler_selector,
                observability=self.observability,
                unified_state_detector=self.unified_detector,  # NEW: unified detector
                rl_engine=self.online_learner,  # NEW: RL online
            )

            # Premium: Connect dashboard bridge to state_manager for match tracking
            if self.dashboard:
                self.state_manager._dashboard_bridge = self.dashboard.bridge
                # Also set wrapper ref on dashboard for bot control
                self.dashboard.set_wrapper(self)

            # Connect lobby automator to unified detector for visual verification
            if self.lobby:
                self.lobby.set_state_detector(self.unified_detector)
                self.lobby.set_screenshot_func(screenshot_source.take)
                self.lobby.set_screen_automation(screen_auto)
                self.lobby.set_diagnostic_mode(self.diagnostic_mode)

            # IMPORTANT: ScreenAutomation thread is NOT started anymore.
            # The UnifiedStateDetector handles all detection in the main StateManager cycle.
            # This eliminates the dual-click conflict that caused "random clicking".
            if screen_auto:
                logger.info("ScreenAutomation created for hints only (NOT started as thread)")
            else:
                logger.info("ScreenAutomation not available, using UnifiedStateDetector only")

            logger.info("Setup completo!")
            return True

        except Exception as e:
            logger.error(f"Erro no setup: {e}", exc_info=True)
            return False

    def start(self) -> bool:
        """Inicia o bot"""
        logger.info("[WRAPPER] Chamando start()")
        if self.running:
            logger.warning("[WRAPPER] Bot já está em execução!")
            return False

        # Install signal handlers for graceful shutdown
        self._install_signal_handlers()

        logger.info("[WRAPPER] Iniciando bot PylaAI Enhanced...")

        # Start dashboard server
        if self.dashboard:
            try:
                self.dashboard.start(daemon=True)
                logger.info("[WRAPPER] Dashboard server iniciado — aceda em http://localhost:%s", self.dashboard.port)
            except Exception as e:
                logger.warning(f"[WRAPPER] Falha ao iniciar dashboard: {e}")

        # Anti-ban schedule check
        if self.anti_ban and not self.anti_ban.should_start_match():
            logger.warning("[WRAPPER] Anti-ban: início bloqueado por schedule/padrão")
            return False

        # Safety check before starting
        logger.debug("[WRAPPER] Verificando limites de segurança")
        trophy_status = self.safety.check_trophy_limit(
            self._get_current_trophies()
        )
        if not trophy_status["can_play"]:
            logger.error(f"[WRAPPER] Safety block: {trophy_status['message']}")
            return False

        if trophy_status["warning"]:
            logger.warning(f"[WRAPPER] Safety warning: {trophy_status['message']}")

        # Start safety session
        logger.debug("[WRAPPER] Iniciando sessão de segurança")
        self.safety.start_session()
        self.session_start = time.time()

        # Initialize current_brawler from queue for dashboard
        if self.state_manager and self.brawler_queue:
            current = self.brawler_queue.get_current()
            if current:
                self.state_manager.current_brawler = current.name

        # Start threads
        logger.debug("[WRAPPER] Iniciando threads")
        self.running = True
        self.stop_event.clear()

        if self.overlay_enabled and self.diagnostic_overlay is None:
            logger.debug("[WRAPPER] Iniciando diagnostic overlay")
            try:
                self.diagnostic_overlay = DiagnosticOverlay(self.get_status)
                self.diagnostic_overlay.start()
                logger.info("[WRAPPER] Diagnostic overlay iniciado")
            except Exception as e:
                logger.warning(f"[WRAPPER] Diagnostic overlay indisponível: {e}")
                self.diagnostic_overlay = None

        # State manager thread
        logger.debug("[WRAPPER] Iniciando thread state-manager")
        self.state_thread = threading.Thread(
            target=self.state_manager.run,
            daemon=True,
            name="state-manager"
        )
        self.state_thread.start()

        # Monitor thread (safety)
        logger.debug("[WRAPPER] Iniciando thread safety-monitor")
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="safety-monitor"
        )
        self.monitor_thread.start()

        # Start gameplay recording if enabled
        if self.recording_enabled and self.gameplay_recorder:
            logger.debug("[WRAPPER] Iniciando gameplay recording")
            self.start_recording()

        # Start data collection if available
        if self.data_collector is not None:
            try:
                self.data_collector.start_episode()
                logger.info("[WRAPPER] Data collector sessão iniciada")
            except Exception as e:
                logger.warning(f"[WRAPPER] Falha ao iniciar data_collector: {e}")

        logger.info("[WRAPPER] PylaAI Enhanced iniciado com sucesso!")
        return True

    def _install_signal_handlers(self):
        """Install signal handlers for graceful shutdown on Ctrl+C."""
        try:
            original_sigint = signal.getsignal(signal.SIGINT)
            original_sigterm = signal.getsignal(signal.SIGTERM)

            def _shutdown_handler(signum, frame):
                sig_name = signal.Signals(signum).name
                logger.info(f"[WRAPPER] Received {sig_name}, initiating graceful shutdown...")
                self.stop()

            signal.signal(signal.SIGINT, _shutdown_handler)
            signal.signal(signal.SIGTERM, _shutdown_handler)
            logger.debug("[WRAPPER] Signal handlers installed (SIGINT, SIGTERM)")
        except (OSError, ValueError) as e:
            # Signal handlers can only be set in main thread
            logger.debug(f"[WRAPPER] Cannot install signal handlers: {e}")

    def record_heartbeat(self):
        """Record that the bot is alive and performing actions."""
        with self._health_lock:
            self._last_action_time = time.time()

    def check_health(self) -> Dict:
        """
        Check bot health status.
        Returns dict with health info and any detected issues.
        """
        with self._health_lock:
            time_since_action = time.time() - self._last_action_time

        health = {
            "healthy": True,
            "time_since_last_action": time_since_action,
            "running": self.running,
            "current_state": self.state_manager.current_state if self.state_manager else "none",
            "issues": [],
        }

        # Check for deadlock (no action for HEARTBEAT_TIMEOUT seconds)
        if self.running and time_since_action > self.HEARTBEAT_TIMEOUT:
            health["healthy"] = False
            health["issues"].append(f"DEADLOCK: No action for {time_since_action:.0f}s")
            logger.error(f"[HEALTH] Deadlock detected! No action for {time_since_action:.0f}s")

        # Check for stuck in unknown state
        if (self.state_manager and
            self.state_manager.current_state == 'unknown' and
            self.state_manager.unknown_since):
            unknown_elapsed = time.time() - self.state_manager.unknown_since
            if unknown_elapsed > self.MAX_UNKNOWN_STATE_DURATION:
                health["healthy"] = False
                health["issues"].append(f"STUCK_UNKNOWN: In unknown state for {unknown_elapsed:.0f}s")
                logger.error(f"[HEALTH] Stuck in unknown state for {unknown_elapsed:.0f}s, forcing reset")
                self.state_manager.current_state = 'lobby'
                self.state_manager.unknown_since = None

        return health

    def register_shutdown_hook(self, hook: callable):
        """Register a function to be called during graceful shutdown."""
        self._shutdown_hooks.append(hook)

    def _cleanup_resources(self):
        """Systematically stop all subsystems in dependency order.

        Order matters: stop active components first, then data sinks,
        then UI/monitoring, finally background threads.
        """
        # --- Phase 1: Stop active gameplay components ---
        if self.state_manager:
            logger.debug("[CLEANUP] Parando state_manager")
            try:
                self.state_manager.stop()
            except Exception as e:
                logger.warning(f"[CLEANUP] Falha ao parar state_manager: {e}")

        if self.state_manager and self.state_manager.screen_automation:
            logger.debug("[CLEANUP] Parando screen_automation")
            try:
                self.state_manager.screen_automation.stop()
            except Exception as e:
                logger.debug(f"[CLEANUP] Falha ao parar screen_automation: {e}")

        if self.play_logic:
            logger.debug("[CLEANUP] Parando play_logic")
            try:
                if hasattr(self.play_logic, 'stop'):
                    self.play_logic.stop()
            except Exception as e:
                logger.debug(f"[CLEANUP] Falha ao parar play_logic: {e}")

        # --- Phase 2: Flush data sinks ---
        if self.data_collector is not None:
            try:
                self.data_collector.flush()
                logger.info("[CLEANUP] Data collector flushed")
            except Exception as e:
                logger.warning(f"[CLEANUP] Falha ao flush data_collector: {e}")

        if self.online_learner is not None:
            try:
                self.online_learner.save()
                stats = self.online_learner.get_stats()
                logger.info(f"[CLEANUP] OnlineLearner salvo: {stats}")
            except Exception as e:
                logger.warning(f"[CLEANUP] Falha ao salvar OnlineLearner: {e}")

        if self.reward_bridge is not None:
            try:
                summary = self.reward_bridge.get_session_summary()
                logger.info(f"[CLEANUP] Reward session summary: {summary}")
                self.reward_bridge.reset()
            except Exception as e:
                logger.warning(f"[CLEANUP] Falha ao reset reward_bridge: {e}")

        # Save brawler selector stats
        if self.brawler_selector is not None:
            try:
                self.brawler_selector._save_stats()
                logger.info("[CLEANUP] Brawler selector stats saved")
            except Exception as e:
                logger.warning(f"[CLEANUP] Falha ao salvar brawler_selector: {e}")

        # Export match history on shutdown
        if self.match_controller is not None:
            try:
                self.match_controller.history.save()
                logger.info("[CLEANUP] Match history saved")
            except Exception as e:
                logger.warning(f"[CLEANUP] Falha ao salvar match_history: {e}")

        # --- Phase 3: Stop UI / monitoring components ---
        if self.diagnostic_overlay:
            logger.debug("[CLEANUP] Parando diagnostic overlay")
            try:
                self.diagnostic_overlay.stop()
            except Exception as e:
                logger.debug(f"[CLEANUP] Falha ao parar diagnostic_overlay: {e}")

        if self.debug_visualizer:
            try:
                self.debug_visualizer.stop()
                logger.info("[CLEANUP] Debug Visualizer parado")
            except Exception as e:
                logger.debug(f"[CLEANUP] Falha ao parar debug visualizer: {e}")

        if self.dashboard:
            try:
                self.dashboard.stop()
                logger.info("[CLEANUP] Dashboard server parado")
            except Exception as e:
                logger.debug(f"[CLEANUP] Falha ao parar dashboard: {e}")

        # --- Phase 4: Stop observability / anti-ban ---
        if self.observability:
            try:
                if hasattr(self.observability, 'stop'):
                    self.observability.stop()
                logger.info("[CLEANUP] Observability collector parado")
            except Exception as e:
                logger.debug(f"[CLEANUP] Falha ao parar observability: {e}")

        if self.anti_ban:
            try:
                if hasattr(self.anti_ban, 'stop'):
                    self.anti_ban.stop()
                logger.info("[CLEANUP] Anti-ban system parado")
            except Exception as e:
                logger.debug(f"[CLEANUP] Falha ao parar anti_ban: {e}")

        # --- Phase 5: Stop state recovery ---
        if self.state_recovery:
            try:
                self.state_recovery.cancel_recovery()
                logger.info("[CLEANUP] State Recovery cancelado")
            except Exception as e:
                logger.debug(f"[CLEANUP] Falha ao cancelar state recovery: {e}")

        # --- Phase 6: Stop recording ---
        if self.recording_enabled and self.gameplay_recorder:
            logger.debug("[CLEANUP] Parando gameplay recording")
            try:
                self.stop_recording()
            except Exception as e:
                logger.debug(f"[CLEANUP] Falha ao parar recording: {e}")

        # --- Phase 7: Anti-detection window randomization ---
        if self.emulator_controller:
            try:
                self.emulator_controller.randomize_window_periodically(interval=0)
            except Exception as e:
                logger.debug(f"[CLEANUP] Falha ao randomizar janela: {e}")

    def stop(self):
        """Para o bot com graceful shutdown completo."""
        logger.info("[WRAPPER] Chamando stop()")
        if not self.running:
            logger.debug("[WRAPPER] Bot não está em execução")
            return False

        logger.info("[WRAPPER] Parando PylaAI Enhanced...")
        self.running = False
        self.stop_event.set()

        # Run user-registered shutdown hooks first
        for hook in self._shutdown_hooks:
            try:
                hook()
            except Exception as e:
                logger.warning(f"[WRAPPER] Shutdown hook failed: {e}")

        # Systematic resource cleanup
        self._cleanup_resources()

        # Wait for threads with reasonable timeouts
        logger.debug("[WRAPPER] Aguardando threads terminarem")
        if self.state_thread and self.state_thread.is_alive():
            self.state_thread.join(timeout=5)
            if self.state_thread.is_alive():
                logger.warning("[WRAPPER] Thread state-manager não terminou em 5s")
            else:
                logger.debug("[WRAPPER] Thread state-manager terminou")
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
            if self.monitor_thread.is_alive():
                logger.warning("[WRAPPER] Thread safety-monitor não terminou em 5s")
            else:
                logger.debug("[WRAPPER] Thread safety-monitor terminou")

        logger.info("[WRAPPER] PylaAI Enhanced parado com sucesso!")
        return True

    def start_recording(self) -> bool:
        """Start gameplay recording for training data collection."""
        if not self.gameplay_recorder:
            logger.warning("[WRAPPER] Gameplay recorder not initialized")
            return False

        try:
            logger.info("[WRAPPER] Starting gameplay recording...")
            self.gameplay_recorder.start()
            logger.info("[WRAPPER] Gameplay recording started")
            return True
        except Exception as e:
            logger.error(f"[WRAPPER] Failed to start recording: {e}")
            return False

    def stop_recording(self) -> bool:
        """Stop gameplay recording and save collected data."""
        if not self.gameplay_recorder:
            logger.warning("[WRAPPER] Gameplay recorder not initialized")
            return False

        try:
            logger.info("[WRAPPER] Stopping gameplay recording...")
            self.gameplay_recorder.stop()
            logger.info("[WRAPPER] Gameplay recording stopped and saved")
            return True
        except Exception as e:
            logger.error(f"[WRAPPER] Failed to stop recording: {e}")
            return False

    # =================================================================
    # DASHBOARD CONTROL METHODS (Phase 1: Controle Total)
    # =================================================================

    def pause(self) -> bool:
        """Pausa o bot sem matar threads. Mantem dashboard ativa."""
        if not self.running:
            logger.warning("[WRAPPER] Bot nao esta em execucao, nada para pausar")
            return False
        if getattr(self, '_paused', False):
            logger.debug("[WRAPPER] Bot ja esta pausado")
            return True
        self._paused = True
        if self.state_manager:
            self.state_manager.pause()
        logger.info("[WRAPPER] Bot PAUSADO — threads mantidas vivas")
        return True

    def resume(self) -> bool:
        """Retoma o bot apos pausa."""
        if not getattr(self, '_paused', False):
            logger.debug("[WRAPPER] Bot nao esta pausado")
            return True
        self._paused = False
        if self.state_manager:
            self.state_manager.resume()
        logger.info("[WRAPPER] Bot RETOMADO")
        return True

    def update_config(self, key: str, value) -> bool:
        """Atualiza uma chave no config.json e central_config em runtime."""
        try:
            self.central_config[key] = value
            config_path = _BOT_ROOT / "config.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    full_config = json.load(f)
                full_config[key] = value
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(full_config, f, indent=2, ensure_ascii=False)
            logger.info(f"[WRAPPER] Config atualizado: {key}={value}")
            return True
        except Exception as e:
            logger.error(f"[WRAPPER] Falha ao atualizar config {key}: {e}")
            return False

    def set_brawler(self, name: str) -> bool:
        """Forca selecao de um brawler especifico."""
        if not self.brawler_queue:
            logger.warning("[WRAPPER] Brawler queue nao inicializada")
            return False
        try:
            # Procurar brawler na fila
            for i, b in enumerate(self.brawler_queue.brawlers):
                if b.name.lower() == name.lower():
                    self.brawler_queue.current_index = i
                    if self.state_manager:
                        self.state_manager.current_brawler = b.name
                    logger.info(f"[WRAPPER] Brawler forçado: {b.name}")
                    return True
            # Se nao encontrou, adiciona com defaults
            self.add_brawler_to_queue(name)
            self.brawler_queue.current_index = len(self.brawler_queue.brawlers) - 1
            if self.state_manager:
                self.state_manager.current_brawler = name
            logger.info(f"[WRAPPER] Brawler adicionado e selecionado: {name}")
            return True
        except Exception as e:
            logger.error(f"[WRAPPER] Falha ao set brawler {name}: {e}")
            return False

    def update_queue(self, queue_data: List[Dict]) -> bool:
        """Substitui a fila de brawlers completamente."""
        if not self.brawler_queue:
            logger.warning("[WRAPPER] Brawler queue nao inicializada")
            return False
        try:
            from .pylaai_real.lobby_automator import BrawlerConfig
            self.brawler_queue.brawlers = []
            self.brawler_queue.current_index = 0
            for item in queue_data:
                cfg = BrawlerConfig(
                    name=item.get("name", "colt"),
                    current_trophies=item.get("current_trophies", 0),
                    target_trophies=item.get("target_trophies", 350),
                    target_wins=item.get("target_wins", 10),
                    priority=item.get("priority", 1),
                    enabled=item.get("enabled", True),
                )
                self.brawler_queue.add_brawler(cfg)
            logger.info(f"[WRAPPER] Fila de brawlers atualizada: {len(queue_data)} brawlers")
            return True
        except Exception as e:
            logger.error(f"[WRAPPER] Falha ao atualizar fila: {e}")
            return False

    def toggle_system(self, system_name: str, enabled: bool) -> bool:
        """Liga/desliga sistemas em runtime."""
        try:
            if system_name == "rl_engine":
                if self.online_learner:
                    self.online_learner.enabled = enabled
                    logger.info(f"[WRAPPER] RL engine {'ativado' if enabled else 'desativado'}")
                else:
                    logger.warning("[WRAPPER] RL engine nao inicializado")
                    return False
            elif system_name == "humanization":
                if self.humanization:
                    self.humanization.enabled = enabled
                    logger.info(f"[WRAPPER] Humanizacao {'ativada' if enabled else 'desativada'}")
                else:
                    logger.warning("[WRAPPER] Humanizacao nao inicializada")
                    return False
            elif system_name == "anti_ban":
                if self.anti_ban:
                    self.anti_ban.enabled = enabled
                    logger.info(f"[WRAPPER] Anti-ban {'ativado' if enabled else 'desativado'}")
                else:
                    logger.warning("[WRAPPER] Anti-ban nao inicializado")
                    return False
            elif system_name == "error_recovery":
                if self.error_recovery:
                    self.error_recovery.enabled = enabled
                    logger.info(f"[WRAPPER] Error recovery {'ativado' if enabled else 'desativado'}")
                else:
                    logger.warning("[WRAPPER] Error recovery nao inicializado")
                    return False
            elif system_name == "recording":
                if enabled and not getattr(self, 'recording_enabled', False):
                    self.recording_enabled = True
                    self.start_recording()
                elif not enabled and getattr(self, 'recording_enabled', False):
                    self.recording_enabled = False
                    self.stop_recording()
                logger.info(f"[WRAPPER] Recording {'ativado' if enabled else 'desativado'}")
            elif system_name == "auto_tuner":
                self.auto_tuning_enabled = enabled
                logger.info(f"[WRAPPER] Auto-tuning {'ativado' if enabled else 'desativado'}")
            else:
                logger.warning(f"[WRAPPER] Sistema desconhecido: {system_name}")
                return False
            return True
        except Exception as e:
            logger.error(f"[WRAPPER] Falha ao toggle {system_name}: {e}")
            return False

    def execute_action(self, action_name: str, **kwargs) -> bool:
        """Executa uma acao manual no bot."""
        try:
            if action_name == "force_goto_lobby":
                if self.state_manager:
                    self.state_manager.current_state = 'lobby'
                    logger.info("[WRAPPER] Acao manual: forçar lobby")
                return True
            elif action_name == "force_click_play":
                if self.lobby and hasattr(self.lobby, '_click'):
                    # Usar coordenadas dinamicas se disponiveis
                    w, h = 1920, 1080
                    if self.play_logic and hasattr(self.play_logic.movement, 'window_w'):
                        w = self.play_logic.movement.window_w
                        h = self.play_logic.movement.window_h
                    play_x, play_y = round(w * 0.9419), round(h * 0.8949)
                    self.lobby._click(play_x, play_y)
                    logger.info("[WRAPPER] Acao manual: clicar Play")
                return True
            elif action_name == "force_attack":
                if self.play_logic and hasattr(self.play_logic, '_execute_attack'):
                    # Atacar na direcao do centro do ecra
                    w, h = 1920, 1080
                    if self.play_logic.movement:
                        w = getattr(self.play_logic.movement, 'window_w', 1920)
                        h = getattr(self.play_logic.movement, 'window_h', 1080)
                    target_x, target_y = round(w * 0.6), round(h * 0.5)
                    self.play_logic._execute_attack((target_x, target_y))
                    logger.info("[WRAPPER] Acao manual: ataque forçado")
                return True
            elif action_name == "force_super":
                if self.emulator_controller:
                    # Super = tap no centro do ecra (simplificado)
                    w, h = 1920, 1080
                    if self.play_logic and self.play_logic.movement:
                        w = getattr(self.play_logic.movement, 'window_w', 1920)
                        h = getattr(self.play_logic.movement, 'window_h', 1080)
                    self.emulator_controller.tap_scaled(round(w/2), round(h/2))
                    logger.info("[WRAPPER] Acao manual: super usado")
                return True
            elif action_name == "force_collect_cube":
                if self.play_logic and hasattr(self.play_logic, '_collect_power_cubes'):
                    self.play_logic._collect_power_cubes([])
                    logger.info("[WRAPPER] Acao manual: coletar power cubes")
                return True
            elif action_name == "screenshot":
                if self.screenshot:
                    img = self.screenshot.take()
                    logger.info(f"[WRAPPER] Acao manual: screenshot capturado {img.shape if img is not None else 'None'}")
                return True
            elif action_name == "back_press":
                if self.emulator_controller:
                    self.emulator_controller.keyevent(4)  # Android BACK
                    logger.info("[WRAPPER] Acao manual: BACK press")
                return True
            else:
                logger.warning(f"[WRAPPER] Acao desconhecida: {action_name}")
                return False
        except Exception as e:
            logger.error(f"[WRAPPER] Falha na acao {action_name}: {e}")
            return False

    def get_system_status(self) -> Dict:
        """Retorna status detalhado de todos os sistemas para a dashboard."""
        status = {
            "paused": getattr(self, '_paused', False),
            "running": self.running,
            "systems": {
                "rl_engine": {
                    "enabled": getattr(self.online_learner, 'enabled', False) if self.online_learner else False,
                    "available": self.online_learner is not None,
                },
                "humanization": {
                    "enabled": getattr(self.humanization, 'enabled', True) if self.humanization else False,
                    "available": self.humanization is not None,
                },
                "anti_ban": {
                    "enabled": getattr(self.anti_ban, 'enabled', True) if self.anti_ban else False,
                    "available": self.anti_ban is not None,
                },
                "error_recovery": {
                    "enabled": getattr(self.error_recovery, 'enabled', True) if self.error_recovery else False,
                    "available": self.error_recovery is not None,
                },
                "recording": {
                    "enabled": getattr(self, 'recording_enabled', False),
                    "available": self.gameplay_recorder is not None,
                },
                "auto_tuner": {
                    "enabled": getattr(self, 'auto_tuning_enabled', False),
                    "available": self.auto_tuner is not None,
                },
                "data_collector": {
                    "enabled": self.data_collector is not None,
                    "available": self.data_collector is not None,
                },
            }
        }
        return status

    def _on_retrain_start(self):
        """Callback when retraining starts."""
        logger.info("[WRAPPER] Retraining started - pausing bot operations")
        # Pause bot operations during retraining
        if self.state_manager:
            self.state_manager.pause()

    def _on_retrain_complete(self, new_model_path: Path):
        """Callback when retraining completes successfully."""
        logger.info(f"[WRAPPER] Retraining completed - new model: {new_model_path}")
        # Reload the new model
        self._load_trained_models()
        # Resume bot operations
        if self.state_manager:
            self.state_manager.resume()

    def _on_retrain_failed(self, error: str):
        """Callback when retraining fails."""
        logger.error(f"[WRAPPER] Retraining failed: {error}")
        # Resume bot operations even if retraining failed
        if self.state_manager:
            self.state_manager.resume()

    def record_performance_metric(self, metric_type: str, **kwargs):
        """Record performance metrics for auto-retraining system."""
        if not self.performance_monitor:
            return

        try:
            if metric_type == "kill":
                self.performance_monitor.record_kill()
            elif metric_type == "death":
                self.performance_monitor.record_death()
            elif metric_type == "damage":
                self.performance_monitor.record_damage(
                    dealt=kwargs.get("dealt", 0),
                    taken=kwargs.get("taken", 0)
                )
            elif metric_type == "match_result":
                self.performance_monitor.record_match_result(
                    won=kwargs.get("won", False),
                    survival_time=kwargs.get("survival_time", 0)
                )
            elif metric_type == "decision":
                self.performance_monitor.record_decision(
                    was_good=kwargs.get("good", True)
                )
            elif metric_type == "detection":
                self.performance_monitor.update_detection_metrics(
                    accuracy=kwargs.get("accuracy", 0.0),
                    tracking=kwargs.get("tracking", 0.0),
                    false_positive=kwargs.get("false_positive", 0.0)
                )
        except Exception as e:
            logger.error(f"[WRAPPER] Failed to record performance metric: {e}")

    def check_retrain_trigger(self) -> tuple[bool, str]:
        """Check if retraining should be triggered based on performance."""
        if not self.retrain_orchestrator:
            return False, "Auto-retrain not enabled"

        try:
            return self.retrain_orchestrator.should_retrain()
        except Exception as e:
            logger.error(f"[WRAPPER] Failed to check retrain trigger: {e}")
            return False, "Error checking trigger"

    def trigger_retrain(self) -> bool:
        """Manually trigger retraining process."""
        if not self.retrain_orchestrator:
            logger.warning("[WRAPPER] Auto-retrain not enabled")
            return False

        try:
            return self.retrain_orchestrator.trigger_retrain()
        except Exception as e:
            logger.error(f"[WRAPPER] Failed to trigger retrain: {e}")
            return False

    def _monitor_loop(self):
        """Loop de monitorizacao (safety + error recovery + state recovery)"""
        logger.info("[WRAPPER] Monitor loop started")
        while not self.stop_event.is_set() and self.running:
            cycle_start = time.time()
            try:
                # Phase 9: Execute state recovery if active
                if self.state_recovery and self.state_recovery.is_recovering():
                    self.state_recovery.execute_recovery_step()

                # Phase 9: Update debug visualizer
                if self.debug_integration:
                    try:
                        self.debug_integration.update()
                    except Exception:
                        pass

                safety_check = self.safety.record_action()

                if not safety_check["safe"]:
                    logger.warning("Problema de seguranca detetado!")
                    if self.safety.emergency_stop_triggered:
                        logger.error("Emergency stop ativado!")
                        self.stop()
                        break

                if safety_check.get("should_delay"):
                    delay = safety_check["delay"]
                    logger.debug(f"Safety delay: {delay:.2f}s")
                    time.sleep(delay)

                if self.safety.should_take_break():
                    break_duration = self.safety.get_break_duration()
                    logger.info(f"Pausa obrigatoria: {break_duration/60:.1f} min")
                    self._take_break(break_duration)

                # Periodic window randomization (anti-detection)
                if self.emulator_controller:
                    self.emulator_controller.randomize_window_periodically()

                # Auto-detect window resolution changes and update components
                if self.screenshot and hasattr(self.screenshot, 'window_handle') and self.screenshot.window_handle:
                    try:
                        import win32gui
                        rect = win32gui.GetWindowRect(self.screenshot.window_handle)
                        new_w = rect[2] - rect[0]
                        new_h = rect[3] - rect[1]
                        if hasattr(self, '_last_window_w'):
                            if new_w != self._last_window_w or new_h != self._last_window_h:
                                logger.info(f"[WRAPPER] Window resized: {new_w}x{new_h}")
                                self._update_all_coordinates(new_w, new_h)
                        self._last_window_w = new_w
                        self._last_window_h = new_h
                    except Exception:
                        pass

                # Phase 10: World Model update (persistent spatial memory)
                if self.world_model and self.state_manager and self.state_manager.current_state == 'in_game':
                    try:
                        if self.play_logic and hasattr(self.play_logic, 'last_combat_snapshot'):
                            snap = self.play_logic.last_combat_snapshot
                            if snap and snap.get('enemies'):
                                enemies = snap['enemies']
                                player = snap.get('player')
                                if player and enemies:
                                    self.world_model.update_enemies(enemies, player)
                    except Exception:
                        pass

                # Phase 10: Pressure Map update (enemy influence zones)
                if self.pressure_map and self.state_manager and self.state_manager.current_state == 'in_game':
                    try:
                        if self.play_logic and hasattr(self.play_logic, 'last_combat_snapshot'):
                            snap = self.play_logic.last_combat_snapshot
                            if snap and snap.get('enemies'):
                                for enemy in snap['enemies']:
                                    cx = (enemy[0] + enemy[2]) // 2
                                    cy = (enemy[1] + enemy[3]) // 2
                                    self.pressure_map.add_pressure(cx, cy, intensity=1.0)
                    except Exception:
                        pass

                # Phase 10: Behavioral Profile update
                if self.behavioral_profile and self.state_manager:
                    try:
                        current_state = self.state_manager.current_state
                        self.behavioral_profile.record_state(current_state)
                    except Exception:
                        pass

                # Observability cycle time
                if self.observability:
                    cycle_duration = time.time() - cycle_start
                    self.observability.record_cycle_time(cycle_duration)
                    cycle_start = time.time()
                    # Update current state from state_manager
                    if self.state_manager:
                        self.observability.update_state(self.state_manager.current_state)

                    # Phase 9: Feed state to State Recovery System
                    if self.state_recovery and self.state_manager:
                        try:
                            # Get confidence from unified_detector if available
                            confidence = 0.8  # Default
                            if self.unified_detector and hasattr(self.unified_detector, '_last_confidence'):
                                confidence = self.unified_detector._last_confidence
                            self.state_recovery.update_state(
                                self.state_manager.current_state, confidence
                            )
                        except Exception:
                            pass

                # Dashboard live data update (REAL data, zero mocks)
                if self.dashboard:
                    try:
                        self.dashboard.update_from_wrapper(self)
                        # Se estivermos em jogo, gravar frame para replay
                        if (self.state_manager and self.state_manager.current_state == 'in_game' and
                                self.dashboard.recorder and self.state_manager._last_screenshot is not None):
                            # Encode screenshot para base64 (thumbnail ~20KB)
                            try:
                                import cv2
                                import numpy as np
                                from io import BytesIO
                                img = self.state_manager._last_screenshot
                                if isinstance(img, np.ndarray) and img.size > 0:
                                    # Resize para thumbnail 320x180
                                    thumb = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
                                    _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 50])
                                    b64 = base64.b64encode(buf).decode('ascii')
                                    self.dashboard.bridge.update(screenshot_b64=b64)
                                    # Gravar replay se ativo
                                    self.dashboard.record_replay_frame(
                                        screenshot=thumb,
                                        state=self.state_manager.current_state,
                                        action=getattr(self.play_logic, '_last_action', 'idle') if self.play_logic else 'idle',
                                        enemies=getattr(self.play_logic, '_last_enemies', 0) if self.play_logic else 0,
                                    )
                            except Exception:
                                pass
                    except Exception as e:
                        logger.debug(f"[WRAPPER] Dashboard update error: {e}")

                # Anti-ban checks
                if self.anti_ban:
                    if self.anti_ban.check_pattern():
                        logger.warning("[WRAPPER] Anti-ban: padrão repetitivo detetado, pausando brevemente")
                        time.sleep(random.uniform(5, 15))
                    if self.anti_ban.check_throttle():
                        logger.warning("[WRAPPER] Anti-ban: throttling ativado, reduzindo eficiência")
                        # Pode ser estendido para ajustar dificuldade do bot

                # Delay adaptativo: mais rapido em jogo, mais lento fora
                if self.state_manager and self.state_manager.current_state == 'in_game':
                    time.sleep(0.3)  # In-game: verificar mais frequentemente
                else:
                    time.sleep(random.uniform(0.8, 1.2))  # Fora de jogo: intervalo medio

            except Exception as e:
                logger.error(f"Erro no monitor: {e}")
                # Phase 9: Try error recovery
                if self.enable_error_recovery and self.error_recovery:
                    try:
                        context = self.error_recovery.classify_error(
                            e, component="wrapper", operation="monitor_loop"
                        )
                        recovered = self.error_recovery.handle_error(context, self)
                        if recovered:
                            logger.info("[WRAPPER] Monitor loop error recovered")
                    except Exception:
                        pass
                time.sleep(random.uniform(0.5, 1.0))

    def _take_break(self, duration: float):
        """Executa pausa obrigatoria"""
        self.stop()
        time.sleep(duration)
        logger.info("Retomando apos pausa...")
        self.start()

    def _update_all_coordinates(self, new_w: int, new_h: int):
        """Atualiza coordenadas em todos os componentes quando a janela muda de tamanho."""
        logger.info(f"[WRAPPER] Atualizando coordenadas para {new_w}x{new_h}")

        # Update unified detector
        if self.unified_detector:
            self.unified_detector.update_window_size(new_w, new_h)

        # Update lobby automator
        if self.lobby and hasattr(self.lobby, 'update_window_size'):
            self.lobby.update_window_size(new_w, new_h)

        # Update movement
        if self.play_logic and self.play_logic.movement:
            self.play_logic.movement.update_window_size(new_w, new_h)

        # Update screen automation (for hints)
        if self.state_manager and self.state_manager.screen_automation:
            self.state_manager.screen_automation.update_window(new_w, new_h)

    def _get_current_trophies(self) -> int:
        """Retorna trofeus atuais do brawler"""
        current = self.brawler_queue.get_current()
        if current:
            return current.current_trophies
        return 0

    def add_brawler_to_queue(
        self,
        name: str,
        current_trophies: int = 0,
        target_trophies: int = 350,
        target_wins: int = 10,
        priority: int = 1
    ):
        """Adiciona brawler a fila"""
        config = BrawlerConfig(
            name=name,
            current_trophies=current_trophies,
            target_trophies=target_trophies,
            target_wins=target_wins,
            priority=priority
        )
        self.brawler_queue.add_brawler(config)
        logger.info(f"Brawler adicionado: {name}")

    def get_queue(self) -> List[Dict]:
        """Retorna fila de brawlers"""
        return self.brawler_queue.get_queue()

    def get_status(self) -> Dict:
        """Retorna status completo"""
        logger.debug("[WRAPPER] get_status() chamado")
        session_duration = 0
        if self.session_start:
            session_duration = time.time() - self.session_start

        match_controller = getattr(self, "match_controller", None)

        window_snapshot = None
        if self.emulator_controller and hasattr(self.emulator_controller, "get_status_snapshot"):
            try:
                window_snapshot = self.emulator_controller.get_status_snapshot()
            except Exception:
                window_snapshot = None

        combat_snapshot = None
        if self.play_logic and hasattr(self.play_logic, "get_last_combat_snapshot"):
            try:
                combat_snapshot = self.play_logic.get_last_combat_snapshot()
            except Exception:
                combat_snapshot = None

        lobby_diagnostic = None
        if self.lobby and hasattr(self.lobby, "get_diagnostic_report"):
            try:
                lobby_diagnostic = self.lobby.get_diagnostic_report()
            except Exception:
                lobby_diagnostic = None

        screen_state = None
        if self.state_manager and self.state_manager.screen_automation and hasattr(self.state_manager.screen_automation, "get_current_state_name"):
            try:
                screen_state = self.state_manager.screen_automation.get_current_state_name()
            except Exception:
                screen_state = None

        tracker_stats = None
        if self.play_logic and hasattr(self.play_logic, "enemy_tracker") and self.play_logic.enemy_tracker:
            try:
                tracker_stats = self.play_logic.enemy_tracker.get_stats()
            except Exception:
                tracker_stats = None

        current_map = None
        if self.play_logic and hasattr(self.play_logic, "movement") and self.play_logic.movement:
            try:
                current_map = self.play_logic.movement.current_map
            except Exception:
                current_map = None

        return {
            "running": self.running,
            "current_state": self.state_manager.current_state if self.state_manager else "unknown",
            "last_known_state": getattr(self.state_manager, "last_known_state", "unknown") if self.state_manager else "unknown",
            "unknown_streak": getattr(self.state_manager, "unknown_streak", 0) if self.state_manager else 0,
            "last_unknown_hint": getattr(self.state_manager, "last_unknown_hint", None) if self.state_manager else None,
            "queue": self.get_queue(),
            "safety": self.safety.get_status() if self.safety else None,
            "session_duration_minutes": session_duration / 60,
            "matches_played": self.matches_played,
            "current_brawler": self.brawler_queue.get_current().name if self.brawler_queue.get_current() else None,
            "current_map": current_map,
            "emulator_controller_active": self.emulator_controller is not None,
            "window_active": window_snapshot.get("window_active") if isinstance(window_snapshot, dict) else None,
            "window_title": window_snapshot.get("window_title") if isinstance(window_snapshot, dict) else None,
            "models_loaded": self.detect_main is not None,
            "diagnostic_overlay_active": bool(getattr(self, "diagnostic_overlay", None)),
            "diagnostics": {
                "diagnostic_mode": self.diagnostic_mode,
                "lobby": lobby_diagnostic,
                "screen_state": screen_state,
                "progress": self.progress.get_stats() if self.progress else None,
                "match": match_controller.get_session_stats() if match_controller else None,
                "combat": combat_snapshot,
            },
            "tracker_stats": tracker_stats,
            # Phase 9: Error Recovery stats
            "error_recovery": self.error_recovery.get_stats() if self.error_recovery else {"enabled": False},
            # Phase 9: State Recovery stats
            "state_recovery": self.state_recovery.get_recovery_status() if self.state_recovery else {"is_recovering": False},
            # Phase 9: AutoCalibrator stats
            "auto_calibrator": {
                "enabled": self.auto_calibrator is not None,
                "cached_coords": len(self.auto_calibrator.coords_cache) if self.auto_calibrator else 0,
            },
            # Phase 9: OCR stats
            "ocr_detector": self.ocr_detector.get_detection_stats() if self.ocr_detector else {"reader_available": False},
            # Phase 9: Debug Visualizer
            "debug_visualizer": {
                "enabled": self.debug_visualizer is not None,
                "running": self.debug_visualizer.is_running if self.debug_visualizer else False,
            },
        }
