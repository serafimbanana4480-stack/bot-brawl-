"""
wrapper_initializers.py

Inicializadores modulares para PylaAIEnhanced.
Extraídos do wrapper.py para reduzir complexidade.

Autor: Sobberana Omega
"""

import logging
from pathlib import Path
from typing import Optional, Any, Dict, List
from dataclasses import dataclass

from .pylaai_real.lobby_automator import BrawlerQueue, BrawlerConfig
from .decision.brawler_selector import BrawlerSelector

logger = logging.getLogger(__name__)


@dataclass
class SubsystemConfig:
    """Configuration for a single subsystem."""
    name: str
    import_path: str
    class_name: str
    init_kwargs: Optional[Dict] = None
    required: bool = False


class WrapperInitializers:
    """Coleção de inicializadores para o wrapper."""

    def __init__(self, central_config: dict, images_path: Path, models_path: Path):
        self.central_config = central_config
        self.images_path = images_path
        self.models_path = models_path

    def init_brawler_queue(self) -> BrawlerQueue:
        """Inicializa fila de brawlers."""
        queue = BrawlerQueue()
        queue_config = self.central_config.get("brawler_queue", [])

        if queue_config:
            for bcfg in queue_config:
                queue.add_brawler(BrawlerConfig(
                    name=bcfg.get("name", "colt"),
                    current_trophies=bcfg.get("current_trophies", 0),
                    target_trophies=bcfg.get("target_trophies", 500),
                    target_wins=bcfg.get("target_wins", 10),
                    priority=bcfg.get("priority", 1),
                    enabled=bcfg.get("enabled", True)
                ))
            logger.info(f"[WRAPPER] {len(queue_config)} brawlers carregados: {[b.get('name') for b in queue_config]}")
        else:
            queue.add_brawler(BrawlerConfig(
                name="colt",
                current_trophies=0,
                target_trophies=400,
                target_wins=10,
                priority=1,
                enabled=True
            ))
            logger.info("[WRAPPER] Brawler 'colt' adicionado à fila (padrão)")

        return queue

    def init_brawler_selector(self) -> Optional[BrawlerSelector]:
        """Inicializa seletor de brawlers."""
        enabled = self.central_config.get("brawler_selection_enabled", True)
        if enabled:
            selector = BrawlerSelector()
            logger.info("[WRAPPER] Brawler selector inicializado")
            return selector
        logger.info("[WRAPPER] Brawler selection desabilitado")
        return None

    def init_phase9_subsystems(self) -> Dict[str, Any]:
        """Inicializa subsistemas Phase 9."""
        subsystems = {}

        try:
            from .pylaai_real.state_recovery import StateRecoverySystem
            subsystems["state_recovery"] = StateRecoverySystem()
            logger.info("[WRAPPER] State Recovery System inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] State Recovery System indisponível: {e}")
            subsystems["state_recovery"] = None

        try:
            from .pylaai_real.auto_calibrator import AutoCalibrator
            subsystems["auto_calibrator"] = AutoCalibrator()
            logger.info("[WRAPPER] AutoCalibrator inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] AutoCalibrator indisponível: {e}")
            subsystems["auto_calibrator"] = None

        try:
            from .pylaai_real.ocr_state_detector import OCRStateDetector
            subsystems["ocr_detector"] = OCRStateDetector()
            logger.info("[WRAPPER] OCR State Detector inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] OCR State Detector indisponível: {e}")
            subsystems["ocr_detector"] = None

        try:
            from .pylaai_real.debug_visualizer import DebugVisualizer, DebugMode
            subsystems["debug_visualizer"] = DebugVisualizer()
            logger.info("[WRAPPER] Debug Visualizer inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] Debug Visualizer indisponível: {e}")
            subsystems["debug_visualizer"] = None

        return subsystems

    def init_decision_subsystems(self) -> Dict[str, Any]:
        """Inicializa subsistemas de decisão."""
        subsystems = {}

        try:
            from .decision.utility_ai import UtilityAI
            subsystems["utility_ai"] = UtilityAI()
            logger.info("[WRAPPER] Utility AI inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] Utility AI indisponível: {e}")
            subsystems["utility_ai"] = None

        try:
            from .decision.sticky_target import StickyTarget
            subsystems["sticky_target"] = StickyTarget()
            logger.info("[WRAPPER] Sticky Target inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] Sticky Target indisponível: {e}")
            subsystems["sticky_target"] = None

        try:
            from .decision.intent_system import IntentSystem
            subsystems["intent_system"] = IntentSystem()
            logger.info("[WRAPPER] Intent System inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] Intent System indisponível: {e}")
            subsystems["intent_system"] = None

        try:
            from .decision.enemy_intention import EnemyIntentionPredictor
            subsystems["enemy_intention"] = EnemyIntentionPredictor()
            logger.info("[WRAPPER] Enemy Intention Predictor inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] Enemy Intention Predictor indisponível: {e}")
            subsystems["enemy_intention"] = None

        try:
            from .decision.meta_awareness import MetaAwareness
            subsystems["meta_awareness"] = MetaAwareness()
            logger.info("[WRAPPER] Meta Awareness inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] Meta Awareness indisponível: {e}")
            subsystems["meta_awareness"] = None

        return subsystems

    def init_core_subsystems(self) -> Dict[str, Any]:
        """Inicializa subsistemas core."""
        subsystems = {}

        try:
            from .core.world_model import WorldModel
            subsystems["world_model"] = WorldModel()
            logger.info("[WRAPPER] World Model inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] World Model indisponível: {e}")
            subsystems["world_model"] = None

        try:
            from .core.pressure_map import PressureMap
            subsystems["pressure_map"] = PressureMap()
            logger.info("[WRAPPER] Pressure Map inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] Pressure Map indisponível: {e}")
            subsystems["pressure_map"] = None

        try:
            from .core.cover_system import CoverSystem
            subsystems["cover_system"] = CoverSystem()
            logger.info("[WRAPPER] Cover System inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] Cover System indisponível: {e}")
            subsystems["cover_system"] = None

        return subsystems

    def init_observability(self) -> Optional[Any]:
        """Inicializa sistema de observabilidade."""
        try:
            from core.observability import ObservabilityCollector
            collector = ObservabilityCollector(
                max_events=2000,
                metrics_dir=Path("pylaai_workspace/observability")
            )
            logger.info("[WRAPPER] Observability collector inicializado")
            return collector
        except Exception as e:
            logger.warning(f"[WRAPPER] Observability indisponível: {e}")
            return None

    def init_dashboard(self, port: int) -> Optional[Any]:
        """Inicializa servidor dashboard."""
        try:
            from .pylaai_real.dashboard_server import DashboardServer
            dashboard = DashboardServer(port=port)
            logger.info(f"[WRAPPER] Dashboard server inicializado (porta {port})")
            return dashboard
        except Exception as e:
            logger.warning(f"[WRAPPER] Dashboard server indisponível: {e}")
            return None

    def init_data_collector(self) -> Optional[Any]:
        """Inicializa coletor de dados."""
        try:
            from dataset.collector import GameplayCollector
            collector = GameplayCollector()
            logger.info("[WRAPPER] Data collector inicializado")
            return collector
        except Exception as e:
            logger.warning(f"[WRAPPER] Data collector indisponível: {e}")
            return None

    def init_reward_bridge(self, data_collector) -> Optional[Any]:
        """Inicializa reward bridge."""
        try:
            from core.reward_bridge import RewardBridge
            bridge = RewardBridge(data_collector=data_collector)
            logger.info("[WRAPPER] Reward bridge inicializado")
            return bridge
        except ImportError as e:
            logger.warning(f"[WRAPPER] RewardBridge indisponível (não instalado): {e}")
            return None
        except Exception as e:
            logger.error(f"[WRAPPER] RewardBridge ERRO (Q-Learning vai usar heurísticas): {e}")
            return None

    def init_anti_ban(self) -> Optional[Any]:
        """Inicializa sistema anti-ban."""
        try:
            from core.anti_ban import AntiBanSystem
            anti_ban = AntiBanSystem()
            logger.info("[WRAPPER] Anti-ban system inicializado")
            return anti_ban
        except ImportError as e:
            logger.warning(f"[WRAPPER] AntiBanSystem indisponível (não instalado): {e}")
            return None
        except Exception as e:
            logger.error(f"[WRAPPER] AntiBanSystem ERRO: {e}")
            return None