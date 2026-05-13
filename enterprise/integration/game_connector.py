"""Game Connector - Integration between Enterprise AI and Real Brawl Stars Game"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger("game_connector")


class GameConnector:
    """
    Conecta o Enterprise AI ao jogo real via ADB e wrapper existente.
    Usa PylaAIEnhanced para captura de ecrã e controlo real.
    """

    def __init__(self, emulator_type: str = "bluestacks"):
        self.emulator_type = emulator_type
        self.wrapper = None
        self.connected = False
        self.screenshot = None
        self.detect = None
        self.model = None

        self._initialize_connection()

    def _initialize_connection(self):
        """Inicializa ligação ao emulador e carrega modelos."""
        try:
            from wrapper import PylaAIEnhanced
            from pylaai_real.detect import Detect
            from pylaai_real.screenshot_taker import ScreenshotTaker

            logger.info("[GameConnector] Inicializando PylaAIEnhanced...")
            self.wrapper = PylaAIEnhanced(
                install_path=Path("c:/Users/rodri/Desktop/bot brawl/pylaai_workspace"),
                diagnostic_mode=False
            )

            logger.info("[GameConnector] A carregar modelos YOLO...")
            if self.wrapper._load_trained_models():
                self.model = self.wrapper.model
                logger.info("[GameConnector] Modelos carregados com sucesso")
            else:
                logger.warning("[GameConnector] Usando modelo genérico")
                from ultralytics import YOLO
                self.model = YOLO("yolov8n.pt")

            classes = self.wrapper.vision_config.get("classes", ["Player", "Bush", "Enemy", "Cubebox"])
            class_dict = {i: name for i, name in enumerate(classes)}

            conf = self.wrapper.vision_config.get("confidence_threshold", 0.5)
            self.detect = Detect(self.model, classes=class_dict, conf=conf)

            self.screenshot = ScreenshotTaker()

            if self.wrapper._try_init_emulator_controller():
                logger.info("[GameConnector] EmulatorController conectado via ADB")
            else:
                logger.warning("[GameConnector] ScreenshotTaker fallback ativo")

            self.connected = True
            logger.info("[GameConnector] Ligação estabelecida com sucesso!")

        except Exception as e:
            logger.error(f"[GameConnector] Erro na inicialização: {e}")
            self.connected = False

    def capture_screen(self) -> Optional[np.ndarray]:
        """Captura ecrã atual do jogo."""
        try:
            if self.wrapper and self.wrapper.emulator_controller:
                return self.wrapper.emulator_controller.get_screenshot()
            elif self.screenshot:
                return self.screenshot.take_screenshot()
        except Exception as e:
            logger.error(f"[GameConnector] Erro capture_screen: {e}")
        return None

    def detect_objects(self, frame: np.ndarray) -> Dict[str, List[List[int]]]:
        """Deteção de objetos no frame usando YOLO real."""
        if self.detect is None:
            return {}
        try:
            return self.detect.detect_objects(frame)
        except Exception as e:
            logger.error(f"[GameConnector] Erro detect_objects: {e}")
            return {}

    def get_game_state(self) -> Dict[str, Any]:
        """Obtém estado atual do jogo (HP, posição, etc)."""
        frame = self.capture_screen()
        if frame is None:
            return {}

        detections = self.detect_objects(frame)

        state = {
            "timestamp": time.time(),
            "frame_shape": frame.shape,
            "detections": detections,
            "enemy_count": len(detections.get("Enemy", [])),
            "player_detected": "Player" in detections,
            "bush_count": len(detections.get("Bush", [])),
        }

        if detections.get("Player"):
            player_bbox = detections["Player"][0]
            state["player_position"] = self._bbox_center(player_bbox)

        if detections.get("Enemy"):
            state["enemy_positions"] = [self._bbox_center(e) for e in detections["Enemy"]]

        return state

    def _bbox_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Calcula centro de bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def send_input(self, action: str, **kwargs) -> bool:
        """Envia input para o jogo (tap, swipe, key)."""
        if not self.wrapper or not self.wrapper.emulator_controller:
            return False

        try:
            if action == "tap":
                x, y = kwargs.get("x", 0), kwargs.get("y", 0)
                return self.wrapper.emulator_controller.tap(x, y)
            elif action == "swipe":
                x1, y1 = kwargs.get("x1", 0), kwargs.get("y1", 0)
                x2, y2 = kwargs.get("x2", 0), kwargs.get("y2", 0)
                duration = kwargs.get("duration", 300)
                return self.wrapper.emulator_controller.swipe(x1, y1, x2, y2, duration)
            elif action == "key":
                key = kwargs.get("key", "")
                return self.wrapper.emulator_controller.press_key(key)
        except Exception as e:
            logger.error(f"[GameConnector] Erro send_input: {e}")

        return False

    def is_connected(self) -> bool:
        """Verifica se está conectado ao jogo."""
        return self.connected and self.wrapper is not None


class RealTimeVisionLoop:
    """Loop de visão em tempo real para o Enterprise AI."""

    def __init__(self, game_connector: GameConnector, event_bus=None):
        self.connector = game_connector
        self.event_bus = event_bus
        self.running = False
        self.fps = 30
        self.frame_count = 0
        self.last_frame_time = 0

    def start(self):
        """Inicia o loop de visão."""
        self.running = True
        logger.info("[VisionLoop] Loop de visão iniciado")

    def stop(self):
        """Para o loop de visão."""
        self.running = False
        logger.info("[VisionLoop] Loop de visão parado")

    def get_latest_state(self) -> Dict[str, Any]:
        """Obtém último estado processado."""
        return self.connector.get_game_state()

    def process_frame(self) -> Dict[str, Any]:
        """Processa um frame e retorna estado."""
        frame = self.connector.capture_screen()
        if frame is None:
            return {}

        detections = self.connector.detect_objects(frame)

        self.frame_count += 1
        current_time = time.time()
        self.last_frame_time = current_time

        return {
            "frame_id": self.frame_count,
            "timestamp": current_time,
            "frame_shape": frame.shape,
            "detections": detections,
            "fps": self.fps,
        }
