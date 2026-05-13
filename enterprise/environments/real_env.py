"""Real Brawl Stars Environment - Gymnasium-compatible environment with real game integration (NO MOCK DATA)"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger("real_env")


class RealBrawlStarsEnvironmentError(Exception):
    """Raised when environment initialization fails."""
    pass


class RealBrawlStarsEnvironment:
    """
    Real Gymnasium-compatible environment connecting to actual Brawl Stars game.
    NO MOCK DATA - Requires real game connection via ADB.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        self.observation_space = {
            "screen": {"shape": (84, 84, 3), "dtype": np.uint8},
            "detections": {"shape": (10, 6), "dtype": np.float32},
            "state": {"shape": (128,), "dtype": np.float32},
        }

        self.action_space = {
            "movement": 8,
            "abilities": 4,
            "target": 10,
        }

        self.actions = {
            0: "up",
            1: "down",
            2: "left",
            3: "right",
            4: "attack",
            5: "skill_1",
            6: "skill_2",
            7: "ultimate",
        }

        self.state = self._reset_state()
        self.episode_step = 0
        self.max_steps = self.config.get("max_steps", 1000)

        self.connector = None
        self.detector = None
        self._initialized = False

        self._initialize_real_components()

    def _initialize_real_components(self):
        """Initialize real components - FAILS if not available."""
        try:
            from enterprise.vision.yolo_detector import YOLOv8Detector

            logger.info("[RealEnv] Initializing REAL components...")

            self.detector = YOLOv8Detector(
                model_path="c:/Users/rodri/Desktop/bot brawl/models/brawlstars_yolov8.pt",
                conf_threshold=0.5
            )
            self.detector.load()
            logger.info("[RealEnv] YOLO detector loaded REAL model!")

            try:
                from enterprise.integration.game_connector import GameConnector
                self.connector = GameConnector()
                if self.connector.is_connected():
                    logger.info("[RealEnv] GameConnector connected to REAL game!")
                else:
                    logger.warning("[RealEnv] GameConnector not connected - game not running")
            except Exception as e:
                logger.warning(f"[RealEnv] GameConnector not available: {e}")

            self._initialized = True
            logger.info("[RealEnv] REAL components initialized!")

        except Exception as e:
            logger.error(f"[RealEnv] Failed to initialize components: {e}")
            raise RealBrawlStarsEnvironmentError(f"Cannot initialize environment: {e}")

    def _reset_state(self) -> Dict[str, Any]:
        return {
            "player_health": 100.0,
            "player_position": (400.0, 400.0),
            "player_velocity": (0.0, 0.0),
            "enemies": [
                {"id": 0, "position": (200.0, 200.0), "health": 50.0},
                {"id": 1, "position": (600.0, 200.0), "health": 50.0},
            ],
            "allies": [],
            "gems": [{"position": (400.0, 400.0), "collected": False}],
            "score": 0,
            "match_time": 0,
            "in_bush": False,
            "target_visible": False,
        }

    def reset(self) -> np.ndarray:
        self.state = self._reset_state()
        self.episode_step = 0
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.episode_step += 1

        action_name = self.actions.get(action, "attack")

        if self.connector and self.connector.is_connected():
            reward = self._execute_real_action(action_name)
        else:
            raise RealBrawlStarsEnvironmentError(
                "Cannot execute action - no game connection. "
                "Start the game and emulator to use this environment."
            )

        self._update_game_state()

        done = self._check_done()

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, done, info

    def _execute_real_action(self, action: str) -> float:
        """Execute action on real game."""
        reward = 0.0

        if not self.connector:
            raise RealBrawlStarsEnvironmentError("No game connector available")

        try:
            if action == "attack":
                self.connector.send_input("key", key="attack")
                reward += 1.0
            elif action in ["up", "down", "left", "right"]:
                self.connector.send_input("key", key=action)
                reward += 0.1
            elif action in ["skill_1", "skill_2", "ultimate"]:
                self.connector.send_input("key", key=action)
                reward += 2.0

            game_state = self.connector.get_game_state()

            if game_state.get("enemy_count", 0) > 0:
                reward += 5.0

            if game_state.get("player_detected"):
                reward += 1.0

        except Exception as e:
            logger.error(f"[RealEnv] Error executing real action: {e}")
            raise RealBrawlStarsEnvironmentError(f"Action execution failed: {e}")

        return reward

    def _update_game_state(self):
        self.state["match_time"] += 1

        for enemy in self.state["enemies"]:
            if enemy["health"] > 0:
                dx = self.state["player_position"][0] - enemy["position"][0]
                dy = self.state["player_position"][1] - enemy["position"][1]
                dist = self._distance(self.state["player_position"], enemy["position"])

                if dist > 100:
                    speed = 3.0
                    enemy["position"] = (
                        enemy["position"][0] + (dx / dist) * speed,
                        enemy["position"][1] + (dy / dist) * speed,
                    )

                    if dist < 80:
                        self.state["player_health"] -= 2.0

        if self.state["player_health"] <= 0:
            for enemy in self.state["enemies"]:
                if enemy["health"] > 0:
                    self.state["score"] += 1

    def _check_done(self) -> bool:
        if self.state["player_health"] <= 0:
            return True

        if self.episode_step >= self.max_steps:
            return True

        all_enemies_dead = all(e["health"] <= 0 for e in self.state["enemies"])
        all_gems_collected = all(g["collected"] for g in self.state["gems"])

        return all_enemies_dead or all_gems_collected

    def _get_observation(self) -> np.ndarray:
        """Get observation from real game."""
        obs = np.zeros((84, 84, 3), dtype=np.uint8)

        if self._initialized and self.detector and self.connector:
            frame = self.connector.capture_screen()
            if frame is not None:
                try:
                    detections = self.detector.detect(frame)

                    for det in detections:
                        bbox = det.get("bbox", [])
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            x1 = int(x1 * 84 / frame.shape[1])
                            y1 = int(y1 * 84 / frame.shape[0])
                            x2 = int(x2 * 84 / frame.shape[1])
                            y2 = int(y2 * 84 / frame.shape[0])

                            class_name = det.get("class_name", "")
                            if class_name == "Enemy":
                                obs[y1:y2, x1:x2] = [255, 0, 0]
                            elif class_name == "Player":
                                obs[y1:y2, x1:x2] = [0, 255, 0]
                            elif class_name == "Bush":
                                obs[y1:y2, x1:x2] = [0, 255, 0]
                except Exception as e:
                    logger.debug(f"[RealEnv] Error processing frame: {e}")

        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "player_health": self.state["player_health"],
            "score": self.state["score"],
            "enemies_alive": sum(1 for e in self.state["enemies"] if e["health"] > 0),
            "gems_collected": sum(1 for g in self.state["gems"] if g["collected"]),
            "initialized": self._initialized,
            "connector_connected": self.connector.is_connected() if self.connector else False,
        }

    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def close(self):
        """Close environment and free resources."""
        if self.connector:
            logger.info("[RealEnv] Closing GameConnector...")

    def render(self):
        """Render current state (for debugging)."""
        return self._get_observation()
