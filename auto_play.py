#!/usr/bin/env python3
"""
Auto-Play System - Soberana Omega
Mini-bot autonomo que usa componentes existentes para jogar Brawl Stars.
"""

import sys
import os
import time
import random
import math
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    handlers=[
        logging.FileHandler('auto_play.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AutoPlayBot:
    """Bot autonomo simplificado que joga Brawl Stars."""

    def __init__(self):
        self.state = 'unknown'
        self.state_start_time = time.time()
        self.cycle_count = 0
        self.combat_actions = 0
        self.play_clicks = 0
        self.matches_completed = 0
        self.emulator = None
        self.detector = None
        self.screenshot_taker = None
        self._init_components()

    def _init_components(self):
        """Inicializa componentes necessarios."""
        try:
            from pylaai_real.screenshot_taker import ScreenshotTaker
            self.screenshot_taker = ScreenshotTaker(window_title='BlueStacks App Player')
            logger.info("[AUTO] ScreenshotTaker inicializado")
        except Exception as e:
            logger.error(f"[AUTO] Falha ao inicializar ScreenshotTaker: {e}")

        try:
            from pylaai_real.unified_state_detector import UnifiedStateDetector
            self.detector = UnifiedStateDetector(images_path=Path('images'))
            logger.info("[AUTO] UnifiedStateDetector inicializado")
        except Exception as e:
            logger.error(f"[AUTO] Falha ao inicializar detector: {e}")

        try:
            from emulator_controller import EmulatorController, EmulatorConfig
            # Detect actual ADB port dynamically
            adb_port = 5554
            try:
                result = subprocess.run(
                    [r'C:\Program Files\BlueStacks_nxt\HD-Adb.exe', 'devices'],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split('\n'):
                    if 'emulator-' in line and 'device' in line:
                        port_str = line.split()[0].split('-')[-1]
                        adb_port = int(port_str)
                        logger.info(f"[AUTO] Detected ADB port: {adb_port}")
                        break
            except Exception as e2:
                logger.debug(f"[AUTO] ADB port detection failed: {e2}, using default 5554")

            config = EmulatorConfig.for_bluestacks()
            config.adb_port = adb_port
            self.emulator = EmulatorController(config)
            if self.emulator.connect():
                logger.info(f"[AUTO] EmulatorController conectado: {self.emulator.adb.device_id}")
            else:
                logger.warning("[AUTO] EmulatorController nao conseguiu conectar")
                self.emulator = None
        except Exception as e:
            logger.error(f"[AUTO] Falha ao inicializar EmulatorController: {e}")
            self.emulator = None

    def get_screenshot(self):
        if self.screenshot_taker:
            try:
                return self.screenshot_taker.take()
            except Exception as e:
                logger.debug(f"[AUTO] Screenshot error: {e}")
        return None

    def detect_state(self, screenshot):
        if self.detector and screenshot is not None:
            try:
                result = self.detector.detect(screenshot)
                return result.state, result.confidence, result.button_coords
            except Exception as e:
                logger.debug(f"[AUTO] Detection error: {e}")
        return 'unknown', 0.0, None

    def tap(self, x, y):
        if self.emulator:
            try:
                self.emulator.tap_scaled(x, y)
                return True
            except Exception as e:
                logger.debug(f"[AUTO] Tap error: {e}")
        return False

    def swipe(self, x1, y1, x2, y2, duration=200):
        if self.emulator:
            try:
                self.emulator.swipe_scaled(x1, y1, x2, y2, duration=duration)
                return True
            except Exception as e:
                logger.debug(f"[AUTO] Swipe error: {e}")
        return False

    def transition_to(self, new_state):
        if new_state != self.state:
            logger.info(f"[AUTO] *** STATE: {self.state} -> {new_state} ***")
            self.state = new_state
            self.state_start_time = time.time()

    def handle_lobby(self, screenshot):
        logger.info("[AUTO] Handling LOBBY")
        self.tap(1751, 985)
        self.play_clicks += 1
        time.sleep(2.0)
        self.transition_to('loading')

    def handle_loading(self, screenshot):
        elapsed = time.time() - self.state_start_time
        if elapsed > 10:
            logger.info("[AUTO] Loading timeout, forcing in_game")
            self.transition_to('in_game')
        else:
            time.sleep(1.0)

    def handle_matchmaking(self, screenshot):
        elapsed = time.time() - self.state_start_time
        if elapsed > 15:
            logger.info("[AUTO] Matchmaking timeout, forcing in_game")
            self.transition_to('in_game')
        else:
            time.sleep(1.0)

    def handle_in_game(self, screenshot):
        if screenshot is None:
            time.sleep(0.5)
            return

        h, w = screenshot.shape[:2]
        joy_x = round(w * 0.10)
        joy_y = round(h * 0.75)
        joy_radius = round(min(w, h) * 0.08)
        atk_x = round(w * 0.90)
        atk_y = round(h * 0.82)

        angle = random.uniform(0, 2 * math.pi)
        target_x = int(joy_x + joy_radius * math.cos(angle))
        target_y = int(joy_y + joy_radius * math.sin(angle))
        self.swipe(joy_x, joy_y, target_x, target_y, duration=300)

        if random.random() > 0.3:
            self.tap(atk_x, atk_y)
            self.combat_actions += 1

        elapsed = time.time() - self.state_start_time
        if elapsed > 180:
            logger.info("[AUTO] Match max duration reached")
            self.transition_to('end')
        else:
            time.sleep(0.8 + random.uniform(0, 0.5))

    def handle_end(self, screenshot):
        self.tap(960, 950)
        time.sleep(1.0)
        self.tap(960, 950)
        time.sleep(1.0)
        if self.emulator:
            try:
                self.emulator.keyevent(4)
            except:
                pass
        self.matches_completed += 1
        self.transition_to('lobby')
        time.sleep(3.0)

    def handle_unknown(self, screenshot):
        elapsed = time.time() - self.state_start_time
        if elapsed > 5:
            self.tap(960, 540)
            time.sleep(1.0)
            if self.emulator:
                try:
                    self.emulator.keyevent(4)
                except:
                    pass
            self.transition_to('lobby')
        time.sleep(1.0)

    def run_cycle(self):
        self.cycle_count += 1
        screenshot = self.get_screenshot()
        if screenshot is None:
            time.sleep(1.0)
            return

        if self.cycle_count % 10 == 0:
            try:
                ts = datetime.now().strftime("%H%M%S")
                Image.fromarray(screenshot).save(f"auto_play_{ts}_{self.state}.png")
            except:
                pass

        detected_state, conf, button_coords = self.detect_state(screenshot)
        if detected_state != self.state:
            self.transition_to(detected_state)

        handlers = {
            'lobby': self.handle_lobby,
            'loading': self.handle_loading,
            'matchmaking': self.handle_matchmaking,
            'in_game': self.handle_in_game,
            'end': self.handle_end,
            'unknown': self.handle_unknown,
        }
        handler = handlers.get(self.state, self.handle_unknown)
        handler(screenshot)

        if self.cycle_count % 5 == 0:
            elapsed = time.time() - self.state_start_time
            logger.info(
                f"[AUTO] Cycle {self.cycle_count:4d} | State: {self.state:12s} | "
                f"Conf: {conf:.2f} | Time: {elapsed:5.1f}s | "
                f"Plays: {self.play_clicks} | Combats: {self.combat_actions} | Matches: {self.matches_completed}"
            )

    def run(self, duration=300):
        logger.info("=" * 60)
        logger.info("AUTO PLAY BOT - Iniciando")
        logger.info(f"ADB: {self.emulator is not None} | Detector: {self.detector is not None} | Screenshot: {self.screenshot_taker is not None}")
        logger.info("=" * 60)

        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                self.run_cycle()
        except KeyboardInterrupt:
            logger.info("[AUTO] Interrupted")
        except Exception as e:
            logger.error(f"[AUTO] Fatal error: {e}")
            import traceback
            traceback.print_exc()

        logger.info("=" * 60)
        logger.info("AUTO PLAY BOT - RESUMO")
        logger.info(f"Ciclos: {self.cycle_count} | Plays: {self.play_clicks} | Combats: {self.combat_actions} | Matches: {self.matches_completed}")
        logger.info("=" * 60)


def main():
    bot = AutoPlayBot()
    bot.run(duration=300)


if __name__ == '__main__':
    main()
