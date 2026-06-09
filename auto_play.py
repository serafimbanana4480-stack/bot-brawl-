#!/usr/bin/env python3
"""Auto-Play System - Soberana Omega"""

import sys
import os
import time
import random
import math
import subprocess
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
from PIL import Image

from core.logging_config import setup_logging, get_logger
from core.metrics import (
    set_bot_state,
    inc_matches_completed,
    inc_errors,
    set_detection_confidence,
    observe_cycle_duration,
)
from core.error_recovery import inference_retry

logger = get_logger(__name__)

BASE_W = 1920
BASE_H = 1080

FALLBACK = {
    'play_button': (1751, 985),
    'cancel_button': (960, 900),
    'center_screen': (960, 540),
    'proceed_area': (1554, 990),
    'news_close': (1868, 70),
    'brawler_select': (960, 750),
    'connection_reload': (540, 628),
}


class AutoPlayBot:
    def __init__(self):
        self.state = 'unknown'
        self.state_start_time = time.time()
        self.cycle_count = 0
        self.combat_actions = 0
        self.play_clicks = 0
        self.matches_completed = 0
        self.recovery_actions = 0
        self.forced_restarts = 0
        self.state_history: deque = deque(maxlen=20)
        self._pending_state = 'unknown'
        self._pending_count = 0
        self._recovery_attempts = 0
        self._last_screenshot = None
        self.emulator = None
        self.detector = None
        self.screenshot_taker = None
        self._init_components()

    def _init_components(self):
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
                inc_errors(error_type="screenshot")
        return None

    @inference_retry
    def _run_detection(self, screenshot):
        if self.detector and screenshot is not None:
            result = self.detector.detect(screenshot)
            return result.state, result.confidence, result.button_coords
        raise ValueError("No detector or screenshot")

    def detect_state(self, screenshot):
        try:
            return self._run_detection(screenshot)
        except Exception as e:
            logger.debug(f"[AUTO] Detection error: {e}")
            inc_errors(error_type="inference")
            return 'unknown', 0.0, None

    def _tap_1080(self, x, y):
        if self.emulator:
            try:
                self.emulator.tap_scaled(x, y)
                return True
            except Exception as e:
                logger.debug(f"[AUTO] Tap error: {e}")
        return False

    def _tap_raw(self, x, y):
        if self.emulator:
            try:
                self.emulator.tap(x, y)
                return True
            except Exception as e:
                logger.debug(f"[AUTO] Tap raw error: {e}")
        return False

    def _swipe_1080(self, x1, y1, x2, y2, duration=200):
        if self.emulator:
            try:
                self.emulator.swipe_scaled(x1, y1, x2, y2, duration=duration)
                return True
            except Exception as e:
                logger.debug(f"[AUTO] Swipe error: {e}")
        return False

    def _keyevent(self, keycode):
        if self.emulator:
            try:
                self.emulator.keyevent(keycode)
                return True
            except Exception as e:
                logger.debug(f"[AUTO] Keyevent error: {e}")
        return False

    def _adb_shell(self, args):
        if not self.emulator or not self.emulator.adb:
            logger.warning("[AUTO] Emulator/ADB not available for shell command")
            return None
        try:
            cmd = [self.emulator.adb.adb_path, "-s", self.emulator.adb.device_id, "shell"] + args
            return subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except Exception as e:
            logger.error(f"[AUTO] ADB shell error: {e}")
            return None

    def _restart_app(self):
        logger.warning("[AUTO] Restarting Brawl Stars via ADB")
        self._adb_shell(["am", "force-stop", "com.supercell.brawlstars"])
        time.sleep(2.0)
        self._adb_shell(["monkey", "-p", "com.supercell.brawlstars", "-c", "android.intent.category.LAUNCHER", "1"])
        self.forced_restarts += 1
        time.sleep(5.0)
        self.transition_to('unknown')
        self._recovery_attempts = 0

    def transition_to(self, new_state):
        if new_state != self.state:
            logger.info(f"[AUTO] *** STATE: {self.state} -> {new_state} ***")
            set_bot_state(new_state)
            self.state = new_state
            self.state_start_time = time.time()
            self._recovery_attempts = 0
            self._pending_state = new_state
            self._pending_count = 0

    def _update_smoothing(self, detected_state):
        self.state_history.append(detected_state)
        if detected_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = detected_state
            self._pending_count = 1
        if self._pending_count >= 3 and self._pending_state != self.state:
            self.transition_to(self._pending_state)

    def _btn_or_fallback(self, button_coords, key):
        return button_coords if button_coords is not None else FALLBACK.get(key)

    def _tap_btn(self, button_coords, key):
        coords = self._btn_or_fallback(button_coords, key)
        if button_coords:
            return self._tap_raw(*coords)
        return self._tap_1080(*coords)

    def handle_lobby(self, screenshot, button_coords):
        logger.info("[AUTO] Handling LOBBY")
        self._tap_btn(button_coords, 'play_button')
        self.play_clicks += 1
        time.sleep(2.0)
        self.transition_to('loading')

    def handle_loading(self, screenshot):
        elapsed = time.time() - self.state_start_time
        if elapsed > 25:
            frozen = False
            if self._last_screenshot is not None and screenshot is not None:
                try:
                    if self._last_screenshot.shape == screenshot.shape:
                        diff = np.mean(np.abs(self._last_screenshot.astype(float) - screenshot.astype(float)))
                        frozen = diff < 2.0
                except Exception:
                    pass
            if frozen:
                logger.warning("[AUTO] Loading screen frozen, restarting app")
                self._restart_app()
            else:
                logger.info("[AUTO] Loading timeout, forcing in_game")
                self.transition_to('in_game')
        else:
            time.sleep(1.0)

    def handle_matchmaking(self, screenshot, button_coords):
        elapsed = time.time() - self.state_start_time
        if elapsed > 30:
            logger.warning("[AUTO] Matchmaking timeout, cancelling")
            self._tap_btn(button_coords, 'cancel_button')
            time.sleep(2.0)
            self.transition_to('lobby')
        else:
            time.sleep(1.0)

    def handle_brawler_selection(self, screenshot):
        logger.info("[AUTO] Handling BRAWLER SELECTION")
        self._tap_1080(*FALLBACK['brawler_select'])
        time.sleep(2.0)

    def handle_connection_lost(self, screenshot, button_coords):
        logger.warning("[AUTO] Handling CONNECTION LOST")
        self._tap_btn(button_coords, 'connection_reload')
        time.sleep(3.0)

    def handle_news(self, screenshot, button_coords):
        logger.info("[AUTO] Handling NEWS popup")
        self._tap_btn(button_coords, 'news_close')
        time.sleep(1.5)

    def handle_season_reset(self, screenshot, button_coords):
        logger.info("[AUTO] Handling SEASON RESET")
        self._tap_btn(button_coords, 'proceed_area')
        time.sleep(2.0)

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
        self._swipe_1080(joy_x, joy_y, target_x, target_y, duration=300)
        if random.random() > 0.3:
            self._tap_raw(atk_x, atk_y)
            self.combat_actions += 1
        elapsed = time.time() - self.state_start_time
        if elapsed > 210:
            logger.info("[AUTO] In-game timeout, forcing end sequence")
            self._keyevent(4)
            time.sleep(1.5)
            self._tap_1080(*FALLBACK['proceed_area'])
            time.sleep(1.0)
            self.transition_to('end')
        else:
            time.sleep(0.8 + random.uniform(0, 0.5))

    def handle_end(self, screenshot, button_coords):
        coords = self._btn_or_fallback(button_coords, 'proceed_area')
        if button_coords:
            self._tap_raw(*coords)
            time.sleep(1.0)
            self._tap_raw(*coords)
        else:
            self._tap_1080(*coords)
            time.sleep(1.0)
            self._tap_1080(*coords)
        time.sleep(1.0)
        self._keyevent(4)
        self.matches_completed += 1
        inc_matches_completed()
        self.transition_to('lobby')
        time.sleep(3.0)

    def handle_unknown(self, screenshot):
        elapsed = time.time() - self.state_start_time
        if elapsed > 12:
            self.recovery_actions += 1
            self._recovery_attempts += 1
            logger.warning(f"[AUTO] Unknown timeout, recovery attempt {self._recovery_attempts}/3")
            self._tap_1080(*FALLBACK['center_screen'])
            time.sleep(2.0)
            self._keyevent(4)
            time.sleep(3.0)
            ss = self.get_screenshot()
            if ss is not None and self.detector:
                try:
                    result = self.detector.detect(ss)
                    if result.state == 'lobby':
                        logger.info("[AUTO] Recovery succeeded: lobby detected")
                        self.transition_to('lobby')
                        return
                except Exception:
                    pass
            if self._recovery_attempts >= 3:
                logger.error("[AUTO] Recovery failed after 3 attempts, restarting app")
                self._restart_app()
        time.sleep(1.0)

    def run_cycle(self):
        cycle_start = time.time()
        self.cycle_count += 1
        screenshot = self.get_screenshot()
        if screenshot is None:
            time.sleep(1.0)
            return
        if self.cycle_count % 10 == 0:
            try:
                ts = datetime.now().strftime("%H%M%S")
                Image.fromarray(screenshot).save(f"auto_play_{ts}_{self.state}.png")
            except Exception:
                pass
        self._last_screenshot = screenshot.copy()
        detected_state, conf, button_coords = self.detect_state(screenshot)
        set_detection_confidence(method="state_detector", value=conf)
        self._update_smoothing(detected_state)
        handlers = {
            'lobby': lambda ss: self.handle_lobby(ss, button_coords),
            'loading': lambda ss: self.handle_loading(ss),
            'matchmaking': lambda ss: self.handle_matchmaking(ss, button_coords),
            'in_game': self.handle_in_game,
            'end': lambda ss: self.handle_end(ss, button_coords),
            'unknown': self.handle_unknown,
            'brawler_selection': self.handle_brawler_selection,
            'connection_lost': lambda ss: self.handle_connection_lost(ss, button_coords),
            'news': lambda ss: self.handle_news(ss, button_coords),
            'season_reset': lambda ss: self.handle_season_reset(ss, button_coords),
        }
        handler = handlers.get(self.state, self.handle_unknown)
        handler(screenshot)
        observe_cycle_duration(time.time() - cycle_start)
        if self.cycle_count % 5 == 0:
            elapsed = time.time() - self.state_start_time
            logger.info(
                f"[AUTO] Cycle {self.cycle_count:4d} | State: {self.state:12s} | "
                f"Conf: {conf:.2f} | Time: {elapsed:5.1f}s | "
                f"Plays: {self.play_clicks} | Combats: {self.combat_actions} | "
                f"Matches: {self.matches_completed} | Recoveries: {self.recovery_actions} | "
                f"Restarts: {self.forced_restarts}"
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
            logger.info("[AUTO] Interrupted by user")
        except Exception as e:
            logger.error(f"[AUTO] Fatal error: {e}")
            inc_errors(error_type="fatal")
            import traceback
            traceback.print_exc()
        self._print_summary()

    def _print_summary(self):
        logger.info("=" * 60)
        logger.info("AUTO PLAY BOT - RESUMO")
        logger.info(f"Ciclos: {self.cycle_count}")
        logger.info(f"Partidas completadas: {self.matches_completed}")
        logger.info(f"Acoes de recuperacao: {self.recovery_actions}")
        logger.info(f"Reinicios forcados: {self.forced_restarts}")
        logger.info(f"Plays clicados: {self.play_clicks}")
        logger.info(f"Acoes de combate: {self.combat_actions}")
        logger.info("=" * 60)
        print("\n" + "=" * 60)
        print("AUTO PLAY BOT - SHUTDOWN SUMMARY")
        print(f"Cycles: {self.cycle_count}")
        print(f"Matches completed: {self.matches_completed}")
        print(f"Recovery actions: {self.recovery_actions}")
        print(f"Forced restarts: {self.forced_restarts}")
        print("=" * 60)


def main():
    setup_logging()
    bot = AutoPlayBot()
    bot.run(duration=300)


if __name__ == '__main__':
    main()
