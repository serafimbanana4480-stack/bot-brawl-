"""
screen_automation.py

Lobby and match automation based on BrawlStarsBot screendetect module.
Uses pixel-matching to detect game states and click appropriate buttons.

Integrated from: https://github.com/Jooi025/BrawlStarsBot
"""

import pyautogui as py
from threading import Thread, Lock
from time import sleep
import logging

logger = logging.getLogger(__name__)


class ScreenState:
    IDLE = 0
    DETECT = 1
    EXIT = 2
    PLAY_AGAIN = 3
    LOAD = 4
    CONNECTION = 5
    PLAY = 6
    PROCEED = 7
    STARDROP = 8


class ScreenAutomation:
    """
    Automates lobby actions using pixel color matching.
    Detects: play button, play again, defeated, loading, proceed, star drop.
    """

    # RGB reference colors (from BrawlStarsBot)
    _DEFEATED_COLOR = (62, 0, 0)
    _PLAY_COLOR = (224, 186, 8)
    _LOAD_COLOR = (0, 1, 0)
    _PROCEED_COLOR = (35, 115, 255)
    _CONNECTION_LOST_COLOR = (66, 66, 66)
    _STAR_DROP_COLOR = (222, 72, 227)

    def __init__(self, window_w: int, window_h: int, offset_x: int = 0, offset_y: int = 0):
        self.state = ScreenState.DETECT
        self.lock = Lock()
        self.w = window_w
        self.h = window_h
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.stopped = True
        self.bot_stopped = True

        # Compute absolute coordinates based on window size
        self._compute_coordinates()

    def _compute_coordinates(self):
        """Pre-compute button coordinates relative to window size."""
        ox, oy = self.offset_x, self.offset_y
        w, h = self.w, self.h

        self.defeated1 = (round(w * 0.9656) + ox, round(h * 0.152) + oy)
        self.defeated2 = (round(w * 0.993) + ox, round(h * 0.2046) + oy)

        self.star_drop1 = (round(w * 0.488) + ox, round(h * 0.9303) + oy)
        self.star_drop2 = (round(w * 0.5228) + ox, round(h * 0.9296) + oy)

        self.play_again_button = (round(w * 0.5903) + ox, round(h * 0.9197) + oy)
        self.play_button = (round(w * 0.9119) + ox, round(h * 0.9122) + oy)
        self.exit_button = (round(w * 0.493) + ox, round(h * 0.9187) + oy)
        self.load_button = (round(w * 0.8057) + ox, round(h * 0.9675) + oy)
        self.proceed_button = (round(w * 0.8093) + ox, round(h * 0.9165) + oy)

        self.connection_lost_cord = (round(w * 0.4912) + ox, round(h * 0.5525) + oy)
        self.reload_button = (round(w * 0.2824) + ox, round(h * 0.5812) + oy)

    def update_window(self, w: int, h: int, offset_x: int = 0, offset_y: int = 0):
        """Update window dimensions (e.g. after resize)."""
        self.w = w
        self.h = h
        self.offset_x = offset_x
        self.offset_y = offset_y
        self._compute_coordinates()

    def set_bot_stopped(self, stopped: bool):
        self.bot_stopped = stopped

    def start(self):
        self.stopped = False
        t = Thread(target=self.run, daemon=True, name="screen-automation")
        t.start()
        logger.info("ScreenAutomation started")

    def stop(self):
        self.stopped = True
        logger.info("ScreenAutomation stopped")

    def run(self):
        while not self.stopped:
            sleep(0.05)
            try:
                if self.state == ScreenState.IDLE:
                    sleep(2)
                    self._set_state(ScreenState.DETECT)

                elif self.state == ScreenState.DETECT:
                    self._detect_state()

                elif self.state == ScreenState.PLAY_AGAIN:
                    self._click(self.play_again_button)
                    self._set_state(ScreenState.IDLE)

                elif self.state == ScreenState.LOAD:
                    sleep(3)  # wait for game to load
                    self._set_state(ScreenState.IDLE)

                elif self.state == ScreenState.EXIT:
                    sleep(4)
                    self._click(self.exit_button)
                    self._set_state(ScreenState.IDLE)

                elif self.state == ScreenState.CONNECTION:
                    sleep(20)
                    self._click(self.reload_button)
                    self._set_state(ScreenState.IDLE)

                elif self.state == ScreenState.PLAY:
                    self._click(self.play_button)
                    self._set_state(ScreenState.IDLE)

                elif self.state == ScreenState.PROCEED:
                    sleep(0.5)
                    self._click(self.proceed_button, clicks=2)
                    sleep(0.5)
                    self._set_state(ScreenState.IDLE)

                elif self.state == ScreenState.STARDROP:
                    py.press("e", presses=5)
                    sleep(6)
                    py.press("e")
                    self._set_state(ScreenState.IDLE)

            except Exception as e:
                logger.debug(f"ScreenAutomation error: {e}")
                sleep(0.5)

    def _set_state(self, new_state: int):
        with self.lock:
            self.state = new_state

    def _click(self, coords, clicks=1):
        x, y = coords
        py.click(x=x, y=y, button="left", clicks=clicks)
        sleep(0.05)

    def _detect_state(self):
        """Check pixel colors to determine game state."""
        try:
            # Play Again button
            if self._pixel_match(self.play_again_button, self._PLAY_COLOR, tolerance=15):
                logger.info("ScreenAutomation: PLAY AGAIN detected")
                self._set_state(ScreenState.PLAY_AGAIN)
                return

            # Loading into match
            if self._pixel_match(self.load_button, self._LOAD_COLOR, tolerance=30):
                logger.info("ScreenAutomation: LOADING detected")
                self._set_state(ScreenState.LOAD)
                return

            # Defeated (red corners)
            if (self._pixel_match(self.defeated1, self._DEFEATED_COLOR, tolerance=15)
                    or self._pixel_match(self.defeated2, self._DEFEATED_COLOR, tolerance=15)) \
                    and not self.bot_stopped:
                logger.info("ScreenAutomation: DEFEATED detected")
                self._set_state(ScreenState.EXIT)
                return

            # Star drop
            if (self._pixel_match(self.star_drop1, self._STAR_DROP_COLOR, tolerance=15)
                    or self._pixel_match(self.star_drop2, self._STAR_DROP_COLOR, tolerance=15)):
                logger.info("ScreenAutomation: STAR DROP detected")
                self._set_state(ScreenState.STARDROP)
                return

            # Play button (main menu)
            if self._pixel_match(self.play_button, self._PLAY_COLOR, tolerance=15):
                logger.info("ScreenAutomation: PLAY button detected")
                self._set_state(ScreenState.PLAY)
                return

            # Proceed button
            if self._pixel_match(self.proceed_button, self._PROCEED_COLOR, tolerance=25):
                logger.info("ScreenAutomation: PROCEED detected")
                self._set_state(ScreenState.PROCEED)
                return

        except OSError:
            pass  # Window may be minimized or off-screen

    @staticmethod
    def _pixel_match(coords, expected_rgb, tolerance=10):
        """Check if pixel at coords matches expected RGB within tolerance."""
        try:
            return py.pixelMatchesColor(coords[0], coords[1], expected_rgb, tolerance=tolerance)
        except Exception:
            return False

    def get_current_state_name(self) -> str:
        """Return human-readable current state."""
        names = {
            ScreenState.IDLE: "idle",
            ScreenState.DETECT: "detecting",
            ScreenState.EXIT: "exiting",
            ScreenState.PLAY_AGAIN: "play_again",
            ScreenState.LOAD: "loading",
            ScreenState.CONNECTION: "connection_lost",
            ScreenState.PLAY: "play",
            ScreenState.PROCEED: "proceed",
            ScreenState.STARDROP: "star_drop",
        }
        return names.get(self.state, "unknown")
