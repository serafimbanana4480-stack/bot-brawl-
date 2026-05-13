"""
lobby_runner.py

Script funcional para navegar o lobby do Brawl Stars ao vivo.
Usa coordenadas mapeadas em 1600x900 (BlueStacks).

Fluxo:
1. Detectar estado atual (lobby, popup, mode select, loading, in_game)
2. Fechar popups
3. Clicar Play -> selecionar modo -> clicar Fight
4. Trocar brawler (com deteccao de locked via HSV)
5. Iniciar partida
6. Farm ate trofeus alvo por brawler

Uso: py lobby_runner.py [--loop N] [--brawler NAME] [--debug]
     py lobby_runner.py --farm colt:500,shelly:400,nita:300
"""

import sys
import os
import time
import argparse
import subprocess
import random
from pathlib import Path

# Fix encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LOBBY")

# --- ADB Controller ---

ADB_PATH = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
DEVICE_ID = "emulator-5554"

# --- Coordinates (percentage-based for any resolution) ---
# Mapped live on BlueStacks 1600x900, May 2025

COORDS = {
    # Play button in nav bar
    "play_btn": (0.9119, 0.9122),
    # Fight button on mode selection screen
    "fight_btn": (0.4950, 0.9356),
    # Brawler icon in lobby
    "brawler_icon": (0.3494, 0.4489),
    # Close popup (X button top-right)
    "close_popup": (0.94, 0.06),
    # Center of screen (for dismissing popups)
    "center": (0.50, 0.50),
    # Play Again button (after match)
    "play_again": (0.5903, 0.9197),
    # Exit button (after defeat)
    "exit_btn": (0.493, 0.9187),
    # Proceed button
    "proceed": (0.8093, 0.9165),
    # Navigation tabs (approximate positions)
    "nav_social": (0.035, 0.86),
    "nav_shop": (0.089, 0.86),
    "nav_brawlers": (0.332, 0.86),
    "nav_events": (0.512, 0.86),
    "nav_play": (0.9119, 0.9122),
}


class LobbyRunner:
    """Live lobby navigation for Brawl Stars."""

    def __init__(self, resolution=(1600, 900), debug=False):
        self.w, self.h = resolution
        self.debug = debug
        self._last_state = None
        self._state_count = 0
        self._match_count = 0

    def _adb(self, *args, timeout=10):
        """Run ADB command."""
        cmd = [ADB_PATH, "-s", DEVICE_ID] + list(args)
        try:
            return subprocess.run(cmd, capture_output=True, timeout=timeout)
        except Exception as e:
            logger.error(f"ADB error: {e}")
            return None

    def tap(self, x, y):
        """Tap at absolute coordinates."""
        logger.info(f"TAP ({x}, {y})")
        self._adb("shell", "input", "tap", str(x), str(y))
        time.sleep(random.uniform(0.3, 0.6))

    def tap_pct(self, x_pct, y_pct):
        """Tap at percentage coordinates."""
        x = round(self.w * x_pct)
        y = round(self.h * y_pct)
        self.tap(x, y)

    def tap_named(self, name):
        """Tap a named coordinate."""
        if name in COORDS:
            x_pct, y_pct = COORDS[name]
            self.tap_pct(x_pct, y_pct)
        else:
            logger.error(f"Unknown coordinate: {name}")

    def back(self):
        """Press Back button."""
        logger.info("BACK")
        self._adb("shell", "input", "keyevent", "4")
        time.sleep(random.uniform(0.5, 1.0))

    def home(self):
        """Press Home button."""
        logger.info("HOME")
        self._adb("shell", "input", "keyevent", "3")
        time.sleep(1.0)

    def swipe(self, x1, y1, x2, y2, duration=300):
        """Swipe from (x1,y1) to (x2,y2)."""
        logger.info(f"SWIPE ({x1},{y1}) -> ({x2},{y2})")
        self._adb("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration))
        time.sleep(0.5)

    def screenshot(self):
        """Take screenshot and return as numpy array."""
        try:
            result = subprocess.run(
                [ADB_PATH, "-s", DEVICE_ID, "exec-out", "screencap", "-p"],
                capture_output=True, timeout=15
            )
            if result.stdout and len(result.stdout) > 1000:
                from PIL import Image
                import numpy as np
                import io
                img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
                return np.array(img)
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
        return None

    def detect_state(self, arr=None):
        """
        Detect current game state from screenshot.
        Returns: (state, details)
        States: lobby, popup, mode_select, loading, in_game, victory, defeat, unknown
        """
        import numpy as np

        if arr is None:
            arr = self.screenshot()
        if arr is None:
            return "unknown", {}

        h, w = arr.shape[:2]
        details = {}

        # Calculate key metrics
        blue_mask = (arr[:,:,2] > 150) & (arr[:,:,0] < 50)
        blue_ratio = blue_mask.sum() / (h * w)
        details["blue_ratio"] = float(blue_ratio)

        dark_mask = arr.max(axis=2) < 50
        dark_ratio = dark_mask.sum() / (h * w)
        details["dark_ratio"] = float(dark_ratio)

        green_mask = (arr[:,:,1] > 100) & (arr[:,:,0] < 50) & (arr[:,:,2] < 50)
        green_ratio = green_mask.sum() / (h * w)
        details["green_ratio"] = float(green_ratio)

        # Check for victory (gold/yellow dominant) or defeat (red dominant)
        gold_mask = (arr[:,:,0] > 150) & (arr[:,:,1] > 120) & (arr[:,:,2] < 80)
        gold_ratio = gold_mask.sum() / (h * w)
        details["gold_ratio"] = float(gold_ratio)

        red_mask = (arr[:,:,0] > 150) & (arr[:,:,1] < 60) & (arr[:,:,2] < 60)
        red_ratio = red_mask.sum() / (h * w)
        details["red_ratio"] = float(red_ratio)

        # Check for overlay (popup)
        corner_brightness = arr[0:50, 0:50].mean()
        center_brightness = arr[h//3:2*h//3, w//3:2*w//3].mean()
        has_overlay = center_brightness > corner_brightness + 20
        details["has_overlay"] = has_overlay

        # Check nav bar area for play button
        nav_area = arr[int(h*0.72):int(h*0.98), int(w*0.80):]
        nav_brightness = nav_area.mean()
        details["nav_brightness"] = float(nav_brightness)

        # Determine state
        if gold_ratio > 0.15:
            state = "victory"
        elif red_ratio > 0.15:
            state = "defeat"
        elif dark_ratio > 0.5:
            state = "loading"
        elif green_ratio > 0.10:
            state = "in_game"
        elif has_overlay and blue_ratio < 0.15:
            state = "popup"
        elif blue_ratio > 0.25:
            # Blue background = lobby or mode select
            # Mode select has less blue and more varied colors
            if blue_ratio > 0.40:
                state = "lobby"
            else:
                state = "mode_select"
        elif dark_ratio > 0.2:
            state = "loading"
        else:
            state = "unknown"

        return state, details

    def handle_popup(self):
        """Close any popup on screen."""
        logger.info("Handling popup...")
        # Try X button first
        self.tap_named("close_popup")
        time.sleep(1.0)

        # Check if popup is gone
        state, _ = self.detect_state()
        if state != "popup":
            logger.info("Popup closed!")
            return True

        # Try clicking center
        self.tap_named("center")
        time.sleep(1.0)

        state, _ = self.detect_state()
        if state != "popup":
            logger.info("Popup dismissed!")
            return True

        # Try back
        self.back()
        time.sleep(1.0)
        return True

    def navigate_to_play(self):
        """
        Full flow: lobby -> click Play -> click Fight -> match starts.
        Returns True if match started, False otherwise.
        """
        logger.info("=== Starting Play Flow ===")

        # Step 1: Ensure we're in lobby
        state, details = self.detect_state()
        logger.info(f"Current state: {state}")

        if state == "popup":
            self.handle_popup()
            time.sleep(1.0)
            state, _ = self.detect_state()

        if state == "loading":
            logger.info("Already loading, waiting...")
            return self._wait_for_match_start()

        if state == "in_game":
            logger.info("Already in game!")
            return True

        if state == "lobby":
            # Step 2: Click Play button
            logger.info("Clicking Play button...")
            self.tap_named("play_btn")
            time.sleep(2.0)

            # Step 3: Check if we're on mode selection
            state, _ = self.detect_state()
            logger.info(f"After Play: state={state}")

            if state == "mode_select" or state == "lobby":
                # Step 4: Click Fight button
                logger.info("Clicking Fight button...")
                self.tap_named("fight_btn")
                time.sleep(2.0)

                # Step 5: Wait for match to start
                return self._wait_for_match_start()
            elif state == "loading":
                return self._wait_for_match_start()
            else:
                logger.warning(f"Unexpected state after Play: {state}")
                self.back()
                return False

        elif state == "mode_select":
            # Already on mode selection, click Fight
            logger.info("Already on mode select, clicking Fight...")
            self.tap_named("fight_btn")
            time.sleep(2.0)
            return self._wait_for_match_start()

        else:
            logger.warning(f"Cannot start play from state: {state}")
            # Try pressing back to get to lobby
            self.back()
            time.sleep(1.0)
            return False

    def _wait_for_match_start(self, timeout=30):
        """Wait for match to start loading."""
        logger.info("Waiting for match to start...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            state, details = self.detect_state()

            if state == "in_game":
                logger.info("Match started!")
                self._match_count += 1
                return True

            if state == "loading":
                logger.debug("Loading...")
                time.sleep(1.0)
                continue

            if state == "lobby":
                # Matchmaking might have failed
                logger.warning("Back in lobby - matchmaking may have failed")
                return False

            time.sleep(1.0)

        logger.warning("Timeout waiting for match")
        return False

    def select_brawler(self, brawler_name="colt"):
        """
        Navigate to brawler selection and pick a brawler.
        Uses HSV saturation to avoid locked brawlers.
        """
        import cv2
        import numpy as np

        logger.info(f"Selecting brawler: {brawler_name}")

        # Step 1: Click brawler icon in lobby
        self.tap_named("brawler_icon")
        time.sleep(2.0)

        # Step 2: Take screenshot of brawler selection screen
        arr = self.screenshot()
        if arr is not None:
            # Detect brawler portraits using circle detection
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=80,
                param1=50, param2=25, minRadius=25, maxRadius=60
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                unlocked_brawlers = []
                locked_brawlers = []

                for c in circles[0]:
                    cx, cy, radius = c
                    # Check saturation to determine if locked
                    r1 = max(0, cy - radius)
                    r2 = min(arr.shape[0], cy + radius)
                    c1 = max(0, cx - radius)
                    c2 = min(arr.shape[1], cx + radius)
                    region = arr[r1:r2, c1:c2]
                    if region.size == 0:
                        continue
                    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
                    avg_sat = hsv[:, :, 1].mean()
                    avg_val = hsv[:, :, 2].mean()

                    is_unlocked = avg_sat > 40
                    entry = {"x": int(cx), "y": int(cy), "radius": int(radius),
                             "sat": float(avg_sat), "val": float(avg_val)}

                    if is_unlocked:
                        unlocked_brawlers.append(entry)
                    else:
                        locked_brawlers.append(entry)

                logger.info(f"Detected {len(unlocked_brawlers)} unlocked, {len(locked_brawlers)} locked brawlers")

                if locked_brawlers:
                    for b in locked_brawlers:
                        logger.info(f"  LOCKED: ({b['x']},{b['y']}) sat={b['sat']:.0f} val={b['val']:.0f}")

                if unlocked_brawlers:
                    # Pick the first unlocked brawler (closest to top-left)
                    target = min(unlocked_brawlers, key=lambda b: b['x'] + b['y'])
                    logger.info(f"Clicking unlocked brawler at ({target['x']},{target['y']}) sat={target['sat']:.0f}")
                    self.tap(target['x'], target['y'])
                    time.sleep(0.8)
                else:
                    logger.warning("No unlocked brawlers found! Clicking center of grid")
                    self.tap_pct(0.40, 0.35)
                    time.sleep(0.8)
            else:
                logger.warning("No brawler portraits detected, clicking center")
                self.tap_pct(0.40, 0.35)
                time.sleep(0.8)
        else:
            logger.warning("Screenshot failed, clicking approximate position")
            self.tap_pct(0.40, 0.35)
            time.sleep(0.8)

        # Step 3: Go back to lobby
        self.back()
        time.sleep(1.0)

        logger.info(f"Brawler selected")
        return True

    def run_loop(self, max_loops=5, brawler="colt", farm_config=None):
        """
        Main loop: navigate lobby and start matches.
        
        farm_config: dict {brawler_name: target_trophies}
                     When set, farms each brawler until target trophies.
        """
        logger.info("=" * 50)
        logger.info("  LOBBY RUNNER - Live Test")
        logger.info(f"  Resolution: {self.w}x{self.h}")
        logger.info(f"  Max loops: {max_loops}")
        if farm_config:
            logger.info(f"  Farm mode: {farm_config}")
        logger.info("=" * 50)

        # Check ADB connection
        result = self._adb("devices")
        if not result or DEVICE_ID not in result.stdout.decode():
            logger.error(f"Device {DEVICE_ID} not found!")
            return False

        # Farm mode: track trophies per brawler
        if farm_config:
            return self._run_farm_loop(farm_config, max_loops)

        for loop in range(1, max_loops + 1):
            logger.info(f"\n--- Loop {loop}/{max_loops} ---")

            # Detect current state
            state, details = self.detect_state()
            logger.info(f"State: {state} (blue={details.get('blue_ratio',0):.3f}, dark={details.get('dark_ratio',0):.3f})")

            # Handle based on state
            if state == "popup":
                self.handle_popup()

            elif state == "lobby":
                # Select brawler on first loop
                if loop == 1 and brawler:
                    self.select_brawler(brawler)
                    time.sleep(1.0)

                # Start match
                success = self.navigate_to_play()
                if success:
                    logger.info(f"Match #{self._match_count} started!")
                    # Wait for match to end
                    self._wait_for_match_end()
                else:
                    logger.warning("Failed to start match, retrying...")

            elif state == "mode_select":
                self.tap_named("fight_btn")
                time.sleep(2.0)

            elif state == "in_game":
                logger.info("In game - waiting for match to end...")
                self._wait_for_match_end()

            elif state == "loading":
                logger.info("Loading - waiting...")
                time.sleep(5.0)

            elif state == "victory" or state == "defeat":
                # Click continue/play again
                self.tap_named("play_again")
                time.sleep(2.0)

            else:
                logger.warning(f"Unknown state: {state}")
                self.back()
                time.sleep(1.0)

            # Brief pause between loops
            time.sleep(2.0)

        logger.info(f"\n{'='*50}")
        logger.info(f"  COMPLETE: {self._match_count} matches started")
        logger.info(f"{'='*50}")

    def _run_farm_loop(self, farm_config: dict, max_loops=50):
        """
        Farm mode: rotate through brawlers until each reaches target trophies.
        farm_config: {"colt": 500, "shelly": 400, "nita": 300}
        """
        import json

        # Load/save trophy tracking
        trophy_file = Path("data/farm_trophies.json")
        trophy_file.parent.mkdir(exist_ok=True)
        
        if trophy_file.exists():
            with open(trophy_file, encoding="utf-8") as f:
                trophy_data = json.load(f)
        else:
            trophy_data = {}

        # Initialize tracking for each brawler
        for name in farm_config:
            if name not in trophy_data:
                trophy_data[name] = {
                    "current_trophies": 0,
                    "matches_played": 0,
                    "wins": 0,
                    "losses": 0,
                    "target_trophies": farm_config[name]
                }

        # Save initial state
        with open(trophy_file, "w", encoding="utf-8") as f:
            json.dump(trophy_data, f, indent=2)

        # Build active brawler list (those that haven't reached target)
        brawler_order = list(farm_config.keys())
        current_brawler_idx = 0

        for loop in range(1, max_loops + 1):
            # Check which brawlers still need farming
            active_brawlers = []
            for name in brawler_order:
                current = trophy_data[name]["current_trophies"]
                target = farm_config[name]
                if current < target:
                    active_brawlers.append(name)
                else:
                    logger.info(f"[FARM] {name} atingiu meta: {current}/{target} trofeus!")

            if not active_brawlers:
                logger.info("[FARM] Todos os brawlers atingiram a meta! Farm concluido.")
                break

            # Pick current brawler (rotate through active ones)
            current_name = active_brawlers[current_brawler_idx % len(active_brawlers)]
            current_target = farm_config[current_name]
            current_trophies = trophy_data[current_name]["current_trophies"]

            logger.info(f"\n--- Farm Loop {loop}/{max_loops} ---")
            logger.info(f"[FARM] Brawler: {current_name} | Trofeus: {current_trophies}/{current_target}")

            # Detect state
            state, details = self.detect_state()
            logger.info(f"State: {state}")

            if state == "popup":
                self.handle_popup()

            elif state == "lobby":
                # Select brawler
                self.select_brawler(current_name)
                time.sleep(1.0)

                # Start match
                success = self.navigate_to_play()
                if success:
                    logger.info(f"[FARM] Partida #{self._match_count} iniciada com {current_name}")
                    # Wait for match to end
                    result = self._wait_for_match_end()
                    
                    # Update stats
                    trophy_data[current_name]["matches_played"] += 1
                    
                    # Check result - take screenshot to detect victory/defeat
                    end_state, end_details = self.detect_state()
                    if end_state == "victory":
                        trophy_data[current_name]["wins"] += 1
                        trophy_data[current_name]["current_trophies"] = max(0, current_trophies + 3)
                        logger.info(f"[FARM] VITORIA! {current_name}: {current_trophies} -> {trophy_data[current_name]['current_trophies']} trofeus")
                        self.tap_named("play_again")
                        time.sleep(2.0)
                    elif end_state == "defeat":
                        trophy_data[current_name]["losses"] += 1
                        trophy_data[current_name]["current_trophies"] = max(0, current_trophies - 3)
                        logger.info(f"[FARM] DERROTA. {current_name}: {current_trophies} -> {trophy_data[current_name]['current_trophies']} trofeus")
                        self.tap_named("exit_btn")
                        time.sleep(2.0)
                    elif end_state == "lobby":
                        # Returned to lobby without victory/defeat screen
                        # Could be matchmaking failure - assume no trophy change
                        logger.info(f"[FARM] Voltou ao lobby sem tela de resultado. Sem mudanca de trofeus.")
                    else:
                        # Unknown result, try to dismiss and assume slight gain
                        logger.info(f"[FARM] Estado pos-partida: {end_state}. Assumindo +1 trofeu.")
                        trophy_data[current_name]["current_trophies"] = max(0, current_trophies + 1)
                        self.tap_named("play_again")
                        time.sleep(2.0)

                    # Save progress
                    with open(trophy_file, "w", encoding="utf-8") as f:
                        json.dump(trophy_data, f, indent=2)

                    # Move to next brawler
                    current_brawler_idx += 1
                else:
                    logger.warning("[FARM] Falha ao iniciar partida, tentando novamente")
                    # Don't increment idx - retry same brawler next loop

            elif state == "mode_select":
                self.tap_named("fight_btn")
                time.sleep(2.0)

            elif state == "in_game":
                logger.info("[FARM] Em jogo - esperando terminar...")
                self._wait_for_match_end()

            elif state == "loading":
                time.sleep(5.0)

            else:
                logger.warning(f"[FARM] Estado desconhecido: {state}")
                self.back()
                time.sleep(1.0)

            time.sleep(2.0)

        # Final summary
        logger.info(f"\n{'='*50}")
        logger.info("  FARM SUMMARY")
        logger.info(f"{'='*50}")
        for name in brawler_order:
            data = trophy_data[name]
            target = farm_config[name]
            status = "CONCLUIDO" if data["current_trophies"] >= target else "EM ANDAMENTO"
            logger.info(f"  {name}: {data['current_trophies']}/{target} trofeus "
                        f"({data['wins']}V/{data['losses']}D) [{status}]")
        logger.info(f"  Total: {self._match_count} partidas")
        logger.info(f"{'='*50}")

    def _wait_for_match_end(self, timeout=180):
        """Wait for current match to end."""
        logger.info("Waiting for match to end...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            state, _ = self.detect_state()

            if state == "lobby" or state == "victory" or state == "defeat":
                logger.info("Match ended!")
                return True

            if state == "popup":
                self.handle_popup()

            time.sleep(3.0)

        logger.warning("Match timeout")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brawl Stars Lobby Runner")
    parser.add_argument("--loop", type=int, default=3, help="Number of loops")
    parser.add_argument("--brawler", type=str, default="colt", help="Brawler name")
    parser.add_argument("--farm", type=str, default=None,
                        help="Farm mode: brawler:trofeus,brawler:trofeus (ex: colt:500,shelly:400,nita:300)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    runner = LobbyRunner(debug=args.debug)

    if args.farm:
        # Parse farm config: "colt:500,shelly:400,nita:300"
        farm_config = {}
        for entry in args.farm.split(","):
            parts = entry.strip().split(":")
            if len(parts) == 2:
                name = parts[0].strip()
                target = int(parts[1].strip())
                farm_config[name] = target
            else:
                logger.warning(f"Invalid farm entry: {entry}")
        if farm_config:
            logger.info(f"Farm mode: {farm_config}")
            runner.run_loop(max_loops=args.loop * 10, farm_config=farm_config)
        else:
            logger.error("No valid farm config parsed")
    else:
        runner.run_loop(max_loops=args.loop, brawler=args.brawler)
