"""
test_lobby_live.py

Script de teste AO VIVO para a navegacao do lobby.
Conecta ao emulador via ADB e executa acoes de lobby:
1. Captura screenshot
2. Detecta estado atual
3. Fecha popups
4. Troca de brawler
5. Clica Play

Uso: py test_lobby_live.py [--step-by-step] [--loop N]
"""

import sys
import os
import time
import argparse
import subprocess
from pathlib import Path

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Fix encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LOBBY_TEST")

# --- ADB Controller (simplified) ---

ADB_PATH = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
DEVICE_ID = "emulator-5554"


def adb_cmd(*args, timeout=10):
    """Run ADB command and return output."""
    cmd = [ADB_PATH, "-s", DEVICE_ID] + list(args)
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return result.returncode, result.stdout.decode(errors="replace")
    except Exception as e:
        logger.error(f"ADB command failed: {e}")
        return -1, str(e)


def adb_tap(x, y):
    """Tap at coordinates."""
    logger.info(f"[ADB] Tap ({x}, {y})")
    rc, out = adb_cmd("shell", "input", "tap", str(x), str(y))
    return rc == 0


def adb_swipe(x1, y1, x2, y2, duration=300):
    """Swipe from (x1,y1) to (x2,y2)."""
    logger.info(f"[ADB] Swipe ({x1},{y1}) -> ({x2},{y2})")
    rc, out = adb_cmd("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration))
    return rc == 0


def adb_keyevent(keycode):
    """Send key event."""
    rc, out = adb_cmd("shell", "input", "keyevent", str(keycode))
    return rc == 0


def adb_back():
    """Press back button (keycode 4)."""
    logger.info("[ADB] Press BACK")
    return adb_keyevent(4)


def take_screenshot(filename="live_screenshot.png"):
    """Take screenshot from emulator and save to file."""
    try:
        result = subprocess.run(
            [ADB_PATH, "-s", DEVICE_ID, "exec-out", "screencap", "-p"],
            capture_output=True, timeout=15
        )
        if result.returncode == 0 and len(result.stdout) > 1000:
            with open(filename, "wb") as f:
                f.write(result.stdout)
            return filename
        else:
            logger.error(f"Screenshot failed: rc={result.returncode}, size={len(result.stdout)}")
            return None
    except Exception as e:
        logger.error(f"Screenshot error: {e}")
        return None


def load_screenshot(filename="live_screenshot.png"):
    """Load screenshot as numpy array."""
    try:
        import numpy as np
        from PIL import Image
        img = Image.open(filename).convert("RGB")
        arr = np.array(img)
        return arr
    except Exception as e:
        logger.error(f"Load screenshot error: {e}")
        return None


# --- State Detection (pixel-based) ---

def detect_state_simple(arr):
    """
    Simple pixel-based state detection for Brawl Stars at 1600x900.
    Returns: (state, confidence, details)
    """
    import numpy as np
    h, w = arr.shape[:2]

    details = {}

    # 1. Check for dark overlay (popup present)
    corner_brightness = arr[0:50, 0:50].mean()
    center_brightness = arr[h//3:2*h//3, w//3:2*w//3].mean()
    has_overlay = center_brightness > corner_brightness + 20
    details["has_overlay"] = has_overlay
    details["corner_brightness"] = float(corner_brightness)
    details["center_brightness"] = float(center_brightness)

    # 2. Check for green play button
    # Play button is typically a large green area in bottom-right
    green_mask = (arr[:,:,1] > 150) & (arr[:,:,0] < 150) & (arr[:,:,2] < 100)
    green_count = green_mask.sum()
    details["green_pixels"] = int(green_count)

    # Find green area center
    if green_count > 500:
        green_ys, green_xs = np.where(green_mask)
        green_cx = int((green_xs.min() + green_xs.max()) / 2)
        green_cy = int((green_ys.min() + green_ys.max()) / 2)
        details["green_center"] = (green_cx, green_cy)
        # Play button is expected at bottom-right (~94% x, ~89% y)
        # At 1600x900: x~1504, y~801
        is_play_button = green_cx > w * 0.7 and green_cy > h * 0.6
        details["is_play_button_position"] = is_play_button

    # 3. Check for blue background (lobby)
    blue_mask = (arr[:,:,2] > 120) & (arr[:,:,0] < 100) & (arr[:,:,1] < 150)
    blue_ratio = blue_mask.sum() / (h * w)
    details["blue_ratio"] = float(blue_ratio)

    # 4. Check for yellow/gold (trophies, shop items)
    yellow_mask = (arr[:,:,0] > 180) & (arr[:,:,1] > 150) & (arr[:,:,2] < 80)
    details["yellow_pixels"] = int(yellow_mask.sum())

    # 5. Check bottom bar area (navigation)
    bottom_bar = arr[h-80:h, :]
    bottom_avg = bottom_bar.mean(axis=(0,1))
    details["bottom_bar_avg"] = [float(x) for x in bottom_avg]

    # 6. Check for "PLAY" text area (bottom right)
    play_region = arr[int(h*0.82):int(h*0.95), int(w*0.85):int(w*0.98)]
    play_avg = play_region.mean(axis=(0,1))
    details["play_region_avg"] = [float(x) for x in play_avg]

    # 7. Check for red (defeat, shop specials)
    red_mask = (arr[:,:,0] > 150) & (arr[:,:,1] < 80) & (arr[:,:,2] < 80)
    details["red_pixels"] = int(red_mask.sum())

    # 8. Check for white text (common in menus)
    white_mask = (arr[:,:,0] > 220) & (arr[:,:,1] > 220) & (arr[:,:,2] > 220)
    details["white_pixels"] = int(white_mask.sum())

    # --- Determine state ---
    state = "unknown"
    confidence = 0.3

    if has_overlay and green_count < 1000:
        # Popup overlay detected
        state = "popup"
        confidence = 0.7
    elif blue_ratio > 0.15 and green_count > 500:
        # Blue background + green button = lobby
        state = "lobby"
        confidence = 0.8
    elif blue_ratio > 0.15 and green_count < 500:
        # Blue background but no green button
        # Could be lobby without play button visible, or menu
        state = "lobby_menu"
        confidence = 0.6
    elif green_count > 2000:
        # Lots of green - could be in-game (grass) or play button
        green_ys2, green_xs2 = np.where(green_mask)
        green_spread = (green_xs2.max() - green_xs2.min()) * (green_ys2.max() - green_ys2.min())
        if green_spread > 100000:
            # Green is spread out = in-game grass
            state = "in_game"
            confidence = 0.6
        else:
            # Green is concentrated = play button
            state = "lobby"
            confidence = 0.7
    elif red_mask.sum() > 50000:
        state = "defeat"
        confidence = 0.5
    elif yellow_mask.sum() > 50000:
        # Lots of yellow = could be victory/reward screen
        state = "victory"
        confidence = 0.5

    return state, confidence, details


# --- Lobby Actions ---

def close_popup(arr, step_by_step=False):
    """Try to close any popup detected on screen."""
    h, w = arr.shape[:2]

    # Check for X button in top-right corner
    x_region = arr[0:h//8, 7*w//8:w]
    import numpy as np

    # Look for close button (usually white/light X on dark overlay)
    white_in_corner = ((x_region[:,:,0] > 200) & (x_region[:,:,1] > 200) & (x_region[:,:,2] > 200)).sum()
    if white_in_corner > 50:
        # Click top-right area
        close_x = int(w * 0.94)
        close_y = int(h * 0.06)
        logger.info(f"[POPUP] Detected close button area, clicking ({close_x}, {close_y})")
        if step_by_step:
            input("  Press Enter to click close button...")
        adb_tap(close_x, close_y)
        time.sleep(0.8)
        return True

    # Try clicking center (for reward/claim popups)
    center_brightness = arr[h//3:2*h//3, w//3:2*w//3].mean()
    corner_brightness = arr[0:50, 0:50].mean()
    if center_brightness > corner_brightness + 20:
        logger.info("[POPUP] Overlay detected, clicking center to dismiss")
        if step_by_step:
            input("  Press Enter to click center...")
        adb_tap(w // 2, h // 2)
        time.sleep(0.8)
        return True

    # Try pressing back
    logger.info("[POPUP] No close button found, pressing Back")
    if step_by_step:
        input("  Press Enter to press Back...")
    adb_back()
    time.sleep(0.8)
    return True


def find_play_button(arr):
    """Find the Play button on screen. Returns (x, y) or None."""
    import numpy as np
    h, w = arr.shape[:2]

    # Method 1: Look for green area in bottom-right
    green_mask = (arr[:,:,1] > 150) & (arr[:,:,0] < 150) & (arr[:,:,2] < 100)
    green_count = green_mask.sum()

    if green_count > 300:
        green_ys, green_xs = np.where(green_mask)
        # Check if green area is in bottom-right (play button position)
        if green_xs.min() > w * 0.6 and green_ys.min() > h * 0.5:
            cx = int((green_xs.min() + green_xs.max()) / 2)
            cy = int((green_ys.min() + green_ys.max()) / 2)
            logger.info(f"[PLAY] Found green button at ({cx}, {cy})")
            return (cx, cy)

    # Method 2: Look for bright/white button area in bottom-right
    # Play button can also be white/bright
    play_region = arr[int(h*0.75):int(h*0.95), int(w*0.80):int(w*0.98)]
    brightness = play_region.mean()
    if brightness > 100:
        # Find brightest spot
        gray = np.mean(play_region, axis=2)
        max_pos = np.unravel_index(gray.argmax(), gray.shape)
        # Convert back to full image coordinates
        abs_x = int(w * 0.80) + max_pos[1]
        abs_y = int(h * 0.75) + max_pos[0]
        logger.info(f"[PLAY] Found bright spot at ({abs_x}, {abs_y})")
        return (abs_x, abs_y)

    # Method 3: Fallback to hardcoded coordinates
    # At 1600x900: play button at 94.19% x, 89.49% y
    fallback_x = int(w * 0.9419)
    fallback_y = int(h * 0.8949)
    logger.info(f"[PLAY] Using fallback coords ({fallback_x}, {fallback_y})")
    return (fallback_x, fallback_y)


def select_brawler(brawler_name="colt", step_by_step=False):
    """
    Navigate to brawler selection and pick a brawler.
    At 1600x900 resolution.
    """
    h, w = 900, 1600

    logger.info(f"[BRAWLER] Selecting brawler: {brawler_name}")

    # Step 1: Click on brawler icon/area (left side of lobby)
    # Brawler icon is typically in the center-left area
    brawler_icon_x = int(w * 0.35)
    brawler_icon_y = int(h * 0.50)
    logger.info(f"[BRAWLER] Clicking brawler icon at ({brawler_icon_x}, {brawler_icon_y})")
    if step_by_step:
        input("  Press Enter to click brawler icon...")
    adb_tap(brawler_icon_x, brawler_icon_y)
    time.sleep(1.5)

    # Step 2: Wait for brawler selection screen
    # Take screenshot to verify
    ss_file = take_screenshot("brawler_select.png")
    if ss_file:
        arr = load_screenshot(ss_file)
        if arr is not None:
            state, conf, details = detect_state_simple(arr)
            logger.info(f"[BRAWLER] After tap: state={state}, conf={conf:.2f}")

    # Step 3: Search for brawler in the grid
    # Brawler grid is typically in center of screen
    # We need to scroll through brawlers to find the right one

    # For now, just click in the brawler grid area to select
    # The grid center
    grid_cx = int(w * 0.50)
    grid_cy = int(h * 0.45)

    # Common brawler positions in the grid (approximate)
    # These are relative to the grid area
    brawler_positions = {
        "colt": (int(w * 0.40), int(h * 0.35)),
        "shelly": (int(w * 0.30), int(h * 0.30)),
        "nita": (int(w * 0.50), int(h * 0.30)),
        "bull": (int(w * 0.35), int(h * 0.45)),
        "jessie": (int(w * 0.45), int(h * 0.45)),
        "brock": (int(w * 0.55), int(h * 0.45)),
        "dynamike": (int(w * 0.40), int(h * 0.55)),
        "bo": (int(w * 0.50), int(h * 0.55)),
    }

    if brawler_name.lower() in brawler_positions:
        bx, by = brawler_positions[brawler_name.lower()]
        logger.info(f"[BRAWLER] Clicking {brawler_name} at ({bx}, {by})")
        if step_by_step:
            input(f"  Press Enter to click {brawler_name}...")
        adb_tap(bx, by)
        time.sleep(0.8)
    else:
        # Try clicking center of grid
        logger.info(f"[BRAWLER] Unknown brawler '{brawler_name}', clicking grid center")
        if step_by_step:
            input("  Press Enter to click grid center...")
        adb_tap(grid_cx, grid_cy)
        time.sleep(0.8)

    # Step 4: Click the selected brawler again to confirm (if needed)
    time.sleep(0.5)

    # Step 5: Press back to return to lobby
    logger.info("[BRAWLER] Pressing Back to return to lobby")
    if step_by_step:
        input("  Press Enter to press Back...")
    adb_back()
    time.sleep(1.0)

    return True


def click_play(arr, step_by_step=False):
    """Click the Play button."""
    play_pos = find_play_button(arr)
    if play_pos:
        x, y = play_pos
        logger.info(f"[PLAY] Clicking Play at ({x}, {y})")
        if step_by_step:
            input("  Press Enter to click Play...")
        adb_tap(x, y)
        time.sleep(1.5)
        return True
    return False


# --- Main Test Loop ---

def run_lobby_test(step_by_step=False, max_loops=5):
    """Main test loop for lobby navigation."""

    logger.info("=" * 60)
    logger.info("  LOBBY LIVE TEST")
    logger.info("  BlueStacks 1600x900 - ADB emulator-5554")
    logger.info("=" * 60)

    # Check ADB connection
    rc, out = adb_cmd("devices")
    if DEVICE_ID not in out:
        logger.error(f"Device {DEVICE_ID} not found! Make sure BlueStacks is running.")
        return False
    logger.info(f"ADB connected: {DEVICE_ID}")

    for loop_num in range(1, max_loops + 1):
        logger.info(f"\n{'='*40}")
        logger.info(f"  LOOP {loop_num}/{max_loops}")
        logger.info(f"{'='*40}")

        # 1. Take screenshot
        ss_file = take_screenshot(f"loop_{loop_num}_screenshot.png")
        if not ss_file:
            logger.error("Failed to take screenshot!")
            time.sleep(2)
            continue

        # 2. Load and analyze
        arr = load_screenshot(ss_file)
        if arr is None:
            logger.error("Failed to load screenshot!")
            time.sleep(2)
            continue

        h, w = arr.shape[:2]
        logger.info(f"Screenshot: {w}x{h}")

        # 3. Detect state
        state, confidence, details = detect_state_simple(arr)
        logger.info(f"State: {state} (confidence={confidence:.2f})")
        logger.info(f"  Details: green={details.get('green_pixels',0)}, blue_ratio={details.get('blue_ratio',0):.3f}, overlay={details.get('has_overlay',False)}")

        # 4. Act based on state
        if state == "popup":
            logger.info(">>> ACTION: Close popup")
            close_popup(arr, step_by_step)

        elif state == "lobby":
            logger.info(">>> ACTION: In lobby - clicking Play")
            if step_by_step:
                input("  Press Enter to click Play...")
            click_play(arr, step_by_step)

        elif state == "lobby_menu":
            logger.info(">>> ACTION: In lobby menu - looking for Play or navigating")
            # Try clicking Play button area
            play_pos = find_play_button(arr)
            if play_pos:
                click_play(arr, step_by_step)
            else:
                # Maybe we need to scroll or navigate
                # Try pressing back first
                logger.info("  Pressing Back to get to main lobby")
                adb_back()
                time.sleep(1.0)

        elif state == "in_game":
            logger.info(">>> ACTION: In game - waiting for match to end")
            logger.info("  (In live test, we just observe)")

        elif state == "defeat" or state == "victory":
            logger.info(f">>> ACTION: {state} screen - clicking continue")
            # Click center or "Play Again" button
            play_again_x = int(w * 0.5903)
            play_again_y = int(h * 0.9197)
            logger.info(f"  Clicking Play Again at ({play_again_x}, {play_again_y})")
            if step_by_step:
                input("  Press Enter to click Play Again...")
            adb_tap(play_again_x, play_again_y)
            time.sleep(1.5)

        else:
            logger.info(f">>> ACTION: Unknown state - pressing Back as recovery")
            adb_back()
            time.sleep(1.0)

        # Wait before next loop
        wait_time = 3 if not step_by_step else 1
        logger.info(f"Waiting {wait_time}s before next check...")
        time.sleep(wait_time)

    logger.info("\n" + "=" * 60)
    logger.info("  LOBBY TEST COMPLETE")
    logger.info("=" * 60)


def interactive_mode():
    """Interactive mode - manual control of each action."""

    logger.info("=" * 60)
    logger.info("  INTERACTIVE LOBBY TEST")
    logger.info("  Commands: s=screenshot, p=play, b=brawler, x=back,")
    logger.info("            t=X,Y=tap, w=wait, q=quit, a=auto-detect")
    logger.info("=" * 60)

    # Check ADB
    rc, out = adb_cmd("devices")
    if DEVICE_ID not in out:
        logger.error(f"Device {DEVICE_ID} not found!")
        return

    current_brawler = "colt"

    while True:
        cmd = input("\n> ").strip().lower()

        if cmd == "q" or cmd == "quit":
            break

        elif cmd == "s":
            ss_file = take_screenshot()
            if ss_file:
                arr = load_screenshot(ss_file)
                if arr is not None:
                    state, conf, details = detect_state_simple(arr)
                    logger.info(f"State: {state} (conf={conf:.2f})")
                    logger.info(f"Green: {details.get('green_pixels',0)}, Blue: {details.get('blue_ratio',0):.3f}, Overlay: {details.get('has_overlay',False)}")
                    if 'green_center' in details:
                        logger.info(f"Green center: {details['green_center']}")

        elif cmd == "p":
            ss_file = take_screenshot()
            if ss_file:
                arr = load_screenshot(ss_file)
                if arr is not None:
                    click_play(arr)

        elif cmd == "b":
            name = input("Brawler name (default=colt): ").strip() or "colt"
            select_brawler(name, step_by_step=True)

        elif cmd == "x":
            adb_back()

        elif cmd.startswith("t="):
            try:
                coords = cmd[2:].split(",")
                x, y = int(coords[0]), int(coords[1])
                adb_tap(x, y)
            except Exception as e:
                logger.error(f"Invalid tap coords: {e}")

        elif cmd == "w":
            secs = input("Wait seconds (default=2): ").strip()
            time.sleep(float(secs) if secs else 2)

        elif cmd == "a":
            # Auto-detect and act
            ss_file = take_screenshot()
            if ss_file:
                arr = load_screenshot(ss_file)
                if arr is not None:
                    state, conf, details = detect_state_simple(arr)
                    logger.info(f"State: {state} (conf={conf:.2f})")

                    if state == "popup":
                        close_popup(arr)
                    elif state == "lobby":
                        click_play(arr)
                    elif state == "defeat" or state == "victory":
                        h, w = arr.shape[:2]
                        adb_tap(int(w * 0.5903), int(h * 0.9197))
                    else:
                        adb_back()

        elif cmd == "loop":
            n = input("How many loops (default=5): ").strip()
            run_lobby_test(step_by_step=False, max_loops=int(n) if n else 5)

        else:
            logger.info("Unknown command. s=screenshot, p=play, b=brawler, x=back, t=X,Y=tap, w=wait, a=auto, loop=auto-loop, q=quit")

    logger.info("Interactive mode ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Lobby Test for Brawl Stars Bot")
    parser.add_argument("--step-by-step", action="store_true", help="Pause between each action")
    parser.add_argument("--loop", type=int, default=0, help="Run N auto-loops (0=interactive)")
    parser.add_argument("--interactive", action="store_true", help="Interactive command mode")
    args = parser.parse_args()

    if args.loop > 0:
        run_lobby_test(step_by_step=args.step_by_step, max_loops=args.loop)
    else:
        interactive_mode()
