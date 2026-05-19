"""
test_lobby_flow.py

Teste de integração para verificar se o fluxo lobby -> jogo funciona.
Simula o ciclo do StateManager sem iniciar threads.
"""

import sys
sys.path.insert(0, 'c:/Users/rodri/Desktop/bot brawl')

import time
import numpy as np
from PIL import Image
from pathlib import Path

from pylaai_real.unified_state_detector import UnifiedStateDetector
from pylaai_real.lobby_automator import LobbyAutomator, BrawlerQueue, BrawlerConfig

# Mock emulator controller
class MockEmulatorController:
    def __init__(self):
        self.taps = []
        self.keys = []
    def tap_scaled(self, x, y):
        self.taps.append((x, y))
        print(f"[MOCK] Tap em ({x}, {y})")
    def keyevent(self, code):
        self.keys.append(code)
        print(f"[MOCK] Keyevent {code}")
    def swipe_scaled(self, x1, y1, x2, y2, duration=300):
        print(f"[MOCK] Swipe ({x1},{y1}) -> ({x2},{y2})")

# Load screenshot
screenshot = np.array(Image.open('c:/Users/rodri/Desktop/bot brawl/screenshot_current.png'))
print(f"Screenshot: {screenshot.shape}")

# Test unified detector
det = UnifiedStateDetector(images_path=Path('c:/Users/rodri/Desktop/bot brawl/images'))
result = det.detect(screenshot)
print(f"\n[UNIFIED DETECTOR]")
print(f"  State: {result.state}")
print(f"  Confidence: {result.confidence:.2f}")
print(f"  Button: {result.button_coords}")

# Test lobby automator
queue = BrawlerQueue()
queue.add_brawler(BrawlerConfig(name="shelly", target_trophies=100))

ctrl = MockEmulatorController()
lobby = LobbyAutomator(
    queue, ctrl,
    window_w=1920, window_h=1080,
    images_path=str(Path('c:/Users/rodri/Desktop/bot brawl/images')),
)
lobby.set_state_detector(det)
lobby.set_screenshot_func(lambda: screenshot)

print("\n[LOBBY AUTOMATOR]")
print("Testing press_play...")
success = lobby.press_play()
print(f"  press_play result: {success}")
print(f"  Taps performed: {ctrl.taps}")
print(f"  Keys performed: {ctrl.keys}")

# Verify state changed simulation
if ctrl.taps:
    print("\n[SIMULATION] Bot clicou no Play button!")
    print("Se o jogo reagir, o estado deveria mudar de 'lobby' para 'loading' ou 'brawler_selection'")
else:
    print("\n[WARNING] Bot NÃO clicou em nada!")

print("\n=== TEST COMPLETE ===")
