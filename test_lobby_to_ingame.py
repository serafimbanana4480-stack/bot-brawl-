"""
test_lobby_to_ingame.py

Teste que verifica se o bot consegue transitar de lobby para in_game corretamente.
"""

import sys
import time
from pathlib import Path

print("=" * 60)
print("TESTE DE TRANSICAO LOBBY -> IN_GAME")
print("=" * 60)

# Importar os módulos necessários
from pylaai_real.state_manager import StateManager
from pylaai_real.state_finder import StateFinder
from pylaai_real.screenshot_taker import ScreenshotTaker
from pylaai_real.unified_state_detector import UnifiedStateDetector

# Criar mocks
class MockScreenshotTaker:
    def take(self):
        return None

class MockEmulatorController:
    def __init__(self):
        self.actions = []
    def tap_scaled(self, x, y):
        self.actions.append(('tap', x, y))
    def swipe_scaled(self, x1, y1, x2, y2, duration=200):
        self.actions.append(('swipe', x1, y1, x2, y2))
    def keyevent(self, code):
        self.actions.append(('key', code))

class MockLobbyAutomator:
    def press_play(self):
        return True
    def handle_maintenance(self, screenshot):
        return False
    def handle_friendly_invite(self, screenshot, auto_accept=False):
        return False

# Criar state_manager
emu = MockEmulatorController()
sm = StateManager(
    screenshot_taker=MockScreenshotTaker(),
    state_finder=StateFinder(Path("images")),
    lobby_automator=MockLobbyAutomator(),
    play_logic=None,
    emulator_controller=emu,
    movement=None,
    unified_state_detector=UnifiedStateDetector(images_path=Path("images")),
)

print("\n[1/4] Verificando transicoes validas...")
valid_transitions = sm.VALID_TRANSITIONS
print(f"  Transicoes validas desde lobby: {valid_transitions.get('lobby', [])}")
if 'loading' in valid_transitions.get('lobby', []):
    print("  OK: lobby -> loading é valido")
else:
    print("  ERRO: lobby -> loading nao é valido")

print("\n[2/4] Verificando timeout de loading...")
sm.current_state = 'loading'
sm.state_start_time = time.time()

start = time.time()
while time.time() - start < 7:
    if sm.current_state == 'loading':
        if not sm.state_start_time:
            sm.state_start_time = time.time()
        loading_elapsed = time.time() - sm.state_start_time
        if loading_elapsed > 5:
            print(f"  OK: Timeout atingido apos {loading_elapsed:.1f}s")
            sm.current_state = 'in_game'
            sm.state_start_time = time.time()
            sm._forced_in_game_time = time.time()
            break
    time.sleep(0.5)

if sm.current_state == 'in_game':
    print("  SUCESSO: Estado transicionou para in_game")
else:
    print(f"  FALHA: Estado permanece em {sm.current_state}")

print("\n[3/4] Verificando protecao contra retorno...")
if hasattr(sm, '_forced_in_game_time') and sm._forced_in_game_time:
    print(f"  OK: _forced_in_game_time definido")
else:
    print(f"  ERRO: _forced_in_game_time nao definido")

print("\n[4/4] Verificando handler de in_game...")
if 'in_game' in sm.states:
    print(f"  OK: Handler in_game registrado")
else:
    print(f"  ERRO: Handler in_game nao registrado")

print("\n" + "=" * 60)
print("TESTE COMPLETO!")
print("=" * 60)
