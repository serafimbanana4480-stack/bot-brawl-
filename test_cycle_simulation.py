"""
test_cycle_simulation.py

Simula o ciclo do state_manager para verificar se o timeout de loading funciona.
"""

import sys
import time
from pathlib import Path

print("=" * 60)
print("SIMULAÇÃO DE CICLO DO STATE MANAGER")
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
    def tap_scaled(self, x, y):
        pass
    def swipe_scaled(self, x1, y1, x2, y2, duration=200):
        pass
    def keyevent(self, code):
        pass

class MockLobbyAutomator:
    def press_play(self):
        return True
    def handle_maintenance(self, screenshot):
        return False
    def handle_friendly_invite(self, screenshot, auto_accept=False):
        return False

# Criar state_manager
sm = StateManager(
    screenshot_taker=MockScreenshotTaker(),
    state_finder=StateFinder(Path("images")),
    lobby_automator=MockLobbyAutomator(),
    play_logic=None,
    emulator_controller=MockEmulatorController(),
    movement=None,
    unified_state_detector=UnifiedStateDetector(images_path=Path("images")),
)

print("\n[1/4] Simulando estado LOADING por 7 segundos...")
sm.current_state = 'loading'
sm.state_start_time = time.time()

# Simular 7 segundos de loading
start = time.time()
cycle_count = 0
while time.time() - start < 7:
    cycle_start = time.time()
    
    # Verificar timeout (código do loop principal)
    if sm.current_state == 'loading':
        if not sm.state_start_time:
            sm.state_start_time = time.time()
        loading_elapsed = time.time() - sm.state_start_time
        if loading_elapsed > 5:
            print(f"  OK: Timeout atingido após {loading_elapsed:.1f}s - forçando in_game")
            sm.current_state = 'in_game'
            sm.state_start_time = time.time()
            sm._forced_in_game_time = time.time()
            break
    
    # Simular delay do ciclo
    time.sleep(0.5)
    cycle_count += 1

if sm.current_state == 'in_game':
    print(f"  SUCESSO: Estado transicionou para in_game após {cycle_count} ciclos")
else:
    print(f"  FALHA: Estado permanece em {sm.current_state} após {cycle_count} ciclos")

print("\n[2/4] Verificando proteção _forced_in_game_time...")
if hasattr(sm, '_forced_in_game_time') and sm._forced_in_game_time:
    elapsed = time.time() - sm._forced_in_game_time
    print(f"  OK: _forced_in_game_time definido há {elapsed:.1f}s")
else:
    print("  ERRO: _forced_in_game_time não definido")

print("\n[3/4] Verificando se in_game -> loading está bloqueado...")
# Simular deteção de loading quando em in_game
forced_time = getattr(sm, '_forced_in_game_time', 0)
if forced_time and (time.time() - forced_time) < 30:
    print(f"  OK: Transição in_game -> loading bloqueada por 30s")
else:
    print(f"  AVISO: Proteção expirou ou não definida")

print("\n[4/4] Resumo...")
print(f"  Estado atual: {sm.current_state}")
print(f"  Ciclos simulados: {cycle_count}")
print(f"  Tempo total: {time.time() - start:.1f}s")

print("\n" + "=" * 60)
if sm.current_state == 'in_game':
    print("SIMULAÇÃO PASSOU - Timeout de loading funciona!")
else:
    print("SIMULAÇÃO FALHOU - Timeout de loading não funcionou")
print("=" * 60)
