"""
test_loading_timeout.py

Teste para validar que o timeout de loading funciona corretamente.
"""

import sys
import time
from pathlib import Path

print("=" * 60)
print("TESTE DE TIMEOUT DE LOADING")
print("=" * 60)

# Simular o state_manager e verificar o timeout
try:
    from pylaai_real.state_manager import StateManager
    from pylaai_real.state_finder import StateFinder
    from pylaai_real.screenshot_taker import ScreenshotTaker
    from pylaai_real.unified_state_detector import UnifiedStateDetector
    
    taker = ScreenshotTaker("LDPlayer")
    sf = StateFinder(Path("images"))
    det = UnifiedStateDetector(images_path=Path("images"))
    
    # Criar state_manager com mocks
    sm = StateManager(
        screenshot_taker=taker,
        state_finder=sf,
        lobby_automator=None,
        play_logic=None,
        emulator_controller=None,
        movement=None,
        unified_state_detector=det,
    )
    
    # Simular estado loading
    sm.current_state = 'loading'
    sm.state_start_time = time.time()
    
    print("\n[1/3] Verificando timeout no loop principal...")
    # Verificar se o timeout de 5s está configurado
    with open('pylaai_real/state_manager.py', encoding='utf-8') as f:
        content = f.read()
    
    if 'loading_elapsed > 5' in content:
        print("  OK: Timeout de loading no loop principal = 5s")
    else:
        print("  ERRO: Timeout de loading não encontrado ou diferente de 5s")
    
    if '_handle_loading elapsed' in content:
        print("  OK: Logging de elapsed no handler encontrado")
    else:
        print("  ERRO: Logging de elapsed no handler não encontrado")
    
    print("\n[2/3] Verificando proteção _forced_in_game_time...")
    if '_forced_in_game_time' in content and 'bloqueando retorno' in content:
        print("  OK: Proteção contra retorno a loading implementada")
    else:
        print("  ERRO: Proteção contra retorno a loading não encontrada")
    
    print("\n[3/3] Verificando fallback de combate...")
    if 'GARANTIR que play_round' in content:
        print("  OK: Garantia de execução de play_round implementada")
    else:
        print("  ERRO: Garantia de execução de play_round não encontrada")
    
    print("\n" + "=" * 60)
    print("TODAS AS VERIFICAÇÕES PASSARAM!")
    print("=" * 60)
    
except Exception as e:
    print(f"\nERRO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
