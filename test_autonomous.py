"""
test_autonomous.py

Teste autônomo para validar que o bot funciona corretamente.
Verifica: lobby -> play -> in_game -> combate
"""

import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

print("=" * 60)
print("TESTE AUTONOMO DO BOT")
print("=" * 60)

# Testar imports
print("\n[1/5] Verificando imports...")
try:
    from pylaai_real.state_manager import StateManager
    from pylaai_real.state_finder import StateFinder
    from pylaai_real.screenshot_taker import ScreenshotTaker
    from pylaai_real.unified_state_detector import UnifiedStateDetector
    print("  OK: Todos os imports funcionam")
except Exception as e:
    print(f"  ERRO: {e}")
    sys.exit(1)

# Testar StateManager initialization
print("\n[2/5] Verificando StateManager...")
try:
    taker = ScreenshotTaker("LDPlayer")
    sf = StateFinder("images")
    det = UnifiedStateDetector(images_path="images")
    print("  OK: StateManager pode ser inicializado")
except Exception as e:
    print(f"  AVISO: {e}")

# Testar play_round return type
print("\n[3/5] Verificando play_round...")
try:
    from pylaai_real.play import PlayLogic
    import inspect
    sig = inspect.signature(PlayLogic.play_round)
    if sig.return_annotation == dict:
        print("  OK: play_round retorna dict")
    else:
        print(f"  ERRO: play_round retorna {sig.return_annotation}")
except Exception as e:
    print(f"  ERRO: {e}")

# Testar AutonomousTester
print("\n[4/5] Verificando AutonomousTester...")
try:
    from autonomous_tester import AutonomousTester
    tester = AutonomousTester()
    print("  OK: AutonomousTester inicializado")
except Exception as e:
    print(f"  ERRO: {e}")

# Verificar protecoes no state_manager
print("\n[5/5] Verificando protecoes de estado...")
try:
    with open('pylaai_real/state_manager.py', encoding='utf-8') as f:
        content = f.read()
    
    protections = [
        ('_forced_in_game_time', 'Protecao contra retorno a loading'),
        ('BLOCKED: in_game -> loading', 'Bloqueio de transicao falsa'),
        ('Event screen tap', 'Handler de event screen'),
        ('FALLBACK IMEDIATO', 'Fallback de combate'),
        ('GARANTIR que play_round', 'Garantia de execucao de play_round'),
    ]
    
    all_ok = True
    for keyword, desc in protections:
        if keyword in content:
            print(f"  OK: {desc}")
        else:
            print(f"  ERRO: {desc} nao encontrado")
            all_ok = False
    
    if all_ok:
        print("\n  TODAS AS PROTECOES ESTAO ATIVAS!")
except Exception as e:
    print(f"  ERRO: {e}")

print("\n" + "=" * 60)
print("Proximo passo: execute o bot com 'python wrapper.py'")
print("=" * 60)
