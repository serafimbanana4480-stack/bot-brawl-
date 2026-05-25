"""
test_all_fixes.py

Teste completo que verifica todas as melhorias aplicadas ao bot.
"""

import sys
import time
from pathlib import Path

print("=" * 60)
print("TESTE DE TODAS AS MELHORIAS")
print("=" * 60)

all_ok = True

# Verificar 1: Timeout de loading reduzido
print("\n[1/8] Verificando timeout de loading...")
with open('pylaai_real/state_manager.py', encoding='utf-8') as f:
    content = f.read()

if 'loading_elapsed > 5' in content and '_handle_loading elapsed' in content:
    print("  OK: Timeout de loading = 5s")
else:
    print("  ERRO: Timeout de loading não encontrado ou incorreto")
    all_ok = False

# Verificar 2: Proteção contra retorno a loading
print("\n[2/8] Verificando protecao contra retorno a loading...")
if '_forced_in_game_time' in content and 'bloqueando retorno' in content:
    print("  OK: Proteção contra retorno a loading implementada")
else:
    print("  ERRO: Proteção contra retorno a loading não encontrada")
    all_ok = False

# Verificar 3: Garantia de execução de play_round
print("\n[3/8] Verificando garantia de execucao de play_round...")
if 'GARANTIR que play_round' in content and 'ERRO CR' in content and 'TICO em play_round' in content:
    print("  OK: Garantia de execução de play_round implementada")
else:
    print("  ERRO: Garantia de execução de play_round não encontrada")
    all_ok = False

# Verificar 4: Coordenadas do botão Play atualizadas
print("\n[4/8] Verificando coordenadas do botao Play...")
play_coords_files = [
    'pylaai_real/state_manager.py',
    'pylaai_real/unified_state_detector.py',
    'pylaai_real/screen_automation.py',
    'core/screenshot_analyzer.py',
    'wrapper.py',
]

for fname in play_coords_files:
    fpath = Path(fname)
    if fpath.exists():
        with open(fpath, encoding='utf-8') as f:
            fcontent = f.read()
        if '0.9119' in fcontent and '0.9122' in fcontent:
            print(f"  OK: {fname} tem coordenadas atualizadas")
        else:
            print(f"  AVISO: {fname} pode ter coordenadas desatualizadas")
    else:
        print(f"  AVISO: {fname} não encontrado")

# Verificar 5: AutonomousTester
print("\n[5/8] Verificando AutonomousTester...")
at_path = Path('autonomous_tester.py')
if at_path.exists():
    with open(at_path, encoding='utf-8') as f:
        at_content = f.read()
    if 'periodic_check' in at_content and '_force_combat_action' in at_content:
        print("  OK: AutonomousTester implementado com checks periódicos")
    else:
        print("  ERRO: AutonomousTester incompleto")
        all_ok = False
else:
    print("  ERRO: AutonomousTester não encontrado")
    all_ok = False

# Verificar 6: Fallback de combate
print("\n[6/8] Verificando fallback de combate...")
if 'FALLBACK IMEDIATO' in content and 'FALLBACK:' in content:
    print("  OK: Fallback de combate implementado")
else:
    print("  ERRO: Fallback de combate não encontrado")
    all_ok = False

# Verificar 7: Timeout de matchmaking
print("\n[7/8] Verificando timeout de matchmaking...")
if 'effective_elapsed > 8' in content:
    print("  OK: Timeout de matchmaking = 8s")
else:
    print("  ERRO: Timeout de matchmaking não encontrado")
    all_ok = False

# Verificar 8: Handler de event screen melhorado
print("\n[8/8] Verificando handler de event screen...")
if 'Event screen tap' in content:
    print("  OK: Handler de event screen melhorado")
else:
    print("  ERRO: Handler de event screen não melhorado")
    all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("TODAS AS MELHORIAS VERIFICADAS COM SUCESSO!")
    print("\nResumo das melhorias:")
    print("- Timeout de loading reduzido para 5s")
    print("- Proteção contra retorno a loading por 30s")
    print("- Garantia de execução de play_round mesmo com erros")
    print("- Coordenadas do botão Play atualizadas")
    print("- AutonomousTester para diagnósticos autônomos")
    print("- Fallback de combate quando bot está parado")
    print("- Timeout de matchmaking reduzido para 8s")
    print("- Handler de event screen com múltiplos cliques")
else:
    print("ALGUMAS MELHORIAS NÃO FORAM VERIFICADAS")
print("=" * 60)

sys.exit(0 if all_ok else 1)
