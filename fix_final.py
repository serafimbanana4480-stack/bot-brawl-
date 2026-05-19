with open('pylaai_real/unified_state_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Procurar por padrão: linha 449 (indice 448) deve ter o bloco loading
# Verificar se há o bloco loading
if "# 2. Loading" not in content:
    print("ERRO: Bloco #2 Loading nao encontrado!")
    
# Verificar se detect_by_pixels esta bem formada
if content.count('def _detect_by_pixels') != 1:
    print(f"ERRO: detect_by_pixels aparece {content.count('def _detect_by_pixels')} vezes")

# Encontrar '    # 3. Defeated' e verificar se esta indentado corretamente
lines = content.split('\n')
for i, line in enumerate(lines):
    if '# 3. Defeated' in line:
        print(f"Linha {i+1}: {repr(line)}")
        # A linha deve comecar com 8 espacos
        if line.startswith('        '):
            print("OK: Indentacao correta (8 espacos)")
        else:
            print(f"ERRO DE INDENTACAO: {repr(line[:20])}")
