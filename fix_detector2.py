with open('pylaai_real/unified_state_detector.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Encontrar o fim do bloco #1 e inserir o bloco #2
# O bloco #1 termina na linha 448 (índice 447) com '            )\n'
# A linha 449 é '\n' (em branco)
# A linha 450 é '    # 3. Defeated...' que deveria estar indentada com 8 espaços

# Primeiro, preciso corrigir a indentação de todas as linhas a partir do #3
# Elas devem ter 8 espaços (dentro da função) mas têm apenas 4

fixed_lines = []
for i, line in enumerate(lines):
    # Se a linha começa com '    # 3.' até '    return DetectedState(state="unknown"'
    # e não é parte de outra função/método, adicionar mais 4 espaços
    if i >= 449:  # A partir da linha 450 (índice 449)
        # Verificar se é uma linha de comentário de bloco, código, ou return
        stripped = line.lstrip()
        if stripped.startswith('# ') or stripped.startswith('return ') or stripped.startswith('if ') or stripped.startswith('d') or stripped.startswith('s') or stripped.startswith('logger'):
            # Adicionar 4 espaços extras se já tiver 4 espaços
            if line.startswith('    ') and not line.startswith('        '):
                fixed_lines.append('    ' + line)
                continue
    fixed_lines.append(line)

# Agora inserir o bloco #2 Loading no índice 449
loading_block = """        # 2. Loading (green pixel) - mas verificar se NÃO estamos em jogo
        match_frac = self._pixel_match_region(
            image, *sc(c.load_button), self._LOAD_COLOR,
            tolerance=self._TOLERANCES['load']
        )
        if match_frac > 0.2:
            # ANTI-OSCILLATION: Verificar se há indícios de gameplay (joystick + attack button)
            # Se ambos estiverem presentes, estamos em jogo, não loading
            joy_x, joy_y = sc(c.joystick_center)
            attack_x, attack_y = sc(c.attack_button)
            
            in_game_indicators = 0
            if 0 <= joy_x < w and 0 <= joy_y < h:
                joy_region = image[max(0,joy_y-10):min(h,joy_y+10), max(0,joy_x-10):min(w,joy_x+10)]
                if joy_region.size > 0:
                    joy_dark = np.mean(joy_region) < 80  # Joystick escuro = in_game
                    if joy_dark:
                        in_game_indicators += 1
            
            if 0 <= attack_x < w and 0 <= attack_y < h:
                attack_region = image[max(0,attack_y-10):min(h,attack_y+10), max(0,attack_x-10):min(w,attack_x+10)]
                if attack_region.size > 0:
                    attack_bright = np.mean(attack_region) > 100  # Attack button visível
                    if attack_bright:
                        in_game_indicators += 1
            
            if in_game_indicators >= 1:
                logger.debug(f"[UNIFIED_DETECTOR] Loading pixel match ({match_frac:.2f}) mas {in_game_indicators} indicadores de in_game detectados - ignorando loading")
            else:
                logger.debug(f"[UNIFIED_DETECTOR] Loading detected at {c.load_button}: {match_frac:.2f}")
                return DetectedState(
                    state="loading",
                    confidence=match_frac,
                    method="pixel",
                    details={"sub_type": "loading", "match_fraction": match_frac, "in_game_checks": in_game_indicators}
                )

"""

# Inserir o bloco #2 antes da linha 450
fixed_lines.insert(449, loading_block)

with open('pylaai_real/unified_state_detector.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("Arquivo corrigido com sucesso!")
print(f"Total de linhas: {len(fixed_lines)}")
