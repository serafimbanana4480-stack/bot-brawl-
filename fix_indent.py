with open('pylaai_real/unified_state_detector.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Encontrar a linha 449 (índice 448) que está em branco depois do bloco #1
# e inserir o bloco #2 Loading correto

insert_idx = 449  # depois da linha em branco

loading_block = """\n        # 2. Loading (green pixel) - mas verificar se NÃO estamos em jogo
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

# Inserir o bloco
lines.insert(insert_idx, loading_block)

with open('pylaai_real/unified_state_detector.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Bloco #2 Loading inserido com sucesso!")
