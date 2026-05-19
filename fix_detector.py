import pathlib
p = pathlib.Path('pylaai_real/unified_state_detector.py')
text = p.read_text(encoding='utf-8')

old = '''        # 8. In-game heuristic: check for joystick area (dark region) + attack button area
        # If joystick area is mostly dark AND attack button area has distinctive color, we're in-game
        joy_x, joy_y = sc(c.joystick_center)
        atk_x, atk_y = sc(c.attack_button_center)
        hp_x, hp_y = sc(c.player_hp_bar)
        timer_x, timer_y = sc(c.match_timer)

        in_game_conf = 0.0
        in_game_details = {}

        if (0 < joy_x < w and 0 < joy_y < h and
            0 < atk_x < w and 0 < atk_y < h):
            # Sample joystick area - should be dark in-game
            joy_region = image[max(0,joy_y-20):min(h,joy_y+20),
                               max(0,joy_x-20):min(w,joy_x+20)]
            if joy_region.size > 0:
                joy_brightness = np.mean(joy_region)
                # Also check attack button area - should have distinctive color in-game
                atk_region = image[max(0,atk_y-30):min(h,atk_y+30),
                                   max(0,atk_x-30):min(w,atk_x+30)]
                atk_brightness = np.mean(atk_region) if atk_region.size > 0 else 255
                atk_std = np.std(atk_region) if atk_region.size > 0 else 0
                in_game_details["brightness"] = float(joy_brightness)
                in_game_details["atk_brightness"] = float(atk_brightness)
                in_game_details["atk_std"] = float(atk_std)
                # In-game: joystick area is dark (brightness < 80) AND attack area has some content
                if joy_brightness < 80:
                    in_game_conf = 0.3
                    if atk_region.size > 0 and atk_std > 20:
                        in_game_conf = 0.5

        # Verificar HP bar (verde no topo esquerdo) para confirmar in_game
        if 0 < hp_x < w and 0 < hp_y < h:
            hp_frac = self._pixel_match_region(
                image, hp_x, hp_y, self._PLAYER_HP_GREEN,
                tolerance=self._TOLERANCES['hp'], sample_radius=5
            )
            if hp_frac > 0.2:
                in_game_conf = max(in_game_conf, 0.6)
                in_game_details["hp_match"] = float(hp_frac)

        # Verificar timer no topo centro (branco)
        if 0 < timer_x < w and 0 < timer_y < h:
            timer_region = image[max(0,timer_y-8):min(h,timer_y+8),
                                 max(0,timer_x-25):min(w,timer_x+25)]
            if timer_region.size > 0:
                timer_brightness = np.mean(timer_region)
                timer_std = np.std(timer_region)
                # Timer: area pequena branca com texto (alto contraste)
                if timer_brightness > 180 and timer_std > 30:
                    in_game_conf = max(in_game_conf, 0.55)
                    in_game_details["timer_brightness"] = float(timer_brightness)
                    in_game_details["timer_std"] = float(timer_std)

        if in_game_conf >= 0.3:
            return DetectedState(
                state="in_game",
                confidence=in_game_conf,
                method="pixel",
                details={"sub_type": "joystick_heuristic", **in_game_details}
            )'''

new = '''        # 8. Matchmaking detection: dark screen with loading spinner or player icons
        # During matchmaking, the screen is mostly dark with occasional UI elements
        # Distinguish from loading by checking for absence of bright green load indicator
        center_region = image[h//3:2*h//3, w//3:2*w//3]
        if center_region.size > 0:
            center_brightness = np.mean(center_region)
            center_std = np.std(center_region)
            # Matchmaking: relatively dark center (brightness 30-140) with some UI variation
            # and NO green load indicator in the bottom-right area
            load_px = sc(c.load_button)
            load_match = self._pixel_match_region(
                image, load_px[0], load_px[1], self._LOAD_COLOR,
                tolerance=self._TOLERANCES['load'], sample_radius=6
            )
            if 20 < center_brightness < 140 and center_std > 8 and load_match < 0.15:
                # Additional check: no bright play button (lobby) and no thumbs down (end)
                play_match = self._pixel_match_region(
                    image, sc(c.play_button)[0], sc(c.play_button)[1], self._PLAY_COLOR,
                    tolerance=self._TOLERANCES['play'], sample_radius=6
                )
                defeated_match = self._pixel_match_region(
                    image, sc(c.defeated1)[0], sc(c.defeated1)[1], self._DEFEATED_COLOR,
                    tolerance=self._TOLERANCES['defeated'], sample_radius=6
                )
                if play_match < 0.2 and defeated_match < 0.3:
                    return DetectedState(
                        state="matchmaking",
                        confidence=0.45,
                        method="pixel",
                        details={"sub_type": "matchmaking_dark_screen",
                                 "center_brightness": float(center_brightness),
                                 "center_std": float(center_std),
                                 "load_match": float(load_match)}
                    )

        # 9. In-game heuristic: check for joystick area (dark region) + attack button area
        # If joystick area is mostly dark AND attack button area has distinctive color, we're in-game
        joy_x, joy_y = sc(c.joystick_center)
        atk_x, atk_y = sc(c.attack_button_center)
        hp_x, hp_y = sc(c.player_hp_bar)
        timer_x, timer_y = sc(c.match_timer)

        in_game_conf = 0.0
        in_game_details = {}

        if (0 < joy_x < w and 0 < joy_y < h and
            0 < atk_x < w and 0 < atk_y < h):
            # Sample joystick area - should be dark in-game
            joy_region = image[max(0,joy_y-20):min(h,joy_y+20),
                               max(0,joy_x-20):min(w,joy_x+20)]
            if joy_region.size > 0:
                joy_brightness = np.mean(joy_region)
                # Also check attack button area - should have distinctive color in-game
                atk_region = image[max(0,atk_y-30):min(h,atk_y+30),
                                   max(0,atk_x-30):min(w,atk_x+30)]
                atk_brightness = np.mean(atk_region) if atk_region.size > 0 else 255
                atk_std = np.std(atk_region) if atk_region.size > 0 else 0
                in_game_details["brightness"] = float(joy_brightness)
                in_game_details["atk_brightness"] = float(atk_brightness)
                in_game_details["atk_std"] = float(atk_std)
                # In-game: joystick area is dark (brightness < 90) AND attack area has some content
                if joy_brightness < 90:
                    in_game_conf = 0.35
                    if atk_region.size > 0 and atk_std > 15:
                        in_game_conf = 0.55

        # Verificar HP bar (verde no topo esquerdo) para confirmar in_game
        if 0 < hp_x < w and 0 < hp_y < h:
            hp_frac = self._pixel_match_region(
                image, hp_x, hp_y, self._PLAYER_HP_GREEN,
                tolerance=self._TOLERANCES['hp'], sample_radius=5
            )
            if hp_frac > 0.15:
                in_game_conf = max(in_game_conf, 0.65)
                in_game_details["hp_match"] = float(hp_frac)

        # Verificar timer no topo centro (branco)
        if 0 < timer_x < w and 0 < timer_y < h:
            timer_region = image[max(0,timer_y-8):min(h,timer_y+8),
                                 max(0,timer_x-25):min(w,timer_x+25)]
            if timer_region.size > 0:
                timer_brightness = np.mean(timer_region)
                timer_std = np.std(timer_region)
                # Timer: area pequena branca com texto (alto contraste)
                if timer_brightness > 160 and timer_std > 25:
                    in_game_conf = max(in_game_conf, 0.6)
                    in_game_details["timer_brightness"] = float(timer_brightness)
                    in_game_details["timer_std"] = float(timer_std)

        if in_game_conf >= 0.35:
            return DetectedState(
                state="in_game",
                confidence=in_game_conf,
                method="pixel",
                details={"sub_type": "joystick_heuristic", **in_game_details}
            )'''

if old in text:
    text = text.replace(old, new)
    p.write_text(text, encoding='utf-8')
    print('OK: unified_state_detector.py atualizado')
else:
    print('ERRO: old string nao encontrada')
    # Debug: mostrar as primeiras linhas do bloco
    start = text.find('# 8. In-game heuristic')
    if start != -1:
        print('Inicio encontrado no indice', start)
        print(repr(text[start:start+200]))
    else:
        print('Inicio NAO encontrado no texto')
