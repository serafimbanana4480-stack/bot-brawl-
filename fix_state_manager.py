import pathlib, re
p = pathlib.Path('pylaai_real/state_manager.py')
text = p.read_text(encoding='utf-8')

# 1. Melhorar _handle_matchmaking: timeout mais agressivo, verificação proativa robusta
old_matchmaking = '''    def _handle_matchmaking(self):
        """Estado de matchmaking - aguarda até a partida começar."""
        logger.info("[STATE] Matchmaking detectado - aguardando início da partida")
        self._diag("matchmaking_handler_start")

        # Verificar se a partida já começou (proactive detection)
        if self.state_start_time:
            elapsed = time.time() - self.state_start_time
            if elapsed > 20:
                logger.warning(f"[STATE] Matchmaking timeout ({elapsed:.0f}s > 20s) - forçando transição para in_game")
                self._diag(f"matchmaking_timeout_force_in_game={elapsed:.1f}")
                self.current_state = 'in_game'
                self.state_start_time = time.time()
                self._remember_known_state('in_game')
                self._forced_in_game_time = time.time()
                logger.info("[STATE] Forçado in_game desde matchmaking - bloqueando retorno por 30s")
                return
            elif elapsed > 10:
                logger.info(f"[STATE] Matchmaking há {elapsed:.0f}s - verificando se partida começou")
                try:
                    img = self._get_cached_screenshot()
                    if img is not None and hasattr(self, 'unified_detector') and self.unified_detector:
                        result = self.unified_detector.detect(img)
                        if result.state in ('in_game', 'loading'):
                            logger.info(f"[STATE] Partida detectada via proactive check: {result.state}")
                            self._diag(f"matchmaking_proactive_detected={result.state}")
                            self.current_state = result.state
                            self.state_start_time = time.time()
                            return
                except Exception as e:
                    logger.debug(f"[STATE] Proactive matchmaking check falhou: {e}")

        # Map detection is now handled automatically in _process_cycle via screen automation hints
        # No need to set default map here anymore
        logger.debug("[STATE] Mapa será detectado automaticamente via screen automation hints")

        time.sleep(1.0)
        if self.screen_automation and hasattr(self.screen_automation, "get_current_state_name"):
            try:
                state_name = self.screen_automation.get_current_state_name()
                self._diag(f"matchmaking_screen_hint={state_name}")
            except Exception as e:
                logger.debug(f"[STATE] Falha ao ler hint de matchmaking: {e}")
        self._diag("matchmaking_handler_done")'''

new_matchmaking = '''    def _handle_matchmaking(self):
        """Estado de matchmaking - aguarda até a partida começar."""
        logger.info("[STATE] Matchmaking detectado - aguardando início da partida")
        self._diag("matchmaking_handler_start")

        # CRITICAL FIX: Usar um timestamp local independente para garantir que o timeout funciona
        # mesmo se state_start_time for modificado externamente
        if not hasattr(self, '_matchmaking_enter_time'):
            self._matchmaking_enter_time = time.time()
        matchmaking_elapsed = time.time() - self._matchmaking_enter_time

        # Verificar se a partida já começou (proactive detection) - a cada ciclo
        try:
            img = self._get_cached_screenshot()
            if img is not None and hasattr(self, 'unified_detector') and self.unified_detector:
                result = self.unified_detector.detect(img)
                if result.state in ('in_game', 'loading'):
                    logger.info(f"[STATE] Partida detectada via proactive check: {result.state}")
                    self._diag(f"matchmaking_proactive_detected={result.state}")
                    self.current_state = result.state
                    self.state_start_time = time.time()
                    self._matchmaking_enter_time = None  # Reset
                    return
        except Exception as e:
            logger.debug(f"[STATE] Proactive matchmaking check falhou: {e}")

        # Verificar se a partida já começou (proactive detection via pixels crus)
        if matchmaking_elapsed > 8:
            logger.info(f"[STATE] Matchmaking há {matchmaking_elapsed:.0f}s - verificando pixels crus")
            try:
                img = self._get_cached_screenshot()
                if img is not None and np is not None:
                    h, w = img.shape[:2]
                    # Verificar joystick area (escura = in_game)
                    joy_y, joy_x = int(h * 0.75), int(w * 0.10)
                    joy_region = img[max(0,joy_y-25):min(h,joy_y+25), max(0,joy_x-25):min(w,joy_x+25)]
                    if joy_region.size > 0 and np.mean(joy_region) < 100:
                        logger.info("[STATE] Joystick escuro detetado - forçando in_game desde matchmaking")
                        self.current_state = 'in_game'
                        self.state_start_time = time.time()
                        self._remember_known_state('in_game')
                        self._forced_in_game_time = time.time()
                        self._matchmaking_enter_time = None
                        return
            except Exception as e:
                logger.debug(f"[STATE] Proactive pixel check falhou: {e}")

        # TIMEOUT AGRESSIVO: forçar in_game após 12s (Brawl Stars matchmaking raramente demora mais)
        if matchmaking_elapsed > 12:
            logger.warning(f"[STATE] Matchmaking timeout ({matchmaking_elapsed:.0f}s > 12s) - FORÇANDO in_game")
            self._diag(f"matchmaking_timeout_force_in_game={matchmaking_elapsed:.1f}")
            self.current_state = 'in_game'
            self.state_start_time = time.time()
            self._remember_known_state('in_game')
            self._forced_in_game_time = time.time()
            self._matchmaking_enter_time = None
            logger.info("[STATE] Forçado in_game desde matchmaking - bloqueando retorno por 25s")
            return

        time.sleep(0.8)
        if self.screen_automation and hasattr(self.screen_automation, "get_current_state_name"):
            try:
                state_name = self.screen_automation.get_current_state_name()
                self._diag(f"matchmaking_screen_hint={state_name}")
            except Exception as e:
                logger.debug(f"[STATE] Falha ao ler hint de matchmaking: {e}")
        self._diag("matchmaking_handler_done")'''

if old_matchmaking in text:
    text = text.replace(old_matchmaking, new_matchmaking)
    print('OK: _handle_matchmaking atualizado')
else:
    print('ERRO: _handle_matchmaking nao encontrado')

# 2. Adicionar stuck detection global no ciclo principal (_process_cycle)
# Vamos adicionar após o check de state timeout (linha ~555)
old_cycle_tail = '''        if not hasattr(self, '_last_log_time'):
            self._last_log_time = 0
        if time.time() - self._last_log_time > 10:
            logger.info(f"Estado atual: {self.current_state}")
            self._last_log_time = time.time()

        if self.current_state in self.states:'''

new_cycle_tail = '''        # === GLOBAL STUCK DETECTION ===
        # Se estamos no mesmo estado há muito tempo e o handler não resolveu,
        # forçar uma ação de recovery autónoma
        if self.state_start_time:
            stuck_elapsed = time.time() - self.state_start_time
            if stuck_elapsed > 25 and self.current_state in ('lobby', 'matchmaking', 'loading', 'brawler_selection'):
                logger.warning(f"[STATE] STUCK DETECTION: {self.current_state} há {stuck_elapsed:.0f}s - forçando recovery")
                self._diag(f"stuck_recovery={self.current_state},elapsed={stuck_elapsed:.1f}")
                if self.current_state == 'lobby':
                    # Se preso no lobby, tentar clicar Play diretamente nas coordenadas padrão
                    self._force_click_play()
                elif self.current_state == 'matchmaking':
                    # Se preso em matchmaking, forçar in_game imediatamente
                    self.current_state = 'in_game'
                    self.state_start_time = time.time()
                    self._forced_in_game_time = time.time()
                    self._matchmaking_enter_time = None
                    return
                elif self.current_state == 'loading':
                    # Se preso em loading, forçar in_game
                    self.current_state = 'in_game'
                    self.state_start_time = time.time()
                    self._forced_in_game_time = time.time()
                    return
                elif self.current_state == 'brawler_selection':
                    # Se preso em brawler selection, tentar confirmar e voltar ao lobby
                    self.current_state = 'lobby'
                    self.state_start_time = time.time()
                    return

        if not hasattr(self, '_last_log_time'):
            self._last_log_time = 0
        if time.time() - self._last_log_time > 10:
            logger.info(f"Estado atual: {self.current_state}")
            self._last_log_time = time.time()

        if self.current_state in self.states:'''

if old_cycle_tail in text:
    text = text.replace(old_cycle_tail, new_cycle_tail)
    print('OK: stuck detection global adicionado')
else:
    print('ERRO: ciclo principal nao encontrado para stuck detection')

# 3. Adicionar método _force_click_play ao StateManager
# Vamos adicionar antes de _handle_lobby
old_handle_lobby_def = '''    def _handle_lobby(self):
        """No lobby - pressiona play com verificações proativas"""
        logger.info("[STATE] No lobby - a pressionar play")
        logger.info(f"[STATE] Lobby automator disponível: {self.lobby is not None}")'''

new_with_force_click = '''    def _force_click_play(self):
        """Força um clique no botão Play usando coordenadas dinâmicas - último recurso autónomo."""
        logger.warning("[STATE] Forçando clique no botão Play (stuck recovery)")
        try:
            if self.movement and hasattr(self.movement, 'window_w'):
                w, h = self.movement.window_w, self.movement.window_h
            else:
                w, h = self._get_window_size()
            play_x = int(w * 0.9119)
            play_y = int(h * 0.9122)
            if self.emulator_controller and hasattr(self.emulator_controller, 'tap'):
                self.emulator_controller.tap(play_x, play_y)
                logger.info(f"[STATE] Clique forçado em Play: ({play_x}, {play_y})")
            elif hasattr(self, '_click') and self._click:
                self._click(play_x, play_y)
                logger.info(f"[STATE] Clique forçado em Play via _click: ({play_x}, {play_y})")
            else:
                logger.warning("[STATE] Não foi possível forçar clique - sem controller disponível")
        except Exception as e:
            logger.error(f"[STATE] Erro ao forçar clique Play: {e}")

    def _handle_lobby(self):
        """No lobby - pressiona play com verificações proativas"""
        logger.info("[STATE] No lobby - a pressionar play")
        logger.info(f"[STATE] Lobby automator disponível: {self.lobby is not None}")'''

if old_handle_lobby_def in text:
    text = text.replace(old_handle_lobby_def, new_with_force_click)
    print('OK: _force_click_play adicionado')
else:
    print('ERRO: _handle_lobby nao encontrado para adicionar _force_click_play')

# 4. Melhorar _handle_lobby para ser mais autónomo e robusto
old_lobby_body = '''    def _handle_lobby(self):
        """No lobby - pressiona play com verificações proativas"""
        logger.info("[STATE] No lobby - a pressionar play")
        logger.info(f"[STATE] Lobby automator disponível: {self.lobby is not None}")
        self._diag("lobby_handler_start")
        self._log_lobby_snapshot("lobby")

        if self.lobby:
            try:'''

new_lobby_body = '''    def _handle_lobby(self):
        """No lobby - pressiona play com verificações proativas e recovery autónomo."""
        logger.info("[STATE] No lobby - a pressionar play")
        logger.info(f"[STATE] Lobby automator disponível: {self.lobby is not None}")
        self._diag("lobby_handler_start")
        self._log_lobby_snapshot("lobby")

        # Se estamos no lobby há muito tempo, limpar estado de matchmaking para permitir novo ciclo
        if hasattr(self, '_matchmaking_enter_time') and self._matchmaking_enter_time is not None:
            self._matchmaking_enter_time = None

        # Verificar se há popups pendentes antes de tudo
        if self.lobby and hasattr(self.lobby, '_popup_manager') and self.lobby._popup_manager:
            try:
                img = self._get_cached_screenshot()
                if img is not None:
                    popup = self.lobby._popup_manager.detect_popup(img)
                    if popup and popup.confidence > 0.3:
                        logger.info(f"[STATE] Popup no lobby: {popup.popup_type} - fechando")
                        self.lobby._popup_manager.handle_popup(
                            popup,
                            click_func=self.lobby._click if hasattr(self.lobby, '_click') else lambda x,y: None,
                            key_func=self.lobby._key_press if hasattr(self.lobby, '_key_press') else lambda k: None
                        )
                        time.sleep(0.5)
            except Exception as e:
                logger.debug(f"[STATE] Erro ao verificar popup no lobby: {e}")

        if self.lobby:
            try:'''

if old_lobby_body in text:
    text = text.replace(old_lobby_body, new_lobby_body)
    print('OK: _handle_lobby melhorado')
else:
    print('ERRO: corpo de _handle_lobby nao encontrado')

# 5. Melhorar _handle_end_game para Play Again automático e mais robusto
old_end_game = '''    def _handle_end_game(self):
        """Ecrã de fim de jogo - clicar em 'Play Again' ou 'Exit'"""
        logger.info("[STATE] Partida terminada - processando ecrã de fim")
        self._diag("end_game_handler_start")'''

new_end_game = '''    def _handle_end_game(self):
        """Ecrã de fim de jogo - clicar em 'Play Again' ou 'Exit' com recovery autónomo."""
        logger.info("[STATE] Partida terminada - processando ecrã de fim")
        self._diag("end_game_handler_start")

        # Reset matchmaking timer para permitir novo ciclo
        if hasattr(self, '_matchmaking_enter_time'):
            self._matchmaking_enter_time = None

        # Se estamos no end há muito tempo, forçar retorno ao lobby
        if self.state_start_time and (time.time() - self.state_start_time) > 20:
            logger.warning("[STATE] End screen timeout - forçando retorno ao lobby")
            self.current_state = 'lobby'
            self.state_start_time = time.time()
            return'''

if old_end_game in text:
    text = text.replace(old_end_game, new_end_game)
    print('OK: _handle_end_game melhorado')
else:
    print('ERRO: _handle_end_game nao encontrado')

# Guardar
p.write_text(text, encoding='utf-8')
print('Guardado.')
