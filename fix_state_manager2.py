import pathlib
p = pathlib.Path('pylaai_real/state_manager.py')
text = p.read_text(encoding='utf-8')

# 1. Melhorar _handle_lobby
old = '''    def _handle_lobby(self):
        """No lobby - pressiona play com verificações proativas"""
        logger.info("[STATE] No lobby - a pressionar play")
        logger.info(f"[STATE] Lobby automator disponível: {self.lobby is not None}")
        self._diag("lobby_handler_start")
        self._log_lobby_snapshot("lobby")

        if self.lobby:
            try:'''

new = '''    def _handle_lobby(self):
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

if old in text:
    text = text.replace(old, new)
    print('OK: _handle_lobby atualizado')
else:
    print('ERRO: _handle_lobby nao encontrado')

# 2. Melhorar _handle_end_game
old2 = '''    def _handle_end_game(self):
        """No fim de uma partida - processar resultado e sair"""
        logger.debug("[STATE] Handler end_game iniciado")'''

new2 = '''    def _handle_end_game(self):
        """No fim de uma partida - processar resultado e sair com recovery autónomo."""
        logger.debug("[STATE] Handler end_game iniciado")

        # Reset matchmaking timer para permitir novo ciclo
        if hasattr(self, '_matchmaking_enter_time'):
            self._matchmaking_enter_time = None

        # Se estamos no end há muito tempo, forçar retorno ao lobby
        if self.state_start_time and (time.time() - self.state_start_time) > 20:
            logger.warning("[STATE] End screen timeout - forçando retorno ao lobby")
            self.current_state = 'lobby'
            self.state_start_time = time.time()
            return'''

if old2 in text:
    text = text.replace(old2, new2)
    print('OK: _handle_end_game atualizado')
else:
    print('ERRO: _handle_end_game nao encontrado')

p.write_text(text, encoding='utf-8')
print('Guardado.')
