import pathlib
p = pathlib.Path('pylaai_real/state_manager.py')
text = p.read_text(encoding='utf-8')

old_str = '    def _handle_lobby(self):\n        """No lobby - pressiona play com verificações proativas"""\n        logger.info("[STATE] No lobby - a pressionar play")\n        logger.info(f"[STATE] Lobby automator disponível: {self.lobby is not None}")\n        self._diag("lobby_handler_start")\n\n        # Verificar popups proativamente antes de tentar clicar no Play'

new_str = '    def _handle_lobby(self):\n        """No lobby - pressiona play com verificações proativas e recovery autónomo."""\n        logger.info("[STATE] No lobby - a pressionar play")\n        logger.info(f"[STATE] Lobby automator disponível: {self.lobby is not None}")\n        self._diag("lobby_handler_start")\n\n        # Se estamos no lobby há muito tempo, limpar estado de matchmaking para permitir novo ciclo\n        if hasattr(self, \'_matchmaking_enter_time\') and self._matchmaking_enter_time is not None:\n            self._matchmaking_enter_time = None\n\n        # Verificar popups proativamente antes de tentar clicar no Play'

if old_str in text:
    text = text.replace(old_str, new_str)
    p.write_text(text, encoding='utf-8')
    print('OK: _handle_lobby atualizado')
else:
    print('ERRO: string nao encontrada')
    # Debug
    idx = text.find('def _handle_lobby')
    if idx != -1:
        print('Encontrado em', idx)
        print(repr(text[idx:idx+400]))
