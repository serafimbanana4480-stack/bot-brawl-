import pathlib
p = pathlib.Path('pylaai_real/state_manager.py')
text = p.read_text(encoding='utf-8')

# Procurar a definição exata do _handle_lobby e substituir
idx = text.find('    def _handle_lobby(self):\n        """No lobby - pressiona play com verificações proativas"""\n        logger.info("[STATE] No lobby - a pressionar play")\n        logger.info(f"[STATE] Lobby automator disponível: {self.lobby is not None}")\n        self._diag("lobby_handler_start")\n\n        # Verificar popups proativamente antes de tentar clicar no Play')

if idx != -1:
    old_block = text[idx:idx+500]
    # Encontrar o fim do bloco até "if self.lobby:"
    end_idx = old_block.find('\n        if self.lobby:')
    if end_idx != -1:
        old_block = old_block[:end_idx]
    else:
        print('ERRO: nao encontrou fim do bloco')
        exit()

    new_block = '''    def _handle_lobby(self):
        """No lobby - pressiona play com verificações proativas e recovery autónomo."""
        logger.info("[STATE] No lobby - a pressionar play")
        logger.info(f"[STATE] Lobby automator disponível: {self.lobby is not None}")
        self._diag("lobby_handler_start")

        # Se estamos no lobby há muito tempo, limpar estado de matchmaking para permitir novo ciclo
        if hasattr(self, '_matchmaking_enter_time') and self._matchmaking_enter_time is not None:
            self._matchmaking_enter_time = None

        # Verificar popups proativamente antes de tentar clicar no Play'''

    text = text.replace(old_block, new_block)
    p.write_text(text, encoding='utf-8')
    print('OK: _handle_lobby atualizado')
else:
    print('ERRO: indice de _handle_lobby nao encontrado')
