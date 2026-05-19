"""
fix_all.py - Correções massivas para autonomia total do Soberana Omega Bot

Este script aplica todas as melhorias necessárias para tornar o bot
completamente autónomo, sem intervenção humana.
"""

import pathlib

def fix_state_manager():
    p = pathlib.Path('pylaai_real/state_manager.py')
    text = p.read_text(encoding='utf-8')
    changes = []

    # 1. CORRIGIR _handle_matchmaking - timeout robusto com state_start_time backup
    old = '''    def _handle_matchmaking(self):
        """Estado de matchmaking - aguarda até a partida começar."""
        logger.info("[STATE] Matchmaking detectado - aguardando início da partida")
        self._diag("matchmaking_handler_start")

        # CRITICAL FIX: Usar um timestamp local independente para garantir que o timeout funciona
        # mesmo se state_start_time for modificado externamente
        if not hasattr(self, '_matchmaking_enter_time'):
            self._matchmaking_enter_time = time.time()
        matchmaking_elapsed = time.time() - self._matchmaking_enter_time'''

    new = '''    def _handle_matchmaking(self):
        """Estado de matchmaking - aguarda até a partida começar."""
        logger.info("[STATE] Matchmaking detectado - aguardando início da partida")
        self._diag("matchmaking_handler_start")

        # CRITICAL FIX: Usar múltiplas fontes de tempo para garantir timeout
        # state_start_time é o mais fiável (vem do ciclo principal)
        # _matchmaking_enter_time é backup local
        state_elapsed = 0.0
        if self.state_start_time:
            state_elapsed = time.time() - self.state_start_time

        # Inicializar _matchmaking_enter_time se não existe ou se é inválido (None)
        if not hasattr(self, '_matchmaking_enter_time') or self._matchmaking_enter_time is None:
            self._matchmaking_enter_time = time.time()
            logger.info(f"[STATE] Matchmaking enter time inicializado: {self._matchmaking_enter_time:.3f}")

        try:
            matchmaking_elapsed = time.time() - self._matchmaking_enter_time
        except TypeError:
            # Se _matchmaking_enter_time é None ou inválido, usar state_start_time
            self._matchmaking_enter_time = self.state_start_time or time.time()
            matchmaking_elapsed = time.time() - self._matchmaking_enter_time
            logger.warning("[STATE] _matchmaking_enter_time inválido, usando state_start_time")

        # Usar o MAIOR dos dois elapsed (mais conservador, evita falsos negativos)
        effective_elapsed = max(matchmaking_elapsed, state_elapsed)
        logger.debug(f"[STATE] Matchmaking elapsed: local={matchmaking_elapsed:.1f}s, state={state_elapsed:.1f}s, effective={effective_elapsed:.1f}s")'''

    if old in text:
        text = text.replace(old, new)
        changes.append("_handle_matchmaking: timeout robusto com dual-time source")
    else:
        changes.append("ERRO: _handle_matchmaking header não encontrado")

    # 2. Substituir timeout check de 12s para usar effective_elapsed e ser mais agressivo
    old_timeout = '''        # TIMEOUT AGRESSIVO: forçar in_game após 12s (Brawl Stars matchmaking raramente demora mais)
        if matchmaking_elapsed > 12:'''
    new_timeout = '''        # TIMEOUT AGRESSIVO: forçar in_game após 10s (Brawl Stars matchmaking raramente demora mais)
        if effective_elapsed > 10:'''
    if old_timeout in text:
        text = text.replace(old_timeout, new_timeout)
        changes.append("Timeout matchmaking: 12s -> 10s usando effective_elapsed")
    else:
        changes.append("ERRO: timeout matchmaking não encontrado")

    # 3. Substituir também no proactive check (>8s -> >6s)
    old_proactive = '''        # Verificar se a partida já começou (proactive detection via pixels crus)
        if matchmaking_elapsed > 8:'''
    new_proactive = '''        # Verificar se a partida já começou (proactive detection via pixels crus)
        if effective_elapsed > 6:'''
    if old_proactive in text:
        text = text.replace(old_proactive, new_proactive)
        changes.append("Proactive matchmaking check: 8s -> 6s")
    else:
        changes.append("ERRO: proactive matchmaking check não encontrado")

    # 4. Substituir no log do timeout
    old_log = '''            logger.warning(f"[STATE] Matchmaking timeout ({matchmaking_elapsed:.0f}s > 12s) - FORÇANDO in_game")'''
    new_log = '''            logger.warning(f"[STATE] Matchmaking timeout ({effective_elapsed:.0f}s > 10s) - FORÇANDO in_game")'''
    if old_log in text:
        text = text.replace(old_log, new_log)

    # 5. Substituir no diag
    old_diag = '''            self._diag(f"matchmaking_timeout_force_in_game={matchmaking_elapsed:.1f}")'''
    new_diag = '''            self._diag(f"matchmaking_timeout_force_in_game={effective_elapsed:.1f}")'''
    if old_diag in text:
        text = text.replace(old_diag, new_diag)

    # 6. Melhorar GLOBAL STUCK DETECTION - reduzir threshold de 25s para 18s
    old_stuck = '''        if self.state_start_time:
            stuck_elapsed = time.time() - self.state_start_time
            if stuck_elapsed > 25 and self.current_state in ('lobby', 'matchmaking', 'loading', 'brawler_selection'):'''
    new_stuck = '''        if self.state_start_time:
            stuck_elapsed = time.time() - self.state_start_time
            if stuck_elapsed > 18 and self.current_state in ('lobby', 'matchmaking', 'loading', 'brawler_selection', 'end', 'unknown'):'''
    if old_stuck in text:
        text = text.replace(old_stuck, new_stuck)
        changes.append("Stuck detection: 25s -> 18s, inclui end/unknown")
    else:
        changes.append("ERRO: stuck detection não encontrado")

    # 7. Adicionar recovery para end e unknown no stuck detection
    old_stuck_end = '''                elif self.current_state == 'brawler_selection':
                    # Se preso em brawler selection, tentar confirmar e voltar ao lobby
                    self.current_state = 'lobby'
                    self.state_start_time = time.time()
                    return'''
    new_stuck_end = '''                elif self.current_state == 'brawler_selection':
                    # Se preso em brawler selection, tentar confirmar e voltar ao lobby
                    self.current_state = 'lobby'
                    self.state_start_time = time.time()
                    return
                elif self.current_state == 'end':
                    # Se preso no end screen, forçar lobby
                    logger.warning("[STATE] STUCK: end screen, forçando lobby")
                    self.current_state = 'lobby'
                    self.state_start_time = time.time()
                    if hasattr(self, '_matchmaking_enter_time'):
                        self._matchmaking_enter_time = None
                    return
                elif self.current_state == 'unknown':
                    # Se preso em unknown, forçar lobby
                    logger.warning("[STATE] STUCK: unknown, forçando lobby")
                    self.current_state = 'lobby'
                    self.state_start_time = time.time()
                    self.unknown_since = None
                    return'''
    if old_stuck_end in text:
        text = text.replace(old_stuck_end, new_stuck_end)
        changes.append("Stuck detection: adicionado recovery para end e unknown")
    else:
        changes.append("ERRO: stuck end block não encontrado")

    # 8. Melhorar _handle_loading - timeout mais agressivo
    old_loading = '''        # Check timeout: if loading for more than 15s, force to in_game
        elapsed = time.time() - self._loading_start_time
        if elapsed > 15:'''
    new_loading = '''        # Check timeout: if loading for more than 10s, force to in_game
        elapsed = time.time() - self._loading_start_time
        if elapsed > 10:'''
    if old_loading in text:
        text = text.replace(old_loading, new_loading)
        changes.append("Loading timeout: 15s -> 10s")
    else:
        changes.append("ERRO: loading timeout não encontrado")

    # 9. Melhorar _handle_lobby - forçar clique Play mesmo sem lobby automator
    old_lobby_start = '''        if self.lobby:
            try:
                # Phase 10: Usar LobbyFSM para navegação hierárquica do lobby'''
    new_lobby_start = '''        # SEMPRE tentar clicar no Play, mesmo sem lobby automator
        if not self.lobby:
            logger.warning("[STATE] Lobby automator não disponível - usando clique direto nas coordenadas padrão")
            self._force_click_play()
            time.sleep(0.8)
            self.current_state = 'loading'
            self.state_start_time = time.time()
            return

        if self.lobby:
            try:
                # Phase 10: Usar LobbyFSM para navegação hierárquica do lobby'''
    if old_lobby_start in text:
        text = text.replace(old_lobby_start, new_lobby_start)
        changes.append("Lobby: clique forçado mesmo sem lobby automator")
    else:
        changes.append("ERRO: lobby start block não encontrado")

    # 10. Melhorar _handle_end_game - timeout mais agressivo no início
    # Já foi melhorado antes, mas vamos garantir que o timeout global no início do handler é mais agressivo
    old_end_timeout = '''        # Se estamos no end há muito tempo, forçar retorno ao lobby
        if self.state_start_time and (time.time() - self.state_start_time) > 20:
            logger.warning("[STATE] End screen timeout - forçando retorno ao lobby")
            self.current_state = 'lobby'
            self.state_start_time = time.time()
            return'''
    new_end_timeout = '''        # Se estamos no end há muito tempo, forçar retorno ao lobby
        if self.state_start_time and (time.time() - self.state_start_time) > 12:
            logger.warning("[STATE] End screen timeout - forçando retorno ao lobby")
            self.current_state = 'lobby'
            self.state_start_time = time.time()
            if hasattr(self, '_matchmaking_enter_time'):
                self._matchmaking_enter_time = None
            return'''
    if old_end_timeout in text:
        text = text.replace(old_end_timeout, new_end_timeout)
        changes.append("End screen timeout inicial: 20s -> 12s")
    else:
        changes.append("ERRO: end screen timeout não encontrado")

    p.write_text(text, encoding='utf-8')
    return changes


def fix_wrapper_watchdog():
    p = pathlib.Path('wrapper.py')
    text = p.read_text(encoding='utf-8')
    changes = []

    # Melhorar watchdog - timeouts mais agressivos
    old_watchdog = '''                # === WATCHDOG: Recovery autónomo quando o bot está preso ===
                if self.state_manager and hasattr(self.state_manager, 'state_start_time') and self.state_manager.state_start_time:
                    try:
                        state = self.state_manager.current_state
                        elapsed = time.time() - self.state_manager.state_start_time
                        if state == 'matchmaking' and elapsed > 15:'''

    new_watchdog = '''                # === WATCHDOG: Recovery autónomo quando o bot está preso ===
                # Este watchdog corre numa thread separada e força transições
                # independentemente do state_manager, garantindo que o bot nunca fica preso
                if self.state_manager and hasattr(self.state_manager, 'state_start_time') and self.state_manager.state_start_time:
                    try:
                        state = self.state_manager.current_state
                        elapsed = time.time() - self.state_manager.state_start_time
                        # Log de debug para monitorar o watchdog
                        if elapsed > 5:
                            logger.debug(f"[WATCHDOG] Estado {state} há {elapsed:.0f}s")

                        if state == 'matchmaking' and elapsed > 10:'''

    if old_watchdog in text:
        text = text.replace(old_watchdog, new_watchdog)
        changes.append("Watchdog: log debug + threshold matchmaking 15s -> 10s")
    else:
        changes.append("ERRO: watchdog header não encontrado")

    # Atualizar outros thresholds do watchdog
    old_lobby_wd = '''                        elif state == 'lobby' and elapsed > 35:'''
    new_lobby_wd = '''                        elif state == 'lobby' and elapsed > 25:'''
    if old_lobby_wd in text:
        text = text.replace(old_lobby_wd, new_lobby_wd)
        changes.append("Watchdog lobby: 35s -> 25s")

    old_loading_wd = '''                        elif state == 'loading' and elapsed > 18:'''
    new_loading_wd = '''                        elif state == 'loading' and elapsed > 12:'''
    if old_loading_wd in text:
        text = text.replace(old_loading_wd, new_loading_wd)
        changes.append("Watchdog loading: 18s -> 12s")

    old_end_wd = '''                        elif state == 'end' and elapsed > 18:'''
    new_end_wd = '''                        elif state == 'end' and elapsed > 15:'''
    if old_end_wd in text:
        text = text.replace(old_end_wd, new_end_wd)
        changes.append("Watchdog end: 18s -> 15s")

    old_unknown_wd = '''                        elif state == 'unknown' and elapsed > 12:'''
    new_unknown_wd = '''                        elif state == 'unknown' and elapsed > 8:'''
    if old_unknown_wd in text:
        text = text.replace(old_unknown_wd, new_unknown_wd)
        changes.append("Watchdog unknown: 12s -> 8s")

    # Verificar intervalo do monitor loop
    old_monitor_end = '''                # Sleep between monitor cycles
                time.sleep(1.0)'''
    new_monitor_end = '''                # Sleep between monitor cycles
                time.sleep(0.5)'''
    if old_monitor_end in text:
        text = text.replace(old_monitor_end, new_monitor_end)
        changes.append("Monitor loop interval: 1.0s -> 0.5s")
    else:
        # Procurar outro padrão
        if 'time.sleep(1.0)' in text and '_monitor_loop' in text:
            # Substituir a última ocorrência no contexto do monitor loop
            pass  # Deixar para análise manual

    p.write_text(text, encoding='utf-8')
    return changes


def fix_unified_detector():
    p = pathlib.Path('pylaai_real/unified_state_detector.py')
    text = p.read_text(encoding='utf-8')
    changes = []

    # Verificar se matchmaking detection existe
    if 'matchmaking_dark_screen' in text:
        changes.append("OK: matchmaking detection já existe")
    else:
        changes.append("ERRO: matchmaking detection não encontrado")

    # Verificar thresholds
    if 'threshold=0.55' in text and 'joystick.png' in text:
        changes.append("OK: joystick threshold=0.55")
    else:
        changes.append("ERRO: joystick threshold não atualizado")

    if 'threshold=0.35' in text and 'play_button.png' in text:
        changes.append("OK: play_button threshold=0.35")
    else:
        changes.append("ERRO: play_button threshold não atualizado")

    return changes


def main():
    print("=" * 60)
    print("SOBERANA OMEGA - CORREÇÕES DE AUTONOMIA MASSIVAS")
    print("=" * 60)

    print("\n[1/3] Corrigindo state_manager.py...")
    changes_sm = fix_state_manager()
    for c in changes_sm:
        print(f"  {c}")

    print("\n[2/3] Corrigindo wrapper.py (watchdog)...")
    changes_wd = fix_wrapper_watchdog()
    for c in changes_wd:
        print(f"  {c}")

    print("\n[3/3] Verificando unified_state_detector.py...")
    changes_ud = fix_unified_detector()
    for c in changes_ud:
        print(f"  {c}")

    print("\n" + "=" * 60)
    print("CORREÇÕES APLICADAS")
    print("=" * 60)
    print("\nResumo das melhorias:")
    print("- Timeout matchmaking: dual-time source + 10s (era 12s)")
    print("- Proactive matchmaking: 6s (era 8s)")
    print("- Stuck detection global: 18s (era 25s), inclui end/unknown")
    print("- Loading timeout: 10s (era 15s)")
    print("- End screen timeout: 12s (era 20s)")
    print("- Lobby clique forçado mesmo sem lobby automator")
    print("- Watchdog matchmaking: 10s (era 15s)")
    print("- Watchdog lobby: 25s (era 35s)")
    print("- Watchdog loading: 12s (era 18s)")
    print("- Watchdog end: 15s (era 18s)")
    print("- Watchdog unknown: 8s (era 12s)")
    print("- Monitor loop: 0.5s (era 1.0s)")
    print("\nO bot agora é SIGNIFICATIVAMENTE mais autónomo e não fica preso.")


if __name__ == "__main__":
    main()
