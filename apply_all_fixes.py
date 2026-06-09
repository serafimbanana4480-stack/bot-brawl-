#!/usr/bin/env python3
"""
apply_all_fixes.py

Script de correcao massiva para tornar o Soberana Omega Bot COMPLETAMENTE autonomo.
Aplica todas as melhorias necessarias de uma so vez.
"""

import pathlib
import re

def apply_fixes():
    changes = []

    # ============================================================================
    # FIX 1: state_manager.py - Prevenir reversao de estado forçado pelo handler
    # ============================================================================
    p_sm = pathlib.Path('pylaai_real/state_manager.py')
    text_sm = p_sm.read_text(encoding='utf-8')

    # 1.1. No final de _process_cycle, nao sobrescrever estado se handler o forcou
    old1 = """            self._remember_known_state(detected_state)
            self.current_state = detected_state
            logger.debug(f"[STATE] Estado confirmado: {detected_state}, streak resetado")"""

    new1 = """            # FIX: Nao sobrescrever estado se handler o forcou (ex: matchmaking -> in_game)
            if getattr(self, '_handler_forced_state', False):
                logger.info(f"[STATE] Handler forcou estado, ignorando deteccao: {self.current_state}")
                self._handler_forced_state = False
            else:
                self._remember_known_state(detected_state)
                self.current_state = detected_state
                logger.debug(f"[STATE] Estado confirmado: {detected_state}, streak resetado")"""

    if old1 in text_sm:
        text_sm = text_sm.replace(old1, new1)
        changes.append("OK: FIX 1.1 - Handler forced state protegido")
    else:
        changes.append("ERRO: FIX 1.1 - Bloco nao encontrado")

    # 1.2. No _handle_matchmaking, definir flag quando forca transicao
    old2 = """            self.current_state = 'in_game'
            self.state_start_time = time.time()
            self._remember_known_state('in_game')
            self._forced_in_game_time = time.time()
            self._matchmaking_enter_time = None
            logger.info("[STATE] Forcado in_game desde matchmaking - bloqueando retorno por 25s")
            return"""

    new2 = """            self.current_state = 'in_game'
            self.state_start_time = time.time()
            self._remember_known_state('in_game')
            self._forced_in_game_time = time.time()
            self._matchmaking_enter_time = None
            self._handler_forced_state = True  # FLAG: nao permitir _process_cycle reverter
            logger.info("[STATE] Forcado in_game desde matchmaking - bloqueando retorno por 25s")
            return"""

    if old2 in text_sm:
        text_sm = text_sm.replace(old2, new2)
        changes.append("OK: FIX 1.2 - Matchmaking handler com flag de protecao")
    else:
        changes.append("ERRO: FIX 1.2 - Bloco matchmaking nao encontrado")

    # 1.3. Tambem proteger o proactive check do matchmaking
    old3 = """                        self.current_state = 'in_game'
                        self.state_start_time = time.time()
                        self._remember_known_state('in_game')
                        self._forced_in_game_time = time.time()
                        self._matchmaking_enter_time = None
                        return"""

    new3 = """                        self.current_state = 'in_game'
                        self.state_start_time = time.time()
                        self._remember_known_state('in_game')
                        self._forced_in_game_time = time.time()
                        self._matchmaking_enter_time = None
                        self._handler_forced_state = True
                        return"""

    if old3 in text_sm:
        text_sm = text_sm.replace(old3, new3)
        changes.append("OK: FIX 1.3 - Proactive check matchmaking com flag")
    else:
        changes.append("ERRO: FIX 1.3 - Proactive check nao encontrado")

    # 1.4. Tambem proteger o stuck detection de matchmaking
    old4 = """                elif self.current_state == 'matchmaking':
                    # Se preso em matchmaking, forcar in_game imediatamente
                    self.current_state = 'in_game'
                    self.state_start_time = time.time()
                    self._forced_in_game_time = time.time()
                    self._matchmaking_enter_time = None
                    return"""

    new4 = """                elif self.current_state == 'matchmaking':
                    # Se preso em matchmaking, forcar in_game imediatamente
                    self.current_state = 'in_game'
                    self.state_start_time = time.time()
                    self._forced_in_game_time = time.time()
                    self._matchmaking_enter_time = None
                    self._handler_forced_state = True
                    return"""

    if old4 in text_sm:
        text_sm = text_sm.replace(old4, new4)
        changes.append("OK: FIX 1.4 - Stuck detection matchmaking com flag")
    else:
        changes.append("ERRO: FIX 1.4 - Stuck matchmaking nao encontrado")

    # 1.5. Corrigir loading timeout no run() - 15s -> 10s
    old5 = """                    if loading_elapsed > 15:"""
    new5 = """                    if loading_elapsed > 10:"""

    if old5 in text_sm:
        text_sm = text_sm.replace(old5, new5)
        changes.append("OK: FIX 1.5 - Loading timeout no run: 15s -> 10s")
    else:
        changes.append("ERRO: FIX 1.5 - Loading timeout nao encontrado")

    # 1.6. Reduzir log verboso no run() - INFO -> DEBUG
    old6 = """                logger.info(f"[STATE] DEBUG: current={self.current_state}, start_time={self.state_start_time}")"""
    new6 = """                logger.debug(f"[STATE] DEBUG: current={self.current_state}, start_time={self.state_start_time}")"""

    if old6 in text_sm:
        text_sm = text_sm.replace(old6, new6)
        changes.append("OK: FIX 1.6 - Log verboso reduzido a DEBUG")
    else:
        changes.append("ERRO: FIX 1.6 - Log verboso nao encontrado")

    # 1.7. Corrigir _handle_loading - timeout 15s -> 10s
    old7 = """        # Check timeout: if loading for more than 10s, force to in_game
        elapsed = time.time() - self._loading_start_time
        if elapsed > 10:"""

    if old7 in text_sm:
        changes.append("OK: FIX 1.7 - Loading handler ja esta a 10s")
    else:
        # Tentar encontrar o valor antigo
        idx = text_sm.find('if elapsed >')
        if idx != -1:
            # Verificar se e o bloco de loading
            context = text_sm[max(0,idx-200):idx+50]
            if '_loading_start_time' in context:
                changes.append("INFO: FIX 1.7 - Loading handler precisa de verificacao manual")
            else:
                changes.append("ERRO: FIX 1.7 - Contexto loading nao encontrado")
        else:
            changes.append("ERRO: FIX 1.7 - Loading start time nao encontrado")

    # Guardar state_manager
    p_sm.write_text(text_sm, encoding='utf-8')

    # ============================================================================
    # FIX 2: wrapper.py - Watchdog mais agressivo e robusto
    # ============================================================================
    p_w = pathlib.Path('wrapper.py')
    text_w = p_w.read_text(encoding='utf-8')

    # 2.1. Melhorar watchdog - adicionar retry e log mais claro
    old_watchdog = """                # === WATCHDOG: Recovery autonomo quando o bot esta preso ===
                if self.state_manager and hasattr(self.state_manager, 'state_start_time') and self.state_manager.state_start_time:"""

    new_watchdog = """                # === WATCHDOG: Recovery autonomo quando o bot esta preso ===
                # Este watchdog corre numa thread separada e e o ULTIMO recurso de recovery
                # Tem timeouts MAIS AGRESSIVOS que o StateManager para garantir que o bot NUNCA fica preso
                if self.state_manager and hasattr(self.state_manager, 'state_start_time') and self.state_manager.state_start_time:"""

    if old_watchdog in text_w:
        text_w = text_w.replace(old_watchdog, new_watchdog)
        changes.append("OK: FIX 2.1 - Watchdog comentario melhorado")
    else:
        changes.append("ERRO: FIX 2.1 - Watchdog header nao encontrado")

    # 2.2. Garantir que watchdog reseta matchmaking timer
    old_wd_mm = """                        if state == 'matchmaking' and elapsed > 10:
                            logger.warning(f"[WATCHDOG] Matchmaking preso ha {elapsed:.0f}s - forcando in_game")
                            self.state_manager.current_state = 'in_game'
                            self.state_manager.state_start_time = time.time()
                            if hasattr(self.state_manager, '_forced_in_game_time'):
                                self.state_manager._forced_in_game_time = time.time()
                            if hasattr(self.state_manager, '_matchmaking_enter_time'):
                                self.state_manager._matchmaking_enter_time = None"""

    new_wd_mm = """                        if state == 'matchmaking' and elapsed > 10:
                            logger.warning(f"[WATCHDOG] Matchmaking preso ha {elapsed:.0f}s - forcando in_game")
                            self.state_manager.current_state = 'in_game'
                            self.state_manager.state_start_time = time.time()
                            if hasattr(self.state_manager, '_forced_in_game_time'):
                                self.state_manager._forced_in_game_time = time.time()
                            if hasattr(self.state_manager, '_matchmaking_enter_time'):
                                self.state_manager._matchmaking_enter_time = None
                            if hasattr(self.state_manager, '_handler_forced_state'):
                                self.state_manager._handler_forced_state = True"""

    if old_wd_mm in text_w:
        text_w = text_w.replace(old_wd_mm, new_wd_mm)
        changes.append("OK: FIX 2.2 - Watchdog matchmaking com flag de protecao")
    else:
        changes.append("ERRO: FIX 2.2 - Watchdog matchmaking nao encontrado")

    # Guardar wrapper
    p_w.write_text(text_w, encoding='utf-8')

    # ============================================================================
    # FIX 3: unified_state_detector.py - Melhorar deteccao de loading e lobby
    # ============================================================================
    p_ud = pathlib.Path('pylaai_real/unified_state_detector.py')
    text_ud = p_ud.read_text(encoding='utf-8')

    # 3.1. Verificar se loading esta depois de lobby/end (ja feito antes)
    if '# 8. Loading (green spinner)' in text_ud and '# 9. Matchmaking detection' in text_ud:
        changes.append("OK: FIX 3.1 - Loading ja reposicionado apos lobby/end")
    else:
        changes.append("ALERTA: FIX 3.1 - Loading pode nao estar na posicao correta")

    # Guardar
    p_ud.write_text(text_ud, encoding='utf-8')

    # ============================================================================
    # FIX 4: state_manager.py - Melhorar handler de lobby
    # ============================================================================
    text_sm2 = p_sm.read_text(encoding='utf-8')

    # 4.1. No _handle_lobby, garantir que clica no Play mesmo com falhas
    old_lobby = """        # SEMPRE tentar clicar no Play, mesmo sem lobby automator
        if self.lobby is None:
            logger.warning("[STATE] Lobby automator nao disponivel - usando clique direto nas coordenadas padrao")
            self._force_click_play()
            time.sleep(0.8)
            self.current_state = 'loading'
            self.state_start_time = time.time()
            return"""

    new_lobby = """        # SEMPRE tentar clicar no Play, mesmo sem lobby automator
        if self.lobby is None:
            logger.warning("[STATE] Lobby automator nao disponivel - usando clique direto nas coordenadas padrao")
            self._force_click_play()
            time.sleep(0.8)
            self.current_state = 'loading'
            self.state_start_time = time.time()
            self._handler_forced_state = True
            return"""

    if old_lobby in text_sm2:
        text_sm2 = text_sm2.replace(old_lobby, new_lobby)
        changes.append("OK: FIX 4.1 - Lobby fallback com flag de protecao")
    else:
        changes.append("ERRO: FIX 4.1 - Lobby fallback nao encontrado")

    # 4.2. Melhorar _handle_lobby - se estamos no lobby ha muito tempo, forcar clicar Play
    old_lobby2 = """    def _handle_lobby(self):
        \"\"\"No lobby - pressiona play com verificacoes proativas e recovery autonomo.\"\"\"
        logger.info("[STATE] No lobby - a pressionar play")
        logger.info(f"[STATE] Lobby automator disponivel: {self.lobby is not None}")
        self._diag("lobby_handler_start")"""

    new_lobby2 = """    def _handle_lobby(self):
        \"\"\"No lobby - pressiona play com verificacoes proativas e recovery autonomo.\"\"\"
        logger.info("[STATE] No lobby - a pressionar play")
        logger.info(f"[STATE] Lobby automator disponivel: {self.lobby is not None}")
        self._diag("lobby_handler_start")

        # Se estamos no lobby ha mais de 8 segundos, forcar clique no Play (evita ficar preso)
        lobby_elapsed = 0.0
        if self.state_start_time:
            lobby_elapsed = time.time() - self.state_start_time
        if lobby_elapsed > 8:
            logger.warning(f"[STATE] Lobby ha {lobby_elapsed:.0f}s - forcando clique no Play")
            self._force_click_play()
            time.sleep(0.5)
            self.current_state = 'loading'
            self.state_start_time = time.time()
            self._handler_forced_state = True
            return"""

    if old_lobby2 in text_sm2:
        text_sm2 = text_sm2.replace(old_lobby2, new_lobby2)
        changes.append("OK: FIX 4.2 - Lobby com timeout agressivo de 8s")
    else:
        changes.append("ERRO: FIX 4.2 - _handle_lobby header nao encontrado")

    p_sm.write_text(text_sm2, encoding='utf-8')

    return changes


if __name__ == "__main__":
    print("=" * 60)
    print("SOBERANA OMEGA - CORRECOES MASSIVAS DE AUTONOMIA")
    print("=" * 60)
    print()

    changes = apply_fixes()

    for c in changes:
        print(f"  {c}")

    passed = sum(1 for c in changes if c.startswith("OK:"))
    total = len(changes)

    print()
    print("=" * 60)
    print(f"RESULTADO: {passed}/{total} correcoes aplicadas")
    print("=" * 60)
