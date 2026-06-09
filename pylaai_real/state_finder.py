"""
state_finder.py

Identifica o estado atual do jogo usando template matching.
Carrega regioes de lobby.toml em vez de hardcoded (Fix Error #13).

DEPRECATED: Use pylaai_real.unified_state_detector.UnifiedStateDetector instead.
This module is kept for backward compatibility only.
"""

import warnings
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

warnings.warn(
    "Deprecated: use pylaai_real.unified_state_detector.UnifiedStateDetector instead",
    DeprecationWarning,
    stacklevel=2,
)


class StateFinder:
    """Encontra o estado atual do jogo via template matching"""

    def __init__(self, images_path: Path):
        logger.info("[STATE_FINDER] Inicializando StateFinder")
        self.images_path = images_path
        self.region_data = self._load_regions()
        self._template_cache: Dict[str, Optional[np.ndarray]] = {}
        self.last_diagnostic: Dict[str, object] = {
            "state": "unknown",
            "reason": "not_evaluated",
            "details": {},
        }
        logger.info(f"[STATE_FINDER] Inicializado com images_path={images_path}")
        logger.info(f"[STATE_FINDER] Regiões carregadas: {list(self.region_data.keys())}")

    def _update_diagnostic(self, state: str, reason: str, **details):
        """Guarda o último diagnóstico do detector de estado."""
        self.last_diagnostic = {
            "state": state,
            "reason": reason,
            "details": details,
        }

    def get_diagnostic_report(self) -> Dict[str, object]:
        """Retorna um snapshot do último estado avaliado."""
        return dict(self.last_diagnostic)

    def _state_from_hint(self, screen_state_hint: Optional[str]):
        """Converte o hint da automação de tela para um estado explícito.

        Returns:
            tuple[str, Optional[str]]: (state, map_name) where map_name may be None,
            or None if the hint cannot be resolved to any state.
        """
        if not screen_state_hint:
            logger.debug("[STATE_FINDER] Nenhum screen_state_hint fornecido")
            return None

        normalized_hint = str(screen_state_hint).strip().lower().replace(" ", "_").replace("-", "_")
        logger.debug(f"[STATE_FINDER] Normalizando hint: '{screen_state_hint}' -> '{normalized_hint}'")
        hint_map = {
            "play": "lobby",
            "idle": "lobby",
            "detecting": "matchmaking",
            "loading": "loading",
            "connection_lost": "connection_lost",
            "connection": "connection_lost",
            "exiting": "end",
            "play_again": "end",
            "playagain": "end",
            "star_drop": "end",
            "stardrop": "end",
            "proceed": "end",
            "game": "in_game",
        }

        # Detectar nome do mapa a partir do hint
        mapped = hint_map.get(normalized_hint)
        if mapped:
            logger.debug(f"[STATE_FINDER] Hint mapeado: '{normalized_hint}' -> '{mapped}'")
        else:
            logger.debug(f"[STATE_FINDER] Hint não mapeado: '{normalized_hint}'")

        # Detect map name as a side-effect; store on the instance for callers that need it
        map_name = self._extract_map_from_hint(screen_state_hint)
        self._last_map_from_hint = map_name

        if not mapped and map_name:
            # Hint contains a map but no explicit state → matchmaking
            return ("matchmaking", map_name)
        if mapped:
            return (mapped, map_name)
        return None

    def _extract_map_from_hint(self, screen_state_hint: str) -> Optional[str]:
        """Extrai nome do mapa a partir do hint do screen automation."""
        if not screen_state_hint:
            return None

        hint_lower = str(screen_state_hint).lower()

        # Mapeamento de keywords de mapa para nomes de mapa (baseado em lobby.toml)
        map_keywords = {
            "island": "Island Invasion",
            "canyon": "Canyon Crossing",
            "brawlball": "Brawl Ball",
            "gem": "Gem Grab",
            "bounty": "Bounty",
            "heist": "Heist",
            "showdown": "Showdown",
            "solo": "Solo Showdown",
            "duo": "Duo Showdown",
            "robo": "Robo Rumble",
            "big": "Big Game",
            "boss": "Boss Fight",
        }

        # Procurar keywords no hint
        for keyword, map_name in map_keywords.items():
            if keyword in hint_lower:
                logger.debug(f"[STATE_FINDER] Mapa detectado do hint: '{keyword}' -> '{map_name}'")
                return map_name

        return None

    def _load_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Carrega regioes de interesse de lobby.toml (Fix Error #13)"""
        regions = {}

        # Tentar carregar de lobby.toml
        lobby_toml_path = self.images_path.parent / "lobby.toml"
        if lobby_toml_path.exists():
            try:
                import toml
                config = toml.load(str(lobby_toml_path))
                toml_regions = config.get("regions", {})

                for name, data in toml_regions.items():
                    if all(k in data for k in ("x1", "y1", "x2", "y2")):
                        regions[name] = (data["x1"], data["y1"], data["x2"], data["y2"])
                        logger.debug(f"Regiao '{name}' carregada de lobby.toml: {regions[name]}")

                if regions:
                    logger.info(f"Carregadas {len(regions)} regioes de lobby.toml")
                    return regions
            except ImportError:
                logger.warning("Modulo 'toml' nao instalado. pip install toml")
            except Exception as e:
                logger.error(f"Erro ao ler lobby.toml: {e}")

        # Fallback para hardcoded (valores de referencia PylaAI para 1920x1080)
        logger.warning("Usando regioes hardcoded (fallback). Crie lobby.toml para ajustar.")
        return {
            'thumbs_down': (100, 800, 300, 1000),
            'play_button': (800, 900, 1120, 1000),
            'brawler_select': (400, 600, 800, 800),
            'virtual_joystick': (50, 600, 350, 950),
        }

    def _get_template(self, template_name: str) -> Optional[np.ndarray]:
        """Carrega template com cache para evitar I/O repetido"""
        if template_name in self._template_cache:
            logger.debug(f"[STATE_FINDER] Template '{template_name}' encontrado em cache")
            return self._template_cache[template_name]

        logger.debug(f"[STATE_FINDER] Carregando template: {template_name}")
        template_path = self.images_path / template_name
        if not template_path.exists():
            logger.warning(f"[STATE_FINDER] Template não encontrado: {template_path}")
            self._template_cache[template_name] = None
            return None

        template = cv2.imread(str(template_path))
        if template is None:
            logger.warning(f"[STATE_FINDER] Falha ao ler template: {template_path}")
            self._template_cache[template_name] = None
            return None

        self._template_cache[template_name] = template
        logger.debug(f"[STATE_FINDER] Template carregado e cacheado: {template_name} ({template.shape})")
        return template

    def is_template_in_region(
        self,
        image: np.ndarray,
        template_name: str,
        region: tuple,
        threshold: float = 0.8,
        scale_factor: float = 1.0
    ) -> bool:
        """Verifica se template existe numa regiao da imagem"""
        logger.debug(f"[TEMPLATE] Verificando template '{template_name}'")
        logger.debug(f"[TEMPLATE] Região: {region}")
        logger.debug(f"[TEMPLATE] Threshold: {threshold}")
        logger.debug(f"[TEMPLATE] Scale factor: {scale_factor}")
        try:
            template = self._get_template(template_name)
            if template is None:
                logger.debug(f"[TEMPLATE] Template '{template_name}' não encontrado no disco")
                return False

            # Aplicar escala na região se necessário
            if scale_factor != 1.0:
                x1, y1, x2, y2 = region
                x1, y1 = int(x1 * scale_factor), int(y1 * scale_factor)
                x2, y2 = int(x2 * scale_factor), int(y2 * scale_factor)
            else:
                x1, y1, x2, y2 = region

            # Escalar o template também para condizer com a imagem
            if scale_factor != 1.0:
                tw = int(template.shape[1] * scale_factor)
                th = int(template.shape[0] * scale_factor)
                template = cv2.resize(template, (tw, th))

            # Validar bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                logger.debug(f"[TEMPLATE] ROI vazio para {template_name} em região {region}")
                return False

            # Verificar que template cabe no ROI
            th, tw = template.shape[:2]
            rh, rw = roi.shape[:2]
            if th > rh or tw > rw:
                # Redimensionar template para caber
                scale = min(rw / tw, rh / th) * 0.9
                template = cv2.resize(template, (int(tw * scale), int(th * scale)))

            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            logger.debug(f"[TEMPLATE] Max_val encontrado: {max_val:.4f}")
            logger.debug(f"[TEMPLATE] Match: {max_val >= threshold} (max_val >= threshold)")
            logger.info(f"[STATE_FINDER] Template '{template_name}': max_val={max_val:.3f}, threshold={threshold:.3f}, match={max_val >= threshold}")

            return max_val >= threshold

        except Exception as e:
            logger.error(f"[STATE_FINDER] Erro template matching '{template_name}': {e}", exc_info=True)
            return False

    def is_in_end_of_match(self, image: np.ndarray, scale_factor: float = 1.0) -> bool:
        """Verifica se esta no ecra de fim de partida"""
        logger.debug("[STATE_FINDER] Verificando se está no fim de partida")
        region = self.region_data.get('thumbs_down', (100, 800, 300, 1000))
        thumbs_match = self.is_template_in_region(image, 'thumbs_down.png', region, threshold=0.15, scale_factor=scale_factor)
        
        # Also check for play_again button as fallback
        if not thumbs_match:
            logger.debug("[STATE_FINDER] thumbs_down não encontrado, tentando play_again")
            play_again_region = self.region_data.get('play_button', (800, 900, 1120, 1000))
            return self.is_template_in_region(image, 'play_button.png', play_again_region, threshold=0.15, scale_factor=scale_factor)
        
        return thumbs_match

    def is_in_main_lobby(self, image: np.ndarray, scale_factor: float = 1.0) -> bool:
        """Verifica se esta no lobby principal"""
        logger.debug("[STATE_FINDER] Verificando se está no lobby principal")
        region = self.region_data.get('play_button', (800, 900, 1120, 1000))
        play_match = self.is_template_in_region(image, 'play_button.png', region, threshold=0.0, scale_factor=scale_factor)
        
        # If template fails, it might be due to UI variations - rely more on screen automation hint
        if not play_match:
            logger.debug("[STATE_FINDER] play_button template falhou, pode ser variação de UI")
        
        return play_match

    def is_in_brawler_select(self, image: np.ndarray, scale_factor: float = 1.0) -> bool:
        """Verifica se esta na selecao de brawler"""
        logger.debug("[STATE_FINDER] Verificando se está na seleção de brawler")
        region = self.region_data.get('brawler_select', (400, 600, 800, 800))
        brawler_match = self.is_template_in_region(image, 'brawler_select.png', region, threshold=0.0, scale_factor=scale_factor)
        
        if not brawler_match:
            logger.debug("[STATE_FINDER] brawler_select template falhou, pode ser variação de UI")
        
        return brawler_match

    def is_in_game(self, image: np.ndarray, scale_factor: float = 1.0) -> bool:
        """Verifica se esta in-game (joystick visivel e ausencia de overlays)"""
        logger.debug("[STATE_FINDER] Verificando se está in-game")
        region = self.region_data.get('virtual_joystick', (50, 600, 350, 950))
        has_joystick = self.is_template_in_region(image, 'joystick.png', region, threshold=0.25, scale_factor=scale_factor)
        
        # Se joystick não está visível, não está in-game
        if not has_joystick:
            logger.debug("[STATE_FINDER] Joystick não visível, não está in-game")
            return False
        
        logger.debug("[STATE_FINDER] Joystick visível, verificando se é falso positivo")
        # Se joystick está visível, verificar se está realmente em jogo ou se é falso positivo
        # Falso positivo se não está em nenhum menu conhecido.
        # Nesse caso, não assumir in-game para evitar o bot avançar com estado errado.
        is_lobby = self.is_in_main_lobby(image, scale_factor)
        is_end = self.is_in_end_of_match(image, scale_factor)
        is_brawler = self.is_in_brawler_select(image, scale_factor)
        logger.debug(f"[STATE_FINDER] Verificação de falso positivo: lobby={is_lobby}, end={is_end}, brawler={is_brawler}")

        # Se o brawler select é identificado, isso não é in-game.
        if is_brawler:
            logger.debug("[STATE_FINDER] Brawler select detectado, não está in-game")
            return False

        # Se joystick está visível mas não está em nenhum menu conhecido, pode estar in-game
        # Removendo a restrição que exigia lobby/end para evitar falsos positivos
        # Se joystick está visível e não é brawler select, assume in-game
        logger.debug("[STATE_FINDER] Confirmado in-game (joystick visível e não é brawler select)")
        return True

    def get_state(self, image: np.ndarray, screen_state_hint: Optional[str] = None) -> str:
        """
        Determina o estado atual do jogo.
        Retorna: 'end', 'lobby', 'brawler_selection', 'in_game', 'unknown'
        """
        logger.debug(f"[STATE_FINDER] get_state chamado com hint: {screen_state_hint}")
        if image is None or image.size == 0:
            logger.warning("[STATE_FINDER] Imagem vazia ou inválida")
            _hint_result = self._state_from_hint(screen_state_hint)
            hinted_state = _hint_result[0] if _hint_result else None
            if hinted_state is not None:
                logger.warning(f"[STATE_FINDER] Imagem vazia; usando hint de tela '{screen_state_hint}' como '{hinted_state}'")
                self._update_diagnostic(
                    hinted_state,
                    "screen_state_hint_fallback",
                    screen_state_hint=screen_state_hint,
                    empty_image=True,
                )
                return hinted_state

            self._update_diagnostic("unknown", "empty_image", screen_state_hint=screen_state_hint)
            return 'unknown'

        # Calcular fator de escala baseado em 1920 (largura base)
        h, w = image.shape[:2]
        scale_factor = w / 1920.0
        logger.debug(f"[STATE_FINDER] Imagem shape: {image.shape}, scale_factor: {scale_factor}")

        logger.debug("[STATE_FINDER] Executando detecções de template")
        is_brawler = self.is_in_brawler_select(image, scale_factor)
        is_end = self.is_in_end_of_match(image, scale_factor)
        is_lobby = self.is_in_main_lobby(image, scale_factor)
        is_game = self.is_in_game(image, scale_factor)

        logger.info(f"[STATE_FINDER] Resultados: End={is_end}, Lobby={is_lobby}, Brawler={is_brawler}, Game={is_game}")
        self._update_diagnostic(
            "pending",
            "templates_evaluated",
            scale_factor=scale_factor,
            is_end=is_end,
            is_lobby=is_lobby,
            is_brawler=is_brawler,
            is_game=is_game,
            screen_state_hint=screen_state_hint,
        )

        # PRIORIDADE: Screen automation hint tem prioridade sobre template matching para evitar conflitos
        _hint_result = self._state_from_hint(screen_state_hint)
        hinted_state = _hint_result[0] if _hint_result else None
        
        # Se joystick está visível e template detecta in_game, priorizar isso sobre screen automation
        # Isso permite que o bot entre em jogos iniciados manualmente
        # Ignoramos lobby/end templates se joystick está visível pois isso indica que está no jogo
        if is_game and not is_brawler:
            logger.info("[STATE_FINDER] Joystick visível e template detecta in_game, priorizando sobre screen automation e outros templates")
            self._update_diagnostic("in_game", "joystick_priority_over_all", screen_state_hint=screen_state_hint)
            return 'in_game'
        
        # Se screen automation diz que NÃO é end, ignorar template de end
        if hinted_state and hinted_state != 'end' and is_end:
            logger.warning(f"[STATE_FINDER] Screen automation diz '{hinted_state}' mas template diz 'end'. Priorizando screen automation.")
            self._update_diagnostic(hinted_state, "screen_automation_priority_over_end_template", screen_state_hint=screen_state_hint)
            return hinted_state
        
        # Se screen automation diz que é end, usar isso mesmo se template não detecta
        if hinted_state == 'end':
            logger.info("[STATE_FINDER] Screen automation indica end state")
            self._update_diagnostic("end", "screen_automation_end", screen_state_hint=screen_state_hint)
            return 'end'

        # Se não há conflito com screen automation, usar template matching normalmente
        if is_end:
            logger.info("[STATE_FINDER] Estado detectado: end (template match)")
            self._update_diagnostic("end", "end_template_match", screen_state_hint=screen_state_hint)
            return 'end'
        if is_brawler:
            logger.info("[STATE_FINDER] Estado detectado: brawler_selection (template match)")
            self._update_diagnostic("brawler_selection", "brawler_template_match", screen_state_hint=screen_state_hint)
            return 'brawler_selection'
        if is_lobby:
            logger.info("[STATE_FINDER] Estado detectado: lobby (template match)")
            self._update_diagnostic("lobby", "lobby_template_match", screen_state_hint=screen_state_hint)
            return 'lobby'
        if is_game:
            logger.info("[STATE_FINDER] Estado detectado: in_game (template match)")
            self._update_diagnostic("in_game", "game_template_match", screen_state_hint=screen_state_hint)
            return 'in_game'

        # FALLBACK: usar hint da automação de tela antes de degradar para unknown.
        logger.debug("[STATE_FINDER] Nenhum template match, tentando fallback para hint")
        if hinted_state is not None:
            logger.warning(f"[STATE_FINDER] Nenhum template match; usando hint de tela '{screen_state_hint}' como '{hinted_state}'")
            self._update_diagnostic(
                hinted_state,
                "screen_state_hint_fallback",
                screen_state_hint=screen_state_hint,
            )
            return hinted_state

        logger.warning("[STATE_FINDER] Nenhum template match detectado; retornando unknown")
        self._update_diagnostic(
            "unknown",
            "no_template_match",
            screen_state_hint=screen_state_hint,
        )
        return 'unknown'
