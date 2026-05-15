"""
movement.py - Soberana Ultimate Tactical Movement v2

Sistema de navegação avançado com:
1. Joystick virtual via ADB swipe (360 graus) — como jogadores reais
2. Bush navigation (hide + ambush pattern do BrawlStarsBot)
3. Storm direction prediction (Showdown)
4. Distância em tiles (resolução-independente)
5. Flanking inteligente
6. Coleta estratégica de Power Cubes

MELHORIA CRÍTICA: Substitui WASD (4 direções) por ADB swipe no joystick virtual,
que permite movimento em qualquer ângulo como um jogador humano.
"""

import math
import time
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dataclasses import dataclass

try:
    import toml
except ImportError:
    toml = None

logger = logging.getLogger(__name__)


@dataclass
class TileMetrics:
    """Métricas de tiles para cálculos resolução-independentes."""
    tile_w: int = 24  # Tiles horizontais no mapa
    tile_h: int = 17  # Tiles verticais no mapa
    speed: float = 2.4  # Tiles por segundo (padrão Colt)
    attack_range: float = 9.33  # Tiles (padrão Colt)
    height_scale_factor: float = 0.15  # Fator de escala de altura

    @classmethod
    def from_brawler(cls, brawler_name: str, stats: Dict = None) -> 'TileMetrics':
        """Cria métricas a partir do nome do brawler e stats."""
        if stats:
            return cls(
                speed=stats.get('speed', 2.4),
                attack_range=stats.get('attack_range', 9.33),
                height_scale_factor=stats.get('heightScaleFactor', 0.15),
            )
        # Defaults por tipo
        defaults = {
            'shelly': cls(speed=2.4, attack_range=4.0, height_scale_factor=0.15),
            'colt': cls(speed=2.4, attack_range=9.33, height_scale_factor=0.158),
            'edgar': cls(speed=2.4, attack_range=3.0, height_scale_factor=0.15),
            'el_primo': cls(speed=2.4, attack_range=2.0, height_scale_factor=0.15),
            'bull': cls(speed=2.4, attack_range=3.33, height_scale_factor=0.15),
            'piper': cls(speed=2.4, attack_range=10.0, height_scale_factor=0.15),
        }
        return defaults.get(brawler_name.lower(), cls())


class Movement:
    """
    Sistema de movimento tático com joystick virtual.

    Usa ADB swipe para simular toque e arraste no joystick virtual do jogo,
    permitindo movimento em qualquer direção (360 graus) como um jogador real.
    """

    # Tile dimensions (from BrawlStarsBot)
    TILE_W = 24
    TILE_H = 17

    def __init__(self, emulator_controller=None, window_w: int = 1920, window_h: int = 1080):
        self.emulator_controller = emulator_controller
        self.last_direction = ""
        self.current_map = None
        self.map_strategy = None
        self._load_map_strategies()

        # Window dimensions for coordinate calculation
        self.window_w = window_w
        self.window_h = window_h

        # Joystick virtual position (bottom-left of screen)
        self.joystick_center_x = round(window_w * 0.10)
        self.joystick_center_y = round(window_h * 0.75)
        self.joystick_radius = round(min(window_w, window_h) * 0.08)

        # Tile size calculation (from BrawlStarsBot)
        self.tile_size = max(1, round(
            (round(window_w / self.TILE_W) + round(window_h / self.TILE_H)) / 2
        ))

        # Center of window (player is always at center in-game)
        # midpoint_offset scales with resolution (12px at 1080p)
        midpoint_offset = round(window_h * 12 / 1080)
        self.center_window = (
            window_w / 2,
            round(window_h / 2) + midpoint_offset
        )

        # Movement state
        self._is_moving = False
        self._move_start_time = 0
        self._current_swipe_id = None
        self._stuck_counter = 0
        self._last_position = None
        self._stuck_threshold = 3  # After N same-direction moves, try different direction

        # Tile metrics (updated when brawler changes)
        self.tile_metrics = TileMetrics()

        # Sharp corner and center order (from BrawlStarsBot)
        self.sharp_corner = True
        self.center_order = True

        logger.info(f"[MOVEMENT] Inicializado: {window_w}x{window_h}, "
                     f"joystick_center=({self.joystick_center_x}, {self.joystick_center_y}), "
                     f"tile_size={self.tile_size}")

    def update_window_size(self, w: int, h: int):
        """Atualiza dimensões da janela e recalcula coordenadas."""
        self.window_w = w
        self.window_h = h
        self.joystick_center_x = round(w * 0.10)
        self.joystick_center_y = round(h * 0.75)
        self.joystick_radius = round(min(w, h) * 0.08)
        self.tile_size = max(1, round(
            (round(w / self.TILE_W) + round(h / self.TILE_H)) / 2
        ))
        midpoint_offset = round(h * 12 / 1080)
        self.center_window = (w / 2, round(h / 2) + midpoint_offset)
        logger.info(f"[MOVEMENT] Atualizado: {w}x{h}, tile_size={self.tile_size}")

    def set_brawler_stats(self, brawler_name: str, speed: float = None,
                          attack_range: float = None):
        """Define stats do brawler atual para cálculos de tiles."""
        if speed is not None:
            self.tile_metrics.speed = speed
        if attack_range is not None:
            self.tile_metrics.attack_range = attack_range
        logger.info(f"[MOVEMENT] Brawler stats: speed={self.tile_metrics.speed}, "
                     f"range={self.tile_metrics.attack_range}")

    def _load_map_strategies(self):
        """Carrega estratégias específicas por mapa do lobby.toml"""
        try:
            config_path = Path(__file__).parent.parent / "lobby.toml"
            if config_path.exists() and toml:
                config = toml.load(str(config_path))
                self.map_strategies = config.get("maps", {})
                logger.info(f"[MOVEMENT] {len(self.map_strategies)} estratégias de mapa carregadas")
            else:
                self.map_strategies = {}
                logger.warning("[MOVEMENT] lobby.toml não encontrado, usando estratégias padrão")
        except Exception as e:
            logger.error(f"[MOVEMENT] Erro ao carregar estratégias de mapa: {e}")
            self.map_strategies = {}

    def set_current_map(self, map_name: str):
        """Define o mapa atual e carrega a estratégia correspondente"""
        self.current_map = map_name
        map_key = map_name.lower().replace(" ", "_")

        if map_key in self.map_strategies:
            self.map_strategy = self.map_strategies[map_key]
            logger.info(f"[MOVEMENT] Mapa definido: {map_name}, estratégia: {self.map_strategy.get('strategy', 'default')}")
        else:
            self.map_strategy = self.map_strategies.get("default", {})
            logger.info(f"[MOVEMENT] Mapa não reconhecido: {map_name}, usando estratégia padrão")

    def get_map_strategy(self) -> Dict:
        """Retorna a estratégia atual do mapa"""
        return self.map_strategy or self.map_strategies.get("default", {})

    # --- Tile-based calculations (from BrawlStarsBot) ---

    def tile_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcula distância em tiles entre duas posições (resolução-independente)."""
        dx = (pos2[0] - pos1[0]) / (self.window_w / self.TILE_W)
        dy = (pos2[1] - pos1[1]) / (self.window_h / self.TILE_H)
        return math.sqrt(dx ** 2 + dy ** 2)

    def pixel_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcula distância em pixels entre duas posições."""
        return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

    def _get_center(self, bbox) -> Tuple[int, int]:
        """Retorna centro de uma bounding box."""
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        return (0, 0)

    # --- Joystick Virtual Movement (360 degrees) ---

    def move_to_position(self, target_x: int, target_y: int, duration: float = None) -> bool:
        """
        Move para uma posição alvo usando joystick virtual via ADB swipe.
        Calcula o ângulo e a duração do swipe baseado na distância em tiles.

        Como o BrawlStarsBot faz: mouseDown no joystick, arrastar na direção,
        segurar pelo tempo calculado, soltar.
        """
        # Calcular ângulo do centro da tela (posição do jogador) para o alvo
        player_x, player_y = self.center_window
        dx = target_x - player_x
        dy = target_y - player_y

        if abs(dx) < 5 and abs(dy) < 5:
            logger.debug("[MOVEMENT] Alvo muito perto, não mover")
            return False

        # Calcular ponto no joystick para a direção desejada
        angle = math.atan2(dy, dx)
        joystick_target_x = int(self.joystick_center_x + self.joystick_radius * math.cos(angle))
        joystick_target_y = int(self.joystick_center_y + self.joystick_radius * math.sin(angle))

        # Calcular duração baseada na distância em tiles
        dist_tiles = self.tile_distance((player_x, player_y), (target_x, target_y))
        if duration is None:
            duration = dist_tiles / self.tile_metrics.speed
            if self.sharp_corner:
                duration *= 1.05  # Aumentar 5% para cantos fechados

        duration = min(duration, 3.0)  # Cap a 3 segundos

        logger.info(f"[MOVEMENT] Joystick swipe: angle={math.degrees(angle):.0f}°, "
                     f"dist={dist_tiles:.1f} tiles, duration={duration:.2f}s")

        # Executar swipe via ADB ou pyautogui
        return self._execute_joystick_swipe(joystick_target_x, joystick_target_y, duration)

    def move_in_direction(self, direction: str, duration: float = 1.0) -> bool:
        """
        Move numa direção cardinal (W/A/S/D) via joystick virtual.
        Compatibilidade com o sistema antigo, mas usando joystick.
        """
        direction_angles = {
            "W": -math.pi / 2,      # Cima
            "S": math.pi / 2,       # Baixo
            "A": math.pi,           # Esquerda
            "D": 0,                 # Direita
            "WA": -3 * math.pi / 4, # Cima-esquerda
            "WD": -math.pi / 4,     # Cima-direita
            "SA": 3 * math.pi / 4,  # Baixo-esquerda
            "SD": math.pi / 4,      # Baixo-direita
        }

        angle = direction_angles.get(direction.upper())
        if angle is None:
            logger.warning(f"[MOVEMENT] Direção desconhecida: {direction}")
            return False

        joystick_target_x = int(self.joystick_center_x + self.joystick_radius * math.cos(angle))
        joystick_target_y = int(self.joystick_center_y + self.joystick_radius * math.sin(angle))

        logger.info(f"[MOVEMENT] Move {direction} for {duration:.1f}s (angle={math.degrees(angle):.0f}°)")
        return self._execute_joystick_swipe(joystick_target_x, joystick_target_y, duration)
    def stop_movement(self):
        """Para o movimento atual (soltar o joystick)."""
        if self._is_moving:
            try:
                if self.emulator_controller:
                    # Swipe de volta ao centro do joystick (soltar)
                    self.emulator_controller.swipe_scaled(
                        self.joystick_center_x + 1, self.joystick_center_y + 1,
                        self.joystick_center_x, self.joystick_center_y,
                        duration=50  # Muito rápido = soltar
                    )
                else:
                    try:
                        import pyautogui
                        pyautogui.mouseUp(button='middle')
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"[MOVEMENT] Erro ao parar: {e}")
            self._is_moving = False
            logger.debug("[MOVEMENT] Movimento parado")

    def _execute_joystick_swipe(self, target_x: int, target_y: int,
                                 duration: float) -> bool:
        """Executa o swipe no joystick virtual via ADB ou pyautogui."""
        try:
            if self.emulator_controller:
                # ADB swipe: tocar no centro do joystick e arrastar para a direção
                self.emulator_controller.swipe_scaled(
                    self.joystick_center_x, self.joystick_center_y,
                    target_x, target_y,
                    duration=int(duration * 1000)  # ms
                )
                self._is_moving = True
                self._move_start_time = time.time()
                logger.debug(f"[MOVEMENT] ADB swipe: ({self.joystick_center_x},{self.joystick_center_y}) "
                             f"-> ({target_x},{target_y}), {duration:.2f}s")
                return True
            else:
                # Fallback: pyautogui drag
                try:
                    import pyautogui
                    pyautogui.moveTo(self.joystick_center_x, self.joystick_center_y)
                    pyautogui.mouseDown(button='middle')
                    pyautogui.moveTo(target_x, target_y, duration=duration)
                    pyautogui.mouseUp(button='middle')
                    self._is_moving = True
                    logger.debug(f"[MOVEMENT] PyAutoGUI swipe: {duration:.2f}s")
                    return True
                except Exception as e:
                    logger.warning(f"[MOVEMENT] PyAutoGUI fallback falhou: {e}")
                    return False
        except Exception as e:
            logger.error(f"[MOVEMENT] Erro ao executar joystick swipe: {e}")
            return False

    # --- Tactical Movement ---

    def get_tactical_movement(self, player, enemies, walls, power_cubes) -> str:
        """
        Determina a melhor direção de movimento baseada na situação tática.
        Retorna string de direção (W/A/S/D/WA/WD/SA/SD) para compatibilidade.
        """
        if not player:
            return ""

        p_center = self._get_center(player)
        strategy = self.get_map_strategy()

        bush_priority = strategy.get("bush_priority", 0.5)
        power_cube_priority = strategy.get("power_cube_priority", 0.5)
        aggression_level = strategy.get("aggression_level", 0.5)

        # 1. EMERGÊNCIA: Fugir de multidões
        if enemies and len(enemies) >= 2:
            closest_enemy = min(enemies, key=lambda e: self.pixel_distance(p_center, self._get_center(e)))
            dist = self.pixel_distance(p_center, self._get_center(closest_enemy))
            if dist < 350:
                logger.debug("[MOVEMENT] EMERGÊNCIA: Fugindo de multidão")
                return self._get_opposite_direction(p_center, self._get_center(closest_enemy))

        # 2. CRESCIMENTO: Coletar Power Cubes seguros
        if power_cubes and power_cube_priority > 0.3:
            safe_cubes = [c for c in power_cubes
                          if not self._is_enemy_near(self._get_center(c), enemies)]
            if safe_cubes:
                target_cube = min(safe_cubes,
                                  key=lambda c: self.pixel_distance(p_center, self._get_center(c)))
                logger.debug("[MOVEMENT] Buscando Power Cube")
                return self._get_direction_to(p_center, self._get_center(target_cube))

        # 3. ATAQUE: Aproximar/mantêr distância de 1 inimigo
        if enemies and len(enemies) == 1:
            enemy_center = self._get_center(enemies[0])
            dist = self.pixel_distance(p_center, enemy_center)
            dist_tiles = self.tile_distance(p_center, enemy_center)

            # Ajustar distância baseada na agressividade e range do brawler
            attack_range = self.tile_metrics.attack_range
            min_dist_tiles = attack_range * 0.3 if aggression_level > 0.7 else attack_range * 0.5
            max_dist_tiles = attack_range * 0.7 if aggression_level > 0.7 else attack_range * 0.8

            if dist_tiles > max_dist_tiles:
                logger.debug(f"[MOVEMENT] Aproximando (dist={dist_tiles:.1f} tiles, max={max_dist_tiles:.1f})")
                return self._get_direction_to(p_center, enemy_center)
            if dist_tiles < min_dist_tiles:
                logger.debug(f"[MOVEMENT] Recuando (dist={dist_tiles:.1f} tiles, min={min_dist_tiles:.1f})")
                return self._get_opposite_direction(p_center, enemy_center)
            # Flanking
            logger.debug("[MOVEMENT] Flanking inimigo")
            return self._get_flank_direction(p_center, enemy_center)

        # 4. EXPLORAÇÃO: Mover para o centro do mapa
        logger.debug("[MOVEMENT] Explorando mapa")
        return self._get_direction_to(p_center, (int(self.center_window[0]), int(self.center_window[1])))

    def get_tactical_movement_target(self, player, enemies, walls, power_cubes,
                                                    game_phase: str = "mid",
                                                    player_hp_estimate: float = 1.0) -> Optional[Tuple[int, int]]:
        """
        Retorna coordenadas do alvo de movimento (para uso com move_to_position).
        Agora com awareness de fase de jogo (early/mid/late) e HP do jogador.
        game_phase: 'early' | 'mid' | 'late' - determina prioridades
        player_hp_estimate: 0.0-1.0 (1.0 = full HP)
        """
        if not player:
            return None

        p_center = self._get_center(player)
        strategy = self.get_map_strategy()

        power_cube_priority = strategy.get("power_cube_priority", 0.5)
        aggression_level = strategy.get("aggression_level", 0.5)

        # --- FASE DE JOGO: ajustar prioridades ---
        if game_phase == "early":
            # Early: farm cubes AGGRESSIVELY, avoid fights
            power_cube_priority = min(1.0, power_cube_priority + 0.3)
            aggression_level = max(0.0, aggression_level - 0.3)
        elif game_phase == "late":
            # Late: survival mode, only fight if advantage
            power_cube_priority = max(0.0, power_cube_priority - 0.2)
            aggression_level = max(0.0, aggression_level - 0.2)

        # --- HP CRÍTICO: fuga prioritária ---
        if player_hp_estimate < 0.25:
            # LOW HP: fuga absoluta, ignore tudo
            if enemies:
                return self._calculate_flee_vector(p_center, enemies, distance=400)
            # Se não há inimigos, ir para bush mais próxima
            if walls:
                bushes = walls
                safe_bushes = [b for b in bushes
                               if not self._is_enemy_near(self._get_center(b), enemies, threshold=250)]
                if safe_bushes:
                    nearest = min(safe_bushes, key=lambda b: self.pixel_distance(p_center, self._get_center(b)))
                    return self._clamp_to_screen(self._get_center(nearest))
            # Fallback: centro do mapa
            return (int(self.center_window[0]), int(self.center_window[1]))

        # 1. EMERGÊNCIA MÚLTIPLOS INIMIGOS
        if enemies and len(enemies) >= 2:
            closest = min(enemies, key=lambda e: self.pixel_distance(p_center, self._get_center(e)))
            dist = self.pixel_distance(p_center, self._get_center(closest))
            if dist < 400:
                # Usar vetor resultante de TODOS os inimigos (kiting inteligente)
                return self._calculate_flee_vector(p_center, enemies, distance=350)

        # 2. POWER CUBES - prioridade baseada em fase
        cube_threshold = 0.3 if game_phase == "early" else 0.5
        if power_cubes and power_cube_priority > cube_threshold:
            # Safe cubes first (no enemy nearby)
            safe_cubes = [c for c in power_cubes
                          if not self._is_enemy_near(self._get_center(c), enemies, threshold=250)]
            if safe_cubes:
                target_cube = min(safe_cubes,
                                  key=lambda c: self.pixel_distance(p_center, self._get_center(c)))
                return self._clamp_to_screen(self._get_center(target_cube))
            # Early game: even risky cubes are OK
            if game_phase == "early" and power_cubes:
                target_cube = min(power_cubes,
                                  key=lambda c: self.pixel_distance(p_center, self._get_center(c)))
                return self._clamp_to_screen(self._get_center(target_cube))

        # 3. BUSH NAVIGATION (cover)
        if walls is not None:
            bushes = walls
            bush_threshold = 0.2 if (player_hp_estimate < 0.5 or game_phase == "late") else 0.4
            if bushes and strategy.get("bush_priority", 0.5) > bush_threshold:
                safe_bushes = [b for b in bushes
                               if not self._is_enemy_near(self._get_center(b), enemies, threshold=200)]
                if safe_bushes:
                    nearest_bush = min(safe_bushes,
                                       key=lambda b: self.pixel_distance(p_center, self._get_center(b)))
                    return self._clamp_to_screen(self._get_center(nearest_bush))

        # 4. ATAQUE POSICIONAMENTO
        if enemies and len(enemies) == 1:
            enemy_center = self._get_center(enemies[0])
            dist_tiles = self.tile_distance(p_center, enemy_center)
            attack_range = self.tile_metrics.attack_range

            ideal_min = attack_range * 0.4
            ideal_max = attack_range * 0.75

            if dist_tiles > ideal_max:
                # Aproximar
                return self._clamp_to_screen(enemy_center)
            elif dist_tiles < ideal_min:
                # Recuar
                dx = p_center[0] - enemy_center[0]
                dy = p_center[1] - enemy_center[1]
                norm = math.sqrt(dx**2 + dy**2) or 1
                target_x = int(p_center[0] + dx / norm * 250)
                target_y = int(p_center[1] + dy / norm * 250)
                return self._clamp_to_screen((target_x, target_y))
            else:
                # Flanking
                dx = enemy_center[0] - p_center[0]
                dy = enemy_center[1] - p_center[1]
                flank_x = -dy
                flank_y = dx
                norm = math.sqrt(flank_x**2 + flank_y**2) or 1
                target_x = int(p_center[0] + flank_x / norm * 180)
                target_y = int(p_center[1] + flank_y / norm * 180)
                return self._clamp_to_screen((target_x, target_y))

        # 5. DEFAULT: explorar para o centro ou andar aleatoriamente
        return (int(self.center_window[0]), int(self.center_window[1]))

    def _calculate_flee_vector(self, p_center, enemies, distance: float = 300) -> Tuple[int, int]:
        """Calcula vetor resultante de fuga considerando TODOS os inimigos."""
        sum_dx, sum_dy, total_weight = 0.0, 0.0, 0.0
        for enemy in enemies:
            ex, ey = self._get_center(enemy)
            dx = p_center[0] - ex
            dy = p_center[1] - ey
            dist = math.sqrt(dx**2 + dy**2) or 1.0
            weight = 1.0 / (dist ** 1.5)
            sum_dx += (dx / dist) * weight
            sum_dy += (dy / dist) * weight
            total_weight += weight
        if total_weight > 0:
            flee_dx = sum_dx / total_weight
            flee_dy = sum_dy / total_weight
            norm = math.sqrt(flee_dx**2 + flee_dy**2) or 1.0
            target_x = int(p_center[0] + flee_dx / norm * distance)
            target_y = int(p_center[1] + flee_dy / norm * distance)
            logger.debug(f"[MOVEMENT] Flee vector to ({target_x}, {target_y})")
            return self._clamp_to_screen((target_x, target_y))
        return self._clamp_to_screen((p_center[0], p_center[1] - int(distance)))

    def _clamp_to_screen(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Garante que as coordenadas não saem do ecrã."""
        margin = 80
        x = max(margin, min(self.window_w - margin, pos[0]))
        y = max(margin, min(self.window_h - margin, pos[1]))
        return (x, y)

    # --- Storm Direction Prediction (from BrawlStarsBot) ---

    def guess_storm_direction(self, player_pos: Optional[Tuple[int, int]] = None) -> List[str]:
        """
        Prevê a direção da tempestade (Showdown) baseado na posição do jogador.
        Se o jogador está à direita do centro, a tempestade vem da direita.
        """
        if player_pos is None:
            player_pos = self.center_window

        x_direction = ""
        y_direction = ""

        x_border = (self.window_w / self.TILE_W) * 1
        y_border = (self.window_h / self.TILE_H) * 1

        p0 = self.center_window
        x_diff = player_pos[0] - p0[0]
        y_diff = player_pos[1] - p0[1]

        if x_diff > x_border:
            x_direction = "right"
        elif x_diff < -x_border:
            x_direction = "left"

        if y_diff > y_border:
            y_direction = "bottom"
        elif y_diff < -y_border:
            y_direction = "top"

        return [x_direction, y_direction]

    def get_storm_movement_keys(self, player_pos: Optional[Tuple[int, int]] = None) -> List[str]:
        """Retorna teclas de movimento para fugir da tempestade."""
        direction = self.guess_storm_direction(player_pos)
        keys = []
        opposites = {"right": "A", "left": "D", "bottom": "W", "top": "S"}

        for d in direction:
            if d in opposites:
                keys.append(opposites[d])

        return keys if keys else ["W"]  # Default: mover para cima

    def get_storm_flee_target(self, player_pos: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Retorna coordenadas para fugir da tempestade."""
        storm_dir = self.guess_storm_direction(player_pos)
        if not player_pos:
            player_pos = (int(self.center_window[0]), int(self.center_window[1]))

        # Move toward center (opposite of storm)
        flee_x = int(self.center_window[0])
        flee_y = int(self.center_window[1])

        # If storm is from right, move left; etc.
        if storm_dir[0] == "right":
            flee_x = int(self.window_w * 0.3)
        elif storm_dir[0] == "left":
            flee_x = int(self.window_w * 0.7)

        if storm_dir[1] == "bottom":
            flee_y = int(self.window_h * 0.3)
        elif storm_dir[1] == "top":
            flee_y = int(self.window_h * 0.7)

        return (flee_x, flee_y)

    # --- Bush Navigation (from BrawlStarsBot) ---

    def find_nearest_bush(self, bushes, player_pos: Optional[Tuple[int, int]] = None,
                          enemies=None) -> Optional[Tuple[int, int]]:
        """
        Encontra a bush mais próxima e segura para se esconder.
        Prioriza bushes no quadrante oposto à tempestade.
        """
        if not bushes:
            return None

        if player_pos is None:
            player_pos = (int(self.center_window[0]), int(self.center_window[1]))

        # Filtrar bushes seguras (sem inimigos perto)
        safe_bushes = [b for b in bushes
                       if not self._is_enemy_near(self._get_center(b), enemies, threshold=200)]

        if not safe_bushes:
            # Se não há bushes seguras, usar a mais próxima mesmo
            safe_bushes = bushes

        # Ordenar por distância
        safe_bushes.sort(key=lambda b: self.pixel_distance(player_pos, self._get_center(b)))

        return self._get_center(safe_bushes[0]) if safe_bushes else None

    # --- Helper Methods ---

    def _get_direction_to(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> str:
        """Retorna direção cardinal (W/A/S/D) de p1 para p2."""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if abs(dx) > abs(dy):
            return "D" if dx > 0 else "A"
        else:
            return "S" if dy > 0 else "W"

    def _get_opposite_direction(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> str:
        """Retorna direção oposta (fugir de p2)."""
        dir_to = self._get_direction_to(p1, p2)
        opposites = {"W": "S", "S": "W", "A": "D", "D": "A"}
        return opposites.get(dir_to, "S")

    def _get_flank_direction(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> str:
        """Retorna direção de flanking (perpendicular)."""
        dir_to = self._get_direction_to(p1, p2)
        flanks = {"W": "D", "S": "A", "A": "W", "D": "S"}
        return flanks.get(dir_to, "D")

    def _is_enemy_near(self, point: Tuple[int, int], enemies, threshold: int = 250) -> bool:
        """Verifica se há inimigos perto de um ponto."""
        if not enemies:
            return False
        for e in enemies:
            if self.pixel_distance(point, self._get_center(e)) < threshold:
                return True
        return False

    # --- Anti-Stuck Detection ---

    def check_stuck(self, current_pos: Tuple[int, int]) -> bool:
        """
        Detecta se o jogador está preso (não se moveu significativamente).
        Retorna True se preso, incrementa o contador de stuck.
        """
        if self._last_position is not None:
            dist = self.pixel_distance(current_pos, self._last_position)
            if dist < 10:  # Menos de 10px de movimento = preso
                self._stuck_counter += 1
                if self._stuck_counter >= self._stuck_threshold:
                    logger.warning(f"[MOVEMENT] STUCK detectado! (counter={self._stuck_counter})")
                    return True
            else:
                self._stuck_counter = 0
        self._last_position = current_pos
        return False

    def get_anti_stuck_movement(self, player_pos: Tuple[int, int],
                                 enemies=None) -> Optional[Tuple[int, int]]:
        """
        Retorna um alvo de movimento para tentar sair de uma posição presa.
        Estratégia: tentar direções alternativas (90°, 180°, 270° da última direção).
        """
        self._stuck_counter = 0  # Reset

        # Se há inimigos, fugir em direção oposta
        if enemies:
            closest = min(enemies, key=lambda e: self.pixel_distance(player_pos, self._get_center(e)))
            enemy_center = self._get_center(closest)
            dx = player_pos[0] - enemy_center[0]
            dy = player_pos[1] - enemy_center[1]
            norm = math.sqrt(dx**2 + dy**2) or 1
            return (int(player_pos[0] + dx / norm * 300),
                    int(player_pos[1] + dy / norm * 300))

        # Tentar direções alternativas em espiral
        import random
        angle = random.uniform(0, 2 * math.pi)
        dist = 200
        return (int(player_pos[0] + dist * math.cos(angle)),
                int(player_pos[1] + dist * math.sin(angle)))
