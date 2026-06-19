"""
combat_engine.py

Combat logic extracted from play.py.
Provides CombatEngineMixin with enemy detection, attack logic,
position prediction, and combat decision making.
"""

import logging
import math
import time
from pathlib import Path

# Utilitarios de humanizacao
from core.humanization_utils import jitter_value

# Sistema de combate avancado (Phase 5)
from pylaai_real.combat_advanced import (
    AdvancedCombatStrategy,
    _center,
)

logger = logging.getLogger(__name__)

try:
    import importlib.util as _ilu
    TRACKER_AVAILABLE = _ilu.find_spec("tracker") is not None
except Exception:
    TRACKER_AVAILABLE = False

try:
    from realtime_logs import get_log_manager
    log_manager = get_log_manager()
except ImportError:
    log_manager = None
    logger.warning("[PLAY] Log manager não disponível")

# Phase 10: Advanced Decision Modules
try:
    from decision.utility_ai import UtilityAI
    HAS_UTILITY_AI = True
except ImportError:
    HAS_UTILITY_AI = False
    UtilityAI = None

try:
    from decision.sticky_target import StickyTarget
    HAS_STICKY_TARGET = True
except ImportError:
    HAS_STICKY_TARGET = False
    StickyTarget = None

try:
    from decision.intent_system import IntentSystem
    HAS_INTENT_SYSTEM = True
except ImportError:
    HAS_INTENT_SYSTEM = False
    IntentSystem = None

try:
    from decision.enemy_intention import EnemyIntentionPredictor
    HAS_ENEMY_INTENTION = True
except ImportError:
    HAS_ENEMY_INTENTION = False
    EnemyIntentionPredictor = None

# Phase 10: Central Coordinator
try:
    from core.central_coordinator import CentralCoordinator, DecisionType, Priority, Recommendation
    HAS_COORDINATOR = True
except ImportError:
    HAS_COORDINATOR = False
    CentralCoordinator = None
    Recommendation = None
    DecisionType = None
    Priority = None

# Phase 10: Game Feature Extractor (wall detection, HP extraction)
try:
    from vision.game_feature_extractor import GameFeatureExtractor
    HAS_FEATURE_EXTRACTOR = True
except ImportError:
    HAS_FEATURE_EXTRACTOR = False
    GameFeatureExtractor = None

# Phase 10: Occupancy Grid (A* pathfinding)
try:
    from core.occupancy_grid import OccupancyGrid
    HAS_OCCUPANCY_GRID = True
except ImportError:
    HAS_OCCUPANCY_GRID = False
    OccupancyGrid = None

# Phase 10: CoverSystem (cover evaluation with raycasting)
try:
    from core.cover_system import CoverSystem
    HAS_COVER_SYSTEM = True
except ImportError:
    HAS_COVER_SYSTEM = False
    CoverSystem = None

# Phase 10: WorldModelIntegrator (bridge detections -> WorldModel)
try:
    from world_model_integration import WorldModelIntegrator
    HAS_WORLD_MODEL_INTEGRATION = True
except ImportError:
    HAS_WORLD_MODEL_INTEGRATION = False
    WorldModelIntegrator = None

# Phase 10: MetaAwareness (matchup-aware combat)
try:
    from decision.meta_awareness import MetaAwareness
    HAS_META_AWARENESS = True
except ImportError:
    HAS_META_AWARENESS = False
    MetaAwareness = None


class CombatEngineMixin:
    """Mixin providing advanced combat logic and enemy interaction."""

    def _load_brawler_strategies(self):
        """Carrega estratégias específicas por brawler do lobby.toml"""
        try:
            config_path = Path(__file__).parent.parent.parent / "lobby.toml"
            if config_path.exists():
                import toml
                config = toml.load(str(config_path))
                self.brawler_strategies = config.get("brawlers", {})
                logger.info(f"[PLAY] {len(self.brawler_strategies)} estratégias de brawler carregadas")
            else:
                self.brawler_strategies = {}
                logger.warning("[PLAY] lobby.toml não encontrado, usando estratégias padrão")
        except Exception as e:
            logger.error(f"[PLAY] Erro ao carregar estratégias de brawler: {e}")
            self.brawler_strategies = {}

    def set_current_brawler(self, brawler_name: str):
        """Define o brawler atual e carrega a estratégia correspondente"""
        self.current_brawler = brawler_name
        brawler_key = brawler_name.lower().replace(" ", "_")

        if brawler_key in self.brawler_strategies:
            self.brawler_strategy = self.brawler_strategies[brawler_key]
            # Ajustar cooldown baseado no brawler
            self.shot_cooldown = self.brawler_strategy.get("attack_cooldown", 0.45)
            logger.info(f"[PLAY] Brawler definido: {brawler_name}, estratégia: {self.brawler_strategy.get('role', 'default')}, cooldown: {self.shot_cooldown}s")
        else:
            self.brawler_strategy = self.brawler_strategies.get("default", {})
            self.shot_cooldown = 0.45
            logger.info(f"[PLAY] Brawler não reconhecido: {brawler_name}, usando estratégia padrão")

        # NOVO: Inicializar sistema de combate avancado para este brawler
        try:
            screen_w = 1920
            screen_h = 1080
            if self.movement and hasattr(self.movement, 'window_w'):
                screen_w = self.movement.window_w
                screen_h = self.movement.window_h
            self._combat_strategy = AdvancedCombatStrategy(brawler_name, screen_w=screen_w, screen_h=screen_h)
            self._leading_engine = self._combat_strategy.leading
            self._combo_manager = self._combat_strategy.combo
            logger.info(f"[PLAY] Combat avancado inicializado para {brawler_name} ({screen_w}x{screen_h})")
        except Exception as e:
            logger.warning(f"[PLAY] Falha ao inicializar combat avancado: {e}")

    def set_current_game_mode(self, game_mode: str):
        """Define o modo de jogo atual e sincroniza sistemas dependentes de modo."""
        normalized = (game_mode or "showdown").lower().replace(" ", "_")
        self.current_game_mode = normalized
        if self._intent_system and hasattr(self._intent_system, "set_game_mode"):
            try:
                self._intent_system.set_game_mode(normalized)
            except Exception as e:
                logger.debug(f"[PLAY] IntentSystem mode update failed: {e}")
        logger.info(f"[PLAY] Game mode definido: {self.current_game_mode}")

    def set_pve_mode(self, pve_type: str):
        """Define que estamos em modo PvE (bots) para ajustar estrategia."""
        self.pve_mode = pve_type
        logger.info(f"[PLAY] Modo PvE ativo: {pve_type}")
        # Ajustar estrategia para bots (mais agressivo, menos kiting)
        if self._intent_system and hasattr(self._intent_system, "set_pve_mode"):
            try:
                self._intent_system.set_pve_mode(pve_type)
            except Exception as e:
                logger.debug(f"[PLAY] IntentSystem PvE mode update failed: {e}")

    def get_brawler_strategy(self) -> dict:
        """Retorna a estratégia atual do brawler"""
        return self.brawler_strategy or self.brawler_strategies.get("default", {})

    def reset_for_new_match(self) -> None:
        """Reseta estado para uma nova partida"""
        self.enemy_history.clear()
        if self.enemy_tracker:
            self.enemy_tracker.reset()
            logger.info("[PLAY] EnemyTracker e enemy_history resetados para nova partida")
        self.last_shot_time = 0
        self.super_ready = False
        self._game_phase = "early"
        self._match_start_time = time.time()
        self._power_cubes_collected = 0
        logger.info("[PLAY] Estado de combate resetado para nova partida (phase=early)")

    def get_last_combat_snapshot(self) -> dict[str, object]:
        """Retorna o último snapshot de combate para overlay/diagnóstico."""
        snapshot = dict(self.last_combat_snapshot)

        # Se vision/state estiver disponível, adicionar resumo do estado
        if hasattr(self, 'vision_state') and self.vision_state and hasattr(self.vision_state, 'get_state_summary'):
            try:
                state_summary = self.vision_state.get_state_summary(self.vision_state)
                snapshot["vision_state_summary"] = state_summary
                logger.debug("[PLAY] State summary adicionado ao snapshot")
            except Exception as e:
                logger.warning(f"[PLAY] Falha ao obter state summary: {e}")

        return snapshot

    def _get_enemy_id(self, bbox) -> int:
        """Gera um ID consistente para um inimigo baseado na posição relativa"""
        # Simplificação: Como os brawlers se movem, usamos a proximidade no frame anterior
        # No futuro, um tracker real (SORT/ByteTrack) seria ideal
        bbox = self._normalize_bbox(bbox)
        if not bbox:
            return 0
        return hash((bbox[0] // 50, bbox[1] // 50, bbox[2] // 50, bbox[3] // 50))

    def _get_track_info(self, enemy_bbox) -> dict | None:
        """Retorna informações detalhadas do track associado ao inimigo se disponível."""
        if not self.enemy_tracker or not TRACKER_AVAILABLE:
            return None

        try:
            enemy_center = ((enemy_bbox[0] + enemy_bbox[2]) // 2, (enemy_bbox[1] + enemy_bbox[3]) // 2)
            best_track = None
            min_dist = float('inf')

            for track in self.enemy_tracker.get_all_tracks():
                if track.confirmed:
                    track_center = ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2)
                    dist = ((enemy_center[0] - track_center[0])**2 + (enemy_center[1] - track_center[1])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_track = track

            if best_track and min_dist < 100:
                track_info = {
                    "id": best_track.id,
                    "hit_streak": best_track.hit_streak,
                    "age": best_track.age,
                    "bbox": best_track.bbox,
                    "confirmed": best_track.confirmed,
                }
                logger.debug(f"[PLAY] Track info obtido: {track_info}")
                return track_info
        except Exception as e:
            logger.warning(f"[PLAY] Falha ao obter track info: {e}")

        return None

    def _get_enemy_tracks(self) -> list:
        """Retorna todos os tracks de classe 'enemy' se disponível."""
        if not self.enemy_tracker or not TRACKER_AVAILABLE:
            return []

        try:
            # Tentar usar get_tracks_by_class se disponível no vision/tracker
            if hasattr(self.enemy_tracker, 'get_tracks_by_class'):
                enemy_tracks = self.enemy_tracker.get_tracks_by_class("enemy")
                logger.debug(f"[PLAY] {len(enemy_tracks)} tracks de classe 'enemy' encontrados")
                return enemy_tracks
            else:
                # Fallback: filtrar tracks por classe manualmente
                enemy_tracks = []
                for track in self.enemy_tracker.get_all_tracks():
                    if hasattr(track, 'class_name') and track.class_name == "enemy":
                        enemy_tracks.append(track)
                logger.debug(f"[PLAY] {len(enemy_tracks)} tracks de classe 'enemy' encontrados (fallback)")
                return enemy_tracks
        except Exception as e:
            logger.warning(f"[PLAY] Falha ao obter tracks de inimigos: {e}")
            return []

    def _predict_position(self, enemy_bbox, time_ahead=0.25) -> tuple[int, int]:
        """Calcula posição predita do inimigo (usa tracker se disponível, fallback para estimativa simples)."""
        # Usar predict_position do EnemyTracker se disponível
        if self.enemy_tracker and TRACKER_AVAILABLE:
            try:
                # Encontrar track mais próximo ao bbox
                enemy_center = ((enemy_bbox[0] + enemy_bbox[2]) // 2, (enemy_bbox[1] + enemy_bbox[3]) // 2)
                best_track = None
                min_dist = float('inf')

                for track in self.enemy_tracker.get_all_tracks():
                    if track.confirmed:
                        track_center = ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2)
                        dist = ((enemy_center[0] - track_center[0])**2 + (enemy_center[1] - track_center[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            best_track = track

                if best_track:
                    pred = self.enemy_tracker.predict_position(best_track.id, time_ahead)
                    if pred:
                        logger.debug(f"[PLAY] Usando predict_position do tracker: {pred}")
                        return pred
            except Exception as e:
                logger.warning(f"[PLAY] Falha ao usar predict_position do tracker: {e}")

        # Fallback para método original com get_velocity() do tracker se disponível
        enemy_center = ((enemy_bbox[0] + enemy_bbox[2]) // 2, (enemy_bbox[1] + enemy_bbox[3]) // 2)

        # Tentar usar get_velocity() do tracker se disponível
        if self.enemy_tracker and TRACKER_AVAILABLE:
            try:
                # Encontrar track mais próximo para obter velocidade
                best_track = None
                min_dist = float('inf')

                for track in self.enemy_tracker.get_all_tracks():
                    if track.confirmed:
                        track_center = ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2)
                        dist = ((enemy_center[0] - track_center[0])**2 + (enemy_center[1] - track_center[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            best_track = track

                if best_track and min_dist < 100:  # Threshold de 100px
                    velocity = self.enemy_tracker.get_velocity(best_track.id)
                    if velocity:
                        vx, vy = velocity
                        pred_x = int(enemy_center[0] + vx * time_ahead)
                        pred_y = int(enemy_center[1] + vy * time_ahead)
                        logger.debug(f"[PLAY] Usando get_velocity do tracker: vx={vx:.1f}, vy={vy:.1f}")
                        return (pred_x, pred_y)
            except Exception as e:
                logger.warning(f"[PLAY] Falha ao usar get_velocity do tracker: {e}")

        # Fallback final: estimativa simples baseada em enemy_history
        e_id = self._get_enemy_id(enemy_bbox)
        curr_center = enemy_center
        curr_time = time.time()

        if e_id not in self.enemy_history or len(self.enemy_history[e_id]) < 2:
            self.enemy_history[e_id] = [(curr_center[0], curr_center[1], curr_time)]
            return curr_center

        # Calcular velocidade usando as duas últimas entradas do histórico
        # Isto evita depender de time.time() atual que pode provocar dt > 0.5
        hist = self.enemy_history[e_id]
        prev_x, prev_y, prev_time = hist[-2]
        last_x, last_y, last_time = hist[-1]
        dt = last_time - prev_time
        if dt <= 0 or dt > 0.5:
            # Histórico inválido ou muito antigo: registar posição atual e retornar sem predição
            self.enemy_history[e_id] = [(curr_center[0], curr_center[1], curr_time)]
            return curr_center

        vx = (last_x - prev_x) / dt
        vy = (last_y - prev_y) / dt

        # Atualizar histórico com posição atual (deque maxlen=5 handles overflow)
        self.enemy_history[e_id].append((curr_center[0], curr_center[1], curr_time))

        # Predição linear: P = last_pos + V*t
        pred_x = int(last_x + vx * time_ahead)
        pred_y = int(last_y + vy * time_ahead)

        return (pred_x, pred_y)

    def _get_leading_shot_position(self, enemy_bbox, projectile_speed=15.0, frame_delay=0) -> tuple[int, int]:
        """
        Calcula posição para leading shot usando o tracker se disponível.

        Args:
            enemy_bbox: Bounding box do inimigo [x1, y1, x2, y2]
            projectile_speed: Velocidade do projétil em pixels/frame
            frame_delay: Delay adicional em frames

        Returns:
            Posição (x, y) para mirar
        """
        # Usar get_leading_shot_position do EnemyTracker se disponível
        if self.enemy_tracker and TRACKER_AVAILABLE:
            try:
                # Encontrar track mais próximo ao bbox
                enemy_center = ((enemy_bbox[0] + enemy_bbox[2]) // 2, (enemy_bbox[1] + enemy_bbox[3]) // 2)
                best_track = None
                min_dist = float('inf')

                for track in self.enemy_tracker.get_all_tracks():
                    if track.hit_streak >= 2:  # Track deve ter pelo menos 2 hits
                        track_center = ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2)
                        dist = ((enemy_center[0] - track_center[0])**2 + (enemy_center[1] - track_center[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            best_track = track

                if best_track and min_dist < 100:  # Threshold de 100px
                    leading_pos = self.enemy_tracker.get_leading_shot_position(
                        best_track.id,
                        projectile_speed=projectile_speed,
                        frame_delay=frame_delay
                    )
                    if leading_pos:
                        logger.debug(f"[PLAY] Usando leading shot do tracker: {leading_pos}")
                        return (int(leading_pos[0]), int(leading_pos[1]))
            except Exception as e:
                logger.warning(f"[PLAY] Falha ao usar leading shot do tracker: {e}")

        # Fallback para _predict_position simples
        return self._predict_position(enemy_bbox, time_ahead=0.25)

    def _get_pressure(self, player_bbox) -> float:
        """Get pressure value at player position from PressureMap."""
        if not self.pressure_map:
            return 0.0
        try:
            px, py = _center(player_bbox)
            return self.pressure_map.get_pressure_at(px, py)
        except Exception as e:
            logger.debug(f"[PLAY] PressureMap error: {e}")
            return 0.0

    def _get_danger(self, player_bbox) -> float:
        """Get danger value at player position from PressureMap."""
        if not self.pressure_map:
            return 0.0
        try:
            px, py = _center(player_bbox)
            pressure = self.pressure_map.get_pressure_at(px, py)
            return min(1.0, pressure * 1.5)
        except Exception as e:
            logger.debug(f"[PLAY] PressureMap error: {e}")
            return 0.0

    def _get_matchup_advantage(self) -> float:
        """Get matchup advantage from MetaAwareness for current enemy."""
        if not self.meta_awareness or not self.current_brawler or not self.enemies:
            return 0.0
        try:
            enemy = self.enemies[0]
            enemy_name = self._get_enemy_brawler_name(enemy)
            if enemy_name:
                matchup = self.meta_awareness.evaluate_matchup(self.current_brawler, enemy_name)
                return matchup.advantage
        except Exception as e:
            logger.debug(f"[META] Matchup evaluation error: {e}")
        return 0.0

    def _get_enemy_brawler_name(self, enemy_bbox) -> str | None:
        """Extract brawler name from enemy bbox if available."""
        return None

    def _should_kite_from_matchup(self) -> bool:
        """Should we kite current enemy based on matchup."""
        if not self.meta_awareness or not self.current_brawler or not self.enemies:
            return False
        try:
            enemy = self.enemies[0]
            enemy_name = self._get_enemy_brawler_name(enemy)
            if enemy_name:
                adj = self.meta_awareness.get_combat_adjustment(self.current_brawler, enemy_name)
                return adj.should_kite
        except Exception as e:
            logger.debug(f"[META] Combat adjustment error: {e}")
        return False

    def _should_rush_from_matchup(self) -> bool:
        """Should we rush current enemy based on matchup."""
        if not self.meta_awareness or not self.current_brawler or not self.enemies:
            return False
        try:
            enemy = self.enemies[0]
            enemy_name = self._get_enemy_brawler_name(enemy)
            if enemy_name:
                adj = self.meta_awareness.get_combat_adjustment(self.current_brawler, enemy_name)
                return adj.should_rush
        except Exception as e:
            logger.debug(f"[META] Combat adjustment error: {e}")
        return False

    def _get_aggression_modifier(self) -> float:
        """Get aggression modifier from MetaAwareness based on matchup."""
        if not self.meta_awareness or not self.current_brawler or not self.enemies:
            return 0.0
        try:
            enemy = self.enemies[0]
            enemy_name = self._get_enemy_brawler_name(enemy)
            if enemy_name:
                adj = self.meta_awareness.get_combat_adjustment(self.current_brawler, enemy_name)
                return adj.aggression_modifier
        except Exception as e:
            logger.debug(f"[META] Combat adjustment error: {e}")
        return 0.0

    def _find_best_cover_position(self, player, enemies) -> tuple[int, int] | None:
        """Find best cover position using CoverSystem."""
        if not self.cover_system or not player:
            return None
        try:
            player_pos = _center(player)
            enemy_dicts = [{"x": _center(e)[0], "y": _center(e)[1], "track_id": i}
                           for i, e in enumerate(enemies)]
            best_cover = self.cover_system.find_best_cover(player_pos, enemy_dicts)
            if best_cover:
                return (int(best_cover.x), int(best_cover.y))
        except Exception as e:
            logger.debug(f"[COVER] Find best cover error: {e}")
        return None

    def _compute_frame_reward(self, player, enemies, attack_taken: bool) -> float:
        """
        Calcula reward heuristico para um frame de combate.
        Usado para treinar o Q-Learning online quando nao ha rewards reais.
        """
        reward = 0.0
        # Reward base: estar vivo
        reward += 0.05

        # Reward por ter inimigos proximos (engajamento bom)
        if enemies:
            px = (player[0] + player[2]) // 2 if len(player) >= 4 else player[0]
            py = (player[1] + player[3]) // 2 if len(player) >= 4 else player[1]
            closest_dist = min(
                math.sqrt(((e[0]+e[2])/2 - px)**2 + ((e[1]+e[3])/2 - py)**2)
                for e in enemies if len(e) >= 4
            ) if enemies else 9999
            if 100 < closest_dist < 350:
                reward += 0.3  # Distancia ideal para atacar
            elif closest_dist < 100:
                reward -= 0.1  # Muito perto, perigoso

        # Reward por atacar com sucesso
        if attack_taken:
            reward += 0.5

        # Penalidade por ficar parado sem inimigos (farming negativo)
        if not enemies and not attack_taken:
            reward -= 0.05

        return reward

    def _get_rl_state(self, player, enemies, can_attack=True, can_super=False, real_hp=None):
        """
        Extrai estado discreto para Q-Learning a partir dos dados de combate.
        Retorna tupla (hp_bucket, enemies_bucket, distance_bucket, ammo_bucket, super_bucket).
        """
        if self.rl_engine is None:
            return None
        try:
            from .rl_engine import CombatQLearning
            state = CombatQLearning.state_from_combat_snapshot(
                player_bbox=player,
                enemies=enemies,
                can_attack=can_attack,
                can_super=can_super,
                player_hp_pct=real_hp,
            )
            return state
        except Exception as e:
            logger.debug(f"[RL] Erro ao extrair estado: {e}")
            return None

    def _apply_rl_action(self, rl_action: str, player, enemies, power_cubes, move_key):
        """
        Aplica acao recomendada pelo Q-Learning, modificando comportamento
        quando confianca e alta.
        Retorna move_key possivelmente modificado.
        """
        if rl_action == "attack" and enemies and self.emulator_controller:
            # RL quer atacar - priorizar ataque (ja sera feito pelo _try_smart_attack)
            logger.info("[RL] Acao escolhida: ATTACK")
            return move_key
        elif rl_action == "move_to_enemy" and enemies and self.movement:
            # RL quer avancar no inimigo
            closest = min(enemies, key=lambda e: self._distance(player, e))
            self.movement.move_to_position(
                (closest[0] + closest[2]) // 2,
                (closest[1] + closest[3]) // 2
            )
            logger.info("[RL] Acao escolhida: MOVE_TO_ENEMY")
            return "forward"
        elif rl_action == "retreat":
            # RL quer recuar - fugir do inimigo mais proximo
            if enemies and self.movement:
                closest = min(enemies, key=lambda e: self._distance(player, e))
                cx = (closest[0] + closest[2]) // 2
                cy = (closest[1] + closest[3]) // 2
                px = (player[0] + player[2]) // 2 if len(player) >= 4 else player[0]
                py = (player[1] + player[3]) // 2 if len(player) >= 4 else player[1]

                # A* pathfinding se OccupancyGrid disponível
                if self._occupancy_grid:
                    goal_x = px + (px - cx)
                    goal_y = py + (py - cy)
                    path = self._occupancy_grid.a_star((px, py), (goal_x, goal_y))
                    if path and len(path) > 1:
                        # Ir para primeiro waypoint do caminho
                        flee_x, flee_y = path[1]
                        self.movement.move_to_position(int(flee_x), int(flee_y))
                        logger.info(f"[RL] Acao escolhida: RETREAT (A* path with {len(path)} waypoints)")
                    else:
                        # Fallback: método simples
                        flee_x = px + (px - cx)
                        flee_y = py + (py - cy)
                        self.movement.move_to_position(int(flee_x), int(flee_y))
                        logger.info("[RL] Acao escolhida: RETREAT (fallback)")
                else:
                    # Sem A*: método simples
                    dx = px - cx
                    dy = py - cy
                    flee_x = px + dx
                    flee_y = py + dy
                    self.movement.move_to_position(int(flee_x), int(flee_y))
                    logger.info("[RL] Acao escolhida: RETREAT")
                return "backward"
            return move_key
        elif rl_action == "use_super" and enemies and self.emulator_controller:
            # RL quer usar super
            if self.movement and hasattr(self.movement, 'window_w'):
                w, h = self.movement.window_w, self.movement.window_h
            else:
                w, h = self._get_safe_resolution()
            super_btn_x = round(w * 0.75)
            super_btn_y = round(h * 0.69)
            self.emulator_controller.ensure_window_active()
            self.emulator_controller.tap_scaled(super_btn_x, super_btn_y)
            logger.info("[RL] Acao escolhida: USE_SUPER")
            return move_key
        elif rl_action == "collect_cube" and power_cubes and self.movement:
            # RL quer coletar power cube
            cube = min(power_cubes, key=lambda c: self._distance(player, c))
            self.movement.move_to_position(
                (cube[0] + cube[2]) // 2,
                (cube[1] + cube[3]) // 2
            )
            logger.info("[RL] Acao escolhida: COLLECT_CUBE")
            return "forward"
        elif rl_action == "idle":
            logger.info("[RL] Acao escolhida: IDLE")
            return ""
        return move_key

    def _get_target_position_for_attack(self, player, enemies):
        """Calcula a posição predita do alvo para ataque (sem executar o ataque)."""
        logger.debug("[TARGET] Calculando posição predita do alvo")
        if not enemies:
            logger.debug("[COMBAT] Nenhum inimigo para calcular alvo")
            return None

        closest_enemy = min(enemies, key=lambda e: self._distance(player, e))
        logger.debug(f"[TARGET] Inimigo selecionado: bbox={closest_enemy}")
        predicted = self._predict_position(closest_enemy)
        logger.debug(f"[TARGET] Posição atual do inimigo: center={((closest_enemy[0]+closest_enemy[2])//2, (closest_enemy[1]+closest_enemy[3])//2)}")
        logger.debug(f"[TARGET] Posição predita: {predicted}")
        logger.debug(f"[TARGET] Diferença (leading shot): dx={predicted[0]-((closest_enemy[0]+closest_enemy[2])//2):.1f}, dy={predicted[1]-((closest_enemy[1]+closest_enemy[3])//2):.1f}")
        return predicted

    def _try_smart_attack(self, player, enemies):
        """
        Ataca com predição de movimento (Leading shots) e mira direcional.
        Usa distância em tiles (resolução-independente) e swipe no botão de ataque
        para mirar na direção do inimigo (como jogador real faz).
        """
        logger.debug("[ATTACK] Avaliando ataque")
        brawler_strategy = self.get_brawler_strategy()

        # Usar cooldown específico do brawler com jitter humano
        effective_cooldown = jitter_value(self.shot_cooldown, self.shot_cooldown_jitter, min_val=0.2)
        cooldown_remaining = self.last_shot_time + effective_cooldown - time.time()
        if cooldown_remaining > 0:
            logger.debug(f"[ATTACK] Cooldown restante: {cooldown_remaining:.2f}s (efetivo: {effective_cooldown:.2f}s)")
            return

        # Delay de reação humano (150-400ms tipico)
        reaction_time = self._human_pause.reaction_delay(base_ms=120)
        logger.debug(f"[ATTACK] Delay de reacao: {reaction_time*1000:.0f}ms")
        time.sleep(reaction_time)

        if not enemies:
            logger.debug("[ATTACK] Nenhum inimigo para atacar")
            return

        # Escolher alvo mais próximo com base em track info
        closest_enemy = None
        closest_dist = float('inf')

        for enemy in enemies:
            dist = self._distance(player, enemy)
            track_info = self._get_track_info(enemy)

            if track_info and track_info.get("hit_streak", 0) >= 2:
                adjusted_dist = dist * 0.9
            else:
                adjusted_dist = dist

            if adjusted_dist < closest_dist:
                closest_dist = adjusted_dist
                closest_enemy = enemy

        # Calcular distância em tiles (resolução-independente)
        attack_range_tiles = brawler_strategy.get("attack_range_tiles", 9.33)
        # Usar tile_size do Movement se disponível, senão estimar
        if self.movement and hasattr(self.movement, 'tile_size') and self.movement.tile_size > 0:
            dist_tiles = closest_dist / self.movement.tile_size
        else:
            dist_tiles = closest_dist / (1920 / 24)

        logger.info(f"[ATTACK] Alvo: dist={closest_dist:.0f}px (~{dist_tiles:.1f} tiles), "
                     f"range={attack_range_tiles:.1f} tiles")

        # Verificar se está na distância ideal (em tiles)
        max_attack_tiles = attack_range_tiles * 1.2  # 20% margem
        if dist_tiles > max_attack_tiles:
            logger.debug(f"[ATTACK] Muito longe ({dist_tiles:.1f} > {max_attack_tiles:.1f} tiles)")
            return

        # Predizer posição do inimigo (leading shot)
        target_pos = self._predict_position(closest_enemy)
        self.last_combat_snapshot = {
            **self.last_combat_snapshot,
            "target_position": target_pos,
            "last_error": None,
        }

        if self.emulator_controller:
            try:
                self.emulator_controller.ensure_window_active()

                # NOVO: Ataque direcional — swipe do botão de ataque na direção do alvo
                # Em vez de apenas tap, fazemos um swipe curto na direção do inimigo
                # Isto é como um jogador real faz: toca e arrasta na direção que quer atirar
                player_center = ((player[0] + player[2]) // 2, (player[1] + player[3]) // 2)
                enemy_center = target_pos

                # Botão de ataque - coordenadas dinâmicas (não hardcoded!)
                if self.movement and hasattr(self.movement, 'window_w'):
                    attack_btn_x = round(self.movement.window_w * 0.90)
                    attack_btn_y = round(self.movement.window_h * 0.82)
                else:
                    attack_btn_x = round(1920 * 0.90)
                    attack_btn_y = round(1080 * 0.82)

                # Calcular direção do swipe (do botão de ataque para o alvo)
                dx = enemy_center[0] - player_center[0]
                dy = enemy_center[1] - player_center[1]
                dist = math.sqrt(dx**2 + dy**2) or 1

                # Swipe curto na direção do alvo (50-80 pixels)
                swipe_len = 60
                swipe_end_x = int(attack_btn_x + dx / dist * swipe_len)
                swipe_end_y = int(attack_btn_y + dy / dist * swipe_len)

                logger.info(f"[ATTACK] Swipe attack: ({attack_btn_x},{attack_btn_y}) -> "
                           f"({swipe_end_x},{swipe_end_y})")

                self.emulator_controller.swipe_scaled(
                    attack_btn_x, attack_btn_y,
                    swipe_end_x, swipe_end_y,
                    duration=150  # Swipe rápido
                )

                self.last_shot_time = time.time()
                self._apm_action_count += 1
                self.last_combat_snapshot = {**self.last_combat_snapshot, "attack_taken": True}
                logger.info("[ATTACK] Ataque direcional executado")

            except Exception as e:
                logger.warning(f"[ATTACK] Erro no ataque direcional, fallback para tap: {e}")
                # Fallback: simples tap no botão de ataque (coordenadas dinâmicas)
                if self.movement and hasattr(self.movement, 'window_w'):
                    self.emulator_controller.tap_scaled(
                        round(self.movement.window_w * 0.90),
                        round(self.movement.window_h * 0.82)
                    )
                else:
                    self.emulator_controller.tap_scaled(1750, 850)
                self.last_shot_time = time.time()
                self._apm_action_count += 1
                self.last_combat_snapshot = {**self.last_combat_snapshot, "attack_taken": True}

    def _try_smart_attack_with_prediction(self, player, enemies, predicted_pos):
        """
        Versao do _try_smart_attack que usa posicao predita do leading shot avancado.
        Similar ao original mas ja recebe predicted_pos calculada.
        """
        logger.debug(f"[ATTACK_AVANCADO] Ataque com leading shot predito: {predicted_pos}")
        self.get_brawler_strategy()

        # Usar cooldown específico do brawler com jitter humano
        effective_cooldown = jitter_value(self.shot_cooldown, self.shot_cooldown_jitter, min_val=0.2)
        cooldown_remaining = self.last_shot_time + effective_cooldown - time.time()
        if cooldown_remaining > 0:
            logger.debug(f"[ATTACK_AVANCADO] Cooldown restante: {cooldown_remaining:.2f}s")
            return

        # Delay de reacao humano
        reaction_time = self._human_pause.reaction_delay(base_ms=120)
        time.sleep(reaction_time)

        if not enemies:
            return

        if self.emulator_controller:
            try:
                self.emulator_controller.ensure_window_active()
                player_center = ((player[0] + player[2]) // 2, (player[1] + player[3]) // 2)
                enemy_center = predicted_pos

                # Botao de ataque
                if self.movement and hasattr(self.movement, 'window_w'):
                    attack_btn_x = round(self.movement.window_w * 0.90)
                    attack_btn_y = round(self.movement.window_h * 0.82)
                else:
                    attack_btn_x = round(1920 * 0.90)
                    attack_btn_y = round(1080 * 0.82)

                dx = enemy_center[0] - player_center[0]
                dy = enemy_center[1] - player_center[1]
                dist = math.sqrt(dx**2 + dy**2) or 1

                swipe_len = 60
                swipe_end_x = int(attack_btn_x + dx / dist * swipe_len)
                swipe_end_y = int(attack_btn_y + dy / dist * swipe_len)

                # --- SOVERANA FIX 2026-06-19: aim humanization (jitter) ---
                try:
                    aim_cfg = getattr(self, 'central_config', {}).get("aim_humanization", {}) if hasattr(self, 'central_config') else {}
                    if aim_cfg.get("enabled", True):
                        base_sigma = aim_cfg.get("base_aim_sigma_px", 3.0)
                        dist_scaling = aim_cfg.get("distance_scaling", 0.02)
                        import random as _r
                        sigma = base_sigma + (dist / 100.0) * dist_scaling
                        swipe_end_x = int(swipe_end_x + _r.gauss(0, sigma))
                        swipe_end_y = int(swipe_end_y + _r.gauss(0, sigma))
                except Exception:
                    pass
                # --- end fix ---

                self.emulator_controller.swipe_scaled(
                    attack_btn_x, attack_btn_y,
                    swipe_end_x, swipe_end_y,
                    duration=150
                )
                self.last_shot_time = time.time()
                self._apm_action_count += 1
                self.last_combat_snapshot = {**self.last_combat_snapshot, "attack_taken": True}
                logger.info(f"[ATTACK_AVANCADO] Ataque com predicao executado: {predicted_pos}")
            except Exception as e:
                logger.warning(f"[ATTACK_AVANCADO] Erro, fallback: {e}")
                self._try_smart_attack(player, enemies)

    def _execute_combo(self, player, enemies):
        """Executa proxima acao de um combo ativo."""
        if not self._combo_manager or not self.emulator_controller:
            return

        next_act = self._combo_manager.next_action()
        if not next_act:
            return

        action, delay = next_act
        logger.info(f"[COMBO] Executando: {action} (delay={delay}s)")

        if self.movement and hasattr(self.movement, 'window_w'):
            w, h = self.movement.window_w, self.movement.window_h
        else:
            w, h = self._get_safe_resolution()

        round(w * 0.90)
        round(h * 0.82)
        super_btn_x = round(w * 0.75)
        super_btn_y = round(h * 0.69)
        gadget_btn_x = round(w * 0.78)
        gadget_btn_y = round(h * 0.58)

        try:
            self.emulator_controller.ensure_window_active()
            if action == "super":
                self.emulator_controller.tap_scaled(super_btn_x, super_btn_y)
                self.last_combat_snapshot = {**self.last_combat_snapshot, "super_taken": True}
            elif action == "attack":
                self._try_smart_attack(player, enemies)
            elif action == "gadget":
                self.emulator_controller.tap_scaled(gadget_btn_x, gadget_btn_y)
                self.last_combat_snapshot = {**self.last_combat_snapshot, "gadget_taken": True}

            # NOTA: Nao fazemos time.sleep aqui. O delay entre acoes do combo
            # e naturalmente o intervalo entre chamadas de play_round (ciclo do state manager).
            # Se precisarmos de delay maior que o ciclo, usamos um sistema de agendamento futuro.
        except Exception as e:
            logger.warning(f"[COMBO] Erro ao executar combo: {e}")

    def _find_enemies(self, detections):
        """
        Find enemies in detections using unified class registry.

        Uses core.class_registry for consistent name normalization.
        Eliminates hardcoded variants (Enemy, enemy, brawler, person).
        """
        from core.class_registry import get_by_type

        logger.debug(f"[ENEMIES] Procurando inimigos em detecções: {list(detections.keys())}")

        # Use unified registry for consistent lookup
        enemies = get_by_type(detections, "enemy", first_only=False)

        logger.info(f"[ENEMIES] {len(enemies)} inimigo(s) encontrados")
        for i, enemy in enumerate(enemies):
            logger.debug(f"[ENEMIES]   Inimigo {i}: bbox={enemy}")

        # Atualizar tracker com novas detecções (mas sempre retornar deteções brutas)
        if self.enemy_tracker:
            if enemies:
                detections_with_conf = [(enemy, 0.9) for enemy in enemies]
                tracked = self.enemy_tracker.update(detections_with_conf)
                logger.debug(f"[TRACKER] {len(tracked)} inimigos rastreados confirmados")
            else:
                tracked = self.enemy_tracker.update([])
                logger.debug(f"[TRACKER] {len(tracked)} inimigos rastreados (sem novas detecções)")

        return enemies

    def _estimate_enemy_hp(self, enemy_bbox):
        """Heuristic HP estimate from bbox size (smaller = lower HP)."""
        if not enemy_bbox or len(enemy_bbox) < 4:
            return 1.0
        w = enemy_bbox[2] - enemy_bbox[0]
        h = enemy_bbox[3] - enemy_bbox[1]
        area = w * h
        # Normal brawler area ~5000-15000 px, low HP shrinks ~20%
        if area < 4000:
            return 0.25
        elif area < 6000:
            return 0.5
        elif area < 10000:
            return 0.75
        return 1.0

    def _map_utility_action_to_combat(self, action_score, player, enemies, bushes, power_cubes):
        """Map UtilityAI ActionScore to combat action dict used by play_round."""
        from decision.utility_ai import Action
        action = action_score.action
        player_c = _center(player) if player else (0, 0)

        if action == Action.ATTACK:
            # Use leading shot prediction if available
            pred_pos = None
            if enemies:
                closest = min(enemies, key=lambda e: self._distance(player, e))
                pred_pos = self._predict_position(closest)
            return {
                "action": "attack",
                "predicted_pos": pred_pos,
                "reason": f"utility_attack_{action_score.reasoning}",
            }
        elif action == Action.RETREAT:
            # Kite away from nearest enemy
            if enemies:
                closest = min(enemies, key=lambda e: self._distance(player, e))
                ec = _center(closest)
                # Move away from enemy (vector from enemy to player, extended)
                dx = player_c[0] - ec[0]
                dy = player_c[1] - ec[1]
                dist = math.sqrt(dx*dx + dy*dy) or 1
                retreat_dist = 200
                retreat_pos = (int(player_c[0] + dx/dist * retreat_dist), int(player_c[1] + dy/dist * retreat_dist))
                return {"action": "kite", "target": retreat_pos, "reason": "utility_retreat"}
            return {"action": "idle", "reason": "utility_retreat_no_enemy"}
        elif action == Action.COLLECT_CUBE:
            if power_cubes:
                target = min(power_cubes, key=lambda c: self._distance(player, c))
                return {"action": "move", "target": _center(target), "reason": "utility_collect_cube"}
            return {"action": "idle", "reason": "utility_no_cube"}
        elif action == Action.TAKE_COVER:
            if bushes:
                target = min(bushes, key=lambda b: self._distance(player, b))
                return {"action": "cover", "target": _center(target), "reason": "utility_take_cover"}
            return {"action": "idle", "reason": "utility_no_cover"}
        elif action == Action.HOLD_POSITION:
            return {"action": "idle", "reason": "utility_hold_position"}
        elif action == Action.HEAL_UP:
            # Find safest bush or just idle
            if bushes:
                target = min(bushes, key=lambda b: self._distance(player, b))
                return {"action": "cover", "target": _center(target), "reason": "utility_heal_up"}
            return {"action": "idle", "reason": "utility_heal_no_cover"}
        elif action == Action.AMBUSH:
            if bushes:
                target = min(bushes, key=lambda b: self._distance(player, b))
                return {"action": "cover", "target": _center(target), "reason": "utility_ambush"}
            return {"action": "idle", "reason": "utility_ambush_no_bush"}
        elif action == Action.CHASE:
            if enemies:
                target = min(enemies, key=lambda e: self._distance(player, e))
                pred_pos = self._predict_position(target)
                return {"action": "attack", "predicted_pos": pred_pos, "reason": "utility_chase"}
            return {"action": "idle", "reason": "utility_chase_no_enemy"}
        elif action == Action.KITE:
            # Attack while retreating
            if enemies:
                closest = min(enemies, key=lambda e: self._distance(player, e))
                ec = _center(closest)
                dx = player_c[0] - ec[0]
                dy = player_c[1] - ec[1]
                dist = math.sqrt(dx*dx + dy*dy) or 1
                retreat_dist = 150
                retreat_pos = (int(player_c[0] + dx/dist * retreat_dist), int(player_c[1] + dy/dist * retreat_dist))
                return {"action": "kite", "target": retreat_pos, "reason": "utility_kite"}
            return {"action": "idle", "reason": "utility_kite_no_enemy"}
        elif action == Action.USE_SUPER:
            pred_pos = None
            if enemies:
                closest = min(enemies, key=lambda e: self._distance(player, e))
                pred_pos = self._predict_position(closest)
            return {
                "action": "attack",
                "predicted_pos": pred_pos,
                "reason": "utility_use_super",
            }
        return {"action": "idle", "reason": "utility_unknown"}

    def _map_coordinator_action_to_combat(self, action_str, player, enemies, bushes, power_cubes):
        """Map CentralCoordinator action string to combat action dict."""
        player_c = _center(player) if player else (0, 0)

        if action_str == "attack":
            pred_pos = None
            if enemies:
                closest = min(enemies, key=lambda e: self._distance(player, e))
                pred_pos = self._predict_position(closest)
            return {"action": "attack", "predicted_pos": pred_pos, "reason": "coordinator_attack"}
        elif action_str == "retreat":
            if enemies:
                closest = min(enemies, key=lambda e: self._distance(player, e))
                ec = _center(closest)
                dx = player_c[0] - ec[0]
                dy = player_c[1] - ec[1]
                dist = math.sqrt(dx*dx + dy*dy) or 1
                retreat_pos = (int(player_c[0] + dx/dist * 200), int(player_c[1] + dy/dist * 200))
                return {"action": "kite", "target": retreat_pos, "reason": "coordinator_retreat"}
            return {"action": "idle", "reason": "coordinator_retreat_no_enemy"}
        elif action_str == "collect_cube":
            if power_cubes:
                target = min(power_cubes, key=lambda c: self._distance(player, c))
                return {"action": "move", "target": _center(target), "reason": "coordinator_collect"}
            return {"action": "idle", "reason": "coordinator_no_cube"}
        elif action_str in ("take_cover", "heal_up", "ambush"):
            if bushes:
                target = min(bushes, key=lambda b: self._distance(player, b))
                return {"action": "cover", "target": _center(target), "reason": f"coordinator_{action_str}"}
            return {"action": "idle", "reason": f"coordinator_{action_str}_no_cover"}
        elif action_str == "hold_position":
            return {"action": "idle", "reason": "coordinator_hold"}
        elif action_str == "chase":
            if enemies:
                target = min(enemies, key=lambda e: self._distance(player, e))
                pred_pos = self._predict_position(target)
                return {"action": "attack", "predicted_pos": pred_pos, "reason": "coordinator_chase"}
            return {"action": "idle", "reason": "coordinator_chase_no_enemy"}
        elif action_str == "kite":
            if enemies:
                closest = min(enemies, key=lambda e: self._distance(player, e))
                ec = _center(closest)
                dx = player_c[0] - ec[0]
                dy = player_c[1] - ec[1]
                dist = math.sqrt(dx*dx + dy*dy) or 1
                retreat_pos = (int(player_c[0] + dx/dist * 150), int(player_c[1] + dy/dist * 150))
                return {"action": "kite", "target": retreat_pos, "reason": "coordinator_kite"}
            return {"action": "idle", "reason": "coordinator_kite_no_enemy"}
        elif action_str == "use_super":
            pred_pos = None
            if enemies:
                closest = min(enemies, key=lambda e: self._distance(player, e))
                pred_pos = self._predict_position(closest)
            return {"action": "attack", "predicted_pos": pred_pos, "reason": "coordinator_super"}
        return {"action": "idle", "reason": "coordinator_unknown"}

