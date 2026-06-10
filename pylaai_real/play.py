"""
play.py - Soberana Ultimate Combat Engine (orquestrador)

Motor de combate avançado — orquestrador principal.
Combat logic moved to core.combat.combat_engine,
movement logic to core.movement.movement_engine,
ability logic to core.abilities.ability_manager.
"""

import time
import math
import numpy as np
import logging
import random
from collections import deque, defaultdict
from typing import Optional, List, Dict, Tuple
from pathlib import Path

# Utilitarios de humanizacao
from .humanization_utils import human_delay, jitter_value, HumanPauseSimulator

# Sistema de combate avancado (Phase 5)
from .combat_advanced import (
    LeadingShotEngine, KitingEngine, CoverEngine, ComboManager, AdvancedCombatStrategy,
    _center, _pixel_distance, BRAWLER_PROJECTILES,
)

logger = logging.getLogger(__name__)

try:
    from tracker import EnemyTracker
    TRACKER_AVAILABLE = True
    logger.debug("[PLAY] EnemyTracker importado com sucesso")
except ImportError:
    TRACKER_AVAILABLE = False
    logger.warning("[PLAY] EnemyTracker não disponível - tracking de inimigos desativado")
    logger.warning("[PLAY] EnemyTracker não disponível, usando enemy_history fallback")

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
    from core.central_coordinator import (
        CentralCoordinator, Recommendation, DecisionType, Priority
    )
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

from core.combat.combat_engine import CombatEngineMixin
from core.movement.movement_engine import MovementEngineMixin
from core.abilities.ability_manager import AbilityManagerMixin


class PlayLogic(CombatEngineMixin, MovementEngineMixin, AbilityManagerMixin):
    """Motor de combate avançado — orquestrador principal."""

    def __init__(
        self,
        detect_main,
        detect_enemies=None,
        movement=None,
        humanization=None,
        emulator_controller=None,
        enemy_tracker=None,
        rl_engine=None,
        central_coordinator=None,
        world_model=None,
        pressure_map=None,
        enemy_intention=None,
        meta_awareness=None,
        cover_system=None,
        world_model_integrator=None,
    ):
        self.detect_main = detect_main
        self.detect_enemies = detect_enemies
        self.movement = movement
        self.humanization = humanization
        self.emulator_controller = emulator_controller
        self.rl_engine = rl_engine
        self.central_coordinator = central_coordinator
        self.world_model = world_model
        self.pressure_map = pressure_map
        self.enemy_intention = enemy_intention
        self.meta_awareness = meta_awareness
        self.cover_system = cover_system
        self.world_model_integrator = world_model_integrator

        # State tracking para predição (deque with maxlen prevents memory leak)
        self.enemy_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=5))
        self.last_shot_time = 0
        self.shot_cooldown = 0.35  # Reduzido para melhorar responsividade (era 0.45)
        self.shot_cooldown_jitter = 0.15  # Variacao de +/-15% no cooldown
        self.super_ready = False
        self._human_pause = HumanPauseSimulator()
        self._apm_action_count = 0
        self._apm_window_start = time.time()
        self._last_rl_state: Optional[Tuple] = None
        self._last_rl_action: Optional[str] = None
        self.last_rl_transition: Optional[Tuple] = None  # (state, action, reward, next_state)
        self.current_game_mode: str = "showdown"
        self.pve_mode: Optional[str] = None  # None = PvP, ou "training_cave", "robo_rumble", etc.

        # NOVO: Sistema de combate avancado (Phase 5)
        self._combat_strategy: Optional[AdvancedCombatStrategy] = None
        self._leading_engine: Optional[LeadingShotEngine] = None
        self._combo_manager: Optional[ComboManager] = None

        # Phase 10: Advanced Decision Modules
        self._utility_ai: Optional[Any] = None
        if HAS_UTILITY_AI:
            try:
                self._utility_ai = UtilityAI()
                logger.info("[PLAY] UtilityAI inicializado")
            except Exception as e:
                logger.warning(f"[PLAY] UtilityAI indisponível: {e}")

        self._sticky_target: Optional[Any] = None
        if HAS_STICKY_TARGET:
            try:
                self._sticky_target = StickyTarget()
                logger.info("[PLAY] StickyTarget inicializado")
            except Exception as e:
                logger.warning(f"[PLAY] StickyTarget indisponível: {e}")

        self._intent_system: Optional[Any] = None
        if HAS_INTENT_SYSTEM:
            try:
                self._intent_system = IntentSystem()
                logger.info("[PLAY] IntentSystem inicializado")
            except Exception as e:
                logger.warning(f"[PLAY] IntentSystem indisponível: {e}")

        # Phase 10: MetaAwareness (FIX #8: removed local fallback - wrapper must provide)
        # If wrapper doesn't pass it, meta_awareness stays None (no duplicate creation)
        if self.meta_awareness is not None:
            logger.info("[PLAY] MetaAwareness recebido do wrapper")
        elif HAS_META_AWARENESS:
            logger.warning("[PLAY] MetaAwareness não fornecido pelo wrapper - combat não terá matchup awareness")

        # Phase 10: CoverSystem (FIX #8: removed local fallback - wrapper must provide)
        # If wrapper doesn't pass it, cover_system stays None (no duplicate creation)
        if self.cover_system is not None:
            logger.info("[PLAY] CoverSystem recebido do wrapper")
        elif HAS_COVER_SYSTEM:
            logger.warning("[PLAY] CoverSystem não fornecido pelo wrapper - combat não terá cover evaluation")

        # Phase 10: Game Feature Extractor (walls, HP, bushes)
        self._feature_extractor: Optional[Any] = None
        if HAS_FEATURE_EXTRACTOR:
            try:
                self._feature_extractor = GameFeatureExtractor()
                logger.info("[PLAY] GameFeatureExtractor inicializado")
            except Exception as e:
                logger.warning(f"[PLAY] GameFeatureExtractor indisponível: {e}")

        # Phase 10: Occupancy Grid (A* pathfinding)
        self._occupancy_grid: Optional[Any] = None
        if HAS_OCCUPANCY_GRID:
            try:
                screen_w = 1920
                screen_h = 1080
                if movement and hasattr(movement, 'window_w'):
                    screen_w = movement.window_w
                    screen_h = movement.window_h
                self._occupancy_grid = OccupancyGrid(map_width=screen_w, map_height=screen_h)
                logger.info("[PLAY] OccupancyGrid inicializado")
            except Exception as e:
                logger.warning(f"[PLAY] OccupancyGrid indisponível: {e}")

        # Feature flags para integração Phase 10 (permite rollback rápido)
        self._enable_utility_ai = True
        self._utility_ai_threshold = 0.70  # FIX #11: Score mínimo para override (era 0.45 - muito baixo)
        self._enable_intent_system = True
        self._enable_enemy_intention = True

        # Parâmetros ajustáveis para auto-tuning
        self.attack_distance = 200  # Distância ideal de ataque (pixels)
        self.aggressiveness = 0.5  # Nível de agressividade (0.0-1.0)

        # Multi-Object Tracking (ByteTrack/SORT)
        # Se um tracker já pronto for passado via parâmetro, usa-se ele diretamente.
        if enemy_tracker is not None:
            self.enemy_tracker = enemy_tracker
            logger.info("[PLAY] EnemyTracker fornecido externamente")
        elif TRACKER_AVAILABLE:
            self.enemy_tracker = EnemyTracker(max_age=30, min_hits=2, use_advanced_prediction=True)
            logger.info("[PLAY] EnemyTracker inicializado com predição avançada")
        else:
            self.enemy_tracker = None
            logger.warning("[PLAY] EnemyTracker não disponível")

        # Brawler-specific strategies
        self.current_brawler = None
        self.brawler_strategy = None
        self._load_brawler_strategies()

        # Dashboard tracking (real data, zero mocks)
        self._last_action = "idle"
        self._last_enemies = 0
        self._game_phase = "early"
        self._match_start_time = None
        self._power_cubes_collected = 0

        self.last_combat_snapshot: Dict[str, object] = {
            "state": "idle",
            "player": None,
            "enemies": 0,
            "bushes": 0,
            "power_cubes": 0,
            "move_key": "",
            "attack_taken": False,
            "super_taken": False,
            "window_active": None,
            "window_title": None,
            "target_position": None,
            "last_error": None,
        }

    def _get_safe_resolution(self) -> tuple:
        """Retorna resolucao segura, consultando ResolutionManager se disponivel."""
        try:
            from core.resolution_manager import ResolutionManager
            rm = ResolutionManager()
            rm.detect()
            if rm.profile.is_reasonable():
                return rm.actual_resolution
        except Exception:
            pass
        return (1920, 1080)

    def _snapshot_window_state(self) -> Dict[str, object]:
        """Lê o estado da janela do emulador, se existir, para diagnosticar input perdido."""
        if not self.emulator_controller:
            return {"window_active": None, "window_title": None}

        try:
            snapshot = self.emulator_controller.get_status_snapshot()
            return {
                "window_active": snapshot.get("window_active"),
                "window_title": snapshot.get("window_title"),
            }
        except Exception as e:
            logger.debug(f"[COMBAT] Falha ao ler snapshot da janela: {e}")
            return {"window_active": None, "window_title": None}

    def _normalize_bbox(self, bbox):
        """Garante bbox no formato [x1, y1, x2, y2]."""
        if bbox is None:
            return None
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        if isinstance(bbox, tuple):
            bbox = list(bbox)
        if not isinstance(bbox, list) or len(bbox) < 4:
            return None
        try:
            return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        except Exception:
            return None

    def _normalize_detection_map(self, detections):
        """Aceita tanto o wrapper Detect quanto saídas brutas/legadas e normaliza para dict[str, list[bbox]]."""
        if detections is None:
            return {}

        if isinstance(detections, dict):
            normalized = {}
            for name, boxes in detections.items():
                clean_boxes = []
                for box in boxes or []:
                    nb = self._normalize_bbox(box)
                    if nb:
                        clean_boxes.append(nb)
                if clean_boxes:
                    normalized[name] = clean_boxes
            return normalized

        # Support list of detection-like objects from newer vision engine output
        normalized = {}
        for det in detections if isinstance(detections, list) else []:
            class_name = getattr(det, "class_name", None) or getattr(det, "class", None) or "unknown"
            bbox = None
            if all(hasattr(det, attr) for attr in ("x", "y", "width", "height")):
                bbox = [int(det.x), int(det.y), int(det.x + det.width), int(det.y + det.height)]
            elif all(hasattr(det, attr) for attr in ("center_x", "center_y", "width", "height")):
                bbox = [
                    int(det.center_x - det.width / 2),
                    int(det.center_y - det.height / 2),
                    int(det.center_x + det.width / 2),
                    int(det.center_y + det.height / 2),
                ]
            if bbox:
                normalized.setdefault(class_name, []).append(bbox)
        return normalized

    def play_round(self, screenshot: np.ndarray) -> dict:
        """
        Executa um round de combate completo.
        Retorna dict com flags de ação: {"attacked": bool, "moved": bool, "super_used": bool, "success": bool}
        """
        logger.debug("[COMBAT] Iniciando play_round")
        if log_manager:
            log_manager.log(
                message="Iniciando round de combate",
                level="DEBUG",
                category="combat",
                data={"action": "play_round_start"}
            )
        if self.detect_main is None:
            logger.warning("[COMBAT] No vision model loaded - skipping combat logic")
            self.last_combat_snapshot = {
                **self.last_combat_snapshot,
                "state": "skipped_no_model",
                "last_error": "no_vision_model",
            }
            return {"attacked": False, "moved": False, "super_used": False, "success": False, "error": "no_vision_model"}
        try:
            logger.debug("[COMBAT] Executando detecção de objetos")
            detections = self.detect_main.detect_objects(screenshot)
            detections = self._normalize_detection_map(detections)
            logger.info(f"[COMBAT] Detecções brutas: {list(detections.keys())}")
            logger.debug(f"[COMBAT] Detalhes das detecções: {detections}")
            if log_manager:
                log_manager.log(
                    message=f"Detecções executadas: {list(detections.keys())}",
                    level="INFO",
                    category="combat",
                    data={"detections": list(detections.keys()), "counts": {k: len(v) for k, v in detections.items()}}
                )
            
            player = self._find_player(detections)
            if player is None:
                # Fallback: assume player is at joystick area and try to play anyway
                logger.warning("[COMBAT] Jogador não detectado! Usando posição assumida.")
                player = (self.movement.joystick_center_x, self.movement.joystick_center_y) if self.movement else (192, 810)
                # Don't abort - try to play with assumed position

            logger.info(f"[COMBAT] Jogador detectado na posição: {player}")

            enemies = self._find_enemies(detections)
            bushes = self._find_bushes(detections)
            power_cubes = self._find_power_cubes(detections)
            logger.info(f"[COMBAT] Inimigos detectados: {len(enemies)}")

            # Phase 10: WorldModelIntegrator - update spatial memory
            if self.world_model_integrator and enemies:
                try:
                    self.world_model_integrator.update_from_detections(
                        enemies=enemies,
                        player=player,
                        power_cubes=power_cubes,
                        bushes=bushes,
                        walls=extracted_walls if extracted_walls else None,
                        player_hp=real_hp if real_hp else 1.0,
                    )
                    logger.debug(f"[WORLD_INTEG] WorldModel atualizado: {len(enemies)} enemies, {len(power_cubes)} cubes")
                except Exception as e:
                    logger.debug(f"[WORLD_INTEG] Update error: {e}")

            # Phase 10: EnemyIntention - update enemy intentions
            if self.enemy_intention and enemies and player:
                try:
                    player_center = _center(player)
                    enemy_dicts = [{"track_id": i, "x": _center(e)[0], "y": _center(e)[1],
                                    "width": e[2]-e[0], "height": e[3]-e[1]}
                                   for i, e in enumerate(enemies)]
                    self.enemy_intention.update(enemy_dicts, player_center)
                    rushers = self.enemy_intention.get_rushers()
                    flankers = self.enemy_intention.get_flankers()
                    baiters = self.enemy_intention.get_baiters()
                    if rushers or flankers or baiters:
                        logger.info(f"[ENEMY_INTENT] rushers={len(rushers)}, flankers={len(flankers)}, baiters={len(baiters)}")
                except Exception as e:
                    logger.debug(f"[ENEMY_INTENT] Update error: {e}")
            logger.debug(f"[COMBAT] Bushes detectados: {len(bushes)}")
            if log_manager:
                log_manager.log(
                    message=f"Detecção de combate: {len(enemies)} inimigos",
                    level="INFO",
                    category="combat",
                    data={"enemies_count": len(enemies), "bushes_count": len(bushes), "player_position": player}
                )

            # Encontrar inimigo mais próximo
            if enemies:
                for i, enemy in enumerate(enemies):
                    dist = self._distance(player, enemy)
                    logger.debug(f"[COMBAT] Inimigo {i}: distância={dist:.1f}px, bbox={enemy}")
                closest_enemy = min(enemies, key=lambda e: self._distance(player, e))
                closest_dist = self._distance(player, closest_enemy)
                # Phase 10: StickyTarget - manter compromisso com alvo
                if self._sticky_target and enemies:
                    try:
                        committed = self._sticky_target.select(
                            player, enemies, current_target=closest_enemy
                        )
                        if committed:
                            closest_enemy = committed
                            closest_dist = self._distance(player, committed)
                            logger.info(f"[STICKY_TARGET] Alvo comprometido: dist={closest_dist:.1f}px")
                    except Exception as e:
                        logger.debug(f"[STICKY_TARGET] Erro: {e}")
                logger.info(f"[COMBAT] Inimigo mais próximo: distância={closest_dist:.1f}px, bbox={closest_enemy}")

            # --- Phase 10: Extract game features (walls, HP, bushes) ---
            extracted_walls = []
            real_hp = 1.0
            if self._feature_extractor and screenshot is not None:
                try:
                    player_bbox = player if player else None
                    features = self._feature_extractor.extract_features(screenshot, player_bbox)
                    extracted_walls = features.get("walls", [])
                    real_hp = features.get("player_hp", 1.0)
                    # Use real HP if available
                    if real_hp < 1.0 and self._combat_strategy:
                        self._combat_strategy.update_hp_estimate(real_hp)
                    # Merge extracted walls with YOLO-detected walls
                    if extracted_walls:
                        if walls is None:
                            walls = []
                        wall_bboxes = [w["bbox"] for w in extracted_walls]
                        walls = walls + wall_bboxes
                        logger.debug(f"[FEATURES] +{len(extracted_walls)} walls, HP={real_hp:.2f}")
                except Exception as e:
                    logger.debug(f"[FEATURES] Extraction error: {e}")

            # --- Atualizar fase de jogo ---
            if self._match_start_time:
                elapsed = time.time() - self._match_start_time
                if elapsed < 30:
                    self._game_phase = "early"
                elif elapsed < 90:
                    self._game_phase = "mid"
                else:
                    self._game_phase = "late"
            # Atualizar contador de power cubes coletados
            if power_cubes and len(power_cubes) < getattr(self, '_last_cube_count', 999):
                self._power_cubes_collected += 1
            self._last_cube_count = len(power_cubes) if power_cubes else 0

            window_snapshot = self._snapshot_window_state()
            logger.info(
                f"[COMBAT] Window snapshot: active={window_snapshot.get('window_active')}, "
                f"title={window_snapshot.get('window_title')}, "
                f"enemies={len(enemies) if enemies else 0}, bushes={len(bushes)}, cubes={len(power_cubes)}"
            )

            if self.emulator_controller and window_snapshot.get("window_active") is False:
                logger.debug("[COMBAT] Janela do emulador não está ativa; tentando ativar antes de agir")
                activated = self.emulator_controller.ensure_window_active()
                logger.info(f"[COMBAT] Resultado da ativação da janela: {activated}")
            elif self.emulator_controller:
                logger.debug("[COMBAT] Janela do emulador já está ativa")
            else:
                logger.warning("[COMBAT] EmulatorController não disponível para verificar foco de janela")

            target_position = None

            # === DECISAO Q-LEARNING (NOVO) ===
            rl_state = self._get_rl_state(
                player, enemies,
                can_attack=True,
                can_super=self.super_ready,
                real_hp=real_hp,
            )
            rl_action = None
            rl_confidence = 0.0
            if rl_state and self.rl_engine:
                # v2.3: Passar dados extras para NeuralPolicy (player_pos, enemies, detections)
                if player and len(player) >= 4:
                    player_center = (
                        ((player[0] + player[2]) / 2) / self._get_safe_resolution()[0],
                        ((player[1] + player[3]) / 2) / self._get_safe_resolution()[1],
                    )
                elif player and len(player) >= 2:
                    res = self._get_safe_resolution()
                    player_center = (player[0] / res[0], player[1] / res[1])
                else:
                    player_center = (0.5, 0.5)
                detections = {"enemy": enemies, "player": [player] if player else []}
                rl_action, rl_confidence = self.rl_engine.get_action(
                    rl_state,
                    player_pos=player_center,
                    enemies=enemies,
                    detections=detections,
                )
                logger.info(f"[RL] Estado={rl_state}, acao={rl_action}, conf={rl_confidence:.2f}")
                self._last_rl_state = rl_state
                self._last_rl_action = rl_action

            # 0. IntentSystem: avaliar intenção estratégica persistente (Phase 10)
            current_intent_value = None
            if self._enable_intent_system and self._intent_system:
                try:
                    intent_context = {
                        "health": self._combat_strategy._hp_estimate if self._combat_strategy else 1.0,
                        "enemies_nearby": len(enemies) if enemies else 0,
                        "pressure": 0.0,
                        "match_phase": self._game_phase,
                        "has_super": self.super_ready,
                        "cube_count": len(power_cubes) if power_cubes else 0,
                        "alive_allies": 0,
                        "is_in_bush": any(self._is_in_bush(player, b) for b in bushes) if bushes else False,
                        "brawler_role": self.current_brawler if self.current_brawler else "damage",
                    }
                    current_intent = self._intent_system.evaluate(intent_context)
                    current_intent_value = current_intent.value if current_intent else None
                    logger.info(f"[INTENT] Intenção: {current_intent_value}")
                except Exception as e:
                    logger.debug(f"[INTENT] Erro: {e}")

            # 0.5. UtilityAI: decisão scored de alto nível (Phase 10)
            utility_decision = None
            utility_override = False
            if self._enable_utility_ai and self._utility_ai and enemies:
                try:
                    hp = self._combat_strategy._hp_estimate if self._combat_strategy else 1.0
                    nearest_enemy = min(enemies, key=lambda e: self._distance(player, e))
                    nearest_dist = self._distance(player, nearest_enemy)
                    enemy_hp = self._estimate_enemy_hp(nearest_enemy)
                    nearest_cube = None
                    nearest_cube_dist = None
                    if power_cubes:
                        nearest_cube = min(power_cubes, key=lambda c: self._distance(player, c))
                        nearest_cube_dist = self._distance(player, nearest_cube)
                    has_cover = bool(bushes) or bool(extracted_walls)
                    cover_pos = None
                    if bushes:
                        cover_pos = _center(min(bushes, key=lambda b: self._distance(player, b)))
                    best_cover_pos = self._find_best_cover_position(player, enemies)
                    utility_context = {
                        "health": hp,
                        "ammo": 3,
                        "max_ammo": 3,
                        "super_charged": self.super_ready,
                        "game_mode": self.current_game_mode,
                        "enemies_nearby": len(enemies),
                        "nearest_enemy_dist": nearest_dist,
                        "nearest_enemy_health": enemy_hp,
                        "player_position": _center(player) if player else (0, 0),
                        "pressure": self._get_pressure(player) if player else 0.0,
                        "danger": self._get_danger(player) if player else 0.0,
                        "has_cover_nearby": has_cover,
                        "cover_position": cover_pos,
                        "best_cover_position": best_cover_pos,
                        "nearest_cube_dist": nearest_cube_dist,
                        "cube_position": _center(nearest_cube) if nearest_cube else None,
                        "match_phase": self._game_phase,
                        "brawler_role": self.current_brawler if self.current_brawler else "damage",
                        "intent": current_intent_value,
                        "matchup_advantage": self._get_matchup_advantage(),
                        "should_kite": self._should_kite_from_matchup(),
                        "should_rush": self._should_rush_from_matchup(),
                        "aggression_modifier": self._get_aggression_modifier(),
                    }
                    utility_decision = self._utility_ai.evaluate(utility_context)
                    logger.info(
                        f"[UTILITY_AI] Ação={utility_decision.action.value}, score={utility_decision.score:.2f}, "
                        f"urgency={utility_decision.urgency:.2f}, reason={utility_decision.reasoning}"
                    )
                    if utility_decision.score >= self._utility_ai_threshold:
                        utility_override = True
                        logger.info(f"[UTILITY_AI] Override ativado (score={utility_decision.score:.2f})")
                except Exception as e:
                    logger.debug(f"[UTILITY_AI] Erro: {e}")

            # 0.75. CentralCoordinator: submit recommendations and resolve conflicts (Phase 10)
            coordinator_action = None
            coordinator_target = None
            if HAS_COORDINATOR and self.central_coordinator and enemies:
                try:
                    # Set current intent
                    if current_intent_value:
                        self.central_coordinator.set_intent(current_intent_value)

                    # Submit UtilityAI recommendation
                    if utility_decision:
                        self.central_coordinator.submit_recommendation(
                            Recommendation(
                                source="utility_ai",
                                decision_type=DecisionType.ACTION,
                                action=utility_decision.action.value,
                                priority=Priority.HIGH if utility_decision.urgency > 0.7 else Priority.MEDIUM,
                                confidence=min(1.0, max(0.0, utility_decision.score)),
                                reason=utility_decision.reasoning,
                                context={"target_position": utility_decision.target_position},
                            )
                        )

                    # Submit StickyTarget recommendation
                    if self._sticky_target and closest_enemy:
                        self.central_coordinator.submit_recommendation(
                            Recommendation(
                                source="sticky_target",
                                decision_type=DecisionType.TARGET,
                                action=closest_enemy,
                                priority=Priority.MEDIUM,
                                confidence=0.8,
                                reason="committed_target",
                            )
                        )

                    # Submit RL recommendation
                    if rl_action and rl_confidence > 0.5:
                        self.central_coordinator.submit_recommendation(
                            Recommendation(
                                source="rl_engine",
                                decision_type=DecisionType.ACTION,
                                action=rl_action,
                                priority=Priority.MEDIUM,
                                confidence=rl_confidence,
                                reason="q_learning",
                            )
                        )

                    # Resolve all recommendations
                    coordinated = self.central_coordinator.resolve()
                    action_decision = coordinated.get(DecisionType.ACTION)
                    if action_decision:
                        coordinator_action = action_decision.action
                        logger.info(
                            f"[COORDINATOR] Ação unificada: {coordinator_action} "
                            f"(fonte={action_decision.source}, razao={action_decision.reason})"
                        )
                        # Check if coordinator overrode UtilityAI (learning opportunity)
                        if utility_override and action_decision.source != "utility_ai":
                            logger.info(
                                f"[COORDINATOR] Override cruzado: UtilityAI -> {action_decision.source}"
                            )
                except Exception as e:
                    logger.debug(f"[COORDINATOR] Erro: {e}")

            # 1. Decisão de Combate: Phase 5 strategy + Phase 10 UtilityAI/Coordinator override
            attack_taken = False
            moved = False
            super_used = False
            action = "idle"
            move_target = None
            if coordinator_action and HAS_COORDINATOR and self.central_coordinator:
                # Phase 10: CentralCoordinator has unified decision from multiple subsystems
                decision = self._map_coordinator_action_to_combat(
                    coordinator_action, player, enemies, bushes, power_cubes
                )
                action = decision.get("action", "idle")
                logger.info(f"[COORDINATOR] Ação mapeada: {action}, razao: {decision.get('reason')}")
            elif utility_override and utility_decision:
                # Phase 10: UtilityAI override with high confidence
                decision = self._map_utility_action_to_combat(
                    utility_decision, player, enemies, bushes, power_cubes
                )
                action = decision.get("action", "idle")
                logger.info(f"[UTILITY_AI] Ação mapeada: {action}, razao: {decision.get('reason')}")
            elif self._combat_strategy and self.emulator_controller:
                decision = self._combat_strategy.decide_combat_action(
                    player, enemies, bushes, power_cubes
                )
                action = decision.get("action", "idle")
                logger.info(f"[COMBAT_AVANCADO] Decisao: {action}, razao: {decision.get('reason')}")
            else:
                decision = {}

            if action == "attack":
                pred_pos = decision.get("predicted_pos")
                if pred_pos:
                    self._try_smart_attack_with_prediction(player, enemies, pred_pos)
                else:
                    self._try_smart_attack(player, enemies)
                attack_taken = True
            elif action == "kite":
                move_target = decision.get("target")
                self._try_smart_attack(player, enemies)
                attack_taken = True
            elif action == "cover":
                move_target = decision.get("target")
            elif action == "combo":
                self._execute_combo(player, enemies)
                attack_taken = True
            elif action == "move":
                move_target = decision.get("target")
            elif action == "idle":
                pass

            if not (utility_override or (self._combat_strategy and self.emulator_controller)):
                # Fallback: ataque básico quando não há combat strategy nem UtilityAI override
                if enemies and self.emulator_controller:
                    self._try_smart_attack(player, enemies)
                    attack_taken = True
                    action = "attack"
                    logger.info("[COMBAT] Fallback: ataque básico (sem combat_strategy)")

            # 2. Movimento Tático: se combat não ditou movimento, calcular um default
            logger.debug("[COMBAT] Calculando movimento tático")
            if move_target and self.movement:
                self.movement.move_to_position(move_target[0], move_target[1])
                moved = True
                move_key = self.movement.get_tactical_movement(player, enemies, bushes, power_cubes)
                logger.info(f"[COMBAT] Movimento combat -> ({move_target[0]:.0f}, {move_target[1]:.0f}), dir={move_key}")
            else:
                use_rl = rl_action is not None and rl_confidence > 0.6
                if use_rl:
                    move_key = self._apply_rl_action(rl_action, player, enemies, power_cubes, "")
                    moved = True  # RL action implies movement
                    logger.info(f"[COMBAT] Movimento RL (conf={rl_confidence:.2f}): {rl_action}")
                else:
                    if self.movement and hasattr(self.movement, 'get_tactical_movement_target'):
                        target_pos = self.movement.get_tactical_movement_target(
                            player, enemies, bushes, power_cubes,
                            game_phase=self._game_phase,
                            player_hp_estimate=self._combat_strategy._hp_estimate if self._combat_strategy else 1.0
                        )
                        if target_pos:
                            self.movement.move_to_position(target_pos[0], target_pos[1])
                            moved = True
                            move_key = self.movement.get_tactical_movement(player, enemies, bushes, power_cubes)
                            logger.info(f"[COMBAT] Movimento tático -> ({target_pos[0]:.0f}, {target_pos[1]:.0f}), dir={move_key}")
                        else:
                            move_key = self.movement.get_tactical_movement(player, enemies, bushes, power_cubes)
                            if move_key:
                                moved = True
                            logger.info(f"[COMBAT] Decisão de movimento (fallback): {move_key}")
                    else:
                        move_key = self.movement.get_tactical_movement(player, enemies, bushes, power_cubes) if self.movement else ""
                        if move_key:
                            moved = True
                        logger.info(f"[COMBAT] Decisão de movimento: {move_key}")

            # Log de tracks por classe usando get_tracks_by_class se disponível
            if self.enemy_tracker and TRACKER_AVAILABLE:
                enemy_tracks = self._get_enemy_tracks()
                if enemy_tracks:
                    logger.debug(f"[COMBAT] {len(enemy_tracks)} tracks de classe 'enemy' encontrados")



            # Calcular reward heuristico para este frame (para RL online)
            frame_reward = self._compute_frame_reward(player, enemies, attack_taken)

            # Guardar transicao para o state_manager chamar learn_from_frame
            if rl_state and rl_action and self.rl_engine:
                next_rl_state = self._get_rl_state(
                    player, enemies,
                    can_attack=True, can_super=self.super_ready,
                )
                # v2.3: Guardar transição com dados extras para NeuralPolicy
                if player and len(player) >= 4:
                    player_center = (
                        ((player[0] + player[2]) / 2) / self._get_safe_resolution()[0],
                        ((player[1] + player[3]) / 2) / self._get_safe_resolution()[1],
                    )
                elif player and len(player) >= 2:
                    res = self._get_safe_resolution()
                    player_center = (player[0] / res[0], player[1] / res[1])
                else:
                    player_center = (0.5, 0.5)
                detections = {"enemy": enemies, "player": [player] if player else []}
                self.last_rl_transition = {
                    "state": rl_state,
                    "action": rl_action,
                    "reward": frame_reward,
                    "next_state": next_rl_state,
                    "player_pos": player_center,
                    "enemies": enemies,
                    "detections": detections,
                }
            else:
                self.last_rl_transition = None

            # Atualizar dados do dashboard (real data)
            self._last_action = action
            self._last_enemies = len(enemies) if enemies else 0

            self.last_combat_snapshot = {
                **self.last_combat_snapshot,
                **window_snapshot,
                "state": "combat_ok",
                "player": player,
                "enemies": enemies,
                "move_key": move_key,
                "attack_taken": attack_taken,
                "moved": moved,
                "super_used": super_used,
            }

            return {"attacked": attack_taken, "moved": moved, "super_used": super_used, "success": True}
        except Exception as e:
            logger.error(f"Erro no cérebro de combate: {e}", exc_info=True)
            self._last_action = "error"
            self.last_combat_snapshot = {
                **self.last_combat_snapshot,
                "state": "combat_error",
                "last_error": str(e),
            }
            return {"attacked": False, "moved": False, "super_used": False, "success": False, "error": str(e)}

