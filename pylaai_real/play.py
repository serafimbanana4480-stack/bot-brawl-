"""
play.py - Soberana Ultimate Combat Engine

Motor de combate avançado com:
- Predição de movimento de inimigos (Leading shots)
- Árvore de decisão de combate (HP vs Distância)
- Gestão de Supers e Gadgets
- Coleta estratégica de Power Cubes
- Multi-Object Tracking (ByteTrack/SORT)
- Brawler-Specific Strategies
"""

import time
import math
import numpy as np
import logging
import random
from collections import deque, defaultdict
from typing import Optional, List, Dict, Tuple
import sys
import os
from pathlib import Path

# Utilitarios de humanizacao
from .humanization_utils import human_delay, jitter_value, HumanPauseSimulator

# Sistema de combate avancado (Phase 5)
from .combat_advanced import (
    LeadingShotEngine, KitingEngine, CoverEngine, ComboManager, AdvancedCombatStrategy,
    _center, _pixel_distance, BRAWLER_PROJECTILES,
)

# Adicionar diretório parent ao path para importar tracker
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

class PlayLogic:
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

    def _load_brawler_strategies(self):
        """Carrega estratégias específicas por brawler do lobby.toml"""
        try:
            config_path = Path(__file__).parent.parent / "lobby.toml"
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

    def get_brawler_strategy(self) -> Dict:
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

    def get_last_combat_snapshot(self) -> Dict[str, object]:
        """Retorna o último snapshot de combate para overlay/diagnóstico."""
        snapshot = dict(self.last_combat_snapshot)

        # Se vision/state estiver disponível, adicionar resumo do estado
        if hasattr(self, 'vision_state') and self.vision_state and hasattr(self.vision_state, 'get_state_summary'):
            try:
                state_summary = self.vision_state.get_state_summary(self.vision_state)
                snapshot["vision_state_summary"] = state_summary
                logger.debug(f"[PLAY] State summary adicionado ao snapshot")
            except Exception as e:
                logger.warning(f"[PLAY] Falha ao obter state summary: {e}")

        return snapshot

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
        
    def _get_enemy_id(self, bbox) -> int:
        """Gera um ID consistente para um inimigo baseado na posição relativa"""
        # Simplificação: Como os brawlers se movem, usamos a proximidade no frame anterior
        # No futuro, um tracker real (SORT/ByteTrack) seria ideal
        bbox = self._normalize_bbox(bbox)
        if not bbox:
            return 0
        return hash((bbox[0] // 50, bbox[1] // 50, bbox[2] // 50, bbox[3] // 50))

    def _get_track_info(self, enemy_bbox) -> Optional[Dict]:
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

    def _get_enemy_tracks(self) -> List:
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

    def _predict_position(self, enemy_bbox, time_ahead=0.25) -> Tuple[int, int]:
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

    def _get_leading_shot_position(self, enemy_bbox, projectile_speed=15.0, frame_delay=0) -> Tuple[int, int]:
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

    def _get_enemy_brawler_name(self, enemy_bbox) -> Optional[str]:
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

    def _find_best_cover_position(self, player, enemies) -> Optional[Tuple[int, int]]:
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

    def _get_target_position_for_attack(self, player, enemies):
        """Calcula a posição predita do alvo para ataque (sem executar o ataque)."""
        logger.debug(f"[TARGET] Calculando posição predita do alvo")
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
        logger.debug(f"[ATTACK] Avaliando ataque")
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
        best_track_info = None

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
                best_track_info = track_info

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
                logger.info(f"[ATTACK] Ataque direcional executado")

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
        brawler_strategy = self.get_brawler_strategy()

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

        attack_btn_x = round(w * 0.90)
        attack_btn_y = round(h * 0.82)
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

    def _manage_abilities(self, player, enemies):
        """Usa Super e Gadgets estrategicamente"""
        logger.debug(f"[COMBAT] Avaliando uso de habilidades: {len(enemies)} inimigos")

        # Coordenadas dinâmicas dos botões
        if self.movement and hasattr(self.movement, 'window_w'):
            w, h = self.movement.window_w, self.movement.window_h
        else:
            w, h = self._get_safe_resolution()

        super_btn_x = round(w * 0.75)
        super_btn_y = round(h * 0.69)
        gadget_btn_x = round(w * 0.78)
        gadget_btn_y = round(h * 0.58)

        # Se houver 2+ inimigos perto, usar Super
        if len(enemies) >= 2 and self._distance(player, enemies[0]) < 300:
            if self.emulator_controller:
                logger.info("[COMBAT] SUPER ATIVADO! Multidao detectada.")
                logger.debug(f"[COMBAT] Executando tap de Super em ({super_btn_x}, {super_btn_y})")
                self.emulator_controller.ensure_window_active()
                self.emulator_controller.tap_scaled(super_btn_x, super_btn_y)
                self.last_combat_snapshot = {**self.last_combat_snapshot, "super_taken": True}
            else:
                logger.debug("[COMBAT] Super disponivel mas EmulatorController nao disponivel")
        else:
            logger.debug(f"[COMBAT] Condicoes para Super nao atendidas: enemies={len(enemies)}, dist={self._distance(player, enemies[0]) if enemies else 'N/A'}")

        # Usar Gadget quando há inimigo próximo e agressividade alta
        brawler_strategy = self.get_brawler_strategy()
        if enemies and brawler_strategy.get("has_gadget", False):
            closest_dist = min(self._distance(player, e) for e in enemies)
            if closest_dist < 400:
                if self.emulator_controller:
                    logger.info(f"[COMBAT] GADGET ATIVADO! Inimigo a {closest_dist:.0f}px.")
                    self.emulator_controller.ensure_window_active()
                    self.emulator_controller.tap_scaled(gadget_btn_x, gadget_btn_y)
                    self.last_combat_snapshot = {**self.last_combat_snapshot, "gadget_taken": True}

    def _distance(self, box1, box2):
        # box1/box2 can be: bbox [x1,y1,x2,y2], center-point (cx,cy), or tuple
        if isinstance(box1, (list, tuple)) and len(box1) >= 4:
            c1 = ((box1[0]+box1[2])//2, (box1[1]+box1[3])//2)
        elif isinstance(box1, (list, tuple)) and len(box1) >= 2:
            c1 = (box1[0], box1[1])
        else:
            c1 = (0, 0)

        if isinstance(box2, (list, tuple)) and len(box2) >= 4:
            c2 = ((box2[0]+box2[2])//2, (box2[1]+box2[3])//2)
        elif isinstance(box2, (list, tuple)) and len(box2) >= 2:
            c2 = (box2[0], box2[1])
        else:
            c2 = (0, 0)

        return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    def _find_player(self, detections):
        """
        Find player in detections using unified class registry.

        Uses core.class_registry for consistent name normalization.
        Eliminates hardcoded variants (Player, player, self, teammate, person).
        """
        from core.class_registry import get_by_type

        logger.debug(f"[PLAYER] Procurando player em detecções: {list(detections.keys())}")

        # Use unified registry for consistent lookup
        player = get_by_type(detections, "player", first_only=True)

        if player is not None:
            logger.info(f"[PLAYER] Player encontrado: bbox={player}")
            logger.debug(f"[PLAYER] Coordenadas: x={player[0]}, y={player[1]}, w={player[2]-player[0]}, h={player[3]-player[1]}")
            return player

        logger.error("[PLAYER] Player não encontrado em detecções")
        return None

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

    def _find_bushes(self, detections):
        """Find bushes to hide in using unified class registry."""
        from core.class_registry import get_by_type
        return get_by_type(detections, "bush", first_only=False)

    def _find_power_cubes(self, detections):
        """Find power cubes using unified class registry."""
        from core.class_registry import get_by_type
        # cubebox is the canonical name for power cubes
        return get_by_type(detections, "cubebox", first_only=False)

    def _execute_movement(self, key):
        if not key:
            logger.debug("[COMBAT] Nenhuma tecla de movimento fornecida")
            return
        logger.debug(f"[MOVEMENT] Iniciando movimento: key={key}")
        if not self.emulator_controller:
            logger.warning("[COMBAT] EmulatorController não disponível para executar movimento")
            return
        jx, jy = 200, 775
        dist = 100
        dx, dy = 0, 0
        if 'W' in key: dy = -dist
        if 'S' in key: dy = dist
        if 'A' in key: dx = -dist
        if 'D' in key: dx = dist
        logger.debug(f"[MOVEMENT] Posição do joystick base: ({jx}, {jy})")
        logger.debug(f"[MOVEMENT] Deslocamento calculado: dx={dx}, dy={dy}")
        logger.debug(f"[MOVEMENT] Coordenada final do swipe: ({jx+dx}, {jy+dy})")
        logger.debug(f"[MOVEMENT] Duração do swipe: 150ms")
        logger.debug(f"[COMBAT] Executando swipe de joystick: ({jx}, {jy}) -> ({jx+dx}, {jy+dy}), key={key}")
        try:
            self.emulator_controller.ensure_window_active()
            self.emulator_controller.swipe_scaled(jx, jy, jx+dx, jy+dy, duration=150)
            logger.info(f"[MOVEMENT] Swipe executado: ({jx}, {jy}) -> ({jx+dx}, {jy+dy})")
            logger.debug("[COMBAT] Movimento executado")
        except Exception as e:
            logger.error(f"[MOVEMENT] Erro ao executar movimento: key={key}, error={e}")
            logger.error(f"[MOVEMENT] EmulatorController disponível: {self.emulator_controller is not None}")

    # --- Phase 10 helper methods ---

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

    def _is_in_bush(self, player_bbox, bush_bbox):
        """Check if player bbox intersects with bush bbox."""
        if not player_bbox or not bush_bbox or len(player_bbox) < 4 or len(bush_bbox) < 4:
            return False
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = bush_bbox
        return not (px2 < bx1 or px1 > bx2 or py2 < by1 or py1 > by2)

    def _map_utility_action_to_combat(self, action_score, player, enemies, bushes, power_cubes):
        """Map UtilityAI ActionScore to combat action dict used by play_round."""
        from decision.utility_ai import Action
        action = action_score.action
        target_pos = action_score.target_position
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
