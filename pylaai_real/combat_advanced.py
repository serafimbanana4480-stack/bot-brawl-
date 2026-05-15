"""
combat_advanced.py  v2.0

Sistema de combate avancado (Phase 5 — Melhorado):
- Leading shot com filtro EMA de velocidade e aim-error model humano
- Kiting multi-inimigo com vetor resultante e preferencia por bushes
- Cover com escape routes, power cubes e quality scoring
- Combos condicionais com delays adaptativos
- Target selection inteligente e combat state machine
"""

import time
import math
import random
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


def _center(box) -> Tuple[float, float]:
    if box is None:
        return (0.0, 0.0)
    if len(box) >= 4:
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return (float(box[0]), float(box[1]))


def _pixel_distance(a, b) -> float:
    c1, c2 = _center(a), _center(b)
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# 5.1  LEADING SHOT  (v2 — EMA filter + human aim-error model)
# ---------------------------------------------------------------------------

BRAWLER_PROJECTILES = {
    "shelly":     {"speed": 5.0,  "width": 1.5, "aim_precision": 0.85},
    "colt":       {"speed": 8.0,  "width": 0.5, "aim_precision": 0.90},
    "piper":      {"speed": 3.5,  "width": 0.5, "aim_precision": 0.95},
    "brock":      {"speed": 3.0,  "width": 1.0, "aim_precision": 0.88},
    "nita":       {"speed": 5.0,  "width": 1.5, "aim_precision": 0.85},
    "el primo":   {"speed": 4.0,  "width": 1.0, "aim_precision": 0.70},
    "dynamike":   {"speed": 2.5,  "width": 2.0, "aim_precision": 0.75},
    "jessie":     {"speed": 5.0,  "width": 1.0, "aim_precision": 0.85},
    "rico":       {"speed": 7.0,  "width": 0.4, "aim_precision": 0.92},
    "edgar":      {"speed": 4.5,  "width": 1.0, "aim_precision": 0.80},
    "mortis":     {"speed": 6.0,  "width": 1.0, "aim_precision": 0.78},
    "tick":       {"speed": 2.0,  "width": 2.0, "aim_precision": 0.72},
    "8-bit":      {"speed": 5.0,  "width": 0.5, "aim_precision": 0.88},
    "emz":        {"speed": 4.5,  "width": 2.0, "aim_precision": 0.82},
    "be":         {"speed": 5.0,  "width": 1.0, "aim_precision": 0.85},
    "spike":      {"speed": 3.5,  "width": 1.5, "aim_precision": 0.87},
    "crow":       {"speed": 4.0,  "width": 0.5, "aim_precision": 0.88},
    "leon":       {"speed": 5.0,  "width": 0.5, "aim_precision": 0.88},
    "darryl":     {"speed": 5.0,  "width": 1.0, "aim_precision": 0.82},
    "rosa":       {"speed": 4.5,  "width": 1.0, "aim_precision": 0.80},
    "poco":       {"speed": 5.0,  "width": 2.5, "aim_precision": 0.85},
    "bo":         {"speed": 5.0,  "width": 0.5, "aim_precision": 0.88},
    "barley":     {"speed": 2.5,  "width": 1.5, "aim_precision": 0.80},
    "pamu":       {"speed": 4.0,  "width": 1.0, "aim_precision": 0.82},
    "default":    {"speed": 5.0,  "width": 1.0, "aim_precision": 0.85},
}

# --- aim-error model ---
# Humanos nao erram uniformemente.  Tendem a:
#   1. overshoot  (atirar A FRENTE demais do inimigo que se afasta)
#   2. ter erro maior a longa distancia
#   3. ter clustering (varios tiros seguidos com erro similar)

class HumanAimError:
    """Modela erro de mira humano com bias direcional e clustering temporal."""

    def __init__(self, precision: float = 0.85):
        self.precision = precision          # 0..1  (1 = perfeito)
        self._last_error_x = 0.0
        self._last_error_y = 0.0
        self._cluster_decay = 0.6           # quanto do erro anterior persiste

    def apply(self, target_x: float, target_y: float,
              enemy_vel_x: float, enemy_vel_y: float,
              dist_px: float) -> Tuple[float, float]:
        """
        Retorna (aim_x, aim_y) com erro humano realistico aplicado.
        """
        # Erro base depende da distancia: mais longe = maior erro
        base_error = dist_px * (1.0 - self.precision) * 0.08

        # Bias direcional: humanos overshoot quando inimigo se afasta
        # (atiram na direcao da velocidade com magnitude exagerada)
        overshoot_bias_x = enemy_vel_x * random.uniform(0.02, 0.06)
        overshoot_bias_y = enemy_vel_y * random.uniform(0.02, 0.06)

        # Componente aleatorio (gaussiano truncado)
        err_x = random.gauss(0, base_error * 0.5)
        err_y = random.gauss(0, base_error * 0.5)
        err_x = _clamp(err_x, -base_error * 2, base_error * 2)
        err_y = _clamp(err_y, -base_error * 2, base_error * 2)

        # Clustering temporal: erro atual correlacionado com o anterior
        err_x = self._last_error_x * self._cluster_decay + err_x * (1 - self._cluster_decay)
        err_y = self._last_error_y * self._cluster_decay + err_y * (1 - self._cluster_decay)
        self._last_error_x = err_x
        self._last_error_y = err_y

        aim_x = target_x + err_x + overshoot_bias_x
        aim_y = target_y + err_y + overshoot_bias_y
        return (aim_x, aim_y)


class LeadingShotEngine:
    """Predicao de posicao com filtro EMA de velocidade e aim-error humano."""

    def __init__(self, brawler_name: str = "default", tile_size_px: float = 80.0):
        self.brawler_name = str(brawler_name).lower().strip()
        self.projectile = BRAWLER_PROJECTILES.get(
            self.brawler_name, BRAWLER_PROJECTILES["default"]
        )
        # Converter velocidade do projétil de tiles/s -> pixels/s
        self._proj_speed_px_s = self.projectile["speed"] * tile_size_px
        self._aim_error = HumanAimError(self.projectile["aim_precision"])

        self._enemy_history: Dict[int, deque] = {}
        self._enemy_vel: Dict[int, Tuple[float, float]] = {}
        self._ema_alpha = 0.3   # fator EMA (mais alto = mais responsivo)
        self._vel_threshold = 15.0  # px/s — abaixo disso considera "parado"

    def update_history(self, enemy_id: int, bbox):
        if enemy_id not in self._enemy_history:
            self._enemy_history[enemy_id] = deque(maxlen=8)
        cx, cy = _center(bbox)
        self._enemy_history[enemy_id].append((cx, cy, time.time()))

    def predict(self, enemy_bbox, player_bbox, enemy_id: int = 0) -> Tuple[int, int]:
        enemy_c = _center(enemy_bbox)
        player_c = _center(player_bbox)
        dist = _pixel_distance(player_c, enemy_c)

        # --- calcular velocidade do inimigo com EMA ---
        vel_x, vel_y = 0.0, 0.0
        history = self._enemy_history.get(enemy_id, deque())

        if len(history) >= 2:
            # Velocidade instantanea da ultima amostra
            x1, y1, t1 = history[-2]
            x2, y2, t2 = history[-1]
            dt = t2 - t1
            if dt > 0.01:
                inst_vx = (x2 - x1) / dt
                inst_vy = (y2 - y1) / dt
                # EMA: vel = alpha * inst + (1-alpha) * vel_anterior
                prev_vx, prev_vy = self._enemy_vel.get(enemy_id, (0.0, 0.0))
                vel_x = self._ema_alpha * inst_vx + (1 - self._ema_alpha) * prev_vx
                vel_y = self._ema_alpha * inst_vy + (1 - self._ema_alpha) * prev_vy
                # se velocidade muito baixa, considerar parado (evita drift)
                speed = math.sqrt(vel_x**2 + vel_y**2)
                if speed < self._vel_threshold:
                    vel_x, vel_y = 0.0, 0.0

        self._enemy_vel[enemy_id] = (vel_x, vel_y)

        # --- tempo de viagem do projétil ---
        # distancia / velocidade_do_projetil  (limitado para nao extrapolar)
        if self._proj_speed_px_s > 0:
            travel_time = dist / self._proj_speed_px_s
        else:
            travel_time = dist * 0.002
        travel_time = min(travel_time, 0.6)

        # --- posicao predita ---
        pred_x = enemy_c[0] + vel_x * travel_time
        pred_y = enemy_c[1] + vel_y * travel_time

        # --- aim-error humano ---
        aim_x, aim_y = self._aim_error.apply(
            pred_x, pred_y, vel_x, vel_y, dist
        )

        logger.debug(
            f"[LEADING] {self.brawler_name}: dist={dist:.0f}px "
            f"travel={travel_time:.3f}s vel=({vel_x:.1f},{vel_y:.1f}) "
            f"pred=({pred_x:.0f},{pred_y:.0f}) aim=({aim_x:.0f},{aim_y:.0f})"
        )

        return (int(_clamp(aim_x, 0, 9999)), int(_clamp(aim_y, 0, 9999)))


# ---------------------------------------------------------------------------
# 5.2  KITING  (v2 — vetor resultante + bushes + limites do mapa)
# ---------------------------------------------------------------------------

class KitingEngine:
    """Kite inteligente considerando multiplos inimigos e bushes."""

    def __init__(self, ideal_range: float = 400.0, retreat_dist: float = 350.0,
                 screen_w: int = 1920, screen_h: int = 1080):
        self.ideal_range = ideal_range
        self.retreat_dist = retreat_dist
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._last_kite_time = 0.0
        self._kite_cooldown = 0.8
        self._consecutive_kites = 0   # penalizar kites repetidos

    def should_kite(self, player, enemies, player_hp_estimate: float = 1.0,
                    brawler_role: str = "general") -> bool:
        if not enemies:
            return False
        if time.time() - self._last_kite_time < self._kite_cooldown:
            return False

        closest_dist = min(_pixel_distance(player, e) for e in enemies)
        num_enemies = len(enemies)

        # Assassinos (edgar, mortis, darryl) sao agressivos, kita menos
        aggressive_roles = {"assassin", "tank", "fighter"}
        hp_threshold = 0.30 if brawler_role in aggressive_roles else 0.40

        if player_hp_estimate < hp_threshold:
            self._consecutive_kites = 0
            return True
        if num_enemies >= 2 and closest_dist < 300:
            return True
        if closest_dist < 140:
            return True
        # Nao kitar se ja kita muito (evitar ficar so a fugir)
        if self._consecutive_kites >= 3:
            return False
        return False

    def get_kite_target(self, player, enemies, bushes: Optional[List] = None) -> Tuple[int, int]:
        """
        Calcula posicao de recuo usando vetor resultante de TODOS os inimigos.
        Se houver bushes, prefere recuar para o bush mais proximo na direcao de fuga.
        """
        player_c = _center(player)

        # Vetor resultante: soma de vetores afastamento de cada inimigo
        # Inimigos mais proximos tem peso maior (1/distancia^2)
        sum_dx, sum_dy, total_weight = 0.0, 0.0, 0.0
        for enemy in enemies:
            ex, ey = _center(enemy)
            dx = player_c[0] - ex
            dy = player_c[1] - ey
            dist = math.sqrt(dx**2 + dy**2) or 1.0
            weight = 1.0 / (dist ** 1.5)   # inimigos proximos dominam
            sum_dx += (dx / dist) * weight
            sum_dy += (dy / dist) * weight
            total_weight += weight

        if total_weight > 0:
            flee_dx = sum_dx / total_weight
            flee_dy = sum_dy / total_weight
        else:
            flee_dx, flee_dy = 0, -1  # recuar para cima como fallback

        # Normalizar
        flee_len = math.sqrt(flee_dx**2 + flee_dy**2) or 1.0
        flee_dx /= flee_len
        flee_dy /= flee_len

        # Base target
        target_x = player_c[0] + flee_dx * self.retreat_dist
        target_y = player_c[1] + flee_dy * self.retreat_dist

        # --- preferir bush na direcao de fuga ---
        if bushes:
            best_bush = None
            best_score = -9999
            for bush in bushes:
                bush_c = _center(bush)
                # Score: quao alinhado com direcao de fuga + quao perto do target ideal
                bush_dx = bush_c[0] - player_c[0]
                bush_dy = bush_c[1] - player_c[1]
                bush_dist = math.sqrt(bush_dx**2 + bush_dy**2) or 1.0
                # Alinhamento com vetor de fuga (dot product)
                alignment = (bush_dx * flee_dx + bush_dy * flee_dy) / bush_dist
                # Queremos bushes que estejam na direcao de fuga (alignment > 0.5)
                # e a uma distancia razoavel (nem muito perto nem muito longe)
                if alignment > 0.3 and 100 < bush_dist < 600:
                    score = alignment * 200 - abs(bush_dist - self.retreat_dist) * 0.3
                    if score > best_score:
                        best_score = score
                        best_bush = bush_c
            if best_bush:
                target_x, target_y = best_bush
                logger.info(f"[KITE] Recuando para bush: ({target_x:.0f},{target_y:.0f})")

        # --- clamp a tela (nao recuar para fora) ---
        margin = 80
        target_x = _clamp(target_x, margin, self.screen_w - margin)
        target_y = _clamp(target_y, margin, self.screen_h - margin)

        # Jitter para nao ser previsivel
        target_x += random.uniform(-25, 25)
        target_y += random.uniform(-25, 25)

        self._last_kite_time = time.time()
        self._consecutive_kites += 1
        return (int(target_x), int(target_y))


# ---------------------------------------------------------------------------
# 5.3  COVER  (v2 — escape routes, power cubes, threat scoring)
# ---------------------------------------------------------------------------

class CoverEngine:
    """Busca de cover com escape routes e threat analysis."""

    def __init__(self, cover_threshold: float = 150.0):
        self.cover_threshold = cover_threshold

    def should_take_cover(self, player, enemies, player_hp_estimate: float = 1.0,
                          brawler_role: str = "general") -> bool:
        if not enemies:
            return False
        # Tanks (El Primo, Rosa) nao buscam cover tanto
        tank_roles = {"tank"}
        hp_thresh = 0.25 if brawler_role in tank_roles else 0.30
        closest_dist = min(_pixel_distance(player, e) for e in enemies)
        if player_hp_estimate < hp_thresh and closest_dist < 450:
            return True
        if closest_dist < 180 and player_hp_estimate < 0.55:
            return True
        return False

    def find_best_cover(self, player, enemies, bushes: List,
                        power_cubes: Optional[List] = None) -> Optional[Tuple[int, int]]:
        if not bushes:
            return None

        player_c = _center(player)
        best_cover = None
        best_score = -float('inf')

        # Pre-computar ameacas
        enemy_centers = [_center(e) for e in enemies]
        num_enemies = len(enemies)

        for bush in bushes:
            bush_c = _center(bush)
            score = 0.0

            # --- distancia ao player (mais perto = melhor) ---
            dist_to_player = _pixel_distance(player_c, bush_c)
            if dist_to_player > 700:
                continue
            score += max(0, 400 - dist_to_player * 0.5)

            # --- distancia aos inimigos (longe = melhor) ---
            min_enemy_dist = min(_pixel_distance(bush_c, e) for e in enemies)
            if min_enemy_dist < 100:
                score -= 600   # Bush com inimigo dentro
            else:
                score += min(250, min_enemy_dist * 0.4)

            # --- line-of-sight break ---
            for ec in enemy_centers:
                if self._is_between(bush_c, player_c, ec, tolerance=70):
                    score += 300
                    break

            # --- escape route (outro bush proximo) ---
            escape_bonus = 0
            for other in bushes:
                if other is bush:
                    continue
                other_c = _center(other)
                d = _pixel_distance(bush_c, other_c)
                if 100 < d < 350:
                    escape_bonus = max(escape_bonus, 150 - d * 0.3)
            score += escape_bonus

            # --- power cube no bush ---
            if power_cubes:
                for cube in power_cubes:
                    if _pixel_distance(bush_c, cube) < 120:
                        score += 200   # coletar enquanto se esconde
                        break

            # --- threat cone: inimigos que APONTAM para este bush? ---
            # Se bush esta no caminho do inimigo, ele pode entrar la
            threat = 0
            for ec in enemy_centers:
                if self._is_between(ec, bush_c, player_c, tolerance=80):
                    threat += 1
            if threat > 0:
                score -= threat * 100   # penalizar bushes no caminho dos inimigos

            if score > best_score:
                best_score = score
                best_cover = bush_c

        if best_cover:
            logger.info(f"[COVER] Best: ({best_cover[0]:.0f},{best_cover[1]:.0f}) score={best_score:.0f}")
        return best_cover

    @staticmethod
    def _is_between(point, a, b, tolerance: float = 50.0) -> bool:
        px, py = point
        ax, ay = a
        bx, by = b
        abx = bx - ax
        aby = by - ay
        ab_len_sq = abx**2 + aby**2
        if ab_len_sq < 1:
            return False
        t = _clamp(((px - ax) * abx + (py - ay) * aby) / ab_len_sq, 0, 1)
        closest_x = ax + t * abx
        closest_y = ay + t * aby
        dist = math.sqrt((px - closest_x)**2 + (py - closest_y)**2)
        return dist < tolerance


# ---------------------------------------------------------------------------
# 5.4  COMBOS  (v2 — condicionais, delays adaptativos, opportunity detection)
# ---------------------------------------------------------------------------

class ComboManager:
    """Combos condicionais com oportunidade de execucao."""

    # (acao, delay, condicao_opcional)
    # condicao: "any", "enemy_low_hp", "enemy_grouped", "player_full_hp"
    COMBOS = {
        "edgar":    [("super", 0.0, "any"),       ("attack", 0.12, "any"),       ("attack", 0.12, "any")],
        "mortis":   [("attack", 0.0, "any"),       ("attack", 0.20, "any"),       ("attack", 0.20, "any")],
        "darryl":   [("super", 0.0, "enemy_grouped"), ("attack", 0.25, "any")],
        "el primo": [("super", 0.0, "enemy_low_hp"),  ("attack", 0.20, "any")],
        "shelly":   [("super", 0.0, "enemy_grouped"), ("attack", 0.40, "any")],
        "colt":     [("super", 0.0, "enemy_grouped")],
        "brock":    [("super", 0.0, "enemy_grouped")],
        "dynamike": [("super", 0.0, "enemy_grouped")],
        "tick":     [("super", 0.0, "enemy_grouped")],
        "emz":      [("super", 0.0, "enemy_grouped"), ("attack", 0.20, "any")],
        "poco":     [("super", 0.0, "any"),        ("attack", 0.20, "any")],
        "rosa":     [("super", 0.0, "player_full_hp"), ("attack", 0.20, "any")],
        "be":       [("super", 0.0, "any"),        ("gadget", 0.15, "any")],
        "8-bit":    [("super", 0.0, "any")],
        "jessie":   [("super", 0.0, "enemy_grouped"), ("attack", 0.25, "any")],
        "nita":     [("super", 0.0, "any"),        ("attack", 0.25, "any")],
        "rico":     [("super", 0.0, "enemy_grouped"), ("attack", 0.20, "any")],
        "spike":    [("super", 0.0, "enemy_grouped")],
        "crow":     [("super", 0.0, "enemy_low_hp"),  ("gadget", 0.15, "any"),       ("attack", 0.15, "any")],
        "leon":     [("super", 0.0, "enemy_low_hp"),  ("attack", 0.20, "any")],
        "piper":    [("super", 0.0, "enemy_low_hp")],
    }

    def __init__(self, brawler_name: str = "default"):
        self.brawler_name = str(brawler_name).lower().strip()
        raw = self.COMBOS.get(self.brawler_name, [])
        # Filtrar combos que satisfazem condicoes (avaliadas dinamicamente)
        self._combo_sequence = list(raw)
        self._combo_index = 0
        self._combo_active = False
        self._last_combo_time = 0.0
        self._combo_cooldown = 4.0

    def _check_condition(self, condition: str, player, enemies) -> bool:
        if condition == "any" or not enemies:
            return True
        if condition == "enemy_low_hp":
            # Heuristica: inimigo proximo com bbox pequeno = low HP
            closest = min(enemies, key=lambda e: _pixel_distance(player, e))
            # bbox area pequena ou distancia muito proxima
            area = (closest[2]-closest[0]) * (closest[3]-closest[1]) if len(closest)>=4 else 9999
            return area < 3000 or _pixel_distance(player, closest) < 200
        if condition == "enemy_grouped":
            # 2+ inimigos a < 400px um do outro
            if len(enemies) < 2:
                return False
            for i, e1 in enumerate(enemies):
                for e2 in enemies[i+1:]:
                    if _pixel_distance(e1, e2) < 400:
                        return True
            return False
        if condition == "player_full_hp":
            return True  # fallback; sem dados reais de HP
        return True

    def can_combo(self, player, enemies) -> bool:
        if not self._combo_sequence:
            return False
        if time.time() - self._last_combo_time < self._combo_cooldown:
            return False
        if not enemies:
            return False
        closest_dist = min(_pixel_distance(player, e) for e in enemies)
        if closest_dist > 350:
            return False
        # Verificar se a 1a acao do combo e valida
        first_action, _, first_cond = self._combo_sequence[0]
        return self._check_condition(first_cond, player, enemies)

    def start_combo(self) -> bool:
        if not self._combo_sequence:
            return False
        self._combo_active = True
        self._combo_index = 0
        self._last_combo_time = time.time()
        logger.info(f"[COMBO] Iniciando {self.brawler_name}: {self._combo_sequence}")
        return True

    def next_action(self) -> Optional[Tuple[str, float]]:
        if not self._combo_active or self._combo_index >= len(self._combo_sequence):
            self._combo_active = False
            return None
        action, delay, _ = self._combo_sequence[self._combo_index]
        self._combo_index += 1
        return action, delay

    def is_active(self) -> bool:
        return self._combo_active

    def reset(self):
        self._combo_active = False
        self._combo_index = 0


# ---------------------------------------------------------------------------
# 5.5  TARGET SELECTION  (novo)
# ---------------------------------------------------------------------------

class TargetSelector:
    """Escolhe qual inimigo atacar primeiro."""

    @staticmethod
    def select_target(player, enemies, brawler_role: str = "general") -> Tuple[int, any]:
        """
        Retorna (index, enemy_bbox) do melhor alvo.
        Prioridade:
        1. Inimigo com menos HP (bbox pequena heuristica)
        2. Suportes / ranged (mais perigosos a distancia)
        3. Mais proximo
        """
        if not enemies:
            return -1, None

        player_c = _center(player)
        scores = []

        for i, enemy in enumerate(enemies):
            ec = _center(enemy)
            dist = _pixel_distance(player_c, ec)
            area = (enemy[2]-enemy[0]) * (enemy[3]-enemy[1]) if len(enemy)>=4 else 9999

            score = 0.0
            # Quanto mais perto, maior score (base)
            score += max(0, 1000 - dist)

            # Inimigo com bbox pequena = provavelmente low HP
            if area < 2500:
                score += 500
            elif area < 4000:
                score += 200

            # Assassinos e brawlers ranged priorizam inimigos proximos
            if brawler_role in {"assassin", "fighter"}:
                score += max(0, 600 - dist) * 0.5

            # Snipers priorizam inimigos parados ou longe
            if brawler_role in {"sniper"}:
                score += dist * 0.2   # preferir alvos mais longe (mais faceis de acertar)

            scores.append((score, i, enemy))

        scores.sort(key=lambda x: x[0], reverse=True)
        best = scores[0]
        logger.debug(f"[TARGET] Alvo selecionado: score={best[0]:.0f}, dist={_pixel_distance(player, best[2]):.0f}px")
        return best[1], best[2]


# ---------------------------------------------------------------------------
# 5.6  COMBAT STATE MACHINE  (orquestrador v2)
# ---------------------------------------------------------------------------

class AdvancedCombatStrategy:
    """
    Orquestra combate com state machine:
    NEUTRAL -> AGGRESSIVE (combo / full HP)
            -> DEFENSIVE  (kite / cover / low HP)
    """

    BRAWLER_ROLES = {
        "shelly": "fighter", "colt": "sharpshooter", "piper": "sniper",
        "brock": "sniper", "nita": "fighter", "el primo": "tank",
        "dynamike": "thrower", "jessie": "sharpshooter", "rico": "sharpshooter",
        "edgar": "assassin", "mortis": "assassin", "tick": "thrower",
        "8-bit": "sharpshooter", "emz": "controller", "be": "sharpshooter",
        "spike": "sharpshooter", "crow": "assassin", "leon": "assassin",
        "darryl": "tank", "rosa": "tank", "poco": "support",
        "bo": "sharpshooter", "barley": "thrower", "pamu": "fighter",
    }

    def __init__(self, brawler_name: str = "default", screen_w: int = 1920, screen_h: int = 1080):
        self.brawler_name = str(brawler_name).lower().strip()
        self.role = self.BRAWLER_ROLES.get(self.brawler_name, "general")
        self.leading = LeadingShotEngine(brawler_name)
        self.kiting = KitingEngine(screen_w=screen_w, screen_h=screen_h)
        self.cover = CoverEngine()
        self.combo = ComboManager(brawler_name)
        self.target_selector = TargetSelector()

        self._state = "neutral"
        self._state_since = time.time()
        self._hp_estimate = 1.0
        self._screen_w = screen_w
        self._screen_h = screen_h

    def update_hp_estimate(self, hp_ratio: float):
        self._hp_estimate = _clamp(hp_ratio, 0.0, 1.0)

    def _transition(self, new_state: str):
        if self._state != new_state:
            logger.info(f"[COMBAT_STATE] {self._state} -> {new_state}")
            self._state = new_state
            self._state_since = time.time()

    def decide_combat_action(self, player, enemies, bushes, power_cubes) -> Dict:
        if not enemies:
            if power_cubes:
                return {
                    "action": "move",
                    "target": _center(min(power_cubes, key=lambda c: _pixel_distance(player, c))),
                    "reason": "collect_cube"
                }
            return {"action": "idle", "reason": "no_enemies"}

        # --- selecionar alvo ---
        target_idx, target_bbox = self.target_selector.select_target(
            player, enemies, self.role
        )
        if target_bbox is None:
            target_bbox = min(enemies, key=lambda e: _pixel_distance(player, e))

        # --- atualizar state machine ---
        if self._hp_estimate < 0.30:
            self._transition("retreating")
        elif self.combo.can_combo(player, enemies) and self._hp_estimate > 0.4:
            self._transition("aggressive")
        elif self._hp_estimate < 0.55:
            self._transition("defensive")
        else:
            self._transition("neutral")

        # --- decidir acao baseada no estado ---
        if self._state == "aggressive":
            if self.combo.can_combo(player, enemies):
                self.combo.start_combo()
                return {"action": "combo", "reason": "combo_opportunity"}

        if self._state in ("defensive", "retreating"):
            # Tentar cover primeiro
            if self.cover.should_take_cover(player, enemies, self._hp_estimate, self.role):
                cover_pos = self.cover.find_best_cover(player, enemies, bushes, power_cubes)
                if cover_pos:
                    return {"action": "cover", "target": cover_pos, "reason": "seeking_cover"}
            # Senao, kite
            if self.kiting.should_kite(player, enemies, self._hp_estimate, self.role):
                kite_pos = self.kiting.get_kite_target(player, enemies, bushes)
                return {"action": "kite", "target": kite_pos, "reason": "kiting_defensive"}

        # --- NEUTRAL / AGGRESSIVE sem combo: atacar com leading shot ---
        enemy_id = hash((int(_center(target_bbox)[0]) // 50,
                        int(_center(target_bbox)[1]) // 50))
        self.leading.update_history(enemy_id, target_bbox)
        predicted = self.leading.predict(target_bbox, player, enemy_id)

        return {
            "action": "attack",
            "predicted_pos": predicted,
            "enemy": target_bbox,
            "reason": f"leading_shot_{self._state}",
        }
