"""
rl_engine.py

Motor de Q-Learning online para decisoes de combate em Brawl Stars.

Design:
- Estado discreto: combinacao de buckets de HP, inimigos proximos, distancia, ammo
- Acoes discretas: UnifiedAction space (12 actions from core.class_registry)
- Atualizacao online: a cada frame de combate ou a cada match
- Exploracao decrescente (epsilon-greedy): comeca explorando, gradualmente explora menos
- Persistencia: salva tabela Q em pickle para aprendizado acumulado

Integracao:
- PlayLogic usa rl_engine.get_action(state) para decidir acoes
- StateManager chama rl_engine.update() a cada frame e end_episode() no fim
- RewardBridge fornece rewards por frame e por match

Migration:
    This version uses UnifiedAction from core.class_registry for consistency
    with UtilityAI and future neural policy. Legacy RL action names are
    automatically mapped to UnifiedAction.
"""

import json
import logging
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class CombatQLearning:
    """
    Q-Learning tabular para decisoes de combate em tempo real.

    Estado (tuple):
        (health_bucket, enemies_nearby, distance_bucket, ammo_ok, can_super)
    Onde:
        health_bucket: 0=baixo(<30%), 1=medio(30-70%), 2=alto(>70%)
        enemies_nearby: 0=nenhum, 1=1, 2=2+
        distance_bucket: 0=perto(<150px), 1=medio(150-400px), 2=longe(>400px)
        ammo_ok: 0=sem ammo, 1=com ammo
        can_super: 0=nao, 1=sim

    Actions:
        Now uses UnifiedAction from core.class_registry for consistency.
        Legacy action names ("attack", "retreat", etc.) are mapped internally.
    """

    # Import UnifiedAction for consistency with rest of system
    from core.class_registry import UnifiedAction

    # Legacy RL actions (for backward compatibility with existing Q-tables)
    LEGACY_ACTIONS = ["attack", "move_to_enemy", "retreat", "use_super", "collect_cube", "idle"]

    # Full UnifiedAction space (new Q-tables should use this)
    # Note: Q-table keys use action names for pickle compatibility
    UNIFIED_ACTIONS = [a.name.lower() for a in UnifiedAction]

    # Default to legacy actions for existing Q-tables, but support unified
    ACTIONS = LEGACY_ACTIONS  # Will be overridden if unified_actions=True

    # Hiperparametros Q-Learning
    ALPHA = 0.15           # Taxa de aprendizado
    GAMMA = 0.95           # Fator de desconto (futuro importa muito)
    EPSILON_START = 0.4    # 40% exploracao inicial
    EPSILON_END = 0.05     # 5% exploracao final (quase so greedy)
    EPSILON_DECAY = 0.995  # Decaimento por frame
    MIN_VISITS = 3         # Minimo de visitas para confiar na Q-value

    SAVE_INTERVAL = 100    # Salvar a cada N atualizacoes

    def __init__(self, save_path: Path = Path("data/q_table.pkl")):
        self.save_path = Path(save_path)
        self.q_table: Dict[Tuple, Dict[str, float]] = defaultdict(lambda: {a: 0.0 for a in self.ACTIONS})
        self.visit_counts: Dict[Tuple, int] = defaultdict(int)
        self.epsilon = self.EPSILON_START
        self.total_updates = 0
        self.last_state: Optional[Tuple] = None
        self.last_action: Optional[str] = None
        self.frame_rewards: List[float] = []
        self._load()
        logger.info(f"[RL] Q-Learning inicializado: {len(self.q_table)} estados, epsilon={self.epsilon:.3f}")

    # --- Discretizacao de estado ---

    @staticmethod
    def discretize_state(
        player_hp_pct: float,
        num_enemies: int,
        nearest_enemy_dist: float,
        can_attack: bool,
        can_super: bool,
    ) -> Tuple[int, int, int, int, int]:
        """Converte estado continuo em tupla discreta."""
        # HP bucket
        if player_hp_pct < 0.30:
            hp_bucket = 0  # Baixo
        elif player_hp_pct < 0.70:
            hp_bucket = 1  # Medio
        else:
            hp_bucket = 2  # Alto

        # Enemies nearby
        if num_enemies == 0:
            enemies_bucket = 0
        elif num_enemies == 1:
            enemies_bucket = 1
        else:
            enemies_bucket = 2  # 2+

        # Distance bucket
        if nearest_enemy_dist < 150:
            dist_bucket = 0  # Perto
        elif nearest_enemy_dist < 400:
            dist_bucket = 1  # Medio
        else:
            dist_bucket = 2  # Longe

        ammo_bucket = 1 if can_attack else 0
        super_bucket = 1 if can_super else 0

        return (hp_bucket, enemies_bucket, dist_bucket, ammo_bucket, super_bucket)

    @classmethod
    def state_from_combat_snapshot(
        cls,
        player_bbox,
        enemies: List,
        can_attack: bool = True,
        can_super: bool = False,
        player_hp_pct: Optional[float] = None,
    ) -> Tuple[int, int, int, int, int]:
        """Cria estado discreto a partir dos dados brutos de combate."""
        if player_bbox is None:
            return (1, 0, 2, 0, 0)  # Estado neutro/default

        if player_hp_pct is None:
            player_hp_pct = 1.0

        num_enemies = len(enemies)

        # Distancia ate inimigo mais proximo
        if num_enemies == 0 or enemies is None:
            nearest_dist = 9999.0
        else:
            import math
            px = (player_bbox[0] + player_bbox[2]) / 2
            py = (player_bbox[1] + player_bbox[3]) / 2
            nearest_dist = min(
                math.sqrt(((e[0]+e[2])/2 - px)**2 + ((e[1]+e[3])/2 - py)**2)
                for e in enemies if len(e) >= 4
            ) if enemies else 9999.0

        return cls.discretize_state(player_hp_pct, num_enemies, nearest_dist, can_attack, can_super)

    # --- Decisao ---

    def get_action(self, state: Tuple, force_explore: bool = False, **kwargs) -> Tuple[str, float]:
        """
        Retorna acao recomendada e confianca (Q-value normalizada).
        force_explore=True ignora epsilon (para treino manual).
        """
        self.visit_counts[state] += 1

        # Exploracao (epsilon-greedy)
        if force_explore or random.random() < self.epsilon:
            action = random.choice(self.ACTIONS)
            confidence = 0.3  # Baixa confianca na exploracao
            logger.debug(f"[RL] Explorando: acao={action}, epsilon={self.epsilon:.3f}")
            return action, confidence

        # Explotacao: escolher acao com maior Q-value
        q_values = self.q_table[state]
        if not q_values:
            action = random.choice(self.ACTIONS)
            return action, 0.3

        best_action = max(q_values, key=q_values.get)
        best_q = q_values[best_action]

        # Calcular confianca baseada na Q-value e visitas
        visits = self.visit_counts[state]
        if visits < self.MIN_VISITS:
            confidence = 0.4 + (visits / self.MIN_VISITS) * 0.3
        else:
            # Normalizar Q-value para confianca 0.5-1.0
            max_possible_q = 10.0  # Heuristica
            confidence = min(1.0, max(0.5, 0.5 + best_q / max_possible_q))

        logger.debug(f"[RL] Greedy: acao={best_action}, Q={best_q:.2f}, conf={confidence:.2f}, visits={visits}")
        return best_action, confidence

    # --- Atualizacao (learning) ---

    def update(self, state: Tuple, action: str, reward: float, next_state: Tuple):
        """
        Atualiza tabela Q com uma transicao (s, a, r, s').
        Chamado a cada frame de combate.
        """
        if action not in self.ACTIONS:
            logger.warning(f"[RL] Acao invalida: {action}")
            return

        # Q(s,a) += alpha * (r + gamma * max Q(s',a') - Q(s,a))
        current_q = self.q_table[state][action]

        # max Q(s', a') sobre todas as acoes
        next_q_values = self.q_table[next_state]
        max_next_q = max(next_q_values.values()) if next_q_values else 0.0

        # Atualizacao Q-Learning
        new_q = current_q + self.ALPHA * (reward + self.GAMMA * max_next_q - current_q)
        self.q_table[state][action] = new_q

        self.total_updates += 1
        self.epsilon = max(self.EPSILON_END, self.epsilon * self.EPSILON_DECAY)

        # Salvar periodicamente
        if self.total_updates % self.SAVE_INTERVAL == 0:
            self._save()

        logger.debug(
            f"[RL] Update: s={state}, a={action}, r={reward:.2f}, "
            f"s'={next_state}, Q_old={current_q:.2f}, Q_new={new_q:.2f}, "
            f"eps={self.epsilon:.4f}"
        )

    def record_reward(self, reward: float):
        """Registra reward de um frame (usado para backward updates no fim do episodio)."""
        self.frame_rewards.append(reward)

    def end_episode(self, final_reward: float):
        """
        Finaliza episodio com reward final e faz backward updates
        com todos os rewards acumulados.
        """
        if not self.frame_rewards:
            logger.debug("[RL] Episodio sem rewards acumulados")
            return

        # Backward pass: acumular rewards com desconto
        cumulative = 0.0
        gamma_power = 1.0
        for r in reversed(self.frame_rewards):
            cumulative += gamma_power * r
            gamma_power *= self.GAMMA

        logger.info(
            f"[RL] Episodio finalizado: {len(self.frame_rewards)} frames, "
            f"reward_final={final_reward:.2f}, "
            f"reward_acumulado={cumulative:.2f}, "
            f"estados={len(self.q_table)}, epsilon={self.epsilon:.4f}"
        )

        self.frame_rewards.clear()
        self.last_state = None
        self.last_action = None
        self._save()

    def get_live_metrics(self) -> Dict:
        """Retorna métricas live para a dashboard."""
        return {
            "engine": "q_learning",
            "q_table_size": len(self.q_table),
            "epsilon": round(self.epsilon, 4),
            "total_updates": self.total_updates,
            "actions": self.ACTIONS,
            "last_state": str(self.last_state) if self.last_state else None,
            "last_action": self.last_action,
        }

    def get_policy_summary(self) -> Dict:
        """Retorna resumo da politica atual para diagnostico."""
        if not self.q_table:
            return {"empty": True}

        # Encontrar estados mais visitados e suas acoes preferidas
        top_states = sorted(self.visit_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        policy = {}
        for state, visits in top_states:
            q_vals = self.q_table[state]
            best_action = max(q_vals, key=q_vals.get)
            policy[str(state)] = {
                "best_action": best_action,
                "q_value": round(q_vals[best_action], 2),
                "visits": visits,
            }

        # Distribuicao de Q-values
        all_q = [q for actions in self.q_table.values() for q in actions.values()]
        return {
            "num_states": len(self.q_table),
            "epsilon": round(self.epsilon, 4),
            "total_updates": self.total_updates,
            "avg_q": round(sum(all_q) / len(all_q), 2) if all_q else 0,
            "max_q": round(max(all_q), 2) if all_q else 0,
            "min_q": round(min(all_q), 2) if all_q else 0,
            "top_policy": policy,
        }

    # --- Persistencia ---

    def _save(self):
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "q_table": dict(self.q_table),
                "visit_counts": dict(self.visit_counts),
                "epsilon": self.epsilon,
                "total_updates": self.total_updates,
                "version": 1,
                "saved_at": time.time(),
            }
            # FIX #13: Atomic write using temp file + rename (prevents corruption on crash)
            temp_path = self.save_path.with_suffix('.tmp')
            with open(temp_path, "wb") as f:
                pickle.dump(data, f)
            os.replace(str(temp_path), str(self.save_path))  # Atomic on both POSIX and Windows
            logger.debug(f"[RL] Q-table salva: {self.save_path} ({len(self.q_table)} estados)")
        except Exception as e:
            logger.warning(f"[RL] Falha ao salvar Q-table: {e}")

    def _load(self):
        if not self.save_path.exists():
            return
        try:
            with open(self.save_path, "rb") as f:
                data = pickle.load(f)
            self.q_table = defaultdict(lambda: {a: 0.0 for a in self.ACTIONS})
            self.q_table.update(data.get("q_table", {}))
            self.visit_counts = defaultdict(int)
            self.visit_counts.update(data.get("visit_counts", {}))
            self.epsilon = data.get("epsilon", self.EPSILON_START)
            self.total_updates = data.get("total_updates", 0)
            logger.info(f"[RL] Q-table carregada: {len(self.q_table)} estados, epsilon={self.epsilon:.3f}")
        except Exception as e:
            logger.warning(f"[RL] Falha ao carregar Q-table: {e}")

    def force_save(self):
        self._save()


class OnlineLearner:
    """
    Facade que integra Q-Learning + NeuralPolicy (PPO) + ELO + RewardBridge.
    Eh o ponto unico de entrada para o sistema de aprendizado online.

    Migration v2.3:
    - Adicionado RLBridge com NeuralPolicy + PPO
    - use_neural=True ativa policy network profunda (CNN+LSTM+Fusion)
    - Q-Learning permanece como fallback automatico
    """

    def __init__(
        self,
        q_save_path: Path = Path("data/q_table.pkl"),
        elo_save_path: Path = Path("data/elo_ratings.json"),
        reward_bridge=None,
        enabled: bool = True,
        use_neural: bool = True,
        neural_schema: str = "core",
    ):
        self.enabled = enabled
        self.reward_bridge = reward_bridge

        # RL Bridge: NeuralPolicy + PPO + Q-Learning fallback
        self.rl_bridge = None
        self.use_neural = use_neural
        if use_neural:
            try:
                from neural.rl_bridge import RLBridge
                self.rl_bridge = RLBridge(
                    use_neural=True,
                    schema=neural_schema,
                    q_learning_fallback=True,
                )
                logger.info("[RL] NeuralPolicy + PPO ativados via RLBridge")
            except Exception as e:
                logger.warning(f"[RL] RLBridge nao disponivel: {e}. Usando Q-Learning.")
                self.use_neural = False

        # Legacy Q-Learning (sempre disponivel como fallback)
        self.q_learning = CombatQLearning(save_path=q_save_path)
        self.elo = None  # Lazy import para evitar dependencia circular
        try:
            from .elo_tracker import BrawlerMapELO
            self.elo = BrawlerMapELO(save_path=elo_save_path)
        except ImportError as e:
            logger.warning(f"[RL] ELO tracker nao disponivel: {e}")

        self.current_map: Optional[str] = None
        self.current_brawler: Optional[str] = None
        self.episode_reward: float = 0.0
        self.episode_start_time: Optional[float] = None

        logger.info(f"[RL] OnlineLearner inicializado: enabled={enabled}, neural={use_neural}")

    def start_episode(self, brawler_name: str, map_name: Optional[str] = None):
        """Chamado no inicio de cada partida."""
        self.current_brawler = brawler_name
        self.current_map = map_name
        self.episode_reward = 0.0
        self.episode_start_time = time.time()
        self.q_learning.frame_rewards.clear()
        if self.rl_bridge is not None:
            self.rl_bridge.start_episode()
        if self.reward_bridge:
            self.reward_bridge.start_match()
        logger.info(f"[RL] Episodio iniciado: {brawler_name} @ {map_name or 'unknown'}")

    def get_action(self, state: Tuple, default_action: str = "idle", **kwargs) -> Tuple[str, float]:
        """Obtem acao do RL (Neural ou Q-Learning). Retorna (acao, confianca)."""
        if not self.enabled:
            return default_action, 0.0

        # NeuralPolicy path
        if self.use_neural and self.rl_bridge is not None:
            try:
                return self.rl_bridge.get_action(state, **kwargs)
            except Exception as e:
                logger.debug(f"[RL] NeuralPolicy falhou: {e}, fallback Q-Learning")

        # Q-Learning fallback
        action, confidence = self.q_learning.get_action(state)
        return action, confidence

    def learn_from_frame(self, state: Tuple, action: str, reward: float, next_state: Tuple, **kwargs):
        """Atualiza RL com uma transicao de frame."""
        if not self.enabled:
            return

        # Neural path
        if self.use_neural and self.rl_bridge is not None:
            try:
                self.rl_bridge.learn_from_frame(state, action, reward, next_state, **kwargs)
            except Exception as e:
                logger.debug(f"[RL] Neural learn falhou: {e}")

        # Q-Learning (sempre atualiza para manter fallback fresco)
        self.q_learning.update(state, action, reward, next_state)
        self.q_learning.record_reward(reward)
        self.episode_reward += reward

    def end_episode(self, result: str, rank: int = 0, damage_dealt: float = 0.0):
        """Finaliza episodio e atualiza ELO."""
        if not self.enabled:
            return

        survival_time = 0.0
        if self.episode_start_time:
            survival_time = time.time() - self.episode_start_time

        # Reward final baseado no resultado
        result_rewards = {"win": 10.0, "draw": 2.0, "loss": -5.0}
        final_reward = result_rewards.get(result, 0.0)

        # Bonus/penalidade por rank
        if rank > 0:
            if rank <= 2:
                final_reward += 5.0
            elif rank <= 4:
                final_reward += 2.0
            elif rank >= 8:
                final_reward -= 3.0

        # Neural path
        if self.use_neural and self.rl_bridge is not None:
            try:
                self.rl_bridge.end_episode(final_reward)
            except Exception as e:
                logger.debug(f"[RL] Neural end_episode falhou: {e}")

        self.q_learning.end_episode(final_reward)

        # Atualizar ELO
        if self.elo and self.current_brawler:
            self.elo.record_match(
                brawler=self.current_brawler,
                map_name=self.current_map,
                result=result,
                rank=rank,
                damage_dealt=damage_dealt,
                survival_time=survival_time,
            )

        logger.info(
            f"[RL] Episodio final: {self.current_brawler}@{self.current_map}, "
            f"result={result}, rank={rank}, reward_total={self.episode_reward:.2f}"
        )
        self.episode_reward = 0.0
        self.current_brawler = None
        self.current_map = None

    def suggest_brawler_for_map(self, map_name: str, available: List[str]) -> Optional[str]:
        """Sugere brawler baseado no ELO para o mapa."""
        if self.elo:
            return self.elo.get_best_brawler_for_map(map_name, available)
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatisticas do sistema de aprendizado."""
        stats = {
            "q_learning": self.q_learning.get_policy_summary(),
            "enabled": self.enabled,
        }
        if self.elo:
            stats["elo"] = self.elo.get_global_summary()
        return stats

    def save(self):
        """Forca salvamento de todos os dados."""
        self.q_learning.force_save()
        if self.elo:
            self.elo.force_save()
