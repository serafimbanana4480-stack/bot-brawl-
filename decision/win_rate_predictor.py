"""
decision/win_rate_predictor.py

Win Rate Prediction — Predição de probabilidade de vitória em tempo real.

Usa um modelo simples (logistic regression ou heurísticas) para estimar
a probabilidade de vitória baseada em:
- Composição de time (brawlers)
- Mapa e modo de jogo
- ELO relativo
- Power level (cubes/gems)
- Tempo restante

Uso:
    predictor = WinRatePredictor()
    win_prob = predictor.predict(
        allies=["Shelly", "Colt", "Poco"],
        enemies=["Bull", "Brock", "Rosa"],
        map_name="Gem_Grab",
        own_cubes=5,
        enemy_cubes=3,
        time_remaining=30,
    )
    # win_prob = 0.72 (72% chance de vitória)
"""

import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class WinRatePredictor:
    """
    Preditor de win rate baseado em heurísticas e dados históricos.
    """

    def __init__(self, data_dir: Path = Path("data/winrate")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Base de dados: mapa + modo -> composição -> win/loss
        self._map_stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "matches": 0})
        self._brawler_synergy: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0})
        self._counter_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_match(
        self,
        map_name: str,
        game_mode: str,
        allies: list[str],
        enemies: list[str],
        result: str,  # "win" ou "loss"
        own_score: int = 0,
        enemy_score: int = 0,
    ):
        """Registra resultado para aprendizado."""
        key = f"{map_name}:{game_mode}"
        self._map_stats[key]["matches"] += 1
        if result in ("win", "victory"):
            self._map_stats[key]["wins"] += 1
        else:
            self._map_stats[key]["losses"] += 1

        # Synergy
        for a1 in allies:
            for a2 in allies:
                if a1 != a2:
                    synergy_key = f"{a1}+{a2}"
                    if result in ("win", "victory"):
                        self._brawler_synergy[synergy_key]["wins"] += 1
                    else:
                        self._brawler_synergy[synergy_key]["losses"] += 1

        # Counter
        for ally in allies:
            for enemy in enemies:
                if result in ("win", "victory"):
                    self._counter_matrix[ally][enemy] += 1
                else:
                    self._counter_matrix[ally][enemy] -= 1

    def predict(
        self,
        allies: list[str],
        enemies: list[str],
        map_name: str,
        game_mode: str = "3v3",
        own_cubes: int = 0,
        enemy_cubes: int = 0,
        time_remaining: int | None = None,
        own_hp_avg: float = 1.0,
        enemy_hp_avg: float = 1.0,
    ) -> float:
        """
        Prediz probabilidade de vitória (0.0 - 1.0).
        """
        score = 0.5  # baseline

        # 1. Synergy bonus
        synergy_bonus = self._compute_synergy_bonus(allies)
        score += synergy_bonus * 0.1

        # 2. Counter bonus
        counter_bonus = self._compute_counter_bonus(allies, enemies)
        score += counter_bonus * 0.15

        # 3. Map familiarity
        map_bonus = self._compute_map_bonus(map_name, game_mode, allies)
        score += map_bonus * 0.1

        # 4. Power cubes / resources
        if own_cubes + enemy_cubes > 0:
            cube_ratio = own_cubes / (own_cubes + enemy_cubes)
            score += (cube_ratio - 0.5) * 0.2

        # 5. HP advantage
        hp_diff = own_hp_avg - enemy_hp_avg
        score += hp_diff * 0.15

        # 6. Time pressure (se estamos ganhando e tempo acabando)
        if time_remaining is not None and time_remaining < 30:
            if own_cubes > enemy_cubes:
                score += 0.1
            elif enemy_cubes > own_cubes:
                score -= 0.1

        # Clamp
        return max(0.05, min(0.95, score))

    def _compute_synergy_bonus(self, allies: list[str]) -> float:
        """Computa bônus de sinergia do time."""
        if len(allies) < 2:
            return 0.0

        total_synergy = 0.0
        pairs = 0
        for i, a1 in enumerate(allies):
            for a2 in allies[i+1:]:
                synergy_key = f"{a1}+{a2}"
                stats = self._brawler_synergy.get(synergy_key)
                if stats:
                    total = stats["wins"] + stats["losses"]
                    if total > 0:
                        total_synergy += (stats["wins"] / total - 0.5) * 2
                pairs += 1

        return total_synergy / pairs if pairs > 0 else 0.0

    def _compute_counter_bonus(self, allies: list[str], enemies: list[str]) -> float:
        """Computa bônus de counter."""
        if not allies or not enemies:
            return 0.0

        total_counter = 0.0
        pairs = 0
        for ally in allies:
            for enemy in enemies:
                score = self._counter_matrix[ally].get(enemy, 0)
                # Normalizar aproximadamente (-10 a +10)
                total_counter += max(-1, min(1, score / 5))
                pairs += 1

        return total_counter / pairs if pairs > 0 else 0.0

    def _compute_map_bonus(self, map_name: str, game_mode: str, allies: list[str]) -> float:
        """Computa bônus de familiaridade com mapa."""
        key = f"{map_name}:{game_mode}"
        stats = self._map_stats.get(key)
        if not stats or stats["matches"] < 5:
            return 0.0
        return (stats["wins"] / stats["matches"] - 0.5) * 2

    def get_matchup_analysis(
        self,
        allies: list[str],
        enemies: list[str],
    ) -> dict[str, any]:
        """Retorna análise detalhada do matchup."""
        analysis = {
            "synergy_score": round(self._compute_synergy_bonus(allies), 3),
            "counter_score": round(self._compute_counter_bonus(allies, enemies), 3),
            "individual_matchups": {},
        }

        for ally in allies:
            for enemy in enemies:
                score = self._counter_matrix[ally].get(enemy, 0)
                analysis["individual_matchups"][f"{ally}_vs_{enemy}"] = {
                    "score": score,
                    "advantage": "ally" if score > 0 else "enemy" if score < 0 else "even",
                }

        return analysis

    def get_status(self) -> dict:
        return {
            "maps_tracked": len(self._map_stats),
            "synergies_tracked": len(self._brawler_synergy),
            "counter_pairs": sum(len(v) for v in self._counter_matrix.values()),
        }
