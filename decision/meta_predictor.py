"""
decision/meta_predictor.py

Meta Prediction — Predição de brawlers e composições do meta atual.

Analisa histórico de partidas para predizer:
- Brawlers mais prováveis de aparecer (por mapa/modo)
- Composições de time comuns
- Counter-picks ótimos
- Tendências do meta ao longo do tempo

Uso:
    predictor = MetaPredictor()
    predictor.record_match(map_name="Gem_Grab", enemies=["Shelly", "Colt", "Poco"])
    brawlers = predictor.predict_enemy_brawlers(map_name="Gem_Grab", top_n=3)
    counters = predictor.suggest_counters(enemies=["Shelly", "Colt"])
"""

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class MetaPredictor:
    """
    Preditor de meta baseado em frequência histórica.
    """

    # Brawlers conhecidos
    ALL_BRAWLERS = [
        "Shelly", "Nita", "Colt", "Bull", "Brock", "Dynamike", "Bo",
        "Tick", "8-Bit", "Emz", "El Primo", "Barley", "Poco", "Rosa",
        "Rico", "Darryl", "Penny", "Carl", "Jacky", "Piper", "Pam",
        "Frank", "Bibi", "Bea", "Nani", "Edgar", "Griff", "Grom",
        "Squeak", "Lou", "Ruffs", "Belle", "Buzz", "Ash", "Lola",
        "Fang", "Eve", "Janet", "Otis", "Sam", "Gus", "Buster",
        "Mandy", "R-T", "Maisie", "Gray", "Chester", "Cordelius",
        "Pearl", "Charlie", "Mico", "Kit", "Draco", "Angelo", "Melodie",
        "Lily", "Berry", "Clancy", "Moe", "Juju", "Shade",
    ]

    def __init__(self, data_dir: Path = Path("data/meta")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.data_dir / "match_history.jsonl"

        # Estatísticas
        self._brawler_frequency: Counter = Counter()
        self._map_brawlers: dict[str, Counter] = defaultdict(Counter)
        self._team_compositions: Counter = Counter()  # frozenset -> count
        self._win_rates: dict[str, dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0})

        self._load_history()

    def _load_history(self):
        """Carrega histórico de partidas."""
        if not self.history_path.exists():
            return
        try:
            with open(self.history_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        match = json.loads(line)
                        self._update_stats(match, save=False)
                    except json.JSONDecodeError:
                        pass
            logger.info("[META] %d partidas carregadas", sum(self._brawler_frequency.values()))
        except Exception as e:
            logger.warning("[META] Erro ao carregar histórico: %s", e)

    def record_match(
        self,
        map_name: str,
        game_mode: str | None = None,
        enemies: list[str] = None,
        allies: list[str] = None,
        result: str | None = None,
    ):
        """Registra uma partida no histórico."""
        match = {
            "timestamp": time.time(),
            "map": map_name,
            "mode": game_mode,
            "enemies": enemies or [],
            "allies": allies or [],
            "result": result,
        }

        # Salvar
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(match) + "\n")

        self._update_stats(match)

    def _update_stats(self, match: dict, save: bool = True):
        """Atualiza estatísticas internas."""
        map_name = match.get("map", "unknown")
        enemies = match.get("enemies", [])
        match.get("allies", [])
        result = match.get("result")

        for brawler in enemies:
            self._brawler_frequency[brawler] += 1
            self._map_brawlers[map_name][brawler] += 1

            if result:
                if result in ("win", "victory"):
                    self._win_rates[brawler]["wins"] += 1
                elif result in ("loss", "defeat"):
                    self._win_rates[brawler]["losses"] += 1

        # Composição de time
        if len(enemies) >= 2:
            comp = frozenset(enemies)
            self._team_compositions[comp] += 1

    # ------------------------------------------------------------------
    # Predições
    # ------------------------------------------------------------------

    def predict_enemy_brawlers(
        self,
        map_name: str | None = None,
        game_mode: str | None = None,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Prediz brawlers inimigos mais prováveis.
        Retorna lista de (brawler, probabilidade).
        """
        if map_name and map_name in self._map_brawlers:
            counter = self._map_brawlers[map_name]
        else:
            counter = self._brawler_frequency

        total = sum(counter.values())
        if total == 0:
            # Retornar brawlers populares como fallback
            return [(b, 1.0 / len(self.ALL_BRAWLERS)) for b in self.ALL_BRAWLERS[:top_n]]

        most_common = counter.most_common(top_n)
        return [(b, count / total) for b, count in most_common]

    def predict_team_composition(
        self,
        map_name: str | None = None,
        top_n: int = 3,
    ) -> list[tuple[frozenset, float]]:
        """Prediz composições de time mais comuns."""
        # Filtrar por mapa se possível (simplificado)
        total = sum(self._team_compositions.values())
        if total == 0:
            return []

        most_common = self._team_compositions.most_common(top_n)
        return [(comp, count / total) for comp, count in most_common]

    def suggest_counters(
        self,
        enemies: list[str],
        owned_brawlers: list[str] | None = None,
        top_n: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Sugere counter-picks para brawlers inimigos.

        Heurística simplificada (pode ser expandida com dados reais):
        - Long-range counters short-range
        - Burst counters squishy
        - Heal counters poke
        """
        counters = Counter()

        # Tabela de counters simplificada
        COUNTER_TABLE = {  # noqa: N806
            "Shelly": ["Brock", "Piper", "Bea"],
            "Bull": ["Shelly", "Brock", "Piper"],
            "El Primo": ["Shelly", "Colt", "Brock"],
            "Colt": ["Brock", "Piper", "Mortis"],
            "Brock": ["Mortis", "Edgar", "Leon"],
            "Piper": ["Mortis", "Edgar", "Leon"],
            "Mortis": ["Shelly", "Bull", "Rosa"],
            "Dynamike": ["Mortis", "Edgar", "Leon"],
            "Tick": ["Mortis", "Edgar", "Leon"],
            "Poco": ["Brock", "Piper", "Colt"],
            "Rosa": ["Brock", "Piper", "Colt"],
        }

        for enemy in enemies:
            for counter in COUNTER_TABLE.get(enemy, []):
                if owned_brawlers is None or counter in owned_brawlers:
                    counters[counter] += 1

        if not counters:
            return []

        most_common = counters.most_common(top_n)
        max_score = most_common[0][1] if most_common else 1
        return [(b, score / max_score) for b, score in most_common]

    def get_brawler_win_rate(self, brawler: str) -> float | None:
        """Retorna win rate de um brawler."""
        stats = self._win_rates.get(brawler)
        if not stats:
            return None
        total = stats["wins"] + stats["losses"]
        return stats["wins"] / total if total > 0 else 0.0

    def get_meta_trend(self, days: int = 7) -> dict[str, list[tuple[str, float]]]:
        """
        Retorna tendência do meta nos últimos N dias.
        """
        # Simplificado: retorna frequências atuais
        cutoff = time.time() - days * 86400
        recent_counter = Counter()

        if self.history_path.exists():
            with open(self.history_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        match = json.loads(line)
                        if match.get("timestamp", 0) > cutoff:
                            for b in match.get("enemies", []):
                                recent_counter[b] += 1
                    except Exception:
                        pass

        total = sum(recent_counter.values())
        if total == 0:
            return {"top_brawlers": [], "total_matches": 0}

        top = recent_counter.most_common(10)
        return {
            "top_brawlers": [(b, count / total) for b, count in top],
            "total_matches": total,
            "period_days": days,
        }

    def get_status(self) -> dict:
        return {
            "total_matches": sum(self._brawler_frequency.values()),
            "unique_brawlers_seen": len(self._brawler_frequency),
            "unique_maps": len(self._map_brawlers),
            "unique_compositions": len(self._team_compositions),
        }
