"""
elo_tracker.py

Sistema de rating ELO por combinacao brawler+mapa.
Permite ao bot aprender quais brawlers performam melhor em quais mapas
(como o BrawlStats faz para jogadores humanos).

ELO inicial: 1000
K-factor: 32 (padrao), ajusta para 16 quando rating > 1400
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class BrawlerMapStats:
    """Estatisticas de performance para um par brawler+mapa."""
    brawler: str
    map_name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    score: float = 1000.0  # ELO score
    avg_rank: float = 0.0
    total_matches: int = 0
    last_played: float = 0.0
    best_streak: int = 0
    current_streak: int = 0
    damage_dealt_avg: float = 0.0
    survival_time_avg: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_matches == 0:
            return 0.0
        return self.wins / self.total_matches

    @property
    def k_factor(self) -> int:
        """K-factor dinamico: menor para ratings altos (mais estabilidade)."""
        if self.score > 1600:
            return 16
        elif self.score > 1400:
            return 20
        elif self.score < 800:
            return 40  # Acelerar recuperacao de ratings muito baixos
        return 32


class BrawlerMapELO:
    """
    Tracking de performance ELO por combinacao brawler+mapa.
    Permite:
    - Saber qual brawler escolher para cada mapa
    - Ajustar a dificuldade do bot baseado no ELO
    - Identificar combinacoes under/over-performing
    """

    DEFAULT_ELO = 1000.0
    SAVE_INTERVAL = 10  # Salvar a cada N partidas

    def __init__(self, save_path: Path = Path("data/elo_ratings.json")):
        self.save_path = Path(save_path)
        self.ratings: Dict[Tuple[str, str], BrawlerMapStats] = {}
        self._matches_since_save = 0
        self._load()
        logger.info(f"[ELO] Inicializado: {len(self.ratings)} combinacoes carregadas")

    def _key(self, brawler: str, map_name: str) -> Tuple[str, str]:
        """Normaliza chave para dict."""
        return (str(brawler).lower().strip(), str(map_name).lower().strip() if map_name else "unknown")

    def record_match(
        self,
        brawler: str,
        map_name: Optional[str],
        result: str,  # "win", "loss", "draw"
        rank: int = 0,  # 1-10 (quanto menor, melhor)
        damage_dealt: float = 0.0,
        survival_time: float = 0.0,
    ) -> Dict:
        """
        Registra resultado de uma partida e atualiza ELO.
        Retorna dict com score anterior, novo score, e delta.
        """
        key = self._key(brawler, map_name or "unknown")
        stats = self.ratings.get(key)
        if stats is None:
            stats = BrawlerMapStats(brawler=brawler, map_name=map_name or "unknown")
            self.ratings[key] = stats

        old_score = stats.score

        # Atualizar contadores
        stats.total_matches += 1
        stats.last_played = time.time()

        # Calcular expected score (baseado no rating atual vs rating padrao de 1000)
        expected = 1.0 / (1.0 + 10.0 ** ((self.DEFAULT_ELO - stats.score) / 400.0))

        if result == "win":
            actual = 1.0
            stats.wins += 1
            stats.current_streak = max(1, stats.current_streak + 1)
            stats.best_streak = max(stats.best_streak, stats.current_streak)
        elif result == "draw":
            actual = 0.5
            stats.draws += 1
            stats.current_streak = 0
        else:  # loss
            actual = 0.0
            stats.losses += 1
            stats.current_streak = min(-1, stats.current_streak - 1)
            stats.best_streak = max(stats.best_streak, abs(stats.current_streak))

        # Atualizar ELO
        k = stats.k_factor
        stats.score += k * (actual - expected)
        # Clamp score para evitar extremos absurdos
        stats.score = max(100, min(3000, stats.score))

        # Atualizar estatisticas auxiliares (media movel simples)
        n = stats.total_matches
        stats.avg_rank = ((n - 1) * stats.avg_rank + rank) / n if n > 0 else rank
        stats.damage_dealt_avg = ((n - 1) * stats.damage_dealt_avg + damage_dealt) / n if n > 0 else damage_dealt
        stats.survival_time_avg = ((n - 1) * stats.survival_time_avg + survival_time) / n if n > 0 else survival_time

        delta = stats.score - old_score

        logger.info(
            f"[ELO] {brawler}@{map_name or 'unknown'}: "
            f"result={result}, old={old_score:.0f}, new={stats.score:.0f}, "
            f"delta={delta:+.0f}, wr={stats.win_rate:.1%}, matches={stats.total_matches}"
        )

        self._matches_since_save += 1
        if self._matches_since_save >= self.SAVE_INTERVAL:
            self._save()
            self._matches_since_save = 0

        return {
            "old_score": old_score,
            "new_score": stats.score,
            "delta": delta,
            "win_rate": stats.win_rate,
            "total_matches": stats.total_matches,
        }

    def get_best_brawler_for_map(self, map_name: str, available_brawlers: List[str]) -> Optional[str]:
        """Retorna o brawler com maior ELO para o mapa dado."""
        map_key = str(map_name).lower().strip() if map_name else "unknown"
        best = None
        best_score = -9999.0

        for brawler in available_brawlers:
            key = self._key(brawler, map_name)
            stats = self.ratings.get(key)
            if stats is None:
                # Brawler nunca jogado neste mapa: dar bonus de exploracao
                score = self.DEFAULT_ELO + 50  # Incentivar experimentar
            else:
                score = stats.score
                # Penalizar win rate muito baixo (< 30%) para evitar picks ruins
                if stats.total_matches >= 5 and stats.win_rate < 0.3:
                    score -= 100

            if score > best_score:
                best_score = score
                best = brawler

        if best:
            logger.info(f"[ELO] Melhor brawler para {map_name}: {best} (score={best_score:.0f})")
        return best

    def get_stats(self, brawler: str, map_name: Optional[str] = None) -> Optional[BrawlerMapStats]:
        """Retorna estatisticas para uma combinacao especifica."""
        key = self._key(brawler, map_name)
        return self.ratings.get(key)

    def get_top_combinations(self, n: int = 10) -> List[BrawlerMapStats]:
        """Retorna as N combinacoes brawler+mapa com maior ELO."""
        sorted_ratings = sorted(
            self.ratings.values(),
            key=lambda x: x.score,
            reverse=True
        )
        return sorted_ratings[:n]

    def get_global_summary(self) -> Dict:
        """Resumo global de todas as combinacoes."""
        if not self.ratings:
            return {"total_combinations": 0}

        scores = [s.score for s in self.ratings.values()]
        total_matches = sum(s.total_matches for s in self.ratings.values())
        total_wins = sum(s.wins for s in self.ratings.values())

        return {
            "total_combinations": len(self.ratings),
            "total_matches": total_matches,
            "total_wins": total_wins,
            "global_win_rate": total_wins / total_matches if total_matches > 0 else 0.0,
            "avg_elo": sum(scores) / len(scores),
            "max_elo": max(scores),
            "min_elo": min(scores),
            "top_brawler": self.get_top_combinations(1)[0].brawler if self.ratings else None,
            "top_map": self.get_top_combinations(1)[0].map_name if self.ratings else None,
        }

    def _save(self):
        """Persiste ratings em JSON."""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "saved_at": time.time(),
                "ratings": {f"{k[0]}|{k[1]}": asdict(v) for k, v in self.ratings.items()},
            }
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"[ELO] Ratings salvos: {self.save_path}")
        except Exception as e:
            logger.warning(f"[ELO] Falha ao salvar ratings: {e}")

    def _load(self):
        """Carrega ratings de JSON."""
        if not self.save_path.exists():
            return
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key_str, stats_dict in data.get("ratings", {}).items():
                parts = key_str.split("|", 1)
                if len(parts) == 2:
                    brawler, map_name = parts
                    stats = BrawlerMapStats(
                        brawler=stats_dict["brawler"],
                        map_name=stats_dict["map_name"],
                        wins=stats_dict.get("wins", 0),
                        losses=stats_dict.get("losses", 0),
                        draws=stats_dict.get("draws", 0),
                        score=stats_dict.get("score", 1000.0),
                        avg_rank=stats_dict.get("avg_rank", 0.0),
                        total_matches=stats_dict.get("total_matches", 0),
                        last_played=stats_dict.get("last_played", 0.0),
                        best_streak=stats_dict.get("best_streak", 0),
                        current_streak=stats_dict.get("current_streak", 0),
                        damage_dealt_avg=stats_dict.get("damage_dealt_avg", 0.0),
                        survival_time_avg=stats_dict.get("survival_time_avg", 0.0),
                    )
                    self.ratings[(brawler, map_name)] = stats
            logger.info(f"[ELO] {len(self.ratings)} combinacoes carregadas de {self.save_path}")
        except Exception as e:
            logger.warning(f"[ELO] Falha ao carregar ratings: {e}")

    def force_save(self):
        """Forca salvamento imediato (chamar no shutdown)."""
        self._save()
