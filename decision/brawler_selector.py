"""
brawler_selector.py

Intelligent brawler selection system based on performance analysis.

Uses multi-armed bandit (Thompson Sampling) to balance exploration
and exploitation of different brawlers.

Features:
- Performance tracking per brawler
- Map-specific recommendations
- Matchup analysis
- Auto-switch between matches
- Thompson Sampling for exploration
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BrawlerStats:
    """Statistics for a single brawler."""
    name: str
    matches_played: int = 0
    matches_won: int = 0
    kills: int = 0
    deaths: int = 0
    damage_dealt: float = 0.0
    trophies_gained: int = 0

    # Map-specific stats
    map_stats: dict[str, dict] = field(default_factory=dict)

    # Thompson Sampling parameters
    alpha: float = 1.0  # Success prior
    beta: float = 1.0   # Failure prior

    @property
    def win_rate(self) -> float:
        if self.matches_played == 0:
            return 0.0
        return self.matches_won / self.matches_played

    @property
    def kda_ratio(self) -> float:
        if self.deaths == 0:
            return self.kills
        return self.kills / self.deaths

    @property
    def average_damage(self) -> float:
        if self.matches_played == 0:
            return 0.0
        return self.damage_dealt / self.matches_played

    def update_match_result(self, won: bool, kills: int, deaths: int, damage: float, trophies: int):
        """Update stats after a match."""
        self.matches_played += 1
        if won:
            self.matches_won += 1
            self.alpha += 1  # Success for Thompson Sampling
        else:
            self.beta += 1  # Failure for Thompson Sampling

        self.kills += kills
        self.deaths += deaths
        self.damage_dealt += damage
        self.trophies_gained += trophies

    def update_map_result(self, map_name: str, won: bool):
        """Update map-specific stats."""
        if map_name not in self.map_stats:
            self.map_stats[map_name] = {'played': 0, 'won': 0}

        self.map_stats[map_name]['played'] += 1
        if won:
            self.map_stats[map_name]['won'] += 1

    def get_map_win_rate(self, map_name: str) -> float:
        """Get win rate for specific map."""
        if map_name not in self.map_stats:
            return 0.0

        stats = self.map_stats[map_name]
        if stats['played'] == 0:
            return 0.0

        return stats['won'] / stats['played']


class BrawlerSelector:
    """
    Intelligent brawler selector using Thompson Sampling.

    Balances exploration (trying new brawlers) with exploitation
    (using known good brawlers).

    Features:
    - Counter-pick logic: select brawlers that counter the meta
    - Map-specific recommendations
    - Performance tracking per brawler
    """

    # Counter-pick table: brawler -> list of brawlers it counters
    COUNTER_PICKS = {
        "Shelly": ["El Primo", "Bull", "Rosa", "Jacky", "Bibi"],
        "Colt": ["Rosa", "Jacky", "Pam", "8-Bit", "Frank"],
        "Brock": ["Rosa", "Jacky", "Pam", "Barley", "Dynamike"],
        "Rico": ["Rosa", "Jacky", "Pam", "Frank", "8-Bit"],
        "Piper": ["Brock", "Colt", "Rico", "Nani", "Belle"],
        "Crow": ["Shelly", "El Primo", "Bull", "Spike", "Tick"],
        "Leon": ["Piper", "Brock", "Colt", "Rico", "Tick"],
        "Mortis": ["Piper", "Brock", "Colt", "Tick", "Barley"],
        "El Primo": ["Piper", "Brock", "Colt", "Tick", "Barley"],
        "Rosa": ["Shelly", "Bull", "El Primo", "Mortis", "Edgar"],
        "Gene": ["Mortis", "Leon", "Crow", "Edgar", "Stu"],
        "Tara": ["Crow", "Leon", "Mortis", "Edgar", "Sandy"],
        "Sprout": ["El Primo", "Bull", "Rosa", "Jacky", "Bibi"],
        "Poco": ["El Primo", "Bull", "Rosa", "Jacky", "Bibi"],
        "Pam": ["Crow", "Leon", "Spike", "Tick", "Barley"],
        "Sandy": ["Crow", "Leon", "Mortis", "Edgar", "Stu"],
        "Max": ["Mortis", "Leon", "Crow", "Edgar", "Stu"],
        "Emz": ["El Primo", "Bull", "Rosa", "Jacky", "Bibi"],
        "Jacky": ["Colt", "Brock", "Rico", "Piper", "Nani"],
        "Gale": ["El Primo", "Bull", "Rosa", "Jacky", "Mortis"],
        "Surge": ["Shelly", "Bull", "El Primo", "Rosa", "Jacky"],
        "Fang": ["Piper", "Brock", "Colt", "Tick", "Barley"],
        "Stu": ["Piper", "Brock", "Colt", "Rico", "Tick"],
        "Ash": ["Crow", "Leon", "Spike", "Tick", "Barley"],
    }

    def __init__(
        self,
        stats_file: Path | None = None,
        exploration_rate: float = 0.2,
        min_matches_before_exploit: int = 5,
    ):
        self.stats_file = stats_file or Path("data/brawler_stats.json")
        self.exploration_rate = exploration_rate
        self.min_matches_before_exploit = min_matches_before_exploit

        # Brawler statistics
        self.brawlers: dict[str, BrawlerStats] = {}

        # Load existing stats
        self._load_stats()

    def _load_stats(self):
        """Load brawler statistics from file."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, encoding='utf-8') as f:
                    data = json.load(f)

                for name, stats_data in data.items():
                    self.brawlers[name] = BrawlerStats(
                        name=name,
                        matches_played=stats_data.get('matches_played', 0),
                        matches_won=stats_data.get('matches_won', 0),
                        kills=stats_data.get('kills', 0),
                        deaths=stats_data.get('deaths', 0),
                        damage_dealt=stats_data.get('damage_dealt', 0.0),
                        trophies_gained=stats_data.get('trophies_gained', 0),
                        alpha=stats_data.get('alpha', 1.0),
                        beta=stats_data.get('beta', 1.0),
                        map_stats=stats_data.get('map_stats', {}),
                    )

                logger.info(f"Loaded stats for {len(self.brawlers)} brawlers")
            except Exception as e:
                logger.error(f"Failed to load stats: {e}")

    def _save_stats(self):
        """Save brawler statistics to file."""
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        for name, stats in self.brawlers.items():
            data[name] = {
                'matches_played': stats.matches_played,
                'matches_won': stats.matches_won,
                'kills': stats.kills,
                'deaths': stats.deaths,
                'damage_dealt': stats.damage_dealt,
                'trophies_gained': stats.trophies_gained,
                'alpha': stats.alpha,
                'beta': stats.beta,
                'map_stats': stats.map_stats,
            }

        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved stats for {len(self.brawlers)} brawlers")

    def add_brawler(self, name: str):
        """Add a new brawler to the selector."""
        if name not in self.brawlers:
            self.brawlers[name] = BrawlerStats(name=name)
            logger.info(f"Added brawler: {name}")

    def record_match(
        self,
        brawler_name: str,
        won: bool,
        kills: int = 0,
        deaths: int = 0,
        damage: float = 0.0,
        trophies: int = 0,
        map_name: str | None = None,
    ):
        """Record match result for a brawler."""
        if brawler_name not in self.brawlers:
            self.add_brawler(brawler_name)

        self.brawlers[brawler_name].update_match_result(won, kills, deaths, damage, trophies)

        if map_name:
            self.brawlers[brawler_name].update_map_result(map_name, won)

        self._save_stats()

    def select_brawler(
        self,
        available_brawlers: list[str],
        map_name: str | None = None,
        force_explore: bool = False,
        enemy_brawlers: list[str] | None = None,
    ) -> str:
        """
        Select a brawler using Thompson Sampling with counter-pick logic.

        Args:
            available_brawlers: List of brawler names to choose from
            map_name: Optional current map name
            force_explore: Force exploration (ignore performance)
            enemy_brawlers: Optional list of enemy brawler names for counter-picking

        Returns:
            Selected brawler name
        """
        if not available_brawlers:
            raise ValueError("No brawlers available")

        # Ensure all available brawlers are in stats
        for brawler in available_brawlers:
            self.add_brawler(brawler)

        # Filter to available brawlers
        available_stats = {
            name: self.brawlers[name]
            for name in available_brawlers
            if name in self.brawlers
        }

        if not available_stats:
            # Fallback to first available
            return available_brawlers[0]

        # Counter-pick: if we know enemy brawlers, prefer brawlers that counter them
        if enemy_brawlers and not force_explore:
            counter_pick = self._select_counter_pick(available_brawlers, enemy_brawlers)
            if counter_pick:
                logger.info(f"Counter-pick: Selected {counter_pick} against {enemy_brawlers}")
                return counter_pick

        # Force exploration or random chance
        if force_explore or random.random() < self.exploration_rate:
            selected = random.choice(list(available_stats.keys()))
            logger.info(f"Exploration: Selected {selected}")
            return selected

        # Map-specific selection if map provided
        if map_name:
            map_best = self._select_best_for_map(available_stats, map_name)
            if map_best:
                logger.info(f"Map-based selection: {map_best} for {map_name}")
                return map_best

        # Thompson Sampling for exploitation
        best_brawler = self._thompson_sample(available_stats)

        # Check if we have enough data to exploit
        if best_brawler.matches_played < self.min_matches_before_exploit:
            # Not enough data, explore instead
            selected = random.choice(list(available_stats.keys()))
            logger.info(f"Insufficient data, exploring: {selected}")
            return selected

        logger.info(f"Exploitation: Selected {best_brawler} (win rate: {best_brawler.win_rate:.2f})")
        return best_brawler.name

    def _thompson_sample(self, brawlers: dict[str, BrawlerStats]) -> BrawlerStats:
        """
        Select brawler using Thompson Sampling.

        Samples from Beta distribution for each brawler and selects
        the one with highest sample.
        """
        samples = []

        for _name, stats in brawlers.items():
            # Sample from Beta(alpha, beta)
            sample = np.random.beta(stats.alpha, stats.beta)
            samples.append((sample, stats))

        # Select brawler with highest sample
        samples.sort(key=lambda x: -x[0])
        return samples[0][1]

    def _select_best_for_map(
        self,
        brawlers: dict[str, BrawlerStats],
        map_name: str,
    ) -> str | None:
        """Select best brawler for specific map."""
        map_win_rates = {}

        for name, stats in brawlers.items():
            win_rate = stats.get_map_win_rate(map_name)
            if stats.map_stats.get(map_name, {}).get('played', 0) > 0:
                map_win_rates[name] = win_rate

        if map_win_rates:
            # Select brawler with highest map win rate
            best = max(map_win_rates.items(), key=lambda x: x[1])
            if best[1] > 0.5:  # Only use if win rate > 50%
                return best[0]

        return None

    def _select_counter_pick(
        self,
        available_brawlers: list[str],
        enemy_brawlers: list[str],
    ) -> str | None:
        """Select a brawler that counters the enemy team composition.

        Scores each available brawler by how many enemy brawlers it counters.
        Weighted by historical win rate against those brawlers.

        Args:
            available_brawlers: Brawlers we can choose from
            enemy_brawlers: Enemy brawler names

        Returns:
            Best counter-pick brawler name, or None if no good counter found
        """
        if not enemy_brawlers:
            return None

        counter_scores: dict[str, float] = {}

        for brawler in available_brawlers:
            countered = self.COUNTER_PICKS.get(brawler, [])
            if not countered:
                counter_scores[brawler] = 0.0
                continue

            # Count how many enemy brawlers this brawler counters
            score = 0.0
            for enemy in enemy_brawlers:
                if enemy in countered:
                    # Base score for countering
                    score += 1.0
                    # Bonus if we have historical win data against this enemy
                    if brawler in self.brawlers:
                        stats = self.brawlers[brawler]
                        enemy_key = f"vs_{enemy}"
                        vs_wr = stats.map_stats.get(enemy_key, {}).get("win_rate", 0.5)
                        score += vs_wr * 0.5  # Up to +0.5 bonus for known good matchup

            counter_scores[brawler] = score

        # Find best counter pick
        if not counter_scores:
            return None

        best = max(counter_scores.items(), key=lambda x: x[1])

        # Only return if the counter score is meaningful (counters at least 1 enemy)
        if best[1] >= 1.0:
            return best[0]

        return None

    def get_recommendation(
        self,
        brawler_name: str,
        map_name: str | None = None,
    ) -> dict:
        """
        Get recommendation for a specific brawler.

        Returns performance metrics and recommendation.
        """
        if brawler_name not in self.brawlers:
            return {
                'brawler': brawler_name,
                'recommendation': 'unknown',
                'reason': 'No data available',
                'win_rate': 0.0,
                'matches_played': 0,
            }

        stats = self.brawlers[brawler_name]

        # Determine recommendation
        if stats.matches_played < 5:
            recommendation = 'explore'
            reason = 'Insufficient data'
        elif stats.win_rate > 0.6:
            recommendation = 'strong'
            reason = 'High win rate'
        elif stats.win_rate > 0.5:
            recommendation = 'good'
            reason = 'Above average'
        else:
            recommendation = 'weak'
            reason = 'Below average win rate'

        return {
            'brawler': brawler_name,
            'recommendation': recommendation,
            'reason': reason,
            'win_rate': stats.win_rate,
            'kda': stats.kda_ratio,
            'matches_played': stats.matches_played,
            'map_win_rate': stats.get_map_win_rate(map_name) if map_name else None,
        }

    def get_leaderboard(self, limit: int = 10) -> list[dict]:
        """Get leaderboard of brawlers by win rate."""
        leaderboard = []

        for name, stats in self.brawlers.items():
            if stats.matches_played >= 5:  # Minimum matches
                leaderboard.append({
                    'name': name,
                    'win_rate': stats.win_rate,
                    'kda': stats.kda_ratio,
                    'matches': stats.matches_played,
                })

        # Sort by win rate
        leaderboard.sort(key=lambda x: -x['win_rate'])

        return leaderboard[:limit]

    def reset_stats(self, brawler_name: str | None = None):
        """Reset statistics for a brawler or all brawlers."""
        if brawler_name:
            if brawler_name in self.brawlers:
                self.brawlers[brawler_name] = BrawlerStats(name=brawler_name)
                logger.info(f"Reset stats for {brawler_name}")
        else:
            self.brawlers.clear()
            logger.info("Reset all brawler stats")

        self._save_stats()


def main():
    """Test brawler selector."""
    logging.basicConfig(level=logging.INFO)

    selector = BrawlerSelector()

    # Add some brawlers
    for brawler in ['colt', 'mortis', 'el_primo', 'spike']:
        selector.add_brawler(brawler)

    # Simulate some matches
    selector.record_match('colt', won=True, kills=3, deaths=1, damage=2000, trophies=10)
    selector.record_match('colt', won=True, kills=2, deaths=0, damage=1500, trophies=8)
    selector.record_match('mortis', won=False, kills=1, deaths=2, damage=800, trophies=-5)
    selector.record_match('el_primo', won=True, kills=4, deaths=1, damage=2500, trophies=12)

    # Select brawler
    available = ['colt', 'mortis', 'el_primo', 'spike']
    selected = selector.select_brawler(available)
    print(f"Selected brawler: {selected}")

    # Get leaderboard
    leaderboard = selector.get_leaderboard()
    print(f"Leaderboard: {leaderboard}")


if __name__ == "__main__":
    main()
