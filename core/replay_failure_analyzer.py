"""
core/replay_failure_analyzer.py

Análise de Replays com IA — identifica padrões de falha automaticamente.

Guardar replays é inútil sem análise. Este módulo:
- Analisa replays de derrotas para identificar failure modes
- Classifica padrões: too_aggressive, caught_out, poor_ability, trapped
- Gera recomendações acionáveis para ajustar parâmetros
- Integra com MetaLearning para adaptação automática
- Exporta relatórios JSON para revisão humana

Integra com o ReplayFailureAnalyzer existente (replay_analyzer.py)
mas adiciona análise estatística e recomendações automáticas.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Counter
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


class FailureMode:
    """Tipos de falha detectáveis em replays."""
    TOO_AGGRESSIVE = "too_aggressive"
    CAUGHT_OUT_OF_POSITION = "caught_out_of_position"
    POOR_ABILITY_USAGE = "poor_ability_usage"
    TRAPPED_BY_WALLS = "trapped_by_walls"
    IGNORED_THREAT = "ignored_threat"
    BAD_SUPER_TIMING = "bad_super_timing"
    OVEREXTENDED = "overextended"
    TEAM_ISOLATION = "team_isolation"


@dataclass
class ReplayAnalysis:
    """Resultado da análise de um replay."""
    replay_id: str
    match_result: str
    brawler: str
    map_name: str
    duration_seconds: float
    failure_modes: Dict[str, int] = field(default_factory=dict)
    critical_moments: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    severity_score: float = 0.0  # 0-1


class ReplayFailureAnalyzer:
    """
    Analisa replays de derrotas para identificar padrões de falha
    e gerar recomendações de ajuste.
    """

    def __init__(self, replay_dir: Path = Path("data/replays"), report_dir: Path = Path("data/replay_reports")):
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Acumuladores estatísticos
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._brawler_failure_map: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._map_failure_map: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._analyses: List[ReplayAnalysis] = []

        logger.info("[REPLAY_ANALYZER] Inicializado (replays=%s)", self.replay_dir)

    # ------------------------------------------------------------------
    # Análise principal
    # ------------------------------------------------------------------

    def analyze_losses(self, limit: int = 50) -> Dict[str, Any]:
        """
        Analisa últimos N replays de derrota e identifica padrões.
        """
        loss_replays = self._find_loss_replays(limit)
        if not loss_replays:
            return {"status": "no_loss_replays_found"}

        for replay_path in loss_replays:
            analysis = self._analyze_single_replay(replay_path)
            if analysis:
                self._analyses.append(analysis)
                self._accumulate(analysis)

        # Gerar relatório consolidado
        primary_failure = self._get_primary_failure_mode()
        recommendations = self._generate_recommendations(primary_failure)

        report = {
            "analyzed_count": len(loss_replays),
            "primary_failure_mode": primary_failure,
            "failure_distribution": dict(self._failure_counts),
            "recommendations": recommendations,
            "brawler_breakdown": {b: dict(f) for b, f in self._brawler_failure_map.items()},
            "map_breakdown": {m: dict(f) for m, f in self._map_failure_map.items()},
            "severity_trend": self._calculate_severity_trend(),
        }

        # Salvar relatório
        self._save_report(report)

        logger.info(
            "[REPLAY_ANALYZER] Análise completa: primary=%s, analyzed=%d",
            primary_failure, len(loss_replays)
        )
        return report

    def _analyze_single_replay(self, replay_path: Path) -> Optional[ReplayAnalysis]:
        """Analisa um replay individual."""
        try:
            with open(replay_path, "r", encoding="utf-8") as f:
                replay_data = json.load(f)
        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
            logger.warning("[REPLAY_ANALYZER] Erro ao ler %s: %s", replay_path.name, e)
            return None

        events = replay_data.get("events", [])
        metadata = replay_data.get("metadata", {})

        analysis = ReplayAnalysis(
            replay_id=replay_path.stem,
            match_result=metadata.get("result", "loss"),
            brawler=metadata.get("brawler", "unknown"),
            map_name=metadata.get("map", "unknown"),
            duration_seconds=metadata.get("duration", 0),
        )

        # Detectar padrões de falha nos eventos
        self._detect_too_aggressive(events, analysis)
        self._detect_caught_out(events, analysis)
        self._detect_poor_ability(events, analysis)
        self._detect_trapped(events, analysis)
        self._detect_ignored_threat(events, analysis)
        self._detect_overextended(events, analysis)

        # Calcular severidade
        total_failures = sum(analysis.failure_modes.values())
        analysis.severity_score = min(1.0, total_failures / 10.0)

        # Gerar recomendações específicas para este replay
        analysis.recommendations = self._replay_recommendations(analysis)

        return analysis

    # ------------------------------------------------------------------
    # Detectores de padrão
    # ------------------------------------------------------------------

    def _detect_too_aggressive(self, events: List[Dict], analysis: ReplayAnalysis):
        """Detecta mortes por agressividade excessiva."""
        deaths = [e for e in events if e.get("type") == "death"]
        for death in deaths:
            # Se morreu com <30% HP e estava atacando
            if death.get("health_before", 1.0) < 0.3 and death.get("action_before") == "attack":
                analysis.failure_modes[FailureMode.TOO_AGGRESSIVE] = analysis.failure_modes.get(FailureMode.TOO_AGGRESSIVE, 0) + 1
                analysis.critical_moments.append({
                    "time": death.get("timestamp"),
                    "type": FailureMode.TOO_AGGRESSIVE,
                    "description": "Morreu atacando com HP baixo",
                })

    def _detect_caught_out(self, events: List[Dict], analysis: ReplayAnalysis):
        """Detecta posicionamento ruim (sem cover, flanqueado)."""
        deaths = [e for e in events if e.get("type") == "death"]
        for death in deaths:
            if not death.get("had_cover", True) and death.get("enemies_nearby", 0) >= 2:
                analysis.failure_modes[FailureMode.CAUGHT_OUT_OF_POSITION] = analysis.failure_modes.get(FailureMode.CAUGHT_OUT_OF_POSITION, 0) + 1
                analysis.critical_moments.append({
                    "time": death.get("timestamp"),
                    "type": FailureMode.CAUGHT_OUT_OF_POSITION,
                    "description": "Morto sem cover com múltiplos inimigos",
                })

    def _detect_poor_ability(self, events: List[Dict], analysis: ReplayAnalysis):
        """Detecta uso ruim de habilidades."""
        # Super usada sem hit
        supers = [e for e in events if e.get("type") == "super_used"]
        for s in supers:
            if not s.get("hit_anything", True):
                analysis.failure_modes[FailureMode.POOR_ABILITY_USAGE] = analysis.failure_modes.get(FailureMode.POOR_ABILITY_USAGE, 0) + 1

    def _detect_trapped(self, events: List[Dict], analysis: ReplayAnalysis):
        """Detecta ficar preso em paredes/zonas."""
        deaths = [e for e in events if e.get("type") == "death"]
        for death in deaths:
            if death.get("near_wall", False) and death.get("escape_attempts", 0) >= 2:
                analysis.failure_modes[FailureMode.TRAPPED_BY_WALLS] = analysis.failure_modes.get(FailureMode.TRAPPED_BY_WALLS, 0) + 1
                analysis.critical_moments.append({
                    "time": death.get("timestamp"),
                    "type": FailureMode.TRAPPED_BY_WALLS,
                    "description": "Preso em parede, múltiplas tentativas de escape",
                })

    def _detect_ignored_threat(self, events: List[Dict], analysis: ReplayAnalysis):
        """Detecta ignorar ameaças óbvias."""
        deaths = [e for e in events if e.get("type") == "death"]
        for death in deaths:
            if death.get("enemy_visible_for", 0) > 3.0 and death.get("action_before") == "collect_item":
                analysis.failure_modes[FailureMode.IGNORED_THREAT] = analysis.failure_modes.get(FailureMode.IGNORED_THREAT, 0) + 1

    def _detect_overextended(self, events: List[Dict], analysis: ReplayAnalysis):
        """Detecta overextension (longe do time, profundo no mapa inimigo)."""
        deaths = [e for e in events if e.get("type") == "death"]
        for death in deaths:
            if death.get("team_distance_ratio", 0.5) > 0.8:
                analysis.failure_modes[FailureMode.OVEREXTENDED] = analysis.failure_modes.get(FailureMode.OVEREXTENDED, 0) + 1

    # ------------------------------------------------------------------
    # Recomendações
    # ------------------------------------------------------------------

    def _get_primary_failure_mode(self) -> str:
        """Retorna o failure mode mais frequente."""
        if not self._failure_counts:
            return "unknown"
        return max(self._failure_counts, key=self._failure_counts.get)

    def _generate_recommendations(self, primary_failure: str) -> List[str]:
        """Gera recomendações baseadas no failure mode primário."""
        fixes = {
            FailureMode.TOO_AGGRESSIVE: [
                "Aumentar danger_threshold no DecisionEngine",
                "Reduzir bush_aggression no perfil do brawler",
                "Aumentar retreat_threshold (recuar com mais HP)",
            ],
            FailureMode.CAUGHT_OUT_OF_POSITION: [
                "Melhorar pathfinding para priorizar cover",
                "Aumentar peso do CoverSystem nas decisões",
                "Reduzir kiting_preference se muito baixo",
            ],
            FailureMode.POOR_ABILITY_USAGE: [
                "Retreinar RL model com ability-focused rewards",
                "Aumentar cooldown entre usos de super",
                "Adicionar line-of-sight check antes de usar super",
            ],
            FailureMode.TRAPPED_BY_WALLS: [
                "Implementar melhor wall avoidance no MovementEngine",
                "Aumentar peso de escape routes no CoverSystem",
                "Adicionar previsão de posição com wall collision",
            ],
            FailureMode.IGNORED_THREAT: [
                "Aumentar threat assessment no WorldModel",
                "Reduzir peso de collect_items quando inimigos próximos",
                "Adicionar 'enemy proximity' ao state space do RL",
            ],
            FailureMode.OVEREXTENDED: [
                "Aumentar peso de 'team distance' nas decisões",
                "Adicionar soft boundary no mapa para retreat",
                "Reduzir approach_speed no perfil do brawler",
            ],
        }
        return fixes.get(primary_failure, ["Revisar parâmetros gerais de conservadorismo"])

    def _replay_recommendations(self, analysis: ReplayAnalysis) -> List[str]:
        """Recomendações específicas para um replay."""
        recs = []
        for mode, count in analysis.failure_modes.items():
            if count > 0:
                recs.extend(self._generate_recommendations(mode))
        return list(dict.fromkeys(recs))  # Remover duplicatas, manter ordem

    def _calculate_severity_trend(self) -> str:
        """Calcula tendência de severidade nas últimas análises."""
        if len(self._analyses) < 5:
            return "insufficient_data"
        recent = [a.severity_score for a in self._analyses[-10:]]
        older = [a.severity_score for a in self._analyses[-20:-10]] if len(self._analyses) >= 20 else recent[:5]
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        if avg_recent > avg_older + 0.1:
            return "worsening"
        elif avg_recent < avg_older - 0.1:
            return "improving"
        return "stable"

    # ------------------------------------------------------------------
    # Acumulação
    # ------------------------------------------------------------------

    def _accumulate(self, analysis: ReplayAnalysis):
        """Acumula estatísticas da análise."""
        for mode, count in analysis.failure_modes.items():
            self._failure_counts[mode] += count
            self._brawler_failure_map[analysis.brawler][mode] += count
            self._map_failure_map[analysis.map_name][mode] += count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_loss_replays(self, limit: int) -> List[Path]:
        """Encontra arquivos de replay de derrotas."""
        if not self.replay_dir.exists():
            return []
        all_replays = sorted(self.replay_dir.glob("*.replay"), key=lambda p: p.stat().st_mtime, reverse=True)
        loss_replays = []
        for path in all_replays:
            if len(loss_replays) >= limit:
                break
            # Ler metadata para verificar se é loss
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("metadata", {}).get("result") in ("loss", "defeat"):
                    loss_replays.append(path)
            except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError):
                # Fallback: nome do arquivo
                if "loss" in path.name.lower() or "defeat" in path.name.lower():
                    loss_replays.append(path)
        return loss_replays

    def _save_report(self, report: Dict[str, Any]):
        """Salva relatório consolidado."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.report_dir / f"failure_analysis_{timestamp}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("[REPLAY_ANALYZER] Relatório salvo: %s", path.name)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def get_failure_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas acumuladas de falhas."""
        return {
            "total_failures_by_mode": dict(self._failure_counts),
            "brawler_breakdown": {b: dict(f) for b, f in self._brawler_failure_map.items()},
            "map_breakdown": {m: dict(f) for m, f in self._map_failure_map.items()},
            "total_analyses": len(self._analyses),
        }

    def get_top_recommendations(self, n: int = 3) -> List[str]:
        """Retorna top N recomendações baseadas em todos os dados."""
        primary = self._get_primary_failure_mode()
        all_recs = self._generate_recommendations(primary)
        return all_recs[:n]

    def reset_stats(self):
        """Limpa estatísticas acumuladas."""
        self._failure_counts.clear()
        self._brawler_failure_map.clear()
        self._map_failure_map.clear()
        self._analyses.clear()
