"""
core/learning_metrics.py

Recolhe e persiste métricas de sessões de aprendizagem na Training Cave.
Não interfere no histórico oficial de partidas (match_history.json).
"""

import json
import time
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class LearningMatchResult:
    """Resultado de uma partida no modo de aprendizagem."""
    match_id: str
    brawler: str
    start_time: str
    duration_seconds: float = 0.0
    kills: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    deaths: int = 0
    detections_total: int = 0
    detections_enemies: int = 0
    detections_player: int = 0
    actions_attack: int = 0
    actions_move: int = 0
    actions_super: int = 0
    survival_time: float = 0.0
    result: str = "unknown"  # completed, died, timeout
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LearningMetricsCollector:
    """
    Acumula métricas frame-a-frame durante uma sessão de aprendizagem
    e persiste num ficheiro JSON separado do histórico oficial.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(__file__).parent.parent / "pylaai_workspace"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.output_dir / "learning_sessions.json"

        self.matches: List[LearningMatchResult] = []
        self.current_match: Optional[LearningMatchResult] = None
        self._frame_counter = 0
        self._session_start = time.time()
        # Histórico de frames para gráficos live (últimos 300 frames ~ 30s a 10fps)
        self.frames_history: deque = deque(maxlen=300)

        logger.info("[LEARNING_METRICS] Inicializado. Ficheiro: %s", self.session_file)

    # ------------------------------------------------------------------
    # Ciclo de vida de um match
    # ------------------------------------------------------------------

    def start_match(self, brawler: str) -> None:
        """Inicia a contagem de métricas para um novo match."""
        self.current_match = LearningMatchResult(
            match_id=f"learn_{int(time.time())}_{len(self.matches)}",
            brawler=brawler,
            start_time=datetime.now().isoformat(),
        )
        self._frame_counter = 0
        logger.info("[LEARNING_METRICS] Match iniciado: %s | Brawler: %s",
                    self.current_match.match_id, brawler)

    def end_match(self, result: str = "completed", duration: Optional[float] = None) -> LearningMatchResult:
        """Finaliza o match atual, calcula duração e guarda na lista."""
        if self.current_match is None:
            logger.warning("[LEARNING_METRICS] end_match chamado sem start_match")
            return LearningMatchResult(match_id="none", brawler="none", start_time="")

        if duration is None:
            duration = time.time() - self._session_start  # aproximado
        self.current_match.duration_seconds = duration
        self.current_match.result = result
        self.current_match.survival_time = duration

        self.matches.append(self.current_match)
        logger.info("[LEARNING_METRICS] Match finalizado: %s | Resultado: %s | Kills: %s | DMG: %.0f",
                    self.current_match.match_id, result,
                    self.current_match.kills, self.current_match.damage_dealt)

        match = self.current_match
        self.current_match = None
        self._persist()
        return match

    # ------------------------------------------------------------------
    # Atualizações frame-a-frame
    # ------------------------------------------------------------------

    def log_frame(
        self,
        enemies_detected: int = 0,
        player_detected: bool = False,
        action_taken: Optional[str] = None,
        damage_dealt: float = 0.0,
        damage_taken: float = 0.0,
    ) -> None:
        """Regista métricas de um único frame de gameplay."""
        if self.current_match is None:
            return

        self._frame_counter += 1
        self.current_match.detections_total += 1
        self.current_match.detections_enemies += enemies_detected
        if player_detected:
            self.current_match.detections_player += 1

        self.current_match.damage_dealt += damage_dealt
        self.current_match.damage_taken += damage_taken

        if action_taken:
            norm = action_taken.lower()
            if "attack" in norm or "shoot" in norm:
                self.current_match.actions_attack += 1
            elif "super" in norm or "ult" in norm:
                self.current_match.actions_super += 1
            elif "move" in norm or "walk" in norm or "kite" in norm:
                self.current_match.actions_move += 1

        # Guardar snapshot do frame para gráficos live
        self.frames_history.append({
            "timestamp": time.time(),
            "frame": self._frame_counter,
            "enemies_detected": enemies_detected,
            "player_detected": player_detected,
            "action": action_taken,
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
        })

    def log_kill(self, count: int = 1) -> None:
        """Regista kill(s) no match atual."""
        if self.current_match is not None:
            self.current_match.kills += count
            logger.debug("[LEARNING_METRICS] Kill +%s (total=%s)", count, self.current_match.kills)

    def log_death(self) -> None:
        """Regista uma morte no match atual."""
        if self.current_match is not None:
            self.current_match.deaths += 1
            self.current_match.result = "died"
            logger.debug("[LEARNING_METRICS] Death registada (total=%s)", self.current_match.deaths)

    def log_note(self, note: str) -> None:
        """Adiciona uma nota textual ao match atual."""
        if self.current_match is not None:
            self.current_match.notes.append(note)

    # ------------------------------------------------------------------
    # Persistência e sumário
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Guarda todos os matches no ficheiro JSON."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "total_sessions": len(self.matches),
                "matches": [m.to_dict() for m in self.matches],
            }
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug("[LEARNING_METRICS] Sessão persistida: %s matches", len(self.matches))
        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
            logger.warning("[LEARNING_METRICS] Falha ao persistir: %s", e)

    def get_summary(self) -> Dict[str, Any]:
        """Calcula estatísticas agregadas de todos os matches da sessão."""
        if not self.matches:
            return {"total_matches": 0, "message": "Nenhum match jogado na sessão de aprendizagem."}

        total = len(self.matches)
        total_kills = sum(m.kills for m in self.matches)
        total_deaths = sum(m.deaths for m in self.matches)
        total_dmg_dealt = sum(m.damage_dealt for m in self.matches)
        total_dmg_taken = sum(m.damage_taken for m in self.matches)
        total_actions = sum(m.actions_attack + m.actions_move + m.actions_super for m in self.matches)
        total_detections = sum(m.detections_total for m in self.matches)
        avg_duration = sum(m.duration_seconds for m in self.matches) / total

        return {
            "total_matches": total,
            "total_kills": total_kills,
            "total_deaths": total_deaths,
            "kdr": round(total_kills / max(1, total_deaths), 2),
            "total_damage_dealt": round(total_dmg_dealt, 1),
            "total_damage_taken": round(total_dmg_taken, 1),
            "avg_match_duration_seconds": round(avg_duration, 1),
            "total_actions": total_actions,
            "total_detections": total_detections,
            "deaths_by_timeout": sum(1 for m in self.matches if m.result == "timeout"),
            "deaths_by_died": sum(1 for m in self.matches if m.result == "died"),
            "completed": sum(1 for m in self.matches if m.result == "completed"),
        }

    def get_frame_history(self, limit: int = 300) -> List[Dict[str, Any]]:
        """Retorna os últimos N frames de histórico para gráficos live."""
        return list(self.frames_history)[-limit:]

    def get_session_history(self) -> List[Dict[str, Any]]:
        """Retorna o histórico completo de sessões anteriores do ficheiro JSON."""
        try:
            if self.session_file.exists():
                with open(self.session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("matches", [])
        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
            logger.debug("[LEARNING_METRICS] Falha ao ler histórico: %s", e)
        return []

    def print_summary(self) -> None:
        """Imprime o sumário da sessão nos logs."""
        summary = self.get_summary()
        if summary.get("total_matches") == 0:
            logger.info("[LEARNING_MODE] Nenhum match registado na sessão de aprendizagem.")
            return

        logger.info("=" * 60)
        logger.info("[LEARNING_MODE] SUMÁRIO DA SESSÃO DE APRENDIZAGEM")
        logger.info("=" * 60)
        logger.info("Partidas jogadas:      %s", summary["total_matches"])
        logger.info("Kills totais:          %s", summary["total_kills"])
        logger.info("Mortes totais:         %s", summary["total_deaths"])
        logger.info("K/D Ratio:             %s", summary["kdr"])
        logger.info("Dano infligido:        %.0f", summary["total_damage_dealt"])
        logger.info("Dano recebido:         %.0f", summary["total_damage_taken"])
        logger.info("Ações totais:          %s", summary["total_actions"])
        logger.info("Deteções totais:       %s", summary["total_detections"])
        logger.info("Duração média/partida: %.1fs", summary["avg_match_duration_seconds"])
        logger.info("Completados:           %s | Mortos: %s | Timeout: %s",
                    summary["completed"], summary["deaths_by_died"], summary["deaths_by_timeout"])
        logger.info("Ficheiro guardado em:  %s", self.session_file)
        logger.info("=" * 60)
