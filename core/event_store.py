"""
core/event_store.py

Event Sourcing + CQRS para Soberana Omega.

Armazena TODOS os eventos do domínio em append-only log para:
- Auditoria completa (debug pós-mortem de bans/crashes)
- Replay de episódios problemáticos
- Análise de padrões comportamentais
- Reconstrução de estado em qualquer ponto temporal

Separação CQRS:
- Command side: append_event (escrita otimizada, sequencial)
- Query side: projeções para leitura otimizada
"""

import gzip
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DomainEventType(Enum):
    """Tipos de eventos de domínio."""
    # Lifecycle
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    MATCH_STARTED = "match_started"
    MATCH_ENDED = "match_ended"

    # Game State
    STATE_TRANSITION = "state_transition"
    BRAWLER_SELECTED = "brawler_selected"
    MAP_DETECTED = "map_detected"

    # Combat
    PLAYER_DAMAGED = "player_damaged"
    PLAYER_HEALED = "player_healed"
    ENEMY_HIT = "enemy_hit"
    ENEMY_ELIMINATED = "enemy_eliminated"
    PLAYER_DIED = "player_died"
    SUPER_USED = "super_used"
    GADGET_USED = "gadget_used"

    # Items
    POWER_CUBE_COLLECTED = "power_cube_collected"
    GEM_COLLECTED = "gem_collected"
    STAR_COLLECTED = "star_collected"

    # Decisions
    ACTION_TAKEN = "action_taken"
    INTENT_CHANGED = "intent_changed"
    TARGET_LOCKED = "target_locked"

    # Safety / Anti-ban
    APM_THROTTLED = "apm_throttled"
    BREAK_STARTED = "break_started"
    BREAK_ENDED = "break_ended"
    RATE_LIMIT_TRIGGERED = "rate_limit_triggered"

    # System
    ERROR_OCCURRED = "error_occurred"
    RECOVERY_TRIGGERED = "recovery_triggered"
    DEGRADATION_CHANGED = "degradation_changed"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_RESTORED = "checkpoint_restored"

    # RL / Learning
    REWARD_RECEIVED = "reward_received"
    Q_TABLE_UPDATED = "q_table_updated"
    STRATEGY_CHANGED = "strategy_changed"


@dataclass(frozen=True)
class DomainEvent:
    """Evento imutável do domínio — a unidade básica do event store."""
    event_type: DomainEventType
    timestamp: float
    aggregate_id: str           # ex: session_id, match_id
    aggregate_type: str         # ex: "session", "match", "player"
    payload: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "payload": self.payload,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DomainEvent":
        return cls(
            event_type=DomainEventType(data["event_type"]),
            timestamp=data["timestamp"],
            aggregate_id=data["aggregate_id"],
            aggregate_type=data["aggregate_type"],
            payload=data.get("payload", {}),
            version=data.get("version", 1),
            metadata=data.get("metadata", {}),
        )


class EventStore:
    """
    Append-only event store com compressão e rotação.

    Design:
    - Cada dia tem seu arquivo events_YYYY-MM-DD.jsonl[.gz]
    - Escrita é sequencial e lock-protegida
    - Leitura via iteradores lazy (não carrega tudo em memória)
    - Reconstrução de estado via replay de eventos
    """

    def __init__(
        self,
        base_dir: Path = Path("core/events"),
        compress_after_days: int = 1,
        max_event_age_days: int = 90,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.compress_after_days = compress_after_days
        self.max_event_age_days = max_event_age_days

        self._lock = threading.RLock()
        self._current_file: Path | None = None
        self._current_date: str | None = None
        self._ensure_daily_file()

        # Métricas
        self._events_appended = 0
        self._events_replayed = 0

        logger.info("[EVENT_STORE] Inicializado em %s", self.base_dir)

    # ------------------------------------------------------------------
    # Escrita (Command Side)
    # ------------------------------------------------------------------

    def append_event(self, event: DomainEvent) -> bool:
        """Persiste evento em append-only log. Thread-safe."""
        with self._lock:
            self._ensure_daily_file()
            try:
                line = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
                with open(self._current_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                self._events_appended += 1
                return True
            except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.error("[EVENT_STORE] Falha ao append evento: %s", e)
                return False

    def append(self, event_type: DomainEventType, aggregate_id: str, aggregate_type: str, payload: dict[str, Any] = None, **metadata) -> bool:
        """Conveniência: cria e persiste evento numa chamada."""
        event = DomainEvent(
            event_type=event_type,
            timestamp=time.time(),
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            payload=payload or {},
            metadata=metadata or {},
        )
        return self.append_event(event)

    # ------------------------------------------------------------------
    # Leitura / Replay (Query Side)
    # ------------------------------------------------------------------

    def replay_events(
        self,
        from_timestamp: float | None = None,
        to_timestamp: float | None = None,
        event_types: list[DomainEventType] | None = None,
        aggregate_id: str | None = None,
    ) -> Iterator[DomainEvent]:
        """
        Reconstrói stream de eventos filtrado.
        Lazy iterator — não carrega tudo em memória.
        """
        files = sorted(self.base_dir.glob("events_*.jsonl*"))
        target_types = {e.value for e in event_types} if event_types else None

        for file_path in files:
            # Se arquivo for gz, precisamos abrir com gzip
            opener = gzip.open if file_path.suffix == ".gz" else open
            try:
                with opener(file_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            event = DomainEvent.from_dict(data)
                        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError):
                            continue

                        # Filtros
                        if from_timestamp and event.timestamp < from_timestamp:
                            continue
                        if to_timestamp and event.timestamp > to_timestamp:
                            continue
                        if target_types and event.event_type.value not in target_types:
                            continue
                        if aggregate_id and event.aggregate_id != aggregate_id:
                            continue

                        self._events_replayed += 1
                        yield event
            except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning("[EVENT_STORE] Erro ao ler %s: %s", file_path.name, e)

    def get_events_for_aggregate(self, aggregate_id: str) -> list[DomainEvent]:
        """Retorna todos os eventos de um aggregate específico."""
        return list(self.replay_events(aggregate_id=aggregate_id))

    def get_events_between(self, t0: float, t1: float, event_types: list[DomainEventType] | None = None) -> list[DomainEvent]:
        """Retorna eventos num intervalo temporal."""
        return list(self.replay_events(from_timestamp=t0, to_timestamp=t1, event_types=event_types))

    # ------------------------------------------------------------------
    # Projeções CQRS (cache de leitura)
    # ------------------------------------------------------------------

    def build_projection(
        self,
        projector: Callable[[Any, DomainEvent], Any],
        initial_state: Any,
        from_timestamp: float | None = None,
        to_timestamp: float | None = None,
        event_types: list[DomainEventType] | None = None,
    ) -> Any:
        """
        Aplica função projector a cada evento, acumulando estado.
        Permite reconstruir QUALQUER view do sistema a partir dos eventos.
        """
        state = initial_state
        for event in self.replay_events(from_timestamp, to_timestamp, event_types):
            state = projector(state, event)
        return state

    def rebuild_session_state(self, session_id: str) -> dict:
        """
        Reconstrói estado completo de uma sessão a partir dos eventos.
        Útil para debug pós-mortem.
        """
        def projector(state: dict, event: DomainEvent) -> dict:
            state["events_count"] = state.get("events_count", 0) + 1
            state["last_event_time"] = event.timestamp

            et = event.event_type
            payload = event.payload

            if et == DomainEventType.SESSION_STARTED:
                state["session_start"] = event.timestamp
                state["profile"] = payload.get("profile", {})
            elif et == DomainEventType.MATCH_STARTED:
                state.setdefault("matches", []).append({
                    "match_id": event.aggregate_id,
                    "start": event.timestamp,
                    "brawler": payload.get("brawler"),
                    "map": payload.get("map"),
                })
            elif et == DomainEventType.MATCH_ENDED:
                for m in state.get("matches", []):
                    if m["match_id"] == event.aggregate_id:
                        m["result"] = payload.get("result")
                        m["duration"] = payload.get("duration", 0)
                        m["end"] = event.timestamp
            elif et == DomainEventType.PLAYER_DIED:
                state.setdefault("deaths", []).append({
                    "time": event.timestamp,
                    "cause": payload.get("cause", "unknown"),
                    "match_id": event.aggregate_id,
                })
            elif et == DomainEventType.ERROR_OCCURRED:
                state.setdefault("errors", []).append({
                    "time": event.timestamp,
                    "error": payload.get("error", "unknown"),
                    "component": payload.get("component", ""),
                })
            return state

        events = self.get_events_for_aggregate(session_id)
        state = {"session_id": session_id}
        for event in events:
            state = projector(state, event)
        return state

    # ------------------------------------------------------------------
    # Diagnóstico pós-mortem
    # ------------------------------------------------------------------

    def post_mortem_analysis(self, session_id: str, minutes_before: float = 10.0) -> dict[str, Any]:
        """
        Análise pós-mortem de uma sessão — identifica sequência de eventos
        que levaram a um crash, ban, ou comportamento anômalo.
        """
        events = self.get_events_for_aggregate(session_id)
        if not events:
            return {"error": "No events found for session"}

        # Encontrar timestamp do último erro crítico ou fim abrupto
        last_event_time = events[-1].timestamp
        window_start = last_event_time - (minutes_before * 60)

        window_events = [e for e in events if e.timestamp >= window_start]

        # Classificar eventos por severidade
        errors = [e for e in window_events if e.event_type == DomainEventType.ERROR_OCCURRED]
        deaths = [e for e in window_events if e.event_type == DomainEventType.PLAYER_DIED]
        actions = [e for e in window_events if e.event_type == DomainEventType.ACTION_TAKEN]
        apm_events = [e for e in window_events if e.event_type == DomainEventType.APM_THROTTLED]
        rate_limits = [e for e in window_events if e.event_type == DomainEventType.RATE_LIMIT_TRIGGERED]

        # Detectar padrões suspeitos
        suspicious_patterns = []
        if len(apm_events) > 5:
            suspicious_patterns.append("high_apm_throttle")
        if len(rate_limits) > 2:
            suspicious_patterns.append("rate_limit_repeated")
        if len(errors) > 10:
            suspicious_patterns.append("cascade_errors")
        if deaths and len(deaths) / max(1, len([e for e in window_events if e.event_type == DomainEventType.MATCH_STARTED])) > 0.8:
            suspicious_patterns.append("high_death_rate")

        # Sequência de eventos críticos nos últimos 60s
        critical_sequence = [
            {
                "time": datetime.fromtimestamp(e.timestamp).isoformat(),
                "type": e.event_type.value,
                "payload_keys": list(e.payload.keys()),
            }
            for e in window_events[-50:]
        ]

        return {
            "session_id": session_id,
            "analysis_window_minutes": minutes_before,
            "total_events_in_window": len(window_events),
            "errors_count": len(errors),
            "deaths_count": len(deaths),
            "actions_count": len(actions),
            "apm_throttles": len(apm_events),
            "rate_limits": len(rate_limits),
            "suspicious_patterns": suspicious_patterns,
            "critical_event_sequence": critical_sequence,
            "recommendation": self._post_mortem_recommendation(suspicious_patterns),
        }

    def _post_mortem_recommendation(self, patterns: list[str]) -> str:
        """Gera recomendação baseada nos padrões encontrados."""
        if "rate_limit_repeated" in patterns:
            return "Aumentar breaks entre sessões; reduzir matches/hora."
        if "high_apm_throttle" in patterns:
            return "APM muito alto — revisar humanization e delays."
        if "cascade_errors" in patterns:
            return "Falhas em cascata — verificar graceful degradation e circuit breakers."
        if "high_death_rate" in patterns:
            return "Taxa de mortes alta — ajustar conservadorismo do RL ou revisar detecção."
        return "Nenhum padrão crítico identificado."

    # ------------------------------------------------------------------
    # Manutenção
    # ------------------------------------------------------------------

    def _ensure_daily_file(self):
        """Garante que o arquivo de hoje está aberto."""
        today = date.today().isoformat()
        if self._current_date != today:
            self._current_date = today
            self._current_file = self.base_dir / f"events_{today}.jsonl"
            logger.debug("[EVENT_STORE] Arquivo do dia: %s", self._current_file)

    def run_maintenance(self):
        """
        Comprime arquivos antigos e remove muito antigos.
        Deve ser chamado periodicamente (ex: a cada 24h).
        """
        now = time.time()
        for file_path in self.base_dir.glob("events_*.jsonl"):
            try:
                # Extrair data do nome
                date_str = file_path.stem.replace("events_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                age_days = (now - file_date.timestamp()) / 86400

                # Comprimir após N dias
                if age_days >= self.compress_after_days:
                    gz_path = file_path.with_suffix(".jsonl.gz")
                    if not gz_path.exists():
                        with open(file_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                            f_out.writelines(f_in)
                        file_path.unlink()
                        logger.info("[EVENT_STORE] Comprimido %s", gz_path.name)

                # Remover após max age
                if age_days >= self.max_event_age_days:
                    file_path.unlink(missing_ok=True)
                    gz_path = file_path.with_suffix(".jsonl.gz")
                    gz_path.unlink(missing_ok=True)
                    logger.info("[EVENT_STORE] Removido %s (expirado)", file_path.name)

            except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning("[EVENT_STORE] Erro na manutenção de %s: %s", file_path.name, e)

    def get_stats(self) -> dict[str, Any]:
        """Retorna estatísticas do event store."""
        total_files = len(list(self.base_dir.glob("events_*")))
        return {
            "events_appended": self._events_appended,
            "events_replayed": self._events_replayed,
            "current_file": str(self._current_file) if self._current_file else None,
            "total_files": total_files,
        }


# ------------------------------------------------------------------------------
# Singleton global (opcional)
# ------------------------------------------------------------------------------

_default_store: EventStore | None = None


def get_event_store() -> EventStore:
    """Retorna instância global do event store."""
    global _default_store
    if _default_store is None:
        _default_store = EventStore()
    return _default_store
