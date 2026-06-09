"""
core/rate_limiter.py

Intelligent Rate Limiter por Account — Anti-Detecção Avançada.

Objetivo: imitar padrões humanos de jogo para evitar análise
comportamental da Supercell. O sistema NUNCA deve parecer um bot
que joga 24/7 com padrões previsíveis.

Features:
- Horários de pico humano por dia da semana
- Duração de sessão realista (30-180 min)
- Breaks aleatórios entre partidas e sessões
- Cooldown após streaks de derrotas (frustração "humana")
- Pausa para "comer", "trabalhar", "dormir"
- Jitter em todos os parâmetros temporais
"""

import random
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class AccountProfile:
    """Perfil de comportamento humano simulado para uma conta."""
    account_id: str
    created_at: float = field(default_factory=time.time)

    # Horários de pico (personalizáveis por conta)
    weekday_peaks: List[Tuple[int, int]] = field(
        default_factory=lambda: [(19, 23), (12, 13)]
    )
    weekend_peaks: List[Tuple[int, int]] = field(
        default_factory=lambda: [(14, 23)]
    )

    # Duração de sessão
    min_session_minutes: int = 30
    max_session_minutes: int = 180

    # Breaks
    break_probability_per_hour: float = 0.30
    break_duration_range: Tuple[int, int] = (5, 15)  # minutos

    # Cooldowns
    min_gap_between_sessions_minutes: int = 60
    loss_streak_break_threshold: int = 3
    loss_streak_cooldown_minutes: int = 30

    # Variação
    jitter_factor: float = 0.20  # +/- 20% em todos os tempos

    # Estado runtime
    play_history: List[Dict] = field(default_factory=list)
    current_session_start: Optional[float] = None
    loss_streak: int = 0
    win_streak: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "AccountProfile":
        # Filtrar apenas campos existentes na dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class IntelligentRateLimiter:
    """
    Rate limiting que imita padrões humanos de jogo.

    Uso:
        limiter = IntelligentRateLimiter()
        limiter.register_account("account_1")

        if limiter.should_play("account_1"):
            # Iniciar partida
            pass
        else:
            # Aguardar break
            limiter.wait_for_next_window("account_1")
    """

    def __init__(self, profiles_dir: Path = Path("data/rate_limiter")):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        self._profiles: Dict[str, AccountProfile] = {}
        self._lock = threading.RLock()

        # Carregar perfis existentes
        self._load_all_profiles()

        logger.info("[RATE_LIMITER] Inicializado com %d contas", len(self._profiles))

    # ------------------------------------------------------------------
    # Gestão de contas
    # ------------------------------------------------------------------

    def register_account(self, account_id: str, **kwargs) -> AccountProfile:
        """Registra nova conta com perfil humano."""
        with self._lock:
            if account_id not in self._profiles:
                # Criar perfil com variação aleatória para parecer único
                profile = AccountProfile(
                    account_id=account_id,
                    min_session_minutes=random.randint(20, 45),
                    max_session_minutes=random.randint(120, 240),
                    break_probability_per_hour=random.uniform(0.20, 0.40),
                    break_duration_range=(random.randint(3, 7), random.randint(10, 20)),
                    loss_streak_break_threshold=random.randint(2, 4),
                    loss_streak_cooldown_minutes=random.randint(15, 45),
                    jitter_factor=random.uniform(0.10, 0.30),
                    **kwargs
                )
                self._profiles[account_id] = profile
                self._save_profile(account_id)
                logger.info("[RATE_LIMITER] Conta registrada: %s", account_id)
            return self._profiles[account_id]

    def get_profile(self, account_id: str) -> Optional[AccountProfile]:
        """Retorna perfil de uma conta."""
        with self._lock:
            return self._profiles.get(account_id)

    # ------------------------------------------------------------------
    # Decisão: deve jogar?
    # ------------------------------------------------------------------

    def should_play(self, account_id: str) -> bool:
        """
        Verifica se é 'realista' jogar agora para esta conta.
        Retorna False se deveria estar em break/cooldown.
        """
        with self._lock:
            profile = self._profiles.get(account_id)
            if not profile:
                profile = self.register_account(account_id)

            now = datetime.now()
            now_ts = time.time()

            # 1. Verificar horários de pico
            if not self._is_peak_hour(profile, now):
                # Fora de pico: 70-90% chance de NÃO jogar
                chance = random.uniform(0.70, 0.90)
                if random.random() < chance:
                    return False

            # 2. Verificar gap entre sessões
            last_session_end = self._get_last_session_end(profile)
            if last_session_end:
                gap_minutes = (now_ts - last_session_end) / 60.0
                min_gap = profile.min_session_minutes * (1 + random.uniform(-profile.jitter_factor, profile.jitter_factor))
                if gap_minutes < min_gap:
                    return False

            # 3. Verificar duração da sessão atual
            if profile.current_session_start:
                session_duration = (now_ts - profile.current_session_start) / 60.0
                max_dur = profile.max_session_minutes * (1 + random.uniform(-profile.jitter_factor, profile.jitter_factor))
                if session_duration >= max_dur:
                    # Forçar fim de sessão
                    self._end_session(profile)
                    return False

            # 4. Verificar loss streak
            if profile.loss_streak >= profile.loss_streak_break_threshold:
                last_loss_time = self._get_last_loss_time(profile)
                if last_loss_time:
                    cooldown = profile.loss_streak_cooldown_minutes * 60
                    if now_ts - last_loss_time < cooldown:
                        return False

            # 5. Break aleatório durante sessão
            if profile.current_session_start and random.random() < profile.break_probability_per_hour / 6:
                # Dividir por 6 para não ser muito frequente
                return False

            # 6. Nunca jogar entre 2h e 6h da manhã (horário de sono)
            if 2 <= now.hour < 6:
                return random.random() > 0.95  # 95% chance de dormir

            return True

    def should_start_match(self, account_id: str) -> bool:
        """
        Verificação mais granular antes de iniciar UMA partida.
        Considera velocidade entre partidas (anti-APM-detection).
        """
        with self._lock:
            profile = self._profiles.get(account_id)
            if not profile:
                return True

            now_ts = time.time()
            last_match = self._get_last_match_time(profile)
            if last_match:
                # Tempo mínimo "realista" entre partidas:
                # - Lobby + loading: 15-45s
                # - Pós-partida: 5-15s
                # Total: 20-60s + jitter
                min_gap = random.uniform(20, 60) * (1 + random.uniform(-0.1, 0.3))
                if now_ts - last_match < min_gap:
                    return False

            return True

    def record_match_start(self, account_id: str):
        """Registra início de partida."""
        with self._lock:
            profile = self._profiles.get(account_id)
            if not profile:
                return
            if not profile.current_session_start:
                profile.current_session_start = time.time()
            profile.play_history.append({
                "type": "match_start",
                "timestamp": time.time(),
            })
            self._trim_history(profile)
            self._save_profile(account_id)

    def record_match_end(self, account_id: str, result: str, duration_seconds: float = 0):
        """Registra fim de partida e ajusta streaks."""
        with self._lock:
            profile = self._profiles.get(account_id)
            if not profile:
                return

            now = time.time()
            if result in ("win", "victory"):
                profile.win_streak += 1
                profile.loss_streak = 0
            elif result in ("loss", "defeat"):
                profile.loss_streak += 1
                profile.win_streak = 0
            else:
                # Draw ou unknown
                profile.win_streak = 0
                profile.loss_streak = 0

            profile.play_history.append({
                "type": "match_end",
                "timestamp": now,
                "result": result,
                "duration": duration_seconds,
            })
            self._trim_history(profile)
            self._save_profile(account_id)

    def record_session_end(self, account_id: str):
        """Força fim de sessão."""
        with self._lock:
            profile = self._profiles.get(account_id)
            if profile:
                self._end_session(profile)
                self._save_profile(account_id)

    def simulate_human_break(self, account_id: str) -> float:
        """
        Simula intervalo humano. Retorna duração em segundos.
        NÃO bloqueia — apenas calcula e retorna o tempo.
        """
        profile = self._profiles.get(account_id)
        if not profile:
            return random.uniform(300, 900)

        min_b, max_b = profile.break_duration_range
        jitter = random.uniform(-profile.jitter_factor, profile.jitter_factor)
        duration = random.uniform(min_b, max_b) * 60 * (1 + jitter)
        return max(60, duration)  # Mínimo 1 minuto

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _is_peak_hour(self, profile: AccountProfile, now: datetime) -> bool:
        """Verifica se está dentro de horário de pico humano."""
        hour = now.hour
        is_weekend = now.weekday() >= 5

        peaks = profile.weekend_peaks if is_weekend else profile.weekday_peaks
        for start, end in peaks:
            if start <= hour < end:
                return True
        return False

    def _get_last_session_end(self, profile: AccountProfile) -> Optional[float]:
        """Retorna timestamp do fim da última sessão."""
        for entry in reversed(profile.play_history):
            if entry.get("type") == "session_end":
                return entry["timestamp"]
        return None

    def _get_last_match_time(self, profile: AccountProfile) -> Optional[float]:
        """Retorna timestamp da última partida."""
        for entry in reversed(profile.play_history):
            if entry.get("type") in ("match_start", "match_end"):
                return entry["timestamp"]
        return None

    def _get_last_loss_time(self, profile: AccountProfile) -> Optional[float]:
        """Retorna timestamp da última derrota."""
        for entry in reversed(profile.play_history):
            if entry.get("type") == "match_end" and entry.get("result") in ("loss", "defeat"):
                return entry["timestamp"]
        return None

    def _end_session(self, profile: AccountProfile):
        """Finaliza sessão atual."""
        profile.current_session_start = None
        profile.win_streak = 0
        profile.loss_streak = 0
        profile.play_history.append({
            "type": "session_end",
            "timestamp": time.time(),
        })
        logger.info("[RATE_LIMITER] Sessão encerrada para %s", profile.account_id)

    def _trim_history(self, profile: AccountProfile, max_entries: int = 500):
        """Mantém histórico limitado."""
        if len(profile.play_history) > max_entries:
            profile.play_history = profile.play_history[-max_entries:]

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def _profile_path(self, account_id: str) -> Path:
        safe_id = "".join(c for c in account_id if c.isalnum() or c in "-_")
        return self.profiles_dir / f"{safe_id}.json"

    def _save_profile(self, account_id: str):
        """Salva perfil em JSON."""
        profile = self._profiles.get(account_id)
        if not profile:
            return
        try:
            with open(self._profile_path(account_id), "w", encoding="utf-8") as f:
                json.dump(profile.to_dict(), f, indent=2, default=str)
        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
            logger.warning("[RATE_LIMITER] Erro ao salvar perfil %s: %s", account_id, e)

    def _load_all_profiles(self):
        """Carrega todos os perfis do disco."""
        for path in self.profiles_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profile = AccountProfile.from_dict(data)
                self._profiles[profile.account_id] = profile
            except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
                logger.warning("[RATE_LIMITER] Erro ao carregar %s: %s", path.name, e)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_account_status(self, account_id: str) -> Dict:
        """Retorna status legível para uma conta."""
        profile = self._profiles.get(account_id)
        if not profile:
            return {"error": "Account not found"}

        can_play = self.should_play(account_id)
        current_session = (
            (time.time() - profile.current_session_start) / 60.0
            if profile.current_session_start else 0
        )

        return {
            "account_id": account_id,
            "should_play_now": can_play,
            "can_start_match": self.should_start_match(account_id),
            "current_session_minutes": round(current_session, 1),
            "win_streak": profile.win_streak,
            "loss_streak": profile.loss_streak,
            "total_matches_today": len([e for e in profile.play_history if e.get("type") == "match_end"]),
            "peak_hour_now": self._is_peak_hour(profile, datetime.now()),
        }

    def get_all_status(self) -> Dict[str, Dict]:
        """Retorna status de todas as contas."""
        return {aid: self.get_account_status(aid) for aid in self._profiles}
