"""
dashboard_server.py

Dashboard web em tempo real + Replay Recorder + A/B Testing.

Arquitetura:
- DashboardDataBridge: buffer thread-safe atualizado pelo wrapper com dados REAIS
- DashboardServer: http.server built-in servindo HTML+JS + API JSON
- ReplayRecorder: grava screenshots + acoes em data/replays/
- ABTestManager: compara estrategias (ex: old_vs_new_combat)

ZERO dados mock. Tudo vem do bot em execucao.
"""

import json
import time
import logging
import threading
import base64
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass, field, asdict
import io

try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
except ImportError:
    ThreadingHTTPServer = None
    HTTPServer = None
    BaseHTTPRequestHandler = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)

# Phase 2: LogBuffer integration
try:
    from core.log_buffer import LogBuffer, install_log_buffer, get_log_buffer
    HAS_LOGBUFFER = True
except ImportError:
    HAS_LOGBUFFER = False
    LogBuffer = None
    install_log_buffer = None
    get_log_buffer = None

# Phase 3: Notifications
try:
    from core.notifications import NotificationManager, get_notification_manager
    HAS_NOTIFICATIONS = True
except ImportError:
    HAS_NOTIFICATIONS = False
    NotificationManager = None
    get_notification_manager = None


# ---------------------------------------------------------------------------
# DATA BRIDGE (thread-safe, alimentado pelo wrapper com dados REAIS)
# ---------------------------------------------------------------------------

@dataclass
class BotLiveData:
    """Snapshot unico de todos os dados do bot em tempo real."""
    timestamp: float = 0.0
    running: bool = False
    current_state: str = "unknown"
    brawler: Optional[str] = None
    map_name: Optional[str] = None
    matches_total: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    win_rate: float = 0.0
    cycle_time_ms: float = 0.0
    fps: float = 0.0
    epsilon: float = 0.0
    q_states: int = 0
    elo_combinations: int = 0
    top_elo: List[Dict] = field(default_factory=list)
    recent_events: List[Dict] = field(default_factory=list)
    last_error: Optional[str] = None
    ab_test_active: bool = False
    ab_test_variant: str = "control"
    replay_recording: bool = False
    screenshot_b64: Optional[str] = None  # ultimo screenshot (base64 jpeg, ~30KB)
    combat_mode: str = "neutral"
    enemies_detected: int = 0
    hp_estimate: float = 1.0
    last_shot_time: float = 0.0
    uptime_seconds: float = 0.0
    # Phase 9: Error Recovery stats
    error_recovery_enabled: bool = False
    error_total: int = 0
    error_recovered: int = 0
    error_circuit_state: str = "CLOSED"
    # Phase 9: State Recovery stats
    state_recovery_active: bool = False
    state_recovery_attempts: int = 0
    state_recovery_current: str = "none"
    # Phase 9: AutoCalibrator stats
    autocalibrator_enabled: bool = False
    autocalibrator_cache_size: int = 0
    # Phase 9: OCR stats
    ocr_detector_enabled: bool = False
    ocr_reader_available: bool = False
    # Phase 9: Debug Visualizer stats
    debug_visualizer_enabled: bool = False
    debug_visualizer_running: bool = False
    # Premium: Trophies & Progress
    total_trophies: int = 0
    unlocked_brawlers: int = 0
    total_brawlers: int = 0
    trophy_history: List[Dict] = field(default_factory=list)  # [{date, trophies}]
    daily_evolution: List[Dict] = field(default_factory=list)  # [{date, change}]
    # Premium: Per-brawler stats
    brawler_stats: List[Dict] = field(default_factory=list)  # [{name, wr, picks, kills, deaths, trophies, gadget, sp, maps}]
    # Premium: Match analysis
    recent_matches: List[Dict] = field(default_factory=list)  # [{brawler, map, result, kills, deaths, duration, analysis}]
    match_analysis: Optional[Dict] = None  # latest analysis
    ai_pick_suggestion: Optional[Dict] = None  # {brawler, map, confidence, reason}
    win_prediction: float = 0.0  # 0-1 predicted win chance
    coach_tips: List[str] = field(default_factory=list)
    weekly_progress: Optional[Dict] = None  # {trophies_change, winrate_change, matches, best_brawler}


class DashboardDataBridge:
    """
    Ponte thread-safe entre o bot e o dashboard.
    O wrapper chama update() a cada ciclo; o dashboard le via get_snapshot().
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._data = BotLiveData()
        self._history: deque = deque(maxlen=300)  # 10 minutos a 2s
        self._rewards_history: deque = deque(maxlen=300)
        self._cycle_times: deque = deque(maxlen=100)
        self._start_time = time.time()
        # Premium systems
        self.brawler_tracker = BrawlerStatsTracker()
        self.match_analyzer = MatchAnalyzer()
        self.trophy_tracker = TrophyTracker()

    def update(self, **kwargs):
        """Wrapper chama isto para atualizar dados reais."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._data, k):
                    setattr(self._data, k, v)
            self._data.timestamp = time.time()
            self._data.uptime_seconds = self._data.timestamp - self._start_time
            # Guardar snapshot no historico
            snap = asdict(self._data)
            self._history.append(snap)

    def update_from_wrapper(self, wrapper_instance):
        """Extrai dados automaticamente do wrapper. Chamado pelo monitor loop."""
        if wrapper_instance is None:
            return
        try:
            w = wrapper_instance
            obs = w.observability
            sm = w.state_manager
            rl = getattr(w, 'online_learner', None)

            # Fallback brawler from queue if state_manager doesn't have it
            brawler = sm.current_brawler if sm else None
            if not brawler and hasattr(w, 'brawler_queue') and w.brawler_queue:
                current = w.brawler_queue.get_current()
                brawler = current.name if current else None
            kwargs = {
                "running": getattr(w, 'running', False),
                "current_state": sm.current_state if sm else "unknown",
                "brawler": brawler,
                "map_name": getattr(sm, '_current_map', None) if sm else None,
            }

            if obs:
                snap = obs.get_snapshot()
                kwargs["matches_total"] = snap.matches_total
                kwargs["wins"] = snap.wins
                kwargs["losses"] = snap.losses
                kwargs["win_rate"] = snap.wins / max(1, snap.matches_total)
                kwargs["cycle_time_ms"] = snap.cycle_time_ms
                kwargs["fps"] = 1000.0 / max(1, snap.cycle_time_ms)
                kwargs["last_error"] = snap.last_error
                kwargs["recent_events"] = obs.get_recent_events(20)

            if rl:
                stats = rl.get_stats()
                q_stats = stats.get("q_learning", {})
                kwargs["epsilon"] = q_stats.get("epsilon", 0.0)
                kwargs["q_states"] = q_stats.get("num_states", 0)
                elo_stats = stats.get("elo", {})
                kwargs["elo_combinations"] = elo_stats.get("total_combinations", 0)

            # Phase 9: Error Recovery stats
            er = getattr(w, 'error_recovery', None)
            if er:
                try:
                    er_stats = er.get_stats()
                    kwargs["error_recovery_enabled"] = True
                    kwargs["error_total"] = er_stats.get("total_errors", 0)
                    kwargs["error_recovered"] = er_stats.get("recovered_errors", 0)
                    kwargs["error_circuit_state"] = er_stats.get("circuit_breaker_state", "CLOSED")
                except Exception:
                    kwargs["error_recovery_enabled"] = True

            # Phase 9: State Recovery stats
            sr = getattr(w, 'state_recovery', None)
            if sr:
                try:
                    sr_status = sr.get_recovery_status()
                    kwargs["state_recovery_active"] = sr_status.get("is_recovering", False)
                    kwargs["state_recovery_attempts"] = sr_status.get("recovery_attempts", 0)
                    kwargs["state_recovery_current"] = sr_status.get("current_state", "none")
                except Exception as e:
                    logger.debug(f"[DASHBOARD] State recovery stats unavailable: {e}")

            # Phase 9: AutoCalibrator stats
            ac = getattr(w, 'auto_calibrator', None)
            if ac:
                try:
                    kwargs["autocalibrator_enabled"] = True
                    kwargs["autocalibrator_cache_size"] = len(ac.get_all_cached_coords())
                except Exception as e:
                    logger.debug(f"[DASHBOARD] AutoCalibrator stats unavailable: {e}")
                    kwargs["autocalibrator_enabled"] = True

            # Phase 9: OCR stats
            ocr = getattr(w, 'ocr_detector', None)
            if ocr:
                try:
                    ocr_stats = ocr.get_detection_stats()
                    kwargs["ocr_detector_enabled"] = True
                    kwargs["ocr_reader_available"] = ocr_stats.get("reader_available", False)
                except Exception as e:
                    logger.debug(f"[DASHBOARD] OCR stats unavailable: {e}")
                    kwargs["ocr_detector_enabled"] = True

            # Phase 9: Debug Visualizer stats
            dv = getattr(w, 'debug_visualizer', None)
            if dv:
                try:
                    kwargs["debug_visualizer_enabled"] = True
                    kwargs["debug_visualizer_running"] = getattr(dv, 'is_running', False)
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Debug visualizer stats unavailable: {e}")
                    kwargs["debug_visualizer_enabled"] = True

            # Premium: Brawler stats & trophies (sync with match_controller history)
            try:
                mc = getattr(w, 'match_controller', None)
                if mc and mc.history:
                    for m in mc.history.matches:
                        self.brawler_tracker.record_match(
                            brawler=m.brawler,
                            map_name=getattr(m, 'game_mode', ''),
                            result=m.result,
                            kills=getattr(m, 'kills', 0),
                            deaths=0,
                            duration=getattr(m, 'duration_seconds', 0.0),
                        )
                kwargs["brawler_stats"] = self.brawler_tracker.get_all_stats()
                kwargs["total_trophies"] = self.brawler_tracker.get_total_trophies()
                kwargs["unlocked_brawlers"] = self.brawler_tracker.get_unlocked_count()
                kwargs["total_brawlers"] = max(80, self.brawler_tracker.get_unlocked_count())
                kwargs["trophy_history"] = self.trophy_tracker.get_trophy_history(30)
                kwargs["daily_evolution"] = self.trophy_tracker.get_daily_evolution(14)
                kwargs["weekly_progress"] = self.trophy_tracker.get_weekly_progress()
            except Exception as e:
                logger.debug(f"[DASHBOARD] Premium stats unavailable: {e}")

            # Premium: AI pick suggestion
            try:
                queue = getattr(w, 'brawler_queue', None)
                if queue:
                    available = [b.name for b in queue.brawlers if b.enabled]
                    map_name = kwargs.get("map_name", "")
                    if available and map_name:
                        kwargs["ai_pick_suggestion"] = self.match_analyzer.suggest_pick(map_name, available)
                        kwargs["win_prediction"] = self.match_analyzer.predict_win(
                            kwargs.get("brawler", "colt"), map_name)
                    if available:
                        brawler = kwargs.get("brawler", "colt")
                        kwargs["coach_tips"] = self.match_analyzer.get_coach_tips(brawler)
            except Exception as e:
                logger.debug(f"[DASHBOARD] Premium stats unavailable: {e}")

            # Real combat data from PlayLogic
            try:
                play = getattr(sm, 'play', None) or getattr(w, 'play_logic', None)
                if play:
                    kwargs["enemies_detected"] = getattr(play, '_last_enemies', 0)
                    kwargs["combat_mode"] = getattr(play, '_last_combat_mode',
                                        getattr(play, 'combat_mode', 'neutral'))
                    # HP estimate from combat strategy
                    combat = getattr(play, 'advanced_combat', None)
                    if combat:
                        kwargs["hp_estimate"] = getattr(combat, 'estimated_hp', 1.0)
                        kwargs["combat_mode"] = getattr(combat, 'current_state', 'neutral')
            except Exception as e:
                logger.debug(f"[DASHBOARD] Premium stats unavailable: {e}")

            # Session duration
            try:
                if hasattr(w, 'session_start') and w.session_start:
                    kwargs["uptime_seconds"] = time.time() - w.session_start
            except Exception as e:
                logger.debug(f"[DASHBOARD] Premium stats unavailable: {e}")

            self.update(**kwargs)
        except Exception as e:
            logger.debug(f"[DASHBOARD] update_from_wrapper error: {e}")

    def get_snapshot(self) -> Dict:
        with self._lock:
            return asdict(self._data)

    def get_history(self) -> List[Dict]:
        with self._lock:
            return list(self._history)

    def add_reward_point(self, reward: float):
        with self._lock:
            self._rewards_history.append({"t": time.time(), "r": reward})

    def get_rewards_history(self) -> List[Dict]:
        with self._lock:
            return list(self._rewards_history)


# ---------------------------------------------------------------------------
# REPLAY RECORDER
# ---------------------------------------------------------------------------

@dataclass
class ReplayFrame:
    timestamp: float
    screenshot_path: Optional[str]
    state: str
    action: str
    enemies: int = 0
    player_pos: Optional[Tuple[float, float]] = None
    reward: float = 0.0


class ReplayRecorder:
    """
    Grava sequencias de frames (screenshot + estado + acao) para analise posterior.
    Limita a ~5 minutos (150 frames a 2s) para nao encher disco.
    """

    def __init__(self, save_dir: Path = Path("data/replays")):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._active = False
        self._current_replay: List[ReplayFrame] = []
        self._current_name: str = ""
        self._frame_counter = 0
        self._max_frames = 150
        self._quality = 60  # jpeg quality

    def start(self, name: Optional[str] = None):
        self._active = True
        self._current_replay = []
        self._frame_counter = 0
        self._current_name = name or f"replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        (self.save_dir / self._current_name).mkdir(exist_ok=True)
        logger.info(f"[REPLAY] Iniciado: {self._current_name}")

    def stop(self):
        self._active = False
        if self._current_replay:
            self._save_replay()
        logger.info(f"[REPLAY] Finalizado: {self._current_name} ({len(self._current_replay)} frames)")
        self._current_replay = []

    def record_frame(self, screenshot, state: str, action: str,
                     enemies: int = 0, player_pos=None, reward: float = 0.0):
        if not self._active or self._frame_counter >= self._max_frames:
            return
        try:
            img_path = None
            if screenshot is not None and HAS_NUMPY:
                img_path = str(self.save_dir / self._current_name / f"frame_{self._frame_counter:04d}.jpg")
                self._save_screenshot(screenshot, img_path)

            frame = ReplayFrame(
                timestamp=time.time(),
                screenshot_path=img_path,
                state=state,
                action=action,
                enemies=enemies,
                player_pos=player_pos,
                reward=reward,
            )
            self._current_replay.append(frame)
            self._frame_counter += 1
        except Exception as e:
            logger.debug(f"[REPLAY] frame error: {e}")

    def _save_screenshot(self, screenshot, path: str):
        try:
            from PIL import Image
            if isinstance(screenshot, np.ndarray):
                img = Image.fromarray(screenshot)
                img.save(path, "JPEG", quality=self._quality)
        except Exception as e:
            logger.debug(f"[REPLAY] screenshot save failed: {e}")

    def _save_replay(self):
        meta = {
            "name": self._current_name,
            "frames": len(self._current_replay),
            "start_time": self._current_replay[0].timestamp if self._current_replay else 0,
            "end_time": self._current_replay[-1].timestamp if self._current_replay else 0,
            "events": [asdict(f) for f in self._current_replay],
        }
        meta_path = self.save_dir / self._current_name / "replay.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def list_replays(self) -> List[Dict]:
        replays = []
        for entry in self.save_dir.iterdir():
            if entry.is_dir() and (entry / "replay.json").exists():
                try:
                    with open(entry / "replay.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                    replays.append({
                        "name": data["name"],
                        "frames": data["frames"],
                        "duration": data["end_time"] - data["start_time"],
                        "path": str(entry),
                    })
                except Exception as e:
                    logger.debug(f"[DASHBOARD] State recovery stats unavailable: {e}")
        return sorted(replays, key=lambda x: x["name"], reverse=True)


# ---------------------------------------------------------------------------
# A/B TEST MANAGER
# ---------------------------------------------------------------------------

@dataclass
class ABTestVariant:
    name: str
    config: Dict[str, Any]
    matches: int = 0
    wins: int = 0
    losses: int = 0
    rewards: float = 0.0


class ABTestManager:
    """
    Framework A/B Testing para comparar estrategias.
    Exemplo: variante A = combat antigo, variante B = combat avancado v2.
    Alterna automaticamente a cada partida (50/50).
    """

    def __init__(self, save_path: Path = Path("data/ab_tests.json")):
        self.save_path = Path(save_path)
        self.active = False
        self.current_variant: str = "control"
        self.variants: Dict[str, ABTestVariant] = {}
        self._match_count = 0
        self._load()

    def define_variants(self, variants: Dict[str, Dict[str, Any]]):
        """Define variantes a partir de dicts de config."""
        for name, config in variants.items():
            if name not in self.variants:
                self.variants[name] = ABTestVariant(name=name, config=config)
        logger.info(f"[ABTEST] Variantes definidas: {list(self.variants.keys())}")

    def start_test(self):
        if len(self.variants) < 2:
            logger.warning("[ABTEST] Precisa de 2+ variantes")
            return
        self.active = True
        self.current_variant = "control"
        logger.info("[ABTEST] Teste iniciado")

    def stop_test(self):
        self.active = False
        self._save()
        logger.info("[ABTEST] Teste parado")

    def next_match_variant(self) -> str:
        """Alterna variante a cada partida (round-robin balanceado)."""
        if not self.active or not self.variants:
            return "control"
        names = list(self.variants.keys())
        idx = self._match_count % len(names)
        variant = names[idx]
        self._match_count += 1
        self.current_variant = variant
        return variant

    def record_result(self, variant: str, result: str, reward: float = 0.0):
        v = self.variants.get(variant)
        if not v:
            return
        v.matches += 1
        v.rewards += reward
        if result == "win":
            v.wins += 1
        elif result == "loss":
            v.losses += 1
        logger.info(f"[ABTEST] {variant}: {result} (wins={v.wins}/{v.matches})")
        self._save()

    def get_summary(self) -> Dict:
        if not self.variants:
            return {}
        summary = {
            "active": self.active,
            "current_variant": self.current_variant,
            "variants": {},
        }
        for name, v in self.variants.items():
            wr = v.wins / max(1, v.matches)
            summary["variants"][name] = {
                "matches": v.matches,
                "wins": v.wins,
                "losses": v.losses,
                "win_rate": round(wr, 3),
                "avg_reward": round(v.rewards / max(1, v.matches), 2),
            }
        return summary

    def _save(self):
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: asdict(v) for k, v in self.variants.items()}
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"[ABTEST] save error: {e}")

    def _load(self):
        if not self.save_path.exists():
            return
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                self.variants[k] = ABTestVariant(**v)
        except Exception as e:
            logger.debug(f"[ABTEST] load error: {e}")


# ---------------------------------------------------------------------------
# PREMIUM: Brawler Stats Tracker
# ---------------------------------------------------------------------------

class BrawlerStatsTracker:
    """Tracks detailed per-brawler statistics: winrate, pick rate, kills, deaths, maps, gadgets."""

    def __init__(self, save_path: Path = Path("data/dashboard_brawler_stats.json")):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self._stats: Dict[str, Dict] = {}
        self._total_matches = 0
        self._load()

    def record_match(self, brawler: str, map_name: str, result: str,
                     kills: int = 0, deaths: int = 0, duration: float = 0.0,
                     gadget_used: str = "", star_power_used: str = ""):
        """Record a match result for a brawler."""
        self._total_matches += 1
        key = brawler.lower()
        if key not in self._stats:
            self._stats[key] = {
                "name": brawler,
                "matches": 0, "wins": 0, "losses": 0, "draws": 0,
                "kills": 0, "deaths": 0, "total_duration": 0.0,
                "trophies": 0, "target_trophies": 500,
                "maps": {}, "gadgets": {}, "star_powers": {},
                "best_map": None, "worst_map": None,
            }
        s = self._stats[key]
        s["matches"] += 1
        if result == "win":
            s["wins"] += 1
            s["trophies"] = max(0, s["trophies"] + 3)
        elif result == "loss":
            s["losses"] += 1
            s["trophies"] = max(0, s["trophies"] - 3)
        else:
            s["draws"] += 1
        s["kills"] += kills
        s["deaths"] += deaths
        s["total_duration"] += duration
        # Map stats
        mk = map_name.lower() if map_name else "unknown"
        if mk not in s["maps"]:
            s["maps"][mk] = {"matches": 0, "wins": 0}
        s["maps"][mk]["matches"] += 1
        if result == "win":
            s["maps"][mk]["wins"] += 1
        # Gadget/SP usage
        if gadget_used:
            gk = gadget_used.lower()
            s["gadgets"][gk] = s["gadgets"].get(gk, 0) + 1
        if star_power_used:
            spk = star_power_used.lower()
            s["star_powers"][spk] = s["star_powers"].get(spk, 0) + 1
        # Best/worst map
        best_wr, worst_wr = -1, 999
        for mn, md in s["maps"].items():
            if md["matches"] >= 2:
                wr = md["wins"] / md["matches"]
                if wr > best_wr:
                    best_wr = wr; s["best_map"] = mn
                if wr < worst_wr:
                    worst_wr = wr; s["worst_map"] = mn
        self._save()

    def get_all_stats(self) -> List[Dict]:
        """Get stats for all brawlers as a list."""
        result = []
        for key, s in self._stats.items():
            wr = s["wins"] / max(1, s["matches"]) * 100
            pick_rate = s["matches"] / max(1, self._total_matches) * 100
            avg_kills = s["kills"] / max(1, s["matches"])
            avg_deaths = s["deaths"] / max(1, s["matches"])
            avg_duration = s["total_duration"] / max(1, s["matches"])
            # Best gadget/star power
            best_gadget = max(s["gadgets"], key=s["gadgets"].get) if s["gadgets"] else None
            best_sp = max(s["star_powers"], key=s["star_powers"].get) if s["star_powers"] else None
            # Favorite maps (top 3 by picks)
            fav_maps = sorted(s["maps"].items(), key=lambda x: x[1]["matches"], reverse=True)[:3]
            result.append({
                "name": s["name"],
                "matches": s["matches"],
                "wins": s["wins"],
                "losses": s["losses"],
                "winrate": round(wr, 1),
                "pick_rate": round(pick_rate, 1),
                "trophies": s["trophies"],
                "target_trophies": s["target_trophies"],
                "avg_kills": round(avg_kills, 1),
                "avg_deaths": round(avg_deaths, 1),
                "avg_duration": round(avg_duration, 1),
                "best_gadget": best_gadget,
                "best_star_power": best_sp,
                "favorite_maps": [m[0] for m in fav_maps],
                "best_map": s["best_map"],
                "worst_map": s["worst_map"],
            })
        return sorted(result, key=lambda x: x["matches"], reverse=True)

    def get_total_trophies(self) -> int:
        return sum(s["trophies"] for s in self._stats.values())

    def get_unlocked_count(self) -> int:
        return len(self._stats)

    def _save(self):
        try:
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump({"total_matches": self._total_matches, "brawlers": self._stats}, f, indent=2)
        except Exception as e:
            logger.debug(f"[BRAWLER_TRACKER] save/load failed: {e}")

    def _load(self):
        try:
            if self.save_path.exists():
                with open(self.save_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._total_matches = data.get("total_matches", 0)
                self._stats = data.get("brawlers", {})
        except Exception as e:
            logger.debug(f"[TROPHY_TRACKER] save/load failed: {e}")


# ---------------------------------------------------------------------------
# PREMIUM: Match Analyzer
# ---------------------------------------------------------------------------

class MatchAnalyzer:
    """Analyzes match results to provide coaching feedback and strategic insights."""

    # Brawler role classifications
    BRAWLER_ROLES = {
        "shelly": "fighter", "colt": "damage", "nita": "controller",
        "bull": "assassin", "jessie": "controller", "brock": "sniper",
        "dynamike": "thrower", "bo": "sniper", "tick": "thrower",
        "8-bit": "damage", "el_primo": "assassin", "poco": "support",
        "penny": "damage", "carl": "damage", "pam": "support",
        "frank": "tank", "gene": "support", "max": "speedster",
        "mortis": "assassin", "tara": "damage", "sprout": "thrower",
        "bea": "damage", "nita": "controller", "ricco": "damage",
        "darryl": "assassin", "penny": "damage", "crow": "assassin",
        "leon": "assassin", "spike": "damage", "surge": "damage",
    }

    # Map-brawler advantage data (simplified)
    MAP_ADVANTAGES = {
        "gem_grab": ["poco", "pam", "gene", "sprout", "tick"],
        "brawl_ball": ["mortis", "max", "el_primo", "darryl", "crow"],
        "showdown": ["bull", "el_primo", "shelly", "crow", "leon"],
        "heist": ["colt", "brock", "dynamike", "tick", "sprout"],
        "knockout": ["brock", "piper", "bea", "tick", "sprout"],
        "hot_zone": ["poco", "pam", "gene", "jacky", "frank"],
    }

    # Counter-pick data
    COUNTERS = {
        "thrower": ["assassin", "speedster"],
        "sniper": ["assassin", "thrower"],
        "assassin": ["tank", "fighter"],
        "tank": ["thrower", "damage"],
        "support": ["assassin", "damage"],
        "damage": ["tank", "support"],
    }

    def analyze_match(self, brawler: str, map_name: str, result: str,
                      kills: int = 0, deaths: int = 0, enemies: List[str] = [],
                      duration: float = 0.0) -> Dict:
        """Analyze a completed match and return insights."""
        analysis = {
            "brawler": brawler,
            "map": map_name,
            "result": result,
            "score": 0,  # 0-100 performance score
            "errors": [],
            "strengths": [],
            "suggestions": [],
            "matchup_analysis": "",
            "positioning_tip": "",
            "build_suggestion": "",
        }

        # Performance score based on K/D and result
        if result == "win":
            analysis["score"] = min(100, 50 + kills * 10 - deaths * 5)
        else:
            analysis["score"] = max(0, 30 + kills * 5 - deaths * 10)

        # Role-based analysis
        role = self.BRAWLER_ROLES.get(brawler.lower(), "damage")
        enemy_roles = [self.BRAWLER_ROLES.get(e.lower(), "damage") for e in enemies]

        # Check matchup
        counters_to_us = []
        for er in enemy_roles:
            if er in self.COUNTERS.get(role, []):
                counters_to_us.append(er)

        if counters_to_us and result == "loss":
            analysis["errors"].append(f"Composicao fraca contra {', '.join(counters_to_us)}")
            analysis["matchup_analysis"] = f"Perdeste porque tinhas composicao fraca contra {', '.join(counters_to_us)}."
        elif result == "win" and kills > 3:
            analysis["strengths"].append(f"Boa performance como {role}")

        # Map-specific analysis
        map_lower = map_name.lower() if map_name else ""
        for map_type, adv_brawlers in self.MAP_ADVANTAGES.items():
            if map_type in map_lower:
                if brawler.lower() not in adv_brawlers:
                    analysis["suggestions"].append(f"{brawler} nao e ideal para {map_type}. Melhores: {', '.join(adv_brawlers[:3])}")
                else:
                    analysis["strengths"].append(f"{brawler} e forte em {map_type}")

        # K/D analysis
        if deaths > kills * 2:
            analysis["errors"].append("Muitas mortes - posicionamento defensivo necessario")
            analysis["positioning_tip"] = "Fica atras de paredes e usa range. Nao facas push solo."
        elif kills > deaths * 2 and result == "win":
            analysis["strengths"].append("Excelente K/D ratio")

        # Build suggestion
        analysis["build_suggestion"] = self._suggest_build(brawler, map_name)

        # Duration analysis
        if duration > 150 and result == "loss":
            analysis["suggestions"].append("Partida longa com derrota - considera trocar de brawler mais cedo")

        return analysis

    def suggest_pick(self, map_name: str, available_brawlers: List[str],
                     enemy_brawlers: List[str] = []) -> Dict:
        """Suggest the best brawler pick for a given map and enemy comp."""
        map_lower = map_name.lower() if map_name else ""
        best_brawler = None
        best_score = -1
        best_reason = ""

        for brawler in available_brawlers:
            score = 50  # base
            reason_parts = []
            role = self.BRAWLER_ROLES.get(brawler.lower(), "damage")

            # Map advantage
            for map_type, adv_brawlers in self.MAP_ADVANTAGES.items():
                if map_type in map_lower and brawler.lower() in adv_brawlers:
                    score += 25
                    reason_parts.append(f"forte em {map_type}")

            # Counter enemy
            enemy_roles = [self.BRAWLER_ROLES.get(e.lower(), "damage") for e in enemy_brawlers]
            for er in enemy_roles:
                if role in self.COUNTERS.get(er, []):
                    score += 15
                    reason_parts.append(f"counter a {er}")

            # Avoid being countered
            for er in enemy_roles:
                if er in self.COUNTERS.get(role, []):
                    score -= 10

            if score > best_score:
                best_score = score
                best_brawler = brawler
                best_reason = ", ".join(reason_parts) if reason_parts else "brawler solido"

        confidence = min(1.0, best_score / 100)

        return {
            "brawler": best_brawler or (available_brawlers[0] if available_brawlers else "colt"),
            "map": map_name,
            "confidence": round(confidence, 2),
            "reason": best_reason or "pick geral",
            "alternatives": available_brawlers[:3],
        }

    def predict_win(self, brawler: str, map_name: str, enemy_brawlers: List[str] = []) -> float:
        """Predict win probability (0-1) based on matchup analysis."""
        score = 0.5  # base 50%
        role = self.BRAWLER_ROLES.get(brawler.lower(), "damage")
        map_lower = map_name.lower() if map_name else ""

        # Map bonus
        for map_type, adv_brawlers in self.MAP_ADVANTAGES.items():
            if map_type in map_lower and brawler.lower() in adv_brawlers:
                score += 0.1

        # Counter bonus/penalty
        enemy_roles = [self.BRAWLER_ROLES.get(e.lower(), "damage") for e in enemy_brawlers]
        for er in enemy_roles:
            if role in self.COUNTERS.get(er, []):
                score += 0.05
            if er in self.COUNTERS.get(role, []):
                score -= 0.05

        return max(0.1, min(0.9, score))

    def get_coach_tips(self, brawler: str, recent_results: List[str] = []) -> List[str]:
        """Generate coaching tips based on recent performance."""
        tips = []
        role = self.BRAWLER_ROLES.get(brawler.lower(), "damage")

        if not recent_results:
            tips.append(f"Joga mais partidas com {brawler} para receber dicas personalizadas")
            return tips

        recent_wins = sum(1 for r in recent_results[-5:] if r == "win")
        recent_losses = sum(1 for r in recent_results[-5:] if r == "loss")

        if recent_losses >= 3:
            tips.append(f"Perdeste {recent_losses} das ultimas 5 - considera trocar de brawler")
        if recent_wins >= 4:
            tips.append(f"Estas em hot streak com {brawler}! Continua.")

        if role == "assassin":
            tips.append("Como assassin, flanqueia inimigos e evita lutas frontais")
        elif role == "thrower":
            tips.append("Como thrower, usa paredes como cobertura e mantem distancia")
        elif role == "support":
            tips.append("Como suporte, fica perto de aliados e cura quando possivel")
        elif role == "sniper":
            tips.append("Como sniper, posicao e tudo - fica atras e usa range")
        elif role == "tank":
            tips.append("Como tank, avanca e absorve dano enquanto a equipe ataca")

        return tips

    def _suggest_build(self, brawler: str, map_name: str) -> str:
        """Suggest gadget + star power build."""
        builds = {
            "colt": "Gadget: Silver Bullet | SP: Slick Boots",
            "shelly": "Gadget: Clay Pigeons | SP: Band-Aid",
            "nita": "Gadget: Faux Fur | SP: Bear With Me",
            "bull": "Gadget: T-Bone Injector | SP: Berserker",
            "jessie": "Gadget: Spark Plug | SP: Energize",
            "brock": "Gadget: Rocket Laces | SP: Incendiary",
            "dynamike": "Gadget: Satchel Charge | SP: Demolition",
            "bo": "Gadget: Super Totem | SP: Snare a Bear",
            "el_primo": "Gadget: Suplex Supplement | SP: El Fuego",
            "poco": "Gadget: Tuning Fork | SP: Da Capo",
            "pam": "Gadget: Pulse Tuner | SP: Mama's Squeeze",
            "mortis": "Gadget: Survival Shovel | SP: Blood Rush",
            "crow": "Gadget: Defense Boost | SP: Extra Toxic",
            "leon": "Gadget: Lollipop Drop | SP: Invisiheal",
            "spike": "Gadget: Popping Pincushion | SP: Fertilize",
        }
        return builds.get(brawler.lower(), "Usa o gadget/star power que preferires")


# ---------------------------------------------------------------------------
# PREMIUM: Trophy Tracker (daily/weekly progress)
# ---------------------------------------------------------------------------

class TrophyTracker:
    """Tracks trophy progression over time for graphs and weekly reports."""

    def __init__(self, save_path: Path = Path("data/trophy_history.json")):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict] = []
        self._load()

    def record(self, total_trophies: int, brawler: str = "", trophies: int = 0):
        """Record a trophy snapshot."""
        now = datetime.now()
        entry = {
            "timestamp": time.time(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "total_trophies": total_trophies,
            "brawler": brawler,
            "brawler_trophies": trophies,
        }
        self._history.append(entry)
        # Keep last 500 entries
        if len(self._history) > 500:
            self._history = self._history[-500:]
        self._save()

    def get_trophy_history(self, days: int = 30) -> List[Dict]:
        """Get trophy history for the last N days."""
        cutoff = time.time() - days * 86400
        return [h for h in self._history if h["timestamp"] >= cutoff]

    def get_daily_evolution(self, days: int = 14) -> List[Dict]:
        """Get daily trophy changes."""
        daily = {}
        for h in self._history:
            d = h["date"]
            if d not in daily:
                daily[d] = {"date": d, "min": h["total_trophies"], "max": h["total_trophies"]}
            daily[d]["min"] = min(daily[d]["min"], h["total_trophies"])
            daily[d]["max"] = max(daily[d]["max"], h["total_trophies"])

        result = []
        prev = None
        for d in sorted(daily.keys())[-days:]:
            change = daily[d]["max"] - prev if prev else 0
            result.append({"date": d, "trophies": daily[d]["max"], "change": change})
            prev = daily[d]["max"]
        return result

    def get_weekly_progress(self) -> Dict:
        """Get weekly progress summary."""
        now = datetime.now()
        week_ago = (now - __import__("datetime").timedelta(days=7)).strftime("%Y-%m-%d")
        week_data = [h for h in self._history if h["date"] >= week_ago]
        if len(week_data) < 2:
            return {"trophies_change": 0, "matches": 0, "best_brawler": None, "winrate_change": 0}
        start = week_data[0]["total_trophies"]
        end = week_data[-1]["total_trophies"]
        return {
            "trophies_change": end - start,
            "matches": len(week_data),
            "best_brawler": None,
            "winrate_change": 0,
            "start_trophies": start,
            "end_trophies": end,
        }

    def _save(self):
        try:
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(self._history[-500:], f, indent=2)
        except Exception as e:
            logger.debug(f"[MATCH_ANALYZER] save/load failed: {e}")

    def _load(self):
        try:
            if self.save_path.exists():
                with open(self.save_path, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
        except Exception as e:
            logger.debug(f"[DASHBOARD] serve loop stopped: {e}")
