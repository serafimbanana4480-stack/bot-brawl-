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


# ---------------------------------------------------------------------------
# HTTP SERVER + HANDLER
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):
    """Handler que serve dashboard HTML e API JSON."""

    bridge: Optional[DashboardDataBridge] = None
    recorder: Optional[ReplayRecorder] = None
    ab_test: Optional[ABTestManager] = None
    wrapper_ref: Optional[Any] = None  # Reference to wrapper for bot control
    log_buffer: Optional[Any] = None  # Phase 2: LogBuffer for real-time logs
    notification_manager: Optional[Any] = None  # Phase 3: NotificationManager

    def log_message(self, format, *args):
        pass  # Silenciar logs de acesso

    def _send_json(self, data: Dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/" or path == "/dashboard":
            self._send_html(_DASHBOARD_HTML)
        elif path == "/api/health":
            self._send_json({
                "ok": True,
                "dashboard": True,
                "bot_connected": bool(self.wrapper_ref),
                "timestamp": time.time(),
            })
        elif path == "/api/live":
            data = self.bridge.get_snapshot() if self.bridge else {}
            # Merge wrapper_ref running state when bridge is stale
            w = self.wrapper_ref
            if w and not data.get("running"):
                data["running"] = getattr(w, 'running', False)
                sm = getattr(w, 'state_manager', None)
                if sm:
                    data["current_state"] = data.get("current_state") or sm.current_state
                    brawler = getattr(sm, 'current_brawler', None)
                    if brawler:
                        data["brawler"] = data.get("brawler") or brawler
            self._send_json(data)
        elif path == "/api/history":
            data = self.bridge.get_history() if self.bridge else []
            self._send_json({"history": data})
        elif path == "/api/rewards":
            data = self.bridge.get_rewards_history() if self.bridge else []
            self._send_json({"rewards": data})
        elif path == "/api/replays":
            replays = self.recorder.list_replays() if self.recorder else []
            self._send_json({"replays": replays})
        elif path == "/api/abtest":
            summary = self.ab_test.get_summary() if self.ab_test else {}
            self._send_json(summary)
        elif path == "/api/recovery":
            # Phase 9: Detailed recovery stats
            recovery_data = {}
            if self.bridge and hasattr(self, 'server') and self.server:
                w = getattr(self.server, '_wrapper_ref', None)
                if w:
                    er = getattr(w, 'error_recovery', None)
                    if er:
                        recovery_data["error_recovery"] = er.get_stats()
                    sr = getattr(w, 'state_recovery', None)
                    if sr:
                        recovery_data["state_recovery"] = sr.get_recovery_status()
                    ac = getattr(w, 'auto_calibrator', None)
                    if ac:
                        recovery_data["auto_calibrator"] = {
                            "cache_size": len(ac.get_all_cached_coords()),
                            "templates_loaded": len(ac.templates) if hasattr(ac, 'templates') else 0,
                        }
                    ocr = getattr(w, 'ocr_detector', None)
                    if ocr:
                        recovery_data["ocr"] = ocr.get_detection_stats()
            self._send_json(recovery_data)
        # Premium API endpoints
        elif path == "/api/brawlers":
            if self.bridge:
                data = self.bridge.get_snapshot()
                self._send_json({
                    "brawlers": data.get("brawler_stats", []),
                    "total_trophies": data.get("total_trophies", 0),
                    "unlocked_brawlers": data.get("unlocked_brawlers", 0),
                    "total_brawlers": data.get("total_brawlers", 80),
                })
            else:
                self._send_json({"brawlers": [], "total_trophies": 0})
        elif path == "/api/match-analysis":
            if self.bridge:
                data = self.bridge.get_snapshot()
                self._send_json({
                    "recent_matches": data.get("recent_matches", []),
                    "match_analysis": data.get("match_analysis"),
                    "coach_tips": data.get("coach_tips", []),
                })
            else:
                self._send_json({"recent_matches": [], "coach_tips": []})
        elif path == "/api/ai-pick":
            if self.bridge:
                data = self.bridge.get_snapshot()
                self._send_json({
                    "suggestion": data.get("ai_pick_suggestion"),
                    "win_prediction": data.get("win_prediction", 0.0),
                    "current_brawler": data.get("brawler"),
                    "map": data.get("map_name"),
                })
            else:
                self._send_json({"suggestion": None, "win_prediction": 0.0})
        elif path == "/api/trophy-history":
            if self.bridge:
                data = self.bridge.get_snapshot()
                self._send_json({
                    "history": data.get("trophy_history", []),
                    "daily_evolution": data.get("daily_evolution", []),
                })
            else:
                self._send_json({"history": [], "daily_evolution": []})
        elif path == "/api/weekly-progress":
            if self.bridge:
                data = self.bridge.get_snapshot()
                self._send_json(data.get("weekly_progress") or {
                    "trophies_change": 0, "matches": 0,
                    "best_brawler": None, "winrate_change": 0,
                })
            else:
                self._send_json({"trophies_change": 0, "matches": 0})
        elif path == "/api/bot/status":
            # Bot control: get current bot status
            w = self.wrapper_ref
            if w:
                status = {
                    "running": getattr(w, 'running', False),
                    "paused": getattr(w, '_paused', False),
                    "current_state": w.state_manager.current_state if w.state_manager else "unknown",
                    "current_brawler": getattr(w.state_manager, 'current_brawler', None) if w.state_manager else None,
                    "current_map": getattr(w.state_manager, '_current_map', None) if w.state_manager else None,
                    "matches_played": getattr(w, 'matches_played', 0),
                    "brawler_queue": [b.name for b in w.brawler_queue.brawlers] if hasattr(w, 'brawler_queue') and w.brawler_queue else [],
                    "session_duration": time.time() - w.session_start if hasattr(w, 'session_start') and w.session_start else 0,
                }
                try:
                    if hasattr(w, 'safety') and w.safety and hasattr(w.safety, 'get_status'):
                        status["safety"] = w.safety.get_status()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] State recovery stats unavailable: {e}")
                try:
                    if hasattr(w, 'error_recovery') and w.error_recovery and hasattr(w.error_recovery, 'get_stats'):
                        status["error_recovery"] = w.error_recovery.get_stats()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] State recovery stats unavailable: {e}")
                # Phase 1: System status
                try:
                    if hasattr(w, 'get_system_status'):
                        status["systems"] = w.get_system_status().get("systems", {})
                except Exception as e:
                    logger.debug(f"[DASHBOARD] State recovery stats unavailable: {e}")
                self._send_json(status)
            else:
                self._send_json({"running": False, "error": "Bot not connected to dashboard"})
        elif path == "/api/system/status":
            w = self.wrapper_ref
            if w and hasattr(w, 'get_system_status'):
                self._send_json(w.get_system_status())
            else:
                # Return default system status when bot is not connected
                self._send_json({
                    "paused": False,
                    "running": False,
                    "systems": {
                        "rl_engine": {"enabled": False, "available": False},
                        "humanization": {"enabled": False, "available": False},
                        "anti_ban": {"enabled": False, "available": False},
                        "error_recovery": {"enabled": False, "available": False},
                        "recording": {"enabled": False, "available": False},
                        "auto_tuner": {"enabled": False, "available": False},
                        "data_collector": {"enabled": False, "available": False},
                    }
                })
        elif path == "/api/bot/queue":
            w = self.wrapper_ref
            if w and w.brawler_queue:
                try:
                    queue = []
                    for i, b in enumerate(w.brawler_queue.brawlers):
                        queue.append({
                            "index": i,
                            "name": b.name,
                            "current_trophies": getattr(b, 'current_trophies', 0),
                            "target_trophies": getattr(b, 'target_trophies', 350),
                            "priority": getattr(b, 'priority', 1),
                            "enabled": getattr(b, 'enabled', True),
                            "current": i == w.brawler_queue.current_index,
                        })
                    self._send_json({"queue": queue, "current_index": w.brawler_queue.current_index})
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
            else:
                self._send_json({"queue": [], "error": "Queue not available"})
        elif path == "/api/notifications/config":
            if self.notification_manager:
                self._send_json(self.notification_manager.get_config())
            else:
                self._send_json({"error": "NotificationManager not available"}, 500)
        elif path == "/api/notifications/history":
            if self.notification_manager:
                self._send_json({"history": self.notification_manager.get_history()})
            else:
                self._send_json({"history": []})
        elif path == "/api/config":
            # Phase 4: Return current config.json
            config_path = Path("config.json")
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._send_json(data)
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
            else:
                self._send_json({})
        elif path == "/api/antiban/status":
            # Phase 6: Anti-ban status from wrapper
            w = self.wrapper_ref
            if w and w.anti_ban:
                try:
                    status = w.anti_ban.get_status() if hasattr(w.anti_ban, 'get_status') else {}
                    self._send_json(status)
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
            else:
                self._send_json({"enabled": False, "error": "Anti-ban not available"})
        elif path == "/api/training/models":
            # Training & Models data: registry + dataset + last training report
            training_data = {"models": [], "dataset": {}, "last_training": None}
            try:
                # Model Registry
                from core.model_registry import ModelRegistry
                registry = ModelRegistry()
                models_list = []
                for name, versions in registry._models.items():
                    for ver_str, ver in versions.items():
                        models_list.append({
                            "name": name,
                            "version": ver_str,
                            "schema": ver.metadata.get("training_data", "—") if ver.metadata else "—",
                            "map50": ver.metrics.get("mAP50", 0.0),
                            "map50_95": ver.metrics.get("mAP50-95", 0.0),
                            "is_active": registry._active.get(name) == ver_str,
                            "created": ver.created_at,
                        })
                training_data["models"] = sorted(models_list, key=lambda x: x["created"], reverse=True)

                # Scan models/ directory if requested
                if "scan" in self.path:
                    models_dir = Path("models")
                    scanned = 0
                    if models_dir.exists():
                        for pt_file in models_dir.glob("*.pt"):
                            if pt_file.stem not in ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]:
                                try:
                                    registry.register(f"yolo_scanned", pt_file, version=None, metrics={})
                                    scanned += 1
                                except Exception:
                                    pass
                    training_data["scanned"] = scanned
            except Exception as e:
                logger.debug(f"[DASHBOARD] Training models error: {e}")

            # Dataset stats from last derived dataset
            try:
                datasets = sorted(Path("dataset").glob("roboflow_raw_v2_*"), key=lambda p: p.stat().st_mtime, reverse=True)
                if datasets:
                    ds = datasets[0]
                    total_imgs = 0
                    total_boxes = 0
                    class_dist = {}
                    for split in ("train", "val"):
                        lbl_dir = ds / split / "labels"
                        if lbl_dir.exists():
                            for lbl in lbl_dir.glob("*.txt"):
                                total_imgs += 1
                                for line in lbl.read_text(encoding="utf-8").splitlines():
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        try:
                                            cls_id = int(parts[0])
                                            total_boxes += 1
                                            class_dist[str(cls_id)] = class_dist.get(str(cls_id), 0) + 1
                                        except ValueError:
                                            pass
                    training_data["dataset"] = {
                        "total_images": total_imgs,
                        "total_boxes": total_boxes,
                        "num_classes": len(class_dist),
                        "class_distribution": dict(sorted(class_dist.items())),
                    }
            except Exception:
                pass

            # Last training report
            try:
                reports = sorted(Path("runs").glob("*/training_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if reports:
                    with open(reports[0], "r", encoding="utf-8") as f:
                        import json as _json
                        report = _json.load(f)
                    training_data["last_training"] = {
                        "run_id": report.get("run_id"),
                        "timestamp": report.get("timestamp"),
                        "schema": report.get("config", {}).get("schema"),
                        "map50": report.get("validation_metrics", {}).get("mAP50"),
                        "map50_95": report.get("validation_metrics", {}).get("mAP50-95"),
                        "duration_seconds": report.get("duration_seconds"),
                    }
            except Exception:
                pass

            self._send_json(training_data)
        elif path == "/api/export/stats":
            # Export all stats as JSON
            w = self.wrapper_ref
            export_data = {}
            if self.bridge:
                export_data["live"] = self.bridge.get_snapshot()
                export_data["history"] = self.bridge.get_history()
            if w:
                export_data["bot_status"] = w.get_status() if hasattr(w, 'get_status') else {}
                if w.brawler_queue:
                    export_data["queue"] = w.get_queue() if hasattr(w, 'get_queue') else []
            self._send_json(export_data)
        elif path == '/api/v2/status':
            w = self.wrapper_ref
            v2_data = {}
            if w and hasattr(w, 'v2_integrator') and w.v2_integrator:
                v2_data = w.v2_integrator.get_dashboard_data()
            self._send_json(v2_data)
        elif path == '/api/v2/degradation':
            w = self.wrapper_ref
            deg = {}
            if w and hasattr(w, 'v2_integrator') and w.v2_integrator and w.v2_integrator._degradation_mgr:
                deg = w.v2_integrator._degradation_mgr.get_status()
            self._send_json(deg)
        elif path == '/api/v2/alerts':
            w = self.wrapper_ref
            alerts = []
            if w and hasattr(w, 'v2_integrator') and w.v2_integrator and w.v2_integrator._alert_system:
                alerts = w.v2_integrator._alert_system.get_active_alerts()
            self._send_json({'alerts': alerts})
        elif path == '/api/v2/rate-limiter':
            w = self.wrapper_ref
            rl = {}
            if w and hasattr(w, 'v2_integrator') and w.v2_integrator and w.v2_integrator._rate_limiter:
                rl = w.v2_integrator._rate_limiter.get_account_status(w.v2_integrator.config.account_id)
            self._send_json(rl)
        elif path == '/api/v2/checkpoints':
            w = self.wrapper_ref
            cp = {}
            if w and hasattr(w, 'v2_integrator') and w.v2_integrator and w.v2_integrator._checkpointer:
                cp = w.v2_integrator._checkpointer.get_stats()
            self._send_json(cp)
        # Learning Mode endpoints
        elif path == '/api/learning-mode/status':
            w = self.wrapper_ref
            data = {"active": False, "metrics": {}}
            if w and hasattr(w, 'learning_mode_controller') and w.learning_mode_controller:
                try:
                    data = w.learning_mode_controller.get_live_metrics()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Learning mode metrics error: {e}")
            self._send_json(data)
        elif path == '/api/learning-mode/history':
            w = self.wrapper_ref
            history = []
            if w and hasattr(w, 'learning_mode_controller') and w.learning_mode_controller and w.learning_mode_controller.metrics:
                try:
                    history = w.learning_mode_controller.metrics.get_session_history()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Learning mode history error: {e}")
            self._send_json({"matches": history})
        elif path == '/api/esp/frame':
            # Retorna screenshot anotado + detecoes JSON
            w = self.wrapper_ref
            data = {"detections": [], "vision_stats": {}, "screenshot_b64": ""}
            if w and hasattr(w, 'get_detection_snapshot'):
                try:
                    data = w.get_detection_snapshot()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] ESP frame error: {e}")
            self._send_json(data)
        elif path == '/api/vision/stats':
            w = self.wrapper_ref
            stats = {}
            if w and hasattr(w, 'get_detection_snapshot'):
                try:
                    stats = w.get_detection_snapshot().get("vision_stats", {})
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Vision stats error: {e}")
            self._send_json(stats)
        elif path == '/api/detections/live':
            w = self.wrapper_ref
            detections = []
            if w and hasattr(w, 'get_detection_snapshot'):
                try:
                    detections = w.get_detection_snapshot().get("detections", [])
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Detections live error: {e}")
            self._send_json({"detections": detections, "timestamp": time.time()})
        elif path == '/api/mode/status':
            w = self.wrapper_ref
            status = {"available": False}
            if w and hasattr(w, 'get_mode_status'):
                try:
                    status = w.get_mode_status()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Mode status error: {e}")
            self._send_json(status)
        elif path == '/api/rl/metrics':
            w = self.wrapper_ref
            metrics = {}
            if w and hasattr(w, 'get_rl_metrics'):
                try:
                    metrics = w.get_rl_metrics()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] RL metrics error: {e}")
            self._send_json(metrics)
        elif path == '/api/logs':
            # Phase 2: Log viewer endpoint
            if self.log_buffer:
                import urllib.parse
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                limit = int(query.get("limit", ["100"])[0])
                level = query.get("level", ["ALL"])[0]
                component = query.get("component", ["ALL"])[0]
                search = query.get("search", [""])[0]
                lines = self.log_buffer.get_lines(limit=limit, level=level if level != "ALL" else None,
                                                  component=component if component != "ALL" else None,
                                                  search=search if search else None)
                self._send_json({"lines": lines, "stats": self.log_buffer.get_stats()})
            else:
                self._send_json({"lines": [], "stats": {"total": 0}, "error": "LogBuffer not available"})
        elif path == "/api/logs/stream":
            # Phase 2: Server-Sent Events for real-time logs
            if self.log_buffer:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                import threading
                listener = threading.Event()
                self.log_buffer.add_listener(listener)
                last_count = len(self.log_buffer.get_lines(limit=1))
                try:
                    while True:
                        # FIX #12: Add safety timeout to prevent infinite loop
                        if listener.wait(timeout=5.0):
                            listener.clear()
                        else:
                            # Timeout - check if server is shutting down
                            if hasattr(self.server, '_shutdown') and self.server._shutdown:
                                break
                            continue
                        lines = self.log_buffer.get_lines(limit=50)
                        if lines:
                            data = json.dumps({"lines": lines}, ensure_ascii=False)
                            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    logger.debug('[DASHBOARD] SSE client disconnected')
                except Exception as e:
                    logger.debug(f"[DASHBOARD] SSE stream error: {e}")
                finally:
                    self.log_buffer.remove_listener(listener)
            else:
                self._send_json({"error": "LogBuffer not available"}, 500)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}

        if path == "/api/abtest/start":
            variants = payload.get("variants", {"control": {}, "test": {}})
            if self.ab_test:
                self.ab_test.define_variants(variants)
                self.ab_test.start_test()
            self._send_json({"ok": True, "status": "started"})
        elif path == "/api/abtest/stop":
            if self.ab_test:
                self.ab_test.stop_test()
            self._send_json({"ok": True, "status": "stopped"})
        elif path == "/api/replay/start":
            name = payload.get("name")
            if self.recorder:
                self.recorder.start(name)
            self._send_json({"ok": True})
        elif path == "/api/replay/stop":
            if self.recorder:
                self.recorder.stop()
            self._send_json({"ok": True})
        # Premium: Record match result
        elif path == "/api/match/record":
            if self.bridge:
                brawler = payload.get("brawler", "colt")
                map_name = payload.get("map", "unknown")
                result = payload.get("result", "loss")
                kills = payload.get("kills", 0)
                deaths = payload.get("deaths", 0)
                duration = payload.get("duration", 0.0)
                self.bridge.brawler_tracker.record_match(
                    brawler, map_name, result, kills, deaths, duration,
                    payload.get("gadget", ""), payload.get("star_power", "")
                )
                # Record trophy snapshot
                total = self.bridge.brawler_tracker.get_total_trophies()
                self.bridge.trophy_tracker.record(total, brawler, 0)
                # Run match analysis
                analysis = self.bridge.match_analyzer.analyze_match(
                    brawler, map_name, result, kills, deaths,
                    payload.get("enemies", []), duration
                )
                # Update live data
                self.bridge.update(
                    recent_matches=self.bridge.match_analyzer.analyze_match(
                        brawler, map_name, result, kills, deaths,
                        payload.get("enemies", []), duration
                    ),
                    match_analysis=analysis,
                )
                self._send_json({"ok": True, "analysis": analysis})
            else:
                self._send_json({"error": "no bridge"}, 500)
        # Bot control endpoints
        elif path == "/api/bot/start":
            w = self.wrapper_ref
            if w:
                try:
                    if not getattr(w, 'running', False):
                        setup_ok = True
                        if hasattr(w, 'setup'):
                            setup_ok = w.setup()
                        if not setup_ok:
                            self._send_json({"ok": False, "error": "Setup failed - verify emulator, window focus and config"})
                            return
                        started = w.start()
                        if not started:
                            self._send_json({"ok": False, "error": "Failed to start bot - safety, anti-ban or runtime block"})
                            return
                        self._send_json({"ok": True, "status": "started"})
                    else:
                        self._send_json({"ok": True, "status": "already_running"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/stop":
            w = self.wrapper_ref
            if w:
                try:
                    if getattr(w, 'running', False):
                        w.stop()
                        self._send_json({"ok": True, "status": "stopped"})
                    else:
                        self._send_json({"ok": True, "status": "already_stopped"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/restart":
            w = self.wrapper_ref
            if w:
                try:
                    if getattr(w, 'running', False):
                        w.stop()
                        time.sleep(2)
                    w.start()
                    self._send_json({"ok": True, "status": "restarted"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/farm/start":
            w = self.wrapper_ref
            if w and hasattr(w, '_run_farm_loop'):
                try:
                    farm_cfg = payload.get("config", {})
                    if farm_cfg:
                        for bcfg in farm_cfg.get("brawlers", []):
                            w.brawler_queue.add_brawler(BrawlerConfig(
                                name=bcfg.get("name", "colt"),
                                target_trophies=bcfg.get("target_trophies", 500),
                                priority=bcfg.get("priority", 1),
                                enabled=bcfg.get("enabled", True),
                            ))
                    if not getattr(w, 'running', False):
                        w.start()
                    self._send_json({"ok": True, "status": "farm_started"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected or farm mode unavailable"})
        # Phase 1: Bot control endpoints
        elif path == "/api/bot/pause":
            w = self.wrapper_ref
            if w:
                try:
                    result = w.pause()
                    self._send_json({"ok": result, "status": "paused" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/resume":
            w = self.wrapper_ref
            if w:
                try:
                    result = w.resume()
                    self._send_json({"ok": result, "status": "resumed" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/queue/update":
            w = self.wrapper_ref
            if w:
                try:
                    queue_data = payload.get("queue", [])
                    result = w.update_queue(queue_data)
                    self._send_json({"ok": result, "queue": [b.name for b in w.brawler_queue.brawlers] if w.brawler_queue else []})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/queue/set-brawler":
            w = self.wrapper_ref
            if w:
                try:
                    name = payload.get("name", "")
                    result = w.set_brawler(name)
                    self._send_json({"ok": result, "current_brawler": w.brawler_queue.get_current().name if w.brawler_queue and w.brawler_queue.get_current() else None})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/queue/add":
            w = self.wrapper_ref
            if w:
                try:
                    name = payload.get("name", "colt")
                    current = payload.get("current_trophies", 0)
                    target = payload.get("target_trophies", 350)
                    priority = payload.get("priority", 1)
                    w.add_brawler_to_queue(name, current, target, priority=priority)
                    self._send_json({"ok": True, "queue": [b.name for b in w.brawler_queue.brawlers] if w.brawler_queue else []})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/queue/remove":
            w = self.wrapper_ref
            if w and w.brawler_queue:
                try:
                    index = payload.get("index", -1)
                    if 0 <= index < len(w.brawler_queue.brawlers):
                        removed = w.brawler_queue.brawlers.pop(index)
                        self._send_json({"ok": True, "removed": removed.name, "queue": [b.name for b in w.brawler_queue.brawlers]})
                    else:
                        self._send_json({"ok": False, "error": "Invalid index"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/config/update":
            w = self.wrapper_ref
            if w:
                try:
                    key = payload.get("key", "")
                    value = payload.get("value")
                    if key:
                        result = w.update_config(key, value)
                        self._send_json({"ok": result})
                    else:
                        self._send_json({"ok": False, "error": "Missing key"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/action":
            w = self.wrapper_ref
            if w:
                try:
                    action = payload.get("action", "")
                    if action:
                        result = w.execute_action(action, **payload.get("params", {}))
                        self._send_json({"ok": result, "action": action})
                    else:
                        self._send_json({"ok": False, "error": "Missing action"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/bot/combat/param":
            w = self.wrapper_ref
            if w and w.play_logic:
                try:
                    param = payload.get("param", "")
                    value = payload.get("value")
                    if param == "aggressiveness":
                        w.play_logic.aggressiveness = float(value)
                        self._send_json({"ok": True, "param": param, "value": w.play_logic.aggressiveness})
                    elif param == "shot_cooldown":
                        w.play_logic.shot_cooldown = float(value) / 1000.0
                        self._send_json({"ok": True, "param": param, "value": w.play_logic.shot_cooldown})
                    elif param == "attack_distance":
                        w.play_logic.attack_distance = int(value)
                        self._send_json({"ok": True, "param": param, "value": w.play_logic.attack_distance})
                    else:
                        self._send_json({"ok": False, "error": f"Unknown param: {param}"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "PlayLogic not available"})
        elif path == "/api/system/toggle":
            w = self.wrapper_ref
            if w:
                try:
                    system = payload.get("system", "")
                    enabled = payload.get("enabled", True)
                    if system:
                        result = w.toggle_system(system, enabled)
                        self._send_json({"ok": result, "system": system, "enabled": enabled})
                    else:
                        self._send_json({"ok": False, "error": "Missing system name"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)})
            else:
                self._send_json({"ok": False, "error": "Bot not connected"})
        elif path == "/api/notifications/config":
            if self.notification_manager:
                try:
                    self.notification_manager.update_config(**payload)
                    self._send_json({"ok": True, "config": self.notification_manager.get_config()})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "NotificationManager not available"}, 500)
        elif path == "/api/notifications/test":
            if self.notification_manager:
                try:
                    title = payload.get("title", "Teste")
                    message = payload.get("message", "Notificacao de teste do Soberana Omega")
                    self.notification_manager.send(title, message, "info", "test")
                    self._send_json({"ok": True})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "NotificationManager not available"}, 500)
        elif path == "/api/learning-mode/start":
            w = self.wrapper_ref
            if w and hasattr(w, 'toggle_learning_mode'):
                try:
                    max_matches = payload.get("max_matches", 5)
                    result = w.toggle_learning_mode(enabled=True, max_matches=max_matches)
                    self._send_json({"ok": result, "status": "started" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Bot not connected or learning mode unavailable"}, 500)
        elif path == "/api/learning-mode/stop":
            w = self.wrapper_ref
            if w and hasattr(w, 'toggle_learning_mode'):
                try:
                    result = w.toggle_learning_mode(enabled=False)
                    self._send_json({"ok": result, "status": "stopped" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Bot not connected or learning mode unavailable"}, 500)
        elif path == "/api/esp/toggle":
            w = self.wrapper_ref
            if w and hasattr(w, 'toggle_esp'):
                try:
                    enabled = payload.get("enabled", None)
                    result = w.toggle_esp(enabled)
                    self._send_json({"ok": result, "status": "on" if result else "off"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "ESP not available"}, 500)
        elif path == "/api/mode/training/start":
            w = self.wrapper_ref
            if w and hasattr(w, 'start_mode'):
                try:
                    config = payload.get("config", {})
                    result = w.start_mode("training", config)
                    self._send_json({"ok": result, "status": "started" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Mode controller not available"}, 500)
        elif path == "/api/mode/training/stop":
            w = self.wrapper_ref
            if w and hasattr(w, 'stop_mode'):
                try:
                    result = w.stop_mode("training")
                    self._send_json({"ok": result, "status": "stopped" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Mode controller not available"}, 500)
        elif path == "/api/mode/farm/start":
            w = self.wrapper_ref
            if w and hasattr(w, 'start_mode'):
                try:
                    config = payload.get("config", {})
                    result = w.start_mode("farm", config)
                    self._send_json({"ok": result, "status": "started" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Mode controller not available"}, 500)
        elif path == "/api/mode/farm/stop":
            w = self.wrapper_ref
            if w and hasattr(w, 'stop_mode'):
                try:
                    result = w.stop_mode("farm")
                    self._send_json({"ok": result, "status": "stopped" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Mode controller not available"}, 500)
        elif path == "/api/mode/learn/start":
            w = self.wrapper_ref
            if w and hasattr(w, 'start_mode'):
                try:
                    config = payload.get("config", {})
                    result = w.start_mode("learn", config)
                    self._send_json({"ok": result, "status": "started" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Mode controller not available"}, 500)
        elif path == "/api/mode/learn/stop":
            w = self.wrapper_ref
            if w and hasattr(w, 'stop_mode'):
                try:
                    result = w.stop_mode("learn")
                    self._send_json({"ok": result, "status": "stopped" if result else "failed"})
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, 500)
            else:
                self._send_json({"ok": False, "error": "Mode controller not available"}, 500)
        elif path == "/api/rl/config":
            # Atualiza hiperparametros RL (placeholder — engine ajusta dinamicamente)
            self._send_json({"ok": True, "message": "RL config update not yet implemented"})
        elif path == "/api/config":
            # Phase 4: Save config.json
            try:
                config_path = Path("config.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                # Also update wrapper central_config
                w = self.wrapper_ref
                if w:
                    w.central_config.update(payload)
                self._send_json({"ok": True})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
        else:
            self._send_json({"error": "not found"}, 404)


# ---------------------------------------------------------------------------
# DASHBOARD HTML (embedded — sem ficheiros externos)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Soberana Omega — Dashboard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.5}
  header{background:#1e293b;padding:1rem 1.5rem;border-bottom:1px solid #334155;display:flex;align-items:center;justify-content:space-between}
  header h1{font-size:1.25rem;color:#38bdf8}
  .status-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px}
  .status-online{background:#22c55e}
  .status-offline{background:#ef4444}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;padding:1rem}
  .card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:1rem}
  .card h2{font-size:.9rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.75rem}
  .metric{font-size:2rem;font-weight:700;color:#38bdf8}
  .metric.small{font-size:1.2rem}
  .label{font-size:.75rem;color:#64748b}
  .row{display:flex;justify-content:space-between;align-items:center;padding:.35rem 0;border-bottom:1px solid #334155}
  .row:last-child{border:none}
  .progress{height:6px;background:#334155;border-radius:3px;overflow:hidden;margin-top:4px}
  .progress-bar{height:100%;background:#38bdf8;border-radius:3px;transition:width .5s}
  .progress-bar.win{background:#22c55e}
  .progress-bar.loss{background:#ef4444}
  .event-log{max-height:220px;overflow-y:auto;font-family:monospace;font-size:.78rem}
  .event-log .event{padding:.2rem 0;border-bottom:1px dashed #334155}
  .event-log .time{color:#64748b}
  .btn{background:#2563eb;color:#fff;border:none;border-radius:4px;padding:.4rem .8rem;font-size:.8rem;cursor:pointer;margin-right:.4rem}
  .btn:hover{background:#1d4ed8}
  .btn.warn{background:#d97706}
  .btn.danger{background:#dc2626}
  table{width:100%;font-size:.8rem;border-collapse:collapse}
  th,td{text-align:left;padding:.4rem;border-bottom:1px solid #334155}
  th{color:#94a3b8}
  .screenshot{max-width:100%;border-radius:4px;border:1px solid #334155;margin-top:.5rem}
  .tabs{display:flex;gap:.5rem;margin-bottom:1rem;flex-wrap:wrap}
  .tab{cursor:pointer;padding:.4rem .8rem;border-radius:4px;font-size:.8rem;background:#334155}
  .tab.active{background:#2563eb}
  .tab.premium{background:#7c3aed}
  .tab.premium.active{background:#9333ea}
  .tab-content{display:none}
  .tab-content.active{display:block}
  canvas{max-width:100%}
  .badge{display:inline-block;padding:.1rem .4rem;border-radius:3px;font-size:.65rem;font-weight:700;margin-left:.3rem}
  .badge.gold{background:#f59e0b;color:#000}
  /* Phase 1: Toggle Switch */
  .toggle{display:inline-block;position:relative;width:40px;height:20px}
  .toggle input{opacity:0;width:0;height:0}
  .toggle .slider{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background:#334155;border-radius:20px;transition:.3s}
  .toggle .slider:before{position:absolute;content:'';height:14px;width:14px;left:3px;bottom:3px;background:#fff;border-radius:50%;transition:.3s}
  .toggle input:checked+.slider{background:#22c55e}
  .toggle input:checked+.slider:before{transform:translateX(20px)}
  /* Phase 1: Range Slider */
  input[type=range]{width:100%;margin:.5rem 0}
  /* Phase 1: Select */
  select{width:100%;padding:.4rem;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;font-size:.8rem}
  /* Phase 1: Control Panel */
  .control-row{display:flex;align-items:center;justify-content:space-between;padding:.4rem 0;border-bottom:1px solid #334155}
  .control-row:last-child{border:none}
  .control-row .label{flex:1}
  .control-row .control{flex-shrink:0;margin-left:.5rem}
  .queue-item{display:flex;align-items:center;gap:.5rem;padding:.3rem;background:#0f172a;border-radius:4px;margin-bottom:.3rem;font-size:.8rem}
  .queue-item .name{flex:1;font-weight:600;color:#38bdf8}
  .queue-item .btn-sm{padding:.2rem .4rem;font-size:.7rem;margin:0}
  .btn-sm{background:#2563eb;color:#fff;border:none;border-radius:4px;padding:.25rem .5rem;font-size:.75rem;cursor:pointer}
  .btn-sm.danger{background:#dc2626}
  .badge.silver{background:#94a3b8;color:#000}
  .badge.green{background:#22c55e;color:#000}
  .badge.red{background:#ef4444;color:#fff}
  .badge.purple{background:#7c3aed;color:#fff}
  .brawler-card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:.75rem;margin-bottom:.5rem}
  .brawler-card .name{font-size:1rem;font-weight:700;color:#38bdf8}
  .brawler-card .stats{display:flex;gap:1rem;margin-top:.3rem;font-size:.75rem;color:#94a3b8}
  .brawler-card .stats span{display:flex;flex-direction:column;align-items:center}
  .brawler-card .stats .val{font-size:1rem;font-weight:700;color:#e2e8f0}
  .esports-bar{background:linear-gradient(90deg,#7c3aed,#2563eb);padding:.5rem 1rem;border-radius:4px;display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem}
  .esports-bar .label{font-size:.7rem;color:#e2e8f0;text-transform:uppercase;letter-spacing:.1em}
  .esports-bar .value{font-size:1.1rem;font-weight:700;color:#fff}
  .coach-tip{background:#1e293b;border-left:3px solid #7c3aed;padding:.5rem .75rem;margin-bottom:.4rem;font-size:.8rem;border-radius:0 4px 4px 0}
  .analysis-score{font-size:2.5rem;font-weight:800;text-align:center}
  .analysis-score.high{color:#22c55e}
  .analysis-score.mid{color:#f59e0b}
  .analysis-score.low{color:#ef4444}
  .win-prediction-bar{height:24px;background:#334155;border-radius:12px;overflow:hidden;position:relative}
  .win-prediction-bar .fill{height:100%;border-radius:12px;transition:width .5s}
  .win-prediction-bar .text{position:absolute;top:0;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;font-size:.7rem;font-weight:700;color:#fff}
  .trophy-chart{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:1rem}
  .weekly-card{display:flex;gap:1rem;flex-wrap:wrap}
  .weekly-card .stat{text-align:center;padding:.5rem 1rem;background:#1e293b;border:1px solid #334155;border-radius:8px;flex:1;min-width:120px}
  .weekly-card .stat .num{font-size:1.5rem;font-weight:700}
  .weekly-card .stat .lbl{font-size:.7rem;color:#94a3b8;text-transform:uppercase}
  /* Health Monitor */
  .health-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px}
  .health-online{background:#22c55e}
  .health-warn{background:#f59e0b}
  .health-offline{background:#ef4444}
  .health-item{display:flex;align-items:center;padding:.25rem 0;font-size:.8rem}
  /* Mobile */
  @media(max-width:600px){
    .tabs{flex-wrap:wrap;gap:.3rem}
    .tab{padding:.3rem .5rem;font-size:.7rem}
    .grid{padding:.5rem;gap:.5rem}
    .card{padding:.75rem}
    .metric{font-size:1.5rem}
  }
  /* Toast */
  @keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
  /* Start button pulse animation */
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(34,197,94,.6)}70%{box-shadow:0 0 0 10px rgba(34,197,94,0)}100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}}
  .btn-start{animation:pulse 2s infinite}
  .btn-start.disabled{opacity:.5;cursor:not-allowed;animation:none}
  .btn-start.running{background:#64748b;animation:none}
  .btn.disabled{opacity:.5;cursor:not-allowed;pointer-events:none}
</style>
</head>
<body>
<header>
  <h1>Soberana Omega <span style="font-size:.75rem;color:#64748b">Dashboard</span></h1>
  <div style="display:flex;align-items:center;gap:1rem">
    <span class="btn btn-sm" onclick="exportStats()" title="Exportar estatisticas">Exportar</span>
    <span class="btn btn-sm" onclick="toggleDarkMode()" id="darkModeBtn" title="Alternar tema">Tema</span>
    <div id="connStatus"><span class="status-dot status-offline"></span>Offline</div>
  </div>
</header>

<div class="tabs" style="padding:1rem 1rem 0">
  <div class="tab active" onclick="showTab('live')">Tempo Real</div>
  <div class="tab" onclick="showTab('history')">Historico</div>
  <div class="tab" onclick="showTab('replays')">Replays</div>
  <div class="tab" onclick="showTab('abtest')">A/B Test</div>
  <div class="tab" onclick="showTab('recovery')">Recovery</div>
  <div class="tab premium" onclick="showTab('brawlers')">Brawlers</div>
  <div class="tab premium" onclick="showTab('analysis')">Match Analyzer</div>
  <div class="tab premium" onclick="showTab('aicoach')">AI Coach</div>
  <div class="tab premium" onclick="showTab('trophies')">Trophies</div>
  <div class="tab premium" onclick="showTab('esports')">Esports</div>
  <div class="tab" onclick="showTab('logs')">Logs <span class="badge green">NEW</span></div>
  <div class="tab" onclick="showTab('notifications')">Alertas <span class="badge green">NEW</span></div>
  <div class="tab" onclick="showTab('config')">Config <span class="badge green">NEW</span></div>
  <div class="tab" onclick="showTab('antiban')">Anti-Ban <span class="badge green">NEW</span></div>
  <div class="tab" onclick="showTab('analytics')">Analytics <span class="badge green">NEW</span></div>
  <div class="tab" onclick="showTab('learning')">Modo Teste <span class="badge green">LIVE</span></div>
  <div class="tab" onclick="showTab('farm')">Executar Bot <span class="badge green">LIVE</span></div>
  <div class="tab" onclick="showTab('learn')">Modo Aprender <span class="badge purple">AI</span></div>
  <div class="tab" onclick="showTab('detections')">Visao <span class="badge green">ESP</span></div>
  <div class="tab premium" onclick="showTab('training')">Training <span class="badge gold">PRO</span></div>
</div>

<div id="tab-live" class="tab-content active">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Controlo do Bot <span class="badge green">NEW</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap">
      <div style="flex:1;min-width:200px">
        <div class="metric small" id="stateVal">—</div>
        <div class="label" id="brawlerVal">—</div>
        <div style="margin-top:.5rem">
          <span class="btn btn-start" id="startBtn" style="background:#22c55e" onclick="botStart()">Iniciar</span>
          <span class="btn danger disabled" id="stopBtn" onclick="botStop()">Parar</span>
          <span class="btn warn disabled" id="restartBtn" onclick="botRestart()">Reiniciar</span>
          <span class="btn disabled" id="pauseBtn" style="background:#f59e0b" onclick="botPauseToggle()">Pausar</span>
        </div>
        <div style="margin-top:.3rem">
          <span class="btn" onclick="fetch('/api/replay/start',{method:'POST',body:'{}'})">Gravar Replay</span>
          <span class="btn warn" onclick="fetch('/api/replay/stop',{method:'POST',body:'{}'})">Parar Replay</span>
          <span class="btn" style="background:#7c3aed" onclick="botAction('screenshot')">Screenshot</span>
        </div>
      </div>
      <div style="flex:1;min-width:200px">
        <div class="label" style="margin-bottom:.3rem">Selecionar Brawler</div>
        <select id="brawlerSelect" onchange="setBrawler(this.value)">
          <option value="">— Escolher —</option>
        </select>
        <div style="margin-top:.5rem">
          <span class="btn btn-sm" onclick="botAction('force_click_play')">Play</span>
          <span class="btn btn-sm" onclick="botAction('force_attack')">Atacar</span>
          <span class="btn btn-sm" onclick="botAction('force_super')">Super</span>
          <span class="btn btn-sm" onclick="botAction('force_goto_lobby')">Lobby</span>
          <span class="btn btn-sm danger" onclick="botAction('back_press')">Back</span>
        </div>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Sistemas <span class="badge green">NEW</span></h2>
    <div class="control-row">
      <span class="label">RL Engine</span>
      <label class="toggle control"><input type="checkbox" id="sysRL" onchange="toggleSystem('rl_engine',this.checked)"><span class="slider"></span></label>
    </div>
    <div class="control-row">
      <span class="label">Humanizacao</span>
      <label class="toggle control"><input type="checkbox" id="sysHuman" onchange="toggleSystem('humanization',this.checked)"><span class="slider"></span></label>
    </div>
    <div class="control-row">
      <span class="label">Anti-Ban</span>
      <label class="toggle control"><input type="checkbox" id="sysAntiBan" onchange="toggleSystem('anti_ban',this.checked)"><span class="slider"></span></label>
    </div>
    <div class="control-row">
      <span class="label">Error Recovery</span>
      <label class="toggle control"><input type="checkbox" id="sysErrRec" onchange="toggleSystem('error_recovery',this.checked)"><span class="slider"></span></label>
    </div>
    <div class="control-row">
      <span class="label">Recording</span>
      <label class="toggle control"><input type="checkbox" id="sysRec" onchange="toggleSystem('recording',this.checked)"><span class="slider"></span></label>
    </div>
    <div class="control-row">
      <span class="label">Auto-Tuner</span>
      <label class="toggle control"><input type="checkbox" id="sysTuner" onchange="toggleSystem('auto_tuner',this.checked)"><span class="slider"></span></label>
    </div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Fila de Brawlers <span class="badge green">NEW</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap">
      <div style="flex:2;min-width:280px">
        <div id="brawlerQueueList"><div class="label">A carregar fila...</div></div>
      </div>
      <div style="flex:1;min-width:200px">
        <h3 style="font-size:.85rem;color:#94a3b8;margin-bottom:.5rem">Adicionar Brawler</h3>
        <input id="newBrawlerName" type="text" placeholder="Nome do brawler" style="width:100%;padding:.4rem;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;font-size:.8rem;margin-bottom:.3rem">
        <input id="newBrawlerTarget" type="number" placeholder="Trofeus alvo" value="350" style="width:100%;padding:.4rem;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;font-size:.8rem;margin-bottom:.3rem">
        <input id="newBrawlerPriority" type="number" placeholder="Prioridade (1-5)" value="1" min="1" max="5" style="width:100%;padding:.4rem;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;font-size:.8rem;margin-bottom:.3rem">
        <span class="btn btn-sm" onclick="addBrawlerToQueue()">Adicionar</span>
        <span class="btn btn-sm warn" onclick="clearQueue()">Limpar Fila</span>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Health Monitor <span class="badge green">NEW</span></h2>
    <div id="healthMonitor">
      <div class="health-item"><span class="health-dot health-online"></span>YOLO Modelo: <span id="healthYOLO">Carregando</span></div>
      <div class="health-item"><span class="health-dot health-online"></span>ADB Conexao: <span id="healthADB">Carregando</span></div>
      <div class="health-item"><span class="health-dot health-online"></span>OCR: <span id="healthOCR">Carregando</span></div>
      <div class="health-item"><span class="health-dot health-online"></span>State Manager: <span id="healthState">Carregando</span></div>
      <div class="health-item"><span class="health-dot health-online"></span>RL Engine: <span id="healthRL">Carregando</span></div>
      <div class="health-item"><span class="health-dot health-online"></span>Anti-Ban: <span id="healthAntiBan">Carregando</span></div>
      <div class="health-item"><span class="health-dot health-online"></span>Emulator: <span id="healthEmulator">Carregando</span></div>
    </div>
    <div class="label" style="margin-top:.5rem">Ultima acao: <span id="healthLastAction">—</span></div>
  </div>
  <div class="card">
    <h2>Parametros de Combate <span class="badge green">NEW</span></h2>
    <div class="control-row">
      <span class="label">Aggressiveness</span>
      <span id="aggVal" class="metric small" style="width:50px;text-align:right">50%</span>
    </div>
    <input type="range" id="aggSlider" min="0" max="100" value="50" onchange="updateCombatParam('aggressiveness',this.value/100)">
    <div class="control-row">
      <span class="label">Shot Cooldown (ms)</span>
      <span id="cdVal" class="metric small" style="width:50px;text-align:right">350</span>
    </div>
    <input type="range" id="cdSlider" min="200" max="600" value="350" onchange="updateCombatParam('shot_cooldown',this.value)">
    <div class="control-row">
      <span class="label">Attack Distance</span>
      <span id="distVal" class="metric small" style="width:50px;text-align:right">200</span>
    </div>
    <input type="range" id="distSlider" min="100" max="400" value="200" onchange="updateCombatParam('attack_distance',this.value)">
  </div>
  <div class="card">
    <h2>Partidas</h2>
    <div class="metric" id="matchesVal">0</div>
    <div class="label">Win Rate: <span id="wrVal">0%</span></div>
    <div class="progress"><div class="progress-bar win" id="wrBar" style="width:0%"></div></div>
  </div>
  <div class="card">
    <h2>Trofeus <span class="badge gold">PRO</span></h2>
    <div class="metric" id="totalTrophiesVal">0</div>
    <div class="label">Brawlers: <span id="unlockedVal">0</span>/<span id="totalBrawlersVal">80</span></div>
    <div class="progress"><div class="progress-bar" id="unlockedBar" style="width:0%;background:#7c3aed"></div></div>
  </div>
  <div class="card">
    <h2>FPS / Ciclo</h2>
    <div class="metric" id="fpsVal">0</div>
    <div class="label">Cycle: <span id="cycleVal">0</span> ms</div>
  </div>
  <div class="card">
    <h2>Combate <span class="badge purple">LIVE</span></h2>
    <div class="metric small" id="combatModeVal">neutral</div>
    <div class="label">Inimigos: <span id="enemiesVal">0</span> | HP: <span id="hpVal">100%</span></div>
    <div class="progress" style="margin-top:.3rem"><div class="progress-bar" id="hpBar" style="width:100%;background:#22c55e"></div></div>
  </div>
  <div class="card">
    <h2>Sessao</h2>
    <div class="metric small" id="uptimeVal">0:00</div>
    <div class="label">Partidas: <span id="sessionMatchesVal">0</span></div>
  </div>
  <div class="card">
    <h2>RL Q-Learning</h2>
    <div class="metric small" id="qStatesVal">0</div>
    <div class="label">Epsilon: <span id="epsVal">0.000</span></div>
    <div class="progress"><div class="progress-bar" id="epsBar" style="width:0%"></div></div>
  </div>
  <div class="card">
    <h2>ELO Combinacoes</h2>
    <div class="metric small" id="eloCountVal">0</div>
    <div class="label">Top brawlers (melhores mapas)</div>
    <div id="topElo" style="font-size:.75rem;margin-top:.3rem"></div>
  </div>
  <div class="card">
    <h2>AI Pick <span class="badge purple">PRO</span></h2>
    <div class="metric small" id="aiPickBrawler">—</div>
    <div class="label">Confianca: <span id="aiPickConf">0%</span></div>
    <div class="label" id="aiPickReason"></div>
    <div style="margin-top:.4rem">
      <div class="label">Previsao Vitoria</div>
      <div class="win-prediction-bar"><div class="fill" id="winPredBar" style="width:50%;background:#38bdf8"></div><div class="text" id="winPredText">50%</div></div>
    </div>
  </div>
  <div class="card">
    <h2>Ultimo Screenshot</h2>
    <img id="lastScreenshot" class="screenshot" src="" alt="screenshot" style="display:none">
    <div class="label" id="ssLabel">Sem screenshot</div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Eventos em Tempo Real</h2>
    <div class="event-log" id="eventLog"><div class="event">A aguardar dados...</div></div>
  </div>
</div>
</div>

<div id="tab-history" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Rewards ao Longo do Tempo</h2>
    <canvas id="rewardChart" height="200"></canvas>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Historico Completo</h2>
    <div id="historyTable"></div>
  </div>
</div>
</div>

<div id="tab-replays" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Replays Gravados</h2>
    <table><thead><tr><th>Nome</th><th>Frames</th><th>Duracao</th><th>Caminho</th></tr></thead>
    <tbody id="replayTable"><tr><td colspan="4">Carregando...</td></tr></tbody></table>
  </div>
</div>
</div>

<div id="tab-abtest" class="tab-content">
<div class="grid">
  <div class="card">
    <h2>A/B Test Status</h2>
    <div class="metric small" id="abStatus">Inativo</div>
    <div style="margin-top:.5rem">
      <span class="btn" onclick="startAB()">Iniciar</span>
      <span class="btn danger" onclick="stopAB()">Parar</span>
    </div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Resultados</h2>
    <table><thead><tr><th>Variante</th><th>Partidas</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>Avg Reward</th></tr></thead>
    <tbody id="abTable"><tr><td colspan="6">Sem dados</td></tr></tbody></table>
  </div>
</div>
</div>

<div id="tab-recovery" class="tab-content">
<div class="grid">
  <div class="card">
    <h2>Error Recovery</h2>
    <div class="row"><span class="label">Ativado</span><span id="erEnabled" class="metric small">—</span></div>
    <div class="row"><span class="label">Erros Total</span><span id="erTotal" class="metric small">0</span></div>
    <div class="row"><span class="label">Recuperados</span><span id="erRecovered" class="metric small">0</span></div>
    <div class="row"><span class="label">Circuit Breaker</span><span id="erCircuit" class="metric small">CLOSED</span></div>
  </div>
  <div class="card">
    <h2>State Recovery</h2>
    <div class="row"><span class="label">Recovery Ativo</span><span id="srActive" class="metric small">Nao</span></div>
    <div class="row"><span class="label">Tentativas</span><span id="srAttempts" class="metric small">0</span></div>
    <div class="row"><span class="label">Estado Atual</span><span id="srState" class="metric small">—</span></div>
  </div>
  <div class="card">
    <h2>AutoCalibrator</h2>
    <div class="row"><span class="label">Ativado</span><span id="acEnabled" class="metric small">—</span></div>
    <div class="row"><span class="label">Cache Size</span><span id="acCache" class="metric small">0</span></div>
  </div>
  <div class="card">
    <h2>OCR Detector</h2>
    <div class="row"><span class="label">Ativado</span><span id="ocrEnabled" class="metric small">—</span></div>
    <div class="row"><span class="label">Reader Disponivel</span><span id="ocrReader" class="metric small">Nao</span></div>
  </div>
  <div class="card">
    <h2>Debug Visualizer</h2>
    <div class="row"><span class="label">Ativado</span><span id="dvEnabled" class="metric small">—</span></div>
    <div class="row"><span class="label">Em execucao</span><span id="dvRunning" class="metric small">Nao</span></div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Detalhes Recovery</h2>
    <div id="recoveryDetail" style="font-family:monospace;font-size:.78rem;max-height:200px;overflow-y:auto">
      A aguardar dados...
    </div>
  </div>
</div>
</div>

<div id="tab-brawlers" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Stats por Brawler <span class="badge gold">PRO</span></h2>
    <div id="brawlerStatsList"><div class="label">A carregar dados...</div></div>
  </div>
</div>
</div>

<div id="tab-analysis" class="tab-content">
<div class="grid">
  <div class="card">
    <h2>Ultima Analise <span class="badge purple">PRO</span></h2>
    <div class="analysis-score mid" id="analysisScore">—</div>
    <div class="label" style="text-align:center" id="analysisResult">Sem dados</div>
  </div>
  <div class="card">
    <h2>Erros & Sugestoes</h2>
    <div id="analysisErrors" style="font-size:.8rem"><div class="label">Sem analise disponivel</div></div>
  </div>
  <div class="card">
    <h2>Pontos Fortes</h2>
    <div id="analysisStrengths" style="font-size:.8rem"><div class="label">Sem analise disponivel</div></div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Matchup & Build</h2>
    <div class="grid" style="padding:0">
      <div class="card" style="border:none">
        <h2>Matchup</h2>
        <div id="matchupAnalysis" class="label">—</div>
      </div>
      <div class="card" style="border:none">
        <h2>Build Sugerida</h2>
        <div id="buildSuggestion" class="label">—</div>
      </div>
      <div class="card" style="border:none">
        <h2>Posicionamento</h2>
        <div id="positioningTip" class="label">—</div>
      </div>
    </div>
  </div>
</div>
</div>

<div id="tab-aicoach" class="tab-content">
<div class="grid">
  <div class="card">
    <h2>AI Pick Suggester <span class="badge purple">PRO</span></h2>
    <div class="metric small" id="coachPickBrawler">—</div>
    <div class="label">Mapa: <span id="coachPickMap">—</span></div>
    <div class="label">Confianca: <span id="coachPickConf">0%</span></div>
    <div class="label" id="coachPickReason"></div>
    <div class="label" style="margin-top:.3rem">Alternativas: <span id="coachPickAlts">—</span></div>
  </div>
  <div class="card">
    <h2>Previsao de Vitoria</h2>
    <div class="metric" id="coachWinPred">50%</div>
    <div class="win-prediction-bar" style="margin-top:.5rem"><div class="fill" id="coachWinBar" style="width:50%;background:#38bdf8"></div><div class="text" id="coachWinText">50%</div></div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Dicas do Coach <span class="badge purple">PRO</span></h2>
    <div id="coachTipsList"><div class="coach-tip">Joga mais partidas para receber dicas personalizadas</div></div>
  </div>
</div>
</div>

<div id="tab-trophies" class="tab-content">
<div class="grid">
  <div class="card">
    <h2>Trofeus Totais <span class="badge gold">PRO</span></h2>
    <div class="metric" id="trophyTotalVal">0</div>
    <div class="label">Brawlers desbloqueados: <span id="trophyUnlockedVal">0</span>/80</div>
  </div>
  <div class="card">
    <h2>Progresso Semanal</h2>
    <div class="weekly-card" id="weeklyProgress">
      <div class="stat"><div class="num" id="weeklyTrophies">0</div><div class="lbl">Trofeus +/-</div></div>
      <div class="stat"><div class="num" id="weeklyMatches">0</div><div class="lbl">Partidas</div></div>
    </div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Grafico de Trofeus <span class="badge gold">PRO</span></h2>
    <canvas id="trophyChart" height="250"></canvas>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Evolucao Diaria</h2>
    <div id="dailyEvolutionTable"><div class="label">A carregar...</div></div>
  </div>
</div>
</div>

<div id="tab-esports" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Esports Overlay <span class="badge purple">PRO</span></h2>
    <div style="background:linear-gradient(135deg,#0f172a,#1e1b4b);border-radius:8px;padding:1rem;border:1px solid #7c3aed">
      <!-- Top bar -->
      <div class="esports-bar">
        <div><div class="label">Brawler</div><div class="value" id="esBrawler">—</div></div>
        <div><div class="label">Estado</div><div class="value" id="esState">—</div></div>
        <div><div class="label">Mapa</div><div class="value" id="esMap">—</div></div>
        <div><div class="label">Win Rate</div><div class="value" id="esWR">0%</div></div>
      </div>
      <!-- Mid stats -->
      <div style="display:flex;gap:.5rem;margin-bottom:.5rem;flex-wrap:wrap">
        <div class="esports-bar" style="flex:1"><div><div class="label">Partidas</div><div class="value" id="esMatches">0</div></div></div>
        <div class="esports-bar" style="flex:1"><div><div class="label">Trofeus</div><div class="value" id="esTrophies">0</div></div></div>
        <div class="esports-bar" style="flex:1"><div class="label">FPS</div><div class="value" id="esFPS">0</div></div>
        <div class="esports-bar" style="flex:1"><div><div class="label">Previsao</div><div class="value" id="esPrediction">50%</div></div></div>
      </div>
      <!-- AI Coach bar -->
      <div class="esports-bar" style="background:linear-gradient(90deg,#7c3aed,#ec4899)">
        <div><div class="label">AI Pick</div><div class="value" id="esAIPick">—</div></div>
        <div><div class="label">Confianca</div><div class="value" id="esAIConf">0%</div></div>
        <div><div class="label">Razao</div><div class="value" id="esAIReason">—</div></div>
      </div>
      <!-- Coach tips scroll -->
      <div id="esCoachTips" style="font-size:.75rem;color:#c4b5fd;max-height:80px;overflow-y:auto;margin-top:.3rem">
        A aguardar dicas do coach...
      </div>
    </div>
  </div>
</div>
</div>

<div id="tab-logs" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Logs em Tempo Real <span class="badge green">NEW</span></h2>
    <div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.5rem">
      <select id="logLevel" onchange="refreshLogs()" style="width:auto">
        <option value="ALL">Todos</option>
        <option value="DEBUG">DEBUG</option>
        <option value="INFO" selected>INFO</option>
        <option value="WARNING">WARNING</option>
        <option value="ERROR">ERROR</option>
        <option value="CRITICAL">CRITICAL</option>
      </select>
      <select id="logComponent" onchange="refreshLogs()" style="width:auto">
        <option value="ALL">Todos componentes</option>
        <option value="wrapper">wrapper</option>
        <option value="state">state_manager</option>
        <option value="play">play</option>
        <option value="lobby">lobby</option>
        <option value="detect">detect</option>
        <option value="dashboard">dashboard</option>
      </select>
      <input id="logSearch" type="text" placeholder="Procurar..." oninput="refreshLogs()" style="padding:.4rem;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;font-size:.8rem;flex:1;min-width:150px">
      <span class="btn" onclick="refreshLogs()">Atualizar</span>
      <span class="btn" onclick="toggleLogStream()" id="logStreamBtn">Stream: OFF</span>
      <span class="btn warn" onclick="clearLogs()">Limpar</span>
    </div>
    <div id="logContainer" style="background:#0a0f1a;border:1px solid #334155;border-radius:4px;padding:.5rem;max-height:500px;overflow-y:auto;font-family:monospace;font-size:.75rem;line-height:1.6">
      <div class="label">A carregar logs...</div>
    </div>
    <div style="margin-top:.3rem;display:flex;justify-content:space-between">
      <span class="label" id="logStats">0 linhas</span>
      <label style="font-size:.75rem;color:#64748b"><input type="checkbox" id="logAutoScroll" checked> Auto-scroll</label>
    </div>
  </div>
</div>
</div>

<div id="tab-notifications" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Notificacoes e Alertas <span class="badge green">NEW</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap">
      <div style="flex:1;min-width:280px">
        <h3 style="font-size:.85rem;color:#94a3b8;margin-bottom:.5rem">Configuracao</h3>
        <div class="control-row">
          <span class="label">Webhook URL</span>
          <input id="notifWebhook" type="text" placeholder="https://discord.com/api/webhooks/..." style="flex:1;margin-left:.5rem;padding:.3rem .5rem;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;font-size:.8rem">
        </div>
        <div class="control-row">
          <span class="label">Notificar Browser</span>
          <label class="toggle control"><input type="checkbox" id="notifBrowser" checked><span class="slider"></span></label>
        </div>
        <div class="control-row">
          <span class="label">Notificar Desktop</span>
          <label class="toggle control"><input type="checkbox" id="notifDesktop"><span class="slider"></span></label>
        </div>
        <div class="control-row">
          <span class="label">On Crash</span>
          <label class="toggle control"><input type="checkbox" id="notifCrash" checked><span class="slider"></span></label>
        </div>
        <div class="control-row">
          <span class="label">On Loss Streak (>=3)</span>
          <label class="toggle control"><input type="checkbox" id="notifLosses" checked><span class="slider"></span></label>
        </div>
        <div class="control-row">
          <span class="label">On Trophy Limit</span>
          <label class="toggle control"><input type="checkbox" id="notifTrophy" checked><span class="slider"></span></label>
        </div>
        <div style="margin-top:.5rem">
          <span class="btn" onclick="saveNotifConfig()">Guardar</span>
          <span class="btn" onclick="testNotification()">Testar</span>
        </div>
      </div>
      <div style="flex:1;min-width:280px">
        <h3 style="font-size:.85rem;color:#94a3b8;margin-bottom:.5rem">Historico</h3>
        <div id="notifHistory" style="max-height:300px;overflow-y:auto;font-size:.8rem">
          <div class="label">A carregar...</div>
        </div>
      </div>
    </div>
  </div>
</div>
</div>

<div id="tab-config" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Editor de Configuracao <span class="badge green">NEW</span></h2>
    <div style="display:flex;gap:.5rem;margin-bottom:.5rem;flex-wrap:wrap">
      <span class="btn" onclick="loadConfig()">Carregar</span>
      <span class="btn" onclick="saveConfig()">Guardar</span>
      <span class="btn warn" onclick="resetConfig()">Restaurar Defaults</span>
    </div>
    <textarea id="configEditor" style="width:100%;min-height:400px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:.5rem;font-family:monospace;font-size:.8rem" placeholder="Carregue a configuracao...">{}</textarea>
    <div id="configStatus" class="label" style="margin-top:.3rem"></div>
  </div>
</div>
</div>

<div id="tab-antiban" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Anti-Ban Dashboard <span class="badge green">NEW</span></h2>
    <div class="grid" style="padding:0">
      <div class="card" style="border:none">
        <h2>Status</h2>
        <div class="metric small" id="abStatusVal">—</div>
        <div class="label">Win Rate Target: <span id="abWinTarget">—</span></div>
        <div class="label">Win Rate Atual: <span id="abWinCurrent">—</span></div>
        <div class="label">Throttle: <span id="abThrottle">—</span></div>
      </div>
      <div class="card" style="border:none">
        <h2>Schedule</h2>
        <div class="label">Proximo jogo: <span id="abNextGame">—</span></div>
        <div class="label">Horario randomizado: <span id="abRandom">—</span></div>
      </div>
      <div class="card" style="border:none">
        <h2>Padroes Detetados</h2>
        <div id="abPatterns"><div class="label">Sem padroes</div></div>
      </div>
      <div class="card" style="border:none">
        <h2>Obfuscation</h2>
        <div class="label">Missclicks: <span id="abMissclicks">0</span></div>
        <div class="label">Delay noise: <span id="abDelayNoise">0</span></div>
        <div class="label">Fingerprint: <span id="abFingerprint">—</span></div>
      </div>
    </div>
  </div>
</div>
</div>

<div id="tab-learning" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Controlo Modo Teste <span class="badge green">LIVE</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:center">
      <div style="flex:1;min-width:200px">
        <div class="metric small" id="lmStatusVal">Inativo</div>
        <div class="label">Partida <span id="lmMatchVal">0</span> / <span id="lmMaxVal">0</span></div>
      </div>
      <div>
        <span class="btn" style="background:#22c55e" onclick="startLearningMode()">Iniciar Modo Teste</span>
        <span class="btn danger" onclick="stopLearningMode()">Parar Modo Teste</span>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Kills</h2>
    <div class="metric" id="lmKillsVal">0</div>
    <div class="label">Mortes: <span id="lmDeathsVal">0</span></div>
  </div>
  <div class="card">
    <h2>Detecoes</h2>
    <div class="metric" id="lmDetectVal">0</div>
    <div class="label">Player: <span id="lmPlayerVal">0</span></div>
  </div>
  <div class="card">
    <h2>Precisao</h2>
    <div class="metric" id="lmAccuracyVal">0%</div>
    <div class="label">Kills / Ataques</div>
  </div>
  <div class="card">
    <h2>Dano</h2>
    <div class="metric small" id="lmDamageVal">0</div>
    <div class="label">Infligido</div>
  </div>
  <div class="card">
    <h2>Sobrevivencia</h2>
    <div class="metric small" id="lmSurvivalVal">0s</div>
    <div class="label">Duracao atual</div>
  </div>
  <div class="card">
    <h2>Brawler</h2>
    <div class="metric small" id="lmBrawlerVal">—</div>
    <div class="label">Em teste</div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Grafico Deteccoes (ultimos 60s)</h2>
    <canvas id="lmDetectChart" height="200"></canvas>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Kills por Partida</h2>
    <canvas id="lmKillsChart" height="200"></canvas>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Historico de Treino</h2>
    <table><thead><tr><th>Brawler</th><th>Resultado</th><th>Kills</th><th>Mortes</th><th>Duracao</th></tr></thead>
    <tbody id="lmHistoryTable"><tr><td colspan="5">Sem dados</td></tr></tbody></table>
  </div>
</div>
</div>

<div id="tab-farm" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Controlo Modo Executar <span class="badge green">LIVE</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:center">
      <div style="flex:1;min-width:200px">
        <div class="metric small" id="farmStatusVal">Inativo</div>
        <div class="label">Partidas: <span id="farmMatchVal">0</span> / <span id="farmTargetVal">0</span></div>
      </div>
      <div>
        <span class="btn" style="background:#22c55e" onclick="startFarmMode()">Iniciar Farm</span>
        <span class="btn danger" onclick="stopFarmMode()">Parar Farm</span>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Trofeus Sessao</h2>
    <div class="metric" id="farmTrophiesVal">0</div>
    <div class="label">Ganhos/perdidos</div>
  </div>
  <div class="card">
    <h2>Win Rate</h2>
    <div class="metric" id="farmWinRateVal">0%</div>
    <div class="label">W / L / D</div>
  </div>
  <div class="card">
    <h2>Tempo Medio</h2>
    <div class="metric small" id="farmAvgTimeVal">0s</div>
    <div class="label">Por partida</div>
  </div>
  <div class="card">
    <h2>APM</h2>
    <div class="metric small" id="farmApmVal">0</div>
    <div class="label">Acoes/min</div>
  </div>
  <div class="card">
    <h2>Estado</h2>
    <div class="metric small" id="farmStateVal">—</div>
    <div class="label">Atual</div>
  </div>
  <div class="card">
    <h2>Brawler</h2>
    <div class="metric small" id="farmBrawlerVal">—</div>
    <div class="label">Em uso</div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Grafico de Trofeus</h2>
    <canvas id="farmTrophyChart" height="200"></canvas>
  </div>
</div>
</div>

<div id="tab-learn" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Controlo Modo Aprender <span class="badge purple">AI</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:center">
      <div style="flex:1;min-width:200px">
        <div class="metric small" id="learnStatusVal">Inativo</div>
        <div class="label">Motor: <span id="learnEngineVal">—</span></div>
      </div>
      <div>
        <span class="btn" style="background:#7c3aed" onclick="startLearnMode()">Iniciar Aprender</span>
        <span class="btn danger" onclick="stopLearnMode()">Parar Aprender</span>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Q-Table</h2>
    <div class="metric" id="learnQTableVal">0</div>
    <div class="label">Estados</div>
  </div>
  <div class="card">
    <h2>Epsilon</h2>
    <div class="metric" id="learnEpsilonVal">0.0</div>
    <div class="label">Exploracao</div>
  </div>
  <div class="card">
    <h2>Reward</h2>
    <div class="metric small" id="learnRewardVal">0</div>
    <div class="label">Ultimo / Episodio</div>
  </div>
  <div class="card">
    <h2>PPO Loss</h2>
    <div class="metric small" id="learnPpoLossVal">0</div>
    <div class="label">Policy / Value</div>
  </div>
  <div class="card">
    <h2>Buffer</h2>
    <div class="metric small" id="learnBufferVal">0</div>
    <div class="label">Experiencias</div>
  </div>
  <div class="card">
    <h2>Acao</h2>
    <div class="metric small" id="learnActionVal">—</div>
    <div class="label">Ultima escolhida</div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Rewards por Episodio</h2>
    <canvas id="learnRewardChart" height="200"></canvas>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Contagem de Acoes</h2>
    <canvas id="learnActionChart" height="200"></canvas>
  </div>
</div>
</div>

<div id="tab-detections" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>ESP / Visao em Tempo Real <span class="badge green">ESP</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:center">
      <div style="flex:1;min-width:200px">
        <div class="metric small" id="espStatusVal">OFF</div>
        <div class="label">FPS: <span id="espFpsVal">0</span> | Objetos: <span id="espObjectsVal">0</span></div>
      </div>
      <div>
        <span class="btn" style="background:#22c55e" onclick="toggleESP()">Ligar ESP</span>
        <span class="btn danger" onclick="toggleESP(false)">Desligar ESP</span>
      </div>
    </div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Lista de Deteccoes</h2>
    <table><thead><tr><th>Classe</th><th>Conf</th><th>X</th><th>Y</th><th>W</th><th>H</th></tr></thead>
    <tbody id="detectionsTable"><tr><td colspan="6">Sem dados</td></tr></tbody></table>
  </div>
  <div class="card">
    <h2>Inimigos</h2>
    <div class="metric" id="detEnemyVal">0</div>
    <div class="label">Detetados</div>
  </div>
  <div class="card">
    <h2>Aliados</h2>
    <div class="metric" id="detTeamVal">0</div>
    <div class="label">Detetados</div>
  </div>
  <div class="card">
    <h2>Paredes</h2>
    <div class="metric" id="detWallVal">0</div>
    <div class="label">Obstaculos</div>
  </div>
  <div class="card">
    <h2>Arbustos</h2>
    <div class="metric" id="detBushVal">0</div>
    <div class="label">Esconderijos</div>
  </div>
  <div class="card">
    <h2>Powerups</h2>
    <div class="metric" id="detPowerVal">0</div>
    <div class="label">Itens</div>
  </div>
  <div class="card">
    <h2>Modelo</h2>
    <div class="metric small" id="detModelVal">—</div>
    <div class="label">Ativo</div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Estatisticas de Visao</h2>
    <div id="visionStats">A carregar...</div>
  </div>
</div>
</div>

<div id="tab-analytics" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Analytics Avancadas <span class="badge green">NEW</span></h2>
    <div class="tabs" style="padding:0">
      <div class="tab active" onclick="showAnalyticsTab('maps')">Mapas</div>
      <div class="tab" onclick="showAnalyticsTab('perf')">Performance</div>
      <div class="tab" onclick="showAnalyticsTab('rl')">RL Insights</div>
      <div class="tab" onclick="showAnalyticsTab('sessions')">Sessoes</div>
    </div>
    <div id="analytics-maps" style="padding:1rem 0">
      <div class="label">Win Rate por Mapa (dados do Match Analyzer)</div>
      <div id="analyticsMapsContent"><div class="label">A carregar...</div></div>
    </div>
    <div id="analytics-perf" style="padding:1rem 0;display:none">
      <div class="label">FPS ao longo do tempo</div>
      <canvas id="perfFPSChart" height="200"></canvas>
      <div class="label" style="margin-top:1rem">Latencia YOLO (ms)</div>
      <div id="perfYOLO"><div class="label">N/A</div></div>
    </div>
    <div id="analytics-rl" style="padding:1rem 0;display:none">
      <div class="label">Epsilon Decay</div>
      <div id="rlEpsilonVal">Epsilon: <span class="metric small">0.000</span></div>
      <div class="label" style="margin-top:1rem">Estados Visitados</div>
      <div id="rlStatesVal">Q-States: <span class="metric small">0</span></div>
    </div>
    <div id="analytics-sessions" style="padding:1rem 0;display:none">
      <div class="label">Historico de Sessoes</div>
      <table><thead><tr><th>Inicio</th><th>Duracao</th><th>Partidas</th><th>Wins</th><th>Brawlers</th></tr></thead>
      <tbody id="sessionsTable"><tr><td colspan="5">A carregar...</td></tr></tbody></table>
    </div>
  </div>
</div>
</div>


<div id="tab-training" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Model Registry <span class="badge gold">PRO</span></h2>
    <div style="display:flex;gap:.5rem;margin-bottom:.5rem">
      <span class="btn btn-sm" onclick="refreshTrainingModels()">Atualizar</span>
      <span class="btn btn-sm" style="background:#7c3aed" onclick="rescanModels()">Scan Models/</span>
    </div>
    <table><thead><tr><th>Modelo</th><th>Versao</th><th>Schema</th><th>mAP50</th><th>Status</th></tr></thead>
    <tbody id="modelsTable"><tr><td colspan="5">A carregar...</td></tr></tbody></table>
  </div>
  <div class="card">
    <h2>Dataset Stats</h2>
    <div class="row"><span class="label">Total Imagens</span><span id="dsImages" class="metric small">—</span></div>
    <div class="row"><span class="label">Total Boxes</span><span id="dsBoxes" class="metric small">—</span></div>
    <div class="row"><span class="label">Classes</span><span id="dsClasses" class="metric small">—</span></div>
  </div>
  <div class="card">
    <h2>Class Distribution</h2>
    <div id="dsClassDist" style="font-size:.75rem;max-height:200px;overflow-y:auto">—</div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Ultimo Treino</h2>
    <div id="lastTraining" style="font-family:monospace;font-size:.78rem;max-height:200px;overflow-y:auto">
      <div class="label">Nenhum treino registado. Execute <code>python train.py</code></div>
    </div>
  </div>
</div>
</div>

<script>
const API = '';
let lastEvents = [];

function showTab(id) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-'+id).classList.add('active');
}

async function poll() {
  try {
    const health = await fetch(API + '/api/health');
    if (!health.ok) throw new Error('health check failed');

    const res = await fetch(API + '/api/live');
    if (!res.ok) throw new Error('live endpoint failed');
    const d = await res.json();
    const botConnected = !!d.running || !!d.current_state || !!d.brawler;
    updateBotButtons(!!d.running);
    document.getElementById('connStatus').innerHTML = botConnected
      ? '<span class="status-dot status-online"></span>Online'
      : '<span class="status-dot status-warning"></span>Dashboard OK • Bot Offline';
    document.getElementById('stateVal').textContent = d.current_state || '—';
    document.getElementById('brawlerVal').textContent = (d.brawler || '—') + ' @ ' + (d.map_name || '—');
    document.getElementById('matchesVal').textContent = d.matches_total || 0;
    document.getElementById('wrVal').textContent = ((d.win_rate||0)*100).toFixed(1) + '%';
    document.getElementById('wrBar').style.width = ((d.win_rate||0)*100) + '%';
    document.getElementById('fpsVal').textContent = (d.fps || 0).toFixed(1);
    document.getElementById('cycleVal').textContent = (d.cycle_time_ms || 0).toFixed(1);
    document.getElementById('qStatesVal').textContent = d.q_states || 0;
    document.getElementById('epsVal').textContent = (d.epsilon || 0).toFixed(3);
    document.getElementById('epsBar').style.width = ((d.epsilon||0)/0.5*100) + '%';
    document.getElementById('eloCountVal').textContent = d.elo_combinations || 0;

    // Top ELO
    const top = d.top_elo || [];
    document.getElementById('topElo').innerHTML = top.slice(0,5).map(
      e => `<div class="row"><span>${e.brawler}@${e.map}</span><span>${e.score.toFixed(0)}</span></div>`
    ).join('') || '<div class="label">Sem dados</div>';

    // Screenshot
    if (d.screenshot_b64) {
      document.getElementById('lastScreenshot').src = 'data:image/jpeg;base64,' + d.screenshot_b64;
      document.getElementById('lastScreenshot').style.display = 'block';
      document.getElementById('ssLabel').textContent = new Date(d.timestamp*1000).toLocaleTimeString();
    }

    // Eventos
    const ev = d.recent_events || [];
    if (JSON.stringify(ev) !== JSON.stringify(lastEvents)) {
      lastEvents = ev;
      const log = document.getElementById('eventLog');
      log.innerHTML = ev.slice().reverse().map(e =>
        `<div class="event"><span class="time">${new Date(e.timestamp*1000).toLocaleTimeString()}</span> ` +
        `<strong>${e.event_type}</strong> ${JSON.stringify(e.details).slice(0,80)}</div>`
      ).join('');
    }

    // Phase 9: Recovery stats from live data
    document.getElementById('erEnabled').textContent = d.error_recovery_enabled ? 'Sim' : 'Nao';
    document.getElementById('erTotal').textContent = d.error_total || 0;
    document.getElementById('erRecovered').textContent = d.error_recovered || 0;
    document.getElementById('erCircuit').textContent = d.error_circuit_state || 'CLOSED';
    document.getElementById('srActive').textContent = d.state_recovery_active ? 'Sim' : 'Nao';
    document.getElementById('srAttempts').textContent = d.state_recovery_attempts || 0;
    document.getElementById('srState').textContent = d.state_recovery_current || '—';
    document.getElementById('acEnabled').textContent = d.autocalibrator_enabled ? 'Sim' : 'Nao';
    document.getElementById('acCache').textContent = d.autocalibrator_cache_size || 0;
    document.getElementById('ocrEnabled').textContent = d.ocr_detector_enabled ? 'Sim' : 'Nao';
    document.getElementById('ocrReader').textContent = d.ocr_reader_available ? 'Sim' : 'Nao';
    document.getElementById('dvEnabled').textContent = d.debug_visualizer_enabled ? 'Sim' : 'Nao';
    document.getElementById('dvRunning').textContent = d.debug_visualizer_running ? 'Sim' : 'Nao';

    // Premium: Trophies
    document.getElementById('totalTrophiesVal').textContent = d.total_trophies || 0;
    document.getElementById('unlockedVal').textContent = d.unlocked_brawlers || 0;
    document.getElementById('totalBrawlersVal').textContent = d.total_brawlers || 80;
    document.getElementById('unlockedBar').style.width = ((d.unlocked_brawlers||0)/(d.total_brawlers||80)*100) + '%';

    // Premium: AI Pick (live tab)
    const pick = d.ai_pick_suggestion;
    if (pick) {
      document.getElementById('aiPickBrawler').textContent = pick.brawler || '—';
      document.getElementById('aiPickConf').textContent = ((pick.confidence||0)*100).toFixed(0) + '%';
      document.getElementById('aiPickReason').textContent = pick.reason || '';
    }
    const wp = d.win_prediction || 0;
    document.getElementById('winPredBar').style.width = (wp*100) + '%';
    document.getElementById('winPredBar').style.background = wp > 0.6 ? '#22c55e' : wp > 0.4 ? '#f59e0b' : '#ef4444';
    document.getElementById('winPredText').textContent = (wp*100).toFixed(0) + '%';

    // Premium: Esports overlay
    document.getElementById('esBrawler').textContent = d.brawler || '—';
    document.getElementById('esState').textContent = d.current_state || '—';
    document.getElementById('esMap').textContent = d.map_name || '—';
    document.getElementById('esWR').textContent = ((d.win_rate||0)*100).toFixed(1) + '%';
    document.getElementById('esMatches').textContent = d.matches_total || 0;
    document.getElementById('esTrophies').textContent = d.total_trophies || 0;
    document.getElementById('esFPS').textContent = (d.fps||0).toFixed(1);
    document.getElementById('esPrediction').textContent = (wp*100).toFixed(0) + '%';
    if (pick) {
      document.getElementById('esAIPick').textContent = pick.brawler || '—';
      document.getElementById('esAIConf').textContent = ((pick.confidence||0)*100).toFixed(0) + '%';
      document.getElementById('esAIReason').textContent = pick.reason || '—';
    }
    const tips = d.coach_tips || [];
    document.getElementById('esCoachTips').innerHTML = tips.map(t => `<div>${t}</div>`).join('') || 'Sem dicas';

    // Combat & Session (real data)
    document.getElementById('combatModeVal').textContent = d.combat_mode || 'neutral';
    document.getElementById('enemiesVal').textContent = d.enemies_detected || 0;
    const hp = d.hp_estimate || 1.0;
    document.getElementById('hpVal').textContent = (hp*100).toFixed(0) + '%';
    document.getElementById('hpBar').style.width = (hp*100) + '%';
    document.getElementById('hpBar').style.background = hp > 0.6 ? '#22c55e' : hp > 0.3 ? '#f59e0b' : '#ef4444';
    // Uptime
    const uptime = d.uptime_seconds || 0;
    const hours = Math.floor(uptime/3600);
    const mins = Math.floor((uptime%3600)/60);
    const secs = Math.floor(uptime%60);
    document.getElementById('uptimeVal').textContent = hours > 0 ? `${hours}:${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')}` : `${mins}:${String(secs).padStart(2,'0')}`;
    document.getElementById('sessionMatchesVal').textContent = d.matches_total || 0;
  } catch (err) {
    document.getElementById('connStatus').innerHTML = '<span class="status-dot status-offline"></span>Offline';
  }
}

async function pollReplays() {
  try {
    const res = await fetch(API + '/api/replays');
    const d = await res.json();
    const tbody = document.getElementById('replayTable');
    tbody.innerHTML = (d.replays || []).map(r =>
      `<tr><td>${r.name}</td><td>${r.frames}</td><td>${r.duration.toFixed(1)}s</td><td>${r.path}</td></tr>`
    ).join('') || '<tr><td colspan="4">Nenhum replay</td></tr>';
  } catch(e){}
}

async function pollAB() {
  try {
    const res = await fetch(API + '/api/abtest');
    const d = await res.json();
    document.getElementById('abStatus').textContent = d.active ? 'Ativo ('+d.current_variant+')' : 'Inativo';
    const tbody = document.getElementById('abTable');
    const vars = d.variants || {};
    tbody.innerHTML = Object.entries(vars).map(([k,v])=>
      `<tr><td>${k}</td><td>${v.matches}</td><td>${v.wins}</td><td>${v.losses}</td><td>${(v.win_rate*100).toFixed(1)}%</td><td>${v.avg_reward}</td></tr>`
    ).join('') || '<tr><td colspan="6">Sem dados</td></tr>';
  } catch(e){}
}

async function startAB() {
  await fetch(API + '/api/abtest/start', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({variants:{control:{},test_v2:{}}})
  });
  pollAB();
}
async function stopAB() {
  await fetch(API + '/api/abtest/stop', {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  pollAB();
}

// Bot control functions
async function botStart() {
  const btn = document.getElementById('startBtn');
  btn.classList.add('disabled'); btn.textContent = 'A iniciar...';
  try {
    const res = await fetch(API + '/api/bot/start', {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const d = await res.json();
    if (d.ok) { toast('Bot iniciado!', 'success'); updateBotButtons(true); }
    else { toast('Erro: ' + (d.error || 'falha'), 'error'); btn.classList.remove('disabled'); btn.textContent = 'Iniciar'; }
  } catch(e) { toast('Erro ao iniciar: ' + e, 'error'); btn.classList.remove('disabled'); btn.textContent = 'Iniciar'; }
}
async function botStop() {
  const btn = document.getElementById('stopBtn');
  btn.classList.add('disabled'); btn.textContent = 'A parar...';
  try {
    const res = await fetch(API + '/api/bot/stop', {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const d = await res.json();
    if (d.ok) { toast('Bot parado!', 'info'); updateBotButtons(false); }
    else { toast('Erro: ' + (d.error || 'falha'), 'error'); btn.classList.remove('disabled'); btn.textContent = 'Parar'; }
  } catch(e) { toast('Erro ao parar: ' + e, 'error'); btn.classList.remove('disabled'); btn.textContent = 'Parar'; }
}
async function botRestart() {
  if (!confirm('Reiniciar o bot?')) return;
  toast('A reiniciar bot...', 'warning', 2000);
  try {
    const res = await fetch(API + '/api/bot/restart', {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const d = await res.json();
    if (d.ok) { toast('Bot reiniciado!', 'success'); updateBotButtons(true); }
    else { toast('Erro: ' + (d.error || 'falha'), 'error'); }
  } catch(e) { toast('Erro ao reiniciar: ' + e, 'error'); }
}
function updateBotButtons(running) {
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const restartBtn = document.getElementById('restartBtn');
  const pauseBtn = document.getElementById('pauseBtn');
  if (startBtn) { startBtn.classList.toggle('disabled', running); startBtn.classList.toggle('running', running); startBtn.textContent = 'Iniciar'; }
  if (stopBtn) { stopBtn.classList.toggle('disabled', !running); stopBtn.textContent = 'Parar'; }
  if (restartBtn) { restartBtn.classList.toggle('disabled', !running); }
  if (pauseBtn) { pauseBtn.classList.toggle('disabled', !running); }
}

// Phase 1: Pause/Resume toggle
async function botPauseToggle() {
  const btn = document.getElementById('pauseBtn');
  const isPaused = btn.textContent === 'Retomar';
  try {
    const endpoint = isPaused ? '/api/bot/resume' : '/api/bot/pause';
    const res = await fetch(API + endpoint, {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const d = await res.json();
    if (d.ok) {
      btn.textContent = isPaused ? 'Pausar' : 'Retomar';
      btn.style.background = isPaused ? '#f59e0b' : '#22c55e';
    } else {
      alert('Erro: ' + (d.error || 'falha'));
    }
  } catch(e) { alert('Erro: ' + e); }
}

// Phase 1: Manual actions
async function botAction(action) {
  try {
    const res = await fetch(API + '/api/bot/action', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({action: action})
    });
    const d = await res.json();
    if (!d.ok) console.warn('Action failed:', d.error);
  } catch(e) { console.warn('Action error:', e); }
}

// Phase 1: Set brawler
async function setBrawler(name) {
  if (!name) return;
  try {
    const res = await fetch(API + '/api/bot/queue/set-brawler', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name: name})
    });
    const d = await res.json();
    if (d.ok) {
      document.getElementById('brawlerVal').textContent = name;
    } else {
      alert('Erro: ' + (d.error || 'falha'));
    }
  } catch(e) { alert('Erro: ' + e); }
}

// Phase 1: Toggle system
async function toggleSystem(system, enabled) {
  try {
    const res = await fetch(API + '/api/system/toggle', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({system: system, enabled: enabled})
    });
    const d = await res.json();
    if (!d.ok) console.warn('Toggle failed:', d.error);
  } catch(e) { console.warn('Toggle error:', e); }
}

async function pollSystemStatus() {
  try {
    const res = await fetch(API + '/api/system/status');
    const d = await res.json();
    if (d.systems) {
      const s = d.systems;
      const setToggle = (id, val) => { const el = document.getElementById(id); if (el) el.checked = !!val; };
      setToggle('sysRL', s.rl_engine?.enabled);
      setToggle('sysHuman', s.humanization?.enabled);
      setToggle('sysAntiBan', s.anti_ban?.enabled);
      setToggle('sysErrRec', s.error_recovery?.enabled);
      setToggle('sysRec', s.recording?.enabled);
      setToggle('sysTuner', s.auto_tuner?.enabled);
    }
    // Update pause button from bot status
    const botRes = await fetch(API + '/api/bot/status');
    const botD = await botRes.json();
    const btn = document.getElementById('pauseBtn');
    if (btn && botD.paused !== undefined) {
      btn.textContent = botD.paused ? 'Retomar' : 'Pausar';
      btn.style.background = botD.paused ? '#22c55e' : '#f59e0b';
    }
    // Update brawler dropdown
    if (botD.brawler_queue && botD.brawler_queue.length > 0) {
      const sel = document.getElementById('brawlerSelect');
      const currentVal = sel.value;
      const options = botD.brawler_queue
        .filter(b => b && (b.enabled === undefined || b.enabled))
        .map(b => `<option value="${b.name || b}">${b.name || b}</option>`).join('');
      if (sel.innerHTML.indexOf(options) === -1) {
        sel.innerHTML = '<option value="">— Escolher —</option>' + options;
        sel.value = currentVal;
      }
    }
  } catch(e) {}
}

// Phase 2: Log viewer functions
let _logStreamActive = false;
let _logStreamAbort = null;
let _logLines = [];

function _logColor(level) {
  const colors = { DEBUG: '#64748b', INFO: '#38bdf8', WARNING: '#f59e0b', ERROR: '#ef4444', CRITICAL: '#dc2626' };
  return colors[level] || '#94a3b8';
}

function _renderLogs(lines) {
  const container = document.getElementById('logContainer');
  if (!container) return;
  const autoScroll = document.getElementById('logAutoScroll')?.checked ?? true;
  container.innerHTML = lines.map(l => {
    const ts = new Date(l.timestamp * 1000).toLocaleTimeString();
    const color = _logColor(l.level);
    return `<div style="color:${color};border-bottom:1px solid #1e293b;padding:2px 0">[${ts}] <strong>${l.level}</strong> [${l.logger}] ${l.message}</div>`;
  }).join('');
  if (autoScroll) container.scrollTop = container.scrollHeight;
  document.getElementById('logStats').textContent = `${lines.length} linhas`;
}

async function refreshLogs() {
  try {
    const level = document.getElementById('logLevel')?.value || 'ALL';
    const component = document.getElementById('logComponent')?.value || 'ALL';
    const search = document.getElementById('logSearch')?.value || '';
    const params = new URLSearchParams({ limit: '200', level, component, search });
    const res = await fetch(API + '/api/logs?' + params.toString());
    const d = await res.json();
    _logLines = d.lines || [];
    _renderLogs(_logLines);
  } catch(e) {
    document.getElementById('logContainer').innerHTML = '<div class="label">Erro ao carregar logs</div>';
  }
}

function clearLogs() {
  _logLines = [];
  document.getElementById('logContainer').innerHTML = '<div class="label">Logs limpos</div>';
  document.getElementById('logStats').textContent = '0 linhas';
}

async function toggleLogStream() {
  const btn = document.getElementById('logStreamBtn');
  if (_logStreamActive) {
    _logStreamActive = false;
    if (_logStreamAbort) _logStreamAbort.abort();
    btn.textContent = 'Stream: OFF';
    btn.style.background = '#2563eb';
  } else {
    _logStreamActive = true;
    btn.textContent = 'Stream: ON';
    btn.style.background = '#22c55e';
    _startLogStream();
  }
}

async function _startLogStream() {
  try {
    const controller = new AbortController();
    _logStreamAbort = controller;
    const res = await fetch(API + '/api/logs/stream', { signal: controller.signal });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (_logStreamActive) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\\n\\n');
      buffer = lines.pop() || '';
      for (const chunk of lines) {
        const match = chunk.match(/^data: (.+)$/m);
        if (match) {
          try {
            const data = JSON.parse(match[1]);
            if (data.lines && data.lines.length) {
              _logLines = _logLines.concat(data.lines);
              if (_logLines.length > 500) _logLines = _logLines.slice(-500);
              _renderLogs(_logLines);
            }
          } catch(e) {}
        }
      }
    }
  } catch(e) {
    if (e.name !== 'AbortError') console.warn('Log stream error:', e);
  } finally {
    _logStreamActive = false;
    const btn = document.getElementById('logStreamBtn');
    if (btn) { btn.textContent = 'Stream: OFF'; btn.style.background = '#2563eb'; }
  }
}

async function drawRewardChart() {
  try {
    const res = await fetch(API + '/api/rewards');
    const d = await res.json();
    const pts = (d.rewards || []).slice(-60);
    const canvas = document.getElementById('rewardChart');
    if (!canvas.getContext) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth;
    const h = canvas.height = 200;
    ctx.clearRect(0,0,w,h);
    if (pts.length < 2) return;
    const vals = pts.map(p=>p.r);
    const minV = Math.min(...vals, -1), maxV = Math.max(...vals, 1);
    const range = maxV - minV || 1;
    ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2; ctx.beginPath();
    pts.forEach((p,i)=>{
      const x = (i/(pts.length-1))*w;
      const y = h - ((p.r-minV)/range)*h;
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
    ctx.fillStyle = '#64748b'; ctx.font = '10px sans-serif';
    ctx.fillText('Reward/frame', 4, 12);
  }catch(e){}
}

async function pollRecovery() {
  try {
    const res = await fetch(API + '/api/recovery');
    const d = await res.json();
    const detail = document.getElementById('recoveryDetail');
    if (Object.keys(d).length === 0) {
      detail.textContent = 'Sem dados de recovery disponiveis';
      return;
    }
    detail.innerHTML = Object.entries(d).map(([k,v]) =>
      `<div style="margin-bottom:.5rem"><strong>${k}:</strong> ` +
      `<pre style="margin:0;white-space:pre-wrap">${JSON.stringify(v, null, 2)}</pre></div>`
    ).join('');
  } catch(e){}
}

// Premium: Brawler stats
async function pollBrawlers() {
  try {
    const res = await fetch(API + '/api/brawlers');
    const d = await res.json();
    const list = document.getElementById('brawlerStatsList');
    const brawlers = d.brawlers || [];
    if (brawlers.length === 0) {
      list.innerHTML = '<div class="label">Sem dados de brawlers. Joga partidas para ver stats!</div>';
      return;
    }
    list.innerHTML = brawlers.sort((a,b) => b.matches - a.matches).map(b => {
      const wrColor = b.winrate >= 60 ? '#22c55e' : b.winrate >= 45 ? '#f59e0b' : '#ef4444';
      const trophyProgress = Math.min(100, (b.trophies / Math.max(1, b.target_trophies)) * 100);
      return `<div class="brawler-card">
        <div class="name">${b.name} <span class="badge ${b.winrate>=55?'green':b.winrate>=40?'silver':'red'}">${b.winrate.toFixed(1)}% WR</span></div>
        <div class="stats">
          <span><div class="val">${b.matches}</div>Picks</span>
          <span><div class="val">${b.wins}</div>Wins</span>
          <span><div class="val">${b.losses}</div>Losses</span>
          <span><div class="val" style="color:${wrColor}">${b.winrate.toFixed(1)}%</div>WR</span>
          <span><div class="val">${b.trophies}</div>Trofeus</span>
          <span><div class="val">${b.avg_kills.toFixed(1)}</div>Kills</span>
          <span><div class="val">${b.avg_deaths.toFixed(1)}</div>Deaths</span>
        </div>
        <div class="progress" style="margin-top:.3rem"><div class="progress-bar" style="width:${trophyProgress}%;background:#7c3aed"></div></div>
        <div class="label" style="margin-top:.2rem">Trofeus: ${b.trophies}/${b.target_trophies} | Melhor mapa: ${b.best_map||'—'} | Build: ${b.best_gadget||'—'}</div>
      </div>`;
    }).join('');
  } catch(e){}
}

// Premium: Match analysis
async function pollMatchAnalysis() {
  try {
    const res = await fetch(API + '/api/match-analysis');
    const d = await res.json();
    const analysis = d.match_analysis;
    if (analysis) {
      const score = analysis.score || 0;
      const scoreClass = score >= 70 ? 'high' : score >= 40 ? 'mid' : 'low';
      document.getElementById('analysisScore').textContent = score;
      document.getElementById('analysisScore').className = 'analysis-score ' + scoreClass;
      document.getElementById('analysisResult').textContent = `${analysis.brawler} @ ${analysis.map} — ${analysis.result}`;
      document.getElementById('analysisErrors').innerHTML = (analysis.errors || []).map(e =>
        `<div style="color:#ef4444;margin-bottom:.2rem">- ${e}</div>`
      ).join('') || '<div class="label">Sem erros</div>';
      document.getElementById('analysisStrengths').innerHTML = (analysis.strengths || []).map(s =>
        `<div style="color:#22c55e;margin-bottom:.2rem">+ ${s}</div>`
      ).join('') || '<div class="label">Sem pontos fortes</div>';
      document.getElementById('matchupAnalysis').textContent = analysis.matchup_analysis || '—';
      document.getElementById('buildSuggestion').textContent = analysis.build_suggestion || '—';
      document.getElementById('positioningTip').textContent = analysis.positioning_tip || '—';
    }
    // Coach tips
    const tips = d.coach_tips || [];
    document.getElementById('coachTipsList').innerHTML = tips.map(t =>
      `<div class="coach-tip">${t}</div>`
    ).join('') || '<div class="coach-tip">Joga mais partidas para receber dicas</div>';
  } catch(e){}
}

// Premium: AI Coach
async function pollAICoach() {
  try {
    const res = await fetch(API + '/api/ai-pick');
    const d = await res.json();
    const s = d.suggestion;
    if (s) {
      document.getElementById('coachPickBrawler').textContent = s.brawler || '—';
      document.getElementById('coachPickMap').textContent = s.map || d.map || '—';
      document.getElementById('coachPickConf').textContent = ((s.confidence||0)*100).toFixed(0) + '%';
      document.getElementById('coachPickReason').textContent = s.reason || '';
      document.getElementById('coachPickAlts').textContent = (s.alternatives || []).join(', ') || '—';
    }
    const wp = d.win_prediction || 0;
    document.getElementById('coachWinPred').textContent = (wp*100).toFixed(0) + '%';
    document.getElementById('coachWinPred').style.color = wp > 0.6 ? '#22c55e' : wp > 0.4 ? '#f59e0b' : '#ef4444';
    document.getElementById('coachWinBar').style.width = (wp*100) + '%';
    document.getElementById('coachWinBar').style.background = wp > 0.6 ? '#22c55e' : wp > 0.4 ? '#f59e0b' : '#ef4444';
    document.getElementById('coachWinText').textContent = (wp*100).toFixed(0) + '%';
  } catch(e){}
}

// Premium: Trophy history chart
async function drawTrophyChart() {
  try {
    const res = await fetch(API + '/api/trophy-history');
    const d = await res.json();
    const history = d.history || [];
    const canvas = document.getElementById('trophyChart');
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth;
    const h = canvas.height = 250;
    ctx.clearRect(0,0,w,h);
    if (history.length < 2) {
      ctx.fillStyle = '#64748b'; ctx.font = '12px sans-serif';
      ctx.fillText('Sem dados suficientes para grafico', w/2 - 120, h/2);
      return;
    }
    const vals = history.map(p => p.total_trophies);
    const minV = Math.min(...vals) - 10;
    const maxV = Math.max(...vals) + 10;
    const range = maxV - minV || 1;
    // Grid lines
    ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = (i/4) * h;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      ctx.fillStyle = '#64748b'; ctx.font = '10px sans-serif';
      ctx.fillText(Math.round(maxV - (i/4)*range), 4, y + 12);
    }
    // Trophy line
    ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 2; ctx.beginPath();
    history.forEach((p, i) => {
      const x = (i/(history.length-1))*w;
      const y = h - ((p.total_trophies - minV)/range)*h;
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
    // Fill under line
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = 'rgba(124,58,237,0.1)'; ctx.fill();
    // Daily evolution table
    const daily = d.daily_evolution || [];
    const table = document.getElementById('dailyEvolutionTable');
    if (daily.length > 0) {
      table.innerHTML = '<table><thead><tr><th>Data</th><th>Trofeus</th><th>+/-</th></tr></thead><tbody>' +
        daily.slice().reverse().map(d => {
          const changeColor = d.change > 0 ? '#22c55e' : d.change < 0 ? '#ef4444' : '#94a3b8';
          const changeStr = d.change > 0 ? '+' + d.change : d.change;
          return `<tr><td>${d.date}</td><td>${d.trophies}</td><td style="color:${changeColor}">${changeStr}</td></tr>`;
        }).join('') + '</tbody></table>';
    }
  } catch(e){}
}

// Premium: Weekly progress
async function pollWeekly() {
  try {
    const res = await fetch(API + '/api/weekly-progress');
    const d = await res.json();
    const change = d.trophies_change || 0;
    const el = document.getElementById('weeklyTrophies');
    el.textContent = (change > 0 ? '+' : '') + change;
    el.style.color = change > 0 ? '#22c55e' : change < 0 ? '#ef4444' : '#94a3b8';
    document.getElementById('weeklyMatches').textContent = d.matches || 0;
    // Also update trophy tab totals
    const live = await (await fetch(API + '/api/live')).json();
    document.getElementById('trophyTotalVal').textContent = live.total_trophies || 0;
    document.getElementById('trophyUnlockedVal').textContent = live.unlocked_brawlers || 0;
  } catch(e){}
}

// Phase 3: Notifications
async function pollNotifications() {
  try {
    const res = await fetch(API + '/api/notifications/history');
    const d = await res.json();
    const container = document.getElementById('notifHistory');
    if (container) {
      const items = d.history || [];
      container.innerHTML = items.slice().reverse().map(n => {
        const color = n.level === 'error' ? '#ef4444' : n.level === 'warning' ? '#f59e0b' : '#38bdf8';
        const ts = new Date(n.timestamp * 1000).toLocaleTimeString();
        return `<div style="border-bottom:1px solid #334155;padding:.3rem 0"><span style="color:${color}"><strong>[${n.level.toUpperCase()}]</strong></span> <span style="color:#64748b">${ts}</span> ${n.title}: ${n.message}</div>`;
      }).join('') || '<div class="label">Sem notificacoes</div>';
    }
  } catch(e){}
}

async function saveNotifConfig() {
  try {
    const config = {
      webhook_url: document.getElementById('notifWebhook').value,
      browser_enabled: document.getElementById('notifBrowser').checked,
      desktop_enabled: document.getElementById('notifDesktop').checked,
      on_crash: document.getElementById('notifCrash').checked,
      on_consecutive_losses: document.getElementById('notifLosses').checked ? 3 : 0,
      on_trophy_limit: document.getElementById('notifTrophy').checked,
    };
    const res = await fetch(API + '/api/notifications/config', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(config)
    });
    const d = await res.json();
    alert(d.ok ? 'Configuracao guardada!' : 'Erro: ' + (d.error || 'falha'));
  } catch(e) { alert('Erro: ' + e); }
}

async function testNotification() {
  try {
    const res = await fetch(API + '/api/notifications/test', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({title:'Teste', message:'Notificacao de teste'})
    });
    const d = await res.json();
    alert(d.ok ? 'Notificacao de teste enviada!' : 'Erro: ' + (d.error || 'falha'));
  } catch(e) { alert('Erro: ' + e); }
}

// Phase 4: Config editor
async function loadConfig() {
  try {
    const res = await fetch(API + '/api/config');
    const d = await res.json();
    document.getElementById('configEditor').value = JSON.stringify(d, null, 2);
    document.getElementById('configStatus').textContent = 'Configuracao carregada.';
  } catch(e) { document.getElementById('configStatus').textContent = 'Erro ao carregar: ' + e; }
}

async function saveConfig() {
  try {
    const raw = document.getElementById('configEditor').value;
    const data = JSON.parse(raw);
    const res = await fetch(API + '/api/config', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)
    });
    const d = await res.json();
    document.getElementById('configStatus').textContent = d.ok ? 'Configuracao guardada!' : 'Erro: ' + (d.error || 'falha');
  } catch(e) { document.getElementById('configStatus').textContent = 'Erro: ' + e; }
}

function resetConfig() {
  document.getElementById('configEditor').value = '{}';
  document.getElementById('configStatus').textContent = 'Resetado (vazio). Carregue para restaurar.';
}

// Phase 6: Anti-Ban
async function pollAntiBan() {
  try {
    const res = await fetch(API + '/api/antiban/status');
    const d = await res.json();
    document.getElementById('abStatusVal').textContent = d.enabled ? 'Ativo' : 'Inativo';
    document.getElementById('abWinTarget').textContent = (d.win_rate_target || 0.5) * 100 + '%';
    document.getElementById('abWinCurrent').textContent = ((d.current_win_rate || 0) * 100).toFixed(1) + '%';
    document.getElementById('abThrottle').textContent = d.throttling ? 'Sim' : 'Nao';
    document.getElementById('abNextGame').textContent = d.next_game_time || '—';
    document.getElementById('abRandom').textContent = d.schedule_randomized ? 'Sim' : 'Nao';
    document.getElementById('abMissclicks').textContent = d.missclicks || 0;
    document.getElementById('abDelayNoise').textContent = d.delay_noise_applied || 0;
    document.getElementById('abFingerprint').textContent = (d.fingerprint || '').slice(0, 20) + '...';
    const patterns = d.patterns_detected || [];
    document.getElementById('abPatterns').innerHTML = patterns.length ? patterns.map(p => `<div style="color:#f59e0b">- ${p}</div>`).join('') : '<div class="label">Sem padroes detetados</div>';
  } catch(e){}
}

// Learning Mode Dashboard Functions
let _lmHistory = [];

async function pollLearningMode() {
  try {
    const res = await fetch(API + '/api/learning-mode/status');
    const d = await res.json();
    const active = d.active || false;
    document.getElementById('lmStatusVal').textContent = active ? 'ATIVO' : 'Inativo';
    document.getElementById('lmStatusVal').style.color = active ? '#22c55e' : '#94a3b8';
    document.getElementById('lmMatchVal').textContent = d.current_match || 0;
    document.getElementById('lmMaxVal').textContent = d.max_matches || 0;
    document.getElementById('lmKillsVal').textContent = d.kills || 0;
    document.getElementById('lmDeathsVal').textContent = d.deaths || 0;
    document.getElementById('lmDetectVal').textContent = d.detections_enemies || 0;
    document.getElementById('lmPlayerVal').textContent = d.detections_player || 0;
    document.getElementById('lmAccuracyVal').textContent = (d.accuracy_percent || 0).toFixed(1) + '%';
    document.getElementById('lmDamageVal').textContent = (d.damage_dealt || 0).toFixed(0);
    document.getElementById('lmSurvivalVal').textContent = (d.match_duration_seconds || 0).toFixed(0) + 's';
    document.getElementById('lmBrawlerVal').textContent = d.current_brawler || '—';

    // Atualizar gráfico de detecções
    drawLearningDetectChart(d.frames_history || []);
  } catch(e) {}
}

async function pollLearningHistory() {
  try {
    const res = await fetch(API + '/api/learning-mode/history');
    const d = await res.json();
    const matches = d.matches || [];
    _lmHistory = matches;
    // Tabela
    const tbody = document.getElementById('lmHistoryTable');
    tbody.innerHTML = matches.slice().reverse().map(m =>
      `<tr><td>${m.brawler || '—'}</td><td>${m.result || '—'}</td><td>${m.kills || 0}</td><td>${m.deaths || 0}</td><td>${(m.duration_seconds || 0).toFixed(0)}s</td></tr>`
    ).join('') || '<tr><td colspan="5">Sem dados</td></tr>';
    // Gráfico kills
    drawLearningKillsChart(matches);
  } catch(e) {}
}

function drawLearningDetectChart(frames) {
  try {
    const canvas = document.getElementById('lmDetectChart');
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth;
    const h = canvas.height = 200;
    ctx.clearRect(0, 0, w, h);
    if (frames.length < 2) {
      ctx.fillStyle = '#64748b'; ctx.font = '12px sans-serif';
      ctx.fillText('A aguardar dados...', w/2 - 60, h/2);
      return;
    }
    const vals = frames.map(f => f.enemies_detected || 0);
    const maxV = Math.max(...vals, 1);
    ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 2; ctx.beginPath();
    frames.forEach((f, i) => {
      const x = (i / (frames.length - 1)) * w;
      const y = h - ((f.enemies_detected || 0) / maxV) * h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = '#64748b'; ctx.font = '10px sans-serif';
    ctx.fillText('Inimigos detetados / frame', 4, 12);
  } catch(e) {}
}

function drawLearningKillsChart(matches) {
  try {
    const canvas = document.getElementById('lmKillsChart');
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth;
    const h = canvas.height = 200;
    ctx.clearRect(0, 0, w, h);
    if (matches.length < 1) {
      ctx.fillStyle = '#64748b'; ctx.font = '12px sans-serif';
      ctx.fillText('Sem partidas registadas', w/2 - 70, h/2);
      return;
    }
    const barW = Math.max(20, (w / matches.length) * 0.7);
    const spacing = w / matches.length;
    const maxKills = Math.max(...matches.map(m => m.kills || 0), 1);
    matches.forEach((m, i) => {
      const kills = m.kills || 0;
      const barH = (kills / maxKills) * (h - 30);
      const x = i * spacing + (spacing - barW) / 2;
      const y = h - barH - 20;
      ctx.fillStyle = kills > 0 ? '#22c55e' : '#334155';
      ctx.fillRect(x, y, barW, barH);
      ctx.fillStyle = '#94a3b8'; ctx.font = '10px sans-serif';
      ctx.fillText(kills.toString(), x + barW/2 - 4, y - 4);
      ctx.fillStyle = '#64748b'; ctx.font = '9px sans-serif';
      ctx.fillText((m.brawler || '').slice(0,4), x + 2, h - 4);
    });
  } catch(e) {}
}

async function startLearningMode() {
  try {
    const res = await fetch(API + '/api/learning-mode/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({max_matches: 5})
    });
    const d = await res.json();
    if (d.ok) {
      toast('Modo Teste iniciado!', 'success');
      pollLearningMode();
    } else {
      toast('Erro: ' + (d.error || 'falha'), 'error');
    }
  } catch(e) { toast('Erro: ' + e, 'error'); }
}

async function stopLearningMode() {
  if (!confirm('Parar o Modo Teste?')) return;
  try {
    const res = await fetch(API + '/api/learning-mode/stop', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: '{}'
    });
    const d = await res.json();
    if (d.ok) {
      toast('Modo Teste parado!', 'info');
      pollLearningMode();
    } else {
      toast('Erro: ' + (d.error || 'falha'), 'error');
    }
  } catch(e) { toast('Erro: ' + e, 'error'); }
}

// Phase 5: Analytics
function showAnalyticsTab(id) {
  ['maps','perf','rl','sessions'].forEach(k => {
    document.getElementById('analytics-'+k).style.display = k === id ? 'block' : 'none';
  });
}

async function pollAnalytics() {
  try {
    const live = await (await fetch(API + '/api/live')).json();
    // Maps tab: use brawler stats
    const maps = await (await fetch(API + '/api/brawlers')).json();
    if (maps.brawlers) {
      const html = maps.brawlers.slice(0,10).map(b => {
        return `<div class="row"><span>${b.name}</span><span>${b.winrate.toFixed(1)}% WR (${b.matches} matches)</span></div>`;
      }).join('');
      document.getElementById('analyticsMapsContent').innerHTML = html || '<div class="label">Sem dados</div>';
    }
    // RL tab
    document.getElementById('rlEpsilonVal').innerHTML = 'Epsilon: <span class="metric small">' + (live.epsilon || 0).toFixed(3) + '</span>';
    document.getElementById('rlStatesVal').innerHTML = 'Q-States: <span class="metric small">' + (live.q_states || 0) + '</span>';
  } catch(e){}
}

// Phase 1: Brawler Queue UI
async function pollQueue() {
  try {
    const res = await fetch(API + '/api/bot/queue');
    const d = await res.json();
    const container = document.getElementById('brawlerQueueList');
    if (!container) return;
    const items = d.queue || [];
    if (items.length === 0) {
      container.innerHTML = '<div class="label">Fila vazia. Adicione brawlers abaixo.</div>';
      return;
    }
    container.innerHTML = items.map((b, idx) => {
      const isCurrent = b.current ? '<span class="badge green">ATIVO</span>' : '';
      const trophyProg = Math.min(100, (b.current_trophies / Math.max(1, b.target_trophies)) * 100);
      return `<div class="queue-item" style="${b.current ? 'border:1px solid #22c55e' : ''}">
        <span class="name">${b.name} ${isCurrent}</span>
        <span class="label">${b.current_trophies}/${b.target_trophies} trofeus</span>
        <span class="label">P:${b.priority}</span>
        <span class="btn-sm" onclick="moveQueueItem(${idx},-1)" ${idx===0?'style="visibility:hidden"':''}>&uarr;</span>
        <span class="btn-sm" onclick="moveQueueItem(${idx},1)" ${idx===items.length-1?'style="visibility:hidden"':''}>&darr;</span>
        <span class="btn-sm danger" onclick="removeBrawlerFromQueue(${idx})">x</span>
      </div>`;
    }).join('');
  } catch(e) {}
}

async function addBrawlerToQueue() {
  const name = document.getElementById('newBrawlerName').value.trim();
  const target = parseInt(document.getElementById('newBrawlerTarget').value) || 350;
  const priority = parseInt(document.getElementById('newBrawlerPriority').value) || 1;
  if (!name) { alert('Insira um nome de brawler'); return; }
  try {
    const res = await fetch(API + '/api/bot/queue/add', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name, target_trophies: target, priority})
    });
    const d = await res.json();
    if (d.ok) {
      document.getElementById('newBrawlerName').value = '';
      pollQueue();
      toast(`${name} adicionado a fila!`, 'success');
    } else {
      alert('Erro: ' + (d.error || 'falha'));
    }
  } catch(e) { alert('Erro: ' + e); }
}

async function removeBrawlerFromQueue(index) {
  try {
    const res = await fetch(API + '/api/bot/queue/remove', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({index})
    });
    const d = await res.json();
    if (d.ok) { pollQueue(); toast('Brawler removido', 'info'); }
  } catch(e) {}
}

async function moveQueueItem(index, direction) {
  try {
    const res = await fetch(API + '/api/bot/queue');
    const d = await res.json();
    const queue = d.queue || [];
    const newIndex = index + direction;
    if (newIndex < 0 || newIndex >= queue.length) return;
    const temp = queue[index];
    queue[index] = queue[newIndex];
    queue[newIndex] = temp;
    // Rebuild queue data for update
    const updateRes = await fetch(API + '/api/bot/queue/update', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({queue: queue.map(b => ({
        name: b.name, current_trophies: b.current_trophies,
        target_trophies: b.target_trophies, priority: b.priority, enabled: b.enabled
      }))})
    });
    if (updateRes.ok) pollQueue();
  } catch(e) {}
}

async function clearQueue() {
  if (!confirm('Limpar toda a fila de brawlers?')) return;
  try {
    const res = await fetch(API + '/api/bot/queue/update', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({queue: []})
    });
    if (res.ok) { pollQueue(); toast('Fila limpa', 'info'); }
  } catch(e) {}
}

// Health Monitor
async function pollHealth() {
  try {
    const res = await fetch(API + '/api/bot/status');
    const d = await res.json();
    const setHealth = (id, status, text) => {
      const el = document.getElementById(id);
      if (!el) return;
      const dot = el.parentElement.querySelector('.health-dot');
      el.textContent = text;
      if (dot) {
        dot.className = 'health-dot ' + (status === 'ok' ? 'health-online' : status === 'warn' ? 'health-warn' : 'health-offline');
      }
    };
    setHealth('healthYOLO', d.models_loaded ? 'ok' : 'error', d.models_loaded ? 'Carregado' : 'Nao carregado');
    setHealth('healthADB', d.emulator_controller_active ? 'ok' : 'error', d.emulator_controller_active ? 'Conectado' : 'Desconectado');
    setHealth('healthOCR', d.ocr_detector?.reader_available ? 'ok' : 'warn', d.ocr_detector?.reader_available ? 'Disponivel' : 'Indisponivel');
    setHealth('healthState', d.running ? 'ok' : 'warn', d.current_state || 'unknown');
    setHealth('healthRL', d.systems?.rl_engine?.enabled ? 'ok' : 'warn', d.systems?.rl_engine?.enabled ? 'Ativo' : 'Inativo');
    setHealth('healthAntiBan', d.systems?.anti_ban?.enabled ? 'ok' : 'warn', d.systems?.anti_ban?.enabled ? 'Ativo' : 'Inativo');
    setHealth('healthEmulator', d.window_active ? 'ok' : 'warn', d.window_active ? 'Janela ativa' : 'Janela inativa');
    document.getElementById('healthLastAction').textContent = d.session_duration ? (d.session_duration / 60).toFixed(1) + ' min' : '—';
  } catch(e) {}
}

// Combat params
async function updateCombatParam(param, value) {
  try {
    const res = await fetch(API + '/api/bot/combat/param', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({param, value: parseFloat(value)})
    });
    const d = await res.json();
    if (d.ok) {
      if (param === 'aggressiveness') document.getElementById('aggVal').textContent = Math.round(value*100) + '%';
      if (param === 'shot_cooldown') document.getElementById('cdVal').textContent = value;
      if (param === 'attack_distance') document.getElementById('distVal').textContent = value;
      toast(param + ' atualizado: ' + value, 'success', 1500);
    }
  } catch(e) { console.warn('Combat param error:', e); }
}

// Phase 7: Export stats
async function exportStats() {
  try {
    const res = await fetch(API + '/api/export/stats');
    const d = await res.json();
    const blob = new Blob([JSON.stringify(d, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `soberana_stats_${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast('Estatisticas exportadas!', 'success');
  } catch(e) { toast('Erro ao exportar', 'error'); }
}

// Phase 7: Dark/Light mode
let _darkMode = true;
function toggleDarkMode() {
  _darkMode = !_darkMode;
  const btn = document.getElementById('darkModeBtn');
  const root = document.documentElement;
  if (_darkMode) {
    root.style.setProperty('--bg', '#0f172a');
    root.style.setProperty('--card-bg', '#1e293b');
    root.style.setProperty('--text', '#e2e8f0');
    root.style.setProperty('--text-muted', '#94a3b8');
    root.style.setProperty('--border', '#334155');
    root.style.setProperty('--input-bg', '#0f172a');
    document.body.style.background = '#0f172a';
    document.body.style.color = '#e2e8f0';
    if (btn) btn.textContent = 'Light';
  } else {
    root.style.setProperty('--bg', '#f8fafc');
    root.style.setProperty('--card-bg', '#ffffff');
    root.style.setProperty('--text', '#1e293b');
    root.style.setProperty('--text-muted', '#64748b');
    root.style.setProperty('--border', '#e2e8f0');
    root.style.setProperty('--input-bg', '#f1f5f9');
    document.body.style.background = '#f8fafc';
    document.body.style.color = '#1e293b';
    if (btn) btn.textContent = 'Dark';
  }
}

// Phase 7: UX Polish
// Toast notifications
function toast(message, type='info', duration=3000) {
  const container = document.getElementById('toastContainer') || (() => {
    const el = document.createElement('div');
    el.id = 'toastContainer';
    el.style.cssText = 'position:fixed;bottom:1rem;right:1rem;z-index:9999;display:flex;flex-direction:column;gap:.5rem';
    document.body.appendChild(el);
    return el;
  })();
  const toast = document.createElement('div');
  const colors = { info:'#2563eb', success:'#22c55e', warning:'#f59e0b', error:'#ef4444' };
  toast.style.cssText = `background:${colors[type]||colors.info};color:#fff;padding:.6rem 1rem;border-radius:4px;font-size:.85rem;box-shadow:0 4px 12px rgba(0,0,0,.3);animation:fadeIn .3s;max-width:300px`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => { toast.style.opacity='0'; toast.style.transition='opacity .3s'; setTimeout(()=>toast.remove(),300); }, duration);
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'p' || e.key === 'P') botPauseToggle();
  if (e.key === 's' || e.key === 'S') botStart();
  if (e.key === 'r' || e.key === 'R') { e.preventDefault(); botRestart(); }
  if (e.key === 'l' || e.key === 'L') showTab('logs');
});

// Auto-reconnect indicator
let _lastPollSuccess = true;
let _reconnectAttempts = 0;
const originalPoll = poll;
poll = async function() {
  try {
    await originalPoll();
    if (!_lastPollSuccess) {
      _lastPollSuccess = true;
      _reconnectAttempts = 0;
      toast('Dashboard reconectada!', 'success', 2000);
    }
  } catch(e) {
    _lastPollSuccess = false;
    _reconnectAttempts++;
    const status = document.getElementById('connStatus');
    if (status) status.innerHTML = '<span class="status-dot status-offline"></span>Offline (' + _reconnectAttempts + ')';
  }
};

// Mode Control Center Functions
async function startFarmMode() { try { const res=await fetch(API+'/api/mode/farm/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({config:{}})}); const d=await res.json(); toast(d.status==='started'?'Farm iniciado':'Falha ao iniciar farm',d.ok?'success':'error'); } catch(e){ toast('Erro: '+e,'error'); } }
async function stopFarmMode() { try { const res=await fetch(API+'/api/mode/farm/stop',{method:'POST',headers:{'Content-Type':'application/json'}}); const d=await res.json(); toast(d.status==='stopped'?'Farm parado':'Falha','info'); } catch(e){} }
async function startLearnMode() { try { const res=await fetch(API+'/api/mode/learn/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({config:{}})}); const d=await res.json(); toast(d.status==='started'?'Aprender iniciado':'Falha',d.ok?'success':'error'); } catch(e){ toast('Erro: '+e,'error'); } }
async function stopLearnMode() { try { const res=await fetch(API+'/api/mode/learn/stop',{method:'POST',headers:{'Content-Type':'application/json'}}); const d=await res.json(); toast(d.status==='stopped'?'Aprender parado':'Falha','info'); } catch(e){} }
async function toggleESP(force) { try { const enabled = force !== undefined ? force : true; const res=await fetch(API+'/api/esp/toggle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled})}); const d=await res.json(); toast('ESP '+d.status,d.ok?'success':'error'); } catch(e){} }

async function pollModeStatus() {
  try {
    const res=await fetch(API+'/api/mode/status'); const d=await res.json();
    // Farm
    const farmActive = d.active_mode==='farm';
    document.getElementById('farmStatusVal').textContent = farmActive?'ATIVO':'Inativo';
    document.getElementById('farmStatusVal').style.color = farmActive?'#22c55e':'#94a3b8';
    document.getElementById('farmMatchVal').textContent = d.matches_completed||0;
    document.getElementById('farmTargetVal').textContent = d.matches_target||0;
    document.getElementById('farmBrawlerVal').textContent = d.current_brawler||'—';
    // Learn
    const learnActive = d.active_mode==='learn';
    document.getElementById('learnStatusVal').textContent = learnActive?'ATIVO':'Inativo';
    document.getElementById('learnStatusVal').style.color = learnActive?'#7c3aed':'#94a3b8';
  } catch(e){}
}

async function pollRLMetrics() {
  try {
    const res=await fetch(API+'/api/rl/metrics'); const d=await res.json();
    document.getElementById('learnEngineVal').textContent = d.engine_type||'—';
    document.getElementById('learnQTableVal').textContent = d.q_table_size||0;
    document.getElementById('learnEpsilonVal').textContent = (d.epsilon||0).toFixed(3);
    document.getElementById('learnRewardVal').textContent = (d.last_reward||0).toFixed(2)+' / '+(d.episode_reward||0).toFixed(2);
    document.getElementById('learnPpoLossVal').textContent = (d.policy_loss||0).toFixed(4)+' / '+(d.value_loss||0).toFixed(4);
    document.getElementById('learnBufferVal').textContent = (d.buffer_size||0)+' / '+(d.buffer_capacity||0);
    document.getElementById('learnActionVal').textContent = d.last_action||'—';
  } catch(e){}
}

async function pollDetections() {
  try {
    const res=await fetch(API+'/api/detections/live'); const d=await res.json();
    const detections = d.detections||[];
    // ESP
    document.getElementById('espFpsVal').textContent = (d.fps||0).toFixed(1);
    document.getElementById('espObjectsVal').textContent = detections.length;
    // Table
    const tbody = document.getElementById('detectionsTable');
    if (!detections.length) { tbody.innerHTML='<tr><td colspan="6">Sem dados</td></tr>'; }
    else { tbody.innerHTML=detections.slice(0,20).map(det=>`<tr><td>${det.class_name}</td><td>${(det.confidence||0).toFixed(2)}</td><td>${det.x}</td><td>${det.y}</td><td>${det.width}</td><td>${det.height}</td></tr>`).join(''); }
    // Counters
    const counts={}; detections.forEach(d=>counts[d.class_name]=(counts[d.class_name]||0)+1);
    document.getElementById('detEnemyVal').textContent = counts['enemy']||0;
    document.getElementById('detTeamVal').textContent = counts['teammate']||0;
    document.getElementById('detWallVal').textContent = counts['wall']||0;
    document.getElementById('detBushVal').textContent = counts['bush']||0;
    document.getElementById('detPowerVal').textContent = counts['powerup']||0;
  } catch(e){}
}

async function pollVisionStats() {
  try {
    const res=await fetch(API+'/api/vision/stats'); const d=await res.json();
    document.getElementById('detModelVal').textContent = (d.device||'—')+' / '+(d.models_loaded||0)+' modelos';
    const statsDiv = document.getElementById('visionStats');
    statsDiv.innerHTML = '<div class="row"><span>Initialized</span><span>'+d.initialized+'</span></div>'
      +'<div class="row"><span>Frame count</span><span>'+(d.frame_count||0)+'</span></div>'
      +'<div class="row"><span>Avg confidence</span><span>'+((d.avg_confidence||0).toFixed(3))+'</span></div>'
      +'<div class="row"><span>Loaded classes</span><span>'+(d.loaded_classes?d.loaded_classes.join(', '):'—')+'</span></div>';
  } catch(e){}
}

// CSS animation for toasts
const toastStyle = document.createElement('style');
toastStyle.textContent = '@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}';
document.head.appendChild(toastStyle);


async function refreshTrainingModels() {
  try {
    const res = await fetch(API + '/api/training/models');
    const d = await res.json();
    const tbody = document.getElementById('modelsTable');
    const models = d.models || [];
    if (!models.length) {
      tbody.innerHTML = '<tr><td colspan="5">Nenhum modelo registado. Execute <code>python train.py</code></td></tr>';
    } else {
      tbody.innerHTML = models.map(m =>
        '<tr><td>' + (m.name || '—') + '</td><td>' + (m.version || '—') + '</td><td>' + (m.schema || '—') + '</td><td>' + ((m.map50 || 0).toFixed(3)) + '</td><td>' + (m.is_active ? '<span style="color:#22c55e">Ativo</span>' : '<span style="color:#64748b">Inativo</span>') + '</td></tr>'
      ).join('');
    }
    // Dataset
    document.getElementById('dsImages').textContent = d.dataset?.total_images || '—';
    document.getElementById('dsBoxes').textContent = d.dataset?.total_boxes || '—';
    document.getElementById('dsClasses').textContent = d.dataset?.num_classes || '—';
    const dist = d.dataset?.class_distribution || {};
    document.getElementById('dsClassDist').innerHTML = Object.entries(dist).map(([k, v]) =>
      '<div class="row"><span>' + k + '</span><span>' + v + '</span></div>'
    ).join('') || '—';
    // Last training
    const lt = d.last_training;
    if (lt) {
      document.getElementById('lastTraining').innerHTML =
        '<div class="row"><span>Run ID</span><span>' + (lt.run_id || '—') + '</span></div>' +
        '<div class="row"><span>Data</span><span>' + (lt.timestamp || '—') + '</span></div>' +
        '<div class="row"><span>Schema</span><span>' + (lt.schema || '—') + '</span></div>' +
        '<div class="row"><span>mAP50</span><span>' + ((lt.map50 || 0).toFixed(4)) + '</span></div>' +
        '<div class="row"><span>mAP50-95</span><span>' + ((lt.map50_95 || 0).toFixed(4)) + '</span></div>' +
        '<div class="row"><span>Duracao</span><span>' + (lt.duration_seconds || 0).toFixed(0) + 's</span></div>';
    }
  } catch(e) {}
}
async function rescanModels() {
  try {
    const res = await fetch(API + '/api/training/models?scan=1');
    const d = await res.json();
    toast(d.scanned + ' modelos encontrados', 'info');
    refreshTrainingModels();
  } catch(e) { toast('Erro ao scan', 'error'); }
}

setInterval(poll, 2000);
setInterval(pollReplays, 5000);
setInterval(pollAB, 5000);
setInterval(pollRecovery, 5000);
setInterval(drawRewardChart, 3000);
setInterval(pollBrawlers, 5000);
setInterval(pollMatchAnalysis, 5000);
setInterval(pollAICoach, 5000);
setInterval(drawTrophyChart, 5000);
setInterval(pollWeekly, 10000);
setInterval(pollSystemStatus, 3000);
setInterval(refreshLogs, 5000);
setInterval(pollNotifications, 10000);
setInterval(pollAntiBan, 10000);
setInterval(pollAnalytics, 10000);
setInterval(pollQueue, 5000);
setInterval(pollHealth, 3000);
setInterval(pollLearningMode, 2000);
setInterval(pollLearningHistory, 10000);
setInterval(pollModeStatus, 2000);
setInterval(pollRLMetrics, 2000);
setInterval(pollDetections, 1000);
setInterval(pollVisionStats, 5000);
setInterval(refreshTrainingModels, 15000);
poll(); pollReplays(); pollAB(); pollRecovery(); drawRewardChart();
pollBrawlers(); pollMatchAnalysis(); pollAICoach(); drawTrophyChart(); pollWeekly(); pollSystemStatus(); refreshLogs(); pollNotifications(); pollAntiBan(); pollAnalytics(); pollQueue(); pollHealth(); pollLearningMode(); pollLearningHistory();
pollModeStatus(); pollRLMetrics(); pollDetections(); pollVisionStats(); refreshTrainingModels();
</script>
</body>
</html>
'''

# ---------------------------------------------------------------------------
# DASHBOARD SERVER (orquestrador)
# ---------------------------------------------------------------------------

class DashboardServer:
    """
    Servidor dashboard completo:
    - HTTP server na thread principal ou daemon
    - DataBridge alimentado pelo wrapper
    - ReplayRecorder para gravar partidas
    - ABTestManager para comparar estrategias
    """

    DEFAULT_PORT = 8765

    def __init__(self, port: int = DEFAULT_PORT, data_dir: Path = Path("data")):
        self.port = port
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = DashboardDataBridge()
        self.recorder = ReplayRecorder(self.data_dir / "replays")
        self.ab_test = ABTestManager(self.data_dir / "ab_tests.json")

        # Phase 2: LogBuffer
        self.log_buffer: Optional[LogBuffer] = None
        if HAS_LOGBUFFER:
            # Use singleton so endpoints read the same buffer the handler writes to
            self.log_buffer = get_log_buffer(max_lines=500)
            try:
                install_log_buffer("", max_lines=500)
                logger.info("[DASHBOARD] LogBuffer instalado no root logger")
            except Exception as e:
                logger.warning(f"[DASHBOARD] Falha ao instalar LogBuffer: {e}")

        # Phase 3: Notifications
        self.notification_manager: Optional[Any] = None
        if HAS_NOTIFICATIONS:
            self.notification_manager = get_notification_manager()

        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._wrapper_ref: Optional[Any] = None  # Set by wrapper for bot control

    def start(self, daemon: bool = True):
        """Inicia servidor HTTP em thread separada."""
        if self._running:
            return
        if ThreadingHTTPServer is None:
            logger.warning("[DASHBOARD] http.server nao disponivel")
            return

        DashboardHandler.bridge = self.bridge
        DashboardHandler.recorder = self.recorder
        DashboardHandler.ab_test = self.ab_test
        DashboardHandler.wrapper_ref = self._wrapper_ref
        DashboardHandler.log_buffer = self.log_buffer
        DashboardHandler.notification_manager = self.notification_manager

        try:
            self._server = ThreadingHTTPServer(("0.0.0.0", self.port), DashboardHandler)
        except OSError as e:
            logger.warning(f"[DASHBOARD] Porta {self.port} ocupada: {e}")
            return

        self._running = True
        self._thread = threading.Thread(target=self._serve_loop, daemon=daemon)
        self._thread.start()
        logger.info(f"[DASHBOARD] Servidor iniciado em http://localhost:{self.port}")

    def _serve_loop(self):
        try:
            self._server.serve_forever(poll_interval=0.5)
        except Exception as e:
            logger.debug(f"[DASHBOARD] serve loop stopped: {e}")

    def stop(self):
        self._running = False
        if self._server:
            try:
                self._server.shutdown()
            except Exception as e:
                logger.debug(f"[DASHBOARD] Premium stats unavailable: {e}")
        logger.info("[DASHBOARD] Servidor parado")

    def update_live_data(self, **kwargs):
        """Wrapper chama isto para atualizar dados em tempo real."""
        self.bridge.update(**kwargs)

    def update_from_wrapper(self, wrapper_instance):
        """Wrapper chama isto periodicamente para sincronizar tudo."""
        self.bridge.update_from_wrapper(wrapper_instance)

    def record_replay_frame(self, screenshot, state: str, action: str, **kwargs):
        self.recorder.record_frame(screenshot, state, action, **kwargs)

    def get_ab_variant(self) -> str:
        """Retorna variante A/B para a proxima partida."""
        return self.ab_test.next_match_variant()

    def record_ab_result(self, variant: str, result: str, reward: float = 0.0):
        self.ab_test.record_result(variant, result, reward)

    def set_wrapper(self, wrapper_instance):
        """Set reference to wrapper for bot control from dashboard."""
        self._wrapper_ref = wrapper_instance
        DashboardHandler.wrapper_ref = wrapper_instance
