"""AI Think Dashboard - Comprehensive AI Thinking Overlay with Full Game State"""

import time
import threading
from typing import Callable, Optional, Dict, Any, List
from collections import deque
import logging

logger = logging.getLogger("ai_think_dashboard")


class DiagnosticOverlay:
    """Lightweight diagnostic overlay that formats a status dict into text lines.

    This class existed in an earlier version of the file and is preserved here
    as a compatibility shim so tests that import it continue to work.
    """

    def __init__(self, status_provider: Callable[[], dict] = None):
        self.status_provider = status_provider

    @staticmethod
    def format_status(status: dict) -> list:
        """Build the overlay text lines from a status dictionary."""
        diagnostics = status.get("diagnostics", {}) if isinstance(status, dict) else {}
        combat = diagnostics.get("combat", {}) if diagnostics else {}
        return [
            f"State: {status.get('current_state', 'unknown')}",
            f"Last known: {status.get('last_known_state', 'unknown')}",
            f"Unknown streak: {status.get('unknown_streak', 0)}",
            f"Unknown hint: {status.get('last_unknown_hint') or 'none'}",
            f"Brawler: {status.get('current_brawler') or 'none'}",
            f"Matches: {status.get('matches_played', 0)}",
            f"Session: {status.get('session_duration_minutes', 0):.1f} min",
            f"Window active: {status.get('window_active', 'n/a')}",
            f"Window title: {status.get('window_title', 'n/a')}",
            f"Lobby: {diagnostics.get('lobby', {}).get('step') if diagnostics.get('lobby') else 'n/a'}",
            f"Screen: {diagnostics.get('screen_state') or 'n/a'}",
            f"Progress: {diagnostics.get('progress', {}).get('total_games') if diagnostics.get('progress') else 'n/a'} games",
            f"Match: {diagnostics.get('match', {}).get('active') if diagnostics.get('match') else 'n/a'}",
            f"Combat state: {combat.get('state', 'n/a')}",
            f"Enemies: {combat.get('enemies', 'n/a')}",
            f"Move key: {combat.get('move_key', 'n/a')}",
            f"Attack taken: {combat.get('attack_taken', 'n/a')}",
        ]

    def start(self) -> bool:
        return False

    def stop(self) -> None:
        pass


class AIThinkDashboard:
    """
    Dashboard completo que mostra TUDO o que a IA está a pensar e fazer.
    Integra com o wrapper.py real para mostrar estado do jogo, lobby, brawlers, etc.
    """

    def __init__(self, status_provider: Callable[[], dict] = None, refresh_interval: float = 0.3):
        self.status_provider = status_provider
        self.refresh_interval = max(0.1, refresh_interval)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._root = None

        self._thought_history = deque(maxlen=50)
        self._decision_history = deque(maxlen=20)
        self._event_history = deque(maxlen=30)
        self._metrics_history = deque(maxlen=60)
        self._action_history = deque(maxlen=30)
        self._state_history = deque(maxlen=20)

        self._game_stats = {
            "matches_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "current_streak": 0,
            "best_streak": 0,
        }

    def add_thought(self, thought: str, agent: str = "system"):
        """Adiciona um pensamento ao histórico."""
        timestamp = time.strftime('%H:%M:%S.%f')[:-3]
        self._thought_history.append({
            "time": timestamp,
            "agent": agent,
            "thought": thought
        })

    def add_decision(self, decision: str, confidence: float, agent: str = "system", reason: str = ""):
        """Adiciona uma decisão ao histórico."""
        self._decision_history.append({
            "decision": decision,
            "confidence": confidence,
            "agent": agent,
            "reason": reason,
            "time": time.time()
        })

    def add_event(self, event: str, event_type: str = "info"):
        """Adiciona um evento ao histórico."""
        self._event_history.append({
            "event": event,
            "type": event_type,
            "time": time.time()
        })

    def add_action(self, action: str, result: str = "", target: str = ""):
        """Adiciona uma ação ao histórico."""
        self._action_history.append({
            "action": action,
            "result": result,
            "target": target,
            "time": time.time()
        })

    def add_metric(self, metric: str, value: float):
        """Adiciona uma métrica."""
        self._metrics_history.append({
            "metric": metric,
            "value": value,
            "time": time.time()
        })

    def update_game_stats(self, stats: Dict[str, Any]):
        """Atualiza estatísticas do jogo."""
        self._game_stats.update(stats)

    def format_full_dashboard(self, status: dict) -> List[str]:
        """Formata dashboard completo com toda a informação."""
        lines = []

        diagnostics = status.get("diagnostics", {}) if isinstance(status, dict) else {}
        tracker_stats = status.get("tracker_stats", {}) if isinstance(status, dict) else {}
        ai_thinking = status.get("ai_thinking", {}) if isinstance(status, dict) else {}
        lobby_info = diagnostics.get("lobby", {}) if diagnostics else {}
        combat_info = diagnostics.get("combat", {}) if diagnostics else {}
        match_info = diagnostics.get("match", {}) if diagnostics else {}
        progress_info = diagnostics.get("progress", {}) if diagnostics else {}

        lines.extend(self._format_header())
        lines.extend(self._format_game_state(status, diagnostics))
        lines.extend(self._format_lobby_state(lobby_info))
        lines.extend(self._format_brawler_queue(status))
        lines.extend(self._format_match_state(match_info, combat_info))
        lines.extend(self._format_progress(progress_info))
        lines.extend(self._format_vision_detailed(ai_thinking, tracker_stats))
        lines.extend(self._format_ai_thinking(ai_thinking))
        lines.extend(self._format_rl_metrics(ai_thinking))
        lines.extend(self._format_action_history())
        lines.extend(self._format_thought_history())
        lines.extend(self._format_recent_events())
        lines.extend(self._format_footer())

        return lines

    def _format_header(self) -> List[str]:
        return [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║           🤖 ENTERPRISE AI - COMPREHENSIVE THINKING DASHBOARD    ║",
            "╠══════════════════════════════════════════════════════════════════╣",
        ]

    def _format_game_state(self, status: dict, diagnostics: dict) -> List[str]:
        lines = ["║ 🏠 GAME STATE                                                   ║"]
        lines.append(f"║   Current State: {status.get('current_state', 'unknown'):<42} ║")
        lines.append(f"║   Last Known: {status.get('last_known_state', 'unknown'):<44} ║")
        lines.append(f"║   Unknown Streak: {status.get('unknown_streak', 0):<41} ║")
        lines.append(f"║   Window: {status.get('window_title', 'n/a'):<48} ║")
        lines.append("║                                                                  ║")
        return lines

    def _format_lobby_state(self, lobby_info: dict) -> List[str]:
        lines = ["║ 🚪 LOBBY & NAVIGATION                                            ║"]
        lines.append(f"║   Lobby Step: {lobby_info.get('step', 'n/a'):<46} ║")
        lines.append(f"║   Brawler Selected: {lobby_info.get('brawler_selected', 'n/a'):<39} ║")
        lines.append(f"║   Map Type: {lobby_info.get('map_type', 'n/a'):<46} ║")
        lines.append("║                                                                  ║")
        return lines

    def _format_brawler_queue(self, status: dict) -> List[str]:
        lines = ["║ 👤 BRAWLER QUEUE                                                 ║"]
        brawler_queue = status.get('brawler_queue', [])
        if brawler_queue:
            current = brawler_queue[0] if brawler_queue else {}
            lines.append(f"║   Current: {current.get('name', 'none'):<47} ║")
            lines.append(f"║   Trophies: {current.get('current_trophies', 0)}/{current.get('target_trophies', 0):<40} ║")
            lines.append(f"║   Wins: {current.get('current_wins', 0)}/{current.get('target_wins', 0):<43} ║")
            if len(brawler_queue) > 1:
                lines.append(f"║   Next: {brawler_queue[1].get('name', 'none'):<48} ║")
        else:
            lines.append(f"║   Current: {status.get('current_brawler', 'none'):<47} ║")
        lines.append("║                                                                  ║")
        return lines

    def _format_match_state(self, match_info: dict, combat_info: dict) -> List[str]:
        lines = ["║ ⚔️  MATCH & COMBAT                                                ║"]
        lines.append(f"║   Match Active: {match_info.get('active', 'n/a'):<43} ║")
        lines.append(f"║   Combat State: {combat_info.get('state', 'n/a'):<43} ║")
        lines.append(f"║   Enemies: {combat_info.get('enemies', 'n/a'):<46} ║")
        lines.append(f"║   Move Key: {combat_info.get('move_key', 'n/a'):<46} ║")
        lines.append(f"║   Attack: {combat_info.get('attack_taken', 'n/a'):<46} ║")
        lines.append("║                                                                  ║")
        return lines

    def _format_progress(self, progress_info: dict) -> List[str]:
        lines = ["║ 📊 SESSION PROGRESS                                              ║"]
        lines.append(f"║   Matches: {progress_info.get('total_games', 0):<46} ║")
        lines.append(f"║   Victories: {self._game_stats.get('wins', 0):<44} ║")
        lines.append(f"║   Defeats: {self._game_stats.get('losses', 0):<45} ║")
        lines.append(f"║   Draws: {self._game_stats.get('draws', 0):<47} ║")

        total = self._game_stats.get('wins', 0) + self._game_stats.get('losses', 0)
        winrate = (self._game_stats.get('wins', 0) / total * 100) if total > 0 else 0
        lines.append(f"║   Win Rate: {winrate:.1f}%{' '*43} ║")
        lines.append(f"║   Streak: {self._game_stats.get('current_streak', 0)} (Best: {self._game_stats.get('best_streak', 0)}){' '*27} ║")
        lines.append("║                                                                  ║")
        return lines

    def _format_vision_detailed(self, ai_thinking: dict, tracker_stats: dict) -> List[str]:
        lines = ["║ 👁️  VISION (YOLO REAL-TIME)                                      ║"]
        detections = ai_thinking.get('detections', {})
        lines.append(f"║   Enemies: {len(detections.get('Enemy', [])):<45} ║")
        lines.append(f"║   Player: {'YES ✓' if detections.get('Player') else 'NO ✗':<47} ║")
        lines.append(f"║   Bushes: {len(detections.get('Bush', [])):<46} ║")
        lines.append(f"║   Cubebox: {len(detections.get('Cubebox', [])):<45} ║")
        lines.append(f"║   Tracker FPS: {tracker_stats.get('fps', 0):<42.1f} ║")
        lines.append(f"║   Frames Processed: {tracker_stats.get('frame_count', 0):<38} ║")

        if detections.get('Enemy'):
            positions = []
            for i, enemy in enumerate(detections['Enemy'][:3]):
                if isinstance(enemy, list) and len(enemy) >= 4:
                    cx = (enemy[0] + enemy[2]) // 2
                    cy = (enemy[1] + enemy[3]) // 2
                    positions.append(f"E{i+1}({cx},{cy})")
            if positions:
                lines.append(f"║   Enemy Pos: {', '.join(positions):<40} ║")

        lines.append("║                                                                  ║")
        return lines

    def _format_ai_thinking(self, ai_thinking: dict) -> List[str]:
        lines = ["║ 🧠 AI THINKING                                                   ║"]
        lines.append(f"║   Active Agent: {ai_thinking.get('active_agent', 'none'):<41} ║")
        lines.append(f"║   Confidence: {ai_thinking.get('confidence', 0):.2f}{' '*40} ║")
        lines.append(f"║   Last Decision: {ai_thinking.get('last_decision', 'none'):<40} ║")

        strategy = ai_thinking.get('current_strategy', 'none')
        lines.append(f"║   Strategy: {strategy:<45} ║")

        objectives = ai_thinking.get('objectives', [])
        if objectives:
            lines.append(f"║   Objectives: {objectives[0] if objectives else 'none':<40} ║")

        lines.append("║                                                                  ║")
        return lines

    def _format_rl_metrics(self, ai_thinking: dict) -> List[str]:
        lines = ["║ 📈 RL TRAINING METRICS                                          ║"]
        rl = ai_thinking.get('rl_metrics', {})
        lines.append(f"║   Algorithm: {rl.get('algorithm', 'PPO'):<44} ║")
        lines.append(f"║   Epsilon: {rl.get('epsilon', 0):.3f}{' '*42} ║")
        lines.append(f"║   Avg Reward: {rl.get('avg_reward', 0):.2f}{' '*40} ║")
        lines.append(f"║   Win Rate: {rl.get('win_rate', 0)*100:.1f}%{' '*42} ║")
        lines.append(f"║   Training Steps: {rl.get('steps', 0):<39} ║")
        lines.append(f"║   Buffer Size: {rl.get('buffer_size', 0):<40} ║")
        lines.append(f"║   SB3 Available: {'YES ✓' if rl.get('sb3_available') else 'NO ✗':<43} ║")
        lines.append("║                                                                  ║")
        return lines

    def _format_action_history(self) -> List[str]:
        lines = ["║ 🎮 RECENT ACTIONS                                                ║"]
        recent = list(self._action_history)[-6:] if self._action_history else []
        if recent:
            for i, action in enumerate(recent):
                act_name = action.get('action', 'unknown')[:15]
                result = action.get('result', '')[:20]
                lines.append(f"║   {i+1}. {act_name:<15} -> {result:<25} ║")
        else:
            lines.append(f"║   No actions yet{' '*41} ║")
        lines.append("║                                                                  ║")
        return lines

    def _format_thought_history(self) -> List[str]:
        lines = ["║ 💭 THOUGHT HISTORY (Most Recent)                                 ║"]
        recent = list(self._thought_history)[-5:] if self._thought_history else []
        if recent:
            for i, thought in enumerate(recent):
                t = thought.get('thought', '')[:42]
                agent = thought.get('agent', '')[:8]
                lines.append(f"║   [{agent:<8}] {t:<34} ║")
        else:
            lines.append(f"║   No thoughts recorded{' '*38} ║")
        lines.append("║                                                                  ║")
        return lines

    def _format_recent_events(self) -> List[str]:
        lines = ["║ 🔄 RECENT EVENTS                                                 ║"]
        recent = list(self._event_history)[-5:] if self._event_history else []
        if recent:
            for i, event in enumerate(recent):
                e = event.get('event', '')[:45]
                lines.append(f"║   {i+1}. {e:<48} ║")
        else:
            lines.append(f"║   No events recorded{' '*41} ║")
        lines.append("║                                                                  ║")
        return lines

    def _format_footer(self) -> List[str]:
        return [
            "╚══════════════════════════════════════════════════════════════════╝",
            f"   Last Update: {time.strftime('%H:%M:%S.%f')[:-3]} | Refresh: {self.refresh_interval}s"
        ]

    def start(self) -> bool:
        """Inicia o dashboard numa thread."""
        if self._thread and self._thread.is_alive():
            return True

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ai-think-dashboard")
        self._thread.start()
        return True

    def stop(self) -> None:
        """Para o dashboard."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            import tkinter as tk
            from tkinter import scrolledtext
        except Exception as exc:
            logger.warning(f"Dashboard requires tkinter: {exc}")
            return

        try:
            root = tk.Tk()
            self._root = root
            root.title("🤖 Enterprise AI - Complete Thinking Dashboard")
            root.geometry("900x800")
            root.attributes("-topmost", True)
            root.configure(bg="#0d1117")

            header = tk.Label(
                root, text="🤖 ENTERPRISE AI THINKING DASHBOARD",
                fg="#58a6ff", bg="#0d1117",
                font=("Consolas", 16, "bold")
            )
            header.pack(pady=5)

            text_area = scrolledtext.ScrolledText(
                root, wrap=tk.WORD, width=100, height=50,
                bg="#0d1117", fg="#c9d1d9",
                font=("Consolas", 8),
                borderwidth=0, padx=10, pady=5
            )
            text_area.pack(fill="both", expand=True)

            footer = tk.Label(
                root,
                text="Real-time AI decision visualization | Close to stop",
                fg="#6e7681", bg="#0d1117", font=("Segoe UI", 8)
            )
            footer.pack(pady=2)

            root.protocol("WM_DELETE_WINDOW", self.stop)
            self._schedule_update(text_area)
            root.mainloop()
        except Exception as exc:
            logger.warning(f"Dashboard error: {exc}")
        finally:
            self._root = None

    def _schedule_update(self, text_area) -> None:
        if self._stop_event.is_set() or self._root is None:
            return

        try:
            status = self.status_provider() if self.status_provider else {}
            lines = self.format_full_dashboard(status)

            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, "\n".join(lines))

        except Exception as exc:
            logger.debug(f"Dashboard update error: {exc}")

        try:
            self._root.after(int(self.refresh_interval * 1000),
                           lambda: self._schedule_update(text_area))
        except Exception:
            pass


class GameStateProvider:
    """
    Provider que recolhe estado do jogo do wrapper PylaAI.
    Usa o status provider existente ou cria um novo.
    """

    def __init__(self, wrapper=None):
        self.wrapper = wrapper
        self._last_status = {}

    def set_wrapper(self, wrapper):
        """Define o wrapper para usar."""
        self.wrapper = wrapper

    def get_status(self) -> Dict[str, Any]:
        """Retorna estado completo do jogo."""
        if self.wrapper is None:
            return self._get_mock_status()

        try:
            status = {
                "current_state": getattr(self.wrapper, 'current_state', 'unknown'),
                "last_known_state": getattr(self.wrapper, 'last_known_state', 'unknown'),
                "unknown_streak": getattr(self.wrapper, 'unknown_streak', 0),
                "current_brawler": self._get_current_brawler(),
                "window_title": self._get_window_title(),
                "diagnostics": self._get_diagnostics(),
                "tracker_stats": self._get_tracker_stats(),
                "ai_thinking": self._get_ai_thinking(),
                "brawler_queue": self._get_brawler_queue(),
            }
            self._last_status = status
            return status
        except Exception as e:
            logger.warning(f"Error getting status: {e}")
            return self._last_status

    def _get_current_brawler(self) -> str:
        """Obtém brawler atual."""
        if self.wrapper and hasattr(self.wrapper, 'brawler_queue'):
            current = self.wrapper.brawler_queue.get_current()
            if current:
                return current.name
        return "colt"

    def _get_window_title(self) -> str:
        """Obtém título da janela."""
        if self.wrapper and self.wrapper.emulator_controller:
            return getattr(self.wrapper.emulator_controller, 'window_title', 'n/a')
        return "n/a"

    def _get_diagnostics(self) -> Dict[str, Any]:
        """Obtém diagnósticos."""
        diagnostics = {}

        if self.wrapper and hasattr(self.wrapper, 'state_manager'):
            sm = self.wrapper.state_manager
            diagnostics["screen_state"] = getattr(sm, 'current_state', 'unknown')

        if self.wrapper and hasattr(self.wrapper, 'progress'):
            p = self.wrapper.progress
            diagnostics["progress"] = {
                "total_games": getattr(p, 'matches_played', 0),
                "wins": getattr(p, 'wins', 0),
                "losses": getattr(p, 'losses', 0),
            }

        if self.wrapper and hasattr(self.wrapper, 'play_logic'):
            pl = self.wrapper.play_logic
            diagnostics["combat"] = {
                "state": getattr(pl, 'combat_state', 'idle'),
                "enemies": getattr(pl, 'enemy_count', 0),
                "move_key": getattr(pl, 'last_move', 'none'),
            }

        if self.wrapper and hasattr(self.wrapper, 'lobby'):
            lobby = self.wrapper.lobby
            diagnostics["lobby"] = {
                "step": getattr(lobby, 'current_step', 'idle'),
            }

        return diagnostics

    def _get_tracker_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do tracker a partir do wrapper quando disponível."""
        if self.wrapper and hasattr(self.wrapper, 'play_logic') and self.wrapper.play_logic:
            pl = self.wrapper.play_logic
            if hasattr(pl, 'enemy_tracker') and pl.enemy_tracker:
                try:
                    stats = pl.enemy_tracker.get_stats()
                    return {
                        "fps": stats.get("fps", 0.0),
                        "frame_count": stats.get("frame_count", 0),
                    }
                except Exception:
                    pass
        return {
            "fps": 0.0,
            "frame_count": 0,
        }

    def _get_ai_thinking(self) -> Dict[str, Any]:
        """Obtém estado de pensamento da IA a partir do wrapper quando disponível."""
        result = {
            "active_agent": "none",
            "confidence": 0.0,
            "last_decision": "none",
            "current_strategy": "none",
            "objectives": [],
            "detections": {},
            "rl_metrics": {
                "algorithm": "none",
                "epsilon": 0.0,
                "avg_reward": 0.0,
                "win_rate": 0.0,
                "steps": 0,
                "buffer_size": 0,
                "sb3_available": False,
            },
        }
        if self.wrapper and hasattr(self.wrapper, 'play_logic') and self.wrapper.play_logic:
            pl = self.wrapper.play_logic
            try:
                result["current_strategy"] = getattr(pl, 'current_strategy', 'none')
                result["detections"] = getattr(pl, 'last_detections', {})
            except Exception:
                pass
        return result

    def _get_brawler_queue(self) -> List[Dict[str, Any]]:
        """Obtém fila de brawlers."""
        if self.wrapper and hasattr(self.wrapper, 'brawler_queue'):
            return self.wrapper.brawler_queue.get_queue()
        return []

    def _get_mock_status(self) -> Dict[str, Any]:
        """Retorna status mock para debugging."""
        return {
            "current_state": "lobby",
            "last_known_state": "lobby",
            "unknown_streak": 0,
            "current_brawler": "colt",
            "window_title": "BlueStacks App Player",
            "diagnostics": {
                "lobby": {"step": "idle"},
                "combat": {"state": "idle", "enemies": 0},
                "progress": {"total_games": 10, "wins": 6, "losses": 4},
            },
            "tracker_stats": {"fps": 30.0, "frame_count": 0},
            "ai_thinking": {
                "active_agent": "SupervisorAgent",
                "confidence": 0.85,
                "last_decision": "wait_for_match",
                "detections": {"Enemy": [], "Player": None, "Bush": [], "Cubebox": []},
                "rl_metrics": {
                    "algorithm": "PPO",
                    "epsilon": 0.5,
                    "avg_reward": 12.5,
                    "win_rate": 0.6,
                    "steps": 50000,
                    "buffer_size": 10000,
                    "sb3_available": True,
                },
            },
            "brawler_queue": [
                {"name": "colt", "current_trophies": 350, "target_trophies": 400, "current_wins": 8, "target_wins": 10},
                {"name": "bull", "current_trophies": 200, "target_trophies": 400, "current_wins": 3, "target_wins": 10},
            ],
        }
