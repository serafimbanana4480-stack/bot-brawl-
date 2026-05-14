"""
core/notifications.py

Sistema de notificacoes multi-canal para o bot Brawl Stars.
Suporta: browser push (via dashboard), webhooks HTTP, desktop notifications.
"""

import json
import time
import logging
import threading
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    webhook_url: str = ""
    webhook_headers: Dict = field(default_factory=dict)
    desktop_enabled: bool = False
    browser_enabled: bool = True
    # Triggers
    on_crash: bool = True
    on_win: bool = False
    on_loss: bool = True
    on_consecutive_losses: int = 3
    on_unknown_timeout: int = 30
    on_trophy_limit: bool = True


@dataclass
class NotificationEvent:
    title: str
    message: str
    level: str  # info, warning, error
    timestamp: float
    category: str


class NotificationManager:
    """Gerencia notificacoes do bot para multiplos canais."""

    def __init__(self, config_path: Optional[Path] = Path("data/notifications.json")):
        self.config_path = Path(config_path) if config_path else None
        if self.config_path:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = NotificationConfig()
        self._load_config()
        self._history: deque = deque(maxlen=100)
        self._listeners: List[Callable] = []  # Browser push listeners
        self._lock = threading.RLock()
        self._last_losses = 0
        self._unknown_since = 0.0

    def _load_config(self):
        if not self.config_path or not self.config_path.exists():
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        except Exception as e:
            logger.warning(f"[NOTIFY] Falha ao carregar config: {e}")

    def save_config(self):
        if not self.config_path:
            return
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config.__dict__, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[NOTIFY] Falha ao guardar config: {e}")

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.save_config()

    def get_config(self) -> Dict:
        return self.config.__dict__

    def add_browser_listener(self, callback: Callable):
        """Adiciona listener para browser push (SSE)."""
        with self._lock:
            self._listeners.append(callback)

    def remove_browser_listener(self, callback: Callable):
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def send(self, title: str, message: str, level: str = "info", category: str = "general"):
        """Envia notificacao para todos os canais ativos."""
        event = NotificationEvent(title, message, level, time.time(), category)
        with self._lock:
            self._history.append(event)
            # Browser push
            if self.config.browser_enabled:
                for cb in list(self._listeners):
                    try:
                        cb(event)
                    except Exception:
                        pass
        # Webhook
        if self.config.webhook_url:
            self._send_webhook(event)
        # Desktop
        if self.config.desktop_enabled:
            self._send_desktop(event)
        logger.info(f"[NOTIFY] {level.upper()}: {title} - {message}")

    def _send_webhook(self, event: NotificationEvent):
        try:
            payload = json.dumps({
                "title": event.title,
                "message": event.message,
                "level": event.level,
                "category": event.category,
                "timestamp": event.timestamp,
            }).encode("utf-8")
            req = urllib.request.Request(
                self.config.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json", **self.config.webhook_headers},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                pass
        except Exception as e:
            logger.debug(f"[NOTIFY] Webhook failed: {e}")

    def _send_desktop(self, event: NotificationEvent):
        try:
            # Try plyer first, then win10toast
            try:
                from plyer import notification
                notification.notify(title=event.title, message=event.message, timeout=5)
            except ImportError:
                try:
                    from win10toast import ToastNotifier
                    ToastNotifier().show_toast(event.title, event.message, duration=5)
                except ImportError:
                    pass
        except Exception as e:
            logger.debug(f"[NOTIFY] Desktop failed: {e}")

    def check_and_notify(self, bot_status: Dict):
        """Verifica triggers e envia notificacoes conforme necessario."""
        # Consecutive losses
        if self.config.on_consecutive_losses > 0:
            current_losses = bot_status.get("losses", 0)
            if current_losses > self._last_losses:
                self._last_losses = current_losses
            # Simplificado: notificar se losses aumentaram e win_rate < 30%
            wr = bot_status.get("win_rate", 0)
            if wr < 0.3 and current_losses >= self.config.on_consecutive_losses:
                self.send("Derrotas Seguidas", f"Win rate baixo: {(wr*100):.1f}%", "warning", "consecutive_losses")

        # Trophy limit
        if self.config.on_trophy_limit and bot_status.get("safety", {}).get("trophy_limit_reached"):
            self.send("Limite de Trofeus", "O bot atingiu o limite de trofeus configurado", "warning", "trophy_limit")

    def get_history(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            return [
                {"title": e.title, "message": e.message, "level": e.level,
                 "category": e.category, "timestamp": e.timestamp}
                for e in list(self._history)[-limit:]
            ]


# Singleton
_global_notification_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    global _global_notification_manager
    if _global_notification_manager is None:
        _global_notification_manager = NotificationManager()
    return _global_notification_manager
