"""
dashboard_server.py

Dashboard web server — orquestrador apenas.
Handler em core.dashboard_handler, templates em core.dashboard_templates,
business logic em core.dashboard_logic.
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Any

try:
    from http.server import HTTPServer
    from socketserver import ThreadingMixIn
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
except ImportError:
    ThreadingHTTPServer = None
    HTTPServer = None

logger = logging.getLogger(__name__)

try:
    from core.log_buffer import LogBuffer, install_log_buffer, get_log_buffer
    HAS_LOGBUFFER = True
except ImportError:
    HAS_LOGBUFFER = False
    LogBuffer = None
    install_log_buffer = None
    get_log_buffer = None

try:
    from core.notifications import NotificationManager, get_notification_manager
    HAS_NOTIFICATIONS = True
except ImportError:
    HAS_NOTIFICATIONS = False
    NotificationManager = None
    get_notification_manager = None

from core.dashboard_logic import (
    DashboardDataBridge,
    ReplayRecorder,
    ABTestManager,
)

from core.dashboard_handler import DashboardHandler

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
