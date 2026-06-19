"""
core/esp_overlay.py

ESP (Extra Sensory Perception) Overlay para BlueStacks/LDPlayer.
Janela Win32 semi-transparente que desenha bounding boxes, labels,
linhas de mira e informações sobre a janela do emulador em tempo real.

Seguranca: apenas overlay visual (GDI), nao injecta no processo do jogo.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

try:
    import win32api
    import win32con
    import win32gui
    import win32ui
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

logger = logging.getLogger(__name__)

CLASS_COLORS = {
    "enemy": (0, 0, 255),         # Vermelho (BGR)
    "teammate": (255, 128, 0),    # Azul
    "player": (0, 255, 0),        # Verde
    "wall": (128, 128, 128),      # Cinza
    "bush": (0, 128, 0),          # Verde escuro
    "powerup": (0, 255, 255),     # Amarelo
    "box": (0, 128, 255),         # Laranja
    "bullet": (0, 165, 255),      # Laranja claro
    "super_indicator": (255, 0, 255),  # Magenta
    "health_bar": (0, 255, 128),    # Verde claro
    "joystick": (128, 128, 128),   # Cinza
    "attack_button": (128, 128, 128),
}

CLASS_LABELS = {
    "enemy": "INIMIGO",
    "teammate": "ALIADO",
    "player": "PLAYER",
    "wall": "PAREDE",
    "bush": "ARBUSTO",
    "powerup": "POWERUP",
    "box": "CAIXA",
    "bullet": "PROJETIL",
    "super_indicator": "SUPER",
    "health_bar": "HP",
    "joystick": "JOYSTICK",
    "attack_button": "ATQ",
}


@dataclass
class ESPDetection:
    class_name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    track_id: int | None = None


class ESPOverlay:
    """
    Overlay GDI sobre a janela do emulador.
    Funciona numa thread separada a ~20fps.
    """

    def __init__(
        self,
        window_title: str = "BlueStacks",
        target_fps: float = 20.0,
    ):
        self.window_title = window_title
        self.target_fps = max(5.0, target_fps)
        self.enabled = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._hwnd: int | None = None
        self._overlay_hwnd: int | None = None
        self._window_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._lock = threading.Lock()
        self._detections: list[ESPDetection] = []
        self._fps_history: deque = deque(maxlen=30)
        self._last_draw_time = 0.0
        self._player_pos: tuple[int, int] | None = None
        self._target_pos: tuple[int, int] | None = None
        self._show_labels = True
        self._show_lines = True
        self._show_hp = True
        self._show_minimap = True
        self._minimap_size = 120

        if not WIN32_AVAILABLE:
            logger.warning("[ESP] Win32 extensions nao disponiveis — overlay desktop desativado")

    # ------------------------------------------------------------------
    # Detections input
    # ------------------------------------------------------------------

    def update_detections(self, detections: list[dict], player_pos: tuple[int, int] | None = None, target_pos: tuple[int, int] | None = None):
        with self._lock:
            self._detections = [ESPDetection(**d) for d in detections]
            self._player_pos = player_pos
            self._target_pos = target_pos

    def set_show_options(self, labels: bool = True, lines: bool = True, hp: bool = True, minimap: bool = True):
        self._show_labels = labels
        self._show_lines = lines
        self._show_hp = hp
        self._show_minimap = minimap

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        if not WIN32_AVAILABLE:
            return False
        if self.enabled:
            return True
        self._stop_event.clear()
        self.enabled = True
        self._thread = threading.Thread(target=self._draw_loop, daemon=True)
        self._thread.start()
        logger.info("[ESP] Overlay iniciado")
        return True

    def stop(self):
        self.enabled = False
        self._stop_event.set()
        if self._overlay_hwnd and WIN32_AVAILABLE:
            try:
                win32gui.DestroyWindow(self._overlay_hwnd)
            except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass
        self._overlay_hwnd = None
        logger.info("[ESP] Overlay parado")

    def toggle(self, state: bool | None = None) -> bool:
        if state is None:
            state = not self.enabled
        if state:
            return self.start()
        else:
            self.stop()
            return True

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def _find_emulator_window(self) -> bool:
        if not WIN32_AVAILABLE:
            return False
        # Try exact title first, then partial match
        hwnd = win32gui.FindWindow(None, self.window_title)
        if not hwnd:
            def callback(hwnd_extra, extra):
                text = win32gui.GetWindowText(hwnd_extra)
                if self.window_title.lower() in text.lower():
                    extra.append(hwnd_extra)
                return True
            matches = []
            win32gui.EnumWindows(callback, matches)
            if matches:
                hwnd = matches[0]
        if hwnd:
            self._hwnd = hwnd
            try:
                rect = win32gui.GetWindowRect(hwnd)
                self._window_rect = rect
                return True
            except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass
        return False

    def _create_overlay_window(self) -> bool:
        if not WIN32_AVAILABLE:
            return False
        if self._overlay_hwnd:
            return True
        try:
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = {win32con.WM_PAINT: self._on_paint, win32con.WM_DESTROY: lambda *a: 0}
            wc.hInstance = win32api.GetModuleHandle(None)
            wc.lpszClassName = "SoberanaESPOverlay"
            try:
                win32gui.RegisterClass(wc)
            except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass  # Already registered

            style = (
                win32con.WS_EX_LAYERED
                | win32con.WS_EX_TRANSPARENT
                | win32con.WS_EX_NOACTIVATE
                | win32con.WS_EX_TOPMOST
                | win32con.WS_EX_TOOLWINDOW
            )
            x, y, r, b = self._window_rect
            w = r - x
            h = b - y
            self._overlay_hwnd = win32gui.CreateWindowEx(
                style,
                "SoberanaESPOverlay",
                "SoberanaESP",
                win32con.WS_POPUP,
                x, y, w, h,
                None, None, wc.hInstance, None
            )
            # Make fully transparent background, opaque content
            from ctypes import windll
            windll.user32.SetLayeredWindowAttributes(self._overlay_hwnd, 0x000000, 0, win32con.LWA_COLORKEY)
            win32gui.ShowWindow(self._overlay_hwnd, win32con.SW_SHOW)
            return True
        except (ImportError, ModuleNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning(f"[ESP] Falha ao criar overlay: {e}")
            return False

    # ------------------------------------------------------------------
    # Drawing loop
    # ------------------------------------------------------------------

    def _draw_loop(self):
        while not self._stop_event.is_set():
            t0 = time.time()
            if not self._find_emulator_window():
                time.sleep(0.5)
                continue
            if not self._overlay_hwnd:
                if not self._create_overlay_window():
                    time.sleep(0.5)
                    continue
            self._position_overlay()
            self._redraw()
            elapsed = time.time() - t0
            fps = 1.0 / max(elapsed, 0.001)
            self._fps_history.append(fps)
            sleep_time = max(0.0, (1.0 / self.target_fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _position_overlay(self):
        if not WIN32_AVAILABLE or not self._overlay_hwnd or not self._hwnd:
            return
        try:
            rect = win32gui.GetWindowRect(self._hwnd)
            if rect != self._window_rect:
                self._window_rect = rect
                x, y, r, b = rect
                win32gui.SetWindowPos(
                    self._overlay_hwnd, win32con.HWND_TOPMOST,
                    x, y, r - x, b - y,
                    win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW
                )
        except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError):
            pass

    def _redraw(self):
        if not WIN32_AVAILABLE or not self._overlay_hwnd:
            return
        try:
            hdc = win32gui.GetDC(self._overlay_hwnd)
            mem_dc = win32gui.CreateCompatibleDC(hdc)
            x, y, r, b = self._window_rect
            w = r - x
            h = b - y
            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(win32ui.CreateDCFromHandle(hdc), w, h)
            old = win32gui.SelectObject(mem_dc, bmp.GetHandle())

            # Clear transparent
            brush = win32gui.CreateSolidBrush(0x000000)
            win32gui.FillRect(mem_dc, (0, 0, w, h), brush)
            win32gui.DeleteObject(brush)

            with self._lock:
                detections = list(self._detections)
                player_pos = self._player_pos
                target_pos = self._target_pos

            # Draw detections
            for det in detections:
                self._draw_detection(mem_dc, det, w, h)

            # Draw line of sight
            if self._show_lines and player_pos and target_pos:
                pen = win32gui.CreatePen(win32con.PS_SOLID, 2, 0x00FF00)
                old_pen = win32gui.SelectObject(mem_dc, pen)
                win32gui.MoveToEx(mem_dc, player_pos[0], player_pos[1])
                win32gui.LineTo(mem_dc, target_pos[0], target_pos[1])
                win32gui.SelectObject(mem_dc, old_pen)
                win32gui.DeleteObject(pen)

            # Draw minimap
            if self._show_minimap:
                self._draw_minimap(mem_dc, detections, w, h)

            # Draw FPS
            avg_fps = sum(self._fps_history) / max(1, len(self._fps_history)) if self._fps_history else 0
            win32gui.SetTextColor(mem_dc, 0x00FF00)
            win32gui.SetBkMode(mem_dc, win32con.TRANSPARENT)
            win32gui.DrawText(mem_dc, f"ESP {avg_fps:.0f}FPS | {len(detections)} objs", -1, (4, 4, 200, 20), win32con.DT_LEFT)

            # Blit to screen
            from ctypes import windll
            windll.gdi32.TransparentBlt(hdc, 0, 0, w, h, mem_dc, 0, 0, w, h, 0x000000)

            # Cleanup
            win32gui.SelectObject(mem_dc, old)
            win32gui.DeleteDC(mem_dc)
            win32gui.DeleteObject(bmp.GetHandle())
            win32gui.ReleaseDC(self._overlay_hwnd, hdc)
        except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as e:
            logger.debug(f"[ESP] Draw error: {e}")

    def _draw_detection(self, hdc, det: ESPDetection, win_w: int, win_h: int):
        color = CLASS_COLORS.get(det.class_name, (255, 255, 255))
        bgr = (color[2] << 16) | (color[1] << 8) | color[0]
        x, y = det.x, det.y
        w, h = det.width, det.height
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return

        # Box
        pen = win32gui.CreatePen(win32con.PS_SOLID, 2, bgr)
        old_pen = win32gui.SelectObject(hdc, pen)
        brush = win32gui.GetStockObject(win32con.NULL_BRUSH)
        old_brush = win32gui.SelectObject(hdc, brush)
        win32gui.Rectangle(hdc, x, y, x + w, y + h)
        win32gui.SelectObject(hdc, old_pen)
        win32gui.DeleteObject(pen)
        win32gui.SelectObject(hdc, old_brush)

        # Label
        if self._show_labels:
            label = CLASS_LABELS.get(det.class_name, det.class_name.upper())
            label_text = f"{label} {det.confidence:.0%}"
            # Background rect for label
            win32gui.SetBkMode(hdc, win32con.OPAQUE)
            win32gui.SetBkColor(hdc, bgr)
            win32gui.SetTextColor(hdc, 0xFFFFFF)
            label_y = max(0, y - 16)
            win32gui.DrawText(hdc, label_text, -1, (x, label_y, x + 120, label_y + 16), win32con.DT_LEFT)
            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)

        # HP bar (mock based on class)
        if self._show_hp and det.class_name in ("enemy", "player", "teammate"):
            hp_h = 4
            hp_y = y + h + 2
            hp_w = w
            # Background
            hp_bg = win32gui.CreateSolidBrush(0x333333)
            win32gui.FillRect(hdc, (x, hp_y, x + hp_w, hp_y + hp_h), hp_bg)
            win32gui.DeleteObject(hp_bg)
            # Foreground (simulate random HP for visual)
            hp_pct = 0.6 + (hash(det.class_name) % 40) / 100.0
            hp_fg = win32gui.CreateSolidBrush(0x00FF00 if hp_pct > 0.5 else 0x0000FF)
            win32gui.FillRect(hdc, (x, hp_y, x + int(hp_w * hp_pct), hp_y + hp_h), hp_fg)
            win32gui.DeleteObject(hp_fg)

    def _draw_minimap(self, hdc, detections: list[ESPDetection], win_w: int, win_h: int):
        size = self._minimap_size
        pad = 8
        mx = win_w - size - pad
        my = pad
        # Background
        bg = win32gui.CreateSolidBrush(0x1a1a2e)
        win32gui.FillRect(hdc, (mx, my, mx + size, my + size), bg)
        win32gui.DeleteObject(bg)
        # Border
        pen = win32gui.CreatePen(win32con.PS_SOLID, 1, 0x444444)
        old_pen = win32gui.SelectObject(hdc, pen)
        win32gui.Rectangle(hdc, mx, my, mx + size, my + size)
        win32gui.SelectObject(hdc, old_pen)
        win32gui.DeleteObject(pen)

        # Normalize positions to minimap
        for det in detections:
            if det.class_name in ("wall", "bush"):
                continue
            color = CLASS_COLORS.get(det.class_name, (255, 255, 255))
            bgr = (color[2] << 16) | (color[1] << 8) | color[0]
            nx = mx + int((det.center_x / win_w) * size)
            ny = my + int((det.center_y / win_h) * size)
            nx = max(mx, min(mx + size, nx))
            ny = max(my, min(my + size, ny))
            dot = win32gui.CreateSolidBrush(bgr)
            win32gui.FillRect(hdc, (nx - 2, ny - 2, nx + 3, ny + 3), dot)
            win32gui.DeleteObject(dot)

        # Player always at center of minimap for relative view
        center = win32gui.CreateSolidBrush(0x00FF00)
        cx = mx + size // 2
        cy = my + size // 2
        win32gui.FillRect(hdc, (cx - 3, cy - 3, cx + 4, cy + 4), center)
        win32gui.DeleteObject(center)

    def get_stats(self) -> dict:
        avg_fps = sum(self._fps_history) / max(1, len(self._fps_history)) if self._fps_history else 0
        return {
            "enabled": self.enabled,
            "win32_available": WIN32_AVAILABLE,
            "hwnd_found": self._hwnd is not None,
            "overlay_created": self._overlay_hwnd is not None,
            "fps": round(avg_fps, 1),
            "detections_count": len(self._detections),
            "show_labels": self._show_labels,
            "show_lines": self._show_lines,
            "show_hp": self._show_hp,
            "show_minimap": self._show_minimap,
        }
