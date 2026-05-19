import subprocess
import time
import random
import shutil
import os
import win32gui
import win32con
import win32api
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmulatorConfig:
    """Configuração do emulador"""
    name: str = "LDPlayer"
    adb_port: int = 5555
    window_title: str = "LDPlayer"
    resolution: Tuple[int, int] = (1920, 1080)
    dpi: int = 280
    # Campos opcionais para compatibilidade com testes e variantes de configuração
    adb_path: Optional[str] = None      # caminho explícito para o executável adb
    window_width: Optional[int] = None  # largura da janela do emulador
    window_height: Optional[int] = None # altura da janela do emulador

    @classmethod
    def for_bluestacks(cls) -> "EmulatorConfig":
        return cls(
            name="BlueStacks",
            adb_port=5554,  # BlueStacks default ADB port (NOT 5555)
            window_title="BlueStacks App Player",
            resolution=(1920, 1080),
            dpi=280
        )

    @classmethod
    def for_ldplayer(cls) -> "EmulatorConfig":
        return cls(
            name="LDPlayer",
            adb_port=5555,
            window_title="LDPlayer",
            resolution=(1920, 1080),
            dpi=280
        )


class ADBController:
    """Controlador ADB para interação com emulador
    
    Integra ResilientADB para retry automático, circuit breaker e health checks.
    Fallback para subprocess direto se ResilientADB não estiver disponível.
    """
    
    @staticmethod
    def _sanitize_device_id(device_id: str) -> str:
        """Sanitiza device_id para evitar caracteres perigosos em subprocess.

        O device_id e gerado internamente (ex: 'emulator-5555'), portanto
        basta garantir que contenha apenas caracteres alfanumericos, hifen,
        dois pontos e ponto. Usa regex para extrair apenas o padrao valido
        (IP:porta ou emulator-porta), rejeitando qualquer injecao de comandos.
        """
        import re
        # Match valid ADB device patterns: "emulator-NNNN" or "IP:port"
        match = re.match(r'^([0-9]{1,3}(\.[0-9]{1,3}){3}:[0-9]+|emulator-[0-9]+)', device_id)
        if match:
            cleaned = match.group(1)
        else:
            # Fallback: allow only safe chars, but strip anything that looks like a command
            allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-:.")
            cleaned = "".join(c for c in device_id if c in allowed)
            # Remove common command patterns that might survive char filtering
            for dangerous in ("rm", "sh", "bash", "cmd", "exec", "eval"):
                cleaned = cleaned.replace(dangerous, "")
        if cleaned != device_id:
            logger.warning(f"[ADB] device_id sanitizado: '{device_id}' -> '{cleaned}'")
        return cleaned

    def __init__(self, config: EmulatorConfig):
        self.config = config
        # Honor explicit adb_path from config before falling back to auto-detection
        if getattr(config, "adb_path", None) and os.path.isfile(config.adb_path):
            self.adb_path = config.adb_path
        else:
            self.adb_path = self._find_adb()
        raw_device = f"emulator-{config.adb_port}"
        self.device_id = self._sanitize_device_id(raw_device)
        
        # Try to use ResilientADB for automatic retry and circuit breaker
        self._resilient_adb = None
        try:
            from adb_resilient import ResilientADB, ADBConfig
            self._resilient_adb = ResilientADB(
                adb_path=self.adb_path,
                adb_id=self.device_id,
                config=ADBConfig(max_retries=3, base_delay=1.0, circuit_breaker_threshold=5)
            )
            logger.info("[ADB] ResilientADB integrado com sucesso (retry + circuit breaker)")
        except ImportError:
            logger.info("[ADB] ResilientADB não disponível, usando subprocess direto")
        
        logger.debug(f"[ADB] ADBController inicializado: adb_path={self.adb_path}, device_id={self.device_id}")
    
    def _find_adb(self) -> str:
        env_adb = os.getenv("ADB_PATH")
        if env_adb and os.path.isfile(env_adb):
            return env_adb

        adb_on_path = shutil.which("adb")
        if adb_on_path:
            return adb_on_path

        bundle_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        possible_paths = [
            os.path.join(bundle_dir, "adb.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"),
            r"C:\Program Files\LDPlayer\LDPlayer9\adb.exe",
            r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
        ]
        
        for path in possible_paths:
            if os.path.isfile(path):
                return path
        
        return "adb"
    
    def connect(self, max_retries: int = 3, backoff: float = 2.0) -> bool:
        """Connect to ADB device with exponential backoff retry."""
        logger.debug(f"[ADB] Tentando conectar ao dispositivo {self.device_id}")
        
        # Try ResilientADB first
        if self._resilient_adb:
            result = self._resilient_adb.run(["connect", self.device_id], timeout=10)
            if result and result.returncode == 0:
                logger.info(f"[ADB] Conectado com sucesso a {self.device_id} (via ResilientADB)")
                return True
            logger.warning("[ADB] ResilientADB connect falhou, tentando fallback")
        
        # Fallback: direct subprocess with manual retry
        for attempt in range(max_retries):
            try:
                result = subprocess.run([self.adb_path, "connect", self.device_id], capture_output=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"[ADB] Conectado com sucesso a {self.device_id}")
                    return True
                else:
                    logger.warning(f"[ADB] Falha ao conectar (attempt {attempt+1}/{max_retries}): returncode={result.returncode}")
            except subprocess.TimeoutExpired:
                logger.warning(f"[ADB] Timeout ao conectar (attempt {attempt+1}/{max_retries})")
            except Exception as e:
                logger.error(f"[ADB] Erro ao conectar (attempt {attempt+1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                wait_time = backoff * (2 ** attempt)
                logger.info(f"[ADB] Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        logger.error(f"[ADB] Failed to connect after {max_retries} attempts")
        return False
    
    def tap(self, x: int, y: int) -> bool:
        """Tap with ResilientADB fallback to direct subprocess."""
        # Randomize tap duration for anti-detection (50-200ms)
        duration_ms = random.randint(50, 200)
        logger.debug(f"[ADB] Executando tap em ({x}, {y}) duration={duration_ms}ms")
        
        if self._resilient_adb:
            return self._resilient_adb.tap(x, y)
        
        cmd = [self.adb_path, "-s", self.device_id, "shell", "input", "tap", str(x), str(y)]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            if result.returncode == 0:
                logger.debug(f"[ADB] Tap executado com sucesso em ({x}, {y})")
            else:
                logger.warning(f"[ADB] Tap falhou: returncode={result.returncode}, stderr={result.stderr.decode()}")
            return result.returncode == 0
        except Exception as e:
            logger.error(f"[ADB] Erro ao executar tap em ({x}, {y}): {e}")
            return False

    def keyevent(self, keycode: int) -> bool:
        """Keyevent with ResilientADB fallback."""
        logger.debug(f"[ADB] Executando keyevent {keycode}")
        
        if self._resilient_adb:
            result = self._resilient_adb.run(["shell", "input", "keyevent", str(keycode)])
            return result is not None and result.returncode == 0
        
        try:
            result = subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", "input", "keyevent", str(keycode)],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                logger.debug(f"[ADB] Keyevent {keycode} executado com sucesso")
            else:
                logger.warning(f"[ADB] Keyevent falhou: returncode={result.returncode}")
            return result.returncode == 0
        except Exception as e:
            logger.error(f"[ADB] Erro ao executar keyevent {keycode}: {e}")
            return False
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300) -> bool:
        """Swipe with ResilientADB fallback."""
        logger.debug(f"[ADB] Executando swipe: ({x1}, {y1}) -> ({x2}, {y2}), duration={duration}ms")
        
        if self._resilient_adb:
            return self._resilient_adb.swipe(x1, y1, x2, y2, duration)
        
        try:
            result = subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", "input", "swipe",
                 str(x1), str(y1), str(x2), str(y2), str(duration)],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                logger.debug(f"[ADB] Swipe executado com sucesso")
            else:
                logger.warning(f"[ADB] Swipe falhou: returncode={result.returncode}")
            return result.returncode == 0
        except Exception as e:
            logger.error(f"[ADB] Erro ao executar swipe: {e}")
            return False
    
    def screenshot(self) -> Optional[bytes]:
        """Screenshot with ResilientADB fallback."""
        logger.debug("[ADB] Capturando screenshot")
        
        if self._resilient_adb:
            data = self._resilient_adb.screenshot()
            if data:
                logger.debug(f"[ADB] Screenshot capturado via ResilientADB: {len(data)} bytes")
                return data
        
        try:
            result = subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", "screencap", "-p"],
                capture_output=True, timeout=10
            )
            if result.returncode == 0:
                screenshot_data = result.stdout.replace(b'\r\n', b'\n')
                logger.debug(f"[ADB] Screenshot capturado com sucesso: {len(screenshot_data)} bytes")
                return screenshot_data
            else:
                logger.warning(f"[ADB] Screenshot falhou: returncode={result.returncode}")
        except Exception as e:
            logger.error(f"[ADB] Erro ao capturar screenshot: {e}")
        return None
    
    def batch_commands(self, commands: List[List[str]]) -> List[bool]:
        """Execute multiple ADB commands in a single subprocess call for performance.
        
        Args:
            commands: List of command arg lists, e.g. [["shell", "input", "tap", "100", "200"], ...]
            
        Returns:
            List of success booleans for each command.
        """
        if not commands:
            return []

        # FIX #17: Execute commands individually to avoid shell command injection risk
        # The previous " && ".join() approach was vulnerable to argument injection
        results = []
        for args in commands:
            cmd = [self.adb_path, "-s", self.device_id] + args
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=5)
                results.append(result.returncode == 0)
            except Exception:
                results.append(False)
        return results
    
    def get_resilient_stats(self) -> dict:
        """Get ResilientADB statistics if available."""
        if self._resilient_adb:
            return self._resilient_adb.get_stats()
        return {"resilient_adb": "not_available"}


class WindowController:
    """Controlador da janela do emulador no Windows"""
    
    def __init__(self, window_title: str):
        self.window_title = window_title
        self.hwnd: Optional[int] = None
        logger.debug(f"[WINDOW] WindowController inicializado para '{window_title}'")
        self._find_window()
    
    def _find_window(self) -> bool:
        self.hwnd = win32gui.FindWindow(None, self.window_title)
        if self.hwnd:
            logger.debug(f"[WINDOW] Janela encontrada: hwnd={self.hwnd}, title='{self.window_title}'")
        else:
            logger.warning(f"[WINDOW] Janela não encontrada: title='{self.window_title}'")
        return self.hwnd is not None
    
    def is_visible(self) -> bool:
        if not self.hwnd:
            self._find_window()
        visible = win32gui.IsWindowVisible(self.hwnd) if self.hwnd else False
        logger.debug(f"[WINDOW] Visibilidade da janela: {visible}")
        return visible

    def get_rect(self):
        if self.hwnd:
            return win32gui.GetWindowRect(self.hwnd)
        return None

    def get_title(self) -> Optional[str]:
        if self.hwnd:
            try:
                return win32gui.GetWindowText(self.hwnd)
            except Exception:
                return None
        return None

    def is_focused(self) -> bool:
        if not self.hwnd and not self._find_window():
            logger.debug("[WINDOW] Sem hwnd para verificar foco")
            return False

        try:
            focused = win32gui.GetForegroundWindow() == self.hwnd
            logger.debug(f"[WINDOW] Janela focada: {focused}")
            return focused
        except Exception as e:
            logger.error(f"[WINDOW] Erro ao verificar foco: {e}")
            return False

    def activate(self) -> bool:
        if not self.hwnd and not self._find_window():
            logger.warning("[WINDOW] Não é possível ativar: hwnd não encontrado")
            return False

        try:
            logger.debug(f"[WINDOW] Ativando janela '{self.window_title}' (hwnd={self.hwnd})")
            if not win32gui.IsWindowVisible(self.hwnd):
                logger.debug("[WINDOW] Janela não visível, tornando visível")
                win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)

            # Try SetForegroundWindow directly first
            try:
                win32gui.SetForegroundWindow(self.hwnd)
                win32gui.BringWindowToTop(self.hwnd)
                logger.info(f"[WINDOW] Janela '{self.window_title}' ativada com sucesso")
                return True
            except Exception:
                pass

            # Fallback: Use AttachThreadInput trick to bypass Windows foreground lock
            # This works even when the calling process doesn't have foreground permission
            try:
                import win32process
                import ctypes

                current_thread = win32api.GetCurrentThreadId()
                foreground_window = win32gui.GetForegroundWindow()
                if foreground_window:
                    foreground_thread, _ = win32process.GetWindowThreadProcessId(foreground_window)
                    if foreground_thread != current_thread:
                        win32api.AttachThreadInput(current_thread, foreground_thread, True)
                        try:
                            win32gui.SetForegroundWindow(self.hwnd)
                            win32gui.BringWindowToTop(self.hwnd)
                        finally:
                            win32api.AttachThreadInput(current_thread, foreground_thread, False)
                    else:
                        win32gui.BringWindowToTop(self.hwnd)
                else:
                    win32gui.BringWindowToTop(self.hwnd)

                # Alternative: Use ALT key trick to gain foreground permission
                if not win32gui.GetForegroundWindow() == self.hwnd:
                    import ctypes
                    user32 = ctypes.windll.user32
                    # Simulate brief ALT key press to unlock foreground
                    user32.keybd_event(0x12, 0, 0, 0)  # ALT down
                    user32.keybd_event(0x12, 0, 2, 0)  # ALT up
                    win32gui.SetForegroundWindow(self.hwnd)

                if win32gui.GetForegroundWindow() == self.hwnd:
                    logger.info(f"[WINDOW] Janela '{self.window_title}' ativada via AttachThreadInput")
                    return True
                else:
                    logger.warning(f"[WINDOW] Janela ativada mas não está em foreground (pode precisar de foco manual)")
                    win32gui.BringWindowToTop(self.hwnd)
                    return True  # Still return True - ADB input works regardless of window focus
            except Exception as e2:
                logger.debug(f"[WINDOW] AttachThreadInput fallback falhou: {e2}")
                # Last resort: just bring to top
                win32gui.BringWindowToTop(self.hwnd)
                return True  # ADB input works regardless of window focus

        except Exception as e:
            logger.error(f"[WINDOW] Falha ao ativar janela '{self.window_title}': {e}")
            return False

    def move(self, x: int, y: int) -> bool:
        """Move a janela para uma nova posição (top-left)."""
        if not self.hwnd and not self._find_window():
            logger.warning("[WINDOW] Não é possível mover: hwnd não encontrado")
            return False
        try:
            rect = self.get_rect()
            if rect:
                width = rect[2] - rect[0]
                height = rect[3] - rect[1]
                win32gui.SetWindowPos(
                    self.hwnd, 0, x, y, width, height,
                    win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE
                )
                logger.debug(f"[WINDOW] Janela movida para ({x}, {y})")
                return True
        except Exception as e:
            logger.error(f"[WINDOW] Falha ao mover janela: {e}")
        return False


class EmulatorController:
    """Controlador principal com Inteligência Soberana"""

    def __init__(self, config: Optional[EmulatorConfig] = None, safety_system=None, humanization_system=None):
        self.config = config or EmulatorConfig()
        self.adb = ADBController(self.config)
        self.window = WindowController(self.config.window_title)
        self.is_connected = False
        self.safety_system = safety_system
        self.humanization_system = humanization_system
        logger.debug(f"[EMULATOR] EmulatorController inicializado com safety_system: {safety_system is not None}")
    
    def connect(self) -> bool:
        logger.info("[EMULATOR] Conectando ao emulador")
        self.is_connected = self.adb.connect()
        if self.is_connected:
            logger.info("[EMULATOR] Conectado com sucesso")
            self._verify_resolution()
        else:
            logger.warning("[EMULATOR] Falha ao conectar")
        return self.is_connected

    def ensure_window_active(self) -> bool:
        """Garante que a janela do emulador está em foreground antes de ações críticas."""
        logger.debug("[EMULATOR] Verificando se janela está ativa antes de ação crítica")
        if not self.window.hwnd:
            self.window._find_window()

        if self.window.is_focused():
            logger.debug("[EMULATOR] Janela já está ativa")
            return True

        logger.info("[EMULATOR] Janela não está ativa, tentando ativar")
        activated = self.window.activate()
        if activated:
            logger.info(f"[EMULATOR] Janela '{self.window.window_title}' ativada para input")
        else:
            logger.warning(f"[EMULATOR] Não foi possível ativar a janela '{self.window.window_title}'")
        return activated
    
    def _verify_resolution(self):
        try:
            import re
            result = subprocess.run([self.adb.adb_path, "-s", self.adb.device_id, "shell", "wm", "size"], 
                                 capture_output=True, text=True)
            match = re.search(r"(\d+)x(\d+)", result.stdout)
            if match:
                w, h = int(match.group(1)), int(match.group(2))
                if h > w: w, h = h, w
                self.config.resolution = (w, h)
                logger.info(f"Resolução real: {w}x{h}")
        except Exception:
            pass

    def tap(self, x: int, y: int) -> bool:
        """Tap com Jitter e Humanização (Anti-Detection) - usa ADB para emuladores"""
        jx, jy = x, y
        if self.humanization_system and getattr(self.humanization_system, "config", None) and self.humanization_system.config.enabled:
            try:
                tremor_x, tremor_y = self.humanization_system.get_tremor()
                jx += int(round(tremor_x))
                jy += int(round(tremor_y))
                delay = self.humanization_system.get_delay("tap")
                if delay > 0:
                    time.sleep(delay)
            except Exception as e:
                logger.debug(f"[TAP] Humanization fallback: {e}")

        jx += random.randint(-3, 3)
        jy += random.randint(-3, 3)
        
        logger.info(f"[TAP] Usando ADB tap em ({jx}, {jy})")
        result = self.adb.tap(jx, jy)
        logger.info(f"[TAP] ADB tap resultado: {result}")
        return result

    def tap_scaled(self, x_1080: int, y_1080: int) -> bool:
        logger.debug(f"[TAP] Iniciando tap")
        logger.debug(f"[TAP] Coordenadas 1080p: ({x_1080}, {y_1080})")
        width, height = self.config.resolution
        logger.debug(f"[TAP] Resolução atual: {width}x{height}")
        scale_factor_x = width / 1920.0
        scale_factor_y = height / 1080.0
        logger.debug(f"[TAP] Fator de escala: x={scale_factor_x:.4f}, y={scale_factor_y:.4f}")
        rx = x_1080 / 1920.0
        ry = y_1080 / 1080.0
        real_x = int(rx * self.config.resolution[0])
        real_y = int(ry * self.config.resolution[1])
        logger.debug(f"[TAP] Coordenadas reais: ({real_x}, {real_y})")
        logger.info(f"[TAP] Coordenadas: 1080p=({x_1080}, {y_1080}) -> Real=({real_x}, {real_y}) @ {self.config.resolution}")
        result = self.tap(real_x, real_y)
        logger.debug(f"[TAP] Usando ADB tap em ({real_x}, {real_y})")
        logger.info(f"[TAP] Resultado: {result}")

        # Record tap for behavioral biometrics
        if result and self.safety_system:
            try:
                self.safety_system.record_tap(float(real_x), float(real_y))
                logger.debug(f"[TAP] Tap registrado no safety_system")
            except Exception as e:
                logger.warning(f"[TAP] Falha ao registrar tap no safety_system: {e}")

        if not result:
            logger.error(f"[TAP] Tap falhou: coordenadas=({real_x}, {real_y})")
            logger.error(f"[TAP] ADB disponível: {self.adb is not None}")
            logger.error(f"[TAP] Device conectado: {self.adb.device_id if self.adb else 'N/A'}")
        return result

    def swipe_scaled(self, x1, y1, x2, y2, duration=300):
        rx1, ry1 = x1/1920, y1/1080
        rx2, ry2 = x2/1920, y2/1080
        w, h = self.config.resolution
        # Randomizar duração para parecer humano
        d = duration + random.randint(-30, 30)
        if self.humanization_system and getattr(self.humanization_system, "config", None) and self.humanization_system.config.enabled:
            try:
                d += int(self.humanization_system.get_delay("movement") * 1000)
            except Exception as e:
                logger.debug(f"[SWIPE] Humanization fallback: {e}")
        real_x1, real_y1 = int(rx1*w), int(ry1*h)
        real_x2, real_y2 = int(rx2*w), int(ry2*h)
        result = self.adb.swipe(real_x1, real_y1, real_x2, real_y2, d)

        # Record swipe for behavioral biometrics
        if result and self.safety_system:
            try:
                self.safety_system.record_swipe(float(real_x1), float(real_y1), float(real_x2), float(real_y2), float(d/1000))
                logger.debug(f"[SWIPE] Swipe registrado no safety_system")
            except Exception as e:
                logger.warning(f"[SWIPE] Falha ao registrar swipe no safety_system: {e}")

        return result
    
    def keyevent(self, keycode: int) -> bool:
        return self.adb.keyevent(keycode)

    def get_screenshot(self) -> Optional[bytes]:
        logger.debug("[EMULATOR] Solicitando screenshot via ADB")
        return self.adb.screenshot()
    
    def randomize_window_periodically(self, interval: int = 300):
        """Randomiza posição da janela periodicamente para anti-detection."""
        try:
            if not hasattr(self, '_last_window_randomize'):
                self._last_window_randomize = 0.0
            now = time.time()
            if now - self._last_window_randomize < interval:
                return
            self._last_window_randomize = now
            # Mover janela ligeiramente (±20px) para evitar fingerprinting por posição
            rect = self.window.get_rect()
            if rect:
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
                new_x = max(0, rect[0] + dx)
                new_y = max(0, rect[1] + dy)
                self.window.move(new_x, new_y)
                logger.debug(f"[EMULATOR] Janela reposicionada para ({new_x}, {new_y})")
        except Exception as e:
            logger.debug(f"[EMULATOR] Falha ao randomizar janela: {e}")

    def get_status_snapshot(self) -> dict:
        """Retorna um snapshot leve de janela/ADB para diagnósticos."""
        rect = self.window.get_rect()
        title = self.window.get_title()
        return {
            "connected": self.is_connected,
            "window_title": title or self.window.window_title,
            "window_active": self.window.is_focused(),
            "window_visible": self.window.is_visible(),
            "window_rect": rect,
            "resolution": self.config.resolution,
            "adb_device_id": self.adb.device_id,
        }
