"""
core/resolution_manager.py

Sistema centralizado de gestão de resolução para o Soberana Omega.

Responsabilidades:
- Detetar resolução real do emulador (Win32 / ADB)
- Manter resolução canónica (1920x1080) para processamento de visão
- Fornecer funções de escala bidirecional (actual <-> canónica)
- Validar resoluções detetadas (razoabilidade, aspect ratio)
- Detetar mudanças de resolução em runtime e alertar
- Centralizar todas as coordenadas hardcoded numa única fonte de verdade

Arquitetura:
    ┌─────────────────┐     detect     ┌──────────────────┐
    │  EmulatorWindow │ ─────────────> │ ResolutionManager│
    │  (Win32/ADB)    │                │                  │
    └─────────────────┘                │  - actual_res    │
                                      │  - canonical_res │
    ┌─────────────────┐    scale     │  - scale_factors │
    │  VisionPipeline │ <────────────│  - validators    │
    │  (1920x1080)    │              └──────────────────┘
    └─────────────────┘                    │
                                            │ scale
    ┌─────────────────┐    scale     ┌───────┘
    │  ADBController  │ <───────────┘
    │  (actual res)   │
    └─────────────────┘

Uso:
    from core.resolution_manager import ResolutionManager

    rm = ResolutionManager(window_title="LDPlayer")
    rm.detect()  # Deteta resolução real

    # Vision: trabalha em canónico (já feito pelo ScreenshotTaker)
    # Input: converte de canónico para actual
    actual_x, actual_y = rm.from_canonical(x_1080, y_1080)
    emulator_controller.tap(actual_x, actual_y)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Resolução canónica usada internamente pelo pipeline de visão
CANONICAL_W = 1920
CANONICAL_H = 1080

# Limites de validação de resolução
MIN_WIDTH = 640
MIN_HEIGHT = 360
MAX_WIDTH = 5120
MAX_HEIGHT = 2880
MIN_ASPECT_RATIO = 1.3  # ~4:3
MAX_ASPECT_RATIO = 2.4  # ~21:9


@dataclass
class ResolutionProfile:
    """Perfil de resolução com metadados de confiança."""

    actual_width: int
    actual_height: int
    canonical_width: int = CANONICAL_W
    canonical_height: int = CANONICAL_H
    dpi_scale: float = 1.0
    validated: bool = False
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"  # "win32", "adb", "config", "fallback"
    change_detected: bool = False
    previous_actual: Optional[Tuple[int, int]] = None

    @property
    def actual_resolution(self) -> Tuple[int, int]:
        return (self.actual_width, self.actual_height)

    @property
    def canonical_resolution(self) -> Tuple[int, int]:
        return (self.canonical_width, self.canonical_height)

    @property
    def scale_x(self) -> float:
        """Fator de escala X: canonical -> actual."""
        return self.actual_width / self.canonical_width

    @property
    def scale_y(self) -> float:
        """Fator de escala Y: canonical -> actual."""
        return self.actual_height / self.canonical_height

    @property
    def aspect_ratio(self) -> float:
        return self.actual_width / max(self.actual_height, 1)

    def is_reasonable(self) -> bool:
        """Verifica se a resolução está dentro de limites razoáveis."""
        w, h = self.actual_width, self.actual_height
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return False
        if w > MAX_WIDTH or h > MAX_HEIGHT:
            return False
        ar = self.aspect_ratio
        if ar < MIN_ASPECT_RATIO or ar > MAX_ASPECT_RATIO:
            return False
        return True


class ResolutionManager:
    """
    Gestor centralizado de resolução.

    Garante que todos os módulos usam coordenadas consistentes,
    independentemente da resolução real do emulador.
    """

    def __init__(
        self,
        window_title: str = "auto",
        canonical_resolution: Tuple[int, int] = (CANONICAL_W, CANONICAL_H),
        enable_change_detection: bool = True,
        change_check_interval_sec: float = 5.0,
        on_resolution_change: Optional[Callable[[ResolutionProfile], None]] = None,
    ):
        self.window_title = window_title
        self.canonical_resolution = canonical_resolution
        self.enable_change_detection = enable_change_detection
        self.change_check_interval_sec = change_check_interval_sec
        self.on_resolution_change = on_resolution_change

        self._profile: Optional[ResolutionProfile] = None
        self._last_check_time: float = 0.0
        self._window_handle: Optional[int] = None

        # Cache de janelas já encontradas para performance
        self._window_cache: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Propriedades de conveniência
    # ------------------------------------------------------------------
    @property
    def profile(self) -> ResolutionProfile:
        """Retorna o perfil atual, forçando deteção se necessário."""
        if self._profile is None:
            self.detect()
        # O type checker não sabe que detect() preenche _profile,
        # mas garantimos com fallback explícito abaixo.
        if self._profile is None:
            self._profile = self._fallback_profile()
        return self._profile

    @property
    def actual_resolution(self) -> Tuple[int, int]:
        return self.profile.actual_resolution

    @property
    def canonical_w(self) -> int:
        return self.canonical_resolution[0]

    @property
    def canonical_h(self) -> int:
        return self.canonical_resolution[1]

    @property
    def scale_x(self) -> float:
        return self.profile.scale_x

    @property
    def scale_y(self) -> float:
        return self.profile.scale_y

    # ------------------------------------------------------------------
    # Deteção de resolução
    # ------------------------------------------------------------------
    def detect(self, force: bool = False) -> ResolutionProfile:
        """
        Deteta resolução real do emulador.

        Ordem de prioridade:
        1. Win32 GetWindowRect (mais fiável para visualização)
        2. ADB wm size (fallback se Win32 falhar)
        3. Configuração guardada (último recurso)
        4. Fallback canónico (1920x1080)
        """
        if not force and self._profile is not None:
            if time.time() - self._last_check_time < self.change_check_interval_sec:
                return self._profile

        # Tentar Win32 primeiro
        profile = self._detect_win32()
        if profile and profile.is_reasonable():
            self._set_profile(profile)
            return self._profile  # type: ignore[return-value]

        # Fallback: ADB
        profile = self._detect_adb()
        if profile and profile.is_reasonable():
            self._set_profile(profile)
            return self._profile  # type: ignore[return-value]

        # Fallback: config ou canónico
        self._set_profile(self._fallback_profile())
        return self._profile  # type: ignore[return-value]

    def _detect_win32(self) -> Optional[ResolutionProfile]:
        """Deteta resolução via Win32 API."""
        try:
            import win32gui
        except ImportError:
            logger.debug("[RES] win32gui não disponível")
            return None

        hwnd = self._find_window_handle()
        if hwnd is None or hwnd == 0:
            return None

        try:
            rect = win32gui.GetWindowRect(hwnd)
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]

            # Em emuladores, procurar janela filha de renderização
            child_hwnd = self._find_render_child(hwnd)
            if child_hwnd:
                child_rect = win32gui.GetWindowRect(child_hwnd)
                cw = child_rect[2] - child_rect[0]
                ch = child_rect[3] - child_rect[1]
                if cw > 400 and ch > 200:
                    w, h = cw, ch

            profile = ResolutionProfile(
                actual_width=w,
                actual_height=h,
                canonical_width=self.canonical_w,
                canonical_height=self.canonical_h,
                source="win32",
            )
            logger.info(f"[RES] Win32 detectado: {w}x{h} (hwnd={hwnd})")
            return profile
        except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.debug(f"[RES] Win32 detection failed: {e}")
            return None

    def _detect_adb(self) -> Optional[ResolutionProfile]:
        """Deteta resolução via ADB wm size."""
        try:
            import subprocess
            result = subprocess.run(
                ["adb", "shell", "wm", "size"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse output: "Physical size: 1920x1080"
                for line in result.stdout.splitlines():
                    if "Physical size:" in line:
                        size_part = line.split(":")[-1].strip()
                        w_str, h_str = size_part.split("x")
                        w, h = int(w_str), int(h_str)
                        profile = ResolutionProfile(
                            actual_width=w,
                            actual_height=h,
                            canonical_width=self.canonical_w,
                            canonical_height=self.canonical_h,
                            source="adb",
                        )
                        logger.info(f"[RES] ADB detectado: {w}x{h}")
                        return profile
        except (ImportError, ModuleNotFoundError, ConnectionError, TimeoutError, TypeError, RuntimeError, OSError) as e:
            logger.debug(f"[RES] ADB detection failed: {e}")
        return None

    def _fallback_profile(self) -> ResolutionProfile:
        """Perfil de fallback canónico."""
        logger.warning(
            f"[RES] Usando fallback canónico {self.canonical_w}x{self.canonical_h}"
        )
        return ResolutionProfile(
            actual_width=self.canonical_w,
            actual_height=self.canonical_h,
            canonical_width=self.canonical_w,
            canonical_height=self.canonical_h,
            source="fallback",
        )

    def _set_profile(self, profile: ResolutionProfile) -> None:
        """Define perfil atual com deteção de mudanças."""
        if self._profile is not None:
            old = self._profile.actual_resolution
            new = profile.actual_resolution
            if old != new and profile.source != "fallback":
                profile.change_detected = True
                profile.previous_actual = old
                logger.warning(
                    f"[RES] Mudança de resolução detetada: {old} -> {new}"
                )
                if self.on_resolution_change:
                    try:
                        self.on_resolution_change(profile)
                    except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                        logger.error(f"[RES] on_resolution_change callback error: {e}")

        # Validar se é razoável
        if profile.is_reasonable():
            profile.validated = True

        self._profile = profile
        self._last_check_time = time.time()

    # ------------------------------------------------------------------
    # Helpers Win32
    # ------------------------------------------------------------------
    def _find_window_handle(self) -> Optional[int]:
        """Encontra handle da janela do emulador."""
        try:
            import win32gui
        except ImportError:
            return None

        # Cache hit
        if self.window_title in self._window_cache:
            cached = self._window_cache[self.window_title]
            if win32gui.IsWindow(cached):
                return cached
            del self._window_cache[self.window_title]

        titles = [self.window_title] if self.window_title and self.window_title != "auto" else []
        titles.extend([
            "BlueStacks App Player", "BlueStacks", "HD-Player",
            "LDPlayer", "NoxPlayer", "MEmu", "MuMuPlayer", "GameLoop",
        ])

        for title in titles:
            hwnd = win32gui.FindWindow(None, title)
            if hwnd != 0:
                self._window_cache[self.window_title] = hwnd
                return hwnd

        # Busca por substring
        def _enum_cb(hwnd, result):
            if win32gui.IsWindowVisible(hwnd):
                t = win32gui.GetWindowText(hwnd)
                for kw in titles:
                    if kw.lower() in t.lower():
                        r = win32gui.GetWindowRect(hwnd)
                        if (r[2] - r[0]) > 100 and (r[3] - r[1]) > 100:
                            result.append((hwnd, t, (r[2] - r[0]) * (r[3] - r[1])))
            return True

        matches: List = []
        win32gui.EnumWindows(_enum_cb, matches)
        if matches:
            matches.sort(key=lambda x: x[2], reverse=True)
            self._window_cache[self.window_title] = matches[0][0]
            return matches[0][0]

        return None

    def _find_render_child(self, parent_hwnd: int) -> Optional[int]:
        """Procura janela filha que parece ser a área de renderização."""
        try:
            import win32gui
            child_hwnd = 0

            def _child_cb(hwnd, _):
                nonlocal child_hwnd
                rect = win32gui.GetWindowRect(hwnd)
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                if w > 400 and h > 200:
                    ratio = w / max(h, 1)
                    if abs(ratio - 16 / 9) < 0.15 or abs(ratio - self.canonical_w / self.canonical_h) < 0.15:
                        child_hwnd = hwnd
                        return False
                return True

            win32gui.EnumChildWindows(parent_hwnd, _child_cb, None)
            return child_hwnd if child_hwnd != 0 else None
        except (ImportError, ModuleNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError):
            return None

    # ------------------------------------------------------------------
    # Escala de coordenadas
    # ------------------------------------------------------------------
    def to_canonical(
        self, x: int, y: int
    ) -> Tuple[int, int]:
        """Converte coordenadas reais (actual) para canónicas (1920x1080)."""
        p = self.profile
        cx = round(x * p.canonical_width / max(p.actual_width, 1))
        cy = round(y * p.canonical_height / max(p.actual_height, 1))
        return (cx, cy)

    def from_canonical(
        self, x: int, y: int
    ) -> Tuple[int, int]:
        """Converte coordenadas canónicas (1920x1080) para reais (actual)."""
        p = self.profile
        ax = round(x * p.actual_width / max(p.canonical_width, 1))
        ay = round(y * p.actual_height / max(p.canonical_height, 1))
        return (ax, ay)

    def scale_relative_to_actual(
        self, rx: float, ry: float
    ) -> Tuple[int, int]:
        """
        Converte coordenadas relativas (0.0–1.0) para pixels reais.
        Usado pelos módulos que já trabalham com coordenadas normalizadas.
        """
        p = self.profile
        ax = round(rx * p.actual_width)
        ay = round(ry * p.actual_height)
        return (ax, ay)

    def scale_relative_to_canonical(
        self, rx: float, ry: float
    ) -> Tuple[int, int]:
        """
        Converte coordenadas relativas (0.0–1.0) para pixels canónicos.
        """
        cx = round(rx * self.canonical_w)
        cy = round(ry * self.canonical_h)
        return (cx, cy)

    def scale_roi_to_actual(
        self, roi: Tuple[float, float, float, float]
    ) -> Tuple[int, int, int, int]:
        """
        Converte ROI normalizada (x1, y1, x2, y2 em 0-1) para pixels reais.
        """
        p = self.profile
        x1 = round(roi[0] * p.actual_width)
        y1 = round(roi[1] * p.actual_height)
        x2 = round(roi[2] * p.actual_width)
        y2 = round(roi[3] * p.actual_height)
        return (x1, y1, x2, y2)

    def scale_roi_to_canonical(
        self, roi: Tuple[float, float, float, float]
    ) -> Tuple[int, int, int, int]:
        """Converte ROI normalizada para pixels canónicos."""
        x1 = round(roi[0] * self.canonical_w)
        y1 = round(roi[1] * self.canonical_h)
        x2 = round(roi[2] * self.canonical_w)
        y2 = round(roi[3] * self.canonical_h)
        return (x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # Atualização runtime
    # ------------------------------------------------------------------
    def check_for_changes(self) -> bool:
        """Verifica se a resolução mudou desde a última deteção."""
        if not self.enable_change_detection:
            return False
        if time.time() - self._last_check_time < self.change_check_interval_sec:
            return False

        old_profile = self._profile
        new_profile = self.detect(force=True)

        if old_profile is None:
            return new_profile.change_detected

        return old_profile.actual_resolution != new_profile.actual_resolution

    def update_window_title(self, window_title: str) -> None:
        """Atualiza título da janela e invalida cache."""
        self.window_title = window_title
        self._window_cache.clear()
        self._window_handle = None
        self._profile = None

    def invalidate(self) -> None:
        """Invalida perfil atual, forçando nova deteção."""
        self._profile = None
        self._window_handle = None
