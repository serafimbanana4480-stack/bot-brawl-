"""
screenshot_taker.py

Captura screenshots do emulador.
"""

import numpy as np
from PIL import Image
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ScreenshotTaker:
    """Captura screenshots da janela do emulador"""

    DEFAULT_TITLES = (
        "BlueStacks App Player",
        "BlueStacks",
        "HD-Player",
        "LDPlayer",
        "NoxPlayer",
        "MEmu",
        "MuMuPlayer",
        "GameLoop",
    )

    def __init__(self, window_title="auto"):
        self.window_title = window_title
        self.window_handle = None

    def _candidate_titles(self):
        """Return window titles to probe, preserving caller preference first."""
        titles = []
        if self.window_title and self.window_title != "auto":
            titles.append(self.window_title)
        titles.extend(self.DEFAULT_TITLES)
        seen = set()
        ordered = []
        for title in titles:
            if title not in seen:
                ordered.append(title)
                seen.add(title)
        return ordered

    def _is_likely_emulator_process(self, hwnd: int) -> bool:
        """Verifica se o processo da janela é realmente um emulador conhecido."""
        try:
            import win32process
            import psutil
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc = psutil.Process(pid)
            exe = proc.name().lower()
            known_exes = {
                "hd-player.exe", "bluestacks.exe", "bstacks.exe",
                "ldplayer.exe", "ldplayer9.exe", "ldplayer64.exe",
                "nox.exe", "noxplayer.exe", "noxvmhandle.exe",
                "memu.exe", "memuc.exe", "memuplayer.exe",
                "mumuplayer.exe", "mumuvm.exe",
                "appmarket.exe", "appplayer.exe",
            }
            return exe in known_exes
        except Exception:
            return False

    def _reject_title(self, title: str) -> bool:
        """Rejeita títulos que claramente NÃO são a janela principal do emulador."""
        t = title.lower()
        # Janelas de sistema
        if title in ["Program Manager", "Start", "Taskbar", "Settings", "Microsoft Store"]:
            return True
        # Instaladores, updaters, notificações, overlays
        rejection_keywords = [
            "installer", "update", "updater", "download", "setup",
            "notification", "toast", "overlay", "tooltip",
            "settings", "configur", "preferences",
            "chrome", "firefox", "edge", "opera", "brave",
            "explorer", "file explorer", "notepad", "cmd", "powershell",
            "discord", "telegram", "whatsapp", "teams", "skype",
            "spotify", "vlc", "media player",
        ]
        for kw in rejection_keywords:
            if kw in t:
                return True
        return False

    def _match_score(self, title: str) -> int:
        """Devolve um score de correspondência; maior = melhor match."""
        t = title.lower()
        # Correspondência exata com título configurado
        if self.window_title and self.window_title != "auto":
            if self.window_title.lower() == t:
                return 1000
            if self.window_title.lower() in t:
                return 500
        # Correspondência exata com títulos conhecidos
        for known in self.DEFAULT_TITLES:
            if known.lower() == t:
                return 900
        # Substring com títulos conhecidos (preferir início do título)
        for known in self.DEFAULT_TITLES:
            kl = known.lower()
            if t.startswith(kl):
                return 400
            if kl in t:
                return 200
        return 0

    def find_window(self) -> bool:
        """Encontra a janela do emulador com validação rigorosa."""
        try:
            import win32gui

            possible_titles = self._candidate_titles()

            # 1. Tentar correspondência exata para os títulos possíveis
            for title in possible_titles:
                hwnd = win32gui.FindWindow(None, title)
                if hwnd != 0:
                    self.window_handle = hwnd
                    logger.info(f"Janela encontrada por correspondência exata: {title} (hwnd: {hwnd})")
                    return True

            # 2. Se título foi explicitamente configurado e FindWindow falhou,
            #    fazemos fallback PARCIAL mas restrito a esse título apenas.
            if self.window_title and self.window_title != "auto":
                logger.warning(
                    f"FindWindow falhou para título configurado '{self.window_title}'. "
                    f"A procurar janelas que contenham esse título..."
                )
            else:
                logger.info("FindWindow falhou para todos os títulos conhecidos. A procurar emuladores via EnumWindows...")

            matches = []

            def enum_callback(hwnd, result):
                if not win32gui.IsWindowVisible(hwnd):
                    return True
                title = win32gui.GetWindowText(hwnd)
                if not title:
                    return True
                if self._reject_title(title):
                    return True

                score = self._match_score(title)
                if score == 0:
                    return True

                rect = win32gui.GetWindowRect(hwnd)
                width = rect[2] - rect[0]
                height = rect[3] - rect[1]
                if width <= 100 or height <= 100:
                    return True

                # Bonus se o processo for confirmado como emulador
                proc_confirmed = self._is_likely_emulator_process(hwnd)
                if proc_confirmed:
                    score += 100

                area = width * height
                result.append((hwnd, title, score, area, proc_confirmed, width, height))
                return True

            win32gui.EnumWindows(enum_callback, matches)

            if matches:
                # Ordenar por score decrescente, depois por área decrescente
                matches.sort(key=lambda x: (x[2], x[3]), reverse=True)
                # Log de todos os candidatos para diagnóstico
                logger.info(f"[EnumWindows] Candidatos encontrados: {len(matches)}")
                for m in matches[:10]:
                    hwnd, title, score, area, proc_confirmed, w, h = m
                    logger.info(
                        f"  - hwnd={hwnd} title='{title}' score={score} "
                        f"area={area} proc_ok={proc_confirmed} dims={w}x{h}"
                    )
                self.window_handle, found_title, score, area, proc_confirmed, _, _ = matches[0]
                logger.info(
                    f"Janela selecionada: '{found_title}' (hwnd={self.window_handle}, "
                    f"score={score}, area={area}, proc_ok={proc_confirmed})"
                )
                return True

            logger.error("Nenhuma janela de emulador encontrada via EnumWindows.")
            return False
        except Exception as e:
            logger.error(f"Erro ao encontrar janela: {e}", exc_info=True)
            return False

    def take(self) -> Optional[np.ndarray]:
        """Captura screenshot e retorna como array numpy usando mss"""
        try:
            import mss
            import win32gui
            import win32con
            import win32api
            import cv2
            import ctypes
            
            logger.info(f"[SCREENSHOT] Iniciando captura, window_title={self.window_title}")
            
            # Garantir consciência de DPI no Windows
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except (OSError, AttributeError):
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except (OSError, AttributeError) as e:
                    logger.debug(f'[SCREENSHOT] DPI awareness fallback unavailable: {e}')

            if not self.window_handle or not win32gui.IsWindow(self.window_handle):
                logger.info("[SCREENSHOT] Janela não encontrada, tentando encontrar...")
                if not self.find_window():
                    logger.error("[SCREENSHOT] Não foi possível encontrar janela")
                    return None
                logger.info(f"[SCREENSHOT] Janela encontrada: hwnd={self.window_handle}")

            # Obter coordenadas da janela principal
            rect = win32gui.GetWindowRect(self.window_handle)
            logger.info(f"[SCREENSHOT] Janela rect: {rect}")
            
            # Verificar se a janela está minimizada
            if win32gui.IsIconic(self.window_handle):
                logger.warning("[SCREENSHOT] Janela do emulador está minimizada!")
                return None

            # Em muitos emuladores (BlueStacks), o conteúdo real está numa janela filha
            # Vamos tentar encontrar a janela de renderização para evitar toolbars
            child_hwnd = 0
            def find_render_window(hwnd, _):
                nonlocal child_hwnd
                child_rect = win32gui.GetWindowRect(hwnd)
                w = child_rect[2] - child_rect[0]
                h = child_rect[3] - child_rect[1]
                # Procurar janela que pareça ter o rácio 16:9 ou o tamanho do jogo
                if w > 400 and h > 200:
                    if abs(w/h - 16/9) < 0.1 or abs(w/h - 1920/1080) < 0.1:
                        child_hwnd = hwnd
                        return False # Parar busca
                return True
            
            try:
                win32gui.EnumChildWindows(self.window_handle, find_render_window, None)
            except Exception as e:
                logger.debug(f"[SCREENSHOT] Child window search failed: {e}")
            
            # Se encontrou janela filha de renderização, usar o rect dela
            if child_hwnd != 0:
                rect = win32gui.GetWindowRect(child_hwnd)
                logger.info(f"[SCREENSHOT] Usando janela filha: hwnd={child_hwnd}, rect={rect}")

            monitor = {
                "top": rect[1],
                "left": rect[0],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1],
            }

            if monitor["width"] <= 10 or monitor["height"] <= 10:
                logger.error(f"[SCREENSHOT] Dimensões inválidas: {monitor}")
                self.window_handle = None
                return None

            logger.info(f"[SCREENSHOT] Capturando região: {monitor}")

            with mss.mss() as sct:
                # Capturar região específica
                sct_img = sct.grab(monitor)
                
                # Converter para numpy array (BGRA para RGB)
                img = np.array(sct_img)
                img = img[:, :, :3] # Remover alpha
                img = np.flip(img, axis=2) # BGR para RGB (mss retorna BGRA)
                
                logger.info(f"[SCREENSHOT] Imagem capturada: shape={img.shape}")

                # VALIDAÇÃO: rejeitar capturas suspeitas (tudo preto, tudo branco, desktop, etc.)
                if not self._validate_screenshot(img):
                    logger.error("[SCREENSHOT] Captura rejeitada pela validação — possivelmente janela errada.")
                    self.window_handle = None  # Forçar re-encontrar janela no próximo tick
                    return None

                # NORMALIZAÇÃO: Redimensionar para 1920x1080
                if img.shape[0] != 1080 or img.shape[1] != 1920:
                    logger.info(f"[SCREENSHOT] Redimensionando de {img.shape} para (1080, 1920)")
                    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)

                logger.info(f"[SCREENSHOT] Captura concluída com sucesso")
                return img

        except Exception as e:
            logger.error(f"[SCREENSHOT] Erro ao capturar screenshot via mss: {e}", exc_info=True)
            return None

    def _validate_screenshot(self, img: np.ndarray) -> bool:
        """Valida se a screenshot parece ser de um jogo/emulador e não de conteúdo errado."""
        try:
            h, w = img.shape[:2]
            # 1. Dimensões mínimas
            if h < 200 or w < 200:
                logger.warning(f"[VALIDATE] Dimensões muito pequenas: {w}x{h}")
                return False

            # 2. Proporção deve ser landscape (w > h) e não extrema
            aspect = w / h if h > 0 else 0
            if aspect < 0.5 or aspect > 4.0:
                logger.warning(f"[VALIDATE] Proporção inválida: {aspect:.2f}")
                return False

            # 3. Rejeitar imagem totalmente preta ou totalmente branca
            mean_val = float(img.mean())
            if mean_val < 5.0:
                logger.warning(f"[VALIDATE] Imagem quase totalmente preta (mean={mean_val:.1f})")
                return False
            if mean_val > 250.0:
                logger.warning(f"[VALIDATE] Imagem quase totalmente branca (mean={mean_val:.1f})")
                return False

            # 4. Rejeitar se 95%+ dos pixels são exatamente a mesma cor (desktop sólido?)
            # Converte para um hash simples: modo de cor mais comum
            # Usar amostra para performance
            sample = img[::10, ::10]
            unique_colors = len(np.unique(sample.reshape(-1, sample.shape[2]), axis=0))
            total_pixels = sample.shape[0] * sample.shape[1]
            if unique_colors < max(50, total_pixels * 0.001):
                logger.warning(f"[VALIDATE] Imagem com pouquíssima variedade de cores ({unique_colors} únicas)")
                return False

            return True
        except Exception as e:
            logger.warning(f"[VALIDATE] Erro na validação: {e}")
            # Em caso de erro na validação, permite a captura (fail-open)
            return True

    def take_pil(self) -> Optional[Image.Image]:
        """Captura screenshot como PIL Image"""
        img_array = self.take()
        if img_array is not None:
            return Image.fromarray(img_array)
        return None
