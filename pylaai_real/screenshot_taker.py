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

    def __init__(self, window_title="LDPlayer"):
        self.window_title = window_title
        self.window_handle = None

    def find_window(self) -> bool:
        """Encontra a janela do emulador"""
        try:
            import win32gui
            
            # Ordem de preferência de títulos para BlueStacks e outros
            possible_titles = [
                self.window_title,
                "BlueStacks App Player",
                "BlueStacks",
                "HD-Player",
                "LDPlayer",
                "NoxPlayer"
            ]
            
            # 1. Tentar correspondência exata para os títulos possíveis
            for title in possible_titles:
                hwnd = win32gui.FindWindow(None, title)
                if hwnd != 0:
                    self.window_handle = hwnd
                    logger.info(f"Janela encontrada por correspondência exata: {title} (hwnd: {hwnd})")
                    return True
            
            # 2. Se não encontrar, procura por janelas que contenham o título
            def enum_callback(hwnd, result):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if not title: return True
                    
                    # Ignorar janelas de sistema ou do desktop
                    if title in ["Program Manager", "Start", "Taskbar"]: return True
                    
                    # Procurar por títulos que contenham palavras-chave do emulador
                    keywords = [self.window_title.lower()]
                    if "bluestacks" in self.window_title.lower():
                        keywords.extend(["bluestacks app player", "hd-player"])
                    
                    for kw in keywords:
                        if kw in title.lower():
                            # Verificar se a janela tem um tamanho razoável (evitar janelas de background)
                            rect = win32gui.GetWindowRect(hwnd)
                            width = rect[2] - rect[0]
                            height = rect[3] - rect[1]
                            if width > 100 and height > 100:
                                result.append((hwnd, title, width * height))
                return True
            
            matches = []
            win32gui.EnumWindows(enum_callback, matches)
            
            if matches:
                # Ordenar por tamanho da janela (provavelmente a maior é o player principal)
                matches.sort(key=lambda x: x[2], reverse=True)
                self.window_handle, found_title, _ = matches[0]
                logger.info(f"Janela encontrada por título parcial: '{found_title}' (hwnd: {self.window_handle})")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Erro ao encontrar janela: {e}")
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
                except (OSError, AttributeError):
                    pass

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
            except Exception:
                pass
            
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
                
                # NORMALIZAÇÃO: Redimensionar para 1920x1080
                if img.shape[0] != 1080 or img.shape[1] != 1920:
                    logger.info(f"[SCREENSHOT] Redimensionando de {img.shape} para (1080, 1920)")
                    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
                
                logger.info(f"[SCREENSHOT] Captura concluída com sucesso")
                return img

        except Exception as e:
            logger.error(f"[SCREENSHOT] Erro ao capturar screenshot via mss: {e}", exc_info=True)
            return None

    def take_pil(self) -> Optional[Image.Image]:
        """Captura screenshot como PIL Image"""
        img_array = self.take()
        if img_array is not None:
            return Image.fromarray(img_array)
        return None
