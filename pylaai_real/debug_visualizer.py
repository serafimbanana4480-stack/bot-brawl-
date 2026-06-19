"""
debug_visualizer.py

Modo de debug visual com OpenCV.
Mostra em tempo real o que o bot está "vendo" e suas decisões.

Funcionalidades:
- Visualização de detecções YOLO (bounding boxes, confiança)
- Visualização de estado atual e confiança
- Visualização de ações planejadas (movimento, ataque)
- Visualização de leading shots e predições
- Visualização de histórico de estados
- Controles interativos (pausar, step-by-step)
- Gravação de sessão de debug
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque

logger = logging.getLogger(__name__)


class DebugMode(Enum):
    """Modos de debug disponíveis."""
    OFF = "off"
    BASIC = "basic"  # Apenas estado e detecções principais
    DETAILED = "detailed"  # Todas as detecções e ações
    COMBAT = "combat"  # Foco em combate (inimigos, leading shots)
    FULL = "full"  # Tudo


@dataclass
class DebugOverlay:
    """Informações para sobrepor na imagem de debug."""
    state: str = "unknown"
    state_confidence: float = 0.0
    enemies: List[dict] = field(default_factory=list)
    player_bbox: Optional[Tuple[int, int, int, int]] = None
    actions: List[str] = field(default_factory=list)
    leading_shots: List[Tuple[int, int]] = field(default_factory=list)
    kiting_vector: Optional[Tuple[float, float]] = None
    cover_zones: List[Tuple[int, int, int, int]] = field(default_factory=list)
    fps: float = 0.0
    cycle_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


class DebugVisualizer:
    """
    Visualizador de debug com OpenCV.
    
    Mostra:
    - Screenshot atual com detecções
    - Estado atual e confiança
    - Bounding boxes de inimigos
    - Ações planejadas
    - Leading shots
    - Vetores de movimento
    - Informações de performance
    """
    
    def __init__(
        self,
        window_name: str = "Bot Debug",
        mode: DebugMode = DebugMode.DETAILED,
        enable_recording: bool = False,
        recording_dir: str = "data/debug_recordings"
    ):
        self.window_name = window_name
        self.mode = mode
        self.enable_recording = enable_recording
        self.recording_dir = recording_dir
        
        # Estado do visualizer
        self.is_running = False
        self.is_paused = False
        self.step_mode = False
        self.current_overlay: Optional[DebugOverlay] = None
        
        # Histórico para gráficos
        self.state_history = deque(maxlen=50)
        self.fps_history = deque(maxlen=50)
        self.action_history = deque(maxlen=20)
        
        # Gravação
        self.video_writer = None
        self.recording_start_time = 0.0
        
        # Cores para visualização
        self.colors = {
            "enemy": (0, 0, 255),      # Vermelho
            "player": (0, 255, 0),     # Verde
            "bush": (0, 255, 255),     # Amarelo
            "cube": (255, 0, 255),     # Magenta
            "leading_shot": (255, 255, 0),  # Ciano
            "kiting": (255, 165, 0),   # Laranja
            "cover": (128, 0, 128),    # Roxo
            "text": (255, 255, 255),   # Branco
            "background": (0, 0, 0)    # Preto
        }
        
        # Thread de renderização
        self.render_thread = None
        self.overlay_lock = threading.Lock()
        
        logger.info(f"[DEBUG] Visualizer inicializado: mode={mode.value}")
    
    def start(self):
        """Inicia o visualizer."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Criar janela
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        # Iniciar thread de renderização
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()
        
        # Iniciar gravação se habilitado
        if self.enable_recording:
            self._start_recording()
        
        logger.info("[DEBUG] Visualizer iniciado")
    
    def stop(self):
        """Para o visualizer."""
        self.is_running = False
        
        if self.render_thread:
            self.render_thread.join(timeout=1.0)
        
        # Parar gravação
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # Destruir janela (com guarda contra NULL pointer do OpenCV headless)
        try:
            if self.window_name:
                try:
                    cv2.destroyWindow(self.window_name)
                except cv2.error:
                    pass  # OpenCV headless: janela nao existe
        except Exception:
            pass
        
        logger.info("[DEBUG] Visualizer parado")
    
    def update_overlay(self, overlay: DebugOverlay):
        """Atualiza informações de sobreposição."""
        with self.overlay_lock:
            self.current_overlay = overlay
            
            # Atualizar históricos
            self.state_history.append(overlay.state)
            self.fps_history.append(overlay.fps)
            self.action_history.extend(overlay.actions)
            
            # Manter tamanho do histórico
            while len(self.action_history) > 20:
                self.action_history.popleft()
    
    def _render_loop(self):
        """Loop de renderização em thread separada."""
        while self.is_running:
            try:
                if not self.is_paused or self.step_mode:
                    self._render_frame()
                
                # Se em step mode, pausar após renderizar um frame
                if self.step_mode:
                    self.is_paused = True
                    self.step_mode = False
                
                time.sleep(0.033)  # ~30 FPS
            
            except Exception as e:
                logger.error(f"[DEBUG] Erro no render loop: {e}")
                time.sleep(0.1)
    
    def _render_frame(self):
        """Renderiza um frame de debug."""
        with self.overlay_lock:
            if self.current_overlay is None:
                return
            
            overlay = self.current_overlay
            
            # Se não tiver screenshot, criar placeholder
            # (na prática, o screenshot viria do overlay)
            if not hasattr(overlay, 'screenshot') or overlay.screenshot is None:
                h, w = 720, 1280
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                frame = overlay.screenshot.copy()
        
        # Redimensionar para display
        frame = cv2.resize(frame, (1280, 720))
        
        # Desenhar overlays baseados no modo
        if self.mode in [DebugMode.BASIC, DebugMode.DETAILED, DebugMode.FULL]:
            self._draw_basic_info(frame, overlay)
        
        if self.mode in [DebugMode.DETAILED, DebugMode.FULL]:
            self._draw_detections(frame, overlay)
        
        if self.mode in [DebugMode.COMBAT, DebugMode.FULL]:
            self._draw_combat_info(frame, overlay)
        
        if self.mode == DebugMode.FULL:
            self._draw_graphs(frame, overlay)
        
        # Mostrar frame
        cv2.imshow(self.window_name, frame)
        
        # Gravar se habilitado
        if self.video_writer:
            self.video_writer.write(frame)
        
        # Processar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Espaço - pausar/continuar
            self.is_paused = not self.is_paused
            logger.info(f"[DEBUG] Pausado: {self.is_paused}")
        elif key == ord('s'):  # S - step
            self.step_mode = True
            logger.info("[DEBUG] Step mode")
        elif key == ord('q'):  # Q - quit
            self.is_running = False
        elif key == ord('1'):  # 1 - modo básico
            self.mode = DebugMode.BASIC
            logger.info("[DEBUG] Modo: BASIC")
        elif key == ord('2'):  # 2 - modo detalhado
            self.mode = DebugMode.DETAILED
            logger.info("[DEBUG] Modo: DETAILED")
        elif key == ord('3'):  # 3 - modo combate
            self.mode = DebugMode.COMBAT
            logger.info("[DEBUG] Modo: COMBAT")
        elif key == ord('4'):  # 4 - modo completo
            self.mode = DebugMode.FULL
            logger.info("[DEBUG] Modo: FULL")
    
    def _draw_basic_info(self, frame: np.ndarray, overlay: DebugOverlay):
        """Desenha informações básicas (estado, FPS, etc.)."""
        
        # Background para texto
        overlay_bg = np.zeros((120, frame.shape[1], 3), dtype=np.uint8)
        overlay_bg[:] = (0, 0, 0)
        frame[0:120, :] = cv2.addWeighted(frame[0:120, :], 0.7, overlay_bg, 0.3, 0)
        
        # Estado atual
        state_text = f"State: {overlay.state}"
        confidence_text = f"Confidence: {overlay.state_confidence:.2f}"
        
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors["text"], 2)
        cv2.putText(frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["text"], 1)
        
        # FPS e ciclo
        fps_text = f"FPS: {overlay.fps:.1f}"
        cycle_text = f"Cycle: {overlay.cycle_time*1000:.0f}ms"
        
        cv2.putText(frame, fps_text, (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["text"], 1)
        cv2.putText(frame, cycle_text, (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["text"], 1)
        
        # Modo atual
        mode_text = f"Mode: {self.mode.value.upper()}"
        cv2.putText(frame, mode_text, (800, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Pausado/Step
        if self.is_paused:
            cv2.putText(frame, "PAUSED", (800, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif self.step_mode:
            cv2.putText(frame, "STEP", (800, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def _draw_detections(self, frame: np.ndarray, overlay: DebugOverlay):
        """Desenha detecções (inimigos, player, etc.)."""
        
        # Player
        if overlay.player_bbox:
            x1, y1, x2, y2 = overlay.player_bbox
            # Escalar para tamanho do frame
            scale_x = frame.shape[1] / 1920  # Assumindo 1920x1080 original
            scale_y = frame.shape[0] / 1080
            
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["player"], 2)
            cv2.putText(frame, "PLAYER", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["player"], 2)
        
        # Inimigos
        for enemy in overlay.enemies:
            bbox = enemy.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                scale_x = frame.shape[1] / 1920
                scale_y = frame.shape[0] / 1080
                
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                # Cor baseada na confiança
                confidence = enemy.get("confidence", 1.0)
                color = (
                    int(255 * (1 - confidence)),
                    int(255 * confidence),
                    0
                )
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"Enemy {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _draw_combat_info(self, frame: np.ndarray, overlay: DebugOverlay):
        """Desenha informações de combate (leading shots, kiting, etc.)."""
        
        # Leading shots
        for shot in overlay.leading_shots:
            x, y = shot
            scale_x = frame.shape[1] / 1920
            scale_y = frame.shape[0] / 1080
            x, y = int(x * scale_x), int(y * scale_y)
            
            cv2.circle(frame, (x, y), 5, self.colors["leading_shot"], -1)
            cv2.circle(frame, (x, y), 10, self.colors["leading_shot"], 2)
        
        # Kiting vector
        if overlay.kiting_vector:
            vx, vy = overlay.kiting_vector
            # Desenhar seta do centro
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            end_x = int(center_x + vx * 0.5)
            end_y = int(center_y + vy * 0.5)
            
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), self.colors["kiting"], 3)
        
        # Cover zones
        for zone in overlay.cover_zones:
            x1, y1, x2, y2 = zone
            scale_x = frame.shape[1] / 1920
            scale_y = frame.shape[0] / 1080
            
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["cover"], 2)
            cv2.putText(frame, "COVER", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["cover"], 2)
        
        # Ações planejadas
        y_offset = 150
        for action in overlay.actions[-5:]:  # Últimas 5 ações
            cv2.putText(frame, f"Action: {action}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 1)
            y_offset += 25
    
    def _draw_graphs(self, frame: np.ndarray, overlay: DebugOverlay):
        """Desenha gráficos de histórico (FPS, estados)."""
        
        # Área para gráficos (canto inferior direito)
        graph_w, graph_h = 300, 150
        graph_x = frame.shape[1] - graph_w - 10
        graph_y = frame.shape[0] - graph_h - 10
        
        # Background
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), 
                    (0, 0, 0), -1)
        
        # Gráfico de FPS
        if len(self.fps_history) > 1:
            fps_points = []
            for i, fps in enumerate(self.fps_history):
                x = graph_x + int(i * graph_w / len(self.fps_history))
                # Normalizar FPS (0-60)
                y = graph_y + graph_h - int(min(fps, 60) * graph_h / 60)
                fps_points.append((x, y))
            
            if len(fps_points) > 1:
                cv2.polylines(frame, [np.array(fps_points)], False, (0, 255, 0), 2)
        
        # Label
        cv2.putText(frame, "FPS", (graph_x, graph_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def _start_recording(self):
        """Inicia gravação de vídeo."""
        import os
        from datetime import datetime
        
        os.makedirs(self.recording_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_{timestamp}.mp4"
        filepath = os.path.join(self.recording_dir, filename)
        
        # Configurar writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filepath, fourcc, 30.0, (1280, 720))
        
        self.recording_start_time = time.time()
        logger.info(f"[DEBUG] Gravação iniciada: {filepath}")
    
    def toggle_pause(self):
        """Alterna pausa."""
        self.is_paused = not self.is_paused
        logger.info(f"[DEBUG] Pausado: {self.is_paused}")
    
    def step(self):
        """Avança um frame (step mode)."""
        self.step_mode = True
        self.is_paused = False
    
    def set_mode(self, mode: DebugMode):
        """Define modo de debug."""
        self.mode = mode
        logger.info(f"[DEBUG] Modo alterado para: {mode.value}")


# Classe de conveniência para integrar com o wrapper
class DebugIntegration:
    """Integração do debug visualizer com o wrapper."""
    
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.visualizer = None
        self.enabled = False
    
    def enable(self, mode: DebugMode = DebugMode.DETAILED):
        """Habilita debug visual."""
        if self.visualizer is None:
            self.visualizer = DebugVisualizer(mode=mode)
        
        self.visualizer.start()
        self.enabled = True
        logger.info("[DEBUG] Debug visual habilitado")
    
    def disable(self):
        """Desabilita debug visual."""
        if self.visualizer:
            self.visualizer.stop()
        self.enabled = False
        logger.info("[DEBUG] Debug visual desabilitado")
    
    def update(self):
        """Atualiza debug visual com estado atual do wrapper."""
        if not self.enabled or self.visualizer is None:
            return
        
        try:
            # Criar overlay com estado atual
            overlay = DebugOverlay()
            
            # Estado
            if self.wrapper.state_manager:
                overlay.state = self.wrapper.state_manager.current_state
                # Confiança seria obtida do detector
                overlay.state_confidence = 0.8  # Placeholder
            
            # Inimigos
            if self.wrapper.play_logic and hasattr(self.wrapper.play_logic, 'last_enemies'):
                overlay.enemies = [
                    {"bbox": e, "confidence": 0.9} 
                    for e in self.wrapper.play_logic.last_enemies
                ]
            
            # Player
            if hasattr(self.wrapper, 'player_bbox'):
                overlay.player_bbox = self.wrapper.player_bbox
            
            # Ações
            if self.wrapper.play_logic and hasattr(self.wrapper.play_logic, '_last_action'):
                overlay.actions = [self.wrapper.play_logic._last_action]
            
            # FPS
            if self.wrapper.observability:
                snap = self.wrapper.observability.get_snapshot()
                overlay.fps = 1000.0 / max(1, snap.cycle_time_ms)
                overlay.cycle_time = snap.cycle_time_ms / 1000.0
            
            # Atualizar visualizer
            self.visualizer.update_overlay(overlay)
        
        except Exception as e:
            logger.debug(f"[DEBUG] Erro ao atualizar overlay: {e}")
