"""
tracker.py

Multi-Object Tracking para Brawl Stars Bot.
Implementação simplificada baseada em SORT (Simple Online and Realtime Tracking).
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)

# Optional movement predictor for advanced prediction
try:
    from vision.movement_predictor import MovementPredictor
    MOVEMENT_PREDICTOR_AVAILABLE = True
except ImportError:
    MOVEMENT_PREDICTOR_AVAILABLE = False


@dataclass
class Track:
    """Representa um objeto rastreado"""
    id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    age: int  # Número de frames desde a última detecção
    hit_streak: int  # Número de frames consecutivos com detecção
    time_since_update: int
    velocity: Tuple[float, float] = (0.0, 0.0)  # Velocidade (vx, vy)
    history: deque = None

    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=10)

    def update(self, bbox: List[int], confidence: float) -> None:
        """Atualiza o track com nova detecção"""
        # Calcular velocidade
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        prev_center_x = (self.bbox[0] + self.bbox[2]) / 2
        prev_center_y = (self.bbox[1] + self.bbox[3]) / 2

        self.velocity = (center_x - prev_center_x, center_y - prev_center_y)
        self.bbox = bbox
        self.confidence = confidence
        self.hit_streak += 1
        self.time_since_update = 0
        self.history.append(bbox)

    def predict(self) -> List[int]:
        """Prediz próxima posição baseada na velocidade"""
        if self.time_since_update > 0:
            # Se não foi atualizado recentemente, extrapolar posição
            dt = self.time_since_update
            x1, y1, x2, y2 = self.bbox
            vx, vy = self.velocity

            # Extrapolar
            new_x1 = x1 + vx * dt
            new_y1 = y1 + vy * dt
            new_x2 = x2 + vx * dt
            new_y2 = y2 + vy * dt

            return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

        return self.bbox

    def mark_missed(self) -> None:
        """Marca que o track não foi detectado neste frame"""
        self.time_since_update += 1
        self.hit_streak = 0


class KalmanFilter:
    """Filtro de Kalman simplificado para tracking"""

    def __init__(self):
        self.x = None  # Estado [x, y, vx, vy]
        self.P = np.eye(4) * 100  # Covariância
        self.F = np.eye(4)  # Matriz de transição
        self.H = np.eye(4, 2)  # Matriz de observação
        self.R = np.eye(2) * 10  # Ruído de medição
        self.Q = np.eye(4) * 0.1  # Ruído de processo

    def initiate(self, bbox: List[int]) -> None:
        """Inicia o filtro com primeira medição"""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.x = np.array([center_x, center_y, 0, 0])  # [x, y, vx, vy]

    def predict(self) -> np.ndarray:
        """Prediz próximo estado"""
        if self.x is None:
            return None

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, bbox: List[int]) -> None:
        """Atualiza com nova medição"""
        if self.x is None:
            self.initiate(bbox)
            return

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        z = np.array([center_x, center_y])

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Atualizar estado
        y = z - (self.H @ self.x)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


class MultiObjectTracker:
    """Tracker multi-objetos simplificado (baseado em SORT)"""

    def __init__(self, max_age: int = 30, min_hits: int = 3):
        self.max_age = max_age  # Máximo de frames sem detecção antes de deletar
        self.min_hits = min_hits  # Mínimo de hits antes de confirmar track
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections: List[Tuple[List[int], float]]) -> List[Track]:
        """
        Atualiza tracker com novas detecções.
        detections: Lista de (bbox, confidence)
        Retorna: Lista de tracks confirmados
        """
        self.frame_count += 1

        # 1. Predição de posições atuais
        for track in self.tracks:
            track.predict()

        # 2. Associação de detecções (Hungarian algorithm simplificado - greedy)
        matched, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(detections)

        # 3. Atualizar tracks com detecções associadas
        for track_idx, det_idx in matched:
            bbox, confidence = detections[det_idx]
            self.tracks[track_idx].update(bbox, confidence)

        # 4. Criar novos tracks para detecções não associadas
        for det_idx in unmatched_detections:
            bbox, confidence = detections[det_idx]
            self._initiate_track(bbox, confidence)

        # 5. Marcar tracks não associados como missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 6. Remover tracks muito antigos
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 7. Retornar apenas tracks confirmados
        confirmed_tracks = [t for t in self.tracks if t.hit_streak >= self.min_hits]

        logger.debug(f"[TRACKER] Frame {self.frame_count}: {len(detections)} detecções, {len(confirmed_tracks)} tracks confirmados")

        return confirmed_tracks

    def _associate_detections_to_tracks(self, detections: List[Tuple[List[int], float]]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associa detecções a tracks usando IoU (Intersection over Union)"""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # Calcular matriz de IoU
        iou_matrix = np.zeros((len(self.tracks), len(detections)))

        for track_idx, track in enumerate(self.tracks):
            track_bbox = track.bbox
            for det_idx, (det_bbox, _) in enumerate(detections):
                iou_matrix[track_idx, det_idx] = self._iou(track_bbox, det_bbox)

        # Greedy matching (simplificado vs Hungarian algorithm)
        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        # Threshold de IoU para associação
        iou_threshold = 0.3

        while True:
            # Encontrar par com maior IoU
            max_iou = 0
            best_pair = None

            for track_idx in unmatched_tracks:
                for det_idx in unmatched_detections:
                    if iou_matrix[track_idx, det_idx] > max_iou:
                        max_iou = iou_matrix[track_idx, det_idx]
                        best_pair = (track_idx, det_idx)

            if best_pair is None or max_iou < iou_threshold:
                break

            matched.append(best_pair)
            unmatched_tracks.remove(best_pair[0])
            unmatched_detections.remove(best_pair[1])

        return matched, unmatched_detections, unmatched_tracks

    def _iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calcula Intersection over Union entre duas bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _initiate_track(self, bbox: List[int], confidence: float) -> None:
        """Inicia novo track"""
        track = Track(
            id=self.next_id,
            bbox=bbox,
            confidence=confidence,
            age=0,
            hit_streak=1,
            time_since_update=0
        )
        self.tracks.append(track)
        self.next_id += 1
        logger.debug(f"[TRACKER] Novo track iniciado: ID={track.id}, bbox={bbox}")

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Retorna track por ID"""
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None

    def get_all_tracks(self) -> List[Track]:
        """Retorna todos os tracks (incluindo não confirmados)"""
        return self.tracks

    def get_stats(self) -> Dict:
        """Retorna estatísticas do tracker"""
        return {
            "active_tracks": len(self.tracks),
            "frame_count": self.frame_count,
            "next_track_id": self.next_id,
        }

    def reset(self) -> None:
        """Reseta o tracker"""
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        logger.info("[TRACKER] Tracker resetado")


class EnemyTracker(MultiObjectTracker):
    """Tracker especializado para inimigos no Brawl Stars"""

    def __init__(self, max_age: int = 30, min_hits: int = 2, use_advanced_prediction: bool = False):
        super().__init__(max_age, min_hits)
        self.enemy_history: Dict[int, List[Tuple[float, float, float]]] = {}  # ID -> [(x, y, time), ...]
        self.use_advanced_prediction = use_advanced_prediction
        
        # Advanced movement predictor
        self.movement_predictor: Optional[MovementPredictor] = None
        if use_advanced_prediction and MOVEMENT_PREDICTOR_AVAILABLE:
            try:
                self.movement_predictor = MovementPredictor()
                logger.info("[TRACKER] Advanced movement predictor initialized")
            except Exception as e:
                logger.warning(f"[TRACKER] Failed to initialize movement predictor: {e}")

    def update(self, detections: List[Tuple[List[int], float]]) -> List[Track]:
        """Atualiza tracker e mantém histórico de movimento"""
        tracks = super().update(detections)

        # Atualizar histórico de movimento
        for track in tracks:
            center_x = (track.bbox[0] + track.bbox[2]) / 2
            center_y = (track.bbox[1] + track.bbox[3]) / 2
            current_time = time.time()

            if track.id not in self.enemy_history:
                self.enemy_history[track.id] = []

            self.enemy_history[track.id].append((center_x, center_y, current_time))

            # Manter apenas histórico recente (últimos 5 segundos)
            self.enemy_history[track.id] = [
                (x, y, t) for x, y, t in self.enemy_history[track.id]
                if current_time - t < 5.0
            ]

            # Update movement predictor if available
            if self.movement_predictor is not None and MOVEMENT_PREDICTOR_AVAILABLE:
                try:
                    self.movement_predictor.add_object(track.id, (center_x, center_y))
                except Exception as e:
                    logger.warning(f"[TRACKER] Failed to update movement predictor: {e}")

        return tracks

    def predict_position(self, track_id: int, time_ahead: float = 0.25) -> Optional[Tuple[float, float]]:
        """Prediz posição futura de um inimigo usando movimento avançado se disponível"""
        if track_id not in self.enemy_history:
            return None

        history = self.enemy_history[track_id]
        if len(history) < 2:
            return None

        # Use advanced predictor if available
        if self.movement_predictor is not None and MOVEMENT_PREDICTOR_AVAILABLE:
            try:
                # Use the movement predictor's predict method
                prediction = self.movement_predictor.predict(
                    track_id=track_id,
                    method="ensemble",
                    horizon=time_ahead
                )
                if prediction and prediction.predicted_position:
                    logger.debug(f"[TRACKER] Using advanced movement predictor for track {track_id}")
                    return prediction.predicted_position
            except Exception as e:
                logger.warning(f"[TRACKER] Movement predictor failed: {e}")

        # Fallback to linear prediction
        # Calcular velocidade média
        recent = history[-min(5, len(history)):]
        velocities = []

        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            dt = recent[i][2] - recent[i-1][2]

            if dt > 0:
                velocities.append((dx/dt, dy/dt))

        if not velocities:
            return None

        # Velocidade média
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)

        # Posição atual
        current_x, current_y, _ = history[-1]

        # Predição linear
        pred_x = current_x + avg_vx * time_ahead
        pred_y = current_y + avg_vy * time_ahead

        return (pred_x, pred_y)

    def get_velocity(self, track_id: int) -> Optional[Tuple[float, float]]:
        """Retorna velocidade atual de um inimigo"""
        if track_id not in self.enemy_history:
            return None

        history = self.enemy_history[track_id]
        if len(history) < 2:
            return None

        # Velocidade mais recente
        recent = history[-2:]
        dx = recent[1][0] - recent[0][0]
        dy = recent[1][1] - recent[0][1]
        dt = recent[1][2] - recent[0][2]

        if dt > 0:
            return (dx/dt, dy/dt)

        return None

    def get_leading_shot_position(
        self,
        track_id: int,
        projectile_speed: float,
        frame_delay: int = 0
    ) -> Optional[Tuple[float, float]]:
        """
        Calcula posição para leading shot (disparo antecipado).
        
        Args:
            track_id: ID do track do inimigo
            projectile_speed: Velocidade do projétil em pixels/frame
            frame_delay: Delay adicional em frames (ex: tempo de reação)
            
        Returns:
            Posição (x, y) para mirar ou None se não for possível calcular
        """
        if track_id not in self.enemy_history:
            return None

        history = self.enemy_history[track_id]
        if len(history) < 2:
            return None

        # Use advanced predictor if available to get velocity
        velocity = None
        if self.movement_predictor is not None and MOVEMENT_PREDICTOR_AVAILABLE:
            try:
                velocity = self.movement_predictor.get_velocity(track_id)
                if velocity:
                    logger.debug(f"[TRACKER] Using velocity from movement predictor for track {track_id}")
            except Exception as e:
                logger.warning(f"[TRACKER] Failed to get velocity from movement predictor: {e}")

        # Fallback to calculate velocity from history
        if velocity is None:
            recent = history[-min(5, len(history)):]
            velocities = []

            for i in range(1, len(recent)):
                dx = recent[i][0] - recent[i-1][0]
                dy = recent[i][1] - recent[i-1][1]
                dt = recent[i][2] - recent[i-1][2]

                if dt > 0:
                    velocities.append((dx/dt, dy/dt))

            if not velocities:
                return None

            # Velocidade média
            avg_vx = sum(v[0] for v in velocities) / len(velocities)
            avg_vy = sum(v[1] for v in velocities) / len(velocities)
            velocity = (avg_vx, avg_vy)

        # Posição atual
        current_x, current_y, _ = history[-1]
        vx, vy = velocity

        # Calcular velocidade
        speed = (vx**2 + vy**2)**0.5
        
        if speed > 0 and projectile_speed > 0:
            # Tempo para interceptação (simplificado)
            # Adiciona frame_delay ao tempo de voo
            intercept_time = 5.0 + frame_delay  # Default: 5 frames ahead + delay
            lead_x = current_x + vx * intercept_time
            lead_y = current_y + vy * intercept_time
            return (lead_x, lead_y)

        # Quando não há movimento suficiente, ao menos retornar a posição atual
        # evita que integrações falhem por ausência de histórico mais rico.
        return (current_x, current_y)
