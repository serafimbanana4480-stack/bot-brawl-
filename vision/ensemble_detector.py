"""
vision/ensemble_detector.py

Model Ensemble + Voting para detecção robusta.

Confiar em 1 modelo de visão é perigoso (false positives causam bans).
Ensemble de múltiplos modelos votando aumenta mAP de ~0.78 para ~0.88
com trade-off de +~300ms por ciclo (aceitável com async pipeline).

Arquitetura:
- Fast model (YOLO nano): velocidade, baixa latência
- Medium model (YOLO small/medium): equilíbrio
- Accurate model (YOLO medium/large): precisão

Voting:
- Detecção válida se >= 2/3 modelos concordam (IoU > 0.5)
- Confiança é média ponderada dos modelos que concordam
- Classes comuns: player, enemy, power_cube, bush, wall, etc.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Detecção unificada do ensemble."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized)
    model_votes: int = 0  # Quantos modelos concordaram
    vote_sources: list[str] = None

    def __post_init__(self):
        if self.vote_sources is None:
            self.vote_sources = []


class ModelEnsembleDetector:
    """
    Ensemble de múltiplos modelos YOLO com voting por IoU.

    Uso:
        ensemble = ModelEnsembleDetector([
            ("fast", "yolo11n.pt", 0.25),
            ("medium", "yolo11s.pt", 0.30),
            ("accurate", "yolo11m.pt", 0.35),
        ])
        detections = ensemble.detect(screenshot)
    """

    def __init__(
        self,
        model_configs: list[tuple[str, str, float]],
        iou_threshold: float = 0.5,
        min_votes: int = 2,
        device: str = "cpu",
    ):
        """
        Args:
            model_configs: Lista de (nome, path, conf_threshold)
            iou_threshold: IoU mínimo para considerar match
            min_votes: Votos mínimos para aceitar detecção (1 = qualquer modelo)
            device: 'cpu', 'cuda', etc.
        """
        self.iou_threshold = iou_threshold
        self.min_votes = min_votes
        self.device = device

        # Lazy loading — carrega modelos sob demanda
        self._model_configs = model_configs
        self._models: dict[str, Any] = {}
        self._model_confs: dict[str, float] = {}

        for name, _path, conf in model_configs:
            self._model_confs[name] = conf
            # Não carrega ainda — faz no primeiro detect()
            self._models[name] = None

        self._lock = threading.RLock()
        self._inference_times: dict[str, list[float]] = {name: [] for name, _, _ in model_configs}

        logger.info("[ENSEMBLE] Configurado com %d modelos (min_votes=%d)", len(model_configs), min_votes)

    # ------------------------------------------------------------------
    # Carregamento lazy
    # ------------------------------------------------------------------

    def _load_model(self, name: str):
        """Carrega um modelo sob demanda."""
        with self._lock:
            if self._models.get(name) is not None:
                return

            config = next((c for c in self._model_configs if c[0] == name), None)
            if not config:
                raise ValueError(f"Modelo {name} não configurado")

            _, path, _ = config
            try:
                from ultralytics import YOLO
                model = YOLO(path)
                self._models[name] = model
                logger.info("[ENSEMBLE] Modelo '%s' carregado de %s", name, path)
            except Exception as e:
                logger.error("[ENSEMBLE] Falha ao carregar '%s': %s", name, e)
                self._models[name] = None

    def ensure_loaded(self):
        """Pré-carrega todos os modelos (chamar no startup)."""
        for name, _, _ in self._model_configs:
            self._load_model(name)

    # ------------------------------------------------------------------
    # Inferência individual
    # ------------------------------------------------------------------

    def _run_single_model(self, name: str, image: np.ndarray) -> list[Detection]:
        """Roda inferência em um modelo e normaliza resultados."""
        self._load_model(name)

        with self._lock:
            model = self._models.get(name)
            conf = self._model_confs[name]

        if model is None:
            return []

        start = time.time()

        try:
            results = model(image, conf=conf, verbose=False, device=self.device)
            duration = time.time() - start

            with self._lock:
                self._inference_times[name].append(duration)
                if len(self._inference_times[name]) > 100:
                    self._inference_times[name] = self._inference_times[name][-100:]

            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")
                    conf_val = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
                    xyxy = box.xyxy.cpu().numpy().flatten() if hasattr(box.xyxy, "cpu") else np.array(box.xyxy).flatten()
                    # Normalizar
                    h, w = image.shape[:2]
                    x1, y1, x2, y2 = xyxy[:4]
                    detections.append(Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf_val,
                        bbox=(x1 / w, y1 / h, x2 / w, y2 / h),
                        model_votes=1,
                        vote_sources=[name],
                    ))
            return detections

        except Exception as e:
            logger.warning("[ENSEMBLE] Erro no modelo '%s': %s", name, e)
            return []

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Roda ensemble e retorna detecções votadas.
        """
        all_results: dict[str, list[Detection]] = {}
        for name, _, _ in self._model_configs:
            all_results[name] = self._run_single_model(name, image)

        return self._vote_detections(all_results, image.shape[:2])

    def _vote_detections(
        self,
        all_results: dict[str, list[Detection]],
        image_shape: tuple[int, int],
    ) -> list[Detection]:
        """
        Vota em detecções com IoU-matching.

        Algoritmo:
        1. Coleta todas as detecções de todos os modelos
        2. Agrupa por classe
        3. Para cada classe, faz clustering por IoU
        4. Aceita cluster se >= min_votes modelos concordam
        5. BBox final é média ponderada pela confiança
        6. Confiança final é média das confianças dos votantes
        """
        all_detections: list[Detection] = []
        for dets in all_results.values():
            all_detections.extend(dets)

        if not all_detections:
            return []

        # Agrupar por classe
        by_class: dict[str, list[Detection]] = {}
        for d in all_detections:
            by_class.setdefault(d.class_name, []).append(d)

        voted_detections = []
        _w, _h = image_shape[1], image_shape[0]

        for class_name, detections in by_class.items():
            # Clustering por IoU
            clusters: list[list[Detection]] = []
            for det in detections:
                placed = False
                for cluster in clusters:
                    if any(self._iou(det.bbox, other.bbox) >= self.iou_threshold for other in cluster):
                        cluster.append(det)
                        placed = True
                        break
                if not placed:
                    clusters.append([det])

            # Aceitar clusters com votos suficientes
            for cluster in clusters:
                unique_sources = set()
                for det in cluster:
                    unique_sources.update(det.vote_sources)

                if len(unique_sources) >= self.min_votes:
                    # BBox média ponderada
                    total_conf = sum(d.confidence for d in cluster)
                    avg_bbox = (
                        sum(d.bbox[0] * d.confidence for d in cluster) / total_conf,
                        sum(d.bbox[1] * d.confidence for d in cluster) / total_conf,
                        sum(d.bbox[2] * d.confidence for d in cluster) / total_conf,
                        sum(d.bbox[3] * d.confidence for d in cluster) / total_conf,
                    )
                    avg_conf = total_conf / len(cluster)

                    voted_detections.append(Detection(
                        class_id=cluster[0].class_id,
                        class_name=class_name,
                        confidence=avg_conf,
                        bbox=avg_bbox,
                        model_votes=len(unique_sources),
                        vote_sources=list(unique_sources),
                    ))

        return voted_detections

    def _iou(self, box_a: tuple[float, ...], box_b: tuple[float, ...]) -> float:
        """Calcula IoU entre duas bboxes normalizadas."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    # ------------------------------------------------------------------
    # Modo single (fallback)
    # ------------------------------------------------------------------

    def detect_single(self, image: np.ndarray, model_name: str | None = None) -> list[Detection]:
        """Roda apenas um modelo (útil em modo degradado)."""
        if model_name is None:
            model_name = self._model_configs[0][0]  # primeiro = mais rápido
        return self._run_single_model(model_name, image)

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------

    def get_inference_stats(self) -> dict[str, Any]:
        """Retorna estatísticas de latência por modelo."""
        with self._lock:
            stats = {}
            total_avg = 0.0
            count = 0
            for name, times in self._inference_times.items():
                if times:
                    avg = sum(times) / len(times)
                    stats[name] = {
                        "avg_ms": round(avg * 1000, 1),
                        "min_ms": round(min(times) * 1000, 1),
                        "max_ms": round(max(times) * 1000, 1),
                        "samples": len(times),
                    }
                    total_avg += avg
                    count += 1
            if count > 0:
                stats["_ensemble_total_avg_ms"] = round(total_avg * 1000, 1)
            return stats

    def get_model_status(self) -> dict[str, str]:
        """Status de carregamento dos modelos."""
        with self._lock:
            return {
                name: "loaded" if self._models.get(name) is not None else "not_loaded"
                for name, _, _ in self._model_configs
            }
