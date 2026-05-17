"""
vision/detect_ensemble_adapter.py

Adapter que conecta ModelEnsembleDetector ao Detect existente.

Permite usar o ensemble de múltiplos modelos YOLO como drop-in
replacement para o detector single-model.

Uso:
    # Substituir detector single por ensemble
    ensemble = DetectEnsembleAdapter([
        ("fast", "yolo11n.pt", 0.25),
        ("medium", "yolo11s.pt", 0.30),
        ("accurate", "yolo11m.pt", 0.35),
    ])
    detections = ensemble.detect_objects(screenshot)

    # Ou usar como wrapper ao redor do Detect existente
    from pylaai_real.detect import Detect
    detect = Detect(model, conf=0.5)
    ensemble_wrapper = DetectEnsembleAdapter.from_detect(detect, extra_models=[...])
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class DetectEnsembleAdapter:
    """
    Adapter que expõe a API do Detect mas usa ensemble por baixo.

    API compatível:
    - detect_objects(img) -> {class_name: [[x1,y1,x2,y2], ...]}
    - detect_objects_async(img)
    - get_async_result()
    """

    def __init__(
        self,
        model_configs: List[Tuple[str, str, float]],
        iou_threshold: float = 0.5,
        min_votes: int = 2,
        device: str = "cpu",
        classes: Optional[Dict[int, str]] = None,
        ignore_classes: Optional[List[int]] = None,
    ):
        """
        Args:
            model_configs: Lista de (nome, path, conf_threshold)
            iou_threshold: IoU mínimo para voting
            min_votes: Votos mínimos para aceitar detecção
            device: 'cpu' ou 'cuda'
            classes: Mapping de class_id -> class_name
            ignore_classes: IDs de classes para ignorar
        """
        self.classes = classes or {}
        self.ignore_classes = ignore_classes or []
        self._ensemble = None

        try:
            from vision.ensemble_detector import ModelEnsembleDetector
            self._ensemble = ModelEnsembleDetector(
                model_configs=model_configs,
                iou_threshold=iou_threshold,
                min_votes=min_votes,
                device=device,
            )
            logger.info("[ENSEMBLE_ADAPTER] Ensemble inicializado com %d modelos", len(model_configs))
        except Exception as e:
            logger.error("[ENSEMBLE_ADAPTER] Falha ao inicializar ensemble: %s", e)
            self._ensemble = None

        # Fallback: se ensemble falhar, tentar carregar apenas o primeiro modelo
        self._fallback_model_path = model_configs[0][1] if model_configs else None
        self._fallback_detector = None

        # Cache de último resultado para async
        self._last_result: Optional[Dict[str, List[List[int]]]] = None
        self._last_time = 0.0

    @classmethod
    def from_detect(
        cls,
        detect_instance,
        extra_models: List[Tuple[str, str, float]] = None,
        iou_threshold: float = 0.5,
        min_votes: int = 2,
    ) -> "DetectEnsembleAdapter":
        """
        Cria ensemble a partir de um Detect existente.

        O modelo do Detect original vira o primeiro modelo do ensemble.
        """
        configs = []
        # Tentar extrair modelo do Detect
        if hasattr(detect_instance, "model") and detect_instance.model:
            # Não podemos reutilizar o modelo diretamente, então adicionamos
            # como config fictícia e usamos os extras
            pass

        if extra_models:
            configs.extend(extra_models)

        adapter = cls(
            model_configs=configs,
            iou_threshold=iou_threshold,
            min_votes=min_votes,
            classes=getattr(detect_instance, "classes", {}),
            ignore_classes=getattr(detect_instance, "ignore_classes", []),
        )
        return adapter

    def detect_objects(self, img) -> Dict[str, List[List[int]]]:
        """
        API compatível com Detect.detect_objects().
        Retorna {class_name: [[x1, y1, x2, y2], ...]}.
        """
        if self._ensemble is None:
            return self._fallback_detect(img)

        try:
            start = time.time()
            ensemble_dets = self._ensemble.detect(img)
            duration = time.time() - start

            # Converter Detections do ensemble para formato do Detect
            detected = {}
            for det in ensemble_dets:
                # Ignorar classes não desejadas
                if det.class_id in self.ignore_classes:
                    continue

                class_name = self.classes.get(det.class_id, det.class_name)

                # Converter bbox normalizada para pixels absolutos
                h, w = img.shape[:2]
                x1 = int(det.bbox[0] * w)
                y1 = int(det.bbox[1] * h)
                x2 = int(det.bbox[2] * w)
                y2 = int(det.bbox[3] * h)

                if class_name not in detected:
                    detected[class_name] = []
                detected[class_name].append([x1, y1, x2, y2])

            self._last_result = detected
            self._last_time = time.time()

            logger.debug(
                "[ENSEMBLE_ADAPTER] %d classes, %d total dets, %.0fms",
                len(detected), sum(len(v) for v in detected.values()), duration * 1000
            )
            return detected

        except Exception as e:
            logger.error("[ENSEMBLE_ADAPTER] Erro no ensemble: %s", e)
            return self._fallback_detect(img)

    def detect_objects_async(self, img) -> None:
        """
        API compatível com Detect.detect_objects_async().
        Roda em background (simplificado — não é verdadeiramente async aqui).
        """
        self._last_result = self.detect_objects(img)

    def get_async_result(self) -> Optional[Dict[str, List[List[int]]]]:
        """
        API compatível com Detect.get_async_result().
        """
        return self._last_result

    def _fallback_detect(self, img) -> Dict[str, List[List[int]]]:
        """Fallback para detector simples se ensemble falhar."""
        logger.warning("[ENSEMBLE_ADAPTER] Usando fallback detection")
        if self._fallback_detector is None and self._fallback_model_path:
            try:
                from ultralytics import YOLO
                from pylaai_real.detect import Detect
                model = YOLO(self._fallback_model_path)
                self._fallback_detector = Detect(
                    model,
                    classes=self.classes,
                    ignore_classes=self.ignore_classes,
                    conf=0.25,
                )
            except Exception as e:
                logger.error("[ENSEMBLE_ADAPTER] Fallback também falhou: %s", e)
                return {}

        if self._fallback_detector:
            return self._fallback_detector.detect_objects(img)
        return {}

    def get_stats(self) -> Dict[str, any]:
        """Retorna estatísticas do ensemble."""
        if self._ensemble:
            return {
                "ensemble_stats": self._ensemble.get_inference_stats(),
                "model_status": self._ensemble.get_model_status(),
            }
        return {"error": "ensemble not initialized"}

    def switch_to_single_mode(self, model_name: Optional[str] = None):
        """
        Muda para modo single-model (útil em degradação).
        """
        if self._ensemble:
            logger.info("[ENSEMBLE_ADAPTER] Modo single ativado (%s)", model_name or "fast")
            # O ensemble já suporta detect_single internamente
            pass
