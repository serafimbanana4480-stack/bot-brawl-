"""
detect.py

Transforma output YOLOv8 num formato mais fácil de usar.
Baseado no PylaAI de ivanyordanovgt.
"""

import logging
import concurrent.futures
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class Detect:
    """
    Classe para detetar objetos usando YOLOv8.
    Requer classes porque modelos exportados perdem o atributo 'names'.
    """

    def __init__(self, model, ignore_classes=None, classes=None, conf=0.5, model_config=None):
        self.model = model
        self.classes = classes or {}
        self.ignore_classes = ignore_classes if ignore_classes else []
        self.conf = conf
        # FIX #1.2: Detect GPU and use CUDA if available
        _device = 'cpu'
        try:
            import torch
            if torch.cuda.is_available():
                _device = 'cuda'
                logger.info(f"[DETECT] GPU detected: {torch.cuda.get_device_name(0)}")
        except ImportError:
            logger.debug("[DETECT] PyTorch not available - using CPU")
        self.model_config = model_config or {
            'conf': 0.65,
            'device': _device,
            'verbose': False
        }
        # FIX #16: Thread pool for async inference - lazy initialization
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._pending_future: Optional[concurrent.futures.Future] = None
        self._last_result: Optional[Dict[str, List[List[int]]]] = None
        self._result_lock = threading.Lock()
        self._max_workers = 1  # Single worker to avoid memory bloat

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Lazy initialization of thread pool."""
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
            logger.debug(f"[DETECT] ThreadPoolExecutor initialized with {self._max_workers} worker(s)")
        return self._executor

    def _do_inference(self, img) -> Dict[str, List[List[int]]]:
        """Internal inference method - runs in thread pool."""
        detected = {}
        results = self.model(img, conf=self.conf)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.ignore_classes:
                    continue

                class_name = self.classes.get(cls_id, f"class_{cls_id}")

                bbox = box.xyxy[0].cpu().numpy()
                bbox = bbox.astype(int).tolist()

                if class_name not in detected:
                    detected[class_name] = []
                detected[class_name].append(bbox)

        return detected

    def detect_objects(self, img) -> Dict[str, List[List[int]]]:
        """
        Processa output do modelo YOLOv8 (SÍNCRONO).
        Retorna: {class_name: [[x1, y1, x2, y2], ...]}

        Para inference assíncrona, use detect_objects_async() em vez deste método.
        """
        # FIX #21: Verify model is loaded before inference
        if self.model is None:
            logger.error("[DETECT] Model not loaded - cannot perform inference")
            return {}
        if not hasattr(self.model, '__call__'):
            logger.error("[DETECT] Model is not callable - cannot perform inference")
            return {}

        return self._do_inference(img)

    def detect_objects_async(self, img) -> None:
        """
        FIX #16: Submete inference para thread pool (NÃO BLOQUEIA).

        Usa o padrão "fire and forget" - a inference corre em background
        e o resultado fica disponível via get_async_result().

        Se houver uma inference pendente, cancela-a e submete a nova.
        Isto evita acumulação de tarefas quando o ciclo principal é mais rápido.
        """
        if self.model is None:
            logger.warning("[DETECT] Model not loaded - skipping async inference")
            return

        # Cancel any pending inference to avoid queue buildup
        if self._pending_future is not None and not self._pending_future.done():
            self._pending_future.cancel()
            logger.debug("[DETECT] Cancelled pending inference")

        # Submit new inference task
        self._pending_future = self._get_executor().submit(self._do_inference, img)

    def get_async_result(self, timeout: float = 0.0) -> Dict[str, List[List[int]]]:
        """
        FIX #16: Obtém resultado da inference assíncrona.

        Args:
            timeout: Tempo máximo a esperar pelo resultado (0 = não espera)

        Returns:
            Dict com detecções, ou dict vazio se timeout ou erro.
        """
        if self._pending_future is None:
            return self._last_result if self._last_result else {}

        try:
            if timeout > 0:
                result = self._pending_future.result(timeout=timeout)
            else:
                if self._pending_future.done():
                    result = self._pending_future.result(timeout=0)
                else:
                    # Not done and no wait - return cached result
                    return self._last_result if self._last_result else {}

            with self._result_lock:
                self._last_result = result
                self._pending_future = None
            return result

        except concurrent.futures.TimeoutError:
            logger.debug("[DETECT] Async inference timeout")
            return self._last_result if self._last_result else {}
        except concurrent.futures.CancelledError:
            logger.debug("[DETECT] Async inference cancelled")
            return self._last_result if self._last_result else {}
        except Exception as e:
            logger.error(f"[DETECT] Async inference error: {e}")
            return self._last_result if self._last_result else {}

    def shutdown(self) -> None:
        """Cleanup thread pool on shutdown."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
            logger.debug("[DETECT] ThreadPoolExecutor shutdown")

    def get_center(self, bbox: List[int]) -> tuple:
        """Calcula centro de uma bounding box [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def get_distance(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calcula distância entre centros de duas bounding boxes"""
        c1 = self.get_center(bbox1)
        c2 = self.get_center(bbox2)
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
