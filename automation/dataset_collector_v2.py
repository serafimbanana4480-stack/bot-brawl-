"""
dataset_collector_v2.py

Pipeline automatizado avançado para coleta de dataset de Brawl Stars durante gameplay real.
Integra captura de screenshots com detecção de estado do jogo e rotulagem automática via YOLO.

Funcionalidades:
- Captura automatizada de screenshots durante gameplay
- Detecção de estado do jogo (lobby, partida, loading, etc.)
- Rotulagem automática de objetos via YOLO (player, enemies, bushes, etc.)
- Organização automática por estado e timestamp
- Metadados enriquecidos com detecções YOLO
- Sistema de priorização de frames interessantes (combate, kills, etc.)
- Suporte a captura contínua ou por eventos

Usage:
    python -m brawl_bot.automation.dataset_collector_v2 --duration 300 --output ./dataset/raw
    python -m brawl_bot.automation.dataset_collector_v2 --mode event --output ./dataset/raw
"""

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import sys
import numpy as np

# Adicionar diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from emulator_detector import get_adb_path, get_emulator_detector
from screenshot_recorder import _adb_screencap

logger = logging.getLogger(__name__)


@dataclass
class YOLODetection:
    """Representa uma detecção YOLO"""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] normalized 0-1


@dataclass
class CaptureMetadata:
    """Metadados enriquecidos para cada screenshot capturado"""
    timestamp: str
    game_state: str
    match_id: Optional[str] = None
    frame_number: int = 0
    resolution: str = "1920x1080"
    emulator_type: str = "bluestacks"
    adb_id: str = ""
    
    # Detections
    detections: List[Dict] = None
    player_detected: bool = False
    enemy_count: int = 0
    bush_count: int = 0
    
    # Frame priority
    priority_score: float = 0.0  # 0-1, higher = more interesting
    priority_reason: str = ""
    
    def __post_init__(self):
        if self.detections is None:
            self.detections = []
    
    def to_dict(self) -> Dict:
        return asdict(self)


class GameStateDetector:
    """Detecta o estado atual do jogo baseado em análise de imagem"""
    
    STATES = {
        "lobby": "Lobby principal",
        "matchmaking": "Procurando partida",
        "loading": "Carregando partida",
        "game": "Em partida",
        "result": "Tela de resultado",
        "unknown": "Estado desconhecido"
    }
    
    def __init__(self):
        self.state_history = []
        self.current_state = "unknown"
    
    def detect_state(self, image: np.ndarray) -> str:
        """Detecta o estado do jogo analisando a imagem"""
        try:
            import cv2
            
            # Análise baseada em características visuais
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Verificar se tem muitos pixels brancos (lobby/loading)
            white_pixels = np.sum(gray > 200) / gray.size
            
            # Verificar distribuição de cores (gameplay tem mais variedade)
            color_std = np.std(gray)
            
            if white_pixels > 0.8:
                return "lobby"
            elif white_pixels > 0.6:
                return "matchmaking"
            elif color_std > 50:  # Alta variabilidade = gameplay
                return "game"
            else:
                return "loading"
                
        except Exception as e:
            logger.error(f"Erro ao detectar estado: {e}")
            return "unknown"
    
    def update_state(self, new_state: str):
        """Atualiza o estado atual com histórico"""
        self.state_history.append((datetime.now().isoformat(), new_state))
        self.current_state = new_state
        
        # Manter apenas os últimos 50 estados
        if len(self.state_history) > 50:
            self.state_history = self.state_history[-50:]


class YOLOAutoLabeler:
    """Rotulagem automática usando YOLO"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.class_mapping = {}
        
        if model_path and model_path.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(model_path))
                
                # Detect class names from model
                if hasattr(self.model, 'names'):
                    self.class_mapping = {v: k for k, v in self.model.names.items()}
                
                logger.info(f"YOLO model loaded: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}")
    
    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.3) -> Tuple[List[YOLODetection], Dict]:
        """Detecta objetos na imagem usando YOLO"""
        detections = []
        summary = {
            "player_detected": False,
            "enemy_count": 0,
            "bush_count": 0,
            "object_count": 0
        }
        
        if self.model is None:
            return detections, summary
        
        try:
            results = self.model(image, conf=conf_threshold, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = self.model.names.get(class_id, f"class_{class_id}")
                        confidence = float(box.conf[0])
                        
                        # Get bbox (normalized 0-1)
                        bbox = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = bbox
                        h, w = image.shape[:2]
                        norm_bbox = [x1/w, y1/h, x2/w, y2/h]
                        
                        detection = YOLODetection(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=norm_bbox
                        )
                        detections.append(detection)
                        
                        # Update summary
                        class_lower = class_name.lower()
                        if "player" in class_lower:
                            summary["player_detected"] = True
                        elif "enemy" in class_lower or "brawler" in class_lower:
                            summary["enemy_count"] += 1
                        elif "bush" in class_lower:
                            summary["bush_count"] += 1
                        summary["object_count"] += 1
        
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
        
        return detections, summary
    
    def calculate_priority_score(self, detections: List[YOLODetection], summary: Dict) -> Tuple[float, str]:
        """Calcula prioridade do frame baseado nas detecções"""
        score = 0.0
        reasons = []
        
        # Combat situations (enemies present)
        if summary["enemy_count"] > 0:
            score += 0.3 + (min(summary["enemy_count"], 3) * 0.1)
            reasons.append(f"{summary['enemy_count']} enemies")
        
        # Player in frame
        if summary["player_detected"]:
            score += 0.2
            reasons.append("player detected")
        
        # Bushes (strategic positions)
        if summary["bush_count"] > 0:
            score += 0.1 + (min(summary["bush_count"], 5) * 0.02)
            reasons.append(f"{summary['bush_count']} bushes")
        
        # Total objects (activity level)
        if summary["object_count"] > 5:
            score += 0.1
            reasons.append("high activity")
        
        # Cap score at 1.0
        score = min(score, 1.0)
        
        reason_str = ", ".join(reasons) if reasons else "baseline"
        return score, reason_str


class DatasetCollectorV2:
    """Coletor automatizado avançado de dataset de Brawl Stars com YOLO labeling"""
    
    def __init__(
        self,
        adb_id: str,
        output_dir: Path,
        adb_path: Optional[str] = None,
        yolo_model_path: Optional[Path] = None,
        enable_yolo_labeling: bool = True
    ):
        self.adb_id = adb_id
        self.output_dir = Path(output_dir)
        self.adb_path = adb_path or get_adb_path()
        self.state_detector = GameStateDetector()
        self.frame_count = 0
        self.match_id = str(int(time.time()))
        self.is_running = False
        self.enable_yolo_labeling = enable_yolo_labeling
        
        # YOLO labeler
        self.yolo_labeler = None
        if enable_yolo_labeling:
            self.yolo_labeler = YOLOAutoLabeler(yolo_model_path)
        
        # Criar diretórios organizados
        self.raw_dir = self.output_dir / "raw"
        self.by_state_dir = self.output_dir / "by_state"
        self.by_priority_dir = self.output_dir / "by_priority"
        self.metadata_dir = self.output_dir / "metadata"
        self.labels_dir = self.output_dir / "labels"  # YOLO format labels
        
        for dir_path in [self.raw_dir, self.by_state_dir, self.by_priority_dir, 
                         self.metadata_dir, self.labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_frames": 0,
            "high_priority_frames": 0,
            "by_state": {},
            "detections": {}
        }
    
    async def capture_frame(self) -> Optional[str]:
        """Captura um frame com rotulagem YOLO automática"""
        timestamp = datetime.now()
        stem = timestamp.strftime("%Y%m%d_%H%M%S_") + f"{timestamp.microsecond // 1000:03d}"
        
        img_path = self.raw_dir / f"{stem}.png"
        meta_path = self.metadata_dir / f"{stem}.json"
        label_path = self.labels_dir / f"{stem}.txt"
        
        # Capturar screenshot
        success = _adb_screencap(self.adb_path, self.adb_id, img_path)
        
        if not success:
            logger.error(f"Falha ao capturar frame {self.frame_count}")
            return None
        
        # Carregar imagem para detecção
        import cv2
        image = cv2.imread(str(img_path))
        if image is None:
            logger.error(f"Falha ao carregar imagem {img_path}")
            return None
        
        # Detectar estado do jogo
        game_state = self.state_detector.detect_state(image)
        self.state_detector.update_state(game_state)
        
        # YOLO detections
        detections = []
        detection_summary = {
            "player_detected": False,
            "enemy_count": 0,
            "bush_count": 0,
            "object_count": 0
        }
        priority_score = 0.0
        priority_reason = "baseline"
        
        if self.yolo_labeler and self.enable_yolo_labeling:
            detections, detection_summary = self.yolo_labeler.detect_objects(image)
            priority_score, priority_reason = self.yolo_labeler.calculate_priority_score(
                detections, detection_summary
            )
            
            # Save YOLO format labels
            self._save_yolo_labels(label_path, detections, image.shape[:2])
        
        # Criar metadados enriquecidos
        metadata = CaptureMetadata(
            timestamp=timestamp.isoformat(),
            game_state=game_state,
            match_id=self.match_id,
            frame_number=self.frame_count,
            adb_id=self.adb_id,
            detections=[d.to_dict() for d in detections],
            player_detected=detection_summary["player_detected"],
            enemy_count=detection_summary["enemy_count"],
            bush_count=detection_summary["bush_count"],
            priority_score=priority_score,
            priority_reason=priority_reason
        )
        
        # Salvar metadados
        with open(meta_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Organizar por estado
        state_dir = self.by_state_dir / game_state
        state_dir.mkdir(exist_ok=True)
        try:
            state_dir.joinpath(img_path.name).symlink_to(img_path.resolve())
        except (OSError, NotImplementedError):
            import shutil
            shutil.copy2(img_path, state_dir.joinpath(img_path.name))
        
        # Organizar por prioridade
        if priority_score > 0.5:
            priority_dir = self.by_priority_dir / "high"
            priority_dir.mkdir(exist_ok=True)
            self.stats["high_priority_frames"] += 1
        elif priority_score > 0.2:
            priority_dir = self.by_priority_dir / "medium"
            priority_dir.mkdir(exist_ok=True)
        else:
            priority_dir = self.by_priority_dir / "low"
            priority_dir.mkdir(exist_ok=True)
        
        try:
            priority_dir.joinpath(img_path.name).symlink_to(img_path.resolve())
        except (OSError, NotImplementedError):
            import shutil
            shutil.copy2(img_path, priority_dir.joinpath(img_path.name))
        
        # Update statistics
        self.stats["total_frames"] += 1
        self.stats["by_state"][game_state] = self.stats["by_state"].get(game_state, 0) + 1
        
        self.frame_count += 1
        
        return str(img_path)
    
    def _save_yolo_labels(self, label_path: Path, detections: List[YOLODetection], image_shape: Tuple[int, int]):
        """Salva labels no formato YOLO (class_id x_center y_center width height)"""
        h, w = image_shape
        
        # Simple class mapping (can be customized)
        class_to_id = {
            "player": 0,
            "enemy": 1,
            "brawler": 1,
            "bush": 2,
            "cubebox": 3,
            "obstacle": 4
        }
        
        labels = []
        for det in detections:
            class_id = class_to_id.get(det.class_name.lower(), 5)  # 5 = unknown
            
            # Convert bbox to YOLO format (x_center, y_center, width, height)
            x1, y1, x2, y2 = det.bbox
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
    
    async def collect_continuous(self, duration_seconds: int, interval_seconds: float = 2.0):
        """Coleta frames continuamente por um período especificado"""
        self.is_running = True
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        logger.info(f"Iniciando coleta contínua v2: {duration_seconds}s, intervalo {interval_seconds}s")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"ADB ID: {self.adb_id}")
        logger.info(f"YOLO labeling: {'enabled' if self.enable_yolo_labeling else 'disabled'}")
        
        try:
            while time.time() < end_time and self.is_running:
                await self.capture_frame()
                
                if self.frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    remaining = end_time - time.time()
                    logger.info(
                        f"Capturados {self.frame_count} frames | "
                        f"{elapsed:.1f}s elapsed | {remaining:.1f}s remaining | "
                        f"High priority: {self.stats['high_priority_frames']}"
                    )
                
                await asyncio.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Coleta interrompida pelo usuário")
        except Exception as e:
            logger.error(f"Erro durante coleta: {e}")
        finally:
            self.is_running = False
            self._save_statistics()
            logger.info(f"Coleta finalizada. Total de frames: {self.frame_count}")
            logger.info(f"Estatísticas: {self.stats}")
    
    async def collect_event_based(self, max_frames: int = 1000):
        """Coleta frames baseado em eventos (mudanças de estado e prioridade)"""
        self.is_running = True
        last_state = "unknown"
        
        logger.info(f"Iniciando coleta baseada em eventos v2: max {max_frames} frames")
        
        try:
            while self.frame_count < max_frames and self.is_running:
                frame_path = await self.capture_frame()
                
                # Detectar mudança de estado
                current_state = self.state_detector.current_state
                if current_state != last_state:
                    logger.info(f"Mudança de estado: {last_state} -> {current_state}")
                    last_state = current_state
                
                await asyncio.sleep(1.0)
                
        except KeyboardInterrupt:
            logger.info("Coleta interrompida pelo usuário")
        except Exception as e:
            logger.error(f"Erro durante coleta: {e}")
        finally:
            self.is_running = False
            self._save_statistics()
            logger.info(f"Coleta finalizada. Total de frames: {self.frame_count}")
    
    def stop(self):
        """Para a coleta"""
        self.is_running = False
    
    def _save_statistics(self):
        """Salva estatísticas da coleta"""
        stats_path = self.output_dir / "collection_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Estatísticas salvas em {stats_path}")


async def main():
    """Função principal para execução via linha de comando"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Collector v2 com YOLO labeling")
    parser.add_argument("--adb-id", required=True, help="ADB device ID")
    parser.add_argument("--duration", type=int, default=300, help="Duração em segundos")
    parser.add_argument("--interval", type=float, default=2.0, help="Intervalo entre frames")
    parser.add_argument("--output", default="./dataset/raw", help="Diretório de output")
    parser.add_argument("--mode", choices=["continuous", "event"], default="continuous", help="Modo de coleta")
    parser.add_argument("--yolo-model", help="Caminho para modelo YOLO")
    parser.add_argument("--no-yolo", action="store_true", help="Desabilitar YOLO labeling")
    
    args = parser.parse_args()
    
    # Detectar emulador
    detector = detect_emulator()
    if not detector:
        logger.error("Nenhum emulador detectado")
        return
    
    # Criar coletor
    yolo_model_path = Path(args.yolo_model) if args.yolo_model else None
    collector = DatasetCollectorV2(
        adb_id=args.adb_id,
        output_dir=Path(args.output),
        yolo_model_path=yolo_model_path,
        enable_yolo_labeling=not args.no_yolo
    )
    
    # Executar coleta
    if args.mode == "continuous":
        await collector.collect_continuous(args.duration, args.interval)
    else:
        await collector.collect_event_based(max_frames=1000)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
