"""
dataset_collector.py

Pipeline automatizado para coleta de dataset de Brawl Stars durante gameplay real.
Integra captura de screenshots com detecção de estado do jogo para rotulagem automática.

Funcionalidades:
- Captura automatizada de screenshots durante gameplay
- Detecção de estado do jogo (lobby, partida, loading, etc.)
- Organização automática por estado e timestamp
- Metadados para facilitar rotulagem
- Suporte a captura contínua ou por eventos

Usage:
    python -m brawl_bot.automation.dataset_collector --duration 300 --output ./dataset/raw
    python -m brawl_bot.automation.dataset_collector --mode event --output ./dataset/raw
"""

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
import sys

# Adicionar diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from emulator_detector import get_adb_path, detect_emulator
from screenshot_recorder import _adb_screencap

logger = logging.getLogger(__name__)


@dataclass
class CaptureMetadata:
    """Metadados para cada screenshot capturado"""
    timestamp: str
    game_state: str
    match_id: Optional[str] = None
    frame_number: int
    resolution: str = "1920x1080"
    emulator_type: str = "bluestacks"
    adb_id: str = ""
    
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
    
    def detect_state(self, image_path: Path) -> str:
        """Detecta o estado do jogo analisando a imagem"""
        try:
            import cv2
            image = cv2.imread(str(image_path))
            if image is None:
                return "unknown"
            
            # Análise simples baseada em características visuais
            # Em uma implementação completa, usaria YOLO para detectar elementos específicos
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Verificar se tem muitos pixels brancos (lobby/loading)
            white_pixels = np.sum(gray > 200) / gray.size
            
            # Verificar presença de UI elements específicos
            # Esta é uma implementação simplificada
            
            if white_pixels > 0.8:
                return "lobby"  # Provavelmente lobby ou loading
            elif white_pixels > 0.6:
                return "matchmaking"
            else:
                return "game"  # Tem mais conteúdo, provavelmente em jogo
                
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


class DatasetCollector:
    """Coletor automatizado de dataset de Brawl Stars"""
    
    def __init__(self, adb_id: str, output_dir: Path, adb_path: Optional[str] = None):
        self.adb_id = adb_id
        self.output_dir = Path(output_dir)
        self.adb_path = adb_path or get_adb_path()
        self.state_detector = GameStateDetector()
        self.frame_count = 0
        self.match_id = str(int(time.time()))
        self.is_running = False
        
        # Criar diretórios organizados
        self.raw_dir = self.output_dir / "raw"
        self.by_state_dir = self.output_dir / "by_state"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.raw_dir, self.by_state_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    async def capture_frame(self) -> Optional[str]:
        """Captura um frame e retorna o caminho"""
        timestamp = datetime.now()
        stem = timestamp.strftime("%Y%m%d_%H%M%S_") + f"{timestamp.microsecond // 1000:03d}"
        
        img_path = self.raw_dir / f"{stem}.png"
        meta_path = self.metadata_dir / f"{stem}.json"
        
        # Capturar screenshot
        success = _adb_screencap(self.adb_path, self.adb_id, img_path)
        
        if not success:
            logger.error(f"Falha ao capturar frame {self.frame_count}")
            return None
        
        # Detectar estado do jogo
        game_state = self.state_detector.detect_state(img_path)
        self.state_detector.update_state(game_state)
        
        # Criar metadados
        metadata = CaptureMetadata(
            timestamp=timestamp.isoformat(),
            game_state=game_state,
            match_id=self.match_id,
            frame_number=self.frame_count,
            adb_id=self.adb_id
        )
        
        # Salvar metadados
        with open(meta_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Organizar por estado (link simbólico)
        state_dir = self.by_state_dir / game_state
        state_dir.mkdir(exist_ok=True)
        try:
            state_dir.joinpath(img_path.name).symlink_to(img_path.resolve())
        except (OSError, NotImplementedError):
            # Se não conseguir criar link simbólico, copiar o arquivo
            import shutil
            shutil.copy2(img_path, state_dir.joinpath(img_path.name))
        
        self.frame_count += 1
        
        return str(img_path)
    
    async def collect_continuous(self, duration_seconds: int, interval_seconds: float = 2.0):
        """Coleta frames continuamente por um período especificado"""
        self.is_running = True
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        logger.info(f"Iniciando coleta contínua: {duration_seconds}s, intervalo {interval_seconds}s")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"ADB ID: {self.adb_id}")
        
        try:
            while time.time() < end_time and self.is_running:
                await self.capture_frame()
                
                if self.frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    remaining = end_time - time.time()
                    logger.info(f"Capturados {self.frame_count} frames | {elapsed:.1f}s elapsed | {remaining:.1f}s remaining")
                
                await asyncio.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Coleta interrompida pelo usuário")
        except Exception as e:
            logger.error(f"Erro durante coleta: {e}")
        finally:
            self.is_running = False
            logger.info(f"Coleta finalizada. Total de frames: {self.frame_count}")
    
    async def collect_event_based(self, max_frames: int = 1000):
        """Coleta frames baseado em eventos (mudanças de estado)"""
        self.is_running = True
        last_state = "unknown"
        
        logger.info(f"Iniciando coleta baseada em eventos: max {max_frames} frames")
        
        try:
            while self.frame_count < max_frames and self.is_running:
                frame_path = await self.capture_frame()
                
                # Detectar mudança de estado
                current_state = self.state_detector.current_state
                if current_state != last_state:
                    logger.info(f"Mudança de estado: {last_state} -> {current_state}")
                    last_state = current_state
                
                await asyncio.sleep(1.0)  # Verificar estado a cada segundo
                
        except KeyboardInterrupt:
            logger.info("Coleta interrompida pelo usuário")
        except Exception as e:
            logger.error(f"Erro durante coleta: {e}")
        finally:
            self.is_running = False
            logger.info(f"Coleta finalizada. Total de frames: {self.frame_count}")
    
    def generate_report(self) -> Dict:
        """Gera relatório da coleta"""
        return {
            "total_frames": self.frame_count,
            "match_id": self.match_id,
            "state_distribution": self._count_states(),
            "duration_seconds": time.time(),
            "output_directory": str(self.output_dir),
            "adb_id": self.adb_id
        }
    
    def _count_states(self) -> Dict[str, int]:
        """Conta distribuição de estados"""
        state_counts = defaultdict(int)
        
        for timestamp, state in self.state_detector.state_history:
            state_counts[state] += 1
        
        return dict(state_counts)


async def main():
    """Função principal para CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Coletor de dataset Brawl Stars")
    parser.add_argument("--adb-id", default="127.0.0.1:5555", 
                       help="ADB device ID (default: 127.0.0.1:5555)")
    parser.add_argument("--output", default="./dataset/raw",
                       help="Diretório de saída (default: ./dataset/raw)")
    parser.add_argument("--duration", type=int, default=300,
                       help="Duração em segundos (default: 300)")
    parser.add_argument("--interval", type=float, default=2.0,
                       help="Intervalo entre captures em segundos (default: 2.0)")
    parser.add_argument("--mode", choices=["continuous", "event"], default="continuous",
                       help="Modo de coleta: continuous ou event-based")
    parser.add_argument("--max-frames", type=int, default=1000,
                       help="Máximo de frames (modo event only)")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    
    # Detectar emulador automaticamente
    logger.info("Detectando emulador...")
    emulator_info = detect_emulator()
    
    if not emulator_info:
        logger.error("Nenhum emulador detectado. Inicie BlueStacks/LDPlayer primeiro.")
        return
    
    logger.info(f"Emulador detectado: {emulator_info}")
    
    # Usar porta detectada
    adb_id = emulator_info.get("adb_id", args.adb_id)
    
    # Criar coletor
    collector = DatasetCollector(
        adb_id=adb_id,
        output_dir=Path(args.output),
        adb_path=None  # Usar detecção automática
    )
    
    # Executar coleta
    if args.mode == "continuous":
        await collector.collect_continuous(args.duration, args.interval)
    else:
        await collector.collect_event_based(args.max_frames)
    
    # Gerar relatório
    report = collector.generate_report()
    
    # Salvar relatório
    report_path = collector.output_dir / "collection_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Relatório salvo em: {report_path}")
    logger.info(f"Distribuição de estados: {report['state_distribution']}")


if __name__ == "__main__":
    asyncio.run(main())
