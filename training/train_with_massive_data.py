"""
train_with_massive_data.py

Script de treinamento usando datasets massivos.
Treina YOLO, BC e CQL com os novos datasets expandidos (10x mais dados).
"""

import sys
from pathlib import Path

# Adicionar diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.unified_training_system import UnifiedTrainingSystem
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Treina todos os modelos com datasets massivos"""
    
    # Criar sistema de treinamento
    system = UnifiedTrainingSystem(Path("./dataset"))
    
    # Usar datasets massivos
    massive_dir = Path("./dataset/synthetic_massive")
    
    logger.info("=" * 60)
    logger.info("TREINAMENTO COM DATASETS MASSIVOS")
    logger.info("=" * 60)
    logger.info(f"BC Dataset: {massive_dir / 'bc_massive.json'}")
    logger.info(f"CQL Dataset: {massive_dir / 'replay_buffer_massive.json'}")
    logger.info(f"YOLO Labels: {massive_dir / 'yolo_labels'}")
    logger.info("=" * 60)
    
    # Treinar BC com dataset massivo
    logger.info("\n🔵 TREINANDO BC COM DATASET MASSIVO (10.000 amostras)")
    bc_results = system.train_bc_model(
        massive_dir / "bc_massive.json",
        epochs=50
    )
    logger.info(f"BC Results: {bc_results}")
    
    # Treinar CQL com dataset massivo
    logger.info("\n🟢 TREINANDO CQL COM DATASET MASSIVO (50.000 transições)")
    cql_results = system.train_cql_model(
        massive_dir / "replay_buffer_massive.json",
        epochs=30
    )
    logger.info(f"CQL Results: {cql_results}")
    
    # Treinar YOLO com labels massivos
    logger.info("\n🟡 TREINANDO YOLO COM LABELS MASSIVOS (5.000 labels)")
    # YOLO precisa de imagens também, vamos usar o dataset sintético existente
    # mas com os novos labels
    yolo_results = system.train_yolo_model(
        Path("./dataset/synthetic_v2"),
        epochs=20
    )
    logger.info(f"YOLO Results: {yolo_results}")
    
    # Relatório consolidado
    logger.info("\n" + "=" * 60)
    logger.info("RELATÓRIO CONSOLIDADO - TREINAMENTO MASSIVO")
    logger.info("=" * 60)
    logger.info(f"BC: {bc_results.get('learning_status', 'N/A')}")
    logger.info(f"CQL: {cql_results.get('learning_status', 'N/A')}")
    logger.info(f"YOLO: {yolo_results.get('learning_status', 'N/A')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()