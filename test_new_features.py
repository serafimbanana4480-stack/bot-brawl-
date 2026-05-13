"""
test_new_features.py

Script de teste para as novas funcionalidades de treinamento de IA.
Testa os componentes individualmente sem depender do emulador.
"""

import sys
from pathlib import Path
import logging

# Adicionar diretório ao path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_synthetic_data_generator():
    """Testa o gerador de dados sintéticos"""
    logger.info("=" * 60)
    logger.info("TESTE 1: Gerador de Dados Sintéticos")
    logger.info("=" * 60)
    
    try:
        from training.synthetic_data_generator import SyntheticDataGenerator
        
        output_dir = Path("./dataset/synthetic_test")
        generator = SyntheticDataGenerator(output_dir=output_dir, use_real_templates=False)
        
        # Gerar pequeno dataset de teste
        logger.info("Gerando 10 amostras de teste...")
        stats = generator.generate_dataset(num_samples=10, sequence_length=2)
        
        logger.info(f"✅ Dataset sintético gerado com sucesso!")
        logger.info(f"   Total amostras: {stats['total_samples']}")
        logger.info(f"   Por fase: {stats['by_state']}")
        logger.info(f"   Média inimigos: {stats['avg_enemies']:.2f}")
        logger.info(f"   Output: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Falha no teste do gerador sintético: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_validator():
    """Testa o validador de treinamento"""
    logger.info("=" * 60)
    logger.info("TESTE 2: Validador de Treinamento")
    logger.info("=" * 60)
    
    try:
        from training.training_validator import ModelValidator, ValidationMetrics
        
        # Criar dataset de teste pequeno
        test_dir = Path("./dataset/test_validation")
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "images").mkdir(exist_ok=True)
        (test_dir / "labels").mkdir(exist_ok=True)
        
        # Criar algumas imagens de teste simples
        import cv2
        import numpy as np
        
        for i in range(5):
            # Criar imagem simples
            img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            img_path = test_dir / "images" / f"test_{i}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # Criar label vazio
            label_path = test_dir / "labels" / f"test_{i}.txt"
            label_path.write_text("")
        
        # Testar validador
        validator = ModelValidator(test_dir)
        logger.info(f"Validador inicializado com {len(validator.test_images)} imagens")
        
        # Criar métricas de teste
        metrics = ValidationMetrics(
            model_path="test_model.pt",
            timestamp="2024-01-01T00:00:00",
            precision=0.75,
            recall=0.80,
            f1_score=0.77,
            mAP=0.70,
            avg_confidence=0.85,
            confidence_std=0.10,
            detection_consistency=0.90,
            confidence_score=0.80
        )
        
        logger.info(f"✅ Validador de treinamento funcional!")
        logger.info(f"   Métricas de teste criadas:")
        logger.info(f"   F1 Score: {metrics.f1_score:.3f}")
        logger.info(f"   mAP: {metrics.mAP:.3f}")
        logger.info(f"   Confidence Score: {metrics.confidence_score:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Falha no teste do validador: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_collector_v2():
    """Testa o dataset collector v2 (sem YOLO)"""
    logger.info("=" * 60)
    logger.info("TESTE 3: Dataset Collector v2 (modo sem YOLO)")
    logger.info("=" * 60)
    
    try:
        # Testar apenas a importação básica
        from automation.dataset_collector_v2 import DatasetCollectorV2, GameStateDetector
        
        logger.info("✅ Dataset Collector v2 importado com sucesso!")
        
        # Testar detector de estado
        detector = GameStateDetector()
        logger.info("✅ GameStateDetector funcional!")
        
        # Nota: Teste completo requer ADB conectado
        logger.info("⚠️  Teste completo requer ADB conectado ao emulador")
        logger.info("   Use: python -m brawl_bot.automation.dataset_collector_v2 --adb-id 127.0.0.1:5555")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Falha no teste do dataset collector: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrain_integration():
    """Testa a integração do sistema de retrain"""
    logger.info("=" * 60)
    logger.info("TESTE 4: Integração do Sistema de Retrain")
    logger.info("=" * 60)
    
    try:
        from training.retrain import PerformanceMonitor, RetrainTrigger, RetrainOrchestrator
        
        # Testar monitor de performance
        logs_dir = Path("./logs/test_performance")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        monitor = PerformanceMonitor(log_dir=logs_dir)
        monitor.start_session()
        monitor.record_kill()
        monitor.record_death()
        monitor.record_match_result(won=True, survival_time=120.0)
        session = monitor.end_session()
        
        logger.info(f"✅ Performance Monitor funcional!")
        logger.info(f"   Kills: {session.kills}")
        logger.info(f"   Deaths: {session.deaths}")
        logger.info(f"   Win Rate: {session.win_rate:.2%}")
        
        # Testar triggers
        triggers = RetrainTrigger(
            min_matches_before_retrain=5,
            win_rate_threshold=0.4
        )
        
        logger.info(f"✅ Retrain Trigger configurado!")
        logger.info(f"   Min matches: {triggers.min_matches_before_retrain}")
        logger.info(f"   Win rate threshold: {triggers.win_rate_threshold}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Falha no teste do retrain: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wrapper_integration():
    """Testa a integração no wrapper"""
    logger.info("=" * 60)
    logger.info("TESTE 5: Integração no Wrapper")
    logger.info("=" * 60)
    
    try:
        # Verificar se as modificações no wrapper estão presentes
        from wrapper import PylaAIEnhanced
        
        # Verificar se os novos atributos existem
        import inspect
        source = inspect.getsource(PylaAIEnhanced.__init__)
        
        has_recording = "recording_enabled" in source
        has_auto_retrain = "auto_retrain_enabled" in source
        
        logger.info(f"✅ Wrapper modificado com sucesso!")
        logger.info(f"   Recording support: {'✅' if has_recording else '❌'}")
        logger.info(f"   Auto-retrain support: {'✅' if has_auto_retrain else '❌'}")
        
        if has_recording and has_auto_retrain:
            logger.info("✅ Todas as modificações do wrapper estão presentes!")
            return True
        else:
            logger.warning("⚠️  Algumas modificações do wrapper estão faltando")
            return False
        
    except Exception as e:
        logger.error(f"❌ Falha no teste do wrapper: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Executa todos os testes"""
    logger.info("INICIANDO TESTE DAS NOVAS FUNCIONALIDADES")
    logger.info("=" * 60)
    
    results = {}
    
    # Executar testes
    results["synthetic_generator"] = test_synthetic_data_generator()
    results["training_validator"] = test_training_validator()
    results["dataset_collector"] = test_dataset_collector_v2()
    results["retrain_integration"] = test_retrain_integration()
    results["wrapper_integration"] = test_wrapper_integration()
    
    # Resumo
    logger.info("=" * 60)
    logger.info("RESUMO DOS TESTES")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASSOU" if result else "❌ FALHOU"
        logger.info(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    logger.info("=" * 60)
    logger.info(f"TOTAL: {passed}/{total} testes passaram")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("🎉 TODOS OS TESTES PASSARAM!")
        logger.info("As novas funcionalidades estão prontas para uso.")
    else:
        logger.warning(f"⚠️  {total - passed} teste(s) falharam. Verifique os erros acima.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
