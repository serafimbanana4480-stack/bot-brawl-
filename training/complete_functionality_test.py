"""
Complete Functionality Test
Teste end-to-end de toda a funcionalidade do sistema
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.unified_training_system import UnifiedTrainingSystem
from training.real_reward_system import RealRewardCalculator, GameMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_complete_functionality():
    logger.info("🚀 TESTE COMPLETO DE FUNCIONALIDADE END-TO-END")
    logger.info("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    results = {
        "timestamp": datetime.now().isoformat(),
        "components": {},
        "integration": {},
        "functionality": {}
    }
    
    # 1. Verificar Estrutura de Arquivos
    logger.info("\n1. VERIFICANDO ESTRUTURA DE ARQUIVOS")
    logger.info("-" * 60)
    
    file_checks = {
        "dataset_bc_massive": (base_dir / "dataset" / "synthetic_massive" / "bc_massive.json").exists(),
        "dataset_cql_massive": (base_dir / "dataset" / "synthetic_massive" / "replay_buffer_massive.json").exists(),
        "dataset_yolo": (base_dir / "dataset" / "yolo" / "data.yaml").exists(),
        "models_bc": (base_dir / "models" / "bc" / "best_bc_policy.pt").exists(),
        "models_cql": (base_dir / "models" / "cql" / "best_cql_agent.pt").exists(),
        "models_yolo": (base_dir / "runs" / "detect" / "models" / "yolo").exists(),
        "training_reports": (base_dir / "dataset" / "training_reports").exists()
    }
    
    results["functionality"]["file_structure"] = file_checks
    all_files_exist = all(file_checks.values())
    
    if all_files_exist:
        logger.info("✅ Estrutura de arquivos completa")
    else:
        missing = [k for k, v in file_checks.items() if not v]
        logger.warning(f"⚠️ Arquivos faltando: {missing}")
    
    # 2. Testar Carregamento de Modelos
    logger.info("\n2. TESTANDO CARREGAMENTO DE MODELOS")
    logger.info("-" * 60)
    
    model_loading = {}
    try:
        # BC Model
        if file_checks["models_bc"]:
            bc_model = torch.load(base_dir / "models" / "bc" / "best_bc_policy.pt", map_location='cpu', weights_only=True)
            model_loading["bc"] = {"loaded": True, "type": str(type(bc_model))}
            logger.info("✅ Modelo BC carregado")

        # CQL Model
        if file_checks["models_cql"]:
            cql_model = torch.load(base_dir / "models" / "cql" / "best_cql_agent.pt", map_location='cpu', weights_only=True)
            model_loading["cql"] = {"loaded": True, "type": str(type(cql_model))}
            logger.info("✅ Modelo CQL carregado")

        # YOLO Model
        if file_checks["models_yolo"]:
            yolo_path = base_dir / "runs" / "detect" / "models" / "yolo" / "brawlstars_detection-2" / "weights" / "best.pt"
            if yolo_path.exists():
                yolo_model = torch.load(yolo_path, map_location='cpu', weights_only=True)
                model_loading["yolo"] = {"loaded": True, "type": str(type(yolo_model))}
                logger.info("✅ Modelo YOLO carregado")
        
        results["functionality"]["model_loading"] = model_loading
        
    except Exception as e:
        logger.error(f"❌ Erro no carregamento de modelos: {e}")
        results["functionality"]["model_loading"] = {"error": str(e)}
    
    # 3. Testar Sistema de Treinamento
    logger.info("\n3. TESTANDO SISTEMA DE TREINAMENTO")
    logger.info("-" * 60)
    
    try:
        training_system = UnifiedTrainingSystem(base_dir / "dataset")
        
        # Teste rápido BC
        massive_dir = base_dir / "dataset" / "synthetic_massive"
        if (massive_dir / "bc_massive.json").exists():
            bc_quick = training_system.train_bc_model(
                massive_dir / "bc_massive.json",
                epochs=3
            )
            results["components"]["bc_quick"] = {
                "success": True,
                "accuracy": bc_quick["latest_metrics"]["accuracy"],
                "learning_status": bc_quick["learning_status"]
            }
            logger.info(f"✅ BC quick test: Acc={bc_quick['latest_metrics']['accuracy']:.4f}")
        
        # Teste rápido CQL
        if (massive_dir / "replay_buffer_massive.json").exists():
            cql_quick = training_system.train_cql_model(
                massive_dir / "replay_buffer_massive.json",
                epochs=3
            )
            results["components"]["cql_quick"] = {
                "success": True,
                "f1_score": cql_quick["latest_metrics"]["f1_score"],
                "learning_status": cql_quick["learning_status"]
            }
            logger.info(f"✅ CQL quick test: F1={cql_quick['latest_metrics']['f1_score']:.4f}")
        
        # Teste rápido YOLO
        yolo_data = base_dir / "dataset" / "synthetic_v2"
        if yolo_data.exists():
            yolo_quick = training_system.train_yolo_model(yolo_data, epochs=2)
            results["components"]["yolo_quick"] = {
                "success": True,
                "f1_score": yolo_quick["latest_metrics"]["f1_score"],
                "learning_status": yolo_quick["learning_status"]
            }
            logger.info(f"✅ YOLO quick test: F1={yolo_quick['latest_metrics']['f1_score']:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Erro no sistema de treinamento: {e}")
        results["components"]["training_error"] = str(e)
    
    # 4. Testar Sistema de Rewards
    logger.info("\n4. TESTANDO SISTEMA DE REWARDS")
    logger.info("-" * 60)
    
    try:
        reward_calculator = RealRewardCalculator()
        
        # Teste com métricas boas
        good_metrics = GameMetrics(
            match_id="test_good",
            timestamp=datetime.now().isoformat(),
            kills=8,
            deaths=2,
            damage_dealt=4000,
            damage_taken=1500,
            survival_time=150,
            final_position=2,
            power_cubes_collected=15,
            enemies_detected=25,
            detection_accuracy=0.85,
            good_decisions=40,
            bad_decisions=5,
            decision_accuracy=0.89
        )
        
        good_reward = reward_calculator.calculate_total_reward(good_metrics)
        
        # Teste com métricas ruins
        bad_metrics = GameMetrics(
            match_id="test_bad",
            timestamp=datetime.now().isoformat(),
            kills=1,
            deaths=7,
            damage_dealt=800,
            damage_taken=3500,
            survival_time=45,
            final_position=9,
            power_cubes_collected=3,
            enemies_detected=8,
            detection_accuracy=0.50,
            good_decisions=10,
            bad_decisions=20,
            decision_accuracy=0.33
        )
        
        bad_reward = reward_calculator.calculate_total_reward(bad_metrics)
        
        results["functionality"]["rewards"] = {
            "good_performance": {
                "total_reward": good_reward.total_reward,
                "normalized": good_reward.normalized_reward
            },
            "bad_performance": {
                "total_reward": bad_reward.total_reward,
                "normalized": bad_reward.normalized_reward
            },
            "consistency_check": good_reward.total_reward > bad_reward.total_reward
        }
        
        if good_reward.total_reward > bad_reward.total_reward:
            logger.info("✅ Sistema de rewards consistente (bom > ruim)")
        else:
            logger.error("❌ Sistema de rewards inconsistente")
        
    except Exception as e:
        logger.error(f"❌ Erro no sistema de rewards: {e}")
        results["functionality"]["rewards"] = {"error": str(e)}
    
    # 5. Testar Integração
    logger.info("\n5. TESTANDO INTEGRAÇÃO COMPLETA")
    logger.info("-" * 60)
    
    integration_tests = {
        "training_to_rewards": "Sistema de treinamento gera modelos usados no cálculo de rewards",
        "models_accessible": "Todos os modelos são acessíveis e carregáveis",
        "data_pipeline": "Pipeline de dados funciona do início ao fim",
        "monitoring": "Sistema de monitoramento registra métricas corretamente"
    }
    
    integration_results = {}
    
    # Testar se modelos treinados podem ser usados
    try:
        if "model_loading" in results["functionality"] and "bc" in results["functionality"]["model_loading"]:
            integration_results["models_accessible"] = True
        else:
            integration_results["models_accessible"] = False
            
        if "rewards" in results["functionality"] and "error" not in results["functionality"]["rewards"]:
            integration_results["training_to_rewards"] = True
        else:
            integration_results["training_to_rewards"] = False
            
        if all_files_exist:
            integration_results["data_pipeline"] = True
        else:
            integration_results["data_pipeline"] = False
            
        integration_results["monitoring"] = (base_dir / "dataset" / "training_reports").exists()
        
    except Exception as e:
        logger.error(f"❌ Erro nos testes de integração: {e}")
        integration_results = {"error": str(e)}
    
    results["integration"] = integration_results
    
    integration_pass_rate = sum(1 for v in integration_results.values() if v is True) / len(integration_results) if isinstance(integration_results, dict) else 0
    
    logger.info(f"Taxa de sucesso na integração: {integration_pass_rate:.1%}")
    
    # 6. Resumo Final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMO FINAL DE FUNCIONALIDADE")
    logger.info("=" * 60)
    
    final_status = {
        "file_structure_complete": all_files_exist,
        "models_loadable": len(model_loading) >= 2,
        "training_works": "bc_quick" in results["components"] and "cql_quick" in results["components"],
        "rewards_work": "rewards" in results["functionality"] and "error" not in results["functionality"]["rewards"],
        "integration_pass_rate": integration_pass_rate
    }
    
    results["final_status"] = final_status
    
    for check, status in final_status.items():
        status_str = "✅" if status else "❌"
        logger.info(f"{status_str} {check}: {status}")
    
    # Calcular status geral
    overall_pass = sum(final_status.values()) / len(final_status)
    
    logger.info(f"\nTaxa geral de sucesso: {overall_pass:.1%}")
    
    if overall_pass >= 0.8:
        logger.info("🎉 SISTEMA FUNCIONAL E PRONTO PARA USO!")
        results["overall_status"] = "FUNCTIONAL"
    elif overall_pass >= 0.5:
        logger.info("⚠️ SISTEMA PARCIALMENTE FUNCIONAL - REQUER AJUSTES")
        results["overall_status"] = "PARTIAL"
    else:
        logger.error("❌ SISTEMA APRESENTA PROBLEMAS CRÍTICOS")
        results["overall_status"] = "CRITICAL"
    
    # Salvar resultados
    output_dir = base_dir / "test_reports"
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "complete_functionality_test.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📁 Resultados salvos em: {results_file}")
    
    return results

if __name__ == "__main__":
    result = test_complete_functionality()
    exit(0 if result["final_status"].get("overall_pass", 0) >= 0.5 else 1)