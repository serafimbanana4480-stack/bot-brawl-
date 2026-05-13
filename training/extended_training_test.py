"""
Extended Training Test - 5 Minutes Complete Training
Executa treinamento completo por 5 minutos com verificação detalhada
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.unified_training_system import UnifiedTrainingSystem
from training.real_reward_system import RealRewardCalculator, GameMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 INICIANDO TREINAMENTO ESTENDIDO - 5 MINUTOS")
    logger.info("=" * 60)
    
    start_time = time.time()
    base_dir = Path(__file__).parent.parent
    
    # Inicializar sistemas
    training_system = UnifiedTrainingSystem(base_dir / "dataset")
    reward_calculator = RealRewardCalculator()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "duration_target": "5 minutes",
        "components": {},
        "rewards": {},
        "integration": {}
    }
    
    # 1. Treinamento BC Estendido
    logger.info("\n" + "=" * 60)
    logger.info("1. TREINAMENTO BC ESTENDIDO")
    logger.info("=" * 60)
    
    bc_start = time.time()
    try:
        massive_dir = base_dir / "dataset" / "synthetic_massive"
        if (massive_dir / "bc_massive.json").exists():
            bc_results = training_system.train_bc_model(
                massive_dir / "bc_massive.json",
                epochs=30  # Mais épocas para treinamento mais longo
            )
            results["components"]["bc"] = {
                "success": True,
                "time": time.time() - bc_start,
                "results": bc_results
            }
            logger.info(f"✅ BC completado em {time.time() - bc_start:.2f}s")
        else:
            logger.warning("⚠️ Dataset BC não encontrado")
            results["components"]["bc"] = {"success": False, "error": "Dataset not found"}
    except Exception as e:
        logger.error(f"❌ Erro no BC: {e}")
        results["components"]["bc"] = {"success": False, "error": str(e)}
    
    # 2. Treinamento CQL Estendido
    logger.info("\n" + "=" * 60)
    logger.info("2. TREINAMENTO CQL ESTENDIDO")
    logger.info("=" * 60)
    
    cql_start = time.time()
    try:
        if (massive_dir / "replay_buffer_massive.json").exists():
            cql_results = training_system.train_cql_model(
                massive_dir / "replay_buffer_massive.json",
                epochs=25  # Mais épocas
            )
            results["components"]["cql"] = {
                "success": True,
                "time": time.time() - cql_start,
                "results": cql_results
            }
            logger.info(f"✅ CQL completado em {time.time() - cql_start:.2f}s")
        else:
            logger.warning("⚠️ Dataset CQL não encontrado")
            results["components"]["cql"] = {"success": False, "error": "Dataset not found"}
    except Exception as e:
        logger.error(f"❌ Erro no CQL: {e}")
        results["components"]["cql"] = {"success": False, "error": str(e)}
    
    # 3. Treinamento YOLO Estendido
    logger.info("\n" + "=" * 60)
    logger.info("3. TREINAMENTO YOLO ESTENDIDO")
    logger.info("=" * 60)
    
    yolo_start = time.time()
    try:
        yolo_data = base_dir / "dataset" / "synthetic_v2"
        if yolo_data.exists():
            yolo_results = training_system.train_yolo_model(
                yolo_data,
                epochs=15  # Mais épocas para YOLO
            )
            results["components"]["yolo"] = {
                "success": True,
                "time": time.time() - yolo_start,
                "results": yolo_results
            }
            logger.info(f"✅ YOLO completado em {time.time() - yolo_start:.2f}s")
        else:
            logger.warning("⚠️ Dataset YOLO não encontrado")
            results["components"]["yolo"] = {"success": False, "error": "Dataset not found"}
    except Exception as e:
        logger.error(f"❌ Erro no YOLO: {e}")
        results["components"]["yolo"] = {"success": False, "error": str(e)}
    
    # 4. Sistema de Rewards Detalhado
    logger.info("\n" + "=" * 60)
    logger.info("4. SISTEMA DE REWARDS DETALHADO")
    logger.info("=" * 60)
    
    rewards_start = time.time()
    try:
        # Criar múltiplas métricas para testar
        test_scenarios = [
            {
                "name": "Cenário Excelente",
                "metrics": GameMetrics(
                    match_id="test_excellent",
                    timestamp=datetime.now().isoformat(),
                    kills=10,
                    deaths=0,
                    damage_dealt=5000,
                    damage_taken=500,
                    survival_time=180,
                    final_position=1,
                    power_cubes_collected=20,
                    enemies_detected=30,
                    detection_accuracy=0.95,
                    good_decisions=50,
                    bad_decisions=2,
                    decision_accuracy=0.96
                )
            },
            {
                "name": "Cenário Bom",
                "metrics": GameMetrics(
                    match_id="test_good",
                    timestamp=datetime.now().isoformat(),
                    kills=5,
                    deaths=2,
                    damage_dealt=3000,
                    damage_taken=1500,
                    survival_time=120,
                    final_position=3,
                    power_cubes_collected=10,
                    enemies_detected=15,
                    detection_accuracy=0.85,
                    good_decisions=30,
                    bad_decisions=5,
                    decision_accuracy=0.86
                )
            },
            {
                "name": "Cenário Ruim",
                "metrics": GameMetrics(
                    match_id="test_poor",
                    timestamp=datetime.now().isoformat(),
                    kills=1,
                    deaths=5,
                    damage_dealt=1000,
                    damage_taken=3000,
                    survival_time=60,
                    final_position=8,
                    power_cubes_collected=3,
                    enemies_detected=5,
                    detection_accuracy=0.60,
                    good_decisions=10,
                    bad_decisions=15,
                    decision_accuracy=0.40
                )
            }
        ]
        
        reward_results = []
        for scenario in test_scenarios:
            reward = reward_calculator.calculate_total_reward(scenario["metrics"])
            reward_results.append({
                "scenario": scenario["name"],
                "total_reward": reward.total_reward,
                "normalized_reward": reward.normalized_reward,
                "breakdown": {
                    "kill_reward": reward.kill_reward,
                    "survival_reward": reward.survival_reward,
                    "damage_reward": reward.damage_reward,
                    "objective_reward": reward.objective_reward,
                    "resource_reward": reward.resource_reward,
                    "detection_reward": reward.detection_reward,
                    "decision_reward": reward.decision_reward,
                    "death_penalty": reward.death_penalty,
                    "bad_decision_penalty": reward.bad_decision_penalty
                }
            })
        
        # Testar tendência
        reward_calculator.add_reward_score(400.0)
        reward_calculator.add_reward_score(350.0)
        reward_calculator.add_reward_score(450.0)
        trend = reward_calculator.get_reward_trend()
        
        results["rewards"] = {
            "success": True,
            "time": time.time() - rewards_start,
            "scenarios": reward_results,
            "trend_analysis": trend
        }
        logger.info(f"✅ Sistema de rewards testado em {time.time() - rewards_start:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Erro no sistema de rewards: {e}")
        results["rewards"] = {"success": False, "error": str(e)}
    
    # 5. Integração Completa
    logger.info("\n" + "=" * 60)
    logger.info("5. TESTE DE INTEGRAÇÃO COMPLETA")
    logger.info("=" * 60)
    
    integration_start = time.time()
    try:
        # Verificar se todos os modelos existem
        models_dir = base_dir / "models"
        integration_checks = {
            "bc_model_exists": (models_dir / "bc" / "best_bc_policy.pt").exists(),
            "cql_model_exists": (models_dir / "cql" / "best_cql_agent.pt").exists(),
            "yolo_model_exists": (models_dir / "yolo_unified_best.pt").exists() or 
                                (base_dir / "runs" / "detect" / "models" / "yolo").exists(),
            "dataset_bc_exists": (massive_dir / "bc_massive.json").exists(),
            "dataset_cql_exists": (massive_dir / "replay_buffer_massive.json").exists(),
            "dataset_yolo_exists": (base_dir / "dataset" / "yolo" / "data.yaml").exists(),
            "training_reports_exist": (base_dir / "dataset" / "training_reports").exists()
        }
        
        # Verificar se os modelos podem ser carregados
        model_loading = {}
        try:
            import torch
            if integration_checks["bc_model_exists"]:
                bc_model = torch.load(models_dir / "bc" / "best_bc_policy.pt", map_location='cpu', weights_only=True)
                model_loading["bc"] = "loaded"
            if integration_checks["cql_model_exists"]:
                cql_model = torch.load(models_dir / "cql" / "best_cql_agent.pt", map_location='cpu', weights_only=True)
                model_loading["cql"] = "loaded"
        except Exception as load_error:
            model_loading["error"] = str(load_error)
        
        results["integration"] = {
            "success": True,
            "time": time.time() - integration_start,
            "checks": integration_checks,
            "model_loading": model_loading
        }
        logger.info(f"✅ Integração testada em {time.time() - integration_start:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Erro na integração: {e}")
        results["integration"] = {"success": False, "error": str(e)}
    
    # Resultado final
    total_time = time.time() - start_time
    results["total_time"] = total_time
    results["target_time"] = 300  # 5 minutos
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMO FINAL")
    logger.info("=" * 60)
    logger.info(f"Tempo total: {total_time:.2f}s ({total_time/60:.2f} minutos)")
    logger.info(f"Tempo alvo: 5 minutos (300s)")
    logger.info(f"Diferença: {abs(total_time - 300):.2f}s")
    
    # Status dos componentes
    for component, data in results["components"].items():
        status = "✅" if data.get("success") else "❌"
        time_str = f" ({data.get('time', 0):.2f}s)" if data.get("success") else ""
        logger.info(f"{status} {component.upper()}{time_str}")
    
    # Status do rewards
    reward_status = "✅" if results["rewards"].get("success") else "❌"
    logger.info(f"{reward_status} REWARDS SYSTEM ({results['rewards'].get('time', 0):.2f}s)")
    
    # Status da integração
    integration_status = "✅" if results["integration"].get("success") else "❌"
    logger.info(f"{integration_status} INTEGRATION ({results['integration'].get('time', 0):.2f}s)")
    
    # Salvar resultados
    output_dir = base_dir / "test_reports"
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "extended_training_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📁 Resultados salvos em: {results_file}")
    
    # Verificar se atingiu o tempo alvo
    if total_time >= 240:  # Pelo menos 4 minutos
        logger.info("✅ Treinamento executado por tempo significativo")
    else:
        logger.info("⚠️ Treinamento mais curto que o esperado")
    
    logger.info("\n🎯 TESTE ESTENDIDO COMPLETO!")
    
    return results

if __name__ == "__main__":
    main()