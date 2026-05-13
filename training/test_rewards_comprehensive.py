"""
Comprehensive Rewards System Test
Testa detalhadamente o sistema de rewards com múltiplos cenários e verificações
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.real_reward_system import RealRewardCalculator, GameMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_rewards_comprehensive():
    logger.info("🎯 TESTE COMPLETO DO SISTEMA DE REWARDS")
    logger.info("=" * 60)
    
    calculator = RealRewardCalculator()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_scenarios": [],
        "edge_cases": [],
        "trend_analysis": [],
        "performance": {}
    }
    
    # 1. Cenários Normais
    logger.info("\n1. TESTANDO CENÁRIOS NORMAIS")
    logger.info("-" * 60)
    
    scenarios = [
        {
            "name": "Performance Perfeita",
            "description": "Jogador ideal com métricas máximas",
            "metrics": GameMetrics(
                match_id="perfect_game",
                timestamp=datetime.now().isoformat(),
                kills=15,
                deaths=0,
                damage_dealt=8000,
                damage_taken=200,
                survival_time=180,
                final_position=1,
                power_cubes_collected=30,
                enemies_detected=50,
                detection_accuracy=0.99,
                good_decisions=100,
                bad_decisions=0,
                decision_accuracy=1.0
            )
        },
        {
            "name": "Performance Boa",
            "description": "Jogador acima da média",
            "metrics": GameMetrics(
                match_id="good_game",
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
        },
        {
            "name": "Performance Média",
            "description": "Jogador mediano",
            "metrics": GameMetrics(
                match_id="average_game",
                timestamp=datetime.now().isoformat(),
                kills=4,
                deaths=4,
                damage_dealt=2000,
                damage_taken=2000,
                survival_time=90,
                final_position=5,
                power_cubes_collected=8,
                enemies_detected=15,
                detection_accuracy=0.70,
                good_decisions=25,
                bad_decisions=10,
                decision_accuracy=0.71
            )
        },
        {
            "name": "Performance Ruim",
            "description": "Jogador abaixo da média",
            "metrics": GameMetrics(
                match_id="poor_game",
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
        },
        {
            "name": "Performance Péssima",
            "description": "Jogador muito ruim",
            "metrics": GameMetrics(
                match_id="terrible_game",
                timestamp=datetime.now().isoformat(),
                kills=0,
                deaths=10,
                damage_dealt=200,
                damage_taken=5000,
                survival_time=20,
                final_position=10,
                power_cubes_collected=0,
                enemies_detected=3,
                detection_accuracy=0.30,
                good_decisions=5,
                bad_decisions=30,
                decision_accuracy=0.14
            )
        }
    ]
    
    for scenario in scenarios:
        reward = calculator.calculate_total_reward(scenario["metrics"])
        scenario_result = {
            "name": scenario["name"],
            "description": scenario["description"],
            "total_reward": reward.total_reward,
            "normalized_reward": reward.normalized_reward,
            "components": {
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
        }
        results["test_scenarios"].append(scenario_result)
        
        logger.info(f"✅ {scenario['name']}: {reward.total_reward:.2f} (normalized: {reward.normalized_reward:.2f})")
    
    # 2. Casos Extremos
    logger.info("\n2. TESTANDO CASOS EXTREMOS")
    logger.info("-" * 60)
    
    edge_cases = [
        {
            "name": "Zero Kills, High Survival",
            "description": "Sobrevive muito sem matar",
            "metrics": GameMetrics(
                match_id="edge_1",
                timestamp=datetime.now().isoformat(),
                kills=0,
                deaths=0,
                damage_dealt=500,
                damage_taken=100,
                survival_time=179,
                final_position=1,
                power_cubes_collected=25,
                enemies_detected=20,
                detection_accuracy=0.90,
                good_decisions=30,
                bad_decisions=2,
                decision_accuracy=0.94
            )
        },
        {
            "name": "High Kills, Low Survival",
            "description": "Mata muito mas morre rápido",
            "metrics": GameMetrics(
                match_id="edge_2",
                timestamp=datetime.now().isoformat(),
                kills=12,
                deaths=1,
                damage_dealt=6000,
                damage_taken=4000,
                survival_time=30,
                final_position=8,
                power_cubes_collected=5,
                enemies_detected=40,
                detection_accuracy=0.95,
                good_decisions=35,
                bad_decisions=15,
                decision_accuracy=0.70
            )
        },
        {
            "name": "Perfect Accuracy, Low Impact",
            "description": "Precisão perfeita mas pouco impacto",
            "metrics": GameMetrics(
                match_id="edge_3",
                timestamp=datetime.now().isoformat(),
                kills=2,
                deaths=3,
                damage_dealt=1000,
                damage_taken=1200,
                survival_time=60,
                final_position=6,
                power_cubes_collected=5,
                enemies_detected=5,
                detection_accuracy=1.0,
                good_decisions=10,
                bad_decisions=0,
                decision_accuracy=1.0
            )
        }
    ]
    
    for case in edge_cases:
        reward = calculator.calculate_total_reward(case["metrics"])
        case_result = {
            "name": case["name"],
            "description": case["description"],
            "total_reward": reward.total_reward,
            "normalized_reward": reward.normalized_reward,
            "components": {
                "kill_reward": reward.kill_reward,
                "survival_reward": reward.survival_reward,
                "damage_reward": reward.damage_reward,
                "decision_reward": reward.decision_reward,
                "death_penalty": reward.death_penalty
            }
        }
        results["edge_cases"].append(case_result)
        
        logger.info(f"✅ {case['name']}: {reward.total_reward:.2f} (normalized: {reward.normalized_reward:.2f})")
    
    # 3. Análise de Tendência
    logger.info("\n3. TESTANDO ANÁLISE DE TENDÊNCIA")
    logger.info("-" * 60)
    
    # Simular progressão
    progression_rewards = [150, 200, 180, 250, 300, 280, 350, 400, 380, 450]
    for reward_val in progression_rewards:
        calculator.add_reward_score(reward_val)
        trend = calculator.get_reward_trend()
        results["trend_analysis"].append({
            "reward": reward_val,
            "trend": trend
        })
        logger.info(f"Reward: {reward_val} → Tendência: {trend}")
    
    # 4. Verificação de Consistência
    logger.info("\n4. VERIFICANDO CONSISTÊNCIA")
    logger.info("-" * 60)
    
    # Verificar se melhores performances têm melhores rewards
    perfect_reward = results["test_scenarios"][0]["total_reward"]
    terrible_reward = results["test_scenarios"][4]["total_reward"]
    
    consistency_check = {
        "perfect_better_than_terrible": perfect_reward > terrible_reward,
        "perfect_reward": perfect_reward,
        "terrible_reward": terrible_reward,
        "difference": perfect_reward - terrible_reward
    }
    
    results["performance"]["consistency"] = consistency_check
    
    if consistency_check["perfect_better_than_terrible"]:
        logger.info(f"✅ Consistência OK: Perfeito ({perfect_reward:.2f}) > Péssimo ({terrible_reward:.2f})")
    else:
        logger.error(f"❌ Consistência FALHOU: Perfeito ({perfect_reward:.2f}) <= Péssimo ({terrible_reward:.2f})")
    
    # Verificar normalização
    normalization_check = {
        "all_normalized": all(0 <= s["normalized_reward"] <= 1 for s in results["test_scenarios"]),
        "perfect_normalized": results["test_scenarios"][0]["normalized_reward"],
        "terrible_normalized": results["test_scenarios"][4]["normalized_reward"]
    }
    
    results["performance"]["normalization"] = normalization_check
    
    if normalization_check["all_normalized"]:
        logger.info(f"✅ Normalização OK: Todos entre 0 e 1")
    else:
        logger.error(f"❌ Normalização FALHOU")
    
    # Salvar resultados
    output_dir = Path(__file__).parent.parent / "test_reports"
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "rewards_comprehensive_test.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📁 Resultados salvos em: {results_file}")
    
    # Resumo final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMO DO TESTE DE REWARDS")
    logger.info("=" * 60)
    logger.info(f"Cenários testados: {len(results['test_scenarios'])}")
    logger.info(f"Casos extremos: {len(results['edge_cases'])}")
    logger.info(f"Pontos de tendência: {len(results['trend_analysis'])}")
    logger.info(f"Consistência: {'✅ OK' if consistency_check['perfect_better_than_terrible'] else '❌ FALHOU'}")
    logger.info(f"Normalização: {'✅ OK' if normalization_check['all_normalized'] else '❌ FALHOU'}")
    
    # Status final
    all_ok = consistency_check["perfect_better_than_terrible"] and normalization_check["all_normalized"]
    
    if all_ok:
        logger.info("\n🎉 SISTEMA DE REWARDS FUNCIONANDO CORRETAMENTE!")
        return {"success": True, "results": results}
    else:
        logger.error("\n❌ SISTEMA DE REWARDS APRESENTA PROBLEMAS!")
        return {"success": False, "results": results}

if __name__ == "__main__":
    result = test_rewards_comprehensive()
    exit(0 if result["success"] else 1)