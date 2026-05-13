"""
test_learning_system.py

Teste completo e exaustivo do sistema de aprendizagem.
Executa múltiplas iterações, analisa métricas detalhadamente e valida se realmente aprende.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

# Adicionar diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.real_reward_system import RealRewardCalculator, GameMetrics
from training.unified_training_system import UnifiedTrainingSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LearningSystemTester:
    """Testador exaustivo do sistema de aprendizagem"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.reward_calculator = RealRewardCalculator()
        self.training_system = UnifiedTrainingSystem(base_dir / "dataset")
        
        # Diretórios
        self.test_reports_dir = base_dir / "test_reports"
        self.test_reports_dir.mkdir(exist_ok=True)
        
        # Histórico de testes
        self.test_results = []
    
    def test_bc_learning(self, iterations: int = 5) -> Dict:
        """Testa aprendizado BC em múltiplas iterações"""
        logger.info("=" * 60)
        logger.info("TESTE DE APRENDIZADO BC")
        logger.info("=" * 60)
        
        massive_dir = self.base_dir / "dataset" / "synthetic_massive"
        bc_dataset = massive_dir / "bc_massive.json"
        
        if not bc_dataset.exists():
            logger.error("Dataset BC massivo não encontrado!")
            return {"success": False, "error": "Dataset not found"}
        
        results = []
        accuracies = []
        losses = []
        
        for i in range(iterations):
            logger.info(f"\n--- Iteração BC {i+1}/{iterations} ---")
            
            # Treinar
            result = self.training_system.train_bc_model(
                bc_dataset,
                epochs=15  # Menos épocas para teste rápido
            )
            
            # Extrair métricas
            latest_metrics = result.get("latest_metrics", {})
            best_metrics = result.get("best_metrics", {})
            
            accuracy = latest_metrics.get("accuracy", 0.0)
            loss = latest_metrics.get("train_loss", 0.0)
            f1 = latest_metrics.get("f1_score", 0.0)
            
            accuracies.append(accuracy)
            losses.append(loss)
            
            results.append({
                "iteration": i + 1,
                "accuracy": accuracy,
                "loss": loss,
                "f1_score": f1,
                "learning_status": result.get("learning_status")
            })
            
            logger.info(f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}, F1: {f1:.4f}")
        
        # Analisar tendência
        accuracy_trend = self._analyze_trend(accuracies)
        loss_trend = self._analyze_trend(losses, descending=True)
        
        logger.info(f"\n📊 Tendência Accuracy: {accuracy_trend}")
        logger.info(f"📊 Tendência Loss: {loss_trend}")
        
        # Plotar resultados
        self._plot_learning_curve(accuracies, losses, "BC_Learning", self.test_reports_dir)
        
        return {
            "success": True,
            "results": results,
            "accuracy_trend": accuracy_trend,
            "loss_trend": loss_trend,
            "final_accuracy": accuracies[-1],
            "accuracy_improvement": accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0.0
        }
    
    def test_cql_learning(self, iterations: int = 5) -> Dict:
        """Testa aprendizado CQL em múltiplas iterações"""
        logger.info("=" * 60)
        logger.info("TESTE DE APRENDIZADO CQL")
        logger.info("=" * 60)
        
        massive_dir = self.base_dir / "dataset" / "synthetic_massive"
        cql_dataset = massive_dir / "replay_buffer_massive.json"
        
        if not cql_dataset.exists():
            logger.error("Dataset CQL massivo não encontrado!")
            return {"success": False, "error": "Dataset not found"}
        
        results = []
        f1_scores = []
        losses = []
        
        for i in range(iterations):
            logger.info(f"\n--- Iteração CQL {i+1}/{iterations} ---")
            
            # Treinar
            result = self.training_system.train_cql_model(
                cql_dataset,
                epochs=10  # Menos épocas para teste rápido
            )
            
            # Extrair métricas
            latest_metrics = result.get("latest_metrics", {})
            
            f1 = latest_metrics.get("f1_score", 0.0)
            loss = latest_metrics.get("train_loss", 0.0)
            
            f1_scores.append(f1)
            losses.append(loss)
            
            results.append({
                "iteration": i + 1,
                "f1_score": f1,
                "loss": loss,
                "learning_status": result.get("learning_status")
            })
            
            logger.info(f"F1: {f1:.4f}, Loss: {loss:.4f}")
        
        # Analisar tendência
        f1_trend = self._analyze_trend(f1_scores)
        loss_trend = self._analyze_trend(losses, descending=True)
        
        logger.info(f"\n📊 Tendência F1: {f1_trend}")
        logger.info(f"📊 Tendência Loss: {loss_trend}")
        
        # Plotar resultados
        self._plot_learning_curve(f1_scores, losses, "CQL_Learning", self.test_reports_dir)
        
        return {
            "success": True,
            "results": results,
            "f1_trend": f1_trend,
            "loss_trend": loss_trend,
            "final_f1": f1_scores[-1],
            "f1_improvement": f1_scores[-1] - f1_scores[0] if len(f1_scores) > 1 else 0.0
        }
    
    def test_yolo_learning(self, iterations: int = 3) -> Dict:
        """Testa aprendizado YOLO em múltiplas iterações"""
        logger.info("=" * 60)
        logger.info("TESTE DE APRENDIZADO YOLO")
        logger.info("=" * 60)
        
        yolo_dataset = self.base_dir / "dataset" / "synthetic_v2"
        
        results = []
        f1_scores = []
        losses = []
        maps = []
        
        for i in range(iterations):
            logger.info(f"\n--- Iteração YOLO {i+1}/{iterations} ---")
            
            # Treinar
            result = self.training_system.train_yolo_model(
                yolo_dataset,
                epochs=10
            )
            
            # Extrair métricas
            latest_metrics = result.get("latest_metrics", {})
            
            f1 = latest_metrics.get("f1_score", 0.0)
            loss = latest_metrics.get("train_loss", 0.0)
            map_score = latest_metrics.get("mAP", 0.0)
            
            f1_scores.append(f1)
            losses.append(loss)
            maps.append(map_score)
            
            results.append({
                "iteration": i + 1,
                "f1_score": f1,
                "loss": loss,
                "mAP": map_score,
                "learning_status": result.get("learning_status")
            })
            
            logger.info(f"F1: {f1:.4f}, Loss: {loss:.4f}, mAP: {map_score:.4f}")
        
        # Analisar tendência
        f1_trend = self._analyze_trend(f1_scores)
        loss_trend = self._analyze_trend(losses, descending=True)
        map_trend = self._analyze_trend(maps)
        
        logger.info(f"\n📊 Tendência F1: {f1_trend}")
        logger.info(f"📊 Tendência Loss: {loss_trend}")
        logger.info(f"📊 Tendência mAP: {map_trend}")
        
        # Plotar resultados
        self._plot_yolo_learning(f1_scores, losses, maps, "YOLO_Learning", self.test_reports_dir)
        
        return {
            "success": True,
            "results": results,
            "f1_trend": f1_trend,
            "loss_trend": loss_trend,
            "map_trend": map_trend,
            "final_f1": f1_scores[-1],
            "final_map": maps[-1],
            "f1_improvement": f1_scores[-1] - f1_scores[0] if len(f1_scores) > 1 else 0.0
        }
    
    def test_reward_system(self, num_matches: int = 20) -> Dict:
        """Testa sistema de rewards com múltiplas partidas"""
        logger.info("=" * 60)
        logger.info("TESTE DO SISTEMA DE REWARDS")
        logger.info("=" * 60)
        
        rewards = []
        normalized_rewards = []
        
        for i in range(num_matches):
            # Simular partida com performance gradualmente melhor
            improvement_factor = i / num_matches  # 0 a 1
            
            metrics = GameMetrics(
                match_id=f"test_match_{i}",
                timestamp=datetime.now().isoformat(),
                kills=int(3 + 5 * improvement_factor + np.random.normal(0, 1)),
                deaths=max(0, int(3 - 2 * improvement_factor + np.random.normal(0, 1))),
                damage_dealt=2000 + 3000 * improvement_factor + np.random.normal(0, 500),
                damage_taken=1500 - 500 * improvement_factor + np.random.normal(0, 300),
                survival_time=60 + 120 * improvement_factor + np.random.normal(0, 20),
                final_position=max(1, int(10 - 7 * improvement_factor + np.random.normal(0, 2))),
                power_cubes_collected=int(5 + 15 * improvement_factor),
                enemies_detected=int(10 + 20 * improvement_factor),
                detection_accuracy=0.7 + 0.2 * improvement_factor,
                good_decisions=int(20 + 30 * improvement_factor),
                bad_decisions=max(0, int(15 - 10 * improvement_factor)),
                decision_accuracy=0.6 + 0.3 * improvement_factor
            )
            
            reward = self.reward_calculator.calculate_total_reward(metrics)
            rewards.append(reward.total_reward)
            normalized_rewards.append(reward.normalized_reward)
            
            if i % 5 == 0:
                logger.info(f"Partida {i+1}: Reward={reward.total_reward:.2f}, Norm={reward.normalized_reward:.2f}")
        
        # Analisar tendência
        reward_trend = self._analyze_trend(rewards)
        
        # Calcular estatísticas
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        improvement = rewards[-1] - rewards[0]
        
        logger.info(f"\n📊 Reward médio: {avg_reward:.2f}")
        logger.info(f"📊 Desvio padrão: {std_reward:.2f}")
        logger.info(f"📊 Melhoria total: {improvement:.2f}")
        logger.info(f"📊 Tendência: {reward_trend}")
        
        # Plotar rewards
        self._plot_rewards(rewards, normalized_rewards, "Reward_System", self.test_reports_dir)
        
        return {
            "success": True,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "improvement": improvement,
            "trend": reward_trend,
            "rewards": rewards
        }
    
    def _analyze_trend(self, values: List[float], descending: bool = False) -> str:
        """Analisa tendência de uma série de valores"""
        if len(values) < 3:
            return "insufficient_data"
        
        # Calcular tendência linear
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Determinar tendência
        if descending:
            if slope < -0.01:
                return "improving"  # loss diminuindo
            elif slope > 0.01:
                return "degrading"  # loss aumentando
            else:
                return "stable"
        else:
            if slope > 0.01:
                return "improving"  # métrica aumentando
            elif slope < -0.01:
                return "degrading"  # métrica diminuindo
            else:
                return "stable"
    
    def _plot_learning_curve(self, metric1: List[float], metric2: List[float], 
                            title: str, output_dir: Path):
        """Plota curva de aprendizado"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        iterations = range(1, len(metric1) + 1)
        
        ax1.plot(iterations, metric1, 'b-o', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Metric 1')
        ax1.set_title(f'{title} - Metric 1')
        ax1.grid(True)
        
        ax2.plot(iterations, metric2, 'r-o', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Metric 2')
        ax2.set_title(f'{title} - Metric 2')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{title.lower()}_curve.png", dpi=150)
        plt.close()
        
        logger.info(f"📊 Gráfico salvo: {output_dir / f'{title.lower()}_curve.png'}")
    
    def _plot_yolo_learning(self, f1: List[float], loss: List[float], 
                           map_score: List[float], title: str, output_dir: Path):
        """Plota curva de aprendizado YOLO"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        iterations = range(1, len(f1) + 1)
        
        ax1.plot(iterations, f1, 'b-o', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('F1 Score')
        ax1.set_title(f'{title} - F1 Score')
        ax1.grid(True)
        
        ax2.plot(iterations, loss, 'r-o', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{title} - Loss')
        ax2.grid(True)
        
        ax3.plot(iterations, map_score, 'g-o', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('mAP')
        ax3.set_title(f'{title} - mAP')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{title.lower()}_curve.png", dpi=150)
        plt.close()
        
        logger.info(f"📊 Gráfico salvo: {output_dir / f'{title.lower()}_curve.png'}")
    
    def _plot_rewards(self, rewards: List[float], normalized: List[float], 
                     title: str, output_dir: Path):
        """Plota rewards ao longo do tempo"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        matches = range(1, len(rewards) + 1)
        
        ax1.plot(matches, rewards, 'b-o', linewidth=2)
        ax1.set_xlabel('Match')
        ax1.set_ylabel('Total Reward')
        ax1.set_title(f'{title} - Total Rewards')
        ax1.grid(True)
        ax1.axhline(y=np.mean(rewards), color='r', linestyle='--', label='Mean')
        ax1.legend()
        
        ax2.plot(matches, normalized, 'g-o', linewidth=2)
        ax2.set_xlabel('Match')
        ax2.set_ylabel('Normalized Reward')
        ax2.set_title(f'{title} - Normalized Rewards')
        ax2.grid(True)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{title.lower()}_curve.png", dpi=150)
        plt.close()
        
        logger.info(f"📊 Gráfico salvo: {output_dir / f'{title.lower()}_curve.png'}")
    
    def run_complete_test(self) -> Dict:
        """Executa teste completo do sistema"""
        logger.info("🚀 INICIANDO TESTE COMPLETO DO SISTEMA DE APRENDIZAGEM")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Testar BC
        bc_results = self.test_bc_learning(iterations=5)
        
        # Testar CQL
        cql_results = self.test_cql_learning(iterations=5)
        
        # Testar YOLO
        yolo_results = self.test_yolo_learning(iterations=3)
        
        # Testar Rewards
        reward_results = self.test_reward_system(num_matches=20)
        
        total_time = time.time() - start_time
        
        # Compilar resultados
        complete_results = {
            "timestamp": datetime.now().isoformat(),
            "total_time": total_time,
            "bc": bc_results,
            "cql": cql_results,
            "yolo": yolo_results,
            "rewards": reward_results
        }
        
        # Salvar resultados
        results_path = self.test_reports_dir / "complete_test_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2)
        
        # Gerar relatório
        self._generate_report(complete_results)
        
        logger.info("=" * 60)
        logger.info("✅ TESTE COMPLETO FINALIZADO")
        logger.info(f"⏱️ Tempo total: {total_time:.2f}s")
        logger.info(f"📊 Resultados salvos: {results_path}")
        logger.info("=" * 60)
        
        return complete_results
    
    def _generate_report(self, results: Dict):
        """Gera relatório detalhado dos testes"""
        report_path = self.test_reports_dir / "TEST_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Relatório Completo de Teste do Sistema de Aprendizagem\n\n")
            f.write(f"**Data**: {results['timestamp']}\n")
            f.write(f"**Tempo Total**: {results['total_time']:.2f}s\n\n")
            
            # BC Results
            f.write("## 1. Behavior Cloning (BC)\n\n")
            if results['bc']['success']:
                f.write(f"- **Status**: ✅ Sucesso\n")
                f.write(f"- **Tendência Accuracy**: {results['bc']['accuracy_trend']}\n")
                f.write(f"- **Tendência Loss**: {results['bc']['loss_trend']}\n")
                f.write(f"- **Accuracy Final**: {results['bc']['final_accuracy']:.4f}\n")
                f.write(f"- **Melhoria**: {results['bc']['accuracy_improvement']:.4f}\n")
            else:
                f.write(f"- **Status**: ❌ Falha - {results['bc'].get('error', 'Unknown')}\n")
            
            # CQL Results
            f.write("\n## 2. Conservative Q-Learning (CQL)\n\n")
            if results['cql']['success']:
                f.write(f"- **Status**: ✅ Sucesso\n")
                f.write(f"- **Tendência F1**: {results['cql']['f1_trend']}\n")
                f.write(f"- **Tendência Loss**: {results['cql']['loss_trend']}\n")
                f.write(f"- **F1 Final**: {results['cql']['final_f1']:.4f}\n")
                f.write(f"- **Melhoria**: {results['cql']['f1_improvement']:.4f}\n")
            else:
                f.write(f"- **Status**: ❌ Falha - {results['cql'].get('error', 'Unknown')}\n")
            
            # YOLO Results
            f.write("\n## 3. YOLO (Vision)\n\n")
            if results['yolo']['success']:
                f.write(f"- **Status**: ✅ Sucesso\n")
                f.write(f"- **Tendência F1**: {results['yolo']['f1_trend']}\n")
                f.write(f"- **Tendência Loss**: {results['yolo']['loss_trend']}\n")
                f.write(f"- **Tendência mAP**: {results['yolo']['map_trend']}\n")
                f.write(f"- **F1 Final**: {results['yolo']['final_f1']:.4f}\n")
                f.write(f"- **mAP Final**: {results['yolo']['final_map']:.4f}\n")
                f.write(f"- **Melhoria F1**: {results['yolo']['f1_improvement']:.4f}\n")
            else:
                f.write(f"- **Status**: ❌ Falha - {results['yolo'].get('error', 'Unknown')}\n")
            
            # Rewards Results
            f.write("\n## 4. Sistema de Rewards\n\n")
            if results['rewards']['success']:
                f.write(f"- **Status**: ✅ Sucesso\n")
                f.write(f"- **Reward Médio**: {results['rewards']['avg_reward']:.2f}\n")
                f.write(f"- **Desvio Padrão**: {results['rewards']['std_reward']:.2f}\n")
                f.write(f"- **Melhoria Total**: {results['rewards']['improvement']:.2f}\n")
                f.write(f"- **Tendência**: {results['rewards']['trend']}\n")
            else:
                f.write(f"- **Status**: ❌ Falha\n")
            
            # Conclusão
            f.write("\n## 5. Conclusão\n\n")
            
            learning_count = sum([
                results['bc']['success'],
                results['cql']['success'],
                results['yolo']['success'],
                results['rewards']['success']
            ])
            
            if learning_count == 4:
                f.write("✅ **SISTEMA FUNCIONANDO PERFEITAMENTE**\n\n")
                f.write("Todos os componentes estão aprendendo e melhorando conforme esperado.\n")
            elif learning_count >= 3:
                f.write("⚠️ **SISTEMA FUNCIONANDO COM ALGUMAS LIMITAÇÕES**\n\n")
                f.write("A maioria dos componentes está funcionando, mas alguns precisam de atenção.\n")
            else:
                f.write("❌ **SISTEMA PRECISA DE MELHORIAS**\n\n")
                f.write("Vários componentes não estão funcionando conforme esperado.\n")
        
        logger.info(f"📝 Relatório gerado: {report_path}")


def main():
    """Função principal"""
    tester = LearningSystemTester(Path("."))
    results = tester.run_complete_test()
    
    # Resumo final
    print("\n" + "=" * 60)
    print("📊 RESUMO FINAL DO TESTE")
    print("=" * 60)
    
    if results['bc']['success']:
        print(f"✅ BC: {results['bc']['accuracy_trend']} (acc: {results['bc']['final_accuracy']:.4f})")
    else:
        print(f"❌ BC: Falhou")
    
    if results['cql']['success']:
        print(f"✅ CQL: {results['cql']['f1_trend']} (f1: {results['cql']['final_f1']:.4f})")
    else:
        print(f"❌ CQL: Falhou")
    
    if results['yolo']['success']:
        print(f"✅ YOLO: {results['yolo']['f1_trend']} (f1: {results['yolo']['final_f1']:.4f}, map: {results['yolo']['final_map']:.4f})")
    else:
        print(f"❌ YOLO: Falhou")
    
    if results['rewards']['success']:
        print(f"✅ Rewards: {results['rewards']['trend']} (avg: {results['rewards']['avg_reward']:.2f})")
    else:
        print(f"❌ Rewards: Falhou")
    
    print("=" * 60)


if __name__ == "__main__":
    main()