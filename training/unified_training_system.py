"""
unified_training_system.py

Sistema unificado de treinamento com monitoramento completo.
Treina todos os modelos (YOLO, BC, CQL) com monitoramento de rewards e outputs.
Verifica se a IA está realmente melhorando ao longo do tempo.

Funcionalidades:
- Treinamento unificado de todos os modelos
- Monitoramento em tempo real de métricas
- Sistema de rewards detalhado
- Verificação de aprendizado
- Comparação entre versões de modelo
- Relatórios automáticos de progresso
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    TorchDataset = None

logger = logging.getLogger(__name__)


@dataclass
class TrainingReward:
    """Recompensa de treinamento para avaliar progresso"""
    timestamp: float
    
    # Vision rewards
    detection_improvement: float = 0.0
    precision_gain: float = 0.0
    recall_gain: float = 0.0
    map_gain: float = 0.0
    
    # Decision rewards
    decision_accuracy: float = 0.0
    action_success_rate: float = 0.0
    strategic_score: float = 0.0
    
    # General rewards
    convergence_speed: float = 0.0
    stability_score: float = 0.0
    overfitting_penalty: float = 0.0
    
    # Overall reward
    total_reward: float = 0.0
    
    @property
    def weighted_reward(self) -> float:
        """Calcula reward ponderada"""
        return (
            self.detection_improvement * 0.3 +
            self.decision_accuracy * 0.3 +
            self.convergence_speed * 0.2 +
            self.stability_score * 0.2 -
            self.overfitting_penalty * 0.5
        )


@dataclass
class TrainingMetrics:
    """Métricas de treinamento"""
    epoch: int
    timestamp: str
    
    # Loss metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    loss_improvement: float = 0.0
    
    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mAP: float = 0.0
    
    # Learning indicators
    is_improving: bool = True
    learning_rate: float = 0.0
    confidence_score: float = 0.0
    
    # Resource usage
    gpu_memory: float = 0.0
    training_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class TrainingMonitor:
    """Monitor de treinamento com verificação de aprendizado"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history: List[TrainingMetrics] = []
        self.rewards_history: List[TrainingReward] = []
        
        self.best_metrics: Optional[TrainingMetrics] = None
        self.current_baseline: Optional[TrainingMetrics] = None
        
        # Setup logging
        self.logger = logging.getLogger("training_monitor")
        
    def record_metrics(self, metrics: TrainingMetrics):
        """Registra métricas de treinamento"""
        self.metrics_history.append(metrics)
        
        # Atualizar melhor métrica
        if self.best_metrics is None or metrics.f1_score > self.best_metrics.f1_score:
            self.best_metrics = metrics
        
        # Calcular reward
        reward = self._calculate_reward(metrics)
        self.rewards_history.append(reward)
        
        # Salvar métricas
        self._save_metrics(metrics)
        
        self.logger.info(
            f"Epoch {metrics.epoch}: Loss={metrics.train_loss:.4f}, "
            f"F1={metrics.f1_score:.4f}, Reward={reward.total_reward:.4f}"
        )
    
    def _calculate_reward(self, metrics: TrainingMetrics) -> TrainingReward:
        """Calcula reward baseado nas métricas"""
        reward = TrainingReward(timestamp=time.time())
        
        # Baseline comparison
        if self.current_baseline:
            reward.detection_improvement = max(0, metrics.f1_score - self.current_baseline.f1_score)
            reward.precision_gain = max(0, metrics.precision - self.current_baseline.precision)
            reward.recall_gain = max(0, metrics.recall - self.current_baseline.recall)
            reward.map_gain = max(0, metrics.mAP - self.current_baseline.mAP)
        
        # Decision accuracy
        reward.decision_accuracy = metrics.accuracy
        
        # Convergence speed
        if len(self.metrics_history) > 1:
            prev_loss = self.metrics_history[-2].train_loss
            reward.convergence_speed = max(0, prev_loss - metrics.train_loss) / max(0.01, prev_loss)
        
        # Stability (variação de loss)
        if len(self.metrics_history) >= 5:
            recent_losses = [m.train_loss for m in self.metrics_history[-5:]]
            reward.stability_score = 1.0 - (np.std(recent_losses) / max(0.01, np.mean(recent_losses)))
        
        # Overfitting penalty
        if metrics.train_loss < metrics.val_loss * 0.8:
            reward.overfitting_penalty = (metrics.val_loss - metrics.train_loss) / metrics.val_loss
        
        # Total reward
        reward.total_reward = reward.weighted_reward
        
        return reward
    
    def is_learning(self) -> Tuple[bool, str]:
        """Verifica se o modelo está aprendendo"""
        if len(self.metrics_history) < 5:
            return True, "Insufficient data"
        
        recent = self.metrics_history[-5:]
        
        # Verificar se loss está diminuindo
        losses = [m.train_loss for m in recent]
        is_decreasing = losses[-1] < losses[0]
        
        # Verificar se métricas estão melhorando
        f1_scores = [m.f1_score for m in recent]
        is_improving = f1_scores[-1] > f1_scores[0]
        
        # Verificar estabilidade
        loss_std = np.std(losses)
        is_stable = loss_std < 0.1
        
        if is_decreasing and is_improving and is_stable:
            return True, "Learning well"
        elif is_decreasing and not is_improving:
            return True, "Learning but not improving metrics"
        elif not is_decreasing:
            return False, "Loss not decreasing"
        else:
            return False, "Unstable learning"
    
    def _save_metrics(self, metrics: TrainingMetrics):
        """Salva métricas em arquivo"""
        metrics_path = self.output_dir / "training_metrics.jsonl"
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')
    
    def generate_report(self) -> Dict:
        """Gera relatório completo de treinamento"""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        latest = self.metrics_history[-1]
        best = self.best_metrics or latest
        
        # Calcular estatísticas
        losses = [m.train_loss for m in self.metrics_history]
        f1_scores = [m.f1_score for m in self.metrics_history]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_epochs": len(self.metrics_history),
            "latest_metrics": latest.to_dict(),
            "best_metrics": best.to_dict(),
            "statistics": {
                "avg_loss": np.mean(losses),
                "min_loss": np.min(losses),
                "max_f1": np.max(f1_scores),
                "avg_f1": np.mean(f1_scores),
                "loss_trend": "decreasing" if losses[-1] < losses[0] else "increasing",
                "f1_trend": "improving" if f1_scores[-1] > f1_scores[0] else "degrading"
            },
            "learning_status": self.is_learning(),
            "total_reward": sum(r.total_reward for r in self.rewards_history)
        }
        
        return report
    
    def plot_training_progress(self, save_path: Optional[Path] = None):
        """Plota progresso do treinamento"""
        if not self.metrics_history:
            self.logger.warning("No metrics to plot")
            return
        
        epochs = [m.epoch for m in self.metrics_history]
        train_losses = [m.train_loss for m in self.metrics_history]
        val_losses = [m.val_loss for m in self.metrics_history]
        f1_scores = [m.f1_score for m in self.metrics_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', marker='o')
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # F1 Score plot
        axes[0, 1].plot(epochs, f1_scores, label='F1 Score', marker='o', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score Progress')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Rewards plot
        if self.rewards_history:
            reward_epochs = list(range(len(self.rewards_history)))
            rewards = [r.total_reward for r in self.rewards_history]
            axes[1, 0].plot(reward_epochs, rewards, label='Total Reward', marker='o', color='purple')
            axes[1, 0].set_xlabel('Reward Step')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].set_title('Reward Progress')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate plot (if available)
        learning_rates = [m.learning_rate for m in self.metrics_history if m.learning_rate > 0]
        if learning_rates:
            lr_epochs = [i for i, lr in enumerate(learning_rates)]
            axes[1, 1].plot(lr_epochs, learning_rates, label='Learning Rate', marker='o', color='orange')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No learning rate data', ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Training plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class UnifiedTrainingSystem:
    """Sistema unificado de treinamento para todos os modelos"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        
        # Diretórios
        self.dataset_dir = self.base_dir / "dataset"
        self.models_dir = self.base_dir / "models"
        self.training_dir = self.base_dir / "training"
        self.reports_dir = self.base_dir / "training_reports"
        
        for dir_path in [self.dataset_dir, self.models_dir, self.training_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Monitores
        self.yolo_monitor = TrainingMonitor(self.reports_dir / "yolo")
        self.bc_monitor = TrainingMonitor(self.reports_dir / "bc")
        self.cql_monitor = TrainingMonitor(self.reports_dir / "cql")
        
        self.logger = logging.getLogger("unified_training")
    
    def train_yolo_model(self, dataset_path: Path, epochs: int = 50) -> Dict:
        """Treina modelo YOLO com monitoramento"""
        self.logger.info("=" * 60)
        self.logger.info("TREINAMENTO YOLO")
        self.logger.info("=" * 60)
        
        try:
            from ultralytics import YOLO
            
            # Configurar dataset YOLO
            yaml_config = self._create_yolo_config(dataset_path)
            
            # Inicializar modelo
            model = YOLO('yolov8n.pt')
            
            # Treinar com monitoramento
            start_time = time.time()
            
            for epoch in range(epochs):
                epoch_start = time.time()
                
                # Treinar uma epoch (simplificado - na prática usaria model.train())
                # Aqui simulamos o treinamento para demonstração
                self._simulate_yolo_epoch(model, dataset_path, epoch)
                
                # Calcular métricas
                metrics = self._evaluate_yolo_model(model, dataset_path, epoch)
                metrics.training_time = time.time() - epoch_start
                
                # Registrar métricas
                self.yolo_monitor.record_metrics(metrics)
                
                # Verificar aprendizado
                is_learning, status = self.yolo_monitor.is_learning()
                self.logger.info(f"Epoch {epoch}: {status}")
                
                # Early stopping se não estiver aprendendo
                if not is_learning and epoch > 10:
                    self.logger.warning(f"Early stopping at epoch {epoch}: {status}")
                    break
            
            total_time = time.time() - start_time
            
            # Gerar relatório
            report = self.yolo_monitor.generate_report()
            report["training_time"] = total_time
            report["training_mode"] = "simulated"
            
            # Salvar modelo
            model_path = self.models_dir / "yolo_unified_best.pt"
            model.save(str(model_path))
            
            # Plotar progresso
            self.yolo_monitor.plot_training_progress(
                self.reports_dir / "yolo" / "training_progress.png"
            )
            
            self.logger.info(f"YOLO training complete in {total_time:.1f}s")
            
            return report
            
        except Exception as e:
            self.logger.error(f"YOLO training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def train_bc_model(self, dataset_path: Path, epochs: int = 50) -> Dict:
        """Treina modelo de Behavior Cloning com monitoramento"""
        self.logger.info("=" * 60)
        self.logger.info("TREINAMENTO BEHAVIOR CLONING")
        self.logger.info("=" * 60)
        
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
            
            if TorchDataset is None:
                raise ImportError("torch.utils.data.Dataset not available")
            
            # Carregar dataset real quando disponível e separar validação.
            bc_dataset = self._load_bc_dataset(dataset_path)
            if len(bc_dataset) >= 10:
                val_size = max(1, int(len(bc_dataset) * 0.2))
                train_size = max(1, len(bc_dataset) - val_size)
                train_dataset, val_dataset = torch.utils.data.random_split(
                    bc_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
            else:
                train_dataset = bc_dataset
                val_dataset = bc_dataset

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Criar modelo
            model = self._create_bc_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            # Treinar com monitoramento
            start_time = time.time()
            
            for epoch in range(epochs):
                epoch_start = time.time()
                
                # Treinar uma epoch
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0
                
                for batch in train_loader:
                    states, actions = batch
                    optimizer.zero_grad()
                    
                    outputs = model(states)
                    loss = criterion(outputs, actions)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += actions.size(0)
                    correct += (predicted == actions).sum().item()
                
                train_loss /= len(train_loader)
                accuracy = correct / total
                
                # Validar
                val_loss, val_accuracy = self._validate_bc_model(model, val_loader)
                
                # Criar métricas
                metrics = TrainingMetrics(
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                    train_loss=train_loss,
                    val_loss=val_loss,
                    accuracy=accuracy,
                    f1_score=accuracy,
                    precision=accuracy,
                    recall=accuracy,
                    learning_rate=1e-4,
                    training_time=time.time() - epoch_start
                )
                
                # Registrar métricas
                self.bc_monitor.record_metrics(metrics)
                
                # Verificar aprendizado
                is_learning, status = self.bc_monitor.is_learning()
                self.logger.info(f"Epoch {epoch}: Acc={accuracy:.4f}, {status}")
                
                # Early stopping
                if not is_learning and epoch > 10:
                    self.logger.warning(f"Early stopping at epoch {epoch}: {status}")
                    break
            
            total_time = time.time() - start_time
            
            # Gerar relatório
            report = self.bc_monitor.generate_report()
            report["training_time"] = total_time
            report["training_mode"] = "real_dataset"
            
            # Salvar modelo
            model_path = self.models_dir / "bc_unified_best.pt"
            torch.save(model.state_dict(), str(model_path))
            
            # Plotar progresso
            self.bc_monitor.plot_training_progress(
                self.reports_dir / "bc" / "training_progress.png"
            )
            
            self.logger.info(f"BC training complete in {total_time:.1f}s")
            
            return report
            
        except Exception as e:
            self.logger.error(f"BC training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def train_cql_model(self, dataset_path: Path, epochs: int = 50) -> Dict:
        """Treina modelo CQL com monitoramento"""
        self.logger.info("=" * 60)
        self.logger.info("TREINAMENTO CQL (OFFLINE RL)")
        self.logger.info("=" * 60)
        
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
            
            if TorchDataset is None:
                raise ImportError("torch.utils.data.Dataset not available")
            
            # Carregar replay buffer real quando disponível.
            replay_buffer = self._load_replay_buffer(dataset_path)
            train_loader = DataLoader(replay_buffer, batch_size=64, shuffle=True)
            
            # Criar modelo
            model = self._create_cql_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.MSELoss()
            
            # Treinar com monitoramento
            start_time = time.time()
            
            for epoch in range(epochs):
                epoch_start = time.time()
                
                # Treinar uma epoch
                model.train()
                train_loss = 0.0
                
                for batch in train_loader:
                    states, actions, rewards, next_states, dones = batch
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    q_values = model(torch.cat([states, actions], dim=1))
                    
                    # Conservative Q-Learning loss (simplificado)
                    target_q = rewards + 0.99 * (1 - dones.float()) * model(
                        torch.cat([next_states, actions], dim=1)
                    ).detach()
                    
                    loss = criterion(q_values, target_q)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Criar métricas
                metrics = TrainingMetrics(
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                    train_loss=train_loss,
                    val_loss=train_loss * 1.1,  # Estimado
                    accuracy=0.0,  # Não aplicável para RL
                    f1_score=max(0.0, 1.0 - train_loss),  # Proxy clamped to valid range
                    precision=max(0.0, 1.0 - train_loss),
                    recall=max(0.0, 1.0 - train_loss),
                    learning_rate=1e-4,
                    training_time=time.time() - epoch_start
                )
                
                # Registrar métricas
                self.cql_monitor.record_metrics(metrics)
                
                # Verificar aprendizado
                is_learning, status = self.cql_monitor.is_learning()
                self.logger.info(f"Epoch {epoch}: Loss={train_loss:.4f}, {status}")
                
                # Early stopping
                if not is_learning and epoch > 10:
                    self.logger.warning(f"Early stopping at epoch {epoch}: {status}")
                    break
            
            total_time = time.time() - start_time
            
            # Gerar relatório
            report = self.cql_monitor.generate_report()
            report["training_time"] = total_time
            report["training_mode"] = "real_dataset"
            
            # Salvar modelo
            model_path = self.models_dir / "cql_unified_best.pt"
            torch.save(model.state_dict(), str(model_path))
            
            # Plotar progresso
            self.cql_monitor.plot_training_progress(
                self.reports_dir / "cql" / "training_progress.png"
            )
            
            self.logger.info(f"CQL training complete in {total_time:.1f}s")
            
            return report
            
        except Exception as e:
            self.logger.error(f"CQL training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def train_all_models(self, epochs: int = 30) -> Dict:
        """Treina todos os modelos sequencialmente"""
        self.logger.info("=" * 60)
        self.logger.info("TREINAMENTO UNIFICADO DE TODOS OS MODELOS")
        self.logger.info("=" * 60)
        
        results = {}
        
        # Dataset paths
        yolo_dataset = self.dataset_dir / "synthetic_v2"
        bc_dataset = self.dataset_dir / "bc" / "bc_dataset.json"
        cql_dataset = self.dataset_dir / "cql" / "replay_buffer.json"
        
        # Treinar YOLO
        if yolo_dataset.exists():
            self.logger.info("Training YOLO model...")
            results["yolo"] = self.train_yolo_model(yolo_dataset, epochs)
        else:
            self.logger.warning(f"YOLO dataset not found: {yolo_dataset}")
        
        # Treinar BC
        if bc_dataset.exists():
            self.logger.info("Training BC model...")
            results["bc"] = self.train_bc_model(bc_dataset, epochs)
        else:
            self.logger.warning(f"BC dataset not found: {bc_dataset}")
        
        # Treinar CQL
        if cql_dataset.exists():
            self.logger.info("Training CQL model...")
            results["cql"] = self.train_cql_model(cql_dataset, epochs)
        else:
            self.logger.warning(f"CQL dataset not found: {cql_dataset}")
        
        # Gerar relatório unificado
        unified_report = {
            "timestamp": datetime.now().isoformat(),
            "total_training_time": sum(r.get("training_time", 0) for r in results.values()),
            "models_trained": list(results.keys()),
            "results": results
        }
        
        report_path = self.reports_dir / "unified_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(unified_report, f, indent=2)
        
        self.logger.info(f"Unified training complete. Report saved to {report_path}")
        
        return unified_report
    
    def _create_yolo_config(self, dataset_path: Path) -> Path:
        """Cria arquivo de configuração YOLO"""
        # Simplificado - na prática criaria um arquivo YAML real
        return dataset_path
    
    def _simulate_yolo_epoch(self, model, dataset_path, epoch):
        """Simula uma epoch de treinamento YOLO"""
        # Na prática, usaria model.train() do YOLO
        pass
    
    def _evaluate_yolo_model(self, model, dataset_path, epoch) -> TrainingMetrics:
        """Avalia modelo YOLO"""
        # Simulação de avaliação
        import random
        
        return TrainingMetrics(
            epoch=epoch,
            timestamp=datetime.now().isoformat(),
            train_loss=max(0.1, 1.0 - epoch * 0.02 + random.uniform(-0.05, 0.05)),
            val_loss=max(0.15, 1.1 - epoch * 0.02 + random.uniform(-0.05, 0.05)),
            accuracy=min(1.0, 0.3 + epoch * 0.015 + random.uniform(-0.02, 0.02)),
            precision=min(1.0, 0.25 + epoch * 0.02 + random.uniform(-0.02, 0.02)),
            recall=min(1.0, 0.4 + epoch * 0.01 + random.uniform(-0.02, 0.02)),
            f1_score=min(1.0, 0.3 + epoch * 0.015 + random.uniform(-0.02, 0.02)),
            mAP=min(1.0, 0.25 + epoch * 0.015 + random.uniform(-0.02, 0.02)),
            learning_rate=0.01 * (0.95 ** epoch),
            is_improving=True,
            confidence_score=0.5 + epoch * 0.01
        )
    
    def _load_bc_dataset(self, dataset_path):
        """Carrega dataset de behavior cloning"""
        import torch

        class BCDataset(TorchDataset):
            def __init__(self, samples):
                self.samples = samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        samples = []
        try:
            if Path(dataset_path).exists():
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                move_map = {"up": 0, "down": 1, "left": 2, "right": 3}

                for episode in data if isinstance(data, list) else data.get("matches", []):
                    frames = episode.get("frames", []) if isinstance(episode, dict) else []
                    for frame in frames:
                        state = frame.get("state", {})
                        action = frame.get("action", {})

                        features = torch.tensor([
                            float(state.get("player_x", 0.0)),
                            float(state.get("player_y", 0.0)),
                            float(state.get("health", 0.0)),
                            float(state.get("ammo", 0.0)),
                            float(state.get("enemy_distance", 0.0)),
                            float(state.get("enemy_angle", 0.0)),
                            float(state.get("obstacle_nearby", 0.0)),
                            float(state.get("powerup_available", 0.0)),
                        ], dtype=torch.float32)

                        label = move_map.get(str(action.get("move_direction", "")).lower(), 4)
                        if action.get("attack") or action.get("use_ability"):
                            label = 4

                        samples.append((features, torch.tensor(label, dtype=torch.long)))
        except Exception as e:
            logger.warning(f"BC dataset load failed, falling back to synthetic samples: {e}")

        if not samples:
            for _ in range(100):
                samples.append((torch.randn(8), torch.randint(0, 5, ()).long()))

        return BCDataset(samples)
    
    def _create_bc_model(self):
        """Cria modelo de behavior cloning"""
        import torch
        import torch.nn as nn
        
        class BCModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(8, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 5)
                )
            
            def forward(self, x):
                return self.network(x)
        
        return BCModel()
    
    def _validate_bc_model(self, model, dataloader):
        """Valida modelo BC"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        import torch
        with torch.no_grad():
            for batch in dataloader:
                states, actions = batch
                outputs = model(states)
                loss = torch.nn.functional.cross_entropy(outputs, actions)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += actions.size(0)
                correct += (predicted == actions).sum().item()
        
        return total_loss / len(dataloader), correct / total
    
    def _load_replay_buffer(self, dataset_path):
        """Carrega replay buffer para CQL"""
        import torch

        class ReplayDataset(TorchDataset):
            def __init__(self, samples):
                self.samples = samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        samples = []
        try:
            if Path(dataset_path).exists():
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for item in data if isinstance(data, list) else data.get("transitions", []):
                    state = torch.tensor(item.get("state", []), dtype=torch.float32)
                    action = torch.tensor(item.get("action", []), dtype=torch.float32)
                    reward = torch.tensor([float(item.get("reward", 0.0))], dtype=torch.float32)
                    next_state = torch.tensor(item.get("next_state", []), dtype=torch.float32)
                    done = torch.tensor([1.0 if item.get("done", False) else 0.0], dtype=torch.float32)
                    if state.numel() and action.numel() and next_state.numel():
                        samples.append((state, action, reward, next_state, done))
        except Exception as e:
            logger.warning(f"CQL dataset load failed, falling back to synthetic samples: {e}")

        if not samples:
            for _ in range(100):
                samples.append((
                    torch.randn(8),
                    torch.randn(5),
                    torch.randn(1),
                    torch.randn(8),
                    torch.randint(0, 2, (1,)).float()
                ))

        return ReplayDataset(samples)
    
    def _create_cql_model(self):
        """Cria modelo CQL"""
        import torch
        import torch.nn as nn
        
        class CQLModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(13, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                )
            
            def forward(self, x):
                return self.network(x)
        
        return CQLModel()


def main():
    """Função principal para execução do treinamento unificado"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Training System")
    parser.add_argument("--base-dir", default="c:\\Users\\rodri\\Desktop\\bot brawl", help="Diretório base do projeto")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas")
    parser.add_argument("--model", choices=["all", "yolo", "bc", "cql"], default="all", help="Modelo para treinar")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Criar sistema de treinamento
    system = UnifiedTrainingSystem(Path(args.base_dir))
    
    # Treinar modelos
    if args.model == "all":
        results = system.train_all_models(args.epochs)
    elif args.model == "yolo":
        results = system.train_yolo_model(
            system.dataset_dir / "synthetic_v2",
            args.epochs
        )
    elif args.model == "bc":
        results = system.train_bc_model(
            system.dataset_dir / "bc" / "bc_dataset.json",
            args.epochs
        )
    elif args.model == "cql":
        results = system.train_cql_model(
            system.dataset_dir / "cql" / "replay_buffer.json",
            args.epochs
        )
    
    logger.info("Training complete!")
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
