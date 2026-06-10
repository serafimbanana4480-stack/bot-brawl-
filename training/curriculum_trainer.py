"""
training/curriculum_trainer.py

Curriculum Learning trainer para Soberana Omega.

Pipeline de 3 fases:
  1. BC (Behavioral Cloning)      — imitação de dados humanos/expert
  2. CQL (Conservative Q-Learning) — Q-learning com regularização conservadora
  3. PPO (Proximal Policy Optimization) — fine-tuning com policy gradient

Cada fase tem critérios de transição configuráveis e guarda checkpoints
em models/curriculum/phase_{1,2,3}/.

Usage:
    from training.curriculum_trainer import CurriculumTrainer
    trainer = CurriculumTrainer(policy, config)
    trainer.run(expert_data, experience_buffer)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment, misc]
    nn = None  # type: ignore[assignment, misc]
    optim = None  # type: ignore[assignment, misc]
    Categorical = None  # type: ignore[assignment, misc]

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None  # type: ignore[assignment, misc]


# ------------------------------------------------------------------
# Phase configuration
# ------------------------------------------------------------------

@dataclass
class PhaseConfig:
    """Configuration for a single curriculum phase."""
    name: str
    max_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 3e-4
    # Transition criteria
    loss_threshold: Optional[float] = None
    win_rate_threshold: Optional[float] = None
    min_epochs: int = 10
    # CQL-specific
    cql_alpha: float = 1.0  # Conservative penalty weight
    # Checkpointing
    checkpoint_dir: Path = Path("models/curriculum")


# ------------------------------------------------------------------
# Curriculum Trainer
# ------------------------------------------------------------------

class CurriculumTrainer:
    """
    Orquestra treino em 3 fases: BC -> CQL -> PPO.

    Args:
        policy: NeuralPolicy (torch.nn.Module)
        config: Dict com configurações de cada fase e MLflow
    """

    def __init__(
        self,
        policy: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise RuntimeError("CurriculumTrainer requer PyTorch")

        self.policy = policy
        self.device = device
        self.config = config or {}
        self.current_phase: int = 0  # 0=BC, 1=CQL, 2=PPO
        self.phase_history: List[Dict[str, Any]] = []

        # MLflow
        self.mlflow_enabled = self.config.get("mlflow", {}).get("enabled", False) and HAS_MLFLOW
        self.mlflow_experiment = self.config.get("mlflow", {}).get("experiment_name", "soberana_omega_curriculum")
        if self.mlflow_enabled:
            uri = self.config.get("mlflow", {}).get("tracking_uri")
            if uri:
                mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(self.mlflow_experiment)

        # Phase configs
        self.phase_configs = [
            PhaseConfig(
                name="bc",
                max_epochs=self.config.get("bc", {}).get("max_epochs", 100),
                batch_size=self.config.get("bc", {}).get("batch_size", 64),
                learning_rate=self.config.get("bc", {}).get("learning_rate", 3e-4),
                loss_threshold=self.config.get("bc", {}).get("loss_threshold", 0.1),
                min_epochs=self.config.get("bc", {}).get("min_epochs", 10),
                checkpoint_dir=Path(self.config.get("bc", {}).get("checkpoint_dir", "models/curriculum/phase_1")),
            ),
            PhaseConfig(
                name="cql",
                max_epochs=self.config.get("cql", {}).get("max_epochs", 200),
                batch_size=self.config.get("cql", {}).get("batch_size", 64),
                learning_rate=self.config.get("cql", {}).get("learning_rate", 3e-4),
                loss_threshold=self.config.get("cql", {}).get("loss_threshold", 0.05),
                win_rate_threshold=self.config.get("cql", {}).get("win_rate_threshold", 0.3),
                min_epochs=self.config.get("cql", {}).get("min_epochs", 20),
                cql_alpha=self.config.get("cql", {}).get("cql_alpha", 1.0),
                checkpoint_dir=Path(self.config.get("cql", {}).get("checkpoint_dir", "models/curriculum/phase_2")),
            ),
            PhaseConfig(
                name="ppo",
                max_epochs=self.config.get("ppo", {}).get("max_epochs", 500),
                batch_size=self.config.get("ppo", {}).get("batch_size", 64),
                learning_rate=self.config.get("ppo", {}).get("learning_rate", 3e-4),
                win_rate_threshold=self.config.get("ppo", {}).get("win_rate_threshold", 0.6),
                min_epochs=self.config.get("ppo", {}).get("min_epochs", 50),
                checkpoint_dir=Path(self.config.get("ppo", {}).get("checkpoint_dir", "models/curriculum/phase_3")),
            ),
        ]

        # Metrics per phase
        self.phase_metrics: Dict[int, Dict[str, List[float]]] = {
            0: {"loss": [], "accuracy": []},
            1: {"loss": [], "q_loss": [], "cql_loss": [], "win_rate": []},
            2: {"policy_loss": [], "value_loss": [], "entropy": [], "win_rate": []},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        expert_data: Optional[List[Tuple[np.ndarray, int]]] = None,
        experience_buffer: Optional[Any] = None,
        win_rate_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Executa o curriculum completo.

        Args:
            expert_data: Lista de (state_vector, action_idx) para BC
            experience_buffer: ExperienceBuffer para CQL/PPO
            win_rate_fn: Callable que retorna win rate atual (float)

        Returns:
            Dict com histórico e métricas finais
        """
        if self.mlflow_enabled:
            mlflow.start_run(run_name=f"curriculum_{time.strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_param("device", self.device)

        logger.info("[CURRICULUM] Iniciando pipeline BC -> CQL -> PPO")

        # Phase 1: BC
        self._run_phase_1_bc(expert_data)

        # Phase 2: CQL
        self._run_phase_2_cql(experience_buffer, win_rate_fn)

        # Phase 3: PPO
        self._run_phase_3_ppo(experience_buffer, win_rate_fn)

        if self.mlflow_enabled:
            mlflow.end_run()

        return {
            "phases_completed": self.current_phase + 1,
            "phase_history": self.phase_history,
            "final_metrics": self.phase_metrics,
        }

    # ------------------------------------------------------------------
    # Phase 1: Behavioral Cloning
    # ------------------------------------------------------------------

    def _run_phase_1_bc(
        self,
        expert_data: Optional[List[Tuple[np.ndarray, int]]] = None,
    ) -> None:
        """Fase 1: Behavioral Cloning com dados humanos/expert."""
        cfg = self.phase_configs[0]
        logger.info(f"[CURRICULUM] Fase 1 BC iniciada (max_epochs={cfg.max_epochs})")

        if expert_data is None or len(expert_data) == 0:
            logger.warning("[CURRICULUM] Fase 1: sem dados expert — pulando BC")
            self.phase_history.append({"phase": "bc", "status": "skipped", "reason": "no_expert_data"})
            return

        self.policy.to(self.device)
        optimizer = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        epochs_without_improvement = 0
        patience = 20

        for epoch in range(cfg.max_epochs):
            # Shuffle data
            np.random.shuffle(expert_data)
            total_loss = 0.0
            correct = 0
            total = 0

            for i in range(0, len(expert_data), cfg.batch_size):
                batch = expert_data[i : i + cfg.batch_size]
                states = torch.FloatTensor(np.stack([s for s, _ in batch])).to(self.device)
                actions = torch.LongTensor([a for _, a in batch]).to(self.device)

                # Forward
                policy_logits, _ = self.policy(states)
                loss = criterion(policy_logits, actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch)
                correct += (policy_logits.argmax(dim=-1) == actions).sum().item()
                total += len(batch)

            avg_loss = total_loss / total
            accuracy = correct / total
            self.phase_metrics[0]["loss"].append(avg_loss)
            self.phase_metrics[0]["accuracy"].append(accuracy)

            if self.mlflow_enabled:
                mlflow.log_metrics({"bc_loss": avg_loss, "bc_accuracy": accuracy}, step=epoch)

            logger.info(f"[CURRICULUM] BC epoch {epoch}: loss={avg_loss:.4f}, acc={accuracy:.3f}")

            # Early stopping / transition criteria
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_without_improvement = 0
                self._save_checkpoint(cfg.checkpoint_dir, f"bc_best.pt")
            else:
                epochs_without_improvement += 1

            if epoch >= cfg.min_epochs and cfg.loss_threshold is not None and avg_loss < cfg.loss_threshold:
                logger.info(f"[CURRICULUM] BC convergiu (loss < {cfg.loss_threshold})")
                break

            if epochs_without_improvement >= patience:
                logger.info("[CURRICULUM] BC early stopping (patience excedido)")
                break

        self._save_checkpoint(cfg.checkpoint_dir, "bc_final.pt")
        self.phase_history.append({
            "phase": "bc",
            "status": "completed",
            "epochs": epoch + 1,
            "final_loss": avg_loss,
            "final_accuracy": accuracy,
        })
        self.current_phase = 1

    # ------------------------------------------------------------------
    # Phase 2: Conservative Q-Learning
    # ------------------------------------------------------------------

    def _run_phase_2_cql(
        self,
        experience_buffer: Optional[Any] = None,
        win_rate_fn: Optional[Any] = None,
    ) -> None:
        """Fase 2: CQL — Q-learning com regularização conservadora."""
        cfg = self.phase_configs[1]
        logger.info(f"[CURRICULUM] Fase 2 CQL iniciada (max_epochs={cfg.max_epochs})")

        if experience_buffer is None or len(experience_buffer) < cfg.batch_size:
            logger.warning("[CURRICULUM] Fase 2: buffer insuficiente — pulando CQL")
            self.phase_history.append({"phase": "cql", "status": "skipped", "reason": "insufficient_buffer"})
            return

        self.policy.to(self.device)
        optimizer = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)

        # Q-network head (reuse policy logits as Q-values for discrete actions)
        # We treat policy_logits as Q-values for simplicity
        gamma = 0.99

        for epoch in range(cfg.max_epochs):
            batch = experience_buffer.sample(cfg.batch_size)
            if batch is None:
                break

            states = torch.FloatTensor(batch["states"]).to(self.device)
            actions = torch.LongTensor(batch["actions"]).to(self.device)
            rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
            next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
            dones = torch.FloatTensor(batch["dones"]).to(self.device)

            # Current Q-values
            q_logits, _ = self.policy(states)
            q_values = q_logits.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target Q-values (Bellman)
            with torch.no_grad():
                next_q_logits, _ = self.policy(next_states)
                next_q_values = next_q_logits.max(dim=1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

            # Standard Q-loss
            q_loss = nn.MSELoss()(q_values, targets)

            # CQL conservative penalty: logsumexp of all Q-values minus dataset Q-values
            cql_penalty = torch.logsumexp(q_logits, dim=1).mean() - q_values.mean()
            loss = q_loss + cfg.cql_alpha * cql_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.phase_metrics[1]["loss"].append(loss.item())
            self.phase_metrics[1]["q_loss"].append(q_loss.item())
            self.phase_metrics[1]["cql_loss"].append(cql_penalty.item())

            if self.mlflow_enabled:
                mlflow.log_metrics({
                    "cql_loss": loss.item(),
                    "cql_q_loss": q_loss.item(),
                    "cql_penalty": cql_penalty.item(),
                }, step=epoch)

            if epoch % 10 == 0:
                logger.info(f"[CURRICULUM] CQL epoch {epoch}: loss={loss:.4f}, q_loss={q_loss:.4f}")

            # Transition criteria
            if epoch >= cfg.min_epochs:
                if cfg.loss_threshold is not None and q_loss.item() < cfg.loss_threshold:
                    logger.info(f"[CURRICULUM] CQL convergiu (q_loss < {cfg.loss_threshold})")
                    break
                if win_rate_fn is not None and cfg.win_rate_threshold is not None:
                    wr = win_rate_fn()
                    self.phase_metrics[1]["win_rate"].append(wr)
                    if wr >= cfg.win_rate_threshold:
                        logger.info(f"[CURRICULUM] CQL win rate threshold atingido ({wr:.2f})")
                        break

        self._save_checkpoint(cfg.checkpoint_dir, "cql_final.pt")
        self.phase_history.append({
            "phase": "cql",
            "status": "completed",
            "epochs": epoch + 1,
            "final_loss": loss.item() if 'loss' in dir() else None,
        })
        self.current_phase = 2

    # ------------------------------------------------------------------
    # Phase 3: PPO Fine-tuning
    # ------------------------------------------------------------------

    def _run_phase_3_ppo(
        self,
        experience_buffer: Optional[Any] = None,
        win_rate_fn: Optional[Any] = None,
    ) -> None:
        """Fase 3: PPO fine-tuning com policy gradient."""
        cfg = self.phase_configs[2]
        logger.info(f"[CURRICULUM] Fase 3 PPO iniciada (max_epochs={cfg.max_epochs})")

        if experience_buffer is None or len(experience_buffer) < cfg.batch_size:
            logger.warning("[CURRICULUM] Fase 3: buffer insuficiente — pulando PPO")
            self.phase_history.append({"phase": "ppo", "status": "skipped", "reason": "insufficient_buffer"})
            return

        from training.ppo_trainer import PPOTrainer

        # Initialize PPO trainer with MLflow if enabled globally
        mlflow_cfg = self.config.get("mlflow", {})
        trainer = PPOTrainer(
            self.policy,
            learning_rate=cfg.learning_rate,
            mlflow_enabled=self.mlflow_enabled,
            mlflow_tracking_uri=mlflow_cfg.get("tracking_uri"),
            mlflow_experiment_name=mlflow_cfg.get("experiment_name", "soberana_omega_ppo"),
        )

        # Run PPO training
        num_updates = cfg.max_epochs
        checkpoint_path = str(cfg.checkpoint_dir / "ppo_final.pt")
        stats = trainer.train(
            experience_buffer,
            num_updates=num_updates,
            batch_size=cfg.batch_size,
            ppo_epochs=4,
            checkpoint_path=checkpoint_path,
            run_name="curriculum_phase3_ppo",
        )

        # Store metrics
        self.phase_metrics[2]["policy_loss"] = stats.get("policy_loss", [])
        self.phase_metrics[2]["value_loss"] = stats.get("value_loss", [])
        self.phase_metrics[2]["entropy"] = stats.get("entropy", [])

        self.phase_history.append({
            "phase": "ppo",
            "status": "completed",
            "updates": len(stats.get("policy_loss", [])),
            "final_policy_loss": stats["policy_loss"][-1] if stats.get("policy_loss") else None,
            "final_value_loss": stats["value_loss"][-1] if stats.get("value_loss") else None,
        })
        self.current_phase = 3

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, checkpoint_dir: Path, filename: str) -> None:
        """Save policy checkpoint."""
        if self.policy is None:
            return
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / filename
        try:
            torch.save(self.policy.state_dict(), path)
            logger.info(f"[CURRICULUM] Checkpoint salvo: {path}")
            if self.mlflow_enabled and HAS_MLFLOW:
                mlflow.log_artifact(str(path))
        except Exception as e:
            logger.warning(f"[CURRICULUM] Falha ao salvar checkpoint: {e}")

    def get_progress(self) -> Dict[str, Any]:
        """Retorna progresso atual do curriculum."""
        return {
            "current_phase": self.current_phase,
            "phase_names": ["bc", "cql", "ppo"],
            "phase_history": self.phase_history,
            "metrics": self.phase_metrics,
        }

    def save_report(self, path: Path = Path("training_reports/curriculum_report.json")) -> None:
        """Salva relatório JSON do curriculum."""
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": time.time(),
            "config": self.config,
            "progress": self.get_progress(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"[CURRICULUM] Relatório salvo em {path}")
