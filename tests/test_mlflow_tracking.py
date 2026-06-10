"""
tests/test_mlflow_tracking.py

Testes para MLflow tracking no PPOTrainer e CurriculumTrainer.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestPPOTrainerMLflow:
    """Testes para MLflow tracking no PPOTrainer."""

    @pytest.mark.skipif(not __import__("torch", __import__("sys").modules[__name__].__dict__), reason="torch not available")
    def test_mlflow_disabled_by_default(self):
        """Por padrão MLflow deve estar desabilitado."""
        import torch
        from training.ppo_trainer import PPOTrainer

        policy = MagicMock()
        policy.parameters = MagicMock(return_value=[torch.randn(2, 2, requires_grad=True)])
        trainer = PPOTrainer(policy)
        assert trainer.mlflow_enabled is False

    @pytest.mark.skipif(not __import__("torch", __import__("sys").modules[__name__].__dict__), reason="torch not available")
    def test_mlflow_start_run(self):
        """Deve iniciar run MLflow quando habilitado."""
        import torch
        from training.ppo_trainer import PPOTrainer

        policy = MagicMock()
        policy.parameters = MagicMock(return_value=[torch.randn(2, 2, requires_grad=True)])

        with patch("training.ppo_trainer.HAS_MLFLOW", True):
            with patch("training.ppo_trainer.mlflow") as mock_mlflow:
                trainer = PPOTrainer(
                    policy,
                    mlflow_enabled=True,
                    mlflow_tracking_uri="runs/mlflow/test",
                    mlflow_experiment_name="test_exp",
                )
                trainer._start_mlflow_run(run_name="test_run")
                assert trainer._mlflow_run_active is True
                mock_mlflow.start_run.assert_called_once_with(run_name="test_run")
                mock_mlflow.log_params.assert_called_once()

    @pytest.mark.skipif(not __import__("torch", __import__("sys").modules[__name__].__dict__), reason="torch not available")
    def test_mlflow_log_metrics(self):
        """Deve logar métricas no MLflow."""
        import torch
        from training.ppo_trainer import PPOTrainer

        policy = MagicMock()
        policy.parameters = MagicMock(return_value=[torch.randn(2, 2, requires_grad=True)])

        with patch("training.ppo_trainer.HAS_MLFLOW", True):
            with patch("training.ppo_trainer.mlflow") as mock_mlflow:
                trainer = PPOTrainer(policy, mlflow_enabled=True)
                trainer._mlflow_run_active = True
                metrics = {"policy_loss": 0.5, "value_loss": 0.3, "entropy": 0.1}
                trainer._log_mlflow_metrics(metrics, step=5)
                mock_mlflow.log_metrics.assert_called_once_with(metrics, step=5)

    @pytest.mark.skipif(not __import__("torch", __import__("sys").modules[__name__].__dict__), reason="torch not available")
    def test_mlflow_end_run_with_artifact(self):
        """Deve terminar run e logar checkpoint como artifact."""
        import torch
        from training.ppo_trainer import PPOTrainer

        policy = MagicMock()
        policy.parameters = MagicMock(return_value=[torch.randn(2, 2, requires_grad=True)])

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(b"fake_checkpoint")
            tmp_path = tmp.name

        with patch("training.ppo_trainer.HAS_MLFLOW", True):
            with patch("training.ppo_trainer.mlflow") as mock_mlflow:
                trainer = PPOTrainer(policy, mlflow_enabled=True)
                trainer._mlflow_run_active = True
                trainer._end_mlflow_run(checkpoint_path=tmp_path)
                mock_mlflow.log_artifact.assert_called_once_with(tmp_path)
                mock_mlflow.end_run.assert_called_once()
                assert trainer._mlflow_run_active is False

        Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.skipif(not __import__("torch", __import__("sys").modules[__name__].__dict__), reason="torch not available")
    def test_mlflow_graceful_without_mlflow(self):
        """Deve funcionar normalmente mesmo sem mlflow instalado."""
        import torch
        from training.ppo_trainer import PPOTrainer

        policy = MagicMock()
        policy.parameters = MagicMock(return_value=[torch.randn(2, 2, requires_grad=True)])

        with patch("training.ppo_trainer.HAS_MLFLOW", False):
            trainer = PPOTrainer(policy, mlflow_enabled=True)
            assert trainer.mlflow_enabled is False
            # Não deve levantar exceção
            trainer._start_mlflow_run()
            trainer._log_mlflow_metrics({"x": 1}, step=0)
            trainer._end_mlflow_run()


class TestCurriculumTrainerMLflow:
    """Testes para MLflow tracking no CurriculumTrainer."""

    @pytest.mark.skipif(not __import__("torch", __import__("sys").modules[__name__].__dict__), reason="torch not available")
    def test_curriculum_mlflow_config(self):
        """Deve respeitar configuração MLflow do config dict."""
        import torch
        from training.curriculum_trainer import CurriculumTrainer

        policy = MagicMock()
        policy.parameters = MagicMock(return_value=[torch.randn(2, 2, requires_grad=True)])
        policy.to = MagicMock(return_value=policy)

        config = {
            "mlflow": {
                "enabled": True,
                "tracking_uri": "runs/mlflow/curriculum",
                "experiment_name": "curriculum_test",
            }
        }

        with patch("training.curriculum_trainer.HAS_MLFLOW", True):
            with patch("training.curriculum_trainer.mlflow") as mock_mlflow:
                trainer = CurriculumTrainer(policy=policy, config=config, device="cpu")
                assert trainer.mlflow_enabled is True
                mock_mlflow.set_tracking_uri.assert_called_once_with("runs/mlflow/curriculum")
                mock_mlflow.set_experiment.assert_called_once_with("curriculum_test")

    @pytest.mark.skipif(not __import__("torch", __import__("sys").modules[__name__].__dict__), reason="torch not available")
    def test_curriculum_save_report(self):
        """Deve salvar relatório JSON do curriculum."""
        import torch
        from training.curriculum_trainer import CurriculumTrainer

        policy = MagicMock()
        policy.parameters = MagicMock(return_value=[torch.randn(2, 2, requires_grad=True)])
        policy.to = MagicMock(return_value=policy)
        policy.state_dict = MagicMock(return_value={})

        config = {"mlflow": {"enabled": False}}
        trainer = CurriculumTrainer(policy=policy, config=config, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "curriculum_report.json"
            trainer.save_report(path=report_path)
            assert report_path.exists()
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "progress" in data
            assert "config" in data


class TestRLBridgeWarmStart:
    """Testes para warm-start de checkpoints no RLBridge."""

    def test_warm_start_no_checkpoints(self):
        """Deve inicializar do zero quando não há checkpoints."""
        from neural.rl_bridge import RLBridge

        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = RLBridge(
                use_neural=False,
                q_learning_fallback=False,
                checkpoint_dir=Path(tmpdir),
            )
            assert bridge.use_neural is False

    @pytest.mark.skipif(not __import__("torch", __import__("sys").modules[__name__].__dict__), reason="torch not available")
    def test_warm_start_loads_latest(self):
        """Deve carregar o checkpoint mais recente."""
        import torch
        from neural.rl_bridge import RLBridge

        with tempfile.TemporaryDirectory() as tmpdir:
            # Criar checkpoints fake
            ckpt1 = Path(tmpdir) / "ckpt_1.pt"
            ckpt2 = Path(tmpdir) / "ckpt_2.pt"
            torch.save({"dummy": 1}, ckpt1)
            torch.save({"dummy": 2}, ckpt2)

            # Simular policy que aceita state_dict e tem parametros
            with patch("neural.neural_policy.NeuralPolicy") as MockPolicy:
                mock_policy = MagicMock()
                mock_policy.to = MagicMock(return_value=mock_policy)
                mock_policy.parameters = MagicMock(return_value=[torch.randn(2, 2, requires_grad=True)])
                MockPolicy.return_value = mock_policy

                with patch("training.ppo_trainer.PPOTrainer") as MockTrainer:
                    mock_trainer = MagicMock()
                    MockTrainer.return_value = mock_trainer

                    bridge = RLBridge(
                        use_neural=True,
                        q_learning_fallback=False,
                        checkpoint_dir=Path(tmpdir),
                    )
                    mock_policy.load_state_dict.assert_called_once()
                    # Deve ter chamado com o mais recente (ckpt2)
                    args, _ = mock_policy.load_state_dict.call_args
                    assert "dummy" in args[0]

    def test_checkpoint_dir_default(self):
        """O diretório padrão de checkpoints deve ser models/checkpoints."""
        from neural.rl_bridge import RLBridge

        bridge = RLBridge(use_neural=False, q_learning_fallback=False)
        assert bridge.checkpoint_dir == Path("models/checkpoints")
