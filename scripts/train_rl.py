#!/usr/bin/env python3
"""
scripts/train_rl.py

Script de treino RL unificado para Soberana Omega.

Fluxo:
  1. Carrega config.json
  2. Valida episódios RL (scripts/validate_episodes.py)
  3. Se dados insuficientes → instrui data collection
  4. Se dados suficientes → treina com curriculum ou PPO direto
  5. Guarda checkpoints e loga no MLflow
  6. Gera relatório final

Usage:
    python scripts/train_rl.py
    python scripts/train_rl.py --mode ppo_direct
    python scripts/train_rl.py --mode curriculum --episodes-dir dataset/raw/episodes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: Path = Path("config.json")) -> Dict[str, Any]:
    """Carrega configuração do projeto."""
    if not path.exists():
        logger.error(f"[TRAIN_RL] config.json não encontrado em {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_episodes(episodes_dir: Path, min_valid_episodes: int = 5) -> Dict[str, Any]:
    """Valida episódios usando validate_episodes.py."""
    try:
        from scripts.validate_episodes import validate_all_episodes
        report = validate_all_episodes(episodes_dir, fix_empty=False)
        return report
    except Exception as e:
        logger.error(f"[TRAIN_RL] Falha na validação de episódios: {e}")
        return {"total_episodes": 0, "valid_episodes": 0, "episodes": []}


def load_expert_data(episodes_dir: Path) -> List[Any]:
    """Carrega dados expert (state, action) dos episódios válidos."""
    expert_data = []
    if not episodes_dir.exists():
        return expert_data

    for episode_dir in sorted(episodes_dir.iterdir()):
        if not episode_dir.is_dir():
            continue
        metadata_path = episode_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            frames = metadata.get("frames", [])
            for frame in frames:
                state = frame.get("state_vector")
                action = frame.get("action_idx")
                if state is not None and action is not None:
                    expert_data.append((np.array(state, dtype=np.float32), int(action)))
        except Exception as e:
            logger.warning(f"[TRAIN_RL] Falha ao carregar {metadata_path}: {e}")

    logger.info(f"[TRAIN_RL] Dados expert carregados: {len(expert_data)} transições")
    return expert_data


def build_experience_buffer(episodes_dir: Path, capacity: int = 10000):
    """Constrói ExperienceBuffer a partir dos episódios."""
    from core.experience_buffer import ExperienceBuffer
    from neural.rl_bridge import Experience

    buffer = ExperienceBuffer(capacity=capacity)
    if not episodes_dir.exists():
        return buffer

    for episode_dir in sorted(episodes_dir.iterdir()):
        if not episode_dir.is_dir():
            continue
        metadata_path = episode_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            frames = metadata.get("frames", [])
            buffer.start_episode()
            for frame in frames:
                state = frame.get("state_vector")
                action = frame.get("action_idx")
                reward = frame.get("reward", 0.0)
                next_state = frame.get("next_state_vector")
                done = frame.get("done", False)
                value = frame.get("value", 0.0)
                log_prob = frame.get("log_prob", 0.0)
                grid = frame.get("grid")
                next_grid = frame.get("next_grid")

                if state is not None and action is not None and next_state is not None:
                    buffer.add(
                        Experience(
                            state_vector=np.array(state, dtype=np.float32),
                            grid=np.array(grid, dtype=np.float32) if grid is not None else None,
                            action_idx=int(action),
                            reward=float(reward),
                            next_state_vector=np.array(next_state, dtype=np.float32),
                            next_grid=np.array(next_grid, dtype=np.float32) if next_grid is not None else None,
                            done=bool(done),
                            value=float(value),
                            log_prob=float(log_prob),
                        )
                    )
            buffer.end_episode()
        except Exception as e:
            logger.warning(f"[TRAIN_RL] Falha ao processar {metadata_path}: {e}")

    logger.info(f"[TRAIN_RL] ExperienceBuffer: {len(buffer)} transições")
    return buffer


def train_ppo_direct(
    policy,
    experience_buffer,
    config: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Treina PPO diretamente (sem curriculum)."""
    from training.ppo_trainer import PPOTrainer

    mlflow_cfg = config.get("mlflow", {})
    trainer = PPOTrainer(
        policy,
        learning_rate=config.get("rl", {}).get("learning_rate", 3e-4),
        mlflow_enabled=mlflow_cfg.get("enabled", False),
        mlflow_tracking_uri=mlflow_cfg.get("tracking_uri"),
        mlflow_experiment_name=mlflow_cfg.get("experiment_name", "soberana_omega_ppo"),
    )

    checkpoint_path = "models/checkpoints/ppo_direct_final.pt"
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    stats = trainer.train(
        experience_buffer,
        num_updates=config.get("rl", {}).get("num_updates", 100),
        batch_size=config.get("rl", {}).get("batch_size", 64),
        ppo_epochs=4,
        checkpoint_path=checkpoint_path,
        run_name="ppo_direct",
    )

    return {
        "mode": "ppo_direct",
        "checkpoint": checkpoint_path,
        "stats": stats,
    }


def train_curriculum(
    policy,
    expert_data,
    experience_buffer,
    config: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Treina com curriculum learning (BC -> CQL -> PPO)."""
    from training.curriculum_trainer import CurriculumTrainer

    curriculum_config = {
        "mlflow": config.get("mlflow", {}),
        "bc": {
            "max_epochs": 50,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "loss_threshold": 0.1,
            "min_epochs": 10,
            "checkpoint_dir": "models/curriculum/phase_1",
        },
        "cql": {
            "max_epochs": 100,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "loss_threshold": 0.05,
            "win_rate_threshold": 0.3,
            "min_epochs": 20,
            "cql_alpha": 1.0,
            "checkpoint_dir": "models/curriculum/phase_2",
        },
        "ppo": {
            "max_epochs": config.get("rl", {}).get("num_updates", 100),
            "batch_size": config.get("rl", {}).get("batch_size", 64),
            "learning_rate": config.get("rl", {}).get("learning_rate", 3e-4),
            "win_rate_threshold": 0.6,
            "min_epochs": 20,
            "checkpoint_dir": "models/curriculum/phase_3",
        },
    }

    trainer = CurriculumTrainer(policy=policy, config=curriculum_config, device=device)
    result = trainer.run(expert_data=expert_data, experience_buffer=experience_buffer)
    trainer.save_report()

    return {
        "mode": "curriculum",
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Treino RL unificado Soberana Omega")
    parser.add_argument("--mode", choices=["ppo_direct", "curriculum", "auto"], default="auto",
                        help="Modo de treino")
    parser.add_argument("--episodes-dir", type=Path, default=Path("dataset/raw/episodes"),
                        help="Diretório com episódios RL")
    parser.add_argument("--config", type=Path, default=Path("config.json"),
                        help="Caminho para config.json")
    parser.add_argument("--min-episodes", type=int, default=5,
                        help="Mínimo de episódios válidos para treinar")
    parser.add_argument("--device", type=str, default=None,
                        help="Device torch (cpu/cuda)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(" SOBERANA OMEGA — Treino RL Unificado")
    logger.info("=" * 60)

    # 1. Carregar config
    config = load_config(args.config)
    logger.info(f"[TRAIN_RL] Config carregada: {args.config}")

    # 2. Validar episódios
    logger.info(f"[TRAIN_RL] Validando episódios em: {args.episodes_dir}")
    report = validate_episodes(args.episodes_dir, min_valid_episodes=args.min_episodes)

    logger.info(f"[TRAIN_RL] Total episódios: {report['total_episodes']}")
    logger.info(f"[TRAIN_RL] Válidos: {report['valid_episodes']}")
    logger.info(f"[TRAIN_RL] Inválidos: {report['invalid_episodes']}")

    if report["valid_episodes"] < args.min_episodes:
        logger.warning("[TRAIN_RL] DADOS INSUFICIENTES para treino RL!")
        logger.warning("[TRAIN_RL] Ação necessária: execute data collection primeiro.")
        logger.warning("  python scripts/collect_episodes.py  # (ou equivalente)")
        logger.warning("  Ou habilite rl.data_collection_mode=true no config.json")
        sys.exit(1)

    # 3. Carregar dados
    expert_data = load_expert_data(args.episodes_dir)
    experience_buffer = build_experience_buffer(args.episodes_dir, capacity=config.get("rl", {}).get("experience_buffer_size", 10000))

    # 4. Inicializar policy
    try:
        import torch
        from neural.neural_policy import NeuralPolicy

        schema = config.get("vision", {}).get("schema", "core")
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy = NeuralPolicy(schema=schema)
        policy.to(device)
        logger.info(f"[TRAIN_RL] NeuralPolicy inicializada em {device}")
    except Exception as e:
        logger.error(f"[TRAIN_RL] Falha ao inicializar NeuralPolicy: {e}")
        sys.exit(1)

    # 5. Escolher modo
    mode = args.mode
    if mode == "auto":
        # Se temos dados expert suficientes (>1000 transições), usar curriculum
        if len(expert_data) > 1000:
            mode = "curriculum"
            logger.info("[TRAIN_RL] Modo auto selecionado: curriculum (dados expert suficientes)")
        else:
            mode = "ppo_direct"
            logger.info("[TRAIN_RL] Modo auto selecionado: ppo_direct (poucos dados expert)")

    # 6. Treinar
    start_time = time.time()
    if mode == "ppo_direct":
        result = train_ppo_direct(policy, experience_buffer, config, device)
    else:
        result = train_curriculum(policy, expert_data, experience_buffer, config, device)

    elapsed = time.time() - start_time
    result["elapsed_seconds"] = elapsed

    # 7. Relatório final
    report_path = Path("training_reports/train_rl_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    logger.info("=" * 60)
    logger.info(" TREINO CONCLUÍDO")
    logger.info("=" * 60)
    logger.info(f"Modo: {result['mode']}")
    logger.info(f"Tempo: {elapsed:.1f}s")
    if result["mode"] == "ppo_direct":
        logger.info(f"Checkpoint: {result.get('checkpoint')}")
    else:
        logger.info(f"Fases completadas: {result['result'].get('phases_completed', 0)}")
    logger.info(f"Relatório: {report_path}")


if __name__ == "__main__":
    main()
