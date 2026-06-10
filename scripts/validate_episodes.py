#!/usr/bin/env python3
"""
scripts/validate_episodes.py

Valida episodios de gameplay coletados para treino RL.
Verifica:
  - Episodio tem >100 frames
  - Cada frame tem state_vector, action_idx, reward, next_state_vector
  - Rewards nao sao todos 0.0
  - Deteccoes YOLO presentes (nao null)
  - Grids espaciais 21x21 presentes (se coletados)

Guarda relatorio de validacao em dataset/raw/validation_report.json

Usage:
    python scripts/validate_episodes.py
    python scripts/validate_episodes.py --episodes-dir dataset/raw/episodes
    python scripts/validate_episodes.py --fix-empty  # remove episodios invalidos
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def validate_episode(episode_dir: Path) -> Dict:
    """Valida um unico episodio e retorna dict com resultados."""
    metadata_path = episode_dir / "metadata.json"
    result = {
        "episode_id": episode_dir.name,
        "valid": False,
        "frame_count": 0,
        "frames_with_state_vector": 0,
        "frames_with_action_idx": 0,
        "frames_with_reward": 0,
        "frames_with_next_state": 0,
        "frames_with_detections": 0,
        "frames_with_grid": 0,
        "all_rewards_zero": True,
        "mean_reward": 0.0,
        "max_reward": 0.0,
        "min_reward": 0.0,
        "issues": [],
    }

    if not metadata_path.exists():
        result["issues"].append("metadata.json nao encontrado")
        return result

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        result["issues"].append(f"falha ao ler metadata.json: {e}")
        return result

    frames = metadata.get("frames", [])
    result["frame_count"] = len(frames)

    if len(frames) < 100:
        result["issues"].append(f"apenas {len(frames)} frames (minimo 100)")

    rewards = []
    for frame in frames:
        if frame.get("state_vector") is not None:
            result["frames_with_state_vector"] += 1
        else:
            result["issues"].append(f"frame {frame.get('frame_id')}: state_vector ausente")

        if frame.get("action_idx") is not None:
            result["frames_with_action_idx"] += 1
        else:
            result["issues"].append(f"frame {frame.get('frame_id')}: action_idx ausente")

        reward = frame.get("reward", 0.0)
        if reward != 0.0:
            result["frames_with_reward"] += 1
        rewards.append(reward)

        if frame.get("next_state_vector") is not None:
            result["frames_with_next_state"] += 1
        else:
            result["issues"].append(f"frame {frame.get('frame_id')}: next_state_vector ausente")

        if frame.get("detections") is not None:
            result["frames_with_detections"] += 1

        if frame.get("grid") is not None:
            result["frames_with_grid"] += 1

    if rewards:
        result["mean_reward"] = float(np.mean(rewards))
        result["max_reward"] = float(np.max(rewards))
        result["min_reward"] = float(np.min(rewards))
        result["all_rewards_zero"] = all(r == 0.0 for r in rewards)

    if result["all_rewards_zero"]:
        result["issues"].append("todos os rewards sao 0.0")

    if result["frames_with_detections"] == 0:
        result["issues"].append("nenhum frame tem deteccoes YOLO")

    # Episodio e valido se passar nos criterios principais
    result["valid"] = (
        result["frame_count"] >= 100
        and not result["all_rewards_zero"]
        and result["frames_with_state_vector"] >= result["frame_count"] * 0.9
        and result["frames_with_action_idx"] >= result["frame_count"] * 0.9
        and result["frames_with_next_state"] >= result["frame_count"] * 0.9
        and result["frames_with_detections"] > 0
    )

    return result


def validate_all_episodes(episodes_dir: Path, fix_empty: bool = False) -> Dict:
    """Valida todos os episodios e retorna relatorio consolidado."""
    report = {
        "episodes_dir": str(episodes_dir),
        "total_episodes": 0,
        "valid_episodes": 0,
        "invalid_episodes": 0,
        "empty_episodes": 0,
        "total_frames": 0,
        "total_valid_frames": 0,
        "episodes": [],
    }

    if not episodes_dir.exists():
        print(f"[ERRO] Diretorio nao encontrado: {episodes_dir}")
        return report

    episode_dirs = sorted([d for d in episodes_dir.iterdir() if d.is_dir()])
    report["total_episodes"] = len(episode_dirs)

    for episode_dir in episode_dirs:
        result = validate_episode(episode_dir)
        report["episodes"].append(result)
        report["total_frames"] += result["frame_count"]

        if result["frame_count"] == 0:
            report["empty_episodes"] += 1
            if fix_empty:
                try:
                    import shutil
                    shutil.rmtree(episode_dir)
                    print(f"[FIX] Removido episodio vazio: {episode_dir.name}")
                    result["issues"].append("REMOVIDO (vazio)")
                except Exception as e:
                    print(f"[ERRO] Falha ao remover {episode_dir.name}: {e}")

        if result["valid"]:
            report["valid_episodes"] += 1
            report["total_valid_frames"] += result["frame_count"]
        else:
            report["invalid_episodes"] += 1

    return report


def print_report(report: Dict):
    """Imprime relatorio de validacao no terminal."""
    print("\n" + "=" * 60)
    print(" RELATORIO DE VALIDACAO DE EPISODIOS RL")
    print("=" * 60)
    print(f"Diretorio:     {report['episodes_dir']}")
    print(f"Total:         {report['total_episodes']} episodios")
    print(f"Validos:       {report['valid_episodes']} episodios")
    print(f"Invalidos:     {report['invalid_episodes']} episodios")
    print(f"Vazios:        {report['empty_episodes']} episodios")
    print(f"Frames total:  {report['total_frames']}")
    print(f"Frames validos: {report['total_valid_frames']}")
    print("-" * 60)

    for ep in report["episodes"]:
        status = "OK" if ep["valid"] else "FALHA"
        print(f"\n[{status}] {ep['episode_id']}")
        print(f"  Frames: {ep['frame_count']} | "
              f"state_vec: {ep['frames_with_state_vector']} | "
              f"action: {ep['frames_with_action_idx']} | "
              f"next_state: {ep['frames_with_next_state']} | "
              f"detections: {ep['frames_with_detections']} | "
              f"reward_mean: {ep['mean_reward']:.4f}")
        if ep["issues"]:
            for issue in ep["issues"][:5]:  # mostra ate 5 issues
                print(f"  ! {issue}")
            if len(ep["issues"]) > 5:
                print(f"  ! ... e mais {len(ep['issues']) - 5} issues")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Valida episodios de gameplay RL")
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=Path("dataset/raw/episodes"),
        help="Diretorio com episodios",
    )
    parser.add_argument(
        "--fix-empty",
        action="store_true",
        help="Remove episodios vazios (0 frames)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/raw/validation_report.json"),
        help="Caminho para salvar relatorio JSON",
    )
    args = parser.parse_args()

    report = validate_all_episodes(args.episodes_dir, fix_empty=args.fix_empty)
    print_report(report)

    # Salva relatorio JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Relatorio salvo em: {args.output}")

    # Exit code: 0 se todos validos, 1 se ha invalidos
    if report["invalid_episodes"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
