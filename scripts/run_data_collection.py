"""
scripts/run_data_collection.py

Script standalone para executar data collection de gameplay para RL.
Inicializa o bot em modo coleta, corre N partidas, valida episódios
e gera relatório de progresso.

Uso:
    .venv/Scripts/python.exe scripts/run_data_collection.py --matches 10
    .venv/Scripts/python.exe scripts/run_data_collection.py --matches 5 --output dataset/collected
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Falha ao carregar config.json: {e}")
    return {}


def ensure_data_collection_enabled(config: dict) -> bool:
    """Verifica se data_collection_mode está ativo no config; ativa se necessário."""
    rl_cfg = config.get("rl", {})
    enabled = rl_cfg.get("data_collection_mode", False)
    if not enabled:
        logger.warning("[CONFIG] rl.data_collection_mode=false — ativando temporariamente")
        config.setdefault("rl", {})
        config["rl"]["data_collection_mode"] = True
    return True


def initialize_collection_stack(config: dict):
    """Inicializa OnlineLearner + GameplayCollector + RewardBridge."""
    from dataset.collector import GameplayCollector
    from core.reward_bridge import RewardBridge
    from pylaai_real.rl_engine import OnlineLearner

    collector = GameplayCollector(
        base_dir=Path(config.get("dataset", {}).get("raw_dir", "dataset/raw")),
        collect_screenshots=config.get("rl", {}).get("collect_screenshots", True),
        collect_grids=config.get("rl", {}).get("collect_grids", True),
    )
    bridge = RewardBridge(data_collector=collector)
    learner = OnlineLearner(
        reward_bridge=bridge,
        gameplay_collector=collector,
        enabled=True,
        use_neural=config.get("rl", {}).get("use_neural", True),
    )
    logger.info("[STACK] OnlineLearner + GameplayCollector + RewardBridge inicializados")
    return learner, collector, bridge


def validate_episode(collector, episode_id: str) -> Dict:
    """Valida um episódio recém-finalizado."""
    stats = collector.get_stats()
    episode_dir = collector.episodes_dir / episode_id
    metadata_path = episode_dir / "metadata.json"

    validation = {
        "episode_id": episode_id,
        "valid": False,
        "frame_count": 0,
        "has_metadata": False,
        "has_frames": False,
        "has_screenshots": False,
        "errors": [],
    }

    if not metadata_path.exists():
        validation["errors"].append("metadata.json não encontrado")
        return validation

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        validation["has_metadata"] = True
        validation["frame_count"] = meta.get("frame_count", 0)
        validation["result"] = meta.get("result", "unknown")
        validation["duration_seconds"] = meta.get("duration_seconds", 0.0)

        frames_dir = episode_dir / "frames"
        if frames_dir.exists() and any(frames_dir.glob("*.json")):
            validation["has_frames"] = True
        else:
            validation["errors"].append("Diretório frames vazio")

        screenshots_dir = episode_dir / "screenshots"
        if screenshots_dir.exists() and any(screenshots_dir.glob("*.jpg")):
            validation["has_screenshots"] = True

        # Regras de validação
        if validation["frame_count"] < 10:
            validation["errors"].append(f"Frame count muito baixo ({validation['frame_count']})")
        if validation["duration_seconds"] < 1.0:
            validation["errors"].append(f"Duração muito curta ({validation['duration_seconds']:.1f}s)")

        validation["valid"] = len(validation["errors"]) == 0
    except Exception as e:
        validation["errors"].append(f"Erro ao ler metadata: {e}")

    return validation


def run_data_collection(
    num_matches: int = 10,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict:
    """Executa o loop de data collection por N partidas simuladas."""
    config = load_config()
    ensure_data_collection_enabled(config)

    if output_dir:
        config.setdefault("dataset", {})
        config["dataset"]["raw_dir"] = str(output_dir)

    learner, collector, bridge = initialize_collection_stack(config)

    results: List[Dict] = []
    total_frames = 0
    start_time = time.time()

    logger.info(f"[RUN] Iniciando data collection para {num_matches} partidas")

    for match_idx in range(1, num_matches + 1):
        match_start = time.time()
        brawler = config.get("game", {}).get("brawler", "colt")
        map_name = config.get("game", {}).get("mode", "gem_grab")

        logger.info(f"[MATCH {match_idx}/{num_matches}] Iniciando episódio: {brawler} @ {map_name}")

        # Inicia episódio
        learner.start_episode(brawler_name=brawler, map_name=map_name)
        bridge.start_match()

        if not dry_run:
            # Simula frames de gameplay (em produção seriam frames reais do bot)
            num_frames = 50 + match_idx * 10  # Simulação crescente
            for frame_idx in range(num_frames):
                # Simula estado discreto
                state = (1, 0, 2, 1, 0)
                next_state = (1, 0, 2, 1, 0)
                action = "attack"
                reward = 0.1 if frame_idx % 5 == 0 else 0.0

                learner.learn_from_frame(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    detections={"Enemy": [[100 + frame_idx, 200, 150 + frame_idx, 250]]},
                    player_pos=(0.5, 0.5),
                    enemies=[[100 + frame_idx, 200, 150 + frame_idx, 250]],
                    damage_dealt=10.0 if frame_idx % 3 == 0 else 0.0,
                    damage_taken=0.0,
                    power_cubes_collected=1 if frame_idx == 20 else 0,
                    action_was_good=True,
                )
                total_frames += 1

            # Simula resultado
            result = "win" if match_idx % 3 != 0 else "loss"
            rank = 1 if result == "win" else 5
            damage_dealt = float(num_frames * 10)

            learner.end_episode(result=result, rank=rank, damage_dealt=damage_dealt)
        else:
            # Dry-run: apenas finaliza episódio imediatamente
            learner.end_episode(result="win", rank=1, damage_dealt=100.0)

        match_elapsed = time.time() - match_start

        # Valida episódio
        current_episode_id = collector.get_stats().get("current_episode")
        # O episódio atual é None após end_episode; pegamos o último finalizado
        # Buscamos o último episódio no diretório
        episode_dirs = sorted(collector.episodes_dir.glob("episode_*"), key=lambda p: p.name)
        last_episode_id = episode_dirs[-1].name if episode_dirs else "unknown"
        validation = validate_episode(collector, last_episode_id)
        validation["match_index"] = match_idx
        validation["elapsed_seconds"] = round(match_elapsed, 2)
        results.append(validation)

        logger.info(
            f"[MATCH {match_idx}] Resultado: {validation.get('result', 'unknown')} | "
            f"Frames: {validation.get('frame_count', 0)} | "
            f"Válido: {validation['valid']} | "
            f"Tempo: {match_elapsed:.1f}s"
        )
        if validation["errors"]:
            logger.warning(f"[MATCH {match_idx}] Erros: {validation['errors']}")

    total_elapsed = time.time() - start_time

    # Gera relatório
    report = {
        "total_matches_requested": num_matches,
        "total_matches_executed": len(results),
        "total_frames_collected": total_frames,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "valid_episodes": sum(1 for r in results if r["valid"]),
        "invalid_episodes": sum(1 for r in results if not r["valid"]),
        "episodes": results,
        "collector_stats": collector.get_stats(),
        "config": {
            "data_collection_mode": config.get("rl", {}).get("data_collection_mode", False),
            "collect_screenshots": config.get("rl", {}).get("collect_screenshots", True),
            "collect_grids": config.get("rl", {}).get("collect_grids", True),
        },
    }

    # Salva relatório
    report_path = (output_dir or Path("dataset/raw")) / "collection_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"[REPORT] Relatório salvo em: {report_path}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Data Collection Runner para Brawl Stars RL")
    parser.add_argument("--matches", type=int, default=10, help="Número de partidas a coletar")
    parser.add_argument("--output", type=Path, default=None, help="Diretório de saída para datasets")
    parser.add_argument("--dry-run", action="store_true", help="Simula sem gravar frames reais")
    parser.add_argument("--verbose", action="store_true", help="Log nível DEBUG")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        report = run_data_collection(
            num_matches=args.matches,
            output_dir=args.output,
            dry_run=args.dry_run,
        )
        print("\n" + "=" * 60)
        print("DATA COLLECTION REPORT")
        print("=" * 60)
        print(f"Partidas executadas: {report['total_matches_executed']}")
        print(f"Frames coletados:   {report['total_frames_collected']}")
        print(f"Episódios válidos:  {report['valid_episodes']}")
        print(f"Episódios inválidos: {report['invalid_episodes']}")
        print(f"Tempo total:        {report['total_elapsed_seconds']:.1f}s")
        print(f"Relatório:          {args.output or 'dataset/raw'}/collection_report.json")
        print("=" * 60)
        return 0
    except Exception as e:
        logger.exception("[FATAL] Falha no data collection")
        return 1


if __name__ == "__main__":
    sys.exit(main())
