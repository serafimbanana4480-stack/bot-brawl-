"""
train_8class_model.py

Script simplificado para treinar modelo YOLO com schema core ou extended.

Classes:
    0: Player   - Jogador controlado
    1: Bush     - Arbustos (cover)
    2: Enemy    - Inimigos
    3: Cubebox  - Caixas de power cubes
    4: Wall     - Paredes/obstaculos
    5: Powerup  - Power-ups diversos
    6: Bullet   - Balas/projeteis
    7: Super    - Indicador de super

Usage:
    py train_8class_model.py --epochs 50 --batch 8 --device cpu
    py train_8class_model.py --epochs 100 --batch 16 --device cuda
"""

import argparse
import sys
from pathlib import Path

import yaml

from training.schema_dataset_builder import build_schema_dataset


def resolve_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Treinar YOLO com schema core ou extended")
    parser.add_argument("--epochs", type=int, default=50, help="Numero de epocas")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem")
    parser.add_argument("--data", type=str, default="dataset/roboflow_raw_v2/data.yaml", help="Dataset config")
    parser.add_argument(
        "--schema",
        type=str,
        default="core",
        choices=["core", "extended"],
        help="Class schema to train (default: core to match current dataset)",
    )
    parser.add_argument("--freeze", type=int, default=0, help="Freeze backbone layers")
    parser.add_argument("--cos-lr", action="store_true", help="Enable cosine LR schedule")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERRO: Dataset nao encontrado em {data_path}")
        print("Execute primeiro: py training/download_roboflow_dataset.py --api-key SUA_CHAVE")
        sys.exit(1)

    from training.class_schema import get_schema
    from ultralytics import YOLO

    classes = get_schema(args.schema)
    source_dataset = data_path.parent
    derived_dataset = source_dataset.parent / f"{source_dataset.name}_{args.schema}"
    build_schema_dataset(source_dataset, derived_dataset, schema=args.schema)
    derived_data = derived_dataset / "data.yaml"
    derived_config = {
        "path": str(derived_dataset.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": classes,
        "nc": len(classes),
    }
    with open(derived_data, "w", encoding="utf-8") as f:
        yaml.safe_dump(derived_config, f, sort_keys=True)

    print("=" * 60)
    print("TREINO YOLO")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Epocas: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Image size: {args.imgsz}")
    print(f"Schema: {args.schema} ({len(classes)} classes)")
    print(f"Derived data yaml: {derived_data}")
    print(f"Freeze: {args.freeze} | Cos LR: {args.cos_lr} | Resume: {args.resume}")
    print("=" * 60)

    resolved_device = resolve_device(args.device)
    print(f"Resolved device: {resolved_device}")

    print()
    print("Carregando modelo base YOLOv8n...")
    model = YOLO("yolov8n.pt")

    print()
    print("Iniciando treino...")
    model.train(
        data=str(derived_data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=resolved_device,
        project="runs",
        name=f"brawlstars_{args.schema}",
        exist_ok=True,
        verbose=True,
        patience=10,
        save=True,
        save_period=10,
        freeze=args.freeze,
        cos_lr=args.cos_lr,
        resume=args.resume,
    )

    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", Path(f"runs/brawlstars_{args.schema}")))
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"

    print()
    print("=" * 60)
    print("TREINO CONCLUIDO")
    print("=" * 60)
    print(f"Modelo guardado em: {best_path}")
    print(f"Ultimo modelo: {last_path}")

    print()
    print("Validando modelo...")
    metrics = model.val(data=str(derived_data), device=resolved_device)
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

    if best_path.exists():
        import shutil
        dest = Path(f"models/brawlstars_yolov8_{args.schema}.pt")
        shutil.copy(best_path, dest)
        print()
        print(f"Modelo copiado para: {dest}")
        if args.schema == "core":
            legacy_dest = Path("models/brawlstars_yolov8.pt")
            try:
                shutil.copy(best_path, legacy_dest)
                print(f"Modelo tambem copiado para legado: {legacy_dest}")
            except Exception as e:
                print(f"Aviso: nao foi possivel atualizar o legado: {e}")


if __name__ == "__main__":
    main()
