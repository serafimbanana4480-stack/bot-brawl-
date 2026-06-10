import os
import shutil
import hashlib
import random
from pathlib import Path
from collections import defaultdict


def detect_classes(labels_dir: Path):
    """Detecta classes únicas a partir dos ficheiros de label YOLO."""
    classes = set()
    for lbl in labels_dir.glob("*.txt"):
        for line in lbl.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    classes.add(int(parts[0]))
                except ValueError:
                    pass
    return sorted(classes)


def find_label_for_image(img_path: Path, dataset_dir: Path):
    """Procura o label correspondente a uma imagem dentro de um dataset."""
    stem = img_path.stem
    candidates = [
        img_path.with_suffix(".txt"),
        dataset_dir / "labels" / f"{stem}.txt",
        dataset_dir / "train" / "labels" / f"{stem}.txt",
        dataset_dir / "test" / "labels" / f"{stem}.txt",
        dataset_dir / "val" / "labels" / f"{stem}.txt",
        dataset_dir / "valid" / "labels" / f"{stem}.txt",
    ]
    # Ajusta para estrutura images/labels dentro de splits
    rel = img_path.relative_to(dataset_dir)
    if rel.parts[0] in ("train", "test", "val", "valid", "images"):
        candidates.append(dataset_dir / rel.parent.parent / "labels" / f"{stem}.txt")
    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    return None


def consolidate_datasets(base_dir: Path, output_name: str = "consolidated",
                         split_ratio=(0.8, 0.1, 0.1), seed: int = 42):
    """
    Consolida todos os datasets YOLO válidos num único dataset.

    - Remove duplicados por hash MD5
    - Remove labels vazios
    - Cria data.yaml unificado
    - Divide em train/val/test
    """
    dataset_base = base_dir / "dataset"
    output_dir = dataset_base / output_name

    # Limpar output anterior
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Datasets a incluir (excluir quarantine e datasets vazios/quebrados)
    skip = {"quarantine", "synthetic_massive", "yolo_final", output_name,
            "raw", "training", "training_reports", "models", "bc", "cql",
            "dataset", "treino"}

    all_pairs = []
    seen_hashes = set()

    for ds_dir in sorted(dataset_base.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name in skip:
            continue

        for root, _dirs, files in os.walk(ds_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    img_path = Path(root) / f
                    lbl_path = find_label_for_image(img_path, ds_dir)
                    if lbl_path:
                        h = hashlib.md5(img_path.read_bytes()).hexdigest()
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            all_pairs.append((img_path, lbl_path))

    # Shuffle e split
    random.seed(seed)
    random.shuffle(all_pairs)
    n = len(all_pairs)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])

    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:n_train + n_val]
    test_pairs = all_pairs[n_train + n_val:]

    def copy_pairs(pairs, split):
        for img, lbl in pairs:
            shutil.copy2(str(img), str(output_dir / split / "images" / img.name))
            shutil.copy2(str(lbl), str(output_dir / split / "labels" / (img.stem + ".txt")))

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    copy_pairs(test_pairs, "test")

    # data.yaml
    all_classes = set()
    for _, lbl in all_pairs:
        for line in lbl.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    all_classes.add(int(parts[0]))
                except ValueError:
                    pass
    all_classes = sorted(all_classes)
    names = {i: f"class_{i}" for i in all_classes}

    yaml_path = output_dir / "data.yaml"
    yaml_content = (
        f"path: {output_dir.absolute().as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n\n"
        f"nc: {len(all_classes)}\n"
        f"names: {names}\n"
    )
    yaml_path.write_text(yaml_content)

    print(f"Consolidação concluída: {output_dir}")
    print(f"  Total: {n} | Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")
    print(f"  Classes: {len(all_classes)} | data.yaml: {yaml_path}")
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Consolida datasets YOLO num único dataset limpo.")
    parser.add_argument("--base", type=Path, default=Path("."),
                        help="Diretório base do projeto (default: .)")
    parser.add_argument("--output", type=str, default="consolidated",
                        help="Nome do dataset de saída (default: consolidated)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed para shuffle (default: 42)")
    args = parser.parse_args()

    consolidate_datasets(args.base, args.output, seed=args.seed)
