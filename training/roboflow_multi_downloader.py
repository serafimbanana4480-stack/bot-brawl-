"""
roboflow_multi_downloader.py

Download e integração de múltiplos datasets do Roboflow Universe.
Suporta download em batch, remapeamento de classes, merge unificado e
validação de compatibilidade com os schemas do projeto.

Uso:
    # Baixar datasets específicos
    python training/roboflow_multi_downloader.py \
        --dataset bloxxy/brawl-stars-dataset \
        --dataset ivan-yordanov-cxrbb/brawl-stars-everything \
        --api-key $ROBOFLOW_API_KEY

    # Baixar datasets compatíveis descobertos automaticamente
    python training/roboflow_multi_downloader.py \
        --discover --schema core --api-key $ROBOFLOW_API_KEY

    # Merge de datasets já baixados
    python training/roboflow_multi_downloader.py \
        --merge-only --input-dir dataset/roboflow_raw --output-dir dataset/merged
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.error import URLError
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.class_registry import (
    CORE_CLASSES,
    EXTENDED_CLASSES,
    FULL_CLASSES,
    ROBOFLOW_TO_CANONICAL,
    VISUAL_CLASSES,
    get_canonical,
    get_class_id,
    get_schema,
)
from training.roboflow_dataset_discoverer import (
    DatasetInfo,
    filter_compatible,
    score_dataset,
    search_universe,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger("roboflow_multi_downloader")


# ============================================================================
# CLASS MAPPING PER DATASET
# ============================================================================

# Alguns datasets usam nomes de classes diferentes dos do bloxxy dataset.
# Mapeamentos customizados por workspace/project.
DATASET_CLASS_MAP: Dict[str, Dict[str, Optional[int]]] = {}


def build_dataset_class_map(schema: str = "core") -> Dict[str, Dict[str, Optional[int]]]:
    """
    Constrói um mapeamento de classes para cada dataset conhecido.
    O mapeamento usa ROBOFLOW_TO_CANONICAL como base e expande com aliases.
    """
    global DATASET_CLASS_MAP
    if DATASET_CLASS_MAP:
        return DATASET_CLASS_MAP

    # Base mapping: Roboflow class name -> canonical class ID in target schema
    base_map: Dict[str, Optional[int]] = {}
    for roboflow_name, canonical_name in ROBOFLOW_TO_CANONICAL.items():
        if canonical_name is not None:
            class_id = get_class_id(canonical_name, schema=schema)
            base_map[roboflow_name] = class_id if class_id is not None else -1
        else:
            base_map[roboflow_name] = -1

    DATASET_CLASS_MAP = {"__default__": base_map}
    return DATASET_CLASS_MAP


def get_class_map_for_dataset(
    dataset_info: DatasetInfo,
    schema: str = "core",
    auto_detect_classes: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Retorna o mapeamento de classes para um dataset específico.

    Se auto_detect_classes for fornecido, tenta mapear automaticamente
    usando aliases conhecidos e heurísticas de nome.
    """
    default_map = build_dataset_class_map(schema).get("__default__", {})
    result: Dict[str, int] = {}

    if auto_detect_classes:
        for cls_name in auto_detect_classes:
            # 1. Tentar mapeamento direto via registry
            mapped = default_map.get(cls_name)
            if mapped is not None and mapped != -1:
                result[cls_name] = mapped
                continue

            # 2. Tentar via canonical name (case-insensitive)
            canonical = get_canonical(cls_name)
            class_id = get_class_id(canonical, schema=schema)
            if class_id is not None:
                result[cls_name] = class_id
                continue

            # 3. Heurísticas de nome (parciais)
            lowered = cls_name.lower()
            if "enemy" in lowered or "opponent" in lowered or "foe" in lowered:
                cid = get_class_id("enemy", schema)
                if cid is not None:
                    result[cls_name] = cid
                    continue
            if "friendly" in lowered or "ally" in lowered or "player" in lowered or "me" in lowered or "self" in lowered:
                cid = get_class_id("player", schema)
                if cid is not None:
                    result[cls_name] = cid
                    continue
            if "box" in lowered or "cube" in lowered or "powercube" in lowered:
                cid = get_class_id("cubebox", schema)
                if cid is not None:
                    result[cls_name] = cid
                    continue
            if "powerup" in lowered or "power" in lowered or "pp" in lowered:
                cid = get_class_id("powerup", schema)
                if cid is not None:
                    result[cls_name] = cid
                    continue
            if "gem" in lowered:
                cid = get_class_id("gem", schema)
                if cid is not None:
                    result[cls_name] = cid
                    continue
            if "ball" in lowered:
                cid = get_class_id("ball", schema)
                if cid is not None:
                    result[cls_name] = cid
                    continue
            if "wall" in lowered:
                cid = get_class_id("wall", schema)
                if cid is not None:
                    result[cls_name] = cid
                    continue
            if "bush" in lowered:
                cid = get_class_id("bush", schema)
                if cid is not None:
                    result[cls_name] = cid
                    continue
            if "bullet" in lowered or "projectile" in lowered or "shot" in lowered:
                cid = get_class_id("bullet_neutral", schema)
                if cid is not None:
                    result[cls_name] = cid
                    continue
            if "super" in lowered:
                cid = get_class_id("super_area", schema)
                if cid is not None:
                    result[cls_name] = cid
                    continue
            if "hot" in lowered or "zone" in lowered:
                cid = get_class_id("hot_zone", schema)
                if cid is not None:
                    result[cls_name] = cid
                    continue
            if "safe" in lowered:
                # safe_zone ou safe_friendly/enemy — verificar contexto
                if "zone" in lowered:
                    cid = get_class_id("safe_zone", schema)
                    if cid is not None:
                        result[cls_name] = cid
                        continue
                elif "friend" in lowered:
                    cid = get_class_id("player", schema)
                    if cid is not None:
                        result[cls_name] = cid
                        continue
                elif "enemy" in lowered:
                    cid = get_class_id("enemy", schema)
                    if cid is not None:
                        result[cls_name] = cid
                        continue

            # Não mapeado -> -1 (skip)
            logger.debug(f"Unmapped class '{cls_name}' in dataset {dataset_info.workspace}/{dataset_info.project}")

    else:
        result = {k: v for k, v in default_map.items() if v is not None and v != -1}

    return result


# ============================================================================
# DOWNLOAD ENGINE
# ============================================================================

def download_single_dataset(
    dataset_info: DatasetInfo,
    output_base_dir: Path,
    api_key: str,
) -> Optional[Path]:
    """
    Faz download de um único dataset do Roboflow.

    Args:
        dataset_info: Metadados do dataset
        output_base_dir: Diretório base onde o dataset será extraído
        api_key: Roboflow API key

    Returns:
        Path para o diretório extraído, ou None se falhar
    """
    dataset_dir = output_base_dir / f"{dataset_info.workspace}_{dataset_info.project}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dataset_dir / "dataset.zip"
    download_url = dataset_info.download_url

    logger.info(f"Downloading {dataset_info.workspace}/{dataset_info.project}...")
    logger.info(f"URL: {download_url}")

    try:
        urlretrieve(download_url, zip_path)
        logger.info(f"Downloaded to {zip_path}")
    except URLError as e:
        logger.error(f"Download failed for {dataset_info.workspace}/{dataset_info.project}: {e}")
        return None

    logger.info("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dataset_dir)
        logger.info(f"Extracted to {dataset_dir}")
    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP for {dataset_info.workspace}/{dataset_info.project}")
        return None
    finally:
        if zip_path.exists():
            zip_path.unlink()

    return dataset_dir


def detect_dataset_structure(dataset_dir: Path) -> Tuple[List[Tuple[Path, Path]], Optional[Path]]:
    """
    Detecta a estrutura do dataset (com splits ou flat).

    Returns:
        (lista de (images_dir, labels_dir) por split, path do data.yaml)
    """
    splits = []
    yaml_path = None

    # Procurar data.yaml
    for candidate in dataset_dir.rglob("data.yaml"):
        yaml_path = candidate
        break

    # Estrutura com splits
    for split in ["train", "valid", "val", "test"]:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if images_dir.exists():
            splits.append((images_dir, labels_dir))

    # Se não encontrou splits, tentar estrutura flat
    if not splits:
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        if images_dir.exists():
            splits.append((images_dir, labels_dir))

    return splits, yaml_path


def read_yaml_classes(yaml_path: Path) -> List[str]:
    """Lê os nomes das classes de um ficheiro data.yaml."""
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names", [])
        if isinstance(names, dict):
            # YOLOv8 style: {0: 'class', 1: 'class'}
            return [names[k] for k in sorted(names.keys())]
        return list(names)
    except Exception as e:
        logger.warning(f"Could not read classes from {yaml_path}: {e}")
        return []


def remap_dataset_classes(
    dataset_dir: Path,
    class_map: Dict[str, int],
    class_names: List[str],
) -> Tuple[int, int]:
    """
    Remapeia as classes de um dataset para o schema do projeto.

    Args:
        dataset_dir: Diretório do dataset
        class_map: Mapeamento {nome_original_classe -> novo_id}
        class_names: Lista de nomes de classes na ordem dos IDs originais

    Returns:
        (número de anotações remapeadas, número de anotações removidas)
    """
    splits, _ = detect_dataset_structure(dataset_dir)
    remapped_total = 0
    removed_total = 0

    for images_dir, labels_dir in splits:
        if not labels_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                orig_cls_id = int(parts[0])
                bbox = parts[1:]

                # Obter nome da classe original
                if orig_cls_id < len(class_names):
                    orig_name = class_names[orig_cls_id]
                else:
                    removed_total += 1
                    continue

                # Mapear para novo ID
                new_cls_id = class_map.get(orig_name, -1)
                if new_cls_id == -1:
                    removed_total += 1
                    continue

                new_lines.append(f"{new_cls_id} {' '.join(bbox)}\n")
                remapped_total += 1

            with open(label_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

    logger.info(f"Remapped {remapped_total} annotations, removed {removed_total}")
    return remapped_total, removed_total


def prefix_filenames(dataset_dir: Path, prefix: str) -> None:
    """
    Renomeia todos os ficheiros de imagem e labels com um prefixo para evitar
    colisões quando múltiplos datasets são mesclados.
    """
    splits, _ = detect_dataset_structure(dataset_dir)

    for images_dir, labels_dir in splits:
        if not images_dir.exists():
            continue

        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                new_name = f"{prefix}_{img_file.name}"
                img_file.rename(images_dir / new_name)

                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    label_file.rename(labels_dir / f"{prefix}_{img_file.stem}.txt")

    logger.info(f"Prefixed files in {dataset_dir} with '{prefix}'")


def get_classes_from_labels(dataset_dir: Path) -> Set[int]:
    """Retorna o conjunto de IDs de classes presentes num dataset."""
    classes: Set[int] = set()
    splits, _ = detect_dataset_structure(dataset_dir)
    for _, labels_dir in splits:
        if not labels_dir.exists():
            continue
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        parts = line.split()
                        if parts:
                            try:
                                classes.add(int(parts[0]))
                            except ValueError:
                                pass
    return classes


# ============================================================================
# MERGE ENGINE
# ============================================================================

def merge_datasets(
    input_dirs: List[Path],
    output_dir: Path,
    schema: str = "core",
) -> Dict[str, int]:
    """
    Mescla múltiplos datasets num único dataset com splits train/val/test.

    Args:
        input_dirs: Lista de diretórios de datasets (cada um com splits)
        output_dir: Diretório de saída
        schema: Schema alvo (core/extended/full)

    Returns:
        Estatísticas do merge
    """
    logger.info("=" * 60)
    logger.info("MERGING DATASETS")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total_images": 0, "total_labels": 0, "datasets_merged": 0}

    # Coletar todos os splits de todos os datasets
    all_splits: Dict[str, List[Tuple[Path, Path]]] = {"train": [], "val": [], "test": []}

    for ds_dir in input_dirs:
        splits, _ = detect_dataset_structure(ds_dir)
        for images_dir, labels_dir in splits:
            # Determinar o split a partir do path
            split_name = "train"
            for candidate in ["train", "val", "valid", "test"]:
                if candidate in str(images_dir).lower():
                    split_name = "val" if candidate == "valid" else candidate
                    break
            all_splits[split_name].append((images_dir, labels_dir))

    # Copiar ficheiros para o diretório de saída
    for split_name, split_sources in all_splits.items():
        if not split_sources:
            continue

        dst_images = output_dir / split_name / "images"
        dst_labels = output_dir / split_name / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        idx = 0
        for images_dir, labels_dir in split_sources:
            if not images_dir.exists():
                continue

            for img_file in sorted(images_dir.glob("*")):
                if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue

                new_name = f"img_{idx:06d}{img_file.suffix}"
                shutil.copy2(img_file, dst_images / new_name)

                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, dst_labels / f"img_{idx:06d}.txt")
                    stats["total_labels"] += 1

                stats["total_images"] += 1
                idx += 1

        logger.info(f"{split_name}: {idx} images")

    stats["datasets_merged"] = len(input_dirs)
    logger.info(f"Total: {stats['total_images']} images, {stats['total_labels']} labels")

    # Criar data.yaml
    create_merged_yaml(output_dir, schema=schema)

    return stats


def create_merged_yaml(dataset_dir: Path, schema: str = "core") -> Path:
    """Cria data.yaml para dataset mesclado."""
    import yaml

    actual_classes = get_classes_from_labels(dataset_dir)
    expected_schema = get_schema(schema)
    classes = {i: name for i, name in expected_schema.items() if i in actual_classes}

    config = {
        "path": str(dataset_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": classes,
        "nc": len(classes),
    }

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Created {yaml_path} with {len(classes)} classes for schema={schema}: {classes}")
    return yaml_path


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def download_and_prepare_dataset(
    dataset_info: DatasetInfo,
    output_base_dir: Path,
    api_key: str,
    schema: str = "core",
) -> Optional[Path]:
    """
    Pipeline completo para um dataset: download, detect, remap, prefix.

    Returns:
        Path para o diretório preparado, ou None se falhar.
    """
    logger.info("-" * 60)
    logger.info(f"Processing: {dataset_info.workspace}/{dataset_info.project}")
    logger.info("-" * 60)

    # 1. Download
    dataset_dir = download_single_dataset(dataset_info, output_base_dir, api_key)
    if dataset_dir is None:
        return None

    # 2. Detectar estrutura e classes
    splits, yaml_path = detect_dataset_structure(dataset_dir)
    if not splits:
        logger.error(f"No valid dataset structure found in {dataset_dir}")
        return None

    class_names = read_yaml_classes(yaml_path) if yaml_path else dataset_info.class_names
    if not class_names:
        logger.warning(f"Could not detect classes for {dataset_info.workspace}/{dataset_info.project}")
        # Tentar usar classes conhecidas
        class_names = dataset_info.class_names

    logger.info(f"Detected classes: {class_names}")

    # 3. Construir mapeamento
    class_map = get_class_map_for_dataset(dataset_info, schema=schema, auto_detect_classes=class_names)
    logger.info(f"Class map: {class_map}")

    if not class_map:
        logger.warning(f"No compatible classes found for {dataset_info.workspace}/{dataset_info.project}, skipping remap")
        return dataset_dir

    # 4. Remapear classes
    remap_dataset_classes(dataset_dir, class_map, class_names)

    # 5. Prefixar ficheiros para evitar colisões
    prefix = f"{dataset_info.workspace}_{dataset_info.project}".replace("-", "_")
    prefix_filenames(dataset_dir, prefix)

    # 6. Validar classes finais
    final_classes = get_classes_from_labels(dataset_dir)
    logger.info(f"Final classes in dataset: {sorted(final_classes)}")

    if not final_classes:
        logger.warning(f"Dataset {dataset_info.workspace}/{dataset_info.project} has no valid annotations after filtering")

    return dataset_dir


def run_multi_download(
    datasets: List[DatasetInfo],
    output_dir: Path,
    api_key: str,
    schema: str = "core",
    merge: bool = True,
    merge_output_dir: Optional[Path] = None,
) -> Dict[str, any]:
    """
    Pipeline completo: download + remap de múltiplos datasets.

    Returns:
        Dicionário com estatísticas e diretórios processados.
    """
    prepared_dirs: List[Path] = []
    failed: List[str] = []
    stats_per_dataset: Dict[str, dict] = {}

    for ds in datasets:
        result = download_and_prepare_dataset(ds, output_dir, api_key, schema=schema)
        if result:
            prepared_dirs.append(result)
            final_classes = get_classes_from_labels(result)
            stats_per_dataset[f"{ds.workspace}/{ds.project}"] = {
                "path": str(result),
                "final_classes": sorted(final_classes),
            }
        else:
            failed.append(f"{ds.workspace}/{ds.project}")

    logger.info("=" * 60)
    logger.info(f"Download complete: {len(prepared_dirs)} succeeded, {len(failed)} failed")
    if failed:
        logger.info(f"Failed: {failed}")
    logger.info("=" * 60)

    # Merge
    if merge and prepared_dirs:
        merge_dir = merge_output_dir or output_dir.parent / "merged"
        merge_stats = merge_datasets(prepared_dirs, merge_dir, schema=schema)
        return {
            "prepared_dirs": [str(d) for d in prepared_dirs],
            "failed": failed,
            "merge_dir": str(merge_dir),
            "merge_stats": merge_stats,
            "stats_per_dataset": stats_per_dataset,
        }

    return {
        "prepared_dirs": [str(d) for d in prepared_dirs],
        "failed": failed,
        "stats_per_dataset": stats_per_dataset,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download multiple Roboflow datasets for Brawl Stars")
    parser.add_argument("--dataset", type=str, action="append", default=None,
                       help="Dataset identifier(s) in format workspace/project (can be used multiple times)")
    parser.add_argument("--discover", action="store_true",
                       help="Auto-discover compatible datasets from known list + web search")
    parser.add_argument("--schema", type=str, default="core", choices=["core", "extended", "full"],
                       help="Target schema for class remapping")
    parser.add_argument("--api-key", type=str, default=None,
                       help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--output-dir", type=str, default="dataset/roboflow_raw",
                       help="Base directory for downloaded datasets")
    parser.add_argument("--merge-output-dir", type=str, default="dataset/merged",
                       help="Output directory for merged dataset")
    parser.add_argument("--no-merge", action="store_true",
                       help="Skip merging datasets after download")
    parser.add_argument("--merge-only", action="store_true",
                       help="Only merge existing downloaded datasets")
    parser.add_argument("--input-dir", type=str, default="dataset/roboflow_raw",
                       help="Input directory for merge-only mode")
    parser.add_argument("--min-score", type=float, default=0.0,
                       help="Minimum compatibility score for auto-discover")
    parser.add_argument("--max-results", type=int, default=20,
                       help="Max datasets to download from discovery")
    parser.add_argument("--json", type=str, default=None,
                       help="Export results summary to JSON")
    args = parser.parse_args()

    api_key = args.api_key or __import__("os").environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        logger.error("Roboflow API key is required. Use --api-key or set ROBOFLOW_API_KEY env var.")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge-only mode
    if args.merge_only:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return 1

        subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
        if not subdirs:
            logger.error(f"No dataset directories found in {input_dir}")
            return 1

        merge_dir = Path(args.merge_output_dir)
        stats = merge_datasets(subdirs, merge_dir, schema=args.schema)
        logger.info(f"Merge complete: {stats}")
        return 0

    # Determine datasets to download
    datasets: List[DatasetInfo] = []

    if args.discover:
        # Combinar known + web search
        from training.roboflow_dataset_discoverer import KNOWN_DATASETS, search_universe
        all_datasets = list(KNOWN_DATASETS)
        try:
            web_results = search_universe(query="brawl stars", max_results=args.max_results)
            all_datasets.extend(web_results)
        except Exception as e:
            logger.warning(f"Web search failed: {e}")

        # Deduplicar e score
        seen: Dict[str, DatasetInfo] = {}
        for ds in all_datasets:
            key = f"{ds.workspace}/{ds.project}"
            if key not in seen:
                seen[key] = ds
        for ds in seen.values():
            score_dataset(ds, schema=args.schema)

        datasets = filter_compatible(list(seen.values()), min_score=args.min_score)
        datasets = datasets[:args.max_results]
        logger.info(f"Discovered {len(datasets)} compatible datasets")

    elif args.dataset:
        for ds_str in args.dataset:
            if "/" not in ds_str:
                logger.error(f"Invalid dataset format (expected workspace/project): {ds_str}")
                continue
            workspace, project = ds_str.split("/", 1)
            datasets.append(DatasetInfo(workspace=workspace, project=project, source="cli"))
    else:
        logger.error("No datasets specified. Use --dataset, --discover, or --merge-only.")
        return 1

    if not datasets:
        logger.error("No datasets to process.")
        return 1

    # Run download pipeline
    results = run_multi_download(
        datasets=datasets,
        output_dir=output_dir,
        api_key=api_key,
        schema=args.schema,
        merge=not args.no_merge,
        merge_output_dir=Path(args.merge_output_dir) if args.merge_output_dir else None,
    )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results exported to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
