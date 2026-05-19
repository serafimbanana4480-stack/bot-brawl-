"""
roboflow_dataset_discoverer.py

Descoberta automática de datasets públicos de Brawl Stars no Roboflow Universe.
Suporta pesquisa via web scraping e lista curada de datasets conhecidos.

Uso:
    python training/roboflow_dataset_discoverer.py --search
    python training/roboflow_dataset_discoverer.py --list-known
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

# Adicionar root ao path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.class_registry import (
    CORE_CLASSES,
    EXTENDED_CLASSES,
    FULL_CLASSES,
    ROBOFLOW_TO_CANONICAL,
    VISUAL_CLASSES,
    get_canonical,
    get_class_id,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger("roboflow_discoverer")


@dataclass
class DatasetInfo:
    """Metadados de um dataset do Roboflow Universe."""

    workspace: str
    project: str
    name: str = ""
    url: str = ""
    type: str = "Object Detection"
    image_count: Optional[int] = None
    class_names: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    compatibility_score: float = 0.0
    source: str = "known"

    @property
    def download_url(self) -> str:
        """URL de download no formato YOLOv8."""
        return (
            f"https://universe.roboflow.com/ds/{self.workspace}/{self.project}"
            f"/download?format=yolov8"
        )

    @property
    def universe_url(self) -> str:
        """URL da página do dataset no Roboflow Universe."""
        return f"https://universe.roboflow.com/{self.workspace}/{self.project}"

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        img_str = f"{self.image_count} imgs" if self.image_count else "? imgs"
        classes_str = f" | {len(self.class_names)} classes" if self.class_names else ""
        return f"{self.workspace}/{self.project} ({self.name}) — {img_str} — score={self.compatibility_score:.2f}{classes_str}"


# ============================================================================
# DATASETS CONHECIDOS (curados da pesquisa no Roboflow Universe)
# ============================================================================

KNOWN_DATASETS: List[DatasetInfo] = [
    DatasetInfo(
        workspace="bloxxy",
        project="brawl-stars-dataset",
        name="Brawl Stars Dataset",
        type="Object Detection",
        image_count=2551,
        class_names=["Ball", "Enemy", "Friendly", "Gem", "Hot_Zone", "Me", "PP", "PP_Box", "Safe_Enemy", "Safe_Friendly"],
        tags=["yolov8", "gameplay"],
        source="known",
    ),
    DatasetInfo(
        workspace="yolo-ifkzx",
        project="brawl-stars-7ybb6",
        name="brawl stars",
        type="Object Detection",
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="brawl-83c6s",
        project="brawl-stars-h7ooo",
        name="Brawl Stars",
        type="Object Detection",
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="bsv",
        project="brawl-stars-vtq0k",
        name="Brawl Stars",
        type="Object Detection",
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="testing-roboflow-oni5k",
        project="brawl-stars-detection",
        name="Brawl Stars detection",
        type="Object Detection",
        image_count=357,
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="ivan-yordanov-cxrbb",
        project="brawl-stars-everything",
        name="brawl-stars-everything",
        type="Object Detection",
        class_names=[],
        tags=["yolo", "comprehensive"],
        source="known",
    ),
    DatasetInfo(
        workspace="ivan-yordanov-cxrbb",
        project="brawl-stars-everything-2",
        name="brawl-stars-everything-2",
        type="Object Detection",
        class_names=[],
        tags=["yolo", "comprehensive"],
        source="known",
    ),
    DatasetInfo(
        workspace="ivan-yordanov-cxrbb",
        project="brawl-stars-everything-biggest",
        name="brawl-stars-everything-biggest",
        type="Object Detection",
        class_names=[],
        tags=["yolo", "comprehensive"],
        source="known",
    ),
    DatasetInfo(
        workspace="yumdaniil",
        project="brawl-stars-ai-q0qkj",
        name="Brawl Stars ai",
        type="Object Detection",
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="ryan-pappa",
        project="brawl-stars-fkyb3",
        name="Brawl Stars",
        type="Object Detection",
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="ryan-pappa",
        project="brawl-stars-2",
        name="Brawl Stars 2",
        type="Object Detection",
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="brawlll",
        project="brawl-stars-bot-16ycf",
        name="brawl stars bot",
        type="Object Detection",
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="uitrial",
        project="brawl-stars-boxes",
        name="Brawl Stars Boxes",
        type="Object Detection",
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="uitrial",
        project="brawl-stars-character-detection",
        name="Brawl Stars Character Detection",
        type="Object Detection",
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="ai-training-wheiu",
        project="brawl-stars-bot-bnvxv",
        name="Brawl Stars Bot",
        type="Object Detection",
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="maskdetection-pfaqq",
        project="brawl-stars-bot-v3",
        name="Brawl Stars BOT V3",
        type="Object Detection",
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="brawl-stars-object-detection",
        project="brawl-stars-ai-03k0r",
        name="Brawl Stars AI",
        type="Object Detection",
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="michael-padilla",
        project="brawl-stars-object-detection2",
        name="brawl-stars-object-detection2",
        type="Object Detection",
        class_names=[],
        tags=["yolo"],
        source="known",
    ),
    DatasetInfo(
        workspace="bloxxy",
        project="brawl-stars-entity-detection-7r2r1",
        name="Brawl Stars Entity Detection",
        type="Object Detection",
        class_names=[],
        tags=["yolo", "entities"],
        source="known",
    ),
    DatasetInfo(
        workspace="bloxxy",
        project="brawl-stars-screen-environment",
        name="Brawl Stars Screen Environment",
        type="Object Detection",
        class_names=[],
        tags=["yolo", "environment"],
        source="known",
    ),
    DatasetInfo(
        workspace="nathan-yan",
        project="brawl-stars-buttons",
        name="Brawl stars buttons",
        type="Object Detection",
        class_names=[],
        tags=["yolo", "ui"],
        source="known",
    ),
    DatasetInfo(
        workspace="project-t9ziv",
        project="brawl-stars-bullets",
        name="brawl-stars-bullets",
        type="Instance Segmentation",
        class_names=[],
        tags=["segmentation", "projectiles"],
        source="known",
    ),
    DatasetInfo(
        workspace="kokskoksowy869-ws",
        project="brawl-stars-segment",
        name="brawl stars segment",
        type="Instance Segmentation",
        class_names=[],
        tags=["segmentation"],
        source="known",
    ),
]


# ============================================================================
# COMPATIBILITY SCORING
# ============================================================================

def compute_compatibility_score(
    dataset_classes: List[str],
    schema: str = "core",
) -> float:
    """
    Calcula um score de compatibilidade entre as classes de um dataset
    e o schema alvo do projeto.

    Score range: 0.0 a 1.0
    - 1.0 = todas as classes do dataset mapeiam perfeitamente para o schema
    - >0.5 = dataset cobre classes úteis mas tem extras ou omissões
    - 0.0 = nenhuma classe compatível
    """
    target_classes = set(VISUAL_CLASSES.get(schema, {}).values())
    if not target_classes or not dataset_classes:
        return 0.0

    canonical_dataset_classes = {get_canonical(c) for c in dataset_classes}
    matched = canonical_dataset_classes & target_classes

    if not matched:
        return 0.0

    # Precision: quantas classes do dataset são úteis?
    precision = len(matched) / len(canonical_dataset_classes) if canonical_dataset_classes else 0
    # Recall: quantas classes do schema são cobertas?
    recall = len(matched) / len(target_classes) if target_classes else 0

    # F1 score como métrica de compatibilidade
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 3)


def score_dataset(info: DatasetInfo, schema: str = "core") -> DatasetInfo:
    """Recomputa o compatibility_score de um DatasetInfo."""
    info.compatibility_score = compute_compatibility_score(info.class_names, schema)
    return info


def filter_compatible(
    datasets: List[DatasetInfo],
    min_score: float = 0.3,
    target_types: Optional[Set[str]] = None,
) -> List[DatasetInfo]:
    """Filtra datasets por score mínimo e tipo."""
    if target_types is None:
        target_types = {"Object Detection"}

    filtered = []
    for ds in datasets:
        if ds.type not in target_types:
            continue
        if ds.compatibility_score >= min_score:
            filtered.append(ds)

    return sorted(filtered, key=lambda x: x.compatibility_score, reverse=True)


# ============================================================================
# WEB SCRAPING SEARCH (Roboflow Universe)
# ============================================================================

def search_universe(
    query: str = "brawl stars",
    max_results: int = 50,
    timeout: int = 30,
) -> List[DatasetInfo]:
    """
    Pesquisa no Roboflow Universe via web scraping.

    Requer `requests` e `beautifulsoup4`:
        pip install requests beautifulsoup4

    Args:
        query: Termo de pesquisa
        max_results: Máximo de resultados a retornar
        timeout: Timeout HTTP em segundos

    Returns:
        Lista de DatasetInfo encontrados
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        logger.error(f"Missing dependency for web scraping: {e}")
        logger.error("Install with: pip install requests beautifulsoup4")
        return []

    search_url = f"https://universe.roboflow.com/search?q={query.replace(' ', '+')}"
    logger.info(f"Searching Roboflow Universe: {search_url}")

    try:
        response = requests.get(search_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to search Roboflow Universe: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results: List[DatasetInfo] = []

    # Extrair links de datasets do HTML
    seen = set()
    for link in soup.find_all("a", href=re.compile(r"^/[^/]+/brawl-stars")):
        href = link.get("href", "")
        if not href or href in seen:
            continue
        seen.add(href)

        # Parse workspace/project do href: /workspace/project
        parts = href.strip("/").split("/")
        if len(parts) != 2:
            continue

        workspace, project = parts
        name = link.get_text(strip=True) or project.replace("-", " ").title()

        info = DatasetInfo(
            workspace=workspace,
            project=project,
            name=name,
            url=f"https://universe.roboflow.com{href}",
            source="scraped",
        )
        results.append(info)

        if len(results) >= max_results:
            break

    logger.info(f"Found {len(results)} datasets via web scraping")
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Discover Brawl Stars datasets on Roboflow Universe")
    parser.add_argument("--search", action="store_true", help="Search via web scraping (requires requests+bs4)")
    parser.add_argument("--list-known", action="store_true", help="List known curated datasets")
    parser.add_argument("--schema", type=str, default="core", choices=["core", "extended", "full"],
                       help="Target schema for compatibility scoring")
    parser.add_argument("--min-score", type=float, default=0.0,
                       help="Minimum compatibility score filter")
    parser.add_argument("--json", type=str, default=None, help="Export results to JSON file")
    parser.add_argument("--max-results", type=int, default=50, help="Max results for web search")
    args = parser.parse_args()

    datasets: List[DatasetInfo] = []

    if args.search:
        datasets.extend(search_universe(max_results=args.max_results))

    if args.list_known or not args.search:
        datasets.extend(KNOWN_DATASETS)

    # Deduplicate by workspace/project
    seen: Dict[str, DatasetInfo] = {}
    for ds in datasets:
        key = f"{ds.workspace}/{ds.project}"
        if key not in seen:
            seen[key] = ds
    datasets = list(seen.values())

    # Score all datasets
    for ds in datasets:
        score_dataset(ds, schema=args.schema)

    # Filter
    datasets = filter_compatible(datasets, min_score=args.min_score)

    logger.info("=" * 70)
    logger.info(f"DATASETS COMPATIBLE WITH SCHEMA='{args.schema}' (min_score={args.min_score})")
    logger.info("=" * 70)
    for ds in datasets:
        logger.info(str(ds))

    if args.json:
        output_path = Path(args.json)
        with open(output_path, "w") as f:
            json.dump([ds.to_dict() for ds in datasets], f, indent=2)
        logger.info(f"Exported {len(datasets)} datasets to {output_path}")

    if not datasets:
        logger.warning("No compatible datasets found.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
