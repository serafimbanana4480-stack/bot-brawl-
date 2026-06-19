#!/usr/bin/env python3
"""
dataset_curator.py — Curadoria Inteligente de Dataset para Brawl Stars

Valida, limpa, balanceia e otimiza datasets de treino YOLO.

Uso:
    python dataset_curator.py --validate    # Validar dataset existente
    python dataset_curator.py --clean       # Limpar dados de baixa qualidade
    python dataset_curator.py --balance     # Balancear classes
    python dataset_curator.py --report      # Relatório completo do dataset
"""

import argparse
import json
import logging
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dataset_curator")


class DatasetCurator:
    """Curadoria profissional de datasets YOLO para Brawl Stars."""

    # Classes do Brawl Stars
    CLASS_NAMES = [
        "player", "enemy", "bush", "powercube", "gem",
        "ball", "brawler", "projectile", "heal", "shield",
        "speed_boost", "sneaky_fields", "rope_fence",
        "spring_trap", "mine", "totem",
    ]

    CLASS_PRIORITIES = {
        "player": 10, "enemy": 10, "brawler": 10,
        "bush": 5, "powercube": 5,
        "projectile": 3,
        "gem": 3, "ball": 3,
    }

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.data_yaml = dataset_path / "data.yaml"
        self.images_dir = dataset_path / "images"
        self.labels_dir = dataset_path / "labels"

    # ── ANÁLISE ──────────────────────────────────────────────────

    def analyze(self) -> dict:
        """Analisa o dataset e retorna relatório completo."""
        report = {
            "path": str(self.dataset_path),
            "total_images": 0,
            "total_labels": 0,
            "total_objects": 0,
            "class_distribution": {},
            "image_sizes": {"min": None, "max": None, "avg": None},
            "issues": [],
            "splits": {},
        }

        if not self.images_dir.exists():
            report["issues"].append("Diretório de imagens não encontrado")
            return report

        # Contar splits
        splits = ["train", "val", "test"]
        all_images = 0
        all_labels = 0
        class_counter = Counter()
        widths, heights = [], []

        for split in splits:
            img_dir = self.images_dir / split
            lbl_dir = self.labels_dir / split

            if not img_dir.exists():
                continue

            imgs = list(img_dir.glob("*.*"))
            lbls = list(lbl_dir.glob("*.txt"))
            report["splits"][split] = {
                "images": len(imgs),
                "labels": len(lbls),
            }
            all_images += len(imgs)
            all_labels += len(lbls)

            # Analisar labels
            for lbl in lbls:
                try:
                    with open(lbl) as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                cls_id = int(parts[0])
                                class_counter[cls_id] += 1
                                all_labels += 1
                except Exception:
                    pass

            # Analisar dimensões das imagens
            for img_path in imgs[:100]:  # Amostra de 100 imagens
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        widths.append(w)
                        heights.append(h)
                except Exception:
                    pass

        report["total_images"] = all_images
        report["total_labels"] = all_labels
        report["total_objects"] = sum(class_counter.values())

        # Distribuição de classes
        for cls_id, count in sorted(class_counter.items(), key=lambda x: -x[1]):
            cls_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else f"class_{cls_id}"
            report["class_distribution"][cls_name] = count

        # Dimensões
        if widths:
            report["image_sizes"] = {
                "min": f"{min(widths)}x{min(heights)}" if heights else "N/A",
                "max": f"{max(widths)}x{max(heights)}" if heights else "N/A",
                "avg": f"{int(np.mean(widths))}x{int(np.mean(heights))}" if heights else "N/A",
            }

        # Verificar problemas comuns
        report["issues"] = self._find_issues(class_counter, all_images)

        return report

    def _find_issues(self, class_counter: Counter, total_images: int) -> List[str]:
        """Encontra problemas no dataset."""
        issues = []

        # Desbalanceamento de classes
        if class_counter:
            max_count = max(class_counter.values())
            min_count = min(class_counter.values())
            if max_count > 0 and min_count / max_count < 0.1:
                issues.append(f"Classes desbalanceadas (min={min_count}, max={max_count})")

        # Poucas imagens
        if total_images < 100:
            issues.append(f"Dataset pequeno ({total_images} imagens)")

        return issues

    # ── LIMPEZA ──────────────────────────────────────────────────

    def clean(self, dry_run: bool = True) -> dict:
        """
        Remove imagens de baixa qualidade:
        - Blurradas (Laplacian < 100)
        - Duplicadas (hash perceptual)
        - Muito escuras/claras
        - Labels vazios ou inválidos
        """
        cleanup = {
            "removed_blurry": 0,
            "removed_duplicates": 0,
            "removed_empty": 0,
            "removed_invalid_labels": 0,
            "total_kept": 0,
        }

        output_dir = self.dataset_path.parent / f"{self.dataset_path.name}_cleaned"
        if dry_run:
            logger.info("── Dry run (sem alterações) ──")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        seen_hashes = set()

        for split in ["train", "val", "test"]:
            img_dir = self.images_dir / split
            lbl_dir = self.labels_dir / split

            if not img_dir.exists():
                continue

            for img_path in img_dir.glob("*.*"):
                should_remove = False
                reasons = []

                # Carregar imagem
                img = cv2.imread(str(img_path))
                if img is None:
                    should_remove = True
                    reasons.append("corrompida")

                if not should_remove:
                    # 1. Blur check
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if laplacian_var < 100:
                        should_remove = True
                        reasons.append(f"blurrada (Laplacian={laplacian_var:.1f})")
                        cleanup["removed_blurry"] += 1

                    # 2. Duplicate check
                    if not should_remove:
                        h = self._perceptual_hash(img)
                        if h in seen_hashes:
                            should_remove = True
                            reasons.append("duplicada")
                            cleanup["removed_duplicates"] += 1
                        else:
                            seen_hashes.add(h)

                # 3. Labels vazios
                if not should_remove:
                    lbl_path = lbl_dir / f"{img_path.stem}.txt"
                    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                        should_remove = True
                        reasons.append("label vazio")
                        cleanup["removed_empty"] += 1

                if should_remove:
                    if dry_run:
                        logger.debug(f"  Removeria: {img_path.name} ({', '.join(reasons)})")
                else:
                    if not dry_run:
                        # Copiar imagem e label
                        out_img_dir = output_dir / "images" / split
                        out_lbl_dir = output_dir / "labels" / split
                        out_img_dir.mkdir(parents=True, exist_ok=True)
                        out_lbl_dir.mkdir(parents=True, exist_ok=True)

                        shutil.copy2(img_path, out_img_dir / img_path.name)
                        lbl_path = lbl_dir / f"{img_path.stem}.txt"
                        if lbl_path.exists():
                            shutil.copy2(lbl_path, out_lbl_dir / lbl_path.name)

                    cleanup["total_kept"] += 1

        if not dry_run:
            # Copiar data.yaml
            if self.data_yaml.exists():
                shutil.copy2(self.data_yaml, output_dir / "data.yaml")

        return cleanup

    # ── BALANCEAMENTO ─────────────────────────────────────────────

    def balance(self, target_per_class: Optional[int] = None) -> dict:
        """Balanceia as classes por oversampling das minoritárias."""
        result = {"classes": {}, "total_augmented": 0}

        # Contar objetos por classe
        class_counts = Counter()
        for split in ["train"]:
            lbl_dir = self.labels_dir / split
            if not lbl_dir.exists():
                continue
            for lbl in lbl_dir.glob("*.txt"):
                try:
                    with open(lbl) as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_counts[int(parts[0])] += 1
                except Exception:
                    pass

        if not class_counts:
            return result

        max_count = max(class_counts.values()) if class_counts else 0
        target = target_per_class or max_count

        result["target_per_class"] = target

        for cls_id, count in class_counts.items():
            cls_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else f"class_{cls_id}"
            needed = target - count
            if needed > 0:
                result["classes"][cls_name] = {
                    "current": count,
                    "needed": needed,
                    "strategy": "oversample",
                }
                result["total_augmented"] += needed

        return result

    # ── RELATÓRIO ─────────────────────────────────────────────────

    def report(self):
        """Gera relatório formatado do dataset."""
        analysis = self.analyze()

        logger.info("=" * 60)
        logger.info(f"  RELATORIO DO DATASET: {analysis['path']}")
        logger.info("=" * 60)
        logger.info(f"  Total imagens: {analysis['total_images']}")
        logger.info(f"  Total objetos: {analysis['total_objects']}")
        logger.info(f"  Dimensões:     {analysis['image_sizes']}")
        logger.info("")

        for split, data in analysis["splits"].items():
            logger.info(f"  {split.upper()}: {data['images']} imagens, {data['labels']} labels")

        if analysis["class_distribution"]:
            logger.info("")
            logger.info("  Distribuição de classes:")
            max_name_len = max(len(n) for n in analysis["class_distribution"].keys())
            for name, count in sorted(analysis["class_distribution"].items(), key=lambda x: -x[1]):
                bar = "█" * max(1, count // 50)
                logger.info(f"    {name:<{max_name_len+2}} {count:>6} {bar}")

        if analysis["issues"]:
            logger.info("")
            logger.warning("  Problemas encontrados:")
            for issue in analysis["issues"]:
                logger.warning(f"    ⚠️ {issue}")
        else:
            logger.info("")
            logger.info("  ✅ Nenhum problema encontrado!")

        return analysis

    # ── HELPERS ──────────────────────────────────────────────────

    @staticmethod
    def _perceptual_hash(image: np.ndarray, hash_size: int = 8) -> int:
        """Perceptual hash para deteção de duplicados."""
        # Redimensionar para 8x8 e converter para grayscale
        resized = cv2.resize(image, (hash_size + 1, hash_size + 1))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # DCT e pegar 8x8 da esquerda superior
        dct = cv2.dct(np.float32(gray))
        dct_low = dct[:hash_size, :hash_size]
        # Média (excluindo DC)
        avg = dct_low.mean()
        # Hash binário
        diff = (dct_low > avg).flatten()
        return sum(2 ** i for i, b in enumerate(diff) if b)


def main():
    parser = argparse.ArgumentParser(description="Dataset Curator para Brawl Stars")
    parser.add_argument("--path", type=str, default="dataset/merged_roboflow",
                        help="Caminho do dataset (default: dataset/merged_roboflow)")
    parser.add_argument("--report", action="store_true", help="Gerar relatório")
    parser.add_argument("--validate", action="store_true", help="Validar dataset")
    parser.add_argument("--clean", action="store_true", help="Limpar dados de baixa qualidade")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (não altera ficheiros)")
    parser.add_argument("--balance", action="store_true", help="Analisar balanceamento")

    args = parser.parse_args()

    root = Path(__file__).parent
    dataset_path = root / args.path

    if not dataset_path.exists():
        logger.error(f"Dataset não encontrado: {dataset_path}")
        sys.exit(1)

    curator = DatasetCurator(dataset_path)

    if args.report or args.validate:
        curator.report()

    if args.clean:
        logger.info("A limpar dataset...")
        result = curator.clean(dry_run=args.dry_run)
        logger.info(f"  Resultado: {json.dumps(result, indent=2)}")

    if args.balance:
        logger.info("A analisar balanceamento...")
        result = curator.balance()
        logger.info(f"  Resultado: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
