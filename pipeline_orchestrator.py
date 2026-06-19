#!/usr/bin/env python3
"""
pipeline_orchestrator.py — Pipeline End-to-End Soberana Omega

Fluxo completo: Dataset → Treino YOLO → Validação → TensorRT → Deploy → Execução

Uso:
    python pipeline_orchestrator.py --full          # Pipeline completa
    python pipeline_orchestrator.py --train         # Só treino
    python pipeline_orchestrator.py --validate      # Só validação dataset
    python pipeline_orchestrator.py --tensorrt      # Só converter para TensorRT
    python pipeline_orchestrator.py --deploy        # Só deploy do modelo
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


class PipelineOrchestrator:
    """
    Orquestrador do pipeline completo:
    1. Dataset → validação + limpeza + balanceamento
    2. Treino YOLO (com curriculum, augmentação avançada)
    3. Validação do modelo treinado (mAP, F1, confusão)
    4. Export para TensorRT (inferência otimizada na GPU)
    5. Deploy (cópia para models/, atualização de config)
    6. Execução (run com o modelo novo)
    """

    ROOT = Path(__file__).parent
    DATASET_DIR = ROOT / "dataset"
    MODELS_DIR = ROOT / "models"
    TRAINING_DIR = ROOT / "training"
    RUNS_DIR = ROOT / "runs"
    DEPLOY_DIR = MODELS_DIR / "deploy"

    def __init__(self, args):
        self.args = args
        self.start_time = time.time()

    # ── ETAPA 0: Diagnóstico do sistema ──────────────────────────

    def diagnose(self) -> bool:
        """Verifica se GPU, datasets e dependências estão OK."""
        logger.info("=" * 60)
        logger.info("ETAPA 0: Diagnóstico do sistema")
        logger.info("=" * 60)

        checks = []

        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
                logger.info(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
                checks.append(True)
            else:
                logger.warning("  GPU: NÃO DISPONÍVEL (CPU mode)")
                checks.append(False)
        except Exception as e:
            logger.warning(f"  GPU: erro ao verificar — {e}")
            checks.append(False)

        # Torch + Ultralytics
        try:
            import ultralytics
            logger.info(f"  Ultralytics: {ultralytics.__version__}")
            checks.append(True)
        except ImportError:
            logger.error("  Ultralytics NÃO instalado!")
            checks.append(False)

        # Dataset
        ds_path = self.DATASET_DIR / "merged_roboflow" / "data.yaml"
        if ds_path.exists():
            logger.info(f"  Dataset merged_roboflow: OK")
            checks.append(True)
        else:
            logger.warning("  Dataset merged_roboflow: não encontrado")
            checks.append(False)

        # Modelo base
        base_model = self.MODELS_DIR / "yolo11n.pt"
        if base_model.exists():
            logger.info(f"  Modelo base yolo11n: OK")
            checks.append(True)
        else:
            logger.warning("  Modelo base yolo11n: não encontrado")
            checks.append(False)

        ok = all(checks)
        logger.info(f"  Resultado: {'✅ PRONTO' if ok else '⚠️ PARCIAL'}")
        return ok

    # ── ETAPA 1: Curadoria do Dataset ────────────────────────────

    def curate_dataset(self):
        """Valida, limpa e balanceia os datasets."""
        logger.info("=" * 60)
        logger.info("ETAPA 1: Curadoria do Dataset")
        logger.info("=" * 60)

        from training.enhanced_training_pipeline import DataCurator

        curator = DataCurator(min_quality=0.5)

        sources = [
            ("merged_roboflow", "curated_roboflow"),
            ("merged_v2", "curated_v2"),
        ]

        for src_name, dst_name in sources:
            src = self.DATASET_DIR / src_name
            dst = self.DATASET_DIR / dst_name

            if not src.exists():
                logger.warning(f"  Dataset {src_name}: não encontrado, a saltar")
                continue

            logger.info(f"  A curar {src_name} → {dst_name}...")
            try:
                report = curator.curate_dataset(src, dst)
                logger.info(f"  Curadoria concluída: {report}")
            except Exception as e:
                logger.error(f"  Erro na curadoria: {e}")

    # ── ETAPA 2: Treino YOLO ────────────────────────────────────

    def train_yolo(self):
        """Treina modelo YOLO usando o pipeline otimizado."""
        logger.info("=" * 60)
        logger.info("ETAPA 2: Treino YOLO + Curriculum")
        logger.info("=" * 60)

        epochs = self.args.epochs or 150
        batch = self.args.batch or 8
        imgsz = self.args.imgsz or 640
        model_size = self.args.model_size or "n"  # n, s, m, l, x

        # Usar yolo11n como base (mais rápido, melhor que yolo8)
        model_path = self.MODELS_DIR / f"yolo11{model_size}.pt"
        if not model_path.exists():
            model_path = self.MODELS_DIR / "yolo11n.pt"
        if not model_path.exists():
            model_path = "yolo11n.pt"  # download automático

        # Dataset config
        data_yaml = self.DATASET_DIR / "merged_roboflow" / "data.yaml"
        if not data_yaml.exists():
            logger.error("Dataset YAML não encontrado!")
            return

        logger.info(f"  Modelo base: {model_path}")
        logger.info(f"  Dataset: {data_yaml}")
        logger.info(f"  Épocas: {epochs} | Batch: {batch} | ImgSize: {imgsz}")

        from ultralytics import YOLO

        model = YOLO(str(model_path))

        # Curriculum training: começar com 640, depois 1280
        for phase, size, ep in [
            ("FASE 1 — Baixa resolução", 416, max(epochs // 3, 30)),
            ("FASE 2 — Resolução alvo", imgsz, epochs // 2),
            ("FASE 3 — Alta resolução (fine-tune)", max(imgsz, 640) * 2, max(epochs // 6, 10)),
        ]:
            logger.info(f"  {phase}: {size}px, {ep} épocas")
            model.train(
                data=str(data_yaml),
                epochs=ep,
                imgsz=size,
                batch=batch,
                device="cuda" if self._has_gpu() else "cpu",
                workers=4,
                patience=20,
                lr0=0.001,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                warmup_momentum=0.8,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.1,
                copy_paste=0.1,
                project=str(self.RUNS_DIR),
                name=f"brawl_train_{datetime.now().strftime('%Y%m%d_%H%M')}",
                exist_ok=True,
                val=True,
                save=True,
                save_period=10,
            )

        logger.info("  ✅ Treino concluído!")

    # ── ETAPA 3: Validação do modelo ────────────────────────────

    def validate_model(self):
        """Valida o modelo treinado e exporta métricas."""
        logger.info("=" * 60)
        logger.info("ETAPA 3: Validação do Modelo")
        logger.info("=" * 60)

        best_pt = self._find_best_model()
        if not best_pt:
            logger.error("Nenhum modelo treinado encontrado!")
            return

        logger.info(f"  Modelo: {best_pt}")
        data_yaml = self.DATASET_DIR / "merged_roboflow" / "data.yaml"
        if not data_yaml.exists():
            logger.error("Dataset YAML não encontrado!")
            return

        from ultralytics import YOLO
        model = YOLO(str(best_pt))
        metrics = model.val(data=str(data_yaml), device="cuda" if self._has_gpu() else "cpu")

        logger.info(f"  mAP@0.5:  {metrics.box.map50:.4f}")
        logger.info(f"  mAP@0.5:.95: {metrics.box.map:.4f}")
        logger.info(f"  Precisão: {metrics.box.mp:.4f}")
        logger.info(f"  Recall:   {metrics.box.mr:.4f}")

        return metrics

    # ── ETAPA 4: Export para TensorRT ───────────────────────────

    def export_tensorrt(self):
        """Converte o modelo para TensorRT (inferência GPU otimizada)."""
        logger.info("=" * 60)
        logger.info("ETAPA 4: Export TensorRT")
        logger.info("=" * 60)

        if not self._has_gpu():
            logger.warning("  GPU não disponível. TensorRT não é possível.")
            return

        best_pt = self._find_best_model()
        if not best_pt:
            logger.error("Nenhum modelo encontrado!")
            return

        logger.info(f"  Modelo PT: {best_pt}")
        tensorrt_path = best_pt.with_suffix(".engine")
        logger.info(f"  Exportando para TensorRT: {tensorrt_path}")

        from ultralytics import YOLO
        model = YOLO(str(best_pt))
        model.export(
            format="engine",
            device="cuda",
            imgsz=640,
            half=True,  # FP16 (2x mais rápido, perda mínima)
            workspace=6,  # 6GB de VRAM
            int8=False,
            batch=1,
        )

        logger.info(f"  ✅ TensorRT exportado: {tensorrt_path}")

    # ── ETAPA 5: Deploy ────────────────────────────────────────

    def deploy(self):
        """Copia o melhor modelo para o diretório de deploy."""
        logger.info("=" * 60)
        logger.info("ETAPA 5: Deploy")
        logger.info("=" * 60)

        best_pt = self._find_best_model()
        if not best_pt:
            logger.error("Nenhum modelo encontrado!")
            return

        self.DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

        import shutil
        deploy_path = self.DEPLOY_DIR / "brawlstars_deploy.pt"
        shutil.copy2(best_pt, deploy_path)
        logger.info(f"  Modelo copiado para: {deploy_path}")

        # Se TensorRT existir, copiar também
        tensorrt_path = best_pt.with_suffix(".engine")
        if tensorrt_path.exists():
            deploy_trt = self.DEPLOY_DIR / "brawlstars_deploy.engine"
            shutil.copy2(tensorrt_path, deploy_trt)
            logger.info(f"  TensorRT copiado para: {deploy_trt}")

        # Atualizar config.json para usar o novo modelo
        config_path = self.ROOT / "config.json"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
                if "vision" not in config:
                    config["vision"] = {}
                config["vision"]["model_path"] = str(deploy_path)
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                logger.info(f"  config.json atualizado!")
            except Exception as e:
                logger.warning(f"  Erro a atualizar config.json: {e}")

        logger.info("  ✅ Deploy concluído!")

    # ── ETAPA 6: Execução ──────────────────────────────────────

    def run_bot(self):
        """Executa o bot com o modelo deployado."""
        logger.info("=" * 60)
        logger.info("ETAPA 6: Execução do Bot")
        logger.info("=" * 60)

        # Verificar deploy
        deploy_model = self.DEPLOY_DIR / "brawlstars_deploy.pt"
        if not deploy_model.exists():
            logger.warning("  Nenhum modelo deployado. A correr com modelo existente.")
        else:
            logger.info(f"  Modelo: {deploy_model}")

        # Iniciar o bot
        try:
            from wrapper import PylaAIEnhanced
            bot = PylaAIEnhanced()
            if bot.setup():
                logger.info("  Bot iniciado!")
                bot.start()
            else:
                logger.error("  Falha no setup do bot")
        except Exception as e:
            logger.error(f"  Erro ao iniciar bot: {e}")

    # ── HELPERS ─────────────────────────────────────────────────

    def _has_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _find_best_model(self) -> Optional[Path]:
        """Encontra o melhor modelo treinado (mais recente ou best.pt)."""
        # Priority 1: runs/detect/train/weights/best.pt (treino mais recente)
        runs_dir = self.RUNS_DIR / "detect"
        if runs_dir.exists():
            train_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for td in train_dirs:
                best = td / "weights" / "best.pt"
                if best.exists():
                    return best

        # Priority 2: models/brawlstars_yolov8_gpu.pt (modelo principal)
        main_model = self.MODELS_DIR / "brawlstars_yolov8_gpu.pt"
        if main_model.exists():
            return main_model

        # Priority 3: models/best.pt
        fallback = self.MODELS_DIR / "best.pt"
        if fallback.exists():
            return fallback

        return None

    # ── EXECUÇÃO PRINCIPAL ──────────────────────────────────────

    def run(self):
        """Executa o pipeline completo."""
        logger.info("=" * 60)
        logger.info("🚀 Soberana Omega — Pipeline End-to-End")
        logger.info(f"   Início: {datetime.now().isoformat()}")
        logger.info("=" * 60)

        diagnose_ok = self.diagnose()

        if self.args.diagnose_only:
            return

        if self.args.validate_dataset or self.args.full:
            self.curate_dataset()

        if self.args.train or self.args.full:
            if not diagnose_ok and not self.args.force:
                logger.error("Diagnóstico falhou. Use --force para ignorar.")
                return
            self.train_yolo()

        if self.args.validate_model or self.args.full:
            self.validate_model()

        if self.args.tensorrt or self.args.full:
            self.export_tensorrt()

        if self.args.deploy or self.args.full:
            self.deploy()

        if self.args.run or self.args.full:
            self.run_bot()

        elapsed = time.time() - self.start_time
        logger.info("=" * 60)
        logger.info(f"✅ Pipeline concluído em {elapsed:.0f}s")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Soberana Omega — Pipeline End-to-End")

    # Ações
    parser.add_argument("--full", action="store_true", help="Pipeline completa")
    parser.add_argument("--diagnose", action="store_true", dest="diagnose_only", help="Só diagnóstico")
    parser.add_argument("--validate-dataset", action="store_true", help="Só curadoria de dataset")
    parser.add_argument("--train", action="store_true", help="Só treino YOLO")
    parser.add_argument("--validate-model", action="store_true", help="Só validação de modelo")
    parser.add_argument("--tensorrt", action="store_true", help="Só export TensorRT")
    parser.add_argument("--deploy", action="store_true", help="Só deploy")
    parser.add_argument("--run", action="store_true", help="Só execução do bot")

    # Parâmetros de treino
    parser.add_argument("--epochs", type=int, default=150, help="Épocas máximas (default: 150)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                        help="Tamanho do modelo YOLO (default: n)")
    parser.add_argument("--force", action="store_true", help="Ignorar falhas de diagnóstico")

    args = parser.parse_args()

    # Se nenhuma ação, mostrar ajuda
    if not any([args.full, args.diagnose_only, args.validate_dataset,
                args.train, args.validate_model, args.tensorrt,
                args.deploy, args.run]):
        parser.print_help()
        return

    pipeline = PipelineOrchestrator(args)
    pipeline.run()


if __name__ == "__main__":
    main()
