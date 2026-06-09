"""
core/model_registry.py

Model Registry com versionamento, warm-start e auto-update.

Gerencia múltiplos modelos YOLO/RL com:
- Versionamento semântico (major.minor.patch)
- Warm-start: carregar checkpoint de treinamento anterior
- Auto-update: detectar novos modelos no diretório
- Rollback: reverter para versão anterior
- Metadata: métricas, dataset, data de treino

Uso:
    registry = ModelRegistry()
    registry.register("yolo", "models/yolo_v1.2.pt", metrics={"mAP": 0.82})
    registry.set_active("yolo", "v1.2")
    model = registry.load_active("yolo")  # YOLO instance carregado
"""

import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Uma versão de um modelo."""
    name: str              # ex: "yolo", "rl_dqn"
    version: str           # ex: "v1.2.0"
    path: Path
    checksum: str          # SHA256 do arquivo
    created_at: float
    metrics: Dict[str, float] = None
    metadata: Dict[str, Any] = None
    parent_version: Optional[str] = None  # Para warm-start
    is_warm_start: bool = False

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.metadata is None:
            self.metadata = {}


class ModelRegistry:
    """
    Registry central de modelos com versionamento.
    """

    REGISTRY_FILE = "models/registry.json"
    CHECKPOINTS_DIR = "models/checkpoints"

    def __init__(self, base_dir: Path = Path(".")):
        self.base_dir = Path(base_dir)
        self.registry_path = self.base_dir / self.REGISTRY_FILE
        self.checkpoints_dir = self.base_dir / self.CHECKPOINTS_DIR
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self._models: Dict[str, Dict[str, ModelVersion]] = defaultdict(dict)
        self._active: Dict[str, str] = {}  # name -> version

        self._load_registry()
        logger.info("[MODEL_REGISTRY] Inicializado com %d famílias", len(self._models))

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def _load_registry(self):
        """Carrega registry do disco."""
        if not self.registry_path.exists():
            return
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, versions in data.get("models", {}).items():
                for ver_str, ver_data in versions.items():
                    self._models[name][ver_str] = ModelVersion(
                        name=name,
                        version=ver_str,
                        path=Path(ver_data["path"]),
                        checksum=ver_data.get("checksum", ""),
                        created_at=ver_data.get("created_at", 0.0),
                        metrics=ver_data.get("metrics", {}),
                        metadata=ver_data.get("metadata", {}),
                        parent_version=ver_data.get("parent_version"),
                        is_warm_start=ver_data.get("is_warm_start", False),
                    )
            self._active = data.get("active", {})
        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
            logger.warning("[MODEL_REGISTRY] Erro ao carregar: %s", e)

    def _save_registry(self):
        """Salva registry no disco."""
        try:
            data = {
                "models": {},
                "active": self._active,
                "saved_at": time.time(),
            }
            for name, versions in self._models.items():
                data["models"][name] = {}
                for ver_str, ver in versions.items():
                    data["models"][name][ver_str] = {
                        "path": str(ver.path),
                        "checksum": ver.checksum,
                        "created_at": ver.created_at,
                        "metrics": ver.metrics,
                        "metadata": ver.metadata,
                        "parent_version": ver.parent_version,
                        "is_warm_start": ver.is_warm_start,
                    }
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
            logger.error("[MODEL_REGISTRY] Erro ao salvar: %s", e)

    # ------------------------------------------------------------------
    # Registro
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        path: Path,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_version: Optional[str] = None,
    ) -> ModelVersion:
        """
        Registra um novo modelo.

        Se version não fornecido, auto-incrementa (v1.0, v1.1, etc.).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {path}")

        # Gerar checksum
        checksum = self._hash_file(path)

        # Auto-version
        if version is None:
            existing = self._models.get(name, {})
            if existing:
                # Incrementar patch do maior version
                latest = self._latest_version(name)
                version = self._bump_patch(latest)
            else:
                version = "v1.0.0"

        ver = ModelVersion(
            name=name,
            version=version,
            path=path,
            checksum=checksum,
            created_at=time.time(),
            metrics=metrics or {},
            metadata=metadata or {},
            parent_version=parent_version,
        )

        self._models[name][version] = ver
        self._save_registry()

        logger.info("[MODEL_REGISTRY] Registrado %s@%s (checksum=%s...)", name, version, checksum[:8])
        return ver

    def set_active(self, name: str, version: str) -> bool:
        """Define versão ativa para um modelo."""
        if version not in self._models.get(name, {}):
            logger.error("[MODEL_REGISTRY] Versão %s não existe para %s", version, name)
            return False
        self._active[name] = version
        self._save_registry()
        logger.info("[MODEL_REGISTRY] %s ativo → %s", name, version)
        return True

    def load_active(self, name: str):
        """
        Carrega e retorna instância do modelo ativo.
        Retorna None se não conseguir carregar.
        """
        version = self._active.get(name)
        if not version:
            logger.warning("[MODEL_REGISTRY] Nenhum modelo ativo para %s", name)
            return None

        ver = self._models.get(name, {}).get(version)
        if not ver:
            logger.error("[MODEL_REGISTRY] Versão ativa não encontrada: %s@%s", name, version)
            return None

        # Carregar modelo
        try:
            if name.startswith("yolo") or str(ver.path).endswith(".pt"):
                from ultralytics import YOLO
                model = YOLO(str(ver.path))
                logger.info("[MODEL_REGISTRY] YOLO carregado: %s@%s", name, version)
                return model
            elif name.startswith("rl") or str(ver.path).endswith(".pt"):
                import torch
                model = torch.load(str(ver.path), weights_only=False)
                logger.info("[MODEL_REGISTRY] RL model carregado: %s@%s", name, version)
                return model
            else:
                logger.warning("[MODEL_REGISTRY] Tipo de modelo desconhecido: %s", name)
                return None
        except (ImportError, ModuleNotFoundError, FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, OSError, IOError) as e:
            logger.error("[MODEL_REGISTRY] Erro ao carregar %s@%s: %s", name, version, e)
            return None

    # ------------------------------------------------------------------
    # Warm-start
    # ------------------------------------------------------------------

    def get_warm_start_path(self, name: str) -> Optional[Path]:
        """
        Retorna path do melhor checkpoint para warm-start.
        Útil para continuar treinamento de onde parou.
        """
        versions = self._models.get(name, {})
        if not versions:
            return None

        # Encontrar versão com melhor métrica (ex: mAP, win_rate)
        best_ver = None
        best_score = -float("inf")
        for ver in versions.values():
            score = ver.metrics.get("mAP", ver.metrics.get("win_rate", 0.0))
            if score > best_score:
                best_score = score
                best_ver = ver

        return best_ver.path if best_ver else None

    def save_training_checkpoint(self, name: str, epoch: int, model_state: Any, metrics: Dict):
        """Salva checkpoint durante treinamento."""
        import torch
        ckpt_path = self.checkpoints_dir / f"{name}_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model_state,
            "metrics": metrics,
            "timestamp": time.time(),
        }, ckpt_path)
        logger.debug("[MODEL_REGISTRY] Checkpoint salvo: %s", ckpt_path.name)
        return ckpt_path

    # ------------------------------------------------------------------
    # Auto-update
    # ------------------------------------------------------------------

    def scan_for_new_models(self, models_dir: Path = Path("models")) -> List[ModelVersion]:
        """
        Escaneia diretório por novos modelos .pt não registrados.
        Retorna lista de modelos recém-registrados.
        """
        new_models = []
        for pt_file in models_dir.glob("*.pt"):
            # Verificar se já registrado
            already_registered = any(
                str(ver.path) == str(pt_file)
                for versions in self._models.values()
                for ver in versions.values()
            )
            if not already_registered:
                # Inferir nome do arquivo
                name = pt_file.stem.split("_")[0] if "_" in pt_file.stem else "model"
                ver = self.register(name, pt_file)
                new_models.append(ver)

        if new_models:
            logger.info("[MODEL_REGISTRY] %d novos modelos encontrados", len(new_models))
        return new_models

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, name: str, steps: int = 1) -> bool:
        """
        Reverte para uma versão anterior.
        """
        versions = self._models.get(name, {})
        if not versions:
            return False

        sorted_vers = sorted(versions.keys(), key=lambda v: versions[v].created_at, reverse=True)
        current = self._active.get(name)
        if not current:
            return False

        try:
            idx = sorted_vers.index(current)
            target_idx = idx + steps
            if target_idx >= len(sorted_vers):
                logger.warning("[MODEL_REGISTRY] Rollback além do histórico disponível")
                return False

            target_ver = sorted_vers[target_idx]
            self.set_active(name, target_ver)
            logger.info("[MODEL_REGISTRY] Rollback %s: %s → %s", name, current, target_ver)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Lista todas as famílias de modelos."""
        return sorted(self._models.keys())

    def list_versions(self, name: str) -> List[str]:
        """Lista versões de um modelo."""
        return sorted(self._models.get(name, {}).keys())

    def get_version_info(self, name: str, version: str) -> Optional[Dict]:
        """Retorna info de uma versão específica."""
        ver = self._models.get(name, {}).get(version)
        if ver:
            return asdict(ver)
        return None

    def get_active_version(self, name: str) -> Optional[str]:
        """Retorna versão ativa de um modelo."""
        return self._active.get(name)

    def compare_versions(self, name: str, v1: str, v2: str) -> Dict[str, Any]:
        """Compara duas versões de um modelo."""
        ver1 = self._models.get(name, {}).get(v1)
        ver2 = self._models.get(name, {}).get(v2)
        if not ver1 or not ver2:
            return {"error": "version not found"}

        comparison = {
            "v1": v1,
            "v2": v2,
            "metrics_diff": {},
            "v1_better_in": [],
            "v2_better_in": [],
        }
        all_metrics = set(ver1.metrics.keys()) | set(ver2.metrics.keys())
        for metric in all_metrics:
            m1 = ver1.metrics.get(metric, 0.0)
            m2 = ver2.metrics.get(metric, 0.0)
            comparison["metrics_diff"][metric] = round(m2 - m1, 4)
            if m1 > m2:
                comparison["v1_better_in"].append(metric)
            elif m2 > m1:
                comparison["v2_better_in"].append(metric)

        return comparison

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Calcula SHA256 de um arquivo."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _latest_version(self, name: str) -> str:
        """Retorna a versão mais recente de um modelo."""
        versions = self._models.get(name, {})
        return max(versions.keys(), key=lambda v: versions[v].created_at) if versions else "v0.0.0"

    @staticmethod
    def _bump_patch(version: str) -> str:
        """Incrementa patch de uma versão semântica."""
        parts = version.lstrip("v").split(".")
        if len(parts) == 3:
            major, minor, patch = parts
            return f"v{major}.{minor}.{int(patch) + 1}"
        return "v1.0.0"
