"""
core/model_auto_updater.py

Sistema de Auto-Update de Modelos.

Monitora diretórios de modelos, detecta novas versões e atualiza
automaticamente o modelo ativo se a nova versão for melhor.

Features:
- Watchdog para detectar novos arquivos .pt
- Comparação de métricas entre versões
- Rollback automático se nova versão degradar performance
- Notificações via AlertSystem
- Warm-start automático ao trocar versão

Uso:
    updater = ModelAutoUpdater(registry, alert_system)
    updater.start_watching("models/")  # thread de monitoramento
    updater.check_for_updates()          # verificação manual
"""

import logging
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    Observer = None
    FileSystemEventHandler = None
    WATCHDOG_AVAILABLE = False

logger = logging.getLogger(__name__)


if WATCHDOG_AVAILABLE:
    class _ModelFileHandler(FileSystemEventHandler):
        """Handler de eventos do watchdog para arquivos .pt."""

        def __init__(self, callback: Callable[[Path], None]):
            self.callback = callback

        def on_created(self, event):
            if not event.is_directory and event.src_path.endswith(".pt"):
                self.callback(Path(event.src_path))

        def on_modified(self, event):
            if not event.is_directory and event.src_path.endswith(".pt"):
                self.callback(Path(event.src_path))
else:
    _ModelFileHandler = None


class ModelAutoUpdater:
    """
    Monitora e auto-atualiza modelos.
    """

    def __init__(
        self,
        model_registry: Any,
        alert_system: Optional[Any] = None,
        min_improvement: float = 0.02,  # 2% mínimo para trocar
        rollback_on_degradation: bool = True,
    ):
        self.registry = model_registry
        self.alert_system = alert_system
        self.min_improvement = min_improvement
        self.rollback_on_degradation = rollback_on_degradation

        self._observer: Optional[Observer] = None
        self._watch_path: Optional[Path] = None
        self._last_scan: float = 0.0
        self._scan_interval = 60.0  # segundos
        self._lock = threading.Lock()
        self._performance_log: Dict[str, list] = {}  # name -> [scores]

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    def start_watching(self, models_dir: Path):
        """Inicia watchdog para monitorar diretório de modelos."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("[AUTO_UPDATE] watchdog não instalado. Usando check_for_updates() manual.")
            return

        models_dir = Path(models_dir)
        if not models_dir.exists():
            logger.warning("[AUTO_UPDATE] Diretório não existe: %s", models_dir)
            return

        self._watch_path = models_dir
        handler = _ModelFileHandler(self._on_new_model)
        self._observer = Observer()
        self._observer.schedule(handler, str(models_dir), recursive=False)
        self._observer.start()
        logger.info("[AUTO_UPDATE] Monitorando: %s", models_dir)

    def stop_watching(self):
        """Para o watchdog."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("[AUTO_UPDATE] Monitoramento parado")

    def _on_new_model(self, path: Path):
        """Callback chamado quando novo .pt é detectado."""
        logger.info("[AUTO_UPDATE] Novo modelo detectado: %s", path.name)
        # Aguardar 2s para o arquivo ser completamente escrito
        time.sleep(2)
        self._evaluate_and_update(path)

    # ------------------------------------------------------------------
    # Verificação Manual
    # ------------------------------------------------------------------

    def check_for_updates(self):
        """Verifica manualmente por modelos novos."""
        if not self._watch_path:
            logger.warning("[AUTO_UPDATE] Nenhum diretório configurado")
            return

        now = time.time()
        if now - self._last_scan < self._scan_interval:
            return
        self._last_scan = now

        new_models = self.registry.scan_for_new_models(self._watch_path)
        for ver in new_models:
            self._evaluate_and_update(ver.path)

    # ------------------------------------------------------------------
    # Avaliação e Troca
    # ------------------------------------------------------------------

    def _evaluate_and_update(self, path: Path):
        """Avalia novo modelo e decide se troca a versão ativa."""
        with self._lock:
            # Determinar nome do modelo a partir do path
            name = path.stem.split("_")[0] if "_" in path.stem else "model"

            # Verificar se já está ativo
            current_ver = self.registry.get_active_version(name)
            if not current_ver:
                # Primeira vez — ativar diretamente
                self._activate_new_version(name, path)
                return

            # Comparar métricas
            comparison = self.registry.compare_versions(name, current_ver, self._latest_version_for_path(name, path))
            if "error" in comparison:
                return

            # Verificar se nova versão é melhor em alguma métrica chave
            new_better = comparison.get("v2_better_in", [])
            old_better = comparison.get("v1_better_in", [])

            if new_better and not old_better:
                logger.info(
                    "[AUTO_UPDATE] Nova versão melhor em: %s. Ativando...",
                    ", ".join(new_better)
                )
                self._activate_new_version(name, path)

                if self.alert_system:
                    self.alert_system.create_alert(
                        severity="info",
                        message=f"Modelo {name} auto-atualizado para nova versão (melhor em: {', '.join(new_better)})",
                        category="model_update",
                        source="auto_updater",
                    )
            else:
                logger.debug("[AUTO_UPDATE] Nova versão não é melhor, mantendo %s", current_ver)

    def _activate_new_version(self, name: str, path: Path):
        """Ativa nova versão e registra performance para monitoramento."""
        try:
            ver = self.registry.register(name, path)
            self.registry.set_active(name, ver.version)
            self._performance_log[name] = []  # Reset log para nova versão
        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            logger.error("[AUTO_UPDATE] Erro ao ativar %s: %s", name, e)

    def _latest_version_for_path(self, name: str, path: Path) -> str:
        """Encontra versão correspondente a um path no registry."""
        for ver_str, ver in self.registry._models.get(name, {}).items():
            if ver.path == path:
                return ver_str
        return ""

    # ------------------------------------------------------------------
    # Monitoramento de Performance Pós-Update
    # ------------------------------------------------------------------

    def report_performance(self, model_name: str, score: float):
        """
        Reporta score de performance do modelo ativo.
        Se degradar muito, faz rollback.
        """
        if model_name not in self._performance_log:
            self._performance_log[model_name] = []

        self._performance_log[model_name].append(score)
        if len(self._performance_log[model_name]) > 20:
            self._performance_log[model_name] = self._performance_log[model_name][-20:]

        # Verificar se performance degradou significativamente
        if len(self._performance_log[model_name]) >= 10:
            recent_avg = sum(self._performance_log[model_name][-5:]) / 5
            baseline = sum(self._performance_log[model_name][:5]) / 5

            if baseline > 0 and recent_avg < baseline * (1 - self.min_improvement * 3):
                logger.warning(
                    "[AUTO_UPDATE] Performance de %s degradou (%.2f -> %.2f). Rollback?",
                    model_name, baseline, recent_avg
                )
                if self.rollback_on_degradation:
                    success = self.registry.rollback(model_name, steps=1)
                    if success and self.alert_system:
                        self.alert_system.create_alert(
                            severity="warning",
                            message=f"Performance de {model_name} degradou. Rollback executado.",
                            category="model_rollback",
                            source="auto_updater",
                        )

    def get_status(self) -> Dict[str, Any]:
        """Status do auto-updater."""
        return {
            "watching": self._observer is not None,
            "watch_path": str(self._watch_path) if self._watch_path else None,
            "min_improvement": self.min_improvement,
            "rollback_enabled": self.rollback_on_degradation,
            "performance_logs": {
                k: {"count": len(v), "avg": round(sum(v)/len(v), 3) if v else 0}
                for k, v in self._performance_log.items()
            },
        }
