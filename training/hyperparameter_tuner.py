"""
hyperparameter_tuner.py

Lightweight automatic hyperparameter tuning for Ultralytics training runs.

The goal here is pragmatic:
- keep the search space small and inspectable
- work without external tuning libraries
- persist trial history for reproducibility
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import json
import logging
import random

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TuningCandidate:
    """One hyperparameter configuration."""

    lr0: float
    lrf: float
    momentum: float
    weight_decay: float
    warmup_epochs: float
    hsv_h: float
    hsv_s: float
    hsv_v: float
    mosaic: float
    mixup: float
    copy_paste: float
    scale: float
    translate: float
    degrees: float
    freeze: int = 0
    cos_lr: bool = True

    def to_ultralytics_kwargs(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class TuningTrialResult:
    candidate: TuningCandidate
    metric_name: str
    metric_value: float
    notes: str = ""


class HyperparameterTuner:
    """Generate and rank training candidates."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _choice(self, values: Sequence):
        return self.rng.choice(list(values))

    def suggest(self, n_trials: int = 8) -> List[TuningCandidate]:
        """Return a small list of candidates for trial runs."""
        candidates = []
        for _ in range(n_trials):
            candidates.append(
                TuningCandidate(
                    lr0=self.rng.choice([0.0005, 0.001, 0.002, 0.003]),
                    lrf=self.rng.choice([0.01, 0.05, 0.1]),
                    momentum=self.rng.choice([0.90, 0.937, 0.95]),
                    weight_decay=self.rng.choice([0.0003, 0.0005, 0.001]),
                    warmup_epochs=self.rng.choice([1.0, 2.0, 3.0, 5.0]),
                    hsv_h=self.rng.choice([0.01, 0.015, 0.02]),
                    hsv_s=self.rng.choice([0.4, 0.6, 0.7]),
                    hsv_v=self.rng.choice([0.2, 0.4, 0.5]),
                    mosaic=self.rng.choice([0.5, 0.75, 1.0]),
                    mixup=self.rng.choice([0.0, 0.05, 0.10, 0.15]),
                    copy_paste=self.rng.choice([0.0, 0.05]),
                    scale=self.rng.choice([0.3, 0.5, 0.7]),
                    translate=self.rng.choice([0.05, 0.1, 0.15]),
                    degrees=self.rng.choice([0.0, 5.0, 10.0]),
                    freeze=self.rng.choice([0, 10, 15]),
                    cos_lr=self.rng.choice([True, False]),
                )
            )
        return candidates

    def run(
        self,
        train_fn: Callable[[TuningCandidate], Dict[str, float]],
        metric_name: str = "mAP50",
        n_trials: int = 8,
        history_path: Optional[Path] = None,
    ) -> Tuple[Optional[TuningCandidate], List[TuningTrialResult]]:
        """Run the search and return the best candidate."""
        results: List[TuningTrialResult] = []
        best_candidate: Optional[TuningCandidate] = None
        best_metric = float("-inf")

        for idx, candidate in enumerate(self.suggest(n_trials), start=1):
            try:
                metrics = train_fn(candidate)
                value = float(metrics.get(metric_name, float("-inf")))
                notes = metrics.get("notes", "")
                result = TuningTrialResult(candidate, metric_name, value, notes)
                results.append(result)
                logger.info(f"Tuning trial {idx}/{n_trials}: {metric_name}={value:.4f}")
                if value > best_metric:
                    best_metric = value
                    best_candidate = candidate
            except Exception as exc:
                logger.exception("Tuning trial failed")
                results.append(
                    TuningTrialResult(candidate, metric_name, float("-inf"), notes=str(exc))
                )

        if history_path is not None:
            history_path = Path(history_path)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "best_candidate": asdict(best_candidate) if best_candidate else None,
                "results": [
                    {
                        "candidate": asdict(item.candidate),
                        "metric_name": item.metric_name,
                        "metric_value": item.metric_value,
                        "notes": item.notes,
                    }
                    for item in results
                ],
            }
            history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return best_candidate, results
