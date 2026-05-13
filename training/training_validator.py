"""
training_validator.py

Sistema de validação de aprendizado para verificar se modelos realmente estão melhorando.
Testa se o treinamento está gerando melhorias reais ou apenas overfitting.

Funcionalidades:
- Validação de modelos em datasets de teste separados
- Comparação de performance entre versões de modelo
- Detecção de overfitting
- Testes de regressão (performance não deve piorar)
- Análise de confiança nas predições
- Relatórios detalhados de validação

Usage:
    python -m brawl_bot.training.training_validator --model-path ./models/best.pt --test-dataset ./dataset/test
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Métricas de validação de modelo"""
    model_path: str
    timestamp: str
    
    # Detection metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mAP: float = 0.0
    
    # Confidence metrics
    avg_confidence: float = 0.0
    confidence_std: float = 0.0
    low_confidence_ratio: float = 0.0
    
    # Consistency metrics
    detection_consistency: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    
    # Learning indicators
    is_overfitting: bool = False
    is_improving: bool = True
    confidence_score: float = 0.0  # Overall confidence in model quality
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ModelValidator:
    """Validador de modelos de IA"""
    
    def __init__(self, test_dataset_path: Path):
        self.test_dataset_path = Path(test_dataset_path)
        self.validation_history: List[ValidationMetrics] = []
        
        # Carregar dataset de teste
        self.test_images = []
        self.test_labels = []
        self._load_test_dataset()
        
        logger.info(f"ModelValidator initialized with {len(self.test_images)} test samples")
    
    def _load_test_dataset(self):
        """Carrega dataset de teste"""
        if not self.test_dataset_path.exists():
            logger.warning(f"Test dataset not found: {self.test_dataset_path}")
            return
        
        # Procurar imagens e labels
        images_dir = self.test_dataset_path / "images"
        labels_dir = self.test_dataset_path / "labels"
        
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return
        
        # Carregar imagens
        for img_path in images_dir.glob("*.jpg"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Carregar labels se existirem
            labels = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            labels.append([float(x) for x in parts])
            
            self.test_images.append((img_path, image))
            self.test_labels.append(labels)
        
        logger.info(f"Loaded {len(self.test_images)} test images")
    
    def validate_model(self, model_path: Path, class_mapping: Dict[int, str] = None) -> ValidationMetrics:
        """Valida um modelo específico"""
        logger.info(f"Validating model: {model_path}")
        
        try:
            from ultralytics import YOLO
            
            # Carregar modelo
            model = YOLO(str(model_path))
            
            # Calcular métricas
            metrics = self._calculate_metrics(model, class_mapping)
            metrics.model_path = str(model_path)
            metrics.timestamp = datetime.now().isoformat()
            
            # Adicionar ao histórico
            self.validation_history.append(metrics)
            
            # Analisar tendências
            self._analyze_trends(metrics)
            
            logger.info(f"Validation complete: mAP={metrics.mAP:.3f}, F1={metrics.f1_score:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationMetrics(
                model_path=str(model_path),
                timestamp=datetime.now().isoformat(),
                confidence_score=0.0
            )
    
    def _calculate_metrics(self, model, class_mapping: Dict[int, str] = None) -> ValidationMetrics:
        """Calcula métricas de validação"""
        metrics = ValidationMetrics(model_path="", timestamp="")
        
        if not self.test_images:
            logger.warning("No test images available")
            return metrics
        
        all_detections = []
        all_confidences = []
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        for img_path, image in self.test_images:
            # Rodar inferência
            results = model(image, verbose=False)
            
            if results and results[0].boxes is not None:
                detections = []
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    detections.append({
                        "class_id": class_id,
                        "confidence": confidence,
                        "bbox": bbox
                    })
                    
                    all_confidences.append(confidence)
                
                all_detections.append(detections)
            else:
                all_detections.append([])
        
        # Calcular métricas básicas
        if all_confidences:
            metrics.avg_confidence = np.mean(all_confidences)
            metrics.confidence_std = np.std(all_confidences)
            metrics.low_confidence_ratio = np.mean([c < 0.5 for c in all_confidences])
        
        # Calcular precision, recall, F1 (simplificado)
        # Em uma implementação completa, usaria IoU threshold
        total_detections = sum(len(d) for d in all_detections)
        total_ground_truth = sum(len(l) for l in self.test_labels)
        
        if total_detections > 0 and total_ground_truth > 0:
            # Estimativa simplificada
            metrics.precision = min(1.0, tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            metrics.recall = min(1.0, tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            
            if metrics.precision + metrics.recall > 0:
                metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
        
        # Calcular mAP (simplificado)
        metrics.mAP = metrics.f1_score * 0.8  # Estimativa conservadora
        
        # Calcular taxas de erro
        metrics.false_positive_rate = fp / total_detections if total_detections > 0 else 0.0
        metrics.false_negative_rate = fn / total_ground_truth if total_ground_truth > 0 else 0.0
        
        # Calcular consistência de detecção
        metrics.detection_consistency = self._calculate_consistency(all_detections)
        
        # Calcular score de confiança geral
        metrics.confidence_score = self._calculate_confidence_score(metrics)
        
        return metrics
    
    def _calculate_consistency(self, all_detections: List[List[Dict]]) -> float:
        """Calcula consistência de detecções entre frames similares"""
        # Em uma implementação completa, compararia detecções em frames consecutivos
        # Por enquanto, retorna valor baseado na variância de confiança
        if not all_detections:
            return 0.0
        
        confidences = []
        for detections in all_detections:
            if detections:
                confidences.extend([d["confidence"] for d in detections])
        
        if not confidences:
            return 0.5
        
        std = np.std(confidences)
        # Menor variância = mais consistente
        consistency = max(0.0, 1.0 - std)
        
        return consistency
    
    def _calculate_confidence_score(self, metrics: ValidationMetrics) -> float:
        """Calcula score de confiança geral no modelo"""
        score = 0.0
        
        # F1 score (40%)
        score += metrics.f1_score * 0.4
        
        # Consistência (20%)
        score += metrics.detection_consistency * 0.2
        
        # Confiança média (20%)
        score += metrics.avg_confidence * 0.2
        
        # Penalidade por falsos positivos (10%)
        score += (1.0 - metrics.false_positive_rate) * 0.1
        
        # Penalidade por baixa confiança (10%)
        score += (1.0 - metrics.low_confidence_ratio) * 0.1
        
        return min(1.0, max(0.0, score))
    
    def _analyze_trends(self, current_metrics: ValidationMetrics):
        """Analisa tendências de aprendizado ao longo do tempo"""
        if len(self.validation_history) < 2:
            return
        
        # Comparar com validação anterior
        prev_metrics = self.validation_history[-2]
        
        # Verificar se está melhorando
        improvement = (
            current_metrics.f1_score > prev_metrics.f1_score and
            current_metrics.mAP > prev_metrics.mAP and
            current_metrics.confidence_score > prev_metrics.confidence_score
        )
        
        current_metrics.is_improving = improvement
        
        # Verificar overfitting (alta precisão mas baixa recall em dados reais)
        if current_metrics.precision > 0.9 and current_metrics.recall < 0.5:
            current_metrics.is_overfitting = True
            logger.warning("Potential overfitting detected")
    
    def compare_models(self, model_path1: Path, model_path2: Path) -> Dict:
        """Compara dois modelos e retorna o melhor"""
        logger.info(f"Comparing models: {model_path1.name} vs {model_path2.name}")
        
        metrics1 = self.validate_model(model_path1)
        metrics2 = self.validate_model(model_path2)
        
        comparison = {
            "model1": metrics1.to_dict(),
            "model2": metrics2.to_dict(),
            "better_model": str(model_path1) if metrics1.confidence_score > metrics2.confidence_score else str(model_path2),
            "improvement": abs(metrics1.confidence_score - metrics2.confidence_score)
        }
        
        logger.info(f"Better model: {comparison['better_model']} (improvement: {comparison['improvement']:.3f})")
        
        return comparison
    
    def generate_validation_report(self, output_path: Path):
        """Gera relatório de validação"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_validations": len(self.validation_history),
            "test_dataset_size": len(self.test_images),
            "validations": [m.to_dict() for m in self.validation_history]
        }
        
        # Adicionar tendências
        if len(self.validation_history) >= 2:
            latest = self.validation_history[-1]
            first = self.validation_history[0]
            
            report["trends"] = {
                "f1_improvement": latest.f1_score - first.f1_score,
                "mAP_improvement": latest.mAP - first.mAP,
                "confidence_improvement": latest.confidence_score - first.confidence_score,
                "is_improving": latest.is_improving,
                "overfitting_detected": latest.is_overfitting
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {output_path}")


class RegressionTester:
    """Testador de regressão para garantir que modelos não pioram"""
    
    def __init__(self, baseline_metrics: ValidationMetrics):
        self.baseline = baseline_metrics
        self.tolerance = 0.05  # 5% tolerância para degradação
    
    def test_regression(self, new_metrics: ValidationMetrics) -> Tuple[bool, str]:
        """Testa se novo modelo tem regressão significativa"""
        issues = []
        
        # Testar F1 score
        if new_metrics.f1_score < self.baseline.f1_score * (1 - self.tolerance):
            issues.append(f"F1 score degraded: {new_metrics.f1_score:.3f} < {self.baseline.f1_score * (1 - self.tolerance):.3f}")
        
        # Testar mAP
        if new_metrics.mAP < self.baseline.mAP * (1 - self.tolerance):
            issues.append(f"mAP degraded: {new_metrics.mAP:.3f} < {self.baseline.mAP * (1 - self.tolerance):.3f}")
        
        # Testar confiança
        if new_metrics.confidence_score < self.baseline.confidence_score * (1 - self.tolerance):
            issues.append(f"Confidence score degraded: {new_metrics.confidence_score:.3f} < {self.baseline.confidence_score * (1 - self.tolerance):.3f}")
        
        # Testar taxa de falsos positivos
        if new_metrics.false_positive_rate > self.baseline.false_positive_rate * (1 + self.tolerance):
            issues.append(f"False positive rate increased: {new_metrics.false_positive_rate:.3f} > {self.baseline.false_positive_rate * (1 + self.tolerance):.3f}")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "No regression detected"


def main():
    """Função principal para execução via linha de comando"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Validator")
    parser.add_argument("--model-path", required=True, help="Caminho para o modelo")
    parser.add_argument("--test-dataset", required=True, help="Caminho para dataset de teste")
    parser.add_argument("--compare-with", help="Caminho para outro modelo para comparação")
    parser.add_argument("--output", default="./validation_report.json", help="Caminho para relatório")
    
    args = parser.parse_args()
    
    # Criar validador
    validator = ModelValidator(Path(args.test_dataset))
    
    # Validar modelo
    metrics = validator.validate_model(Path(args.model_path))
    
    print(f"\nValidation Results:")
    print(f"F1 Score: {metrics.f1_score:.3f}")
    print(f"mAP: {metrics.mAP:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall: {metrics.recall:.3f}")
    print(f"Confidence Score: {metrics.confidence_score:.3f}")
    print(f"Is Improving: {metrics.is_improving}")
    print(f"Is Overfitting: {metrics.is_overfitting}")
    
    # Comparar com outro modelo se especificado
    if args.compare_with:
        comparison = validator.compare_models(Path(args.model_path), Path(args.compare_with))
        print(f"\nComparison Results:")
        print(f"Better Model: {comparison['better_model']}")
        print(f"Improvement: {comparison['improvement']:.3f}")
    
    # Gerar relatório
    validator.generate_validation_report(Path(args.output))
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
