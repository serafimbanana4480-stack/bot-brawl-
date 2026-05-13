"""
validate_and_register_model.py

Valida e registra modelo YOLO treinado no sistema Brawl Stars Bot.
Verifica se o modelo é válido para Brawl Stars e o registra no model_registry.json.

Usage:
    python -m brawl_bot.training.validate_and_register_model --model ./models/brawlstars/weights/best.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import hashlib

# Adicionar diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


BRAWL_STARS_CLASSES = [
    "enemy",      # 0
    "teammate",   # 1
    "player",     # 2
    "wall",       # 3
    "bush",       # 4
    "powerup",    # 5
    "box",        # 6
    "bullet",     # 7
]


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def calculate_sha256(file_path: Path) -> str:
    """Calcula hash SHA256 de um arquivo"""
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def inspect_model(model_path: Path) -> Dict[str, Any]:
    """Inspeciona modelo YOLO para obter informações"""
    try:
        from ultralytics import YOLO
        
        logger.info(f"Carregando modelo: {model_path}")
        model = YOLO(str(model_path))
        
        # Obter informações do modelo
        info = {
            "task": model.task,
            "names": model.names,
            "nc": model.nc,
            "model_type": str(type(model.model))
        }
        
        logger.info(f"Modelo carregado: task={info['task']}, classes={info['nc']}")
        logger.info(f"Classes detectadas: {info['names']}")
        
        return info
        
    except ImportError:
        logger.error("Ultralytics não está instalado")
        return None
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        return None


def validate_brawl_stars_model(model_info: Dict[str, Any]) -> tuple[bool, str]:
    """Valida se o modelo é válido para Brawl Stars"""
    if model_info is None:
        return False, "Não foi possível inspecionar o modelo"
    
    # Verificar número de classes
    if model_info['nc'] != len(BRAWL_STARS_CLASSES):
        return False, f"Número de classes incorreto: {model_info['nc']} (esperado {len(BRAWL_STARS_CLASSES)})"
    
    # Verificar se as classes correspondem
    model_classes = list(model_info['names'].values())
    
    # Verificar se tem classes COCO (indicador de modelo não treinado)
    coco_overlap = set(model_classes) & set(COCO_CLASSES)
    if len(coco_overlap) > 5:  # Se tiver mais de 5 classes COCO, provavelmente não é Brawl Stars
        return False, f"Modelo contém classes COCO: {list(coco_overlap)[:5]}... (indicador de modelo não treinado para Brawl Stars)"
    
    # Verificar se tem classes esperadas do Brawl Stars
    brawl_stars_overlap = set(model_classes) & set(BRAWL_STARS_CLASSES)
    if len(brawl_stars_overlap) < 3:
        return False, f"Modelo não contém classes esperadas de Brawl Stars (apenas {len(brawl_stars_overlap)}/{len(BRAWL_STARS_CLASSES)})"
    
    return True, "Modelo válido para Brawl Stars"


def register_model(model_path: Path, model_info: Dict[str, Any], 
                  validation_result: tuple[bool, str], registry_path: Path):
    """Registra o modelo no model_registry.json"""
    try:
        # Carregar registry existente
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {
                "generated_at": "2026-05-09T00:00:00.000000",
                "models": {},
                "quarantined": []
            }
        
        # Calcular hash
        sha256 = calculate_sha256(model_path)
        
        # Criar entrada do modelo
        model_name = model_path.name
        registry['models'][model_name] = {
            "name": model_name,
            "path": str(model_path.absolute()),
            "sha256": sha256,
            "classes": list(model_info['names'].values()),
            "status": "valid" if validation_result[0] else "invalid",
            "reason": validation_result[1]
        }
        
        # Salvar registry atualizado
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Modelo registrado no registry: {model_name}")
        logger.info(f"SHA256: {sha256}")
        logger.info(f"Status: {validation_result[1]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao registrar modelo: {e}")
        return False


def update_config_json(model_name: str, config_path: Path):
    """Atualiza config.json para usar o novo modelo"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Atualizar modelo principal
        config['vision']['main_model'] = model_name
        
        # Salvar config atualizado
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"config.json atualizado para usar: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao atualizar config.json: {e}")
        return False


def main():
    """Função principal para CLI"""
    parser = argparse.ArgumentParser(description="Validar e registrar modelo YOLO")
    parser.add_argument("--model", required=True,
                       help="Caminho para o modelo treinado (.pt)")
    parser.add_argument("--config", default="config.json",
                       help="Caminho para config.json (default: config.json)")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    
    model_path = Path(args.model)
    config_path = Path(args.config)
    registry_path = Path(__file__).parent.parent / "models" / "model_registry.json"
    
    # Verificar se modelo existe
    if not model_path.exists():
        logger.error(f"Modelo não encontrado: {model_path}")
        return
    
    # Inspecionar modelo
    model_info = inspect_model(model_path)
    
    if model_info is None:
        return
    
    # Validar modelo
    is_valid, validation_message = validate_brawl_stars_model(model_info)
    
    if not is_valid:
        logger.error(f"❌ Validação falhou: {validation_message}")
        logger.error("O modelo não é adequado para Brawl Stars.")
        logger.info("Treine o modelo com dataset real de Brawl Stars primeiro.")
        return
    
    logger.info(f"✅ Validação passou: {validation_message}")
    
    # Registrar modelo
    success = register_model(model_path, model_info, (is_valid, validation_message), registry_path)
    
    if success:
        # Atualizar config.json
        update_config_json(model_path.name, config_path)
        
        logger.info("✅ Modelo registrado e configurado com sucesso!")
        logger.info(f"Pronto para uso em: {config_path}")


if __name__ == "__main__":
    main()
