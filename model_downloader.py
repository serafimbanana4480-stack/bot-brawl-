"""
model_downloader.py

Sistema de download automático de modelos pré-treinados para visão computacional.
Suporta download de modelos YOLOv8 e modelos específicos para Brawl Stars.
"""

import os
import requests
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Informações sobre um modelo disponível"""
    name: str
    url: str
    description: str
    size_mb: float
    checksum: Optional[str] = None
    type: str = "yolov8"  # yolov8, custom
    source: str = "ultralytics"  # ultralytics, neuroNeon, custom


class ModelDownloader:
    """Gerenciador de download de modelos"""
    
    # Modelos disponíveis
    AVAILABLE_MODELS: Dict[str, ModelInfo] = {
        "yolov8n": ModelInfo(
            name="yolov8n.pt",
            url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            description="YOLOv8 Nano - Modelo mais leve e rápido",
            size_mb=6.2,
            type="yolov8",
            source="ultralytics"
        ),
        "yolov8s": ModelInfo(
            name="yolov8s.pt",
            url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
            description="YOLOv8 Small - Equilíbrio entre velocidade e precisão",
            size_mb=21.5,
            type="yolov8",
            source="ultralytics"
        ),
        "yolov8m": ModelInfo(
            name="yolov8m.pt",
            url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
            description="YOLOv8 Medium - Maior precisão",
            size_mb=49.7,
            type="yolov8",
            source="ultralytics"
        ),
        # Modelos NeuroNeon para Brawl Stars - comentado até URL real estar disponível
        # "neoneon_brawl": ModelInfo(
        #     name="neoneon_brawl.pt",
        #     url="",  # URL placeholder - needs real URL from NeuroNeon repository
        #     description="NeuroNeon - Modelo específico para Brawl Stars",
        #     size_mb=0,
        #     type="custom",
        #     source="neoneon"
        # )
    }
    
    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            # Diretório padrão: backend/brawl_bot/models/
            self.models_dir = Path(__file__).parent / "models"
        else:
            self.models_dir = Path(models_dir)
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Arquivo de metadados
        self.metadata_file = self.models_dir / "models_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Carrega metadados dos modelos baixados"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar metadados: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Salva metadados dos modelos"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao salvar metadados: {e}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calcula checksum SHA256 de um arquivo"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verifica se checksum do arquivo bate com o esperado"""
        if not expected_checksum:
            return True  # Sem checksum para verificar
        
        actual_checksum = self._calculate_checksum(file_path)
        return actual_checksum == expected_checksum
    
    def download_model(
        self, 
        model_key: str, 
        force: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Baixa um modelo específico.
        
        Args:
            model_key: Chave do modelo em AVAILABLE_MODELS
            force: Força re-download mesmo se já existir
            progress_callback: Função callback para progresso (bytes_downloaded, total_bytes)
        
        Returns:
            Dict com status do download
        """
        if model_key not in self.AVAILABLE_MODELS:
            return {
                "success": False,
                "error": f"Modelo '{model_key}' não encontrado",
                "model_key": model_key
            }
        
        model_info = self.AVAILABLE_MODELS[model_key]
        model_path = self.models_dir / model_info.name
        
        # Verificar se já existe
        if model_path.exists() and not force:
            logger.info(f"Modelo {model_key} já existe")
            
            # Atualizar metadados
            self.metadata[model_key] = {
                "name": model_info.name,
                "path": str(model_path),
                "size_mb": model_path.stat().st_size / (1024 * 1024),
                "downloaded_at": self.metadata.get(model_key, {}).get("downloaded_at"),
                "checksum": self._calculate_checksum(model_path)
            }
            self._save_metadata()
            
            return {
                "success": True,
                "message": "Modelo já existe",
                "model_key": model_key,
                "path": str(model_path)
            }
        
        # Verificar se URL é válida
        if not model_info.url:
            return {
                "success": False,
                "error": f"Modelo '{model_key}' não tem URL configurada",
                "model_key": model_key
            }
        
        # Download
        try:
            logger.info(f"Baixando modelo {model_key} de {model_info.url}")
            
            response = requests.get(model_info.url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback:
                            progress_callback(downloaded, total_size)
            
            # Verificar checksum se disponível
            if model_info.checksum and not self._verify_checksum(model_path, model_info.checksum):
                model_path.unlink()  # Remover arquivo corrompido
                return {
                    "success": False,
                    "error": "Checksum não bate - arquivo corrompido",
                    "model_key": model_key
                }
            
            import datetime
            
            # Atualizar metadados
            self.metadata[model_key] = {
                "name": model_info.name,
                "path": str(model_path),
                "size_mb": model_path.stat().st_size / (1024 * 1024),
                "downloaded_at": datetime.datetime.now().isoformat(),
                "checksum": self._calculate_checksum(model_path) if model_info.checksum else None
            }
            self._save_metadata()
            
            logger.info(f"Modelo {model_key} baixado com sucesso")
            
            return {
                "success": True,
                "message": "Modelo baixado com sucesso",
                "model_key": model_key,
                "path": str(model_path),
                "size_mb": model_path.stat().st_size / (1024 * 1024)
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao baixar modelo: {e}")
            if model_path.exists():
                model_path.unlink()
            
            return {
                "success": False,
                "error": f"Erro de download: {str(e)}",
                "model_key": model_key
            }
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            if model_path.exists():
                model_path.unlink()
            
            return {
                "success": False,
                "error": f"Erro inesperado: {str(e)}",
                "model_key": model_key
            }
    
    def list_available_models(self) -> List[Dict]:
        """Lista todos os modelos disponíveis para download"""
        return [
            {
                "key": key,
                "name": info.name,
                "description": info.description,
                "size_mb": info.size_mb,
                "type": info.type,
                "source": info.source,
                "downloaded": key in self.metadata
            }
            for key, info in self.AVAILABLE_MODELS.items()
        ]
    
    def list_downloaded_models(self) -> List[Dict]:
        """Lista todos os modelos baixados"""
        downloaded = []
        
        for key, meta in self.metadata.items():
            model_path = Path(meta["path"])
            if model_path.exists():
                downloaded.append({
                    "key": key,
                    "name": meta["name"],
                    "path": meta["path"],
                    "size_mb": meta["size_mb"],
                    "downloaded_at": meta.get("downloaded_at"),
                    "checksum": meta.get("checksum")
                })
        
        return downloaded
    
    def delete_model(self, model_key: str) -> Dict:
        """Remove um modelo baixado"""
        if model_key not in self.metadata:
            return {
                "success": False,
                "error": f"Modelo '{model_key}' não está baixado"
            }
        
        try:
            model_path = Path(self.metadata[model_key]["path"])
            if model_path.exists():
                model_path.unlink()
            
            del self.metadata[model_key]
            self._save_metadata()
            
            logger.info(f"Modelo {model_key} removido")
            
            return {
                "success": True,
                "message": "Modelo removido",
                "model_key": model_key
            }
            
        except Exception as e:
            logger.error(f"Erro ao remover modelo: {e}")
            return {
                "success": False,
                "error": f"Erro ao remover: {str(e)}"
            }
    
    def get_model_path(self, model_key: str) -> Optional[Path]:
        """Retorna caminho do modelo se estiver baixado"""
        if model_key in self.metadata:
            model_path = Path(self.metadata[model_key]["path"])
            if model_path.exists():
                return model_path
        return None


# Singleton instance
_downloader_instance: Optional[ModelDownloader] = None


def get_model_downloader(models_dir: Optional[Path] = None) -> ModelDownloader:
    """Retorna instância singleton do downloader"""
    global _downloader_instance
    if _downloader_instance is None:
        _downloader_instance = ModelDownloader(models_dir)
    return _downloader_instance
