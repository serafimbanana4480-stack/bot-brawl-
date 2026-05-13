"""
Model download automation.
Provides triggers and actions for automatic downloading of missing YOLO models.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import urlretrieve
import hashlib

from .rule_engine import Trigger, Action, TriggerContext, ActionResult

logger = logging.getLogger(__name__)


class MissingModelTrigger(Trigger):
    """Trigger that detects missing YOLO models."""
    
    def __init__(self, model_dir: str, config: Optional[Dict[str, Any]] = None):
        super().__init__("missing_model_trigger", config)
        self.model_dir = Path(model_dir)
        self.required_models = config.get('required_models', [
            'yolov8n.pt',
            'yolov8s.pt'
        ]) if config else ['yolov8n.pt', 'yolov8s.pt']
        self.model_urls = config.get('model_urls', {
            'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt'
        }) if config else {
            'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt'
        }
        self.check_interval = config.get('check_interval', 60) if config else 60
    
    async def evaluate(self, context: TriggerContext) -> bool:
        """Check if any required models are missing."""
        missing_models = []
        
        for model_name in self.required_models:
            model_path = self.model_dir / model_name
            if not model_path.exists():
                missing_models.append(model_name)
                logger.warning(f"Missing model: {model_name}")
        
        if missing_models:
            context.data['event'] = 'missing_models'
            context.data['missing_models'] = missing_models
            return True
        
        return False


class DownloadModelAction(Action):
    """Action to download missing models."""
    
    def __init__(self, model_dir: str, config: Optional[Dict[str, Any]] = None):
        super().__init__("download_model_action", config)
        self.model_dir = Path(model_dir)
        self.model_urls = config.get('model_urls', {
            'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt'
        }) if config else {
            'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt'
        }
        self.verify_checksum = config.get('verify_checksum', False) if config else False
    
    async def execute(self, context: TriggerContext) -> ActionResult:
        """Download missing models."""
        missing_models = context.data.get('missing_models', [])
        downloaded = []
        failed = []
        
        # Ensure model directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name in missing_models:
            if model_name not in self.model_urls:
                logger.error(f"No URL configured for model: {model_name}")
                failed.append(model_name)
                continue
            
            url = self.model_urls[model_name]
            model_path = self.model_dir / model_name
            
            try:
                logger.info(f"Downloading model: {model_name} from {url}")
                
                # Download in async-friendly way
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    urlretrieve,
                    url,
                    str(model_path)
                )
                
                # Verify checksum if enabled
                if self.verify_checksum:
                    checksum = self._calculate_checksum(model_path)
                    logger.info(f"Model {model_name} checksum: {checksum}")
                
                downloaded.append(model_name)
                logger.info(f"Successfully downloaded: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
                failed.append(model_name)
        
        success = len(failed) == 0
        message = f"Downloaded {len(downloaded)} models"
        if failed:
            message += f", failed to download {len(failed)} models"
        
        return ActionResult(
            success=success,
            message=message,
            data={
                'downloaded': downloaded,
                'failed': failed
            }
        )
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


class ModelDownloadAutomation:
    """Model download automation setup."""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.rules = []
    
    def setup_auto_download(self, required_models: Optional[list] = None, model_urls: Optional[Dict[str, str]] = None):
        """Setup automatic download of missing models."""
        from .rule_engine import Rule
        
        trigger = MissingModelTrigger(
            model_dir=self.model_dir,
            config={
                'required_models': required_models or ['yolov8n.pt'],
                'model_urls': model_urls
            }
        )
        
        action = DownloadModelAction(
            model_dir=self.model_dir,
            config={'model_urls': model_urls}
        )
        
        rule = Rule(
            name="model_auto_download",
            trigger=trigger,
            actions=[action],
            cooldown_seconds=60
        )
        
        self.rules.append(rule)
        logger.info(f"Setup model auto-download automation (dir: {self.model_dir})")
        return rule
    
    def get_rules(self):
        """Get all configured rules."""
        return self.rules
