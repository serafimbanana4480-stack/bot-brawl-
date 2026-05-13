"""
dataset_manager.py

Dataset versioning and quality management for Brawl Stars training.

Manages dataset versions, calculates quality metrics, and tracks
dataset evolution over time.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for a dataset version."""
    version: str
    created_at: str
    description: str
    num_images: int
    num_annotations: int
    classes: List[str]
    quality_score: float
    sha256: str = ""
    parent_version: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DatasetManager:
    """
    Manage dataset versions and quality.
    
    Provides versioning, quality scoring, and metadata management
    for training datasets.
    """
    
    def __init__(self, base_dir: Path):
        """
        Initialize dataset manager.
        
        Args:
            base_dir: Base directory for datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_dir / "metadata.json"
        
    def create_version(
        self,
        version_name: str,
        description: str,
        parent_version: str = ""
    ) -> Path:
        """
        Create a new dataset version.
        
        Args:
            version_name: Name of the version (e.g., "v1.0")
            description: Description of the version
            parent_version: Parent version name (if derived from another)
            
        Returns:
            Path to the new version directory
        """
        version_dir = self.base_dir / version_name
        version_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (version_dir / "images").mkdir(exist_ok=True)
        (version_dir / "labels").mkdir(exist_ok=True)
        
        # Create metadata
        metadata = DatasetMetadata(
            version=version_name,
            created_at=datetime.now().isoformat(),
            description=description,
            num_images=0,
            num_annotations=0,
            classes=[],
            quality_score=0.0,
            parent_version=parent_version
        )
        
        self._save_metadata(version_dir, metadata)
        
        logger.info(f"Created dataset version {version_name} at {version_dir}")
        return version_dir
    
    def update_metadata(
        self,
        version_name: str,
        num_images: Optional[int] = None,
        num_annotations: Optional[int] = None,
        classes: Optional[List[str]] = None
    ) -> bool:
        """
        Update metadata for a dataset version.
        
        Args:
            version_name: Name of the version
            num_images: Number of images (optional)
            num_annotations: Number of annotations (optional)
            classes: List of class names (optional)
            
        Returns:
            True if successful, False otherwise
        """
        version_dir = self.base_dir / version_name
        if not version_dir.exists():
            logger.error(f"Version {version_name} does not exist")
            return False
        
        metadata = self._load_metadata(version_dir)
        if metadata is None:
            return False
        
        # Update fields
        if num_images is not None:
            metadata.num_images = num_images
        if num_annotations is not None:
            metadata.num_annotations = num_annotations
        if classes is not None:
            metadata.classes = classes
        
        # Recalculate quality score
        metadata.quality_score = self._calculate_quality_score(metadata)
        
        self._save_metadata(version_dir, metadata)
        
        logger.info(f"Updated metadata for version {version_name}")
        return True
    
    def calculate_quality(self, version_name: str) -> float:
        """
        Calculate dataset quality score.
        
        Args:
            version_name: Name of the version
            
        Returns:
            Quality score (0-1)
        """
        version_dir = self.base_dir / version_name
        if not version_dir.exists():
            logger.error(f"Version {version_name} does not exist")
            return 0.0
        
        metadata = self._load_metadata(version_dir)
        if metadata is None:
            return 0.0
        
        quality = self._calculate_quality_score(metadata)
        
        # Update metadata
        metadata.quality_score = quality
        self._save_metadata(version_dir, metadata)
        
        return quality
    
    def _calculate_quality_score(self, metadata: DatasetMetadata) -> float:
        """Calculate quality score from metadata."""
        if metadata.num_images == 0:
            return 0.0
        
        # Simple quality metric: annotations per image
        annotations_per_image = metadata.num_annotations / metadata.num_images
        
        # Normalize to 0-1 (assuming 10 annotations per image is "perfect")
        quality = min(annotations_per_image / 10.0, 1.0)
        
        return quality
    
    def get_version_info(self, version_name: str) -> Optional[Dict]:
        """
        Get information about a dataset version.
        
        Args:
            version_name: Name of the version
            
        Returns:
            Dictionary with version info or None if not found
        """
        version_dir = self.base_dir / version_name
        if not version_dir.exists():
            logger.error(f"Version {version_name} does not exist")
            return None
        
        metadata = self._load_metadata(version_dir)
        if metadata is None:
            return None
        
        return metadata.to_dict()
    
    def list_versions(self) -> List[Dict]:
        """List all dataset versions."""
        versions = []
        
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                metadata = self._load_metadata(item)
                if metadata:
                    versions.append(metadata.to_dict())
        
        # Sort by creation date
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        
        return versions
    
    def calculate_sha256(self, version_name: str) -> Optional[str]:
        """
        Calculate SHA256 hash of dataset version.
        
        Args:
            version_name: Name of the version
            
        Returns:
            SHA256 hash or None if failed
        """
        version_dir = self.base_dir / version_name
        if not version_dir.exists():
            return None
        
        sha256_hash = hashlib.sha256()
        
        # Hash all files in the version directory
        for file_path in version_dir.rglob("*"):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _load_metadata(self, version_dir: Path) -> Optional[DatasetMetadata]:
        """Load metadata from version directory."""
        metadata_file = version_dir / "metadata.json"
        
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            return DatasetMetadata(**data)
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None
    
    def _save_metadata(self, version_dir: Path, metadata: DatasetMetadata) -> None:
        """Save metadata to version directory."""
        metadata_file = version_dir / "metadata.json"
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def get_global_metadata(self) -> Dict:
        """Get global metadata file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load global metadata: {e}")
        
        return {
            "created_at": datetime.now().isoformat(),
            "versions": [],
            "current_version": ""
        }
    
    def update_global_metadata(self, current_version: str) -> None:
        """Update global metadata file."""
        versions = self.list_versions()
        version_names = [v["version"] for v in versions]
        
        global_meta = {
            "created_at": self.get_global_metadata().get("created_at", datetime.now().isoformat()),
            "versions": version_names,
            "current_version": current_version,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(global_meta, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save global metadata: {e}")
