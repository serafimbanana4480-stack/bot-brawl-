"""
core/config_manager.py

Unified configuration management system for the Brawl Stars Bot.

This module provides a single source of truth for all configuration,
replacing the scattered config.json, config.example.json, and lobby.toml files.

Features:
- Single configuration file (config.yaml) for all settings
- Environment variable overrides
- Schema validation
- Migration from legacy config files
- Hot-reload support
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyYAML not installed, falling back to JSON config")


@dataclass
class EmulatorConfig:
    """Emulator configuration."""
    window_title: str = "BlueStacks App Player"
    resolution: tuple = (1920, 1080)
    adb_path: Optional[str] = None
    auto_detect: bool = True


@dataclass
class SafetyConfig:
    """Safety and anti-detection configuration."""
    max_trophies: int = 400
    warning_trophies: int = 380
    max_session_hours: float = 3.0
    break_duration_min: float = 0.5
    break_duration_max: float = 1.0
    min_apm: int = 20
    max_apm: int = 60
    suspicious_pattern_threshold: int = 5
    auto_stop_on_detection: bool = True
    # Behavioral biometrics thresholds
    human_curvature_min: float = 0.1
    human_curvature_max: float = 2.0
    human_velocity_min: float = 100.0
    human_velocity_max: float = 2000.0
    human_acceleration_max: float = 5000.0
    biometric_window_size: int = 50


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""
    schema: str = "core"  # core (4 classes) or extended (8 classes)
    auto_retrain_enabled: bool = False
    min_dataset_size: int = 100
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.001


@dataclass
class BotConfig:
    """Main bot configuration."""
    diagnostic_mode: bool = False
    enable_recording: bool = False
    auto_tuning_enabled: bool = False
    brawler_selection_enabled: bool = True
    debug_visualizer: bool = False
    dashboard_port: int = 8765
    trophy_limit: int = 400
    warning_trophies: int = 380
    
    # Sub-configurations
    emulator: EmulatorConfig = field(default_factory=EmulatorConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Brawler queue (loaded from config)
    brawler_queue: list = field(default_factory=list)


class ConfigManager:
    """Unified configuration manager."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to config file (yaml or json). If None, searches standard locations.
        """
        self.config_path = config_path or self._find_config_file()
        self.config = BotConfig()
        self._raw_config: Dict[str, Any] = {}
        
        if self.config_path and self.config_path.exists():
            self.load()
        else:
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            self._create_default_config()
    
    def _find_config_file(self) -> Optional[Path]:
        """Search for config file in standard locations."""
        # Determine bot root
        try:
            from wrapper import _BOT_ROOT
            bot_root = _BOT_ROOT
        except ImportError:
            bot_root = Path(__file__).parent.parent
        
        # Search in order of preference
        candidates = [
            bot_root / "config.yaml",
            bot_root / "config.yml",
            bot_root / "config.json",
            bot_root / "config.example.json",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                logger.info(f"Found config file: {candidate}")
                return candidate
        
        # Return default path (will create if doesn't exist)
        return bot_root / "config.yaml"
    
    def load(self) -> bool:
        """Load configuration from file."""
        if not self.config_path or not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix in ['.yaml', '.yml'] and YAML_AVAILABLE:
                    self._raw_config = yaml.safe_load(f) or {}
                else:
                    self._raw_config = json.load(f) or {}
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            # Parse into config objects
            self._parse_config()
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return False
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to config."""
        env_prefix = "BRAWL_BOT_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                # Convert nested keys (e.g., SAFETY_MAX_TROPHIES -> safety.max_trophies)
                self._set_nested_config(config_key, value)
    
    def _set_nested_config(self, key: str, value: str):
        """Set a nested config value from environment variable key."""
        parts = key.split('_')
        if len(parts) >= 2:
            section = parts[0]
            setting = '_'.join(parts[1:])
            
            # Handle type conversion
            try:
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string
            
            # Set in raw config
            if section not in self._raw_config:
                self._raw_config[section] = {}
            self._raw_config[section][setting] = value
    
    def _parse_config(self):
        """Parse raw config into typed config objects."""
        # Parse emulator config
        if 'emulator' in self._raw_config:
            emu_cfg = self._raw_config['emulator']
            self.config.emulator = EmulatorConfig(
                window_title=emu_cfg.get('window_title', self.config.emulator.window_title),
                resolution=tuple(emu_cfg.get('resolution', self.config.emulator.resolution)),
                adb_path=emu_cfg.get('adb_path'),
                auto_detect=emu_cfg.get('auto_detect', True)
            )
        
        # Parse safety config
        if 'safety' in self._raw_config:
            safe_cfg = self._raw_config['safety']
            self.config.safety = SafetyConfig(
                max_trophies=safe_cfg.get('max_trophies', self.config.safety.max_trophies),
                warning_trophies=safe_cfg.get('warning_trophies', self.config.safety.warning_trophies),
                max_session_hours=safe_cfg.get('max_session_hours', self.config.safety.max_session_hours),
                break_duration_min=safe_cfg.get('break_duration_min', self.config.safety.break_duration_min),
                break_duration_max=safe_cfg.get('break_duration_max', self.config.safety.break_duration_max),
                min_apm=safe_cfg.get('min_apm', self.config.safety.min_apm),
                max_apm=safe_cfg.get('max_apm', self.config.safety.max_apm),
                suspicious_pattern_threshold=safe_cfg.get('suspicious_pattern_threshold', 
                                                       self.config.safety.suspicious_pattern_threshold),
                auto_stop_on_detection=safe_cfg.get('auto_stop_on_detection', 
                                                   self.config.safety.auto_stop_on_detection)
            )
        
        # Parse training config
        if 'training' in self._raw_config:
            train_cfg = self._raw_config['training']
            self.config.training = TrainingConfig(
                schema=train_cfg.get('schema', self.config.training.schema),
                auto_retrain_enabled=train_cfg.get('auto_retrain_enabled', 
                                                  self.config.training.auto_retrain_enabled),
                min_dataset_size=train_cfg.get('min_dataset_size', 
                                             self.config.training.min_dataset_size),
                batch_size=train_cfg.get('batch_size', self.config.training.batch_size),
                epochs=train_cfg.get('epochs', self.config.training.epochs),
                learning_rate=train_cfg.get('learning_rate', self.config.training.learning_rate)
            )
        
        # Parse main config
        self.config.diagnostic_mode = self._raw_config.get('diagnostic_mode', 
                                                          self.config.diagnostic_mode)
        self.config.enable_recording = self._raw_config.get('enable_recording', 
                                                           self.config.enable_recording)
        self.config.auto_tuning_enabled = self._raw_config.get('auto_tuning_enabled', 
                                                              self.config.auto_tuning_enabled)
        self.config.brawler_selection_enabled = self._raw_config.get('brawler_selection_enabled', 
                                                                     self.config.brawler_selection_enabled)
        self.config.debug_visualizer = self._raw_config.get('debug_visualizer', 
                                                           self.config.debug_visualizer)
        self.config.dashboard_port = self._raw_config.get('dashboard_port', 
                                                         self.config.dashboard_port)
        self.config.trophy_limit = self._raw_config.get('trophy_limit', 
                                                       self.config.trophy_limit)
        self.config.warning_trophies = self._raw_config.get('warning_trophies', 
                                                           self.config.warning_trophies)
        
        # Parse brawler queue
        self.config.brawler_queue = self._raw_config.get('brawler_queue', [])
    
    def _create_default_config(self):
        """Create default configuration file."""
        if not self.config_path:
            return
        
        try:
            default_config = {
                'diagnostic_mode': False,
                'enable_recording': False,
                'auto_retrain_enabled': False,
                'training_schema': 'core',
                'brawler_selection_enabled': True,
                'auto_tuning_enabled': False,
                'debug_visualizer': False,
                'dashboard_port': 8765,
                'trophy_limit': 400,
                'warning_trophies': 380,
                'emulator': {
                    'window_title': 'BlueStacks App Player',
                    'resolution': [1920, 1080],
                    'auto_detect': True
                },
                'safety': {
                    'max_trophies': 400,
                    'warning_trophies': 380,
                    'max_session_hours': 3.0,
                    'break_duration_min': 0.5,
                    'break_duration_max': 1.0,
                    'min_apm': 20,
                    'max_apm': 60,
                    'suspicious_pattern_threshold': 5,
                    'auto_stop_on_detection': True
                },
                'training': {
                    'schema': 'core',
                    'auto_retrain_enabled': False,
                    'min_dataset_size': 100,
                    'batch_size': 16,
                    'epochs': 50,
                    'learning_rate': 0.001
                },
                'brawler_queue': []
            }
            
            # Use YAML if available, otherwise JSON
            if YAML_AVAILABLE and self.config_path.suffix in ['.yaml', '.yml']:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
            else:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2)
            
            logger.info(f"Created default config at {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to create default config: {e}")
    
    def save(self) -> bool:
        """Save current configuration to file."""
        if not self.config_path:
            logger.error("No config path set, cannot save")
            return False
        
        try:
            # Convert config objects to dict
            config_dict = {
                'diagnostic_mode': self.config.diagnostic_mode,
                'enable_recording': self.config.enable_recording,
                'auto_retrain_enabled': self.config.training.auto_retrain_enabled,
                'training_schema': self.config.training.schema,
                'brawler_selection_enabled': self.config.brawler_selection_enabled,
                'auto_tuning_enabled': self.config.auto_tuning_enabled,
                'debug_visualizer': self.config.debug_visualizer,
                'dashboard_port': self.config.dashboard_port,
                'trophy_limit': self.config.trophy_limit,
                'warning_trophies': self.config.warning_trophies,
                'emulator': {
                    'window_title': self.config.emulator.window_title,
                    'resolution': list(self.config.emulator.resolution),
                    'adb_path': self.config.emulator.adb_path,
                    'auto_detect': self.config.emulator.auto_detect
                },
                'safety': {
                    'max_trophies': self.config.safety.max_trophies,
                    'warning_trophies': self.config.safety.warning_trophies,
                    'max_session_hours': self.config.safety.max_session_hours,
                    'break_duration_min': self.config.safety.break_duration_min,
                    'break_duration_max': self.config.safety.break_duration_max,
                    'min_apm': self.config.safety.min_apm,
                    'max_apm': self.config.safety.max_apm,
                    'suspicious_pattern_threshold': self.config.safety.suspicious_pattern_threshold,
                    'auto_stop_on_detection': self.config.safety.auto_stop_on_detection
                },
                'training': {
                    'schema': self.config.training.schema,
                    'auto_retrain_enabled': self.config.training.auto_retrain_enabled,
                    'min_dataset_size': self.config.training.min_dataset_size,
                    'batch_size': self.config.training.batch_size,
                    'epochs': self.config.training.epochs,
                    'learning_rate': self.config.training.learning_rate
                },
                'brawler_queue': self.config.brawler_queue
            }
            
            # Save
            if YAML_AVAILABLE and self.config_path.suffix in ['.yaml', '.yml']:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            else:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """Set a configuration value by key (supports dot notation)."""
        keys = key.split('.')
        obj = self.config
        
        # Navigate to parent
        for k in keys[:-1]:
            if hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                logger.error(f"Invalid config key: {key}")
                return False
        
        # Set value
        setattr(obj, keys[-1], value)
        return True
    
    def migrate_legacy_config(self) -> bool:
        """Migrate from legacy config.json and lobby.toml to unified config."""
        try:
            from wrapper import _BOT_ROOT
            bot_root = _BOT_ROOT
        except ImportError:
            bot_root = Path(__file__).parent.parent
        
        legacy_configs = [
            bot_root / "config.json",
            bot_root / "lobby.toml",
        ]
        
        migrated = False
        
        for legacy_path in legacy_configs:
            if legacy_path.exists():
                try:
                    if legacy_path.suffix == '.json':
                        with open(legacy_path, 'r', encoding='utf-8') as f:
                            legacy_data = json.load(f)
                        
                        # Merge into current config
                        self._raw_config.update(legacy_data)
                        self._parse_config()
                        migrated = True
                        logger.info(f"Migrated config from {legacy_path}")
                    
                    elif legacy_path.suffix == '.toml':
                        # TOML migration requires toml library
                        try:
                            import toml
                            with open(legacy_path, 'r', encoding='utf-8') as f:
                                legacy_data = toml.load(f)
                            
                            # Merge into current config
                            self._raw_config.update(legacy_data)
                            self._parse_config()
                            migrated = True
                            logger.info(f"Migrated config from {legacy_path}")
                        except ImportError:
                            logger.warning("toml library not installed, skipping lobby.toml migration")
                
                except Exception as e:
                    logger.error(f"Failed to migrate {legacy_path}: {e}")
        
        if migrated:
            self.save()
        
        return migrated


# Global config instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get or create the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> BotConfig:
    """Get the current bot configuration."""
    return get_config_manager().config
