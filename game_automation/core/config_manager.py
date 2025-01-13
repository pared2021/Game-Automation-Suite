"""
Configuration management system
"""

import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.error_handler import log_exception, GameAutomationError

logger = get_logger(__name__)

class ConfigError(GameAutomationError):
    """Configuration related errors"""
    pass

class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_dir: str = "data/config"):
        """Initialize configuration manager
        
        Args:
            config_dir: Configuration directory
        """
        self.config_dir = config_dir
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        # Default configuration
        self.default_config = {
            "game": {
                "window_title": "",
                "process_name": "",
                "resolution": {
                    "width": 1280,
                    "height": 720
                },
                "fps_limit": 60,
                "input_delay": 50
            },
            "recognition": {
                "confidence_threshold": 0.8,
                "match_method": "template",
                "max_matches": 5,
                "scale_factor": 1.0,
                "use_grayscale": True
            },
            "task": {
                "auto_retry": True,
                "max_retries": 3,
                "retry_delay": 5,
                "timeout": 300
            },
            "debug": {
                "save_screenshots": False,
                "log_level": "INFO",
                "show_matches": False,
                "record_video": False
            }
        }
        
        # Current configuration
        self.config: Dict[str, Any] = {}
        
        # Load default config
        self.reset_config()
        
    def reset_config(self):
        """Reset configuration to default"""
        self.config = self.default_config.copy()
        
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration
        
        Args:
            section: Configuration section name
            
        Returns:
            Dict: Configuration data
        """
        if section:
            return self.config.get(section, {})
        return self.config
        
    def set_config(self, config: Dict[str, Any], section: Optional[str] = None):
        """Set configuration
        
        Args:
            config: Configuration data
            section: Configuration section name
        """
        if section:
            self.config[section] = config
        else:
            self.config = config
            
    def update_config(self, updates: Dict[str, Any], section: Optional[str] = None):
        """Update configuration
        
        Args:
            updates: Configuration updates
            section: Configuration section name
        """
        target = self.config if not section else self.config.setdefault(section, {})
        self._deep_update(target, updates)
        
    def _deep_update(self, target: Dict, updates: Dict):
        """Deep update dictionary
        
        Args:
            target: Target dictionary
            updates: Update dictionary
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
                
    def save_config(self, filename: str = "config.json"):
        """Save configuration to file
        
        Args:
            filename: Configuration file name
        """
        filepath = os.path.join(self.config_dir, filename)
        
        # Create backup
        if os.path.exists(filepath):
            backup_name = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = os.path.join(self.config_dir, "backup", backup_name)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            os.rename(filepath, backup_path)
            
        # Save new config
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
            
    def load_config(self, filename: str = "config.json"):
        """Load configuration from file
        
        Args:
            filename: Configuration file name
        """
        filepath = os.path.join(self.config_dir, filename)
        if not os.path.exists(filepath):
            raise ConfigError(f"Configuration file not found: {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # Validate and merge with default config
        self.config = self._merge_config(self.default_config, config)
        
    def _merge_config(self, default: Dict, config: Dict) -> Dict:
        """Merge configuration with default values
        
        Args:
            default: Default configuration
            config: User configuration
            
        Returns:
            Dict: Merged configuration
        """
        result = default.copy()
        
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def validate_config(self, config: Dict) -> bool:
        """Validate configuration
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: Whether configuration is valid
        """
        # TODO: Implement validation rules
        return True
        
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema
        
        Returns:
            Dict: Configuration schema
        """
        # TODO: Implement schema generation
        return {}
