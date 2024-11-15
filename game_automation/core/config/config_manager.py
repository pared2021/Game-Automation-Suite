from typing import Any, Dict, Optional, Union
import yaml
import json
import os
from pathlib import Path
from ..base.service_base import ServiceBase

class ConfigManager(ServiceBase):
    """Centralized configuration management service."""
    
    def __init__(self):
        super().__init__("ConfigManager")
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._config_paths: Dict[str, str] = {}
        self._default_config: Dict[str, Any] = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "game_automation.log"
            },
            "performance": {
                "metrics_enabled": True,
                "metrics_interval": 60
            },
            "security": {
                "encryption_enabled": False,
                "access_control_enabled": False
            }
        }
    
    async def _on_start(self) -> None:
        """Initialize configuration on service start."""
        self._configs["default"] = self._default_config.copy()
        await self._load_config_files()
    
    async def _load_config_files(self) -> None:
        """Load all configuration files."""
        config_dir = Path("config")
        if not config_dir.exists():
            self.log_info("Creating config directory")
            config_dir.mkdir(parents=True)
        
        # Load YAML configs
        for config_file in config_dir.glob("*.yaml"):
            await self.load_config_file(config_file, config_file.stem)
        
        # Load JSON configs
        for config_file in config_dir.glob("*.json"):
            await self.load_config_file(config_file, config_file.stem)
    
    async def load_config_file(self, file_path: Union[str, Path], namespace: str = "default") -> bool:
        """Load configuration from file."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                self.log_error(f"Config file not found: {file_path}")
                return False
            
            self.log_info(f"Loading config from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.yaml' or file_path.suffix == '.yml':
                    config = yaml.safe_load(f)
                elif file_path.suffix == '.json':
                    config = json.load(f)
                else:
                    self.log_error(f"Unsupported config file format: {file_path.suffix}")
                    return False
            
            self._configs[namespace] = config
            self._config_paths[namespace] = str(file_path)
            self.log_info(f"Successfully loaded config for namespace: {namespace}")
            return True
            
        except Exception as e:
            self.log_error(f"Error loading config file: {file_path}", e)
            return False
    
    async def save_config(self, namespace: str = "default") -> bool:
        """Save configuration to file."""
        try:
            if namespace not in self._config_paths:
                self.log_error(f"No file path defined for namespace: {namespace}")
                return False
            
            file_path = Path(self._config_paths[namespace])
            config = self._configs[namespace]
            
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.suffix == '.yaml' or file_path.suffix == '.yml':
                    yaml.safe_dump(config, f, default_flow_style=False)
                elif file_path.suffix == '.json':
                    json.dump(config, f, indent=4)
                else:
                    self.log_error(f"Unsupported config file format: {file_path.suffix}")
                    return False
            
            self.log_info(f"Successfully saved config for namespace: {namespace}")
            return True
            
        except Exception as e:
            self.log_error(f"Error saving config for namespace: {namespace}", e)
            return False
    
    def get_config(self, namespace: str = "default") -> Dict[str, Any]:
        """Get configuration for specified namespace."""
        return self._configs.get(namespace, {}).copy()
    
    def set_config(self, config: Dict[str, Any], namespace: str = "default") -> None:
        """Set configuration for specified namespace."""
        self._configs[namespace] = config.copy()
        self.log_info(f"Updated config for namespace: {namespace}")
    
    def update_config(self, updates: Dict[str, Any], namespace: str = "default") -> None:
        """Update configuration with new values."""
        if namespace not in self._configs:
            self._configs[namespace] = {}
        
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        self._configs[namespace] = deep_update(self._configs[namespace], updates)
        self.log_info(f"Updated config values for namespace: {namespace}")
    
    def get_value(self, key: str, namespace: str = "default", default: Any = None) -> Any:
        """Get specific configuration value."""
        config = self._configs.get(namespace, {})
        keys = key.split('.')
        
        try:
            value = config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_value(self, key: str, value: Any, namespace: str = "default") -> None:
        """Set specific configuration value."""
        if namespace not in self._configs:
            self._configs[namespace] = {}
        
        keys = key.split('.')
        config = self._configs[namespace]
        
        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        self.log_info(f"Set config value for {key} in namespace: {namespace}")
    
    async def process(self, input_data: Dict[str, Any]) -> Any:
        """Process configuration updates."""
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
        
        namespace = input_data.get('namespace', 'default')
        action = input_data.get('action')
        
        if action == 'update':
            self.update_config(input_data.get('config', {}), namespace)
            await self.save_config(namespace)
            return {'status': 'success', 'message': 'Configuration updated'}
        
        elif action == 'get':
            return {'status': 'success', 'config': self.get_config(namespace)}
        
        else:
            raise ValueError(f"Unsupported action: {action}")
    
    async def validate(self, data: Any) -> bool:
        """Validate configuration data."""
        if not isinstance(data, dict):
            return False
        
        required_fields = ['action']
        return all(field in data for field in required_fields)
