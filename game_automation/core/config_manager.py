"""
Configuration management system
"""

import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import aiofiles
import asyncio

from .error.error_manager import GameAutomationError, ErrorCategory, ErrorSeverity
from .events import Event
from .events.event_dispatcher import EventDispatcher
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ConfigError(GameAutomationError):
    """Configuration related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )

class ConfigEvent(Event):
    """Configuration change event"""
    
    def __init__(self, section: str, key: str, old_value: Any, new_value: Any):
        super().__init__(type="config_changed")
        self.section = section
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = datetime.now()

class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_dir: str = "data/config"):
        """Initialize configuration manager
        
        Args:
            config_dir: Configuration directory
        """
        self.config_dir = config_dir
        self._initialized = False
        self._event_dispatcher = EventDispatcher()
        
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
        
    async def initialize(self):
        """Initialize configuration manager asynchronously"""
        if self._initialized:
            return
            
        try:
            # 创建配置目录
            config_path = Path(self.config_dir)
            config_path.mkdir(parents=True, exist_ok=True)
            
            # 启动事件调度器
            await self._event_dispatcher.start()
            
            # 加载默认配置
            self.reset_config()
            
            # 尝试加载已有配置
            config_file = config_path / "config.json"
            if config_file.exists():
                await self.load_config()
                
            self._initialized = True
            logger.info("Configuration manager initialized")
            
        except Exception as e:
            raise ConfigError(
                "Failed to initialize configuration manager",
                context={
                    "config_dir": self.config_dir,
                    "error": str(e)
                }
            )
            
    async def ensure_initialized(self):
        """Ensure configuration manager is initialized"""
        if not self._initialized:
            await self.initialize()
            
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
        if section:
            if section not in self.config:
                self.config[section] = {}
            # 获取旧值并更新配置
            old_values = {}
            for key, value in updates.items():
                old_values[key] = self.config[section].get(key)
            
            # 更新配置
            self._deep_update(self.config[section], updates)
            
            # 触发配置变更事件
            for key, value in updates.items():
                asyncio.create_task(self.on_config_changed(section, key, old_values[key], value))
        else:
            # 更新整个配置
            for section_name, section_updates in updates.items():
                if section_name not in self.config:
                    self.config[section_name] = {}
                # 获取旧值并更新配置
                old_values = {}
                for key, value in section_updates.items():
                    old_values[key] = self.config[section_name].get(key)
                
                # 更新配置
                self._deep_update(self.config[section_name], section_updates)
                
                # 触发配置变更事件
                for key, value in section_updates.items():
                    asyncio.create_task(self.on_config_changed(section_name, key, old_values[key], value))
                
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
                
    async def save_config(self, filename: str = "config.json"):
        """Save configuration to file
        
        Args:
            filename: Configuration file name
        """
        await self.ensure_initialized()
        filepath = os.path.join(self.config_dir, filename)
        
        try:
            # Create backup
            if os.path.exists(filepath):
                backup_name = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                backup_path = os.path.join(self.config_dir, "backup", backup_name)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                os.rename(filepath, backup_path)
                
            # Save new config
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.config, indent=4, ensure_ascii=False))
                
        except Exception as e:
            raise ConfigError(
                "Failed to save configuration",
                context={
                    "file": filepath,
                    "error": str(e)
                }
            )
            
    async def load_config(self, filename: str = "config.json"):
        """Load configuration from file
        
        Args:
            filename: Configuration file name
            
        Raises:
            ConfigError: If configuration file not found or invalid
        """
        await self.ensure_initialized()
        filepath = os.path.join(self.config_dir, filename)
        if not os.path.exists(filepath):
            raise ConfigError(
                f"Configuration file not found: {filepath}",
                context={
                    "file": filepath,
                    "current_dir": os.getcwd()
                }
            )
            
        try:
            async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                content = await f.read()
                config = json.loads(content)
        except json.JSONDecodeError as e:
            raise ConfigError(
                f"Invalid JSON in configuration file: {filepath}",
                context={
                    "file": filepath,
                    "error": str(e),
                    "line": e.lineno,
                    "column": e.colno
                }
            )
        except Exception as e:
            raise ConfigError(
                f"Failed to load configuration file: {filepath}",
                context={
                    "file": filepath,
                    "error": str(e)
                }
            )
            
        # Validate and merge with default config
        self.config = self._merge_config(self.default_config, config)
        
    async def on_config_changed(self, section: str, key: str, old_value: Any, new_value: Any):
        """Handle configuration change event
        
        Args:
            section: Configuration section
            key: Configuration key
            old_value: Old value
            new_value: New value
        """
        event = ConfigEvent(section, key, old_value, new_value)
        await self._event_dispatcher.dispatch(event)
        
    def validate_config(self, config: Dict) -> bool:
        """Validate configuration
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: Whether configuration is valid
            
        Raises:
            ConfigError: If configuration is invalid
        """
        try:
            schema = self.get_config_schema()
            if not schema:  # 如果没有模式定义，暂时跳过验证
                return True
                
            # TODO: 实现完整的验证逻辑
            for section, section_schema in schema.items():
                if section not in config:
                    raise ConfigError(f"Missing required section: {section}")
                    
                section_config = config[section]
                if not isinstance(section_config, dict):
                    raise ConfigError(f"Invalid section type: {section}")
                    
                for key, key_schema in section_schema.items():
                    if key not in section_config:
                        if key_schema.get("required", False):
                            raise ConfigError(f"Missing required key: {section}.{key}")
                        continue
                        
                    value = section_config[key]
                    expected_type = key_schema.get("type")
                    if expected_type and not isinstance(value, expected_type):
                        raise ConfigError(f"Invalid type for {section}.{key}: expected {expected_type.__name__}")
                        
                    if "validator" in key_schema:
                        if not key_schema["validator"](value):
                            raise ConfigError(f"Validation failed for {section}.{key}")
                            
            return True
            
        except Exception as e:
            if not isinstance(e, ConfigError):
                raise ConfigError(f"Configuration validation failed: {str(e)}")
            raise
            
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
        
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema
        
        Returns:
            Dict: Configuration schema
        """
        return {
            "game": {
                "window_title": {
                    "type": str,
                    "required": False
                },
                "process_name": {
                    "type": str,
                    "required": False
                },
                "resolution": {
                    "type": dict,
                    "required": True,
                    "schema": {
                        "width": {
                            "type": int,
                            "required": True,
                            "validator": lambda x: x > 0
                        },
                        "height": {
                            "type": int,
                            "required": True,
                            "validator": lambda x: x > 0
                        }
                    }
                },
                "fps_limit": {
                    "type": int,
                    "required": True,
                    "validator": lambda x: x > 0
                },
                "input_delay": {
                    "type": int,
                    "required": True,
                    "validator": lambda x: x >= 0
                }
            },
            "recognition": {
                "confidence_threshold": {
                    "type": float,
                    "required": True,
                    "validator": lambda x: 0 <= x <= 1
                },
                "match_method": {
                    "type": str,
                    "required": True,
                    "validator": lambda x: x in ["template", "feature"]
                },
                "max_matches": {
                    "type": int,
                    "required": True,
                    "validator": lambda x: x > 0
                },
                "scale_factor": {
                    "type": float,
                    "required": True,
                    "validator": lambda x: x > 0
                },
                "use_grayscale": {
                    "type": bool,
                    "required": True
                }
            },
            "task": {
                "auto_retry": {
                    "type": bool,
                    "required": True
                },
                "max_retries": {
                    "type": int,
                    "required": True,
                    "validator": lambda x: x >= 0
                },
                "retry_delay": {
                    "type": int,
                    "required": True,
                    "validator": lambda x: x >= 0
                },
                "timeout": {
                    "type": int,
                    "required": True,
                    "validator": lambda x: x > 0
                }
            },
            "debug": {
                "save_screenshots": {
                    "type": bool,
                    "required": True
                },
                "log_level": {
                    "type": str,
                    "required": True,
                    "validator": lambda x: x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                },
                "show_matches": {
                    "type": bool,
                    "required": True
                },
                "record_video": {
                    "type": bool,
                    "required": True
                }
            }
        }
