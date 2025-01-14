"""配置资源实现"""

import os
import json
import yaml
import jsonschema
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from copy import deepcopy

from ..base import ResourceBase, ResourceType
from ..errors import ResourceLoadError


class ConfigFormat:
    """配置格式枚举"""
    JSON = 'json'
    YAML = 'yaml'
    
    @staticmethod
    def from_extension(path: str) -> str:
        """从文件扩展名获取配置格式
        
        Args:
            path: 文件路径
            
        Returns:
            配置格式
        """
        ext = Path(path).suffix.lower()
        if ext in ['.json']:
            return ConfigFormat.JSON
        elif ext in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        else:
            raise ValueError(f"Unsupported config format: {ext}")


class ConfigResource(ResourceBase):
    """配置资源
    
    支持的格式：
    - JSON
    - YAML
    
    特性：
    - 配置合并
    - 环境变量替换
    - JSON Schema 验证
    """
    
    def __init__(
        self,
        key: str,
        path: str,
        schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        env_prefix: str = 'APP_'
    ):
        """初始化配置资源
        
        Args:
            key: 资源标识符
            path: 配置文件路径
            schema: JSON Schema
            metadata: 资源元数据
            env_prefix: 环境变量前缀
        """
        super().__init__(key, ResourceType.CONFIG, metadata)
        self._path = Path(path)
        self._schema = schema
        self._env_prefix = env_prefix
        self._format = ConfigFormat.from_extension(str(path))
        self._config: Optional[Dict[str, Any]] = None
        
    @property
    def path(self) -> Path:
        """获取配置路径"""
        return self._path
        
    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """获取配置数据"""
        return deepcopy(self._config) if self._config else None
        
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项
        
        Args:
            key: 配置项键
            default: 默认值
            
        Returns:
            配置项值
        """
        if not self._config:
            return default
            
        # 支持点号分隔的键
        keys = key.split('.')
        value = self._config
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                return default
            value = value[k]
        return value
        
    async def _do_load(self) -> None:
        """加载配置
        
        Raises:
            ResourceLoadError: 配置加载失败
        """
        try:
            # 检查文件是否存在
            if not self._path.exists():
                raise ResourceLoadError(
                    self.key,
                    f"Config file not found: {self._path}"
                )
                
            # 读取配置文件
            with open(self._path, 'r', encoding='utf-8') as f:
                if self._format == ConfigFormat.JSON:
                    self._config = json.load(f)
                else:  # YAML
                    self._config = yaml.safe_load(f)
                    
            # 替换环境变量
            if self._config:
                self._replace_env_vars(self._config)
                
            # 验证配置
            if self._schema:
                try:
                    jsonschema.validate(self._config, self._schema)
                except jsonschema.exceptions.ValidationError as e:
                    raise ResourceLoadError(
                        self.key,
                        f"Config validation failed: {e.message}"
                    )
                    
        except Exception as e:
            if not isinstance(e, ResourceLoadError):
                raise ResourceLoadError(self.key, cause=e)
            raise
            
    async def _do_unload(self) -> None:
        """释放配置"""
        self._config = None
        
    def _replace_env_vars(self, config: Any) -> None:
        """替换环境变量
        
        Args:
            config: 配置数据
        """
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and value.startswith('${'):
                    env_key = value[2:-1]
                    if self._env_prefix:
                        env_key = f"{self._env_prefix}{env_key}"
                    config[key] = os.environ.get(env_key, value)
                else:
                    self._replace_env_vars(value)
        elif isinstance(config, list):
            for i, value in enumerate(config):
                if isinstance(value, str) and value.startswith('${'):
                    env_key = value[2:-1]
                    if self._env_prefix:
                        env_key = f"{self._env_prefix}{env_key}"
                    config[i] = os.environ.get(env_key, value)
                else:
                    self._replace_env_vars(value)
                    
    def merge(self, other: Union[Dict[str, Any], 'ConfigResource']) -> None:
        """合并配置
        
        Args:
            other: 要合并的配置
            
        Raises:
            ValueError: 配置未加载
        """
        if not self._config:
            raise ValueError("Config not loaded")
            
        other_config: Optional[Dict[str, Any]] = None
        if isinstance(other, ConfigResource):
            other_config = other.config
        else:
            other_config = other
            
        if not other_config:
            return
            
        self._merge_dict(self._config, other_config)
        
    def _merge_dict(
        self,
        base: Dict[str, Any],
        other: Dict[str, Any]
    ) -> None:
        """递归合并字典
        
        Args:
            base: 基础字典
            other: 要合并的字典
        """
        for key, value in other.items():
            if (
                key in base and
                isinstance(base[key], dict) and
                isinstance(value, dict)
            ):
                self._merge_dict(base[key], value)
            else:
                base[key] = deepcopy(value)
