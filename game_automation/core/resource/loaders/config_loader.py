"""配置资源加载器实现"""

import os
from typing import Type, Dict, Any, Optional, List
from pathlib import Path

from ..base import ResourceBase
from ..loader import ResourceLoader
from ..types.config import ConfigResource
from ..errors import ResourceLoadError


class ConfigLoader(ResourceLoader):
    """配置资源加载器
    
    功能：
    - 支持多种配置格式
    - 支持配置验证
    - 支持环境变量
    - 支持配置覆盖
    """
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        schemas: Optional[Dict[str, Dict[str, Any]]] = None,
        env_prefix: str = 'APP_'
    ):
        """初始化配置加载器
        
        Args:
            base_path: 配置基础路径
            schemas: JSON Schema 字典
            env_prefix: 环境变量前缀
        """
        self._base_path = Path(base_path) if base_path else None
        self._schemas = schemas or {}
        self._env_prefix = env_prefix
        
    async def load(
        self,
        key: str,
        resource_type: Type[ResourceBase],
        **kwargs: Any
    ) -> ResourceBase:
        """加载配置资源
        
        Args:
            key: 资源标识符
            resource_type: 资源类型
            **kwargs: 加载参数
                path: 配置路径（相对或绝对）
                schema: JSON Schema
                metadata: 资源元数据
                env_prefix: 环境变量前缀
                
        Returns:
            配置资源对象
            
        Raises:
            ResourceLoadError: 资源加载失败
        """
        if not issubclass(resource_type, ConfigResource):
            raise ResourceLoadError(
                key,
                f"Invalid resource type: {resource_type}"
            )
            
        # 获取配置路径
        path = kwargs.get('path')
        if not path:
            raise ResourceLoadError(key, "Config path not specified")
            
        # 处理相对路径
        if self._base_path and not os.path.isabs(path):
            path = self._base_path / path
            
        # 获取 schema
        schema = kwargs.get('schema')
        if not schema and key in self._schemas:
            schema = self._schemas[key]
            
        # 获取环境变量前缀
        env_prefix = kwargs.get('env_prefix', self._env_prefix)
        
        # 创建资源对象
        return ConfigResource(
            key,
            str(path),
            schema=schema,
            metadata=kwargs.get('metadata'),
            env_prefix=env_prefix
        )
        
    async def unload(self, resource: ResourceBase) -> None:
        """释放配置资源
        
        Args:
            resource: 资源对象
        """
        if not isinstance(resource, ConfigResource):
            return
            
        await resource.unload()
