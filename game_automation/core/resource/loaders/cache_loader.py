"""缓存加载器实现"""

import os
from typing import Type, Dict, Any, Optional, List, Callable
from pathlib import Path

from ..base import ResourceBase
from ..loader import ResourceLoader
from ..types.cache import CacheResource
from ..errors import ResourceLoadError


class CacheLoader(ResourceLoader):
    """缓存加载器
    
    功能：
    - 支持多种缓存格式
    - 支持缓存过期
    - 支持缓存压缩
    - 支持自定义序列化
    """
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        default_ttl: Optional[int] = None,
        compress: bool = False,
        serializer: Optional[Callable[[Any], bytes]] = None,
        deserializer: Optional[Callable[[bytes], Any]] = None
    ):
        """初始化缓存加载器
        
        Args:
            base_path: 缓存基础路径
            default_ttl: 默认缓存生存时间（秒）
            compress: 是否压缩
            serializer: 自定义序列化函数
            deserializer: 自定义反序列化函数
        """
        self._base_path = Path(base_path) if base_path else None
        self._default_ttl = default_ttl
        self._compress = compress
        self._serializer = serializer
        self._deserializer = deserializer
        
    async def load(
        self,
        key: str,
        resource_type: Type[ResourceBase],
        **kwargs: Any
    ) -> ResourceBase:
        """加载缓存资源
        
        Args:
            key: 资源标识符
            resource_type: 资源类型
            **kwargs: 加载参数
                path: 缓存路径（相对或绝对）
                metadata: 资源元数据
                ttl: 缓存生存时间（秒）
                compress: 是否压缩
                serializer: 自定义序列化函数
                deserializer: 自定义反序列化函数
                
        Returns:
            缓存资源对象
            
        Raises:
            ResourceLoadError: 资源加载失败
        """
        if not issubclass(resource_type, CacheResource):
            raise ResourceLoadError(
                key,
                f"Invalid resource type: {resource_type}"
            )
            
        # 获取缓存路径
        path = kwargs.get('path')
        if not path:
            raise ResourceLoadError(key, "Cache path not specified")
            
        # 处理相对路径
        if self._base_path and not os.path.isabs(path):
            path = self._base_path / path
            
        # 获取缓存参数
        ttl = kwargs.get('ttl', self._default_ttl)
        compress = kwargs.get('compress', self._compress)
        serializer = kwargs.get('serializer', self._serializer)
        deserializer = kwargs.get('deserializer', self._deserializer)
        
        # 创建资源对象
        return CacheResource(
            key,
            str(path),
            metadata=kwargs.get('metadata'),
            ttl=ttl,
            compress=compress,
            serializer=serializer,
            deserializer=deserializer
        )
        
    async def unload(self, resource: ResourceBase) -> None:
        """释放缓存资源
        
        Args:
            resource: 资源对象
        """
        if not isinstance(resource, CacheResource):
            return
            
        await resource.unload()
        
    def clear_all(self) -> None:
        """清除所有缓存"""
        if not self._base_path or not self._base_path.exists():
            return
            
        for path in self._base_path.glob('*'):
            if path.is_file():
                path.unlink()
