"""资源管理器"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar, Generic

from .cache import CacheManager
from .errors import (
    ResourceError,
    ResourceLoadError,
    ResourceUnloadError,
    ResourceVerifyError,
    ResourceNotFoundError,
    ResourceAlreadyExistsError,
    ResourceInvalidError,
    ResourceBusyError
)
from .types.base import BaseResource
from .loaders.base import BaseLoader

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseResource)


class ResourceManager(Generic[T]):
    """资源管理器
    
    特性：
    - 资源加载和卸载
    - 资源缓存
    - 资源验证
    - 错误处理
    """
    
    def __init__(
        self,
        base_path: str,
        loader: Type[BaseLoader],
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """初始化资源管理器
        
        Args:
            base_path: 基础路径
            loader: 加载器类型
            cache_dir: 缓存目录
            **kwargs: 其他参数
        """
        self._base_path = Path(base_path)
        self._loader = loader(base_path, **kwargs)
        self._resources: Dict[str, T] = {}
        
        # 创建缓存管理器
        if cache_dir:
            self._cache = CacheManager(cache_dir)
        else:
            self._cache = None
            
    @property
    def base_path(self) -> Path:
        """获取基础路径"""
        return self._base_path
        
    @property
    def loader(self) -> BaseLoader:
        """获取加载器"""
        return self._loader
        
    @property
    def resources(self) -> Dict[str, T]:
        """获取资源字典"""
        return self._resources
        
    async def load(
        self,
        key: str,
        resource_type: Type[T],
        path: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> T:
        """加载资源
        
        Args:
            key: 资源键
            resource_type: 资源类型
            path: 资源路径
            use_cache: 是否使用缓存
            **kwargs: 其他参数
            
        Returns:
            资源对象
        """
        # 检查资源是否已存在
        if key in self._resources:
            raise ResourceAlreadyExistsError(key)
            
        try:
            # 尝试从缓存加载
            if use_cache and self._cache:
                cache_key = self._get_cache_key(key, path)
                data = self._cache.get(cache_key)
                if data is not None:
                    kwargs['data'] = data
                    
            # 加载资源
            resource = await self._loader.load(
                key,
                resource_type,
                path,
                **kwargs
            )
            
            # 验证资源
            if not await resource.verify():
                raise ResourceInvalidError(
                    f"Resource {key} failed verification"
                )
                
            # 添加到资源字典
            self._resources[key] = resource
            
            # 添加到缓存
            if use_cache and self._cache:
                self._cache.put(cache_key, resource.data)
                
            return resource
            
        except Exception as e:
            logger.error(f"Failed to load resource {key}: {e}")
            if isinstance(e, ResourceError):
                raise
            raise ResourceLoadError(str(e))
            
    async def unload(self, key: str) -> None:
        """卸载资源
        
        Args:
            key: 资源键
        """
        # 检查资源是否存在
        if key not in self._resources:
            raise ResourceNotFoundError(key)
            
        try:
            # 卸载资源
            resource = self._resources[key]
            await self._loader.unload(resource)
            
            # 从资源字典中移除
            del self._resources[key]
            
        except Exception as e:
            logger.error(f"Failed to unload resource {key}: {e}")
            if isinstance(e, ResourceError):
                raise
            raise ResourceUnloadError(str(e))
            
    async def reload(self, key: str) -> T:
        """重新加载资源
        
        Args:
            key: 资源键
            
        Returns:
            资源对象
        """
        # 检查资源是否存在
        if key not in self._resources:
            raise ResourceNotFoundError(key)
            
        # 获取资源信息
        resource = self._resources[key]
        resource_type = type(resource)
        path = str(resource.path)
        kwargs = resource.kwargs
        
        # 卸载资源
        await self.unload(key)
        
        # 重新加载资源
        return await self.load(
            key,
            resource_type,
            path,
            **kwargs
        )
        
    async def verify(self, key: str) -> bool:
        """验证资源
        
        Args:
            key: 资源键
            
        Returns:
            验证是否通过
        """
        # 检查资源是否存在
        if key not in self._resources:
            raise ResourceNotFoundError(key)
            
        try:
            # 验证资源
            resource = self._resources[key]
            return await resource.verify()
            
        except Exception as e:
            logger.error(f"Failed to verify resource {key}: {e}")
            if isinstance(e, ResourceError):
                raise
            raise ResourceVerifyError(str(e))
            
    def clear(self) -> None:
        """清除所有资源"""
        for key in list(self._resources.keys()):
            try:
                self.unload(key)
            except Exception as e:
                logger.error(f"Failed to unload resource {key}: {e}")
                
        self._resources.clear()
        
    def _get_cache_key(self, key: str, path: Optional[str] = None) -> str:
        """获取缓存键
        
        Args:
            key: 资源键
            path: 资源路径
            
        Returns:
            缓存键
        """
        if path is None:
            path = key
        return f"{key}:{path}"
