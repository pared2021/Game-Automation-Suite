"""资源管理器实现"""

from typing import Dict, Type, Any, Optional
from datetime import datetime
from .base import ResourceBase, ResourceState, ResourceType
from .loader import ResourceLoader
from .cache import CacheManager
from .monitor import MonitorManager
from .errors import (
    ResourceNotFoundError,
    ResourceLoadError,
    ResourceStateError
)


class ResourceManager:
    """资源管理器
    
    负责资源的生命周期管理，包括：
    - 资源注册和查找
    - 资源加载和释放
    - 资源缓存管理
    - 资源使用监控
    """
    
    def __init__(
        self,
        loader: ResourceLoader,
        cache_manager: Optional[CacheManager] = None,
        monitor_manager: Optional[MonitorManager] = None
    ):
        """初始化资源管理器
        
        Args:
            loader: 资源加载器
            cache_manager: 缓存管理器
            monitor_manager: 监控管理器
        """
        self._loader = loader
        self._cache_manager = cache_manager or CacheManager()
        self._monitor_manager = monitor_manager or MonitorManager()
        self._resources: Dict[str, ResourceBase] = {}
        
    async def get_resource(
        self,
        key: str,
        resource_type: Type[ResourceBase],
        **kwargs: Any
    ) -> ResourceBase:
        """获取资源
        
        如果资源不存在，则加载资源。
        如果资源存在但未加载，则加载资源。
        
        Args:
            key: 资源标识符
            resource_type: 资源类型
            **kwargs: 加载参数
            
        Returns:
            资源对象
            
        Raises:
            ResourceNotFoundError: 资源未找到
            ResourceLoadError: 资源加载失败
            ResourceStateError: 资源状态错误
        """
        # 1. 检查资源是否存在
        if key not in self._resources:
            # 2. 尝试从缓存加载
            cached_resource = await self._cache_manager.get(key)
            if cached_resource is not None and isinstance(cached_resource, resource_type):
                self._resources[key] = cached_resource
            else:
                # 3. 加载新资源
                try:
                    resource = await self._loader.load(key, resource_type, **kwargs)
                    self._resources[key] = resource
                    await self._cache_manager.put(key, resource)
                except Exception as e:
                    raise ResourceLoadError(key, cause=e)
                
        resource = self._resources[key]
        
        # 4. 检查资源类型
        if not isinstance(resource, resource_type):
            raise ResourceStateError(
                key,
                str(type(resource)),
                str(resource_type),
                f"Resource type mismatch: expected {resource_type}, got {type(resource)}"
            )
            
        # 5. 加载资源（如果未加载）
        if resource.state == ResourceState.UNLOADED:
            try:
                await resource.load()
            except Exception as e:
                raise ResourceLoadError(key, cause=e)
                
        # 6. 更新引用计数和监控信息
        resource.increment_ref()
        await self._monitor_manager.record_usage(resource)
        
        return resource
        
    async def release_resource(self, key: str) -> None:
        """释放资源
        
        减少资源的引用计数。当引用计数为 0 时，释放资源。
        
        Args:
            key: 资源标识符
            
        Raises:
            ResourceNotFoundError: 资源未找到
            ResourceStateError: 资源状态错误
        """
        if key not in self._resources:
            raise ResourceNotFoundError(key)
            
        resource = self._resources[key]
        ref_count = resource.decrement_ref()
        
        # 更新监控信息
        await self._monitor_manager.record_usage(resource)
        
        # 如果引用计数为 0，释放资源
        if ref_count == 0:
            try:
                await resource.unload()
            except Exception as e:
                raise ResourceStateError(
                    key,
                    str(resource.state),
                    str(ResourceState.UNLOADED),
                    cause=e
                )
                
    async def cleanup(self, max_age: float = 3600.0) -> None:
        """清理旧资源
        
        Args:
            max_age: 最大资源年龄（秒）
        """
        # 检查资源泄漏
        leaks = await self._monitor_manager.check_leaks()
        for leak in leaks:
            try:
                await self.release_resource(leak.key)
            except Exception:
                pass  # 忽略清理错误
                
        # 清理缓存
        # TODO: 实现缓存清理策略
        
    async def get_stats(self) -> Dict[str, Any]:
        """获取资源统计信息
        
        Returns:
            统计信息字典
        """
        metrics = await self._monitor_manager.get_metrics()
        return {
            'total_resources': len(self._resources),
            'loaded_resources': sum(
                1 for r in self._resources.values()
                if r.state == ResourceState.LOADED
            ),
            'error_resources': sum(
                1 for r in self._resources.values()
                if r.state == ResourceState.ERROR
            ),
            **metrics
        }
