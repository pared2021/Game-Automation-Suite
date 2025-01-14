"""资源加载器实现"""

from abc import ABC, abstractmethod
from typing import Type, Dict, Any, Optional
from .base import ResourceBase, ResourceType


class ResourceLoader(ABC):
    """资源加载器基类
    
    负责资源的加载和释放操作。
    """
    
    @abstractmethod
    async def load(
        self,
        key: str,
        resource_type: Type[ResourceBase],
        **kwargs: Any
    ) -> ResourceBase:
        """加载资源
        
        Args:
            key: 资源标识符
            resource_type: 资源类型
            **kwargs: 加载参数
            
        Returns:
            加载的资源对象
        """
        pass
        
    @abstractmethod
    async def unload(self, resource: ResourceBase) -> None:
        """释放资源
        
        Args:
            resource: 要释放的资源对象
        """
        pass
