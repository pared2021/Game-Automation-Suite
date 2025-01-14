"""基础加载器类"""

import os
import abc
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Generic

from ..types.base import BaseResource

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseResource)


class BaseLoader(abc.ABC, Generic[T]):
    """基础加载器类
    
    所有加载器类型都应该继承自这个类。
    """
    
    def __init__(
        self,
        base_path: str,
        **kwargs
    ):
        """初始化加载器
        
        Args:
            base_path: 基础路径
            **kwargs: 其他参数
        """
        self._base_path = Path(base_path)
        self._kwargs = kwargs
        
    @property
    def base_path(self) -> Path:
        """获取基础路径"""
        return self._base_path
        
    @property
    def kwargs(self) -> Dict[str, Any]:
        """获取其他参数"""
        return self._kwargs
        
    async def load(
        self,
        key: str,
        resource_type: Type[T],
        path: Optional[str] = None,
        **kwargs
    ) -> T:
        """加载资源
        
        Args:
            key: 资源键
            resource_type: 资源类型
            path: 资源路径
            **kwargs: 其他参数
            
        Returns:
            资源对象
        """
        if path is None:
            path = key
            
        # 处理相对路径
        if not os.path.isabs(path):
            path = os.path.join(self.base_path, path)
            
        try:
            # 创建资源对象
            resource = resource_type(
                key=key,
                path=path,
                **{**self.kwargs, **kwargs}
            )
            
            # 加载资源
            await resource.load()
            
            return resource
            
        except Exception as e:
            logger.error(f"Failed to load resource {key}: {e}")
            raise
            
    async def unload(self, resource: T) -> None:
        """卸载资源
        
        Args:
            resource: 资源对象
        """
        try:
            await resource.unload()
        except Exception as e:
            logger.error(f"Failed to unload resource {resource.key}: {e}")
            raise
