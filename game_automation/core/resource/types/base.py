"""基础资源类"""

import os
import abc
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Generic

logger = logging.getLogger(__name__)
T = TypeVar('T')


class BaseResource(abc.ABC):
    """基础资源类
    
    所有资源类型都应该继承自这个类。
    """
    
    def __init__(
        self,
        key: str,
        path: str,
        data: Optional[Any] = None,
        **kwargs
    ):
        """初始化资源
        
        Args:
            key: 资源键
            path: 资源路径
            data: 资源数据
            **kwargs: 其他参数
        """
        self._key = key
        self._path = Path(path)
        self._data = data
        self._loaded = False
        self._kwargs = kwargs
        
    @property
    def key(self) -> str:
        """获取资源键"""
        return self._key
        
    @property
    def path(self) -> Path:
        """获取资源路径"""
        return self._path
        
    @property
    def data(self) -> Optional[Any]:
        """获取资源数据"""
        return self._data
        
    @property
    def loaded(self) -> bool:
        """获取资源是否已加载"""
        return self._loaded
        
    @property
    def kwargs(self) -> Dict[str, Any]:
        """获取其他参数"""
        return self._kwargs
        
    async def load(self) -> None:
        """加载资源
        
        这个方法应该由子类实现。
        """
        if self.loaded:
            logger.warning(f"Resource {self.key} is already loaded")
            return
            
        try:
            await self._load()
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load resource {self.key}: {e}")
            raise
            
    async def unload(self) -> None:
        """卸载资源
        
        这个方法应该由子类实现。
        """
        if not self.loaded:
            logger.warning(f"Resource {self.key} is not loaded")
            return
            
        try:
            await self._unload()
            self._loaded = False
            self._data = None
        except Exception as e:
            logger.error(f"Failed to unload resource {self.key}: {e}")
            raise
            
    async def reload(self) -> None:
        """重新加载资源"""
        await self.unload()
        await self.load()
        
    async def verify(self) -> bool:
        """验证资源
        
        这个方法应该由子类实现。
        
        Returns:
            验证是否通过
        """
        if not self.loaded:
            logger.warning(f"Resource {self.key} is not loaded")
            return False
            
        try:
            return await self._verify()
        except Exception as e:
            logger.error(f"Failed to verify resource {self.key}: {e}")
            return False
            
    @abc.abstractmethod
    async def _load(self) -> None:
        """加载资源的具体实现"""
        pass
        
    @abc.abstractmethod
    async def _unload(self) -> None:
        """卸载资源的具体实现"""
        pass
        
    @abc.abstractmethod
    async def _verify(self) -> bool:
        """验证资源的具体实现"""
        pass
