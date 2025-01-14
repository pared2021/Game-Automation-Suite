"""资源管理系统基础类和接口"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Optional
from datetime import datetime


class ResourceState(Enum):
    """资源状态枚举"""
    UNLOADED = auto()  # 未加载
    LOADING = auto()   # 加载中
    LOADED = auto()    # 已加载
    UNLOADING = auto() # 释放中
    ERROR = auto()     # 错误状态


class ResourceType(Enum):
    """资源类型枚举"""
    IMAGE = auto()    # 图像资源
    CONFIG = auto()   # 配置资源
    MODEL = auto()    # 模型资源
    CACHE = auto()    # 缓存资源
    OTHER = auto()    # 其他资源


class ResourceBase(ABC):
    """资源基类
    
    所有具体的资源类型都应该继承自这个基类。
    """
    
    def __init__(
        self,
        key: str,
        type: ResourceType,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """初始化资源
        
        Args:
            key: 资源标识符
            type: 资源类型
            metadata: 资源元数据
        """
        self._key = key
        self._type = type
        self._metadata = metadata or {}
        self._state = ResourceState.UNLOADED
        self._ref_count = 0
        self._created_at = datetime.now()
        self._last_accessed = self._created_at
        self._error: Optional[Exception] = None
        
    @property
    def key(self) -> str:
        """获取资源标识符"""
        return self._key
        
    @property
    def type(self) -> ResourceType:
        """获取资源类型"""
        return self._type
        
    @property
    def state(self) -> ResourceState:
        """获取资源状态"""
        return self._state
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """获取资源元数据"""
        return self._metadata.copy()
        
    @property
    def ref_count(self) -> int:
        """获取引用计数"""
        return self._ref_count
        
    @property
    def created_at(self) -> datetime:
        """获取创建时间"""
        return self._created_at
        
    @property
    def last_accessed(self) -> datetime:
        """获取最后访问时间"""
        return self._last_accessed
        
    @property
    def error(self) -> Optional[Exception]:
        """获取错误信息"""
        return self._error
        
    def increment_ref(self) -> int:
        """增加引用计数
        
        Returns:
            新的引用计数
        """
        self._ref_count += 1
        self._last_accessed = datetime.now()
        return self._ref_count
        
    def decrement_ref(self) -> int:
        """减少引用计数
        
        Returns:
            新的引用计数
        """
        if self._ref_count > 0:
            self._ref_count -= 1
        return self._ref_count
        
    @abstractmethod
    async def _do_load(self) -> None:
        """执行实际的加载操作
        
        由子类实现具体的加载逻辑。
        """
        pass
        
    @abstractmethod
    async def _do_unload(self) -> None:
        """执行实际的释放操作
        
        由子类实现具体的释放逻辑。
        """
        pass
        
    async def load(self) -> None:
        """加载资源
        
        加载过程中会更新资源状态，处理错误情况。
        """
        if self._state == ResourceState.LOADED:
            return
            
        try:
            self._state = ResourceState.LOADING
            await self._do_load()
            self._state = ResourceState.LOADED
            self._error = None
        except Exception as e:
            self._state = ResourceState.ERROR
            self._error = e
            raise
            
    async def unload(self) -> None:
        """释放资源
        
        释放过程中会更新资源状态，处理错误情况。
        """
        if self._state == ResourceState.UNLOADED:
            return
            
        try:
            self._state = ResourceState.UNLOADING
            await self._do_unload()
            self._state = ResourceState.UNLOADED
            self._error = None
        except Exception as e:
            self._state = ResourceState.ERROR
            self._error = e
            raise
