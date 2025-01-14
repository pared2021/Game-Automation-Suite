"""缓存管理实现"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class CacheManager:
    """缓存管理器
    
    负责资源的缓存管理，支持内存缓存和磁盘缓存。
    """
    
    def __init__(self, max_memory_items: int = 1000):
        """初始化缓存管理器
        
        Args:
            max_memory_items: 最大内存缓存项数
        """
        self._memory_cache: Dict[str, Any] = {}
        self._access_times: Dict[str, datetime] = {}
        self._max_memory_items = max_memory_items
        
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存项
        
        Args:
            key: 缓存项标识符
            
        Returns:
            缓存项值，如果不存在则返回 None
        """
        if key in self._memory_cache:
            self._access_times[key] = datetime.now()
            return self._memory_cache[key]
        return None
        
    async def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """存储缓存项
        
        Args:
            key: 缓存项标识符
            value: 缓存项值
            ttl: 生存时间（秒）
        """
        if len(self._memory_cache) >= self._max_memory_items:
            await self._evict_old_items()
            
        self._memory_cache[key] = value
        self._access_times[key] = datetime.now()
        
    async def invalidate(self, key: str) -> None:
        """使缓存项失效
        
        Args:
            key: 缓存项标识符
        """
        if key in self._memory_cache:
            del self._memory_cache[key]
            del self._access_times[key]
            
    async def _evict_old_items(self) -> None:
        """清理旧的缓存项"""
        if not self._memory_cache:
            return
            
        # 按最后访问时间排序
        sorted_items = sorted(
            self._access_times.items(),
            key=lambda x: x[1]
        )
        
        # 移除最旧的 10% 的项
        num_to_remove = max(1, len(sorted_items) // 10)
        for key, _ in sorted_items[:num_to_remove]:
            del self._memory_cache[key]
            del self._access_times[key]
