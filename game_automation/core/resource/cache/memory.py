"""内存缓存实现"""

import sys
import time
import threading
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime

from .base import CacheStats


class LRUCache:
    """LRU 缓存
    
    特性：
    - 支持 LRU（最近最少使用）淘汰策略
    - 支持 TTL（过期时间）
    - 支持缓存大小限制
    - 支持自动清理
    - 支持详细统计
    - 支持内存限制
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory: Optional[int] = None,
        ttl: Optional[float] = None,
        cleanup_interval: Optional[int] = None
    ):
        """初始化 LRU 缓存
        
        Args:
            max_size: 最大缓存项数
            max_memory: 最大内存使用量（字节）
            ttl: 过期时间（秒）
            cleanup_interval: 清理间隔（秒）
        """
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._access_times: Dict[str, float] = {}
        self._sizes: Dict[str, int] = {}
        self._max_size = max_size
        self._max_memory = max_memory
        self._ttl = ttl
        self._lock = threading.Lock()
        self._stats = CacheStats(max_size)
        
        # 启动清理线程
        if cleanup_interval and (ttl or max_memory):
            self._start_cleanup_thread(cleanup_interval)
            
    def _get_size(self, value: Any) -> int:
        """获取对象大小
        
        Args:
            value: 对象
            
        Returns:
            对象大小（字节）
        """
        return sys.getsizeof(value)
        
    def _remove(self, key: str) -> None:
        """移除缓存项
        
        Args:
            key: 缓存键
        """
        if key in self._cache:
            # 更新统计信息
            self._stats.total_size -= self._sizes[key]
            self._stats.size -= 1
            self._stats.evictions += 1
            self._stats.record_hourly(evict=True)
            
            # 移除缓存项
            del self._cache[key]
            del self._timestamps[key]
            del self._access_times[key]
            del self._sizes[key]
            
    def _remove_lru(self) -> None:
        """移除最近最少使用的缓存项"""
        if not self._cache:
            return
            
        lru_key = min(
            self._access_times.keys(),
            key=lambda k: self._access_times[k]
        )
        self._remove(lru_key)
        
    def _ensure_memory_limit(self, new_size: int) -> None:
        """确保内存使用量不超过限制
        
        Args:
            new_size: 新对象大小
        """
        if not self._max_memory:
            return
            
        while (self._stats.total_size + new_size) > self._max_memory and self._cache:
            self._remove_lru()
            
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值
        """
        start_time = time.time()
        
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                self._stats.record_hourly(miss=True)
                self._stats.record_access(start_time)
                return None
                
            # 检查过期时间
            if self._ttl is not None:
                if time.time() - self._timestamps[key] > self._ttl:
                    self._remove(key)
                    self._stats.misses += 1
                    self._stats.expired += 1
                    self._stats.record_hourly(miss=True)
                    self._stats.record_access(start_time)
                    return None
                    
            # 更新访问时间和统计信息
            self._access_times[key] = time.time()
            self._stats.hits += 1
            self._stats.record_hourly(hit=True)
            self._stats.record_access(start_time)
            
            return self._cache[key]
            
    def put(self, key: str, value: Any) -> None:
        """添加缓存项
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            # 计算对象大小
            size = self._get_size(value)
            
            # 如果键已存在，更新值
            if key in self._cache:
                old_size = self._sizes[key]
                self._stats.total_size -= old_size
                self._stats.updates += 1
                self._remove(key)
                
            # 确保内存限制
            self._ensure_memory_limit(size)
            
            # 如果缓存已满，移除最近最少使用的项
            while len(self._cache) >= self._max_size:
                self._remove_lru()
                
            # 添加新项
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._access_times[key] = time.time()
            self._sizes[key] = size
            self._stats.size += 1
            self._stats.total_size += size
            
    def remove(self, key: str) -> None:
        """移除缓存项
        
        Args:
            key: 缓存键
        """
        with self._lock:
            self._remove(key)
            
    def clear(self) -> None:
        """清除所有缓存项"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._access_times.clear()
            self._sizes.clear()
            self._stats.clear()
            
    def cleanup(self) -> None:
        """清理过期和超出内存限制的缓存项"""
        with self._lock:
            current_time = time.time()
            
            # 清理过期项
            if self._ttl is not None:
                expired_keys = [
                    key for key, timestamp in self._timestamps.items()
                    if current_time - timestamp > self._ttl
                ]
                for key in expired_keys:
                    self._remove(key)
                    
            # 清理超出内存限制的项
            if self._max_memory and self._stats.total_size > self._max_memory:
                while self._stats.total_size > self._max_memory and self._cache:
                    self._remove_lru()
                    
            self._stats.last_cleanup = datetime.now()
            
    def get_stats(self) -> CacheStats:
        """获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        return self._stats
        
    def _start_cleanup_thread(self, cleanup_interval: int) -> None:
        """启动清理线程
        
        Args:
            cleanup_interval: 清理间隔（秒）
        """
        def cleanup() -> None:
            while True:
                time.sleep(cleanup_interval)
                self.cleanup()
                
        thread = threading.Thread(
            target=cleanup,
            name='LRUCache-Cleanup',
            daemon=True
        )
        thread.start()
