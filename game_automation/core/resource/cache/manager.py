"""缓存管理器"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Callable

from .memory import LRUCache
from .disk import DiskCache
from .enhancer import CacheEnhancer

logger = logging.getLogger(__name__)


class CacheManager:
    """缓存管理器
    
    特性：
    - 支持内存缓存和磁盘缓存
    - 支持 TTL（过期时间）
    - 支持缓存大小限制
    - 支持磁盘空间限制
    - 支持自动清理
    - 支持持久化
    - 支持缓存压缩
    - 支持缓存验证
    - 支持缓存预热
    - 支持缓存监控
    """
    
    def __init__(
        self,
        cache_dir: str,
        memory_max_size: int = 1000,
        disk_max_size: int = 1000,
        disk_max_disk_size: int = 1024 * 1024 * 1024,
        ttl: Optional[float] = None,
        cleanup_interval: Optional[int] = None,
        compression_level: int = 6,
        compression_threshold: int = 1024,
        verify_data: bool = True,
        preload_keys: Optional[List[str]] = None,
        preload_callback: Optional[Callable[[str], Any]] = None,
        monitor_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            memory_max_size: 内存缓存最大项数
            disk_max_size: 磁盘缓存最大项数
            disk_max_disk_size: 磁盘缓存最大空间（字节）
            ttl: 过期时间（秒）
            cleanup_interval: 清理间隔（秒）
            compression_level: 压缩级别 (0-9)
            compression_threshold: 压缩阈值（字节）
            verify_data: 是否验证数据
            preload_keys: 预加载的键列表
            preload_callback: 预加载回调函数
            monitor_callback: 监控回调函数
        """
        self._memory_cache = LRUCache(
            max_size=memory_max_size,
            ttl=ttl,
            cleanup_interval=cleanup_interval
        )
        self._disk_cache = DiskCache(
            cache_dir=cache_dir,
            max_size=disk_max_size,
            max_disk_size=disk_max_disk_size,
            ttl=ttl,
            cleanup_interval=cleanup_interval
        )
        self._enhancer = CacheEnhancer(
            compression_level=compression_level,
            compression_threshold=compression_threshold,
            verify_data=verify_data,
            preload_keys=preload_keys,
            preload_callback=preload_callback,
            monitor_callback=monitor_callback
        )
        
        # 预加载缓存
        self._enhancer.preload()
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值
        """
        # 先从内存缓存获取
        value = self._memory_cache.get(key)
        if value is not None:
            # 验证数据
            if not self._enhancer.verify(key, value):
                logger.warning(f"Cache verification failed for key: {key}")
                self.remove(key)
                return None
                
            # 监控访问
            stats = self._memory_cache.get_stats()
            self._enhancer.monitor(key, {
                "source": "memory",
                "hit": True,
                "stats": stats.__dict__
            })
            return value
            
        # 再从磁盘缓存获取
        value = self._disk_cache.get(key)
        if value is not None:
            # 解压数据
            try:
                is_compressed = isinstance(value, tuple) and len(value) == 2
                if is_compressed:
                    value = self._enhancer.decompress(*value)
            except Exception as e:
                logger.error(f"Failed to decompress cache value for key {key}: {e}")
                self.remove(key)
                return None
                
            # 验证数据
            if not self._enhancer.verify(key, value):
                logger.warning(f"Cache verification failed for key: {key}")
                self.remove(key)
                return None
                
            # 写入内存缓存
            self._memory_cache.put(key, value)
            
            # 监控访问
            stats = self._disk_cache.get_stats()
            self._enhancer.monitor(key, {
                "source": "disk",
                "hit": True,
                "stats": stats.__dict__
            })
            return value
            
        # 监控未命中
        self._enhancer.monitor(key, {
            "source": "none",
            "hit": False
        })
        return None
        
    def put(self, key: str, value: Any) -> None:
        """添加缓存项
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        # 压缩数据
        compressed_value = self._enhancer.compress(value)
        
        # 写入内存缓存
        self._memory_cache.put(key, value)
        
        # 写入磁盘缓存
        self._disk_cache.put(key, compressed_value)
        
        # 监控写入
        stats = {
            "memory_stats": self._memory_cache.get_stats().__dict__,
            "disk_stats": self._disk_cache.get_stats().__dict__
        }
        self._enhancer.monitor(key, {
            "operation": "put",
            "stats": stats
        })
        
    def remove(self, key: str) -> None:
        """移除缓存项
        
        Args:
            key: 缓存键
        """
        self._memory_cache.remove(key)
        self._disk_cache.remove(key)
        
        # 监控删除
        self._enhancer.monitor(key, {
            "operation": "remove"
        })
        
    def clear(self) -> None:
        """清除所有缓存项"""
        self._memory_cache.clear()
        self._disk_cache.clear()
        
        # 监控清除
        self._enhancer.monitor("all", {
            "operation": "clear"
        })
        
    def get_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        memory_stats = self._memory_cache.get_stats()
        disk_stats = self._disk_cache.get_stats()
        
        return {
            'memory': {
                'hits': memory_stats.hits,
                'misses': memory_stats.misses,
                'evictions': memory_stats.evictions,
                'size': memory_stats.size,
                'max_size': memory_stats.max_size,
                'hit_rate': memory_stats.hit_rate(),
                'eviction_rate': memory_stats.eviction_rate()
            },
            'disk': {
                'hits': disk_stats.hits,
                'misses': disk_stats.misses,
                'evictions': disk_stats.evictions,
                'size': disk_stats.size,
                'max_size': disk_stats.max_size,
                'hit_rate': disk_stats.hit_rate(),
                'eviction_rate': disk_stats.eviction_rate(),
                'disk_size': disk_stats.disk_size,
                'max_disk_size': disk_stats.max_disk_size
            }
        }
        
    def get_resource_key(self, resource: Any) -> str:
        """获取资源键
        
        Args:
            resource: 资源对象
            
        Returns:
            资源键
        """
        return f'resource:{hash(resource)}'
        
    def get_loader_key(self, loader: Any) -> str:
        """获取加载器键
        
        Args:
            loader: 加载器对象
            
        Returns:
            加载器键
        """
        return f'loader:{hash(loader)}'
