"""磁盘缓存实现"""

import os
import json
import time
import shutil
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Set, Callable

from .base import CacheStats
from .compression import CacheCompressor
from .validation import CacheValidator
from .storage import CacheStorage

logger = logging.getLogger(__name__)


class DiskCacheStats(CacheStats):
    """磁盘缓存统计信息"""
    
    def __init__(self, max_size: int = 1000, max_disk_size: int = 1024 * 1024 * 1024):
        super().__init__(max_size)
        self.disk_size = 0
        self.max_disk_size = max_disk_size
        self.storage_stats = {}
        

class DiskCache:
    """磁盘缓存
    
    特性：
    - 支持 TTL（过期时间）
    - 支持缓存大小限制
    - 支持磁盘空间限制
    - 支持自动清理
    - 支持持久化
    - 支持压缩
    - 支持验证
    - 支持分片
    """
    
    def __init__(
        self,
        cache_dir: str,
        max_size: int = 1000,
        max_disk_size: int = 1024 * 1024 * 1024,
        ttl: Optional[float] = None,
        cleanup_interval: Optional[int] = None,
        compression_level: int = 6,
        compression_threshold: int = 1024,
        auto_compress: bool = True,
        verify_data: bool = True,
        verify_interval: Optional[int] = None,
        custom_validator: Optional[Callable[[str, Any], bool]] = None,
        max_chunk_size: int = 1024 * 1024  # 1MB
    ):
        """初始化磁盘缓存
        
        Args:
            cache_dir: 缓存目录
            max_size: 最大缓存项数
            max_disk_size: 最大磁盘空间（字节）
            ttl: 过期时间（秒）
            cleanup_interval: 清理间隔（秒）
            compression_level: 压缩级别 (0-9)
            compression_threshold: 压缩阈值（字节）
            auto_compress: 是否自动压缩
            verify_data: 是否验证数据
            verify_interval: 验证间隔（秒）
            custom_validator: 自定义验证函数
            max_chunk_size: 最大分片大小（字节）
        """
        self._ttl = ttl
        self._lock = threading.Lock()
        self._stats = DiskCacheStats(max_size, max_disk_size)
        self._keys: Set[str] = set()
        
        # 初始化存储
        self._storage = CacheStorage(
            cache_dir,
            max_chunk_size=max_chunk_size,
            max_storage_size=max_disk_size
        )
        
        # 初始化压缩器
        self._compressor = CacheCompressor(
            compression_level,
            compression_threshold,
            auto_compress
        )
        
        # 初始化验证器
        self._validator = CacheValidator(
            verify_data,
            verify_interval,
            custom_validator
        )
        
        # 加载缓存
        self._load_cache()
        
        # 启动清理线程
        if cleanup_interval and ttl:
            self._start_cleanup_thread(cleanup_interval)
            
    def _load_cache(self) -> None:
        """加载缓存"""
        # 遍历存储目录
        storage_dir = Path(self._storage._storage_dir)
        for metadata_file in storage_dir.rglob('metadata.json'):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    key = metadata['key']
                    self._keys.add(key)
                    self._stats.size += 1
            except Exception as e:
                logger.error(f'Failed to load cache file {metadata_file}: {e}')
                
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值
        """
        with self._lock:
            # 从存储中读取数据
            data = self._storage.read(key)
            if data is None:
                self._stats.misses += 1
                return None
                
            try:
                # 解压数据
                cache_data = self._compressor.decompress(data)
                
                # 检查过期时间
                if self._ttl is not None:
                    if time.time() - cache_data['timestamp'] > self._ttl:
                        self._remove(key)
                        self._stats.misses += 1
                        return None
                        
                # 验证数据
                if not self._validator.validate(key, cache_data['value']):
                    logger.warning(f'Cache validation failed for key: {key}')
                    self._remove(key)
                    self._validator.invalidate(key)
                    self._stats.misses += 1
                    return None
                    
                self._stats.hits += 1
                return cache_data['value']
                
            except Exception as e:
                logger.error(f'Failed to get cache data for key {key}: {e}')
                self._stats.misses += 1
                return None
                
    def put(self, key: str, value: Any) -> None:
        """添加缓存项
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            # 如果键已存在，更新值
            if key in self._keys:
                self._remove(key)
                self._validator.invalidate(key)
                
            # 检查缓存大小
            if self._stats.size >= self._stats.max_size:
                self._remove_oldest()
                
            # 检查磁盘配额
            while self._storage.get_storage_size() >= self._stats.max_disk_size:
                self._remove_oldest()
                
            # 准备缓存数据
            cache_data = {
                'key': key,
                'value': value,
                'timestamp': time.time()
            }
            
            # 压缩数据
            compressed_data = self._compressor.compress(cache_data)
            
            # 写入存储
            if self._storage.write(key, compressed_data):
                self._keys.add(key)
                self._stats.size += 1
                self._stats.disk_size = self._storage.get_storage_size()
                
    def remove(self, key: str) -> None:
        """移除缓存项
        
        Args:
            key: 缓存键
        """
        with self._lock:
            self._remove(key)
            self._validator.invalidate(key)
            
    def _remove(self, key: str) -> None:
        """移除缓存项
        
        Args:
            key: 缓存键
        """
        if key in self._keys:
            if self._storage.delete(key):
                self._keys.remove(key)
                self._stats.size -= 1
                self._stats.evictions += 1
                self._stats.disk_size = self._storage.get_storage_size()
                
    def _remove_oldest(self) -> None:
        """移除最旧的缓存项"""
        oldest_key = None
        oldest_time = float('inf')
        
        # 遍历所有缓存项
        for key in self._keys:
            data = self._storage.read(key)
            if data is not None:
                try:
                    cache_data = self._compressor.decompress(data)
                    timestamp = cache_data['timestamp']
                    if timestamp < oldest_time:
                        oldest_time = timestamp
                        oldest_key = key
                except Exception as e:
                    logger.error(f'Failed to read cache data for key {key}: {e}')
                    
        # 移除最旧的缓存项
        if oldest_key is not None:
            self._remove(oldest_key)
            
    def clear(self) -> None:
        """清除所有缓存项"""
        with self._lock:
            if self._storage.clear():
                self._keys.clear()
                self._stats.size = 0
                self._stats.disk_size = 0
                self._validator.clear_cache()
                
    def cleanup(self) -> None:
        """清理过期的缓存项"""
        with self._lock:
            for key in list(self._keys):
                data = self._storage.read(key)
                if data is not None:
                    try:
                        cache_data = self._compressor.decompress(data)
                        if self._ttl is not None:
                            if time.time() - cache_data['timestamp'] > self._ttl:
                                self._remove(key)
                    except Exception as e:
                        logger.error(f'Failed to cleanup cache data for key {key}: {e}')
                        self._remove(key)
                        
    def get_stats(self) -> CacheStats:
        """获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        # 更新存储统计信息
        self._stats.storage_stats = self._storage.get_stats()
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
            name='DiskCache-Cleanup',
            daemon=True
        )
        thread.start()
