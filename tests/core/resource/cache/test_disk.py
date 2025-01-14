"""磁盘缓存测试"""

import os
import time
import json
import pytest
import threading
import tempfile
from pathlib import Path
from typing import Dict, Any

from game_automation.core.resource.cache.disk import DiskCache, DiskCacheStats


@pytest.fixture
def cache_dir():
    """创建临时缓存目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_cache_basic(cache_dir):
    """测试基本缓存操作"""
    cache = DiskCache(cache_dir, max_size=2)
    
    # 添加缓存项
    cache.put('key1', 'value1')
    cache.put('key2', 'value2')
    
    # 验证缓存项
    assert cache.get('key1') == 'value1'
    assert cache.get('key2') == 'value2'
    assert cache.get('key3') is None
    
    # 验证统计信息
    stats = cache.get_stats()
    assert stats.hits == 2
    assert stats.misses == 1
    assert stats.size == 2
    assert stats.max_size == 2
    assert stats.disk_size > 0
    assert stats.max_disk_size > 0


def test_cache_size_limit(cache_dir):
    """测试缓存大小限制"""
    cache = DiskCache(cache_dir, max_size=2)
    
    # 添加缓存项
    cache.put('key1', 'value1')
    cache.put('key2', 'value2')
    cache.put('key3', 'value3')  # key1 应该被淘汰
    
    # 验证缓存项
    assert cache.get('key1') is None
    assert cache.get('key2') == 'value2'
    assert cache.get('key3') == 'value3'
    
    # 验证统计信息
    stats = cache.get_stats()
    assert stats.evictions == 1
    assert stats.size == 2


def test_cache_disk_limit(cache_dir):
    """测试磁盘大小限制"""
    # 设置很小的磁盘限制
    cache = DiskCache(
        cache_dir,
        max_size=10,
        max_disk_size=100  # 100字节
    )
    
    # 添加大缓存项
    large_data = 'x' * 200  # 200字节
    cache.put('key1', large_data)
    
    # 验证缓存项被淘汰
    assert cache.get('key1') is None
    
    # 验证统计信息
    stats = cache.get_stats()
    assert stats.evictions > 0
    assert stats.disk_size < 100


def test_cache_ttl(cache_dir):
    """测试缓存过期"""
    cache = DiskCache(cache_dir, max_size=2, ttl=1)  # 1秒后过期
    
    # 添加缓存项
    cache.put('key1', 'value1')
    assert cache.get('key1') == 'value1'
    
    # 等待缓存过期
    time.sleep(1.1)
    
    # 验证缓存项
    assert cache.get('key1') is None
    
    # 验证统计信息
    stats = cache.get_stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.evictions == 1


def test_cache_thread_safety(cache_dir):
    """测试线程安全"""
    cache = DiskCache(cache_dir, max_size=1000)
    num_threads = 10
    num_operations = 100
    
    def worker():
        for i in range(num_operations):
            key = f'key{i}'
            cache.put(key, i)
            assert cache.get(key) == i
            
    # 创建多个线程
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()
        
    # 等待所有线程完成
    for thread in threads:
        thread.join()
        
    # 验证统计信息
    stats = cache.get_stats()
    assert stats.hits == num_threads * num_operations
    assert stats.size <= 1000


def test_cache_cleanup(cache_dir):
    """测试自动清理"""
    cache = DiskCache(
        cache_dir,
        max_size=2,
        ttl=1,
        cleanup_interval=1
    )
    
    # 添加缓存项
    cache.put('key1', 'value1')
    cache.put('key2', 'value2')
    
    # 等待清理
    time.sleep(1.5)
    
    # 验证缓存项
    assert cache.get('key1') is None
    assert cache.get('key2') is None
    
    # 验证统计信息
    stats = cache.get_stats()
    assert stats.evictions == 2
    assert stats.size == 0


def test_cache_clear(cache_dir):
    """测试清除缓存"""
    cache = DiskCache(cache_dir, max_size=2)
    
    # 添加缓存项
    cache.put('key1', 'value1')
    cache.put('key2', 'value2')
    
    # 清除缓存
    cache.clear()
    
    # 验证缓存项
    assert cache.get('key1') is None
    assert cache.get('key2') is None
    
    # 验证统计信息
    stats = cache.get_stats()
    assert stats.size == 0
    assert stats.hits == 0
    assert stats.misses == 2
    assert stats.disk_size == 0


def test_cache_persistence(cache_dir):
    """测试缓存持久化"""
    # 创建缓存并添加数据
    cache1 = DiskCache(cache_dir, max_size=2)
    cache1.put('key1', 'value1')
    cache1.put('key2', 'value2')
    
    # 创建新的缓存实例
    cache2 = DiskCache(cache_dir, max_size=2)
    
    # 验证数据仍然存在
    assert cache2.get('key1') == 'value1'
    assert cache2.get('key2') == 'value2'


def test_cache_stats(cache_dir):
    """测试缓存统计"""
    cache = DiskCache(cache_dir, max_size=2)
    
    # 添加和获取缓存项
    cache.put('key1', 'value1')
    cache.get('key1')  # hit
    cache.get('key2')  # miss
    cache.put('key2', 'value2')
    cache.put('key3', 'value3')  # evict key1
    
    # 验证统计信息
    stats = cache.get_stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.evictions == 1
    assert stats.size == 2
    assert stats.max_size == 2
    assert stats.disk_size > 0
    assert stats.max_disk_size > 0
    assert 0 < stats.hit_rate() < 1
    assert 0 < stats.eviction_rate() < 1
