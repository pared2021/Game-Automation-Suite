"""内存缓存测试"""

import time
import pytest
import threading
from typing import Dict, Any

from game_automation.core.resource.cache.memory import LRUCache, CacheStats


def test_cache_basic():
    """测试基本缓存操作"""
    cache = LRUCache(max_size=2)
    
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


def test_cache_lru():
    """测试 LRU 淘汰策略"""
    cache = LRUCache(max_size=2)
    
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


def test_cache_ttl():
    """测试缓存过期"""
    cache = LRUCache(max_size=2, ttl=1)  # 1秒后过期
    
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


def test_cache_thread_safety():
    """测试线程安全"""
    cache = LRUCache(max_size=1000)
    num_threads = 10
    num_operations = 1000
    
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


def test_cache_cleanup():
    """测试自动清理"""
    cache = LRUCache(max_size=2, ttl=1, cleanup_interval=1)
    
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


def test_cache_clear():
    """测试清除缓存"""
    cache = LRUCache(max_size=2)
    
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


def test_cache_stats():
    """测试缓存统计"""
    cache = LRUCache(max_size=2)
    
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
    assert 0 < stats.hit_rate() < 1
    assert 0 < stats.eviction_rate() < 1
