"""缓存管理器测试"""

import os
import time
import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from game_automation.core.resource.cache.manager import CacheManager


@pytest.fixture
def cache_dir():
    """创建临时缓存目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_cache_basic(cache_dir):
    """测试基本缓存操作"""
    manager = CacheManager(cache_dir)
    
    # 添加缓存项
    manager.put('key1', 'value1')
    manager.put('key2', 'value2')
    
    # 验证缓存项
    assert manager.get('key1') == 'value1'
    assert manager.get('key2') == 'value2'
    assert manager.get('key3') is None
    
    # 验证统计信息
    stats = manager.get_stats()
    assert stats['memory']['hits'] == 2
    assert stats['memory']['misses'] == 1
    assert stats['disk']['hits'] == 0
    assert stats['disk']['misses'] == 0


def test_cache_levels(cache_dir):
    """测试多级缓存"""
    manager = CacheManager(cache_dir)
    
    # 添加缓存项
    manager.put('key1', 'value1')
    
    # 从内存缓存获取
    assert manager.get('key1') == 'value1'
    
    # 清除内存缓存
    manager._memory_cache.clear()
    
    # 从磁盘缓存获取
    assert manager.get('key1') == 'value1'
    
    # 验证统计信息
    stats = manager.get_stats()
    assert stats['memory']['hits'] == 1
    assert stats['disk']['hits'] == 1


def test_cache_ttl(cache_dir):
    """测试缓存过期"""
    manager = CacheManager(cache_dir, ttl=1)  # 1秒后过期
    
    # 添加缓存项
    manager.put('key1', 'value1')
    assert manager.get('key1') == 'value1'
    
    # 等待缓存过期
    time.sleep(1.1)
    
    # 验证缓存项
    assert manager.get('key1') is None
    
    # 验证统计信息
    stats = manager.get_stats()
    assert stats['memory']['evictions'] > 0
    assert stats['disk']['evictions'] > 0


def test_cache_remove(cache_dir):
    """测试移除缓存"""
    manager = CacheManager(cache_dir)
    
    # 添加缓存项
    manager.put('key1', 'value1')
    assert manager.get('key1') == 'value1'
    
    # 移除缓存项
    manager.remove('key1')
    
    # 验证缓存项
    assert manager.get('key1') is None
    
    # 验证统计信息
    stats = manager.get_stats()
    assert stats['memory']['evictions'] == 1
    assert stats['disk']['evictions'] == 1


def test_cache_clear(cache_dir):
    """测试清除缓存"""
    manager = CacheManager(cache_dir)
    
    # 添加缓存项
    manager.put('key1', 'value1')
    manager.put('key2', 'value2')
    
    # 清除缓存
    manager.clear()
    
    # 验证缓存项
    assert manager.get('key1') is None
    assert manager.get('key2') is None
    
    # 验证统计信息
    stats = manager.get_stats()
    assert stats['memory']['size'] == 0
    assert stats['disk']['size'] == 0


def test_cache_stats(cache_dir):
    """测试缓存统计"""
    manager = CacheManager(cache_dir)
    
    # 添加和获取缓存项
    manager.put('key1', 'value1')
    manager.get('key1')  # memory hit
    manager.get('key2')  # memory miss, disk miss
    
    # 清除内存缓存
    manager._memory_cache.clear()
    
    # 从磁盘获取
    manager.get('key1')  # memory miss, disk hit
    
    # 验证统计信息
    stats = manager.get_stats()
    assert stats['memory']['hits'] == 1
    assert stats['memory']['misses'] == 2
    assert stats['disk']['hits'] == 1
    assert stats['disk']['misses'] == 1
    assert 0 < stats['memory']['hit_rate'] < 1
    assert 0 < stats['disk']['hit_rate'] < 1


def test_resource_key(cache_dir):
    """测试资源缓存键"""
    manager = CacheManager(cache_dir)
    
    # 创建模拟资源
    class MockResource:
        def __init__(self, key: str, path: str):
            self.key = key
            self.path = path
            
    resource = MockResource('test', '/path/to/resource')
    
    # 验证缓存键
    key = manager.get_resource_key(resource)
    assert key == 'resource:test:/path/to/resource'


def test_loader_key(cache_dir):
    """测试加载器缓存键"""
    manager = CacheManager(cache_dir)
    
    # 创建模拟加载器
    class MockLoader:
        def __init__(self, base_path: str):
            self.base_path = base_path
            
    loader = MockLoader('/path/to/loader')
    
    # 验证缓存键
    key = manager.get_loader_key(loader)
    assert key == 'loader:MockLoader:/path/to/loader'
