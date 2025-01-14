"""缓存资源测试"""

import os
import json
import pickle
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

from game_automation.core.resource.types.cache import CacheResource
from game_automation.core.resource.errors import ResourceLoadError


@pytest.fixture
def test_json_cache(tmp_path) -> Path:
    """创建测试 JSON 缓存文件"""
    cache_path = tmp_path / "test.json"
    data = {
        "name": "test",
        "value": 123
    }
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return cache_path


@pytest.fixture
def test_pickle_cache(tmp_path) -> Path:
    """创建测试 Pickle 缓存文件"""
    cache_path = tmp_path / "test.pkl"
    data = {
        "name": "test",
        "value": 123
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    return cache_path


@pytest.fixture
def test_custom_cache(tmp_path) -> Path:
    """创建测试自定义缓存文件"""
    cache_path = tmp_path / "test.custom"
    data = b'test data'
    with open(cache_path, 'wb') as f:
        f.write(data)
    return cache_path


@pytest.mark.asyncio
async def test_load_json_cache(test_json_cache):
    """测试加载 JSON 缓存"""
    resource = CacheResource('test', str(test_json_cache))
    
    # 测试加载前的状态
    assert resource.data is None
    
    # 加载缓存
    await resource.load()
    
    # 测试加载后的状态
    assert resource.data is not None
    assert resource.data['name'] == 'test'
    assert resource.data['value'] == 123


@pytest.mark.asyncio
async def test_load_pickle_cache(test_pickle_cache):
    """测试加载 Pickle 缓存"""
    resource = CacheResource('test', str(test_pickle_cache))
    await resource.load()
    
    assert resource.data is not None
    assert resource.data['name'] == 'test'
    assert resource.data['value'] == 123


@pytest.mark.asyncio
async def test_load_custom_cache(test_custom_cache):
    """测试加载自定义缓存"""
    def deserializer(data: bytes) -> str:
        return data.decode('utf-8')
        
    resource = CacheResource(
        'test',
        str(test_custom_cache),
        deserializer=deserializer
    )
    await resource.load()
    
    assert resource.data == 'test data'


@pytest.mark.asyncio
async def test_save_cache(tmp_path):
    """测试保存缓存"""
    cache_path = tmp_path / "test.json"
    resource = CacheResource('test', str(cache_path))
    
    # 保存数据
    data = {"name": "test", "value": 123}
    await resource.save(data)
    
    # 验证文件是否存在
    assert cache_path.exists()
    
    # 验证数据是否正确保存
    with open(cache_path, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
        assert loaded_data == data


@pytest.mark.asyncio
async def test_cache_compression(tmp_path):
    """测试缓存压缩"""
    cache_path = tmp_path / "test.json"
    resource = CacheResource('test', str(cache_path), compress=True)
    
    # 保存数据
    data = {"name": "test", "value": "x" * 1000}  # 创建大数据
    await resource.save(data)
    
    # 加载压缩的数据
    await resource.load()
    assert resource.data == data
    
    # 验证文件大小是否减小
    uncompressed_size = len(json.dumps(data).encode('utf-8'))
    compressed_size = cache_path.stat().st_size
    assert compressed_size < uncompressed_size


@pytest.mark.asyncio
async def test_cache_ttl(tmp_path):
    """测试缓存过期"""
    cache_path = tmp_path / "test.json"
    resource = CacheResource('test', str(cache_path), ttl=1)  # 1秒后过期
    
    # 保存数据
    data = {"name": "test"}
    await resource.save(data)
    
    # 验证未过期的缓存
    assert not resource.is_expired
    assert resource.verify()
    
    # 等待缓存过期
    await asyncio.sleep(1.1)
    
    # 验证已过期的缓存
    assert resource.is_expired
    assert not resource.verify()


@pytest.mark.asyncio
async def test_cache_verification(test_json_cache):
    """测试缓存验证"""
    resource = CacheResource('test', str(test_json_cache))
    
    # 未加载时验证应该失败
    assert not resource.verify()
    
    # 加载后验证应该成功
    await resource.load()
    assert resource.verify()
    
    # 清除后验证应该失败
    resource.clear()
    assert not resource.verify()


@pytest.mark.asyncio
async def test_load_nonexistent_cache():
    """测试加载不存在的缓存"""
    resource = CacheResource('test', 'nonexistent.json')
    with pytest.raises(ResourceLoadError):
        await resource.load()


@pytest.mark.asyncio
async def test_unload_cache(test_json_cache):
    """测试释放缓存"""
    resource = CacheResource('test', str(test_json_cache))
    await resource.load()
    
    # 测试释放前的状态
    assert resource.data is not None
    
    # 释放缓存
    await resource.unload()
    
    # 测试释放后的状态
    assert resource.data is None


@pytest.mark.asyncio
async def test_clear_cache(tmp_path):
    """测试清除缓存"""
    cache_path = tmp_path / "test.json"
    resource = CacheResource('test', str(cache_path))
    
    # 保存数据
    data = {"name": "test"}
    await resource.save(data)
    
    # 验证文件存在
    assert cache_path.exists()
    
    # 清除缓存
    resource.clear()
    
    # 验证文件已删除
    assert not cache_path.exists()
    assert resource.data is None
