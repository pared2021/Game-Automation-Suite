"""缓存加载器测试"""

import os
import json
import pytest
from pathlib import Path
from typing import Dict, Any

from game_automation.core.resource.types.cache import CacheResource
from game_automation.core.resource.loaders.cache_loader import CacheLoader
from game_automation.core.resource.errors import ResourceLoadError


@pytest.fixture
def test_cache_path(tmp_path) -> Path:
    """创建测试缓存文件"""
    cache_path = tmp_path / "test.json"
    data = {
        "name": "test",
        "value": 123
    }
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return cache_path


@pytest.fixture
def cache_loader(tmp_path):
    """创建缓存加载器"""
    return CacheLoader(
        str(tmp_path),
        default_ttl=3600,  # 1小时
        compress=False
    )


@pytest.mark.asyncio
async def test_load_cache(cache_loader, test_cache_path):
    """测试加载缓存"""
    # 使用相对路径
    resource = await cache_loader.load(
        'test',
        CacheResource,
        path=test_cache_path.name
    )
    
    assert isinstance(resource, CacheResource)
    assert resource.key == 'test'
    assert resource.path == test_cache_path
    
    # 加载缓存
    await resource.load()
    assert resource.data is not None
    assert resource.data['name'] == 'test'


@pytest.mark.asyncio
async def test_load_with_absolute_path(cache_loader, test_cache_path):
    """测试使用绝对路径加载缓存"""
    resource = await cache_loader.load(
        'test',
        CacheResource,
        path=str(test_cache_path)
    )
    
    assert isinstance(resource, CacheResource)
    assert resource.path == test_cache_path


@pytest.mark.asyncio
async def test_load_with_ttl(cache_loader, test_cache_path):
    """测试使用指定 TTL 加载缓存"""
    resource = await cache_loader.load(
        'test',
        CacheResource,
        path=test_cache_path.name,
        ttl=60  # 1分钟
    )
    
    await resource.load()
    assert not resource.is_expired


@pytest.mark.asyncio
async def test_load_with_compression(cache_loader, test_cache_path):
    """测试使用压缩加载缓存"""
    resource = await cache_loader.load(
        'test',
        CacheResource,
        path=test_cache_path.name,
        compress=True
    )
    
    # 保存数据
    data = {"name": "test", "value": "x" * 1000}
    await resource.save(data)
    
    # 加载数据
    await resource.load()
    assert resource.data == data


@pytest.mark.asyncio
async def test_load_with_custom_serializer(cache_loader, tmp_path):
    """测试使用自定义序列化器加载缓存"""
    def serializer(data: Any) -> bytes:
        return str(data).encode('utf-8')
        
    def deserializer(data: bytes) -> str:
        return data.decode('utf-8')
        
    resource = await cache_loader.load(
        'test',
        CacheResource,
        path='test.custom',
        serializer=serializer,
        deserializer=deserializer
    )
    
    # 保存数据
    await resource.save("test data")
    
    # 加载数据
    await resource.load()
    assert resource.data == "test data"


@pytest.mark.asyncio
async def test_load_invalid_type(cache_loader, test_cache_path):
    """测试加载无效的资源类型"""
    class InvalidResource:
        pass
    
    with pytest.raises(ResourceLoadError):
        await cache_loader.load(
            'test',
            InvalidResource,
            path=str(test_cache_path)
        )


@pytest.mark.asyncio
async def test_unload_cache(cache_loader, test_cache_path):
    """测试释放缓存"""
    resource = await cache_loader.load(
        'test',
        CacheResource,
        path=test_cache_path.name
    )
    
    # 加载缓存
    await resource.load()
    assert resource.data is not None
    
    # 释放缓存
    await cache_loader.unload(resource)
    assert resource.data is None


def test_clear_all_caches(tmp_path):
    """测试清除所有缓存"""
    # 创建多个缓存文件
    for i in range(3):
        cache_path = tmp_path / f"test{i}.json"
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({"name": f"test{i}"}, f)
            
    # 创建加载器
    loader = CacheLoader(str(tmp_path))
    
    # 验证文件存在
    assert len(list(tmp_path.glob('*'))) == 3
    
    # 清除所有缓存
    loader.clear_all()
    
    # 验证文件已删除
    assert len(list(tmp_path.glob('*'))) == 0
