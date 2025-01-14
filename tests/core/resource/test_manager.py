"""资源管理器测试"""

import pytest
from typing import Dict, Any, Optional
from datetime import datetime
from game_automation.core.resource.base import ResourceBase, ResourceState, ResourceType
from game_automation.core.resource.loader import ResourceLoader
from game_automation.core.resource.cache import CacheManager
from game_automation.core.resource.monitor import MonitorManager
from game_automation.core.resource.manager import ResourceManager
from game_automation.core.resource.errors import (
    ResourceNotFoundError,
    ResourceLoadError,
    ResourceStateError
)


class MockResource(ResourceBase):
    """模拟资源类"""
    
    def __init__(self, key: str, data: Any = None):
        super().__init__(key, ResourceType.OTHER)
        self.data = data
        self.load_called = False
        self.unload_called = False
        
    async def _do_load(self) -> None:
        self.load_called = True
        
    async def _do_unload(self) -> None:
        self.unload_called = True


class MockLoader(ResourceLoader):
    """模拟资源加载器"""
    
    def __init__(self, resources: Optional[Dict[str, Any]] = None):
        self.resources = resources or {}
        
    async def load(
        self,
        key: str,
        resource_type: type[ResourceBase],
        **kwargs: Any
    ) -> ResourceBase:
        if key not in self.resources:
            raise ResourceNotFoundError(key)
        return MockResource(key, self.resources[key])
        
    async def unload(self, resource: ResourceBase) -> None:
        pass


@pytest.fixture
def loader():
    """创建模拟加载器"""
    return MockLoader({
        'test1': 'data1',
        'test2': 'data2'
    })


@pytest.fixture
def cache_manager():
    """创建缓存管理器"""
    return CacheManager()


@pytest.fixture
def monitor_manager():
    """创建监控管理器"""
    return MonitorManager()


@pytest.fixture
def manager(loader, cache_manager, monitor_manager):
    """创建资源管理器"""
    return ResourceManager(loader, cache_manager, monitor_manager)


@pytest.mark.asyncio
async def test_get_resource(manager):
    """测试获取资源"""
    # 获取不存在的资源
    with pytest.raises(ResourceLoadError):
        await manager.get_resource('not_exist', MockResource)
        
    # 获取存在的资源
    resource = await manager.get_resource('test1', MockResource)
    assert isinstance(resource, MockResource)
    assert resource.key == 'test1'
    assert resource.data == 'data1'
    assert resource.state == ResourceState.LOADED
    assert resource.ref_count == 1
    assert resource.load_called
    
    # 再次获取同一个资源
    resource2 = await manager.get_resource('test1', MockResource)
    assert resource2 is resource
    assert resource2.ref_count == 2


@pytest.mark.asyncio
async def test_release_resource(manager):
    """测试释放资源"""
    # 释放不存在的资源
    with pytest.raises(ResourceNotFoundError):
        await manager.release_resource('not_exist')
        
    # 获取并释放资源
    resource = await manager.get_resource('test1', MockResource)
    assert resource.ref_count == 1
    
    await manager.release_resource('test1')
    assert resource.ref_count == 0
    assert resource.state == ResourceState.UNLOADED
    assert resource.unload_called


@pytest.mark.asyncio
async def test_cleanup(manager):
    """测试清理资源"""
    # 获取多个资源
    resource1 = await manager.get_resource('test1', MockResource)
    resource2 = await manager.get_resource('test2', MockResource)
    
    # 释放资源1
    await manager.release_resource('test1')
    
    # 清理资源
    await manager.cleanup()
    
    # 检查资源状态
    assert resource1.state == ResourceState.UNLOADED
    assert resource2.state == ResourceState.LOADED


@pytest.mark.asyncio
async def test_get_stats(manager):
    """测试获取统计信息"""
    # 初始状态
    stats = await manager.get_stats()
    assert stats['total_resources'] == 0
    assert stats['loaded_resources'] == 0
    assert stats['error_resources'] == 0
    
    # 加载资源后
    await manager.get_resource('test1', MockResource)
    await manager.get_resource('test2', MockResource)
    
    stats = await manager.get_stats()
    assert stats['total_resources'] == 2
    assert stats['loaded_resources'] == 2
    assert stats['error_resources'] == 0
