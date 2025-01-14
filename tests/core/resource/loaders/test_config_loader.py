"""配置加载器测试"""

import os
import json
import pytest
from pathlib import Path
from typing import Dict, Any

from game_automation.core.resource.types.config import ConfigResource
from game_automation.core.resource.loaders.config_loader import ConfigLoader
from game_automation.core.resource.errors import ResourceLoadError


@pytest.fixture
def test_config_path(tmp_path) -> Path:
    """创建测试配置文件"""
    config_path = tmp_path / "test.json"
    config = {
        "name": "test",
        "settings": {
            "value1": 123,
            "value2": "test"
        }
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f)
    return config_path


@pytest.fixture
def test_schema() -> Dict[str, Any]:
    """创建测试 Schema"""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "settings": {
                "type": "object",
                "properties": {
                    "value1": {"type": "integer"},
                    "value2": {"type": "string"}
                }
            }
        }
    }


@pytest.fixture
def config_loader(tmp_path, test_schema):
    """创建配置加载器"""
    return ConfigLoader(
        str(tmp_path),
        schemas={'test': test_schema}
    )


@pytest.mark.asyncio
async def test_load_config(config_loader, test_config_path):
    """测试加载配置"""
    # 使用相对路径
    resource = await config_loader.load(
        'test',
        ConfigResource,
        path=test_config_path.name
    )
    
    assert isinstance(resource, ConfigResource)
    assert resource.key == 'test'
    assert resource.path == test_config_path
    
    # 加载配置
    await resource.load()
    assert resource.config is not None
    assert resource.config['name'] == 'test'


@pytest.mark.asyncio
async def test_load_with_absolute_path(config_loader, test_config_path):
    """测试使用绝对路径加载配置"""
    resource = await config_loader.load(
        'test',
        ConfigResource,
        path=str(test_config_path)
    )
    
    assert isinstance(resource, ConfigResource)
    assert resource.path == test_config_path


@pytest.mark.asyncio
async def test_load_with_schema(config_loader, test_config_path):
    """测试使用 Schema 加载配置"""
    # Schema 从加载器获取
    resource = await config_loader.load(
        'test',
        ConfigResource,
        path=test_config_path.name
    )
    await resource.load()  # 应该通过验证
    
    # 使用自定义 Schema
    custom_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"]
    }
    resource = await config_loader.load(
        'test',
        ConfigResource,
        path=test_config_path.name,
        schema=custom_schema
    )
    await resource.load()  # 应该通过验证


@pytest.mark.asyncio
async def test_load_with_env_prefix(config_loader, test_config_path):
    """测试使用环境变量前缀加载配置"""
    resource = await config_loader.load(
        'test',
        ConfigResource,
        path=test_config_path.name,
        env_prefix='TEST_'
    )
    
    assert isinstance(resource, ConfigResource)
    # env_prefix 会被传递给 ConfigResource


@pytest.mark.asyncio
async def test_load_invalid_type(config_loader, test_config_path):
    """测试加载无效的资源类型"""
    class InvalidResource:
        pass
    
    with pytest.raises(ResourceLoadError):
        await config_loader.load(
            'test',
            InvalidResource,
            path=str(test_config_path)
        )


@pytest.mark.asyncio
async def test_unload_config(config_loader, test_config_path):
    """测试释放配置"""
    resource = await config_loader.load(
        'test',
        ConfigResource,
        path=test_config_path.name
    )
    
    # 加载配置
    await resource.load()
    assert resource.config is not None
    
    # 释放配置
    await config_loader.unload(resource)
    assert resource.config is None
