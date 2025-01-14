"""配置资源测试"""

import os
import json
import yaml
import pytest
from pathlib import Path
from typing import Dict, Any

from game_automation.core.resource.types.config import ConfigResource
from game_automation.core.resource.errors import ResourceLoadError


@pytest.fixture
def test_json_config(tmp_path) -> Path:
    """创建测试 JSON 配置文件"""
    config_path = tmp_path / "test.json"
    config = {
        "name": "test",
        "settings": {
            "value1": 123,
            "value2": "test"
        },
        "env_value": "${TEST_VALUE}"
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f)
    return config_path


@pytest.fixture
def test_yaml_config(tmp_path) -> Path:
    """创建测试 YAML 配置文件"""
    config_path = tmp_path / "test.yaml"
    config = {
        "name": "test",
        "settings": {
            "value1": 123,
            "value2": "test"
        },
        "env_value": "${TEST_VALUE}"
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
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
                },
                "required": ["value1", "value2"]
            },
            "env_value": {"type": "string"}
        },
        "required": ["name", "settings"]
    }


@pytest.mark.asyncio
async def test_load_json_config(test_json_config, test_schema):
    """测试加载 JSON 配置"""
    resource = ConfigResource('test', str(test_json_config), test_schema)
    
    # 测试加载前的状态
    assert resource.config is None
    
    # 加载配置
    await resource.load()
    
    # 测试加载后的状态
    config = resource.config
    assert config is not None
    assert config['name'] == 'test'
    assert config['settings']['value1'] == 123
    assert config['settings']['value2'] == 'test'


@pytest.mark.asyncio
async def test_load_yaml_config(test_yaml_config, test_schema):
    """测试加载 YAML 配置"""
    resource = ConfigResource('test', str(test_yaml_config), test_schema)
    await resource.load()
    
    config = resource.config
    assert config is not None
    assert config['name'] == 'test'
    assert config['settings']['value1'] == 123
    assert config['settings']['value2'] == 'test'


@pytest.mark.asyncio
async def test_env_vars(test_json_config):
    """测试环境变量替换"""
    # 设置环境变量
    os.environ['APP_TEST_VALUE'] = 'env_test'
    
    resource = ConfigResource('test', str(test_json_config))
    await resource.load()
    
    config = resource.config
    assert config is not None
    assert config['env_value'] == 'env_test'


@pytest.mark.asyncio
async def test_config_validation(test_json_config, test_schema):
    """测试配置验证"""
    # 修改配置文件为无效配置
    invalid_config = {
        "name": "test",
        "settings": {
            "value1": "invalid",  # 应该是整数
            "value2": "test"
        }
    }
    with open(test_json_config, 'w', encoding='utf-8') as f:
        json.dump(invalid_config, f)
        
    resource = ConfigResource('test', str(test_json_config), test_schema)
    with pytest.raises(ResourceLoadError):
        await resource.load()


@pytest.mark.asyncio
async def test_config_merge(test_json_config):
    """测试配置合并"""
    resource = ConfigResource('test', str(test_json_config))
    await resource.load()
    
    # 合并新配置
    other_config = {
        "settings": {
            "value3": "new_value"
        },
        "new_section": {
            "key": "value"
        }
    }
    resource.merge(other_config)
    
    config = resource.config
    assert config is not None
    assert config['settings']['value1'] == 123  # 原值保持不变
    assert config['settings']['value3'] == "new_value"  # 新值已添加
    assert config['new_section']['key'] == "value"  # 新节点已添加


@pytest.mark.asyncio
async def test_config_get(test_json_config):
    """测试配置获取"""
    resource = ConfigResource('test', str(test_json_config))
    await resource.load()
    
    # 测试不同层级的键
    assert resource.get('name') == 'test'
    assert resource.get('settings.value1') == 123
    assert resource.get('settings.value2') == 'test'
    
    # 测试默认值
    assert resource.get('not_exist', 'default') == 'default'
    assert resource.get('settings.not_exist', 100) == 100


@pytest.mark.asyncio
async def test_load_nonexistent_config():
    """测试加载不存在的配置"""
    resource = ConfigResource('test', 'nonexistent.json')
    with pytest.raises(ResourceLoadError):
        await resource.load()


@pytest.mark.asyncio
async def test_unload_config(test_json_config):
    """测试释放配置"""
    resource = ConfigResource('test', str(test_json_config))
    await resource.load()
    
    # 测试释放前的状态
    assert resource.config is not None
    
    # 释放配置
    await resource.unload()
    
    # 测试释放后的状态
    assert resource.config is None
