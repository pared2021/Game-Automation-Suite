"""Test configuration manager"""

import os
import json
import pytest
import pytest_asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import asyncio

from game_automation.core.config_manager import ConfigManager, ConfigError, ConfigEvent

@pytest_asyncio.fixture
async def config_manager(tmp_path):
    """Create a config manager instance for testing"""
    config_dir = tmp_path / "config"
    manager = ConfigManager(str(config_dir))
    await manager.initialize()
    yield manager
    
    # 停止事件调度器
    await manager._event_dispatcher.stop()
    
    # 清理配置文件
    if config_dir.exists():
        for file in config_dir.glob("**/*"):
            if file.is_file():
                file.unlink()
        for dir in reversed(list(config_dir.glob("**/*"))):
            if dir.is_dir():
                dir.rmdir()
        config_dir.rmdir()

@pytest.mark.asyncio
async def test_config_initialization(tmp_path):
    """Test configuration manager initialization"""
    config_dir = tmp_path / "config"
    manager = ConfigManager(str(config_dir))
    
    # 验证初始状态
    assert not manager._initialized
    assert not config_dir.exists()
    
    # 初始化配置管理器
    await manager.initialize()
    
    # 验证初始化结果
    assert manager._initialized
    assert config_dir.exists()
    assert isinstance(manager.get_config(), dict)
    assert all(section in manager.get_config() for section in ["game", "recognition", "task", "debug"])

@pytest.mark.asyncio
async def test_config_reset(config_manager):
    """Test configuration reset"""
    # 修改配置
    test_config = {"test": "value"}
    config_manager.set_config(test_config)
    
    # 重置配置
    config_manager.reset_config()
    
    # 验证配置已重置为默认值
    config = config_manager.get_config()
    assert config == config_manager.default_config
    assert "test" not in config

@pytest.mark.asyncio
async def test_config_get_set(config_manager):
    """Test getting and setting configuration"""
    # 测试获取整个配置
    full_config = config_manager.get_config()
    assert isinstance(full_config, dict)
    
    # 测试获取特定部分
    game_config = config_manager.get_config("game")
    assert isinstance(game_config, dict)
    assert "window_title" in game_config
    
    # 测试设置特定部分
    new_game_config = {
        "window_title": "Test Game",
        "process_name": "test.exe"
    }
    config_manager.set_config(new_game_config, "game")
    updated_game_config = config_manager.get_config("game")
    assert updated_game_config["window_title"] == "Test Game"
    assert updated_game_config["process_name"] == "test.exe"

@pytest.mark.asyncio
async def test_config_update(config_manager):
    """Test configuration update"""
    # 测试深度更新
    updates = {
        "game": {
            "window_title": "Updated Game",
            "resolution": {
                "width": 1920
            }
        }
    }
    config_manager.update_config(updates)
    
    # 验证更新结果
    game_config = config_manager.get_config("game")
    assert game_config["window_title"] == "Updated Game"
    assert game_config["resolution"]["width"] == 1920
    assert game_config["resolution"]["height"] == 720  # 未更新的值保持不变

@pytest.mark.asyncio
async def test_config_save_load(config_manager):
    """Test saving and loading configuration"""
    # 准备测试配置
    test_config = {
        "game": {
            "window_title": "Test Game",
            "process_name": "test.exe"
        }
    }
    config_manager.set_config(test_config)
    
    # 保存配置
    await config_manager.save_config("test_config.json")
    
    # 重置配置
    config_manager.reset_config()
    
    # 加载配置
    await config_manager.load_config("test_config.json")
    
    # 验证加载的配置
    loaded_config = config_manager.get_config()
    assert loaded_config["game"]["window_title"] == "Test Game"
    assert loaded_config["game"]["process_name"] == "test.exe"

@pytest.mark.asyncio
async def test_config_backup(config_manager):
    """Test configuration backup"""
    # 保存初始配置
    await config_manager.save_config()
    
    # 修改并再次保存配置
    test_config = {"test": "value"}
    config_manager.set_config(test_config)
    await config_manager.save_config()
    
    # 验证备份文件创建
    backup_dir = Path(config_manager.config_dir) / "backup"
    assert backup_dir.exists()
    assert len(list(backup_dir.glob("*.json"))) > 0

@pytest.mark.asyncio
async def test_config_error_handling(config_manager):
    """Test configuration error handling"""
    # 测试加载不存在的配置文件
    with pytest.raises(ConfigError):
        await config_manager.load_config("nonexistent.json")
    
    # 测试无效的配置数据
    invalid_config = {"invalid": {"key": None}}
    with pytest.raises(ConfigError):
        config_manager.validate_config(invalid_config)

@pytest.mark.asyncio
async def test_config_merge(config_manager):
    """Test configuration merge"""
    # 准备测试数据
    default_config = {
        "section1": {
            "key1": "value1",
            "key2": {
                "subkey1": "subvalue1"
            }
        }
    }
    user_config = {
        "section1": {
            "key2": {
                "subkey2": "subvalue2"
            }
        }
    }
    
    # 测试合并
    merged = config_manager._merge_config(default_config, user_config)
    
    # 验证合并结果
    assert merged["section1"]["key1"] == "value1"  # 保留默认值
    assert merged["section1"]["key2"]["subkey1"] == "subvalue1"  # 保留默认子值
    assert merged["section1"]["key2"]["subkey2"] == "subvalue2"  # 添加新子值

@pytest.mark.asyncio
async def test_config_events(config_manager):
    """Test configuration events"""
    events = []
    
    async def on_config_changed(event: ConfigEvent):
        events.append(event)
    
    # 注册事件监听器
    config_manager._event_dispatcher.register_handler("config_changed", on_config_changed)
    
    # 更新配置
    old_value = config_manager.get_config("game")["window_title"]
    new_value = "Test Game"
    config_manager.update_config({"game": {"window_title": new_value}})
    
    # 等待事件处理器处理事件
    await asyncio.sleep(0.1)
    
    # 验证事件
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ConfigEvent)
    assert event.section == "game"
    assert event.key == "window_title"
    assert event.old_value == old_value
    assert event.new_value == new_value
