"""测试日志系统"""

import os
import json
import asyncio
import logging
import pytest
from pathlib import Path

from game_automation.utils.logger import (
    GameLogger,
    get_logger,
    setup_logging,
    LoggingError
)

@pytest.fixture
def test_log_dir(tmp_path):
    """创建临时日志目录"""
    log_dir = tmp_path / "test_logs"
    log_dir.mkdir()
    return str(log_dir)

@pytest.fixture
def logger(test_log_dir):
    """创建测试日志记录器"""
    logger = GameLogger("test", log_dir=test_log_dir)
    return logger

@pytest.mark.asyncio
async def test_logger_initialization(test_log_dir):
    """测试日志记录器初始化"""
    logger = GameLogger("test", log_dir=test_log_dir)
    
    # 验证日志目录创建
    assert Path(test_log_dir).exists()
    assert Path(test_log_dir).is_dir()
    
    # 验证日志文件创建
    log_file = Path(test_log_dir) / "test.log"
    assert log_file.exists()
    assert log_file.is_file()
    
    # 验证日志级别设置
    assert logger.log_level == logging.INFO
    
    # 验证处理器设置
    assert len(logger.logger.handlers) == 3  # 控制台、文件和异步队列处理器

@pytest.mark.asyncio
async def test_logger_async_logging(test_log_dir):
    """测试异步日志记录"""
    logger = GameLogger("test", log_dir=test_log_dir)
    await logger.start()
    
    # 记录一些日志
    test_message = "Test async logging"
    logger.info(test_message)
    
    # 等待日志处理
    await asyncio.sleep(0.1)
    
    # 验证日志文件内容
    log_file = Path(test_log_dir) / "test.log"
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert test_message in content
        
    await logger.stop()

@pytest.mark.asyncio
async def test_logger_context(test_log_dir):
    """测试日志上下文"""
    logger = GameLogger("test", log_dir=test_log_dir)
    await logger.start()
    
    # 测试上下文设置
    context = {"user": "test_user", "session": "123"}
    logger.set_context(**context)
    
    test_message = "Test context logging"
    logger.info(test_message)
    
    # 等待日志处理
    await asyncio.sleep(0.1)
    
    # 验证日志文件内容包含上下文信息
    log_file = Path(test_log_dir) / "test.log"
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert test_message in content
        assert "test_user" in content
        assert "123" in content
        
    await logger.stop()

@pytest.mark.asyncio
async def test_logger_json_format(test_log_dir):
    """测试JSON格式日志"""
    logger = GameLogger("test", log_dir=test_log_dir, json_format=True)
    await logger.start()
    
    # 记录带上下文的日志
    context = {"user": "test_user", "session": "123"}
    test_message = "Test JSON logging"
    logger.info(test_message, context=context)
    
    # 等待日志处理
    await asyncio.sleep(0.1)
    
    # 验证日志文件内容是有效的JSON
    log_file = Path(test_log_dir) / "test.log"
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # 验证基本字段
                assert "time" in data
                assert "level" in data
                assert "name" in data
                assert "message" in data
                assert "context" in data
                
                # 验证字段值
                assert data["level"] == "INFO"
                assert data["name"] == "test"
                assert test_message in data["message"]
                assert data["context"] == context
                break
        else:
            pytest.fail("Log message not found")
            
    await logger.stop()

@pytest.mark.asyncio
async def test_logger_with_context_decorator(test_log_dir):
    """测试上下文装饰器"""
    logger = GameLogger("test", log_dir=test_log_dir)
    await logger.start()
    
    # 定义测试函数
    @logger.with_context(module="test_module")
    def test_function():
        logger.info("Test decorator")
        
    # 执行测试函数
    test_function()
    
    # 等待日志处理
    await asyncio.sleep(0.1)
    
    # 验证日志文件内容
    log_file = Path(test_log_dir) / "test.log"
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "Test decorator" in content
        assert "test_module" in content
        
    await logger.stop()

@pytest.mark.asyncio
async def test_get_logger(test_log_dir):
    """测试获取日志记录器"""
    # 获取同名日志记录器
    logger1 = get_logger("test", log_dir=test_log_dir)
    logger2 = get_logger("test")
    
    # 验证返回相同的实例
    assert logger1 is logger2
    
    # 验证日志记录
    test_message = "Test get_logger"
    await logger1.start()
    logger1.info(test_message)
    
    # 等待日志处理
    await asyncio.sleep(0.1)
    
    # 验证日志文件内容
    log_file = Path(test_log_dir) / "test.log"
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert test_message in content
        
    await logger1.stop()

@pytest.mark.asyncio
async def test_setup_logging(test_log_dir):
    """测试日志系统设置"""
    # 设置日志系统
    setup_logging(
        log_dir=test_log_dir,
        log_level="DEBUG",
        json_format=True
    )
    
    # 获取根日志记录器
    logger = get_logger("game_automation")
    
    # 验证日志记录
    test_message = "Test setup_logging"
    logger.info(test_message)
    
    # 等待日志处理
    await asyncio.sleep(0.1)
    
    # 验证日志文件内容
    log_file = Path(test_log_dir) / "game_automation.log"
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if test_message in data["message"]:
                    assert data["level"] == "INFO"
                    assert data["name"] == "game_automation"
                    break
        else:
            pytest.fail("Log message not found")
