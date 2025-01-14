import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from game_automation.core.error.error_manager import (
    ErrorManager,
    ErrorCategory,
    ErrorSeverity,
    ErrorState,
    GameAutomationError,
)

@pytest_asyncio.fixture
async def error_manager():
    manager = ErrorManager()
    await manager.initialize()
    yield manager
    # 确保在测试结束后清理资源
    await manager.clear_errors()
    if hasattr(manager, '_event_loop'):
        manager._event_loop = None

@pytest.mark.asyncio
async def test_error_manager_initialization(error_manager):
    """测试错误管理器初始化"""
    assert error_manager._initialized
    assert error_manager._handlers
    assert error_manager._errors == []

@pytest.mark.asyncio
async def test_error_registration_and_handling():
    """测试错误注册和处理"""
    manager = ErrorManager()
    await manager.initialize()
    
    handled_errors = []
    
    async def test_handler(error: GameAutomationError):
        handled_errors.append(error)
    
    # 注册处理器
    await manager.register_handler(
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.ERROR,
        handler=test_handler
    )
    
    # 创建并处理错误
    error = GameAutomationError(
        message="Test error",
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.ERROR
    )
    
    await manager.handle_error(error)
    
    # 验证错误被正确处理
    assert len(handled_errors) == 1
    assert handled_errors[0] == error
    assert error in manager._errors

@pytest.mark.asyncio
async def test_error_filtering():
    """测试错误过滤"""
    manager = ErrorManager()
    await manager.initialize()
    
    # 创建不同类型的错误
    system_error = GameAutomationError(
        message="System error",
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.ERROR
    )
    
    network_error = GameAutomationError(
        message="Network error",
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.WARNING
    )
    
    # 处理错误
    await manager.handle_error(system_error)
    await manager.handle_error(network_error)
    
    # 测试过滤
    system_errors = await manager.get_errors(category=ErrorCategory.SYSTEM)
    assert len(system_errors) == 1
    assert system_errors[0] == system_error
    
    warning_errors = await manager.get_errors(severity=ErrorSeverity.WARNING)
    assert len(warning_errors) == 1
    assert warning_errors[0] == network_error

@pytest.mark.asyncio
async def test_error_recovery():
    """测试错误恢复机制"""
    manager = ErrorManager()
    await manager.initialize()
    
    recovery_count = 0
    
    async def recovery_handler(error: GameAutomationError):
        nonlocal recovery_count
        recovery_count += 1
        return True  # 表示恢复成功
    
    # 注册恢复处理器
    await manager.register_handler(
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.ERROR,
        handler=recovery_handler
    )
    
    # 创建可恢复的错误
    error = GameAutomationError(
        message="Recoverable error",
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.ERROR,
        context={"recoverable": True}
    )
    
    # 处理错误
    await manager.handle_error(error)
    
    # 验证恢复处理器被调用
    assert recovery_count == 1
    assert error.state == ErrorState.RESOLVED

@pytest.mark.asyncio
async def test_error_context():
    """测试错误上下文"""
    manager = ErrorManager()
    await manager.initialize()
    
    # 创建带有上下文的错误
    original_error = ValueError("Original error")
    context = {
        "function": "test_function",
        "params": {"x": 1, "y": 2},
        "stack_trace": True
    }
    
    error = GameAutomationError(
        message="Context test error",
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.ERROR,
        original_error=original_error,
        context=context
    )
    
    # 处理错误
    await manager.handle_error(error)
    
    # 获取处理后的错误
    errors = await manager.get_errors()
    processed_error = errors[0]
    
    # 验证上下文信息
    assert processed_error.context == context
    assert processed_error.original_error == original_error
    assert "stack_trace" in processed_error.context
