"""Tests for async utilities."""

import pytest
import asyncio
import time
from game_automation.core.async_io.async_utils import (
    with_retry, with_timeout,
    AsyncResourceManager, AsyncPerformanceMonitor
)

@pytest.mark.asyncio
async def test_retry_decorator():
    """Test retry decorator functionality."""
    attempt_count = 0
    
    @with_retry(retries=3, delay=0.1)
    async def failing_function():
        nonlocal attempt_count
        attempt_count += 1
        raise ValueError("Test error")
        
    # Function should retry 3 times then fail
    with pytest.raises(ValueError):
        await failing_function()
        
    assert attempt_count == 4  # Initial attempt + 3 retries
    
@pytest.mark.asyncio
async def test_retry_success():
    """Test retry decorator with eventual success."""
    attempt_count = 0
    
    @with_retry(retries=3, delay=0.1)
    async def eventually_succeeds():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError("Test error")
        return "success"
        
    result = await eventually_succeeds()
    
    assert result == "success"
    assert attempt_count == 3
    
@pytest.mark.asyncio
async def test_timeout_decorator():
    """Test timeout decorator."""
    @with_timeout(0.1)
    async def slow_function():
        await asyncio.sleep(0.2)
        
    with pytest.raises(asyncio.TimeoutError):
        await slow_function()
        
@pytest.mark.asyncio
async def test_timeout_success():
    """Test timeout decorator with successful completion."""
    @with_timeout(0.2)
    async def fast_function():
        await asyncio.sleep(0.1)
        return "success"
        
    result = await fast_function()
    assert result == "success"
    
@pytest.mark.asyncio
async def test_resource_manager():
    """Test async resource manager."""
    manager = AsyncResourceManager()
    
    # Store and retrieve resource
    await manager.put("test_key", "test_value")
    value = await manager.get("test_key")
    assert value == "test_value"
    
    # Test cleanup
    await manager.put("old_key", "old_value")
    manager._usage["old_key"] = time.time() - 3700  # Make resource old
    
    await manager.cleanup(max_age=3600)
    
    with pytest.raises(KeyError):
        await manager.get("old_key")
        
@pytest.mark.asyncio
async def test_performance_monitor():
    """Test async performance monitor."""
    monitor = AsyncPerformanceMonitor()
    
    # Test operation timing
    monitor.start_operation("test_op")
    await asyncio.sleep(0.1)
    duration = monitor.end_operation("test_op")
    
    assert duration >= 0.1
    
    # Test statistics
    stats = monitor.get_stats()
    assert stats["counts"]["test_op"] == 1
    assert len(stats["active_operations"]) == 0
    
@pytest.mark.asyncio
async def test_retry_with_backoff():
    """Test retry decorator with backoff."""
    attempt_times = []
    
    @with_retry(retries=2, delay=0.1, backoff=2.0)
    async def failing_function():
        attempt_times.append(time.time())
        raise ValueError("Test error")
        
    start_time = time.time()
    
    with pytest.raises(ValueError):
        await failing_function()
        
    # Check that delays between attempts increase
    delays = [t - start_time for t in attempt_times]
    assert delays[1] - delays[0] < delays[2] - delays[1]
    
@pytest.mark.asyncio
async def test_resource_manager_concurrent():
    """Test async resource manager with concurrent access."""
    manager = AsyncResourceManager()
    
    async def access_resource(key: str, value: str):
        await manager.put(key, value)
        result = await manager.get(key)
        return result
        
    # Create multiple concurrent operations
    tasks = [
        access_resource(f"key_{i}", f"value_{i}")
        for i in range(5)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Check results
    for i, result in enumerate(results):
        assert result == f"value_{i}"
        
@pytest.mark.asyncio
async def test_performance_monitor_concurrent():
    """Test performance monitor with concurrent operations."""
    monitor = AsyncPerformanceMonitor()
    
    async def monitored_operation(name: str):
        monitor.start_operation(name)
        await asyncio.sleep(0.1)
        return monitor.end_operation(name)
        
    # Run multiple operations concurrently
    tasks = [
        monitored_operation(f"op_{i}")
        for i in range(3)
    ]
    
    durations = await asyncio.gather(*tasks)
    
    # Check results
    stats = monitor.get_stats()
    assert sum(stats["counts"].values()) == 3
    assert all(d >= 0.1 for d in durations)
