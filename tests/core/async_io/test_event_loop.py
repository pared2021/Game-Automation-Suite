"""Tests for event loop functionality."""

import pytest
import asyncio
import pytest_asyncio
from datetime import datetime
from game_automation.core.async_io.event_loop import EventLoop, EventLoopStats

@pytest_asyncio.fixture
async def test_loop():
    """Create an event loop instance for testing."""
    loop = EventLoop(loop=asyncio.get_running_loop())
    yield loop
    if loop._running:
        await loop.stop()

@pytest.mark.asyncio
async def test_event_loop_start_stop(test_loop):
    """Test starting and stopping the event loop."""
    # Start the loop
    start_task = asyncio.create_task(test_loop.start())
    await asyncio.sleep(0.01)  # Give time for loop to start
    
    assert test_loop._running
    assert isinstance(test_loop.stats, EventLoopStats)
    assert test_loop.stats.total_tasks == 0
    
    # Stop the loop
    await test_loop.stop()
    try:
        await asyncio.wait_for(start_task, timeout=0.1)
    except asyncio.TimeoutError:
        start_task.cancel()
        await asyncio.sleep(0)
    
    assert not test_loop._running
    
@pytest.mark.asyncio
async def test_event_loop_task_creation(test_loop):
    """Test task creation and tracking."""
    async def dummy_task():
        await asyncio.sleep(0.01)
        
    # Start the loop
    start_task = asyncio.create_task(test_loop.start())
    await asyncio.sleep(0.01)
    
    # Create a task
    task = await test_loop.create_task(dummy_task(), name="dummy")
    
    assert test_loop.stats.total_tasks == 1
    assert "dummy" in test_loop.stats.current_tasks
    
    # Wait for task completion
    await task
    await asyncio.sleep(0.01)
    
    assert test_loop.stats.completed_tasks == 1
    assert "dummy" not in test_loop.stats.current_tasks
    
    # Stop the loop
    await test_loop.stop()
    try:
        await asyncio.wait_for(start_task, timeout=0.1)
    except asyncio.TimeoutError:
        start_task.cancel()
        await asyncio.sleep(0)
    
@pytest.mark.asyncio
async def test_event_loop_task_priority(test_loop):
    """Test task priority handling."""
    results = []
    
    async def priority_task(priority: int):
        results.append(priority)
        await asyncio.sleep(0.01)
        
    # Start the loop
    start_task = asyncio.create_task(test_loop.start())
    await asyncio.sleep(0.01)
    
    # Create tasks with different priorities
    tasks = []
    for i in range(3):
        task = await test_loop.create_task(
            priority_task(i),
            name=f"task_{i}",
            priority=i
        )
        tasks.append(task)
        
    # Wait for tasks to complete
    await asyncio.gather(*tasks)
    await asyncio.sleep(0.01)
    
    # Higher priority tasks should complete first
    assert results == [2, 1, 0]
    
    # Stop the loop
    await test_loop.stop()
    try:
        await asyncio.wait_for(start_task, timeout=0.1)
    except asyncio.TimeoutError:
        start_task.cancel()
        await asyncio.sleep(0)
    
@pytest.mark.asyncio
async def test_event_loop_error_handling(test_loop):
    """Test error handling in tasks."""
    async def error_task():
        raise ValueError("Test error")
        
    # Start the loop
    start_task = asyncio.create_task(test_loop.start())
    await asyncio.sleep(0.01)
    
    # Create a task that will fail
    task = await test_loop.create_task(error_task(), name="error_task")
    
    # Wait for task to fail
    with pytest.raises(ValueError, match="Test error"):
        await task
        
    await asyncio.sleep(0.01)
    
    assert test_loop.stats.failed_tasks == 1
    assert "error_task" not in test_loop.stats.current_tasks
    
    # Stop the loop
    await test_loop.stop()
    try:
        await asyncio.wait_for(start_task, timeout=0.1)
    except asyncio.TimeoutError:
        start_task.cancel()
        await asyncio.sleep(0)
    
@pytest.mark.asyncio
async def test_event_loop_stats(test_loop):
    """Test event loop statistics tracking."""
    async def dummy_task():
        await asyncio.sleep(0.01)
        
    # Start the loop
    start_task = asyncio.create_task(test_loop.start())
    await asyncio.sleep(0.01)
    
    # Initial stats
    assert isinstance(test_loop.stats.start_time, datetime)
    assert test_loop.stats.total_tasks == 0
    assert test_loop.stats.completed_tasks == 0
    assert test_loop.stats.failed_tasks == 0
    
    # Create and run tasks
    tasks = []
    for i in range(3):
        task = await test_loop.create_task(dummy_task(), name=f"task_{i}")
        tasks.append(task)
        
    await asyncio.gather(*tasks)
    await asyncio.sleep(0.01)
    
    # Check final stats
    assert test_loop.stats.total_tasks == 3
    assert test_loop.stats.completed_tasks == 3
    assert test_loop.stats.failed_tasks == 0
    assert len(test_loop.stats.current_tasks) == 0
    
    # Stop the loop
    await test_loop.stop()
    try:
        await asyncio.wait_for(start_task, timeout=0.1)
    except asyncio.TimeoutError:
        start_task.cancel()
        await asyncio.sleep(0)
