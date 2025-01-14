"""Tests for event dispatcher functionality."""

import pytest
import asyncio
import pytest_asyncio
from datetime import datetime
from game_automation.core.async_io.event_dispatcher import (
    EventDispatcher, Event, EventHandler
)

@pytest_asyncio.fixture
async def test_dispatcher():
    """Create an event dispatcher instance for testing."""
    disp = EventDispatcher(loop=asyncio.get_running_loop())
    yield disp
    if disp._running:
        await disp.stop()

@pytest.mark.asyncio
async def test_event_dispatcher_start_stop(test_dispatcher):
    """Test starting and stopping the event dispatcher."""
    # Start the dispatcher
    start_task = asyncio.create_task(test_dispatcher.start())
    await asyncio.sleep(0.01)
    
    assert test_dispatcher._running
    
    # Stop the dispatcher
    await test_dispatcher.stop()
    try:
        await asyncio.wait_for(start_task, timeout=0.1)
    except asyncio.TimeoutError:
        start_task.cancel()
        await asyncio.sleep(0)
    
    assert not test_dispatcher._running
    
@pytest.mark.asyncio
async def test_event_handler_registration(test_dispatcher):
    """Test registering and unregistering event handlers."""
    async def handler(event: Event):
        pass
        
    # Register handler
    test_dispatcher.register_handler("test_event", handler)
    
    assert "test_event" in test_dispatcher._handlers
    assert len(test_dispatcher._handlers["test_event"]) == 1
    assert test_dispatcher._handlers["test_event"][0].callback == handler
    
    # Unregister handler
    test_dispatcher.unregister_handler("test_event", handler)
    
    assert len(test_dispatcher._handlers["test_event"]) == 0
    
@pytest.mark.asyncio
async def test_event_dispatch(test_dispatcher):
    """Test event dispatching."""
    received_events = []
    
    async def handler(event: Event):
        received_events.append(event)
        
    # Register handler and start dispatcher
    test_dispatcher.register_handler("test_event", handler)
    await test_dispatcher.start()
    
    # Dispatch event
    event = Event(type="test_event", data="test_data")
    await test_dispatcher.dispatch(event)
    await asyncio.sleep(0.1)  # Wait longer for event to be processed
    
    assert len(received_events) == 1
    assert received_events[0].type == "test_event"
    assert received_events[0].data == "test_data"
    
    # Stop the dispatcher
    await test_dispatcher.stop()
    
@pytest.mark.asyncio
async def test_event_priority(test_dispatcher):
    """Test event handler priority."""
    received_events = []
    
    async def handler1(event: Event):
        received_events.append(1)
        await asyncio.sleep(0.01)
        
    async def handler2(event: Event):
        received_events.append(2)
        await asyncio.sleep(0.01)
        
    # Register handlers with different priorities
    test_dispatcher.register_handler("test_event", handler1, priority=0)
    test_dispatcher.register_handler("test_event", handler2, priority=1)
    
    # Start dispatcher
    start_task = asyncio.create_task(test_dispatcher.start())
    await asyncio.sleep(0.01)
    
    # Dispatch event
    event = Event(type="test_event")
    await test_dispatcher.dispatch(event)
    await asyncio.sleep(0.03)
    
    # Higher priority handler should execute first
    assert received_events == [2, 1]
    
    # Stop the dispatcher
    await test_dispatcher.stop()
    try:
        await asyncio.wait_for(start_task, timeout=0.1)
    except asyncio.TimeoutError:
        start_task.cancel()
        await asyncio.sleep(0)
    
@pytest.mark.asyncio
async def test_event_filtering(test_dispatcher):
    """Test event filtering."""
    received_events = []
    
    async def handler(event: Event):
        received_events.append(event)
        
    # Register handler with filter
    test_dispatcher.register_handler(
        "test_event",
        handler,
        filters={"source1"}
    )
    
    # Start dispatcher
    await test_dispatcher.start()
    
    # Dispatch events from different sources
    event1 = Event(type="test_event", source="source1")
    event2 = Event(type="test_event", source="source2")
    
    await test_dispatcher.dispatch(event1)
    await test_dispatcher.dispatch(event2)
    await asyncio.sleep(0.1)  # Wait longer for events to be processed
    
    # Only event from source1 should be received
    assert len(received_events) == 1
    assert received_events[0].source == "source1"
    
    # Stop the dispatcher
    await test_dispatcher.stop()
    
@pytest.mark.asyncio
async def test_event_error_handling(test_dispatcher):
    """Test error handling in event handlers."""
    error_count = 0
    
    async def error_handler(event: Event):
        raise ValueError("Test error")
        
    async def success_handler(event: Event):
        nonlocal error_count
        error_count += 1
        
    # Register handlers
    test_dispatcher.register_handler("test_event", error_handler)
    test_dispatcher.register_handler("test_event", success_handler)
    
    # Start dispatcher
    await test_dispatcher.start()
    
    # Dispatch event
    event = Event(type="test_event")
    await test_dispatcher.dispatch(event)
    await asyncio.sleep(0.1)  # Wait longer for event to be processed
    
    # Second handler should still execute despite first handler's error
    assert error_count == 1
    
    # Stop the dispatcher
    await test_dispatcher.stop()
