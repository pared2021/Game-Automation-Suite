"""Event Dispatcher for Game Automation Suite.

This module provides an event dispatching system that handles:
- Event registration and unregistration
- Event prioritization
- Event filtering
- Async event handling
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Event:
    """Base event class."""
    type: str
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    priority: int = 0

@dataclass
class EventHandler:
    """Event handler registration."""
    callback: Callable
    priority: int = 0
    filters: Set[str] = field(default_factory=set)

class EventDispatcher:
    """Event dispatcher implementation.
    
    Features:
    - Async event handling
    - Event prioritization
    - Event filtering
    - Error handling
    """
    
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Initialize the event dispatcher.
        
        Args:
            loop: Optional event loop to use. If None, current loop will be used.
        """
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._event_queue = asyncio.Queue()
        self._running = False
        self._loop = loop or asyncio.get_event_loop()
        
    async def start(self):
        """Start the event dispatcher."""
        if self._running:
            logger.warning("Event dispatcher is already running")
            return
            
        logger.info("Starting event dispatcher")
        self._running = True
        
        # Start the event processing task
        try:
            self._process_task = asyncio.create_task(self._process_events())
            logger.info("Event processing task created")
        except Exception as e:
            logger.error(f"Failed to create event processing task: {e}")
            self._running = False
            raise
            
    async def stop(self):
        """Stop the event dispatcher."""
        if not self._running:
            logger.warning("Event dispatcher is not running")
            return
            
        logger.info("Stopping event dispatcher")
        self._running = False
        
        # Wait for the process task to complete
        if hasattr(self, '_process_task'):
            logger.debug("Waiting for event processing task to complete")
            try:
                await asyncio.wait_for(self._process_task, timeout=0.1)
                logger.debug("Event processing task completed")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for event processing to stop")
                self._process_task.cancel()
                try:
                    await self._process_task
                except asyncio.CancelledError:
                    logger.debug("Event processing task cancelled")
                    pass
            
        # Clear any remaining events
        events_cleared = 0
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
                self._event_queue.task_done()
                events_cleared += 1
            except asyncio.QueueEmpty:
                break
        if events_cleared > 0:
            logger.debug(f"Cleared {events_cleared} remaining events")
            
    def register_handler(self, event_type: str, callback: Callable,
                        priority: int = 0, filters: Optional[Set[str]] = None):
        """Register an event handler.
        
        Args:
            event_type: Type of event to handle
            callback: Async callback function
            priority: Handler priority (0-9, higher is more important)
            filters: Set of event filters
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
            
        handler = EventHandler(
            callback=callback,
            priority=priority,
            filters=filters or set()
        )
        
        # Insert handler in priority order (higher priority first)
        handlers = self._handlers[event_type]
        for i, h in enumerate(handlers):
            if handler.priority > h.priority:
                handlers.insert(i, handler)
                break
        else:
            handlers.append(handler)
            
        logger.debug(f"Registered handler for {event_type} with priority {priority}")
        
    def unregister_handler(self, event_type: str, callback: Callable):
        """Unregister an event handler.
        
        Args:
            event_type: Type of event
            callback: Callback function to remove
        """
        if event_type not in self._handlers:
            return
            
        self._handlers[event_type] = [
            h for h in self._handlers[event_type]
            if h.callback != callback
        ]
        
        logger.debug(f"Unregistered handler for {event_type}")
        
    async def dispatch(self, event: Event):
        """Dispatch an event.
        
        Args:
            event: Event to dispatch
        """
        if not self._running:
            raise RuntimeError("Event dispatcher is not running")
            
        await self._event_queue.put(event)
        
    async def _process_events(self):
        """Process events from the queue."""
        logger.info("Event processing loop started")
        while self._running:
            try:
                # Get next event with timeout
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                    logger.debug(f"Processing event: {event.type}")
                except asyncio.TimeoutError:
                    continue
                
                # Get handlers
                handlers = self._handlers.get(event.type, [])
                if not handlers:
                    logger.debug(f"No handlers found for event type: {event.type}")
                
                # Process handlers
                for handler in handlers:
                    if not self._should_handle(event, handler):
                        logger.debug(f"Handler filtered out for event: {event.type}")
                        continue
                        
                    try:
                        await handler.callback(event)
                        logger.debug(f"Handler completed for event: {event.type}")
                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")
                        
                self._event_queue.task_done()
                logger.debug(f"Event processed: {event.type}")
                
            except asyncio.CancelledError:
                logger.info("Event processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing events: {e}")
                await asyncio.sleep(0.1)
                
        logger.info("Event processing loop stopped")
        
    def _should_handle(self, event: Event, handler: EventHandler) -> bool:
        """Check if handler should process event.
        
        Args:
            event: Event to check
            handler: Handler to check
            
        Returns:
            bool: True if handler should process event
        """
        # If no filters are set, always handle
        if not handler.filters:
            return True
            
        # If filters are set, check if event source matches
        return event.source in handler.filters
