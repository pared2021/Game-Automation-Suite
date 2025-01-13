"""
Centralized event management system
"""

from typing import Any, Dict, List, Callable, Optional, Set
import asyncio
from datetime import datetime
import json
from enum import Enum, auto
from ..base.service_base import ServiceBase

class EventType(str, Enum):
    """System event types"""
    # Task events
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    TASK_SCHEDULED = "task_scheduled"
    TASK_RETRYING = "task_retrying"
    
    # Game events
    GAME_STARTED = "game_started"
    GAME_STOPPED = "game_stopped"
    GAME_STATE_CHANGED = "game_state_changed"
    GAME_ERROR = "game_error"
    
    # System events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    SYSTEM_ERROR = "system_error"
    INITIALIZED = "initialized"
    SHUTDOWN = "shutdown"
    ERROR = "error"
    
    # GUI events
    GUI_ACTION = "gui_action"
    GUI_STATE_CHANGED = "gui_state_changed"
    GUI_ERROR = "gui_error"
    
    # State events
    STATE_CHANGED = "state_changed"
    STATE_TIMEOUT = "state_timeout"
    STATE_ACTION = "state_action"
    
    # Window events
    WINDOW_CHANGED = "window_changed"
    WINDOW_CLOSED = "window_closed"
    
    # Image events
    IMAGE_CAPTURED = "image_captured"
    IMAGE_PROCESSED = "image_processed"

class Event:
    """Represents a system event"""
    
    def __init__(self, event_type: EventType, data: Any = None, source: str = None):
        """Initialize event
        
        Args:
            event_type: Event type
            data: Event data
            source: Event source
        """
        self.type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.now()
        self.id = f"{event_type}_{self.timestamp.timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary
        
        Returns:
            Dict[str, Any]: Event data dictionary
        """
        return {
            'id': self.id,
            'type': self.type,
            'data': self.data,
            'source': self.source,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary
        
        Args:
            data: Event data dictionary
            
        Returns:
            Event: Created event instance
        """
        event = cls(
            EventType[data['type']], 
            data['data'], 
            data['source']
        )
        event.id = data['id']
        event.timestamp = datetime.fromisoformat(data['timestamp'])
        return event

class EventManager(ServiceBase):
    """Centralized event management service"""
    
    def __init__(self):
        """Initialize event manager"""
        super().__init__("EventManager")
        self._handlers: Dict[EventType, Set[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._is_processing = False
        self._processing_task: Optional[asyncio.Task] = None
        
    async def _on_start(self):
        """Start event processing"""
        if not self._is_processing:
            self._is_processing = True
            self._processing_task = asyncio.create_task(self._process_events())
            
    async def _on_stop(self):
        """Stop event processing"""
        if self._is_processing:
            self._is_processing = False
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass
                self._processing_task = None
                
    async def _process_events(self):
        """Process events from queue"""
        while self._is_processing:
            try:
                event = await self._event_queue.get()
                await self._handle_event(event)
                self._event_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Error processing event")
                
    async def _handle_event(self, event: Event):
        """Handle a single event
        
        Args:
            event: Event to handle
        """
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
            
        # Call handlers
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.exception(f"Error in event handler for {event.type}")
                    
    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe to events
        
        Args:
            event_type: Event type to subscribe to
            handler: Handler function
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = set()
        self._handlers[event_type].add(handler)
        
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from events
        
        Args:
            event_type: Event type to unsubscribe from
            handler: Handler function
        """
        if event_type in self._handlers:
            self._handlers[event_type].discard(handler)
            
    async def emit(self, event: Event):
        """Emit an event
        
        Args:
            event: Event to emit
        """
        await self._event_queue.put(event)
        
    def emit_immediate(self, event: Event):
        """Emit and handle an event immediately
        
        Args:
            event: Event to emit
        """
        asyncio.create_task(self._handle_event(event))
        
    def get_history(self,
                   event_type: Optional[EventType] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Event]:
        """Get event history
        
        Args:
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List[Event]: Filtered event history
        """
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
            
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
            
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
            
        return events
        
    def clear_history(self):
        """Clear event history"""
        self._event_history.clear()
        
    def save_history(self, file_path: str):
        """Save event history
        
        Args:
            file_path: Path to save file
        """
        history_data = [event.to_dict() for event in self._event_history]
        with open(file_path, 'w') as f:
            json.dump(history_data, f, indent=2)
            
    def load_history(self, file_path: str):
        """Load event history
        
        Args:
            file_path: Path to load file
        """
        with open(file_path, 'r') as f:
            history_data = json.load(f)
            
        self._event_history = [
            Event.from_dict(data) for data in history_data
        ]
        
    async def process(self, input_data: Dict[str, Any]) -> Any:
        """Process event-related requests
        
        Args:
            input_data: Request data
            
        Returns:
            Any: Response data
        """
        action = input_data.get('action')
        
        if action == 'emit':
            event_data = input_data.get('event')
            event = Event.from_dict(event_data)
            await self.emit(event)
            return {'status': 'success', 'message': 'Event emitted'}
        
        elif action == 'get_history':
            event_type = input_data.get('event_type')
            start_time = input_data.get('start_time')
            end_time = input_data.get('end_time')
            
            if start_time:
                start_time = datetime.fromisoformat(start_time)
            if end_time:
                end_time = datetime.fromisoformat(end_time)
            
            events = self.get_history(event_type, start_time, end_time)
            return {'status': 'success', 'events': [e.to_dict() for e in events]}
        
        else:
            raise ValueError(f"Unsupported action: {action}")
            
    async def validate(self, data: Any) -> bool:
        """Validate event-related request data
        
        Args:
            data: Request data
            
        Returns:
            bool: Whether data is valid
        """
        if not isinstance(data, dict):
            return False
        
        required_fields = ['action']
        return all(field in data for field in required_fields)
