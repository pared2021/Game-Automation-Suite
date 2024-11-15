from typing import Any, Dict, List, Callable, Optional, Set
import asyncio
from datetime import datetime
import json
from ..base.service_base import ServiceBase

class Event:
    """Represents a system event."""
    
    def __init__(self, event_type: str, data: Any = None, source: str = None):
        self.type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.now()
        self.id = f"{event_type}_{self.timestamp.timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'id': self.id,
            'type': self.type,
            'data': self.data,
            'source': self.source,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        event = cls(data['type'], data['data'], data['source'])
        event.id = data['id']
        event.timestamp = datetime.fromisoformat(data['timestamp'])
        return event

class EventManager(ServiceBase):
    """Centralized event management service."""
    
    def __init__(self):
        super().__init__("EventManager")
        self._handlers: Dict[str, Set[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._is_processing = False
        self._processing_task: Optional[asyncio.Task] = None
    
    async def _on_start(self) -> None:
        """Start event processing on service start."""
        self._is_processing = True
        self._processing_task = asyncio.create_task(self._process_events())
        self.log_info("Event processing started")
    
    async def _on_stop(self) -> None:
        """Stop event processing on service stop."""
        self._is_processing = False
        if self._processing_task:
            await self._processing_task
        self.log_info("Event processing stopped")
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._is_processing:
            try:
                event = await self._event_queue.get()
                await self._handle_event(event)
                self._event_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_error("Error processing event", e)
    
    async def _handle_event(self, event: Event) -> None:
        """Handle a single event."""
        handlers = self._handlers.get(event.type, set())
        handlers.update(self._handlers.get('*', set()))  # Add global handlers
        
        if not handlers:
            self.log_debug(f"No handlers registered for event type: {event.type}")
            return
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self.log_error(f"Error in event handler for {event.type}", e)
        
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to events of specified type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = set()
        self._handlers[event_type].add(handler)
        self.log_debug(f"Subscribed handler to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from events of specified type."""
        if event_type in self._handlers:
            self._handlers[event_type].discard(handler)
            if not self._handlers[event_type]:
                del self._handlers[event_type]
            self.log_debug(f"Unsubscribed handler from event type: {event_type}")
    
    async def emit(self, event: Event) -> None:
        """Emit an event."""
        await self._event_queue.put(event)
        self.log_debug(f"Emitted event: {event.type}")
    
    async def emit_immediate(self, event: Event) -> None:
        """Emit and handle an event immediately."""
        await self._handle_event(event)
    
    def get_history(self, event_type: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Event]:
        """Get event history with optional filtering."""
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return events
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        self.log_info("Event history cleared")
    
    def save_history(self, file_path: str) -> None:
        """Save event history to file."""
        try:
            history_data = [event.to_dict() for event in self._event_history]
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2)
            self.log_info(f"Event history saved to: {file_path}")
        except Exception as e:
            self.log_error(f"Error saving event history", e)
    
    def load_history(self, file_path: str) -> None:
        """Load event history from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            self._event_history = [Event.from_dict(data) for data in history_data]
            self.log_info(f"Event history loaded from: {file_path}")
        except Exception as e:
            self.log_error(f"Error loading event history", e)
    
    async def process(self, input_data: Dict[str, Any]) -> Any:
        """Process event-related requests."""
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
        """Validate event-related request data."""
        if not isinstance(data, dict):
            return False
        
        required_fields = ['action']
        return all(field in data for field in required_fields)
