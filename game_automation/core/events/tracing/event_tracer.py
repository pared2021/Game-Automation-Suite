"""Event tracing and logging implementation."""

import abc
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union

from .events import (
    Event,
    EventCategory,
    EventPriority,
    IEvent
)

# 设置日志
logger = logging.getLogger(__name__)

class TraceLevel(Enum):
    """Trace level enumeration."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

@dataclass
class TraceEvent:
    """Trace event data."""
    
    timestamp: float
    level: TraceLevel
    event_id: str
    event_type: str
    category: EventCategory
    source: Optional[str]
    message: str
    duration: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "level": self.level.name,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "category": self.category.name,
            "source": self.source,
            "message": self.message,
            "duration": self.duration,
            "error": self.error,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceEvent':
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            level=TraceLevel[data["level"]],
            event_id=data["event_id"],
            event_type=data["event_type"],
            category=EventCategory[data["category"]],
            source=data["source"],
            message=data["message"],
            duration=data.get("duration"),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )

class IEventTracer(abc.ABC):
    """Event tracer interface."""
    
    @abc.abstractmethod
    def start_trace(self, event: IEvent, message: str) -> None:
        """Start tracing an event.
        
        Args:
            event: Event to trace
            message: Trace message
        """
        pass
        
    @abc.abstractmethod
    def end_trace(self, event: IEvent, error: Optional[str] = None) -> None:
        """End tracing an event.
        
        Args:
            event: Event to trace
            error: Optional error message
        """
        pass
        
    @abc.abstractmethod
    def log(self, level: TraceLevel, event: IEvent,
            message: str, error: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a trace event.
        
        Args:
            level: Trace level
            event: Event to trace
            message: Log message
            error: Optional error message
            metadata: Optional metadata
        """
        pass
        
    @abc.abstractmethod
    def get_traces(self, event_id: Optional[str] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None) -> List[TraceEvent]:
        """Get trace events.
        
        Args:
            event_id: Optional event ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List[TraceEvent]: List of trace events
        """
        pass

@dataclass
class EventTracer(IEventTracer):
    """Event tracer implementation."""
    
    traces: List[TraceEvent] = field(default_factory=list)
    _active_traces: Dict[str, float] = field(default_factory=dict)
    max_traces: int = 10000
    
    def start_trace(self, event: IEvent, message: str) -> None:
        """Start tracing an event."""
        self._active_traces[event.id] = time.time()
        self.log(TraceLevel.INFO, event, f"Start: {message}")
        
    def end_trace(self, event: IEvent, error: Optional[str] = None) -> None:
        """End tracing an event."""
        if event.id in self._active_traces:
            start_time = self._active_traces.pop(event.id)
            duration = time.time() - start_time
            
            level = TraceLevel.ERROR if error else TraceLevel.INFO
            message = f"End: Duration {duration:.3f}s"
            
            self.log(level, event, message, error, {
                "duration": duration
            })
        
    def log(self, level: TraceLevel, event: IEvent,
            message: str, error: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a trace event."""
        trace = TraceEvent(
            timestamp=time.time(),
            level=level,
            event_id=event.id,
            event_type=event.type,
            category=event.category,
            source=event.source,
            message=message,
            error=error,
            metadata=metadata or {}
        )
        
        self.traces.append(trace)
        
        # 限制追踪事件数量
        if len(self.traces) > self.max_traces:
            self.traces = self.traces[-self.max_traces:]
            
        # 同时记录到日志系统
        log_message = f"[{event.id}] {event.type}: {message}"
        if error:
            log_message += f" Error: {error}"
            
        if level == TraceLevel.DEBUG:
            logger.debug(log_message)
        elif level == TraceLevel.INFO:
            logger.info(log_message)
        elif level == TraceLevel.WARNING:
            logger.warning(log_message)
        elif level == TraceLevel.ERROR:
            logger.error(log_message)
        elif level == TraceLevel.CRITICAL:
            logger.critical(log_message)
        
    def get_traces(self, event_id: Optional[str] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None) -> List[TraceEvent]:
        """Get trace events."""
        filtered = self.traces
        
        if event_id:
            filtered = [t for t in filtered if t.event_id == event_id]
            
        if start_time:
            filtered = [t for t in filtered if t.timestamp >= start_time]
            
        if end_time:
            filtered = [t for t in filtered if t.timestamp <= end_time]
            
        return filtered
        
    def export_traces(self, filepath: str) -> None:
        """Export traces to JSON file.
        
        Args:
            filepath: Output file path
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(
                    [t.to_dict() for t in self.traces],
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Error exporting traces: {e}")
            
    def import_traces(self, filepath: str) -> None:
        """Import traces from JSON file.
        
        Args:
            filepath: Input file path
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.traces = [
                    TraceEvent.from_dict(t) for t in data
                ]
        except Exception as e:
            logger.error(f"Error importing traces: {e}")
            
    def clear_traces(self) -> None:
        """Clear all traces."""
        self.traces.clear()
        self._active_traces.clear()

# 创建全局事件追踪器实例
event_tracer = EventTracer()
