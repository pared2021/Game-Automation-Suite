"""Event Type System for Game Automation Suite.

This module defines the event type system, including:
- Event interfaces and base classes
- Event validation system
- Event factory system
"""

import abc
import enum
import uuid
from typing import Any, Dict, Optional, Set, Type, TypeVar
from dataclasses import dataclass, field
from datetime import datetime

T = TypeVar('T', bound='IEvent')

class EventPriority(enum.IntEnum):
    """Event priority levels."""
    LOWEST = 0
    LOW = 2
    NORMAL = 5
    HIGH = 7
    HIGHEST = 9
    
    @classmethod
    def from_int(cls, value: int) -> 'EventPriority':
        """Convert integer to EventPriority."""
        if value <= 1:
            return cls.LOWEST
        elif value <= 3:
            return cls.LOW
        elif value <= 6:
            return cls.NORMAL
        elif value <= 8:
            return cls.HIGH
        else:
            return cls.HIGHEST

class EventCategory(enum.Enum):
    """Event categories."""
    SYSTEM = "system"
    GAME = "game"
    INPUT = "input"
    NETWORK = "network"
    RESOURCE = "resource"
    CUSTOM = "custom"

class IEvent(abc.ABC):
    """Event interface."""
    
    @property
    @abc.abstractmethod
    def id(self) -> str:
        """Get event ID."""
        pass
        
    @property
    @abc.abstractmethod
    def type(self) -> str:
        """Get event type."""
        pass
        
    @property
    @abc.abstractmethod
    def category(self) -> EventCategory:
        """Get event category."""
        pass
        
    @property
    @abc.abstractmethod
    def priority(self) -> EventPriority:
        """Get event priority."""
        pass
        
    @property
    @abc.abstractmethod
    def timestamp(self) -> datetime:
        """Get event timestamp."""
        pass
        
    @property
    @abc.abstractmethod
    def source(self) -> Optional[str]:
        """Get event source."""
        pass
        
    @property
    @abc.abstractmethod
    def data(self) -> Any:
        """Get event data."""
        pass
        
    @property
    @abc.abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Get event metadata."""
        pass
        
    @abc.abstractmethod
    def validate(self) -> bool:
        """Validate the event."""
        pass
        
    @abc.abstractmethod
    def clone(self: T) -> T:
        """Create a copy of the event."""
        pass

@dataclass
class Event(IEvent):
    """Base event implementation."""
    
    _type: str
    _category: EventCategory
    _priority: EventPriority = field(default=EventPriority.NORMAL)
    _data: Any = None
    _source: Optional[str] = None
    _metadata: Dict[str, Any] = field(default_factory=dict)
    _id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def id(self) -> str:
        return self._id
        
    @property
    def type(self) -> str:
        return self._type
        
    @property
    def category(self) -> EventCategory:
        return self._category
        
    @property
    def priority(self) -> EventPriority:
        return self._priority
        
    @property
    def timestamp(self) -> datetime:
        return self._timestamp
        
    @property
    def source(self) -> Optional[str]:
        return self._source
        
    @property
    def data(self) -> Any:
        return self._data
        
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata
        
    def validate(self) -> bool:
        """Basic validation."""
        return True
        
    def clone(self: T) -> T:
        """Create a copy of the event."""
        return self.__class__(
            type=self._type,
            category=self._category,
            priority=self._priority,
            data=self._data,
            source=self._source,
            metadata=self._metadata.copy()
        )

@dataclass
class SystemEvent(Event):
    """System event implementation."""
    
    def __init__(self, type: str, data: Any = None,
                 priority: EventPriority = EventPriority.HIGH,
                 source: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            _type=type,
            _category=EventCategory.SYSTEM,
            _priority=priority,
            _data=data,
            _source=source,
            _metadata=metadata or {}
        )
        
    def validate(self) -> bool:
        """Validate system event."""
        return (
            super().validate() and
            self.category == EventCategory.SYSTEM and
            self.priority >= EventPriority.NORMAL
        )

@dataclass
class GameEvent(Event):
    """Game event implementation."""
    
    def __init__(self, type: str, data: Any = None,
                 priority: EventPriority = EventPriority.NORMAL,
                 source: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            _type=type,
            _category=EventCategory.GAME,
            _priority=priority,
            _data=data,
            _source=source,
            _metadata=metadata or {}
        )
        
    def validate(self) -> bool:
        """Validate game event."""
        return (
            super().validate() and
            self.category == EventCategory.GAME
        )

@dataclass
class InputEvent(Event):
    """Input event implementation."""
    
    def __init__(self, type: str, data: Any = None,
                 priority: EventPriority = EventPriority.HIGH,
                 source: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            _type=type,
            _category=EventCategory.INPUT,
            _priority=priority,
            _data=data,
            _source=source,
            _metadata=metadata or {}
        )
        
    def validate(self) -> bool:
        """Validate input event."""
        return (
            super().validate() and
            self.category == EventCategory.INPUT and
            self.priority >= EventPriority.NORMAL
        )

@dataclass
class NetworkEvent(Event):
    """Network event implementation."""
    
    def __init__(self, type: str, data: Any = None,
                 priority: EventPriority = EventPriority.NORMAL,
                 source: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            _type=type,
            _category=EventCategory.NETWORK,
            _priority=priority,
            _data=data,
            _source=source,
            _metadata=metadata or {}
        )
        
    def validate(self) -> bool:
        """Validate network event."""
        return (
            super().validate() and
            self.category == EventCategory.NETWORK
        )

@dataclass
class ResourceEvent(Event):
    """Resource event implementation."""
    
    def __init__(self, type: str, data: Any = None,
                 priority: EventPriority = EventPriority.NORMAL,
                 source: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            _type=type,
            _category=EventCategory.RESOURCE,
            _priority=priority,
            _data=data,
            _source=source,
            _metadata=metadata or {}
        )
        
    def validate(self) -> bool:
        """Validate resource event."""
        return (
            super().validate() and
            self.category == EventCategory.RESOURCE
        )

class IEventValidator(abc.ABC):
    """Event validator interface."""
    
    @abc.abstractmethod
    def validate(self, event: IEvent) -> bool:
        """Validate an event.
        
        Args:
            event: Event to validate
            
        Returns:
            bool: True if event is valid
        """
        pass
        
    @abc.abstractmethod
    def get_error(self) -> Optional[str]:
        """Get the last validation error.
        
        Returns:
            Optional[str]: Error message if validation failed, None otherwise
        """
        pass
        
    def __and__(self, other: 'IEventValidator') -> 'CompositeEventValidator':
        """Combine validators with AND operation."""
        return CompositeEventValidator([self, other], all_must_pass=True)
        
    def __or__(self, other: 'IEventValidator') -> 'CompositeEventValidator':
        """Combine validators with OR operation."""
        return CompositeEventValidator([self, other], all_must_pass=False)

@dataclass
class BaseEventValidator(IEventValidator):
    """Base event validator implementation."""
    
    _last_error: Optional[str] = None
    
    def validate(self, event: IEvent) -> bool:
        """Validate basic event properties."""
        if not isinstance(event, IEvent):
            self._last_error = f"Expected IEvent, got {type(event)}"
            return False
            
        if not event.type:
            self._last_error = "Event type cannot be empty"
            return False
            
        if not event.category:
            self._last_error = "Event category cannot be empty"
            return False
            
        if not event.priority:
            self._last_error = "Event priority cannot be empty"
            return False
            
        if not event.timestamp:
            self._last_error = "Event timestamp cannot be empty"
            return False
            
        self._last_error = None
        return True
        
    def get_error(self) -> Optional[str]:
        return self._last_error

@dataclass
class TypeEventValidator(IEventValidator):
    """Event type validator implementation."""
    
    allowed_types: Set[str]
    _last_error: Optional[str] = None
    
    def validate(self, event: IEvent) -> bool:
        """Validate event type."""
        if event.type not in self.allowed_types:
            self._last_error = f"Event type {event.type} not allowed"
            return False
            
        self._last_error = None
        return True
        
    def get_error(self) -> Optional[str]:
        return self._last_error

@dataclass
class CategoryEventValidator(IEventValidator):
    """Event category validator implementation."""
    
    allowed_categories: Set[EventCategory]
    _last_error: Optional[str] = None
    
    def validate(self, event: IEvent) -> bool:
        """Validate event category."""
        if event.category not in self.allowed_categories:
            self._last_error = f"Event category {event.category} not allowed"
            return False
            
        self._last_error = None
        return True
        
    def get_error(self) -> Optional[str]:
        return self._last_error

@dataclass
class PriorityEventValidator(IEventValidator):
    """Event priority validator implementation."""
    
    min_priority: EventPriority
    max_priority: EventPriority = EventPriority.HIGHEST
    _last_error: Optional[str] = None
    
    def validate(self, event: IEvent) -> bool:
        """Validate event priority."""
        if not (self.min_priority <= event.priority <= self.max_priority):
            self._last_error = f"Event priority {event.priority} not in range [{self.min_priority}, {self.max_priority}]"
            return False
            
        self._last_error = None
        return True
        
    def get_error(self) -> Optional[str]:
        return self._last_error

@dataclass
class DataTypeEventValidator(IEventValidator):
    """Event data type validator implementation."""
    
    allowed_types: Set[Type]
    _last_error: Optional[str] = None
    
    def validate(self, event: IEvent) -> bool:
        """Validate event data type."""
        if event.data is not None and not any(isinstance(event.data, t) for t in self.allowed_types):
            self._last_error = f"Event data type {type(event.data)} not allowed"
            return False
            
        self._last_error = None
        return True
        
    def get_error(self) -> Optional[str]:
        return self._last_error

@dataclass
class CompositeEventValidator(IEventValidator):
    """Composite event validator implementation."""
    
    validators: list[IEventValidator]
    all_must_pass: bool = True
    _last_error: Optional[str] = None
    
    def validate(self, event: IEvent) -> bool:
        """Validate event using all validators."""
        errors = []
        passed = 0
        
        for validator in self.validators:
            if validator.validate(event):
                passed += 1
            else:
                error = validator.get_error()
                if error:
                    errors.append(error)
                    
        if self.all_must_pass:
            is_valid = passed == len(self.validators)
        else:
            is_valid = passed > 0
            
        if not is_valid:
            self._last_error = "; ".join(errors)
        else:
            self._last_error = None
            
        return is_valid
        
    def get_error(self) -> Optional[str]:
        return self._last_error

class EventValidatorFactory:
    """Factory for creating common event validators."""
    
    @staticmethod
    def create_base_validator() -> BaseEventValidator:
        """Create a base validator."""
        return BaseEventValidator()
        
    @staticmethod
    def create_system_validator() -> CompositeEventValidator:
        """Create a system event validator."""
        return CompositeEventValidator([
            BaseEventValidator(),
            CategoryEventValidator({EventCategory.SYSTEM}),
            PriorityEventValidator(EventPriority.NORMAL)
        ])
        
    @staticmethod
    def create_game_validator() -> CompositeEventValidator:
        """Create a game event validator."""
        return CompositeEventValidator([
            BaseEventValidator(),
            CategoryEventValidator({EventCategory.GAME})
        ])
        
    @staticmethod
    def create_input_validator() -> CompositeEventValidator:
        """Create an input event validator."""
        return CompositeEventValidator([
            BaseEventValidator(),
            CategoryEventValidator({EventCategory.INPUT}),
            PriorityEventValidator(EventPriority.NORMAL)
        ])
        
    @staticmethod
    def create_network_validator() -> CompositeEventValidator:
        """Create a network event validator."""
        return CompositeEventValidator([
            BaseEventValidator(),
            CategoryEventValidator({EventCategory.NETWORK})
        ])
        
    @staticmethod
    def create_resource_validator() -> CompositeEventValidator:
        """Create a resource event validator."""
        return CompositeEventValidator([
            BaseEventValidator(),
            CategoryEventValidator({EventCategory.RESOURCE})
        ])

class IEventFactory(abc.ABC):
    """Event factory interface."""
    
    @abc.abstractmethod
    def create_event(self, type: str, data: Any = None,
                    priority: Optional[EventPriority] = None,
                    source: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> IEvent:
        """Create an event.
        
        Args:
            type: Event type
            data: Event data
            priority: Event priority
            source: Event source
            metadata: Event metadata
            
        Returns:
            IEvent: Created event
        """
        pass
        
    @abc.abstractmethod
    def register_validator(self, validator: IEventValidator) -> None:
        """Register a validator for created events.
        
        Args:
            validator: Validator to register
        """
        pass

class EventBuilder:
    """Event builder implementation."""
    
    def __init__(self, category: EventCategory):
        """Initialize the event builder.
        
        Args:
            category: Event category
        """
        self._category = category
        self._type: Optional[str] = None
        self._data: Any = None
        self._priority: EventPriority = EventPriority.NORMAL
        self._source: Optional[str] = None
        self._metadata: Dict[str, Any] = {}
        self._validator: Optional[IEventValidator] = None
        
    def type(self, type: str) -> 'EventBuilder':
        """Set event type."""
        self._type = type
        return self
        
    def data(self, data: Any) -> 'EventBuilder':
        """Set event data."""
        self._data = data
        return self
        
    def priority(self, priority: EventPriority) -> 'EventBuilder':
        """Set event priority."""
        self._priority = priority
        return self
        
    def source(self, source: str) -> 'EventBuilder':
        """Set event source."""
        self._source = source
        return self
        
    def metadata(self, metadata: Dict[str, Any]) -> 'EventBuilder':
        """Set event metadata."""
        self._metadata = metadata
        return self
        
    def add_metadata(self, key: str, value: Any) -> 'EventBuilder':
        """Add metadata entry."""
        self._metadata[key] = value
        return self
        
    def validator(self, validator: IEventValidator) -> 'EventBuilder':
        """Set event validator."""
        self._validator = validator
        return self
        
    def build(self) -> IEvent:
        """Build the event."""
        if not self._type:
            raise ValueError("Event type is required")
            
        # Create event based on category
        if self._category == EventCategory.SYSTEM:
            event = SystemEvent(
                self._type,
                self._data,
                self._priority,
                self._source,
                self._metadata
            )
        elif self._category == EventCategory.GAME:
            event = GameEvent(
                self._type,
                self._data,
                self._priority,
                self._source,
                self._metadata
            )
        elif self._category == EventCategory.INPUT:
            event = InputEvent(
                self._type,
                self._data,
                self._priority,
                self._source,
                self._metadata
            )
        elif self._category == EventCategory.NETWORK:
            event = NetworkEvent(
                self._type,
                self._data,
                self._priority,
                self._source,
                self._metadata
            )
        elif self._category == EventCategory.RESOURCE:
            event = ResourceEvent(
                self._type,
                self._data,
                self._priority,
                self._source,
                self._metadata
            )
        else:
            event = Event(
                _type=self._type,
                _category=self._category,
                _priority=self._priority,
                _data=self._data,
                _source=self._source,
                _metadata=self._metadata
            )
            
        # Validate event if validator is set
        if self._validator and not self._validator.validate(event):
            raise ValueError(f"Event validation failed: {self._validator.get_error()}")
            
        return event

class EventFactory(IEventFactory):
    """Event factory implementation."""
    
    def __init__(self):
        """Initialize the event factory."""
        self._validators: Dict[EventCategory, IEventValidator] = {
            EventCategory.SYSTEM: EventValidatorFactory.create_system_validator(),
            EventCategory.GAME: EventValidatorFactory.create_game_validator(),
            EventCategory.INPUT: EventValidatorFactory.create_input_validator(),
            EventCategory.NETWORK: EventValidatorFactory.create_network_validator(),
            EventCategory.RESOURCE: EventValidatorFactory.create_resource_validator(),
        }
        
    def create_event(self, type: str, data: Any = None,
                    priority: Optional[EventPriority] = None,
                    source: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> IEvent:
        """Create an event using the builder pattern."""
        # Determine category from type prefix
        category = self._get_category_from_type(type)
        
        # Create builder
        builder = EventBuilder(category)
        
        # Build event
        builder.type(type).data(data)
        
        if priority is not None:
            builder.priority(priority)
            
        if source is not None:
            builder.source(source)
            
        if metadata is not None:
            builder.metadata(metadata)
            
        # Add validator
        if category in self._validators:
            builder.validator(self._validators[category])
            
        return builder.build()
        
    def register_validator(self, validator: IEventValidator,
                         category: Optional[EventCategory] = None) -> None:
        """Register a validator for a category."""
        if category is None:
            # Register for all categories
            for cat in EventCategory:
                if cat in self._validators:
                    self._validators[cat] = (
                        self._validators[cat] & validator
                    )
        else:
            # Register for specific category
            if category in self._validators:
                self._validators[category] = (
                    self._validators[category] & validator
                )
            else:
                self._validators[category] = validator
                
    def _get_category_from_type(self, type: str) -> EventCategory:
        """Determine event category from type."""
        type_lower = type.lower()
        
        if type_lower.startswith("system."):
            return EventCategory.SYSTEM
        elif type_lower.startswith("game."):
            return EventCategory.GAME
        elif type_lower.startswith("input."):
            return EventCategory.INPUT
        elif type_lower.startswith("network."):
            return EventCategory.NETWORK
        elif type_lower.startswith("resource."):
            return EventCategory.RESOURCE
        else:
            return EventCategory.CUSTOM

# Create global event factory instance
event_factory = EventFactory()
