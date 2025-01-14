"""Event filter system implementation."""

import abc
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Type, Union

from .events import (
    Event,
    EventCategory,
    EventPriority,
    IEvent
)

class IEventFilter(abc.ABC):
    """Event filter interface."""
    
    @abc.abstractmethod
    def filter(self, event: IEvent) -> bool:
        """Filter an event.
        
        Args:
            event: Event to filter
            
        Returns:
            bool: True if event passes filter
        """
        pass
        
    def __and__(self, other: 'IEventFilter') -> 'CompositeEventFilter':
        """Combine filters with AND operation."""
        return CompositeEventFilter([self, other], all_must_pass=True)
        
    def __or__(self, other: 'IEventFilter') -> 'CompositeEventFilter':
        """Combine filters with OR operation."""
        return CompositeEventFilter([self, other], all_must_pass=False)
        
    def __invert__(self) -> 'InvertEventFilter':
        """Invert filter result."""
        return InvertEventFilter(self)

@dataclass
class TypeEventFilter(IEventFilter):
    """Event type filter implementation."""
    
    types: Set[str]
    use_regex: bool = False
    _patterns: Optional[List[Pattern]] = None
    
    def __post_init__(self):
        """Initialize regex patterns if needed."""
        if self.use_regex:
            self._patterns = [re.compile(t) for t in self.types]
    
    def filter(self, event: IEvent) -> bool:
        """Filter event by type."""
        if self.use_regex:
            return any(p.match(event.type) for p in self._patterns)
        else:
            return event.type in self.types

@dataclass
class CategoryEventFilter(IEventFilter):
    """Event category filter implementation."""
    
    categories: Set[EventCategory]
    
    def filter(self, event: IEvent) -> bool:
        """Filter event by category."""
        return event.category in self.categories

@dataclass
class PriorityEventFilter(IEventFilter):
    """Event priority filter implementation."""
    
    min_priority: EventPriority
    max_priority: EventPriority = EventPriority.HIGHEST
    
    def filter(self, event: IEvent) -> bool:
        """Filter event by priority range."""
        return self.min_priority <= event.priority <= self.max_priority

@dataclass
class SourceEventFilter(IEventFilter):
    """Event source filter implementation."""
    
    sources: Set[str]
    use_regex: bool = False
    _patterns: Optional[List[Pattern]] = None
    
    def __post_init__(self):
        """Initialize regex patterns if needed."""
        if self.use_regex:
            self._patterns = [re.compile(s) for s in self.sources]
    
    def filter(self, event: IEvent) -> bool:
        """Filter event by source."""
        if event.source is None:
            return False
            
        if self.use_regex:
            return any(p.match(event.source) for p in self._patterns)
        else:
            return event.source in self.sources

@dataclass
class DataTypeEventFilter(IEventFilter):
    """Event data type filter implementation."""
    
    types: Set[Type]
    allow_none: bool = True
    
    def filter(self, event: IEvent) -> bool:
        """Filter event by data type."""
        if event.data is None:
            return self.allow_none
            
        return any(isinstance(event.data, t) for t in self.types)

@dataclass
class MetadataEventFilter(IEventFilter):
    """Event metadata filter implementation."""
    
    key: str
    value: Any
    
    def filter(self, event: IEvent) -> bool:
        """Filter event by metadata key-value pair."""
        return (
            self.key in event.metadata and
            event.metadata[self.key] == self.value
        )

@dataclass
class PredicateEventFilter(IEventFilter):
    """Event predicate filter implementation."""
    
    predicate: Callable[[IEvent], bool]
    
    def filter(self, event: IEvent) -> bool:
        """Filter event using predicate."""
        return self.predicate(event)

@dataclass
class CompositeEventFilter(IEventFilter):
    """Composite event filter implementation."""
    
    filters: List[IEventFilter]
    all_must_pass: bool = True
    
    def filter(self, event: IEvent) -> bool:
        """Filter event using all filters."""
        if self.all_must_pass:
            return all(f.filter(event) for f in self.filters)
        else:
            return any(f.filter(event) for f in self.filters)

@dataclass
class InvertEventFilter(IEventFilter):
    """Invert event filter implementation."""
    
    base_filter: IEventFilter
    
    def filter(self, event: IEvent) -> bool:
        """Invert base filter result."""
        return not self.base_filter.filter(event)

class EventFilterFactory:
    """Factory for creating common event filters."""
    
    @staticmethod
    def create_type_filter(types: Union[str, Set[str]], use_regex: bool = False) -> TypeEventFilter:
        """Create a type filter."""
        if isinstance(types, str):
            types = {types}
        return TypeEventFilter(types, use_regex)
        
    @staticmethod
    def create_category_filter(categories: Union[EventCategory, Set[EventCategory]]) -> CategoryEventFilter:
        """Create a category filter."""
        if isinstance(categories, EventCategory):
            categories = {categories}
        return CategoryEventFilter(categories)
        
    @staticmethod
    def create_priority_filter(min_priority: EventPriority,
                             max_priority: Optional[EventPriority] = None) -> PriorityEventFilter:
        """Create a priority filter."""
        if max_priority is None:
            max_priority = EventPriority.HIGHEST
        return PriorityEventFilter(min_priority, max_priority)
        
    @staticmethod
    def create_source_filter(sources: Union[str, Set[str]], use_regex: bool = False) -> SourceEventFilter:
        """Create a source filter."""
        if isinstance(sources, str):
            sources = {sources}
        return SourceEventFilter(sources, use_regex)
        
    @staticmethod
    def create_data_type_filter(types: Union[Type, Set[Type]], allow_none: bool = True) -> DataTypeEventFilter:
        """Create a data type filter."""
        if isinstance(types, type):
            types = {types}
        return DataTypeEventFilter(types, allow_none)
        
    @staticmethod
    def create_metadata_filter(key: str, value: Any) -> MetadataEventFilter:
        """Create a metadata filter."""
        return MetadataEventFilter(key, value)
        
    @staticmethod
    def create_predicate_filter(predicate: Callable[[IEvent], bool]) -> PredicateEventFilter:
        """Create a predicate filter."""
        return PredicateEventFilter(predicate)
        
    @staticmethod
    def create_system_filter() -> CategoryEventFilter:
        """Create a system event filter."""
        return CategoryEventFilter({EventCategory.SYSTEM})
        
    @staticmethod
    def create_game_filter() -> CategoryEventFilter:
        """Create a game event filter."""
        return CategoryEventFilter({EventCategory.GAME})
        
    @staticmethod
    def create_input_filter() -> CategoryEventFilter:
        """Create an input event filter."""
        return CategoryEventFilter({EventCategory.INPUT})
        
    @staticmethod
    def create_network_filter() -> CategoryEventFilter:
        """Create a network event filter."""
        return CategoryEventFilter({EventCategory.NETWORK})
        
    @staticmethod
    def create_resource_filter() -> CategoryEventFilter:
        """Create a resource event filter."""
        return CategoryEventFilter({EventCategory.RESOURCE})
        
    @staticmethod
    def create_high_priority_filter() -> PriorityEventFilter:
        """Create a high priority filter."""
        return PriorityEventFilter(EventPriority.HIGH)
