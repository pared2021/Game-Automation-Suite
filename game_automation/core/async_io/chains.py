"""Event chain and group implementation."""

import abc
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Type, Union

from .events import (
    Event,
    EventCategory,
    EventPriority,
    IEvent,
    event_factory
)
from .filters import (
    IEventFilter,
    EventFilterFactory
)
from .dispatcher import (
    EventHandler,
    IEventDispatcher,
    event_dispatcher
)

# 设置日志
logger = logging.getLogger(__name__)

class IEventTransformer(abc.ABC):
    """Event transformer interface."""
    
    @abc.abstractmethod
    async def transform(self, event: IEvent) -> Optional[IEvent]:
        """Transform an event.
        
        Args:
            event: Event to transform
            
        Returns:
            Optional[IEvent]: Transformed event or None if event should be dropped
        """
        pass

@dataclass
class EventTypeTransformer(IEventTransformer):
    """Event type transformer implementation."""
    
    target_type: str
    copy_data: bool = True
    copy_metadata: bool = True
    
    async def transform(self, event: IEvent) -> Optional[IEvent]:
        """Transform event type."""
        return event_factory.create_event(
            type=self.target_type,
            data=event.data if self.copy_data else None,
            priority=event.priority,
            source=event.source,
            metadata=event.metadata if self.copy_metadata else None
        )

@dataclass
class EventCategoryTransformer(IEventTransformer):
    """Event category transformer implementation."""
    
    target_category: EventCategory
    
    async def transform(self, event: IEvent) -> Optional[IEvent]:
        """Transform event category."""
        # 保持原始类型的后缀部分
        type_parts = event.type.split(".")
        new_type = f"{self.target_category.name.lower()}.{'.'.join(type_parts[1:])}"
        
        return event_factory.create_event(
            type=new_type,
            data=event.data,
            priority=event.priority,
            source=event.source,
            metadata=event.metadata
        )

class IEventChain(abc.ABC):
    """Event chain interface."""
    
    @abc.abstractmethod
    async def process(self, event: IEvent) -> List[IEvent]:
        """Process an event through the chain.
        
        Args:
            event: Event to process
            
        Returns:
            List[IEvent]: List of resulting events
        """
        pass

@dataclass
class EventChain(IEventChain):
    """Event chain implementation."""
    
    name: str
    filters: List[IEventFilter] = field(default_factory=list)
    transformers: List[IEventTransformer] = field(default_factory=list)
    handlers: List[EventHandler] = field(default_factory=list)
    
    async def process(self, event: IEvent) -> List[IEvent]:
        """Process event through chain."""
        events = [event]
        result_events = []
        
        try:
            # 应用所有过滤器
            for filter in self.filters:
                events = [e for e in events if filter.filter(e)]
                if not events:
                    return []
                    
            # 应用所有转换器
            for transformer in self.transformers:
                new_events = []
                for e in events:
                    transformed = await transformer.transform(e)
                    if transformed:
                        new_events.append(transformed)
                events = new_events
                if not events:
                    return []
                    
            # 调用所有处理器
            for handler in self.handlers:
                for e in events:
                    try:
                        await handler(e)
                        result_events.append(e)
                    except Exception as ex:
                        logger.error(f"Error in chain {self.name} handler: {ex}")
                        
            return result_events
            
        except Exception as e:
            logger.error(f"Error processing event chain {self.name}: {e}")
            return []

class IEventGroup(abc.ABC):
    """Event group interface."""
    
    @abc.abstractmethod
    async def process(self, event: IEvent) -> bool:
        """Process an event through all chains in the group.
        
        Args:
            event: Event to process
            
        Returns:
            bool: True if event was processed by any chain
        """
        pass
        
    @abc.abstractmethod
    def add_chain(self, chain: IEventChain) -> None:
        """Add a chain to the group.
        
        Args:
            chain: Chain to add
        """
        pass
        
    @abc.abstractmethod
    def remove_chain(self, chain: IEventChain) -> None:
        """Remove a chain from the group.
        
        Args:
            chain: Chain to remove
        """
        pass

@dataclass
class EventGroup(IEventGroup):
    """Event group implementation."""
    
    name: str
    chains: List[IEventChain] = field(default_factory=list)
    parallel: bool = True
    
    async def process(self, event: IEvent) -> bool:
        """Process event through all chains."""
        try:
            if self.parallel:
                # 并行处理所有链
                tasks = [
                    asyncio.create_task(chain.process(event))
                    for chain in self.chains
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 检查结果
                success = False
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in group {self.name}: {result}")
                    elif result:
                        success = True
                return success
                
            else:
                # 串行处理所有链
                success = False
                for chain in self.chains:
                    try:
                        result = await chain.process(event)
                        if result:
                            success = True
                    except Exception as e:
                        logger.error(f"Error in group {self.name} chain: {e}")
                return success
                
        except Exception as e:
            logger.error(f"Error processing event group {self.name}: {e}")
            return False
            
    def add_chain(self, chain: IEventChain) -> None:
        """Add chain to group."""
        if chain not in self.chains:
            self.chains.append(chain)
            
    def remove_chain(self, chain: IEventChain) -> None:
        """Remove chain from group."""
        if chain in self.chains:
            self.chains.remove(chain)

class EventChainBuilder:
    """Event chain builder."""
    
    def __init__(self, name: str):
        """Initialize builder."""
        self._name = name
        self._filters: List[IEventFilter] = []
        self._transformers: List[IEventTransformer] = []
        self._handlers: List[EventHandler] = []
        
    def add_filter(self, filter: IEventFilter) -> 'EventChainBuilder':
        """Add filter to chain."""
        self._filters.append(filter)
        return self
        
    def add_transformer(self, transformer: IEventTransformer) -> 'EventChainBuilder':
        """Add transformer to chain."""
        self._transformers.append(transformer)
        return self
        
    def add_handler(self, handler: EventHandler) -> 'EventChainBuilder':
        """Add handler to chain."""
        self._handlers.append(handler)
        return self
        
    def build(self) -> EventChain:
        """Build event chain."""
        return EventChain(
            name=self._name,
            filters=self._filters,
            transformers=self._transformers,
            handlers=self._handlers
        )

class EventGroupBuilder:
    """Event group builder."""
    
    def __init__(self, name: str):
        """Initialize builder."""
        self._name = name
        self._chains: List[IEventChain] = []
        self._parallel = True
        
    def add_chain(self, chain: IEventChain) -> 'EventGroupBuilder':
        """Add chain to group."""
        self._chains.append(chain)
        return self
        
    def set_parallel(self, parallel: bool) -> 'EventGroupBuilder':
        """Set parallel processing flag."""
        self._parallel = parallel
        return self
        
    def build(self) -> EventGroup:
        """Build event group."""
        return EventGroup(
            name=self._name,
            chains=self._chains,
            parallel=self._parallel
        )
