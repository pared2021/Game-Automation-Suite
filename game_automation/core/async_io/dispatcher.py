"""Event dispatcher implementation."""

import abc
import asyncio
import logging
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union

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
from .tracing import (
    TraceLevel,
    IEventTracer,
    event_tracer
)

# 设置日志
logger = logging.getLogger(__name__)

# 事件处理器类型
EventHandler = Callable[[IEvent], Awaitable[None]]

@dataclass
class EventSubscription:
    """Event subscription information."""
    
    handler: EventHandler
    filters: List[IEventFilter]
    priority: EventPriority = EventPriority.NORMAL
    _active: bool = True
    
    @property
    def active(self) -> bool:
        """Get subscription active state."""
        return self._active
        
    def activate(self) -> None:
        """Activate subscription."""
        self._active = True
        
    def deactivate(self) -> None:
        """Deactivate subscription."""
        self._active = False
        
    async def handle(self, event: IEvent) -> bool:
        """Handle event if it passes all filters.
        
        Args:
            event: Event to handle
            
        Returns:
            bool: True if event was handled
        """
        if not self._active:
            return False
            
        # Check all filters
        for filter in self.filters:
            if not filter.filter(event):
                return False
                
        # Handle event
        try:
            await self.handler(event)
            return True
        except Exception as e:
            logger.error(f"Error handling event {event}: {e}")
            return False

class IEventDispatcher(abc.ABC):
    """Event dispatcher interface."""
    
    @abc.abstractmethod
    async def dispatch(self, event: IEvent) -> bool:
        """Dispatch event to handlers.
        
        Args:
            event: Event to dispatch
            
        Returns:
            bool: True if event was handled by any handler
        """
        pass
        
    @abc.abstractmethod
    def subscribe(self, handler: EventHandler,
                 filters: Optional[List[IEventFilter]] = None,
                 priority: EventPriority = EventPriority.NORMAL) -> EventSubscription:
        """Subscribe to events.
        
        Args:
            handler: Event handler
            filters: Event filters
            priority: Handler priority
            
        Returns:
            EventSubscription: Subscription object
        """
        pass
        
    @abc.abstractmethod
    def unsubscribe(self, subscription: EventSubscription) -> None:
        """Unsubscribe from events.
        
        Args:
            subscription: Subscription to remove
        """
        pass

@dataclass
class EventDispatcher(IEventDispatcher):
    """Event dispatcher implementation."""
    
    # 按优先级存储订阅
    _subscriptions: Dict[EventPriority, Set[EventSubscription]] = field(
        default_factory=lambda: {
            priority: set() for priority in EventPriority
        }
    )
    
    # 性能统计
    _dispatch_count: int = 0
    _handler_count: int = 0
    _error_count: int = 0
    _total_dispatch_time: float = 0
    
    async def dispatch(self, event: IEvent) -> bool:
        """Dispatch event to handlers."""
        start_time = time.time()
        handled = False
        
        try:
            # 开始追踪
            event_tracer.start_trace(event, "Dispatching event")
            
            # 按优先级从高到低分发
            for priority in sorted(EventPriority, reverse=True):
                if priority not in self._subscriptions:
                    continue
                    
                # 获取当前优先级的所有订阅
                subscriptions = self._subscriptions[priority]
                if not subscriptions:
                    continue
                    
                # 记录处理器数量
                event_tracer.log(
                    TraceLevel.DEBUG,
                    event,
                    f"Processing {len(subscriptions)} handlers at priority {priority}"
                )
                    
                # 并发调用所有处理器
                tasks = []
                for subscription in subscriptions:
                    tasks.append(asyncio.create_task(subscription.handle(event)))
                    
                # 等待所有处理器完成
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 检查结果
                for result in results:
                    if isinstance(result, Exception):
                        self._error_count += 1
                        error_msg = str(result)
                        logger.error(f"Error dispatching event {event}: {error_msg}")
                        event_tracer.log(
                            TraceLevel.ERROR,
                            event,
                            "Handler error",
                            error=error_msg
                        )
                    elif result:
                        handled = True
                        self._handler_count += 1
                        
            self._dispatch_count += 1
            self._total_dispatch_time += time.time() - start_time
            
            # 结束追踪
            event_tracer.end_trace(event)
            return handled
            
        except Exception as e:
            self._error_count += 1
            error_msg = str(e)
            logger.error(f"Error dispatching event {event}: {error_msg}")
            event_tracer.end_trace(event, error=error_msg)
            return False
            
    def subscribe(self, handler: EventHandler,
                 filters: Optional[List[IEventFilter]] = None,
                 priority: EventPriority = EventPriority.NORMAL) -> EventSubscription:
        """Subscribe to events."""
        # 创建订阅
        subscription = EventSubscription(
            handler=handler,
            filters=filters or [],
            priority=priority
        )
        
        # 添加到对应优先级的集合
        if priority not in self._subscriptions:
            self._subscriptions[priority] = set()
        self._subscriptions[priority].add(subscription)
        
        return subscription
        
    def unsubscribe(self, subscription: EventSubscription) -> None:
        """Unsubscribe from events."""
        if subscription.priority in self._subscriptions:
            self._subscriptions[subscription.priority].discard(subscription)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher statistics."""
        return {
            "dispatch_count": self._dispatch_count,
            "handler_count": self._handler_count,
            "error_count": self._error_count,
            "avg_dispatch_time": (
                self._total_dispatch_time / self._dispatch_count
                if self._dispatch_count > 0 else 0
            ),
            "subscription_count": sum(
                len(subs) for subs in self._subscriptions.values()
            )
        }
        
    def reset_stats(self) -> None:
        """Reset dispatcher statistics."""
        self._dispatch_count = 0
        self._handler_count = 0
        self._error_count = 0
        self._total_dispatch_time = 0

# 创建全局事件分发器实例
event_dispatcher = EventDispatcher()
