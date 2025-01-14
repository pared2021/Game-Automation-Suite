"""
Async IO Framework for Game Automation Suite.

This package provides the core async IO functionality including:
- Event loop management
- Event dispatching system
- Async utilities
"""

from .event_loop import EventLoop
from .event_dispatcher import EventDispatcher
from .async_utils import (
    with_retry,
    with_timeout,
    AsyncResourceManager,
    AsyncPerformanceMonitor
)

__all__ = [
    'EventLoop',
    'EventDispatcher',
    'with_retry',
    'with_timeout',
    'AsyncResourceManager',
    'AsyncPerformanceMonitor'
]
