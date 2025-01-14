"""Async Utilities for Game Automation Suite.

This module provides utility functions and classes for async operations:
- Task management
- Timeouts and retries
- Resource management
- Performance monitoring
"""

import asyncio
import logging
import time
from typing import TypeVar, Callable, Optional, Any
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')

def with_retry(
    retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
    backoff: float = 2.0
):
    """Decorator for retrying async functions.
    
    Args:
        retries: Maximum number of retries
        delay: Initial delay between retries in seconds
        exceptions: Tuple of exceptions to catch
        backoff: Multiplier for delay between retries
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == retries:
                        break
                        
                    logger.warning(
                        f"Retry {attempt + 1}/{retries} for {func.__name__}: {e}"
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
                    
            raise last_exception
            
        return wrapper
    return decorator

def with_timeout(timeout: float):
    """Decorator for adding timeout to async functions.
    
    Args:
        timeout: Timeout in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout
            )
        return wrapper
    return decorator

class AsyncResourceManager:
    """Manager for async resources.
    
    Features:
    - Resource pooling
    - Automatic cleanup
    - Usage tracking
    """
    
    def __init__(self):
        """Initialize the resource manager."""
        self._resources = {}
        self._usage = {}
        
    async def get(self, key: str) -> Any:
        """Get a resource.
        
        Args:
            key: Resource key
            
        Returns:
            Resource value
        """
        if key not in self._resources:
            raise KeyError(f"Resource {key} not found")
            
        self._usage[key] = time.time()
        return self._resources[key]
        
    async def put(self, key: str, value: Any):
        """Store a resource.
        
        Args:
            key: Resource key
            value: Resource value
        """
        self._resources[key] = value
        self._usage[key] = time.time()
        
    async def cleanup(self, max_age: float = 3600.0):
        """Clean up old resources.
        
        Args:
            max_age: Maximum resource age in seconds
        """
        current_time = time.time()
        to_remove = []
        
        for key, last_use in self._usage.items():
            if current_time - last_use > max_age:
                to_remove.append(key)
                
        for key in to_remove:
            del self._resources[key]
            del self._usage[key]
            
class AsyncPerformanceMonitor:
    """Monitor for async operation performance.
    
    Features:
    - Operation timing
    - Resource usage tracking
    - Performance logging
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self._timings = {}
        self._counts = {}
        
    def start_operation(self, name: str):
        """Start timing an operation.
        
        Args:
            name: Operation name
        """
        self._timings[name] = time.time()
        
    def end_operation(self, name: str):
        """End timing an operation.
        
        Args:
            name: Operation name
            
        Returns:
            float: Operation duration in seconds
        """
        if name not in self._timings:
            return 0.0
            
        duration = time.time() - self._timings[name]
        del self._timings[name]
        
        self._counts[name] = self._counts.get(name, 0) + 1
        
        return duration
        
    def get_stats(self) -> dict:
        """Get performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        return {
            'counts': self._counts.copy(),
            'active_operations': list(self._timings.keys())
        }
