"""资源管理系统错误定义"""

from typing import Optional
from ...core.error.error_base import GameError, ErrorType, ErrorLevel


class ResourceError(GameError):
    """资源管理错误基类"""
    
    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.RESOURCE_ERROR,
        error_level: ErrorLevel = ErrorLevel.ERROR,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, error_type, error_level, cause)


class ResourceNotFoundError(ResourceError):
    """资源未找到错误"""
    
    def __init__(
        self,
        resource_key: str,
        message: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        message = message or f"Resource not found: {resource_key}"
        super().__init__(
            message,
            ErrorType.RESOURCE_NOT_FOUND,
            ErrorLevel.ERROR,
            cause
        )
        self.resource_key = resource_key


class ResourceLoadError(ResourceError):
    """资源加载错误"""
    
    def __init__(
        self,
        resource_key: str,
        message: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        message = message or f"Failed to load resource: {resource_key}"
        super().__init__(
            message,
            ErrorType.RESOURCE_LOAD_ERROR,
            ErrorLevel.ERROR,
            cause
        )
        self.resource_key = resource_key


class ResourceStateError(ResourceError):
    """资源状态错误"""
    
    def __init__(
        self,
        resource_key: str,
        current_state: str,
        expected_state: str,
        message: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        message = message or (
            f"Invalid resource state for {resource_key}: "
            f"expected {expected_state}, got {current_state}"
        )
        super().__init__(
            message,
            ErrorType.RESOURCE_STATE_ERROR,
            ErrorLevel.ERROR,
            cause
        )
        self.resource_key = resource_key
        self.current_state = current_state
        self.expected_state = expected_state


class CacheError(ResourceError):
    """缓存错误"""
    
    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            ErrorType.CACHE_ERROR,
            ErrorLevel.ERROR,
            cause
        )


class MonitorError(ResourceError):
    """监控错误"""
    
    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            ErrorType.MONITOR_ERROR,
            ErrorLevel.ERROR,
            cause
        )
