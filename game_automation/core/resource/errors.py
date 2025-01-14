"""资源错误类型"""

from typing import Any, Dict, Optional

from ...core.error.error_base import GameError, ErrorType, ErrorLevel


class ResourceError(GameError):
    """资源错误基类"""
    
    def __init__(
        self,
        message: str,
        error_level: ErrorLevel = ErrorLevel.ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        """初始化错误
        
        Args:
            message: 错误消息
            error_level: 错误级别
            details: 错误详情
        """
        super().__init__(
            message,
            error_type=ErrorType.RESOURCE,
            error_level=error_level,
            details=details
        )


class ResourceLoadError(ResourceError):
    """资源加载错误"""
    
    def __init__(
        self,
        resource_key: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """初始化错误
        
        Args:
            resource_key: 资源键
            reason: 失败原因
            details: 错误详情
        """
        super().__init__(
            f"Failed to load resource {resource_key}: {reason}",
            error_level=ErrorLevel.ERROR,
            details=details
        )


class ResourceUnloadError(ResourceError):
    """资源卸载错误"""
    
    def __init__(
        self,
        resource_key: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """初始化错误
        
        Args:
            resource_key: 资源键
            reason: 失败原因
            details: 错误详情
        """
        super().__init__(
            f"Failed to unload resource {resource_key}: {reason}",
            error_level=ErrorLevel.ERROR,
            details=details
        )


class ResourceVerifyError(ResourceError):
    """资源验证错误"""
    
    def __init__(
        self,
        resource_key: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """初始化错误
        
        Args:
            resource_key: 资源键
            reason: 失败原因
            details: 错误详情
        """
        super().__init__(
            f"Failed to verify resource {resource_key}: {reason}",
            error_level=ErrorLevel.ERROR,
            details=details
        )


class ResourceNotFoundError(ResourceError):
    """资源未找到错误"""
    
    def __init__(
        self,
        resource_key: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """初始化错误
        
        Args:
            resource_key: 资源键
            details: 错误详情
        """
        super().__init__(
            f"Resource not found: {resource_key}",
            error_level=ErrorLevel.ERROR,
            details=details
        )


class ResourceAlreadyExistsError(ResourceError):
    """资源已存在错误"""
    
    def __init__(
        self,
        resource_key: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """初始化错误
        
        Args:
            resource_key: 资源键
            details: 错误详情
        """
        super().__init__(
            f"Resource already exists: {resource_key}",
            error_level=ErrorLevel.ERROR,
            details=details
        )


class ResourceInvalidError(ResourceError):
    """资源无效错误"""
    
    def __init__(
        self,
        resource_key: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """初始化错误
        
        Args:
            resource_key: 资源键
            reason: 无效原因
            details: 错误详情
        """
        super().__init__(
            f"Invalid resource {resource_key}: {reason}",
            error_level=ErrorLevel.ERROR,
            details=details
        )


class ResourceBusyError(ResourceError):
    """资源忙错误"""
    
    def __init__(
        self,
        resource_key: str,
        operation: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """初始化错误
        
        Args:
            resource_key: 资源键
            operation: 操作名称
            details: 错误详情
        """
        super().__init__(
            f"Resource {resource_key} is busy: {operation} in progress",
            error_level=ErrorLevel.WARNING,
            details=details
        )


class ResourceTimeoutError(ResourceError):
    """资源超时错误"""
    
    def __init__(
        self,
        resource_key: str,
        operation: str,
        timeout: float,
        details: Optional[Dict[str, Any]] = None
    ):
        """初始化错误
        
        Args:
            resource_key: 资源键
            operation: 操作名称
            timeout: 超时时间（秒）
            details: 错误详情
        """
        super().__init__(
            f"Resource {resource_key} operation {operation} timed out after {timeout}s",
            error_level=ErrorLevel.ERROR,
            details=details
        )


class ResourceCorruptedError(ResourceError):
    """资源损坏错误"""
    
    def __init__(
        self,
        resource_key: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """初始化错误
        
        Args:
            resource_key: 资源键
            reason: 损坏原因
            details: 错误详情
        """
        super().__init__(
            f"Resource {resource_key} is corrupted: {reason}",
            error_level=ErrorLevel.ERROR,
            details=details
        )


class ResourceLimitExceededError(ResourceError):
    """资源限制超出错误"""
    
    def __init__(
        self,
        resource_type: str,
        current: int,
        limit: int,
        details: Optional[Dict[str, Any]] = None
    ):
        """初始化错误
        
        Args:
            resource_type: 资源类型
            current: 当前数量
            limit: 限制数量
            details: 错误详情
        """
        super().__init__(
            f"Resource limit exceeded for {resource_type}: {current}/{limit}",
            error_level=ErrorLevel.WARNING,
            details=details
        )
