"""错误处理基类"""

import enum
from typing import Any, Dict, Optional


class ErrorLevel(enum.Enum):
    """错误级别"""
    DEBUG = 0  # 调试级别
    INFO = 1  # 信息级别
    WARNING = 2  # 警告级别
    ERROR = 3  # 错误级别
    CRITICAL = 4  # 严重错误级别


class ErrorType(enum.Enum):
    """错误类型"""
    UNKNOWN = 0  # 未知错误
    SYSTEM = 1  # 系统错误
    RESOURCE = 2  # 资源错误
    NETWORK = 3  # 网络错误
    DEVICE = 4  # 设备错误
    CONFIG = 5  # 配置错误
    RUNTIME = 6  # 运行时错误
    USER = 7  # 用户错误


class GameError(Exception):
    """游戏错误基类
    
    所有错误类型都应该继承自这个类。
    """
    
    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.UNKNOWN,
        error_level: ErrorLevel = ErrorLevel.ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        """初始化错误
        
        Args:
            message: 错误消息
            error_type: 错误类型
            error_level: 错误级别
            details: 错误详情
        """
        super().__init__(message)
        self._message = message
        self._error_type = error_type
        self._error_level = error_level
        self._details = details or {}
        
    @property
    def message(self) -> str:
        """获取错误消息"""
        return self._message
        
    @property
    def error_type(self) -> ErrorType:
        """获取错误类型"""
        return self._error_type
        
    @property
    def error_level(self) -> ErrorLevel:
        """获取错误级别"""
        return self._error_level
        
    @property
    def details(self) -> Dict[str, Any]:
        """获取错误详情"""
        return self._details
        
    def __str__(self) -> str:
        """获取错误字符串表示"""
        return (
            f"{self.error_type.name} Error ({self.error_level.name}): "
            f"{self.message}"
        )
        
    def __repr__(self) -> str:
        """获取错误字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_type={self.error_type}, "
            f"error_level={self.error_level}, "
            f"details={self.details})"
        )
