"""资源错误通知系统"""

import json
import logging
import asyncio
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

from .errors import ResourceError

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """通知级别"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


@dataclass
class Notification:
    """通知消息"""
    level: NotificationLevel
    title: str
    message: str
    timestamp: datetime
    error: Optional[ResourceError] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            字典形式的通知
        """
        data = asdict(self)
        data["level"] = self.level.name
        data["timestamp"] = self.timestamp.isoformat()
        if self.error:
            data["error"] = {
                "type": self.error.__class__.__name__,
                "message": str(self.error),
                "details": self.error.details
            }
        return data


class NotificationHandler:
    """通知处理器基类"""
    
    async def handle(self, notification: Notification) -> None:
        """处理通知
        
        Args:
            notification: 通知消息
        """
        raise NotImplementedError


class LoggingHandler(NotificationHandler):
    """日志通知处理器"""
    
    def __init__(self, logger_name: Optional[str] = None):
        """初始化处理器
        
        Args:
            logger_name: 日志器名称
        """
        self._logger = logging.getLogger(logger_name or __name__)
        
    async def handle(self, notification: Notification) -> None:
        """处理通知
        
        Args:
            notification: 通知消息
        """
        level = {
            NotificationLevel.DEBUG: logging.DEBUG,
            NotificationLevel.INFO: logging.INFO,
            NotificationLevel.WARNING: logging.WARNING,
            NotificationLevel.ERROR: logging.ERROR,
            NotificationLevel.CRITICAL: logging.CRITICAL
        }[notification.level]
        
        self._logger.log(
            level,
            f"{notification.title}: {notification.message}",
            extra={"notification": notification.to_dict()}
        )


class CallbackHandler(NotificationHandler):
    """回调通知处理器"""
    
    def __init__(self, callback: Callable[[Notification], None]):
        """初始化处理器
        
        Args:
            callback: 回调函数
        """
        self._callback = callback
        
    async def handle(self, notification: Notification) -> None:
        """处理通知
        
        Args:
            notification: 通知消息
        """
        try:
            self._callback(notification)
        except Exception as e:
            logger.error(f"Failed to execute notification callback: {e}")


class FileHandler(NotificationHandler):
    """文件通知处理器"""
    
    def __init__(self, filepath: str):
        """初始化处理器
        
        Args:
            filepath: 文件路径
        """
        self._filepath = filepath
        
    async def handle(self, notification: Notification) -> None:
        """处理通知
        
        Args:
            notification: 通知消息
        """
        try:
            with open(self._filepath, "a", encoding="utf-8") as f:
                json.dump(notification.to_dict(), f)
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write notification to file: {e}")


class NotificationManager:
    """通知管理器
    
    特性：
    - 多级别通知
    - 多处理器支持
    - 通知过滤
    - 通知限流
    """
    
    def __init__(
        self,
        min_level: NotificationLevel = NotificationLevel.INFO,
        rate_limit: Optional[float] = None
    ):
        """初始化管理器
        
        Args:
            min_level: 最小通知级别
            rate_limit: 限流时间间隔（秒）
        """
        self._min_level = min_level
        self._rate_limit = rate_limit
        self._handlers: List[NotificationHandler] = []
        self._last_notification: Dict[str, datetime] = {}
        self._notification_history: List[Notification] = []
        self._max_history = 1000
        
    def add_handler(self, handler: NotificationHandler) -> None:
        """添加处理器
        
        Args:
            handler: 通知处理器
        """
        self._handlers.append(handler)
        
    def remove_handler(self, handler: NotificationHandler) -> None:
        """移除处理器
        
        Args:
            handler: 通知处理器
        """
        if handler in self._handlers:
            self._handlers.remove(handler)
            
    async def notify(
        self,
        level: NotificationLevel,
        title: str,
        message: str,
        error: Optional[ResourceError] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """发送通知
        
        Args:
            level: 通知级别
            title: 通知标题
            message: 通知消息
            error: 资源错误
            details: 详细信息
        """
        # 检查通知级别
        if level.value < self._min_level.value:
            return
            
        # 检查限流
        if self._rate_limit is not None:
            key = f"{level.name}:{title}"
            last_time = self._last_notification.get(key)
            if last_time:
                elapsed = (datetime.now() - last_time).total_seconds()
                if elapsed < self._rate_limit:
                    return
                    
        # 创建通知
        notification = Notification(
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            error=error,
            details=details
        )
        
        # 更新限流记录
        if self._rate_limit is not None:
            key = f"{level.name}:{title}"
            self._last_notification[key] = notification.timestamp
            
        # 添加到历史记录
        self._notification_history.append(notification)
        if len(self._notification_history) > self._max_history:
            self._notification_history.pop(0)
            
        # 分发通知
        for handler in self._handlers:
            try:
                await handler.handle(notification)
            except Exception as e:
                logger.error(f"Failed to handle notification: {e}")
                
    def get_history(
        self,
        min_level: Optional[NotificationLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        error_types: Optional[Set[str]] = None
    ) -> List[Notification]:
        """获取通知历史
        
        Args:
            min_level: 最小通知级别
            start_time: 开始时间
            end_time: 结束时间
            error_types: 错误类型集合
            
        Returns:
            通知历史记录
        """
        history = []
        
        for notification in self._notification_history:
            # 过滤级别
            if min_level and notification.level.value < min_level.value:
                continue
                
            # 过滤时间范围
            if start_time and notification.timestamp < start_time:
                continue
            if end_time and notification.timestamp > end_time:
                continue
                
            # 过滤错误类型
            if error_types and notification.error:
                error_type = notification.error.__class__.__name__
                if error_type not in error_types:
                    continue
                    
            history.append(notification)
            
        return history
