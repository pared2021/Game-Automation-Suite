"""资源错误恢复机制"""

import logging
import asyncio
from typing import Dict, List, Optional, Type, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from .errors import (
    ResourceError,
    ResourceLoadError,
    ResourceUnloadError,
    ResourceVerifyError,
    ResourceNotFoundError,
    ResourceAlreadyExistsError,
    ResourceInvalidError,
    ResourceBusyError,
    ResourceTimeoutError,
    ResourceCorruptedError,
    ResourceLimitExceededError
)

logger = logging.getLogger(__name__)


@dataclass
class RecoveryAttempt:
    """恢复尝试记录"""
    error: ResourceError
    timestamp: datetime
    success: bool
    recovery_time: float


class ResourceRecoveryManager:
    """资源错误恢复管理器
    
    特性：
    - 错误恢复策略
    - 重试机制
    - 恢复记录
    - 错误通知
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        recovery_timeout: float = 30.0,
        notify_callback: Optional[Callable[[ResourceError], None]] = None
    ):
        """初始化恢复管理器
        
        Args:
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            retry_backoff: 重试延迟增长系数
            recovery_timeout: 恢复超时时间（秒）
            notify_callback: 错误通知回调函数
        """
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._retry_backoff = retry_backoff
        self._recovery_timeout = recovery_timeout
        self._notify_callback = notify_callback
        self._recovery_history: Dict[str, List[RecoveryAttempt]] = {}
        
    async def handle_error(
        self,
        error: ResourceError,
        recovery_func: Callable[[], Any]
    ) -> bool:
        """处理资源错误
        
        Args:
            error: 资源错误
            recovery_func: 恢复函数
            
        Returns:
            是否恢复成功
        """
        # 通知错误
        if self._notify_callback:
            try:
                self._notify_callback(error)
            except Exception as e:
                logger.error(f"Failed to notify error: {e}")
                
        # 获取资源键
        resource_key = self._get_resource_key(error)
        
        # 检查是否可以恢复
        if not self._can_recover(error):
            logger.warning(f"Cannot recover from error: {error}")
            self._record_attempt(resource_key, error, False, 0)
            return False
            
        # 尝试恢复
        retries = 0
        delay = self._retry_delay
        start_time = datetime.now()
        
        while retries < self._max_retries:
            try:
                # 等待延迟
                await asyncio.sleep(delay)
                
                # 执行恢复
                logger.info(f"Attempting to recover resource {resource_key} (attempt {retries + 1}/{self._max_retries})")
                await recovery_func()
                
                # 记录成功
                recovery_time = (datetime.now() - start_time).total_seconds()
                self._record_attempt(resource_key, error, True, recovery_time)
                return True
                
            except Exception as e:
                logger.error(f"Recovery attempt {retries + 1} failed: {e}")
                retries += 1
                delay *= self._retry_backoff
                
                # 检查是否超时
                if (datetime.now() - start_time).total_seconds() > self._recovery_timeout:
                    logger.error(f"Recovery timeout for resource {resource_key}")
                    break
                    
        # 记录失败
        recovery_time = (datetime.now() - start_time).total_seconds()
        self._record_attempt(resource_key, error, False, recovery_time)
        return False
        
    def get_recovery_history(
        self,
        resource_key: Optional[str] = None,
        error_type: Optional[Type[ResourceError]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, List[RecoveryAttempt]]:
        """获取恢复历史
        
        Args:
            resource_key: 资源键
            error_type: 错误类型
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            恢复历史记录
        """
        history = {}
        
        for key, attempts in self._recovery_history.items():
            # 过滤资源键
            if resource_key and key != resource_key:
                continue
                
            filtered_attempts = []
            for attempt in attempts:
                # 过滤错误类型
                if error_type and not isinstance(attempt.error, error_type):
                    continue
                    
                # 过滤时间范围
                if start_time and attempt.timestamp < start_time:
                    continue
                if end_time and attempt.timestamp > end_time:
                    continue
                    
                filtered_attempts.append(attempt)
                
            if filtered_attempts:
                history[key] = filtered_attempts
                
        return history
        
    def _can_recover(self, error: ResourceError) -> bool:
        """检查错误是否可以恢复
        
        Args:
            error: 资源错误
            
        Returns:
            是否可以恢复
        """
        # 不可恢复的错误类型
        if isinstance(error, (ResourceNotFoundError, ResourceAlreadyExistsError)):
            return False
            
        # 检查错误级别
        if error.error_level.value >= 3:  # CRITICAL
            return False
            
        return True
        
    def _get_resource_key(self, error: ResourceError) -> str:
        """从错误中获取资源键
        
        Args:
            error: 资源错误
            
        Returns:
            资源键
        """
        details = error.details or {}
        if "resource_key" in details:
            return str(details["resource_key"])
            
        # 从错误消息中提取
        message = str(error)
        for prefix in ["Resource ", "Failed to "]:
            if prefix in message:
                parts = message[message.index(prefix):].split(":")
                if len(parts) > 1:
                    return parts[0].replace(prefix, "").strip()
                    
        return "unknown"
        
    def _record_attempt(
        self,
        resource_key: str,
        error: ResourceError,
        success: bool,
        recovery_time: float
    ) -> None:
        """记录恢复尝试
        
        Args:
            resource_key: 资源键
            error: 资源错误
            success: 是否成功
            recovery_time: 恢复时间
        """
        attempt = RecoveryAttempt(
            error=error,
            timestamp=datetime.now(),
            success=success,
            recovery_time=recovery_time
        )
        
        if resource_key not in self._recovery_history:
            self._recovery_history[resource_key] = []
            
        self._recovery_history[resource_key].append(attempt)
        
        # 清理过期记录
        cutoff = datetime.now() - timedelta(days=7)
        self._recovery_history[resource_key] = [
            a for a in self._recovery_history[resource_key]
            if a.timestamp > cutoff
        ]
