"""Logging utilities."""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import asyncio
import json
from functools import wraps

from ..core.error.error_manager import GameAutomationError, ErrorCategory, ErrorSeverity

class LoggingError(GameAutomationError):
    """日志记录相关错误"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )

class AsyncQueueHandler(logging.Handler):
    """异步队列处理器，用于异步日志记录"""
    
    def __init__(self, queue: asyncio.Queue):
        """初始化异步队列处理器
        
        Args:
            queue: 异步队列
        """
        super().__init__()
        self.queue = queue
        
    def emit(self, record: logging.LogRecord):
        """发送日志记录到队列
        
        Args:
            record: 日志记录
        """
        try:
            # 将日志记录放入队列
            asyncio.create_task(self.queue.put(record))
        except Exception as e:
            self.handleError(record)

class ContextFilter(logging.Filter):
    """上下文过滤器，用于添加上下文信息到日志记录"""
    
    def __init__(self, context: Dict[str, Any], json_format: bool):
        """初始化上下文过滤器
        
        Args:
            context: 上下文信息
            json_format: 是否使用JSON格式输出
        """
        super().__init__()
        self.context = context
        self.json_format = json_format
        
    def filter(self, record: logging.LogRecord) -> bool:
        """过滤日志记录，添加上下文信息
        
        Args:
            record: 日志记录
            
        Returns:
            bool: 是否保留该记录
        """
        if self.context:
            if hasattr(record, 'context'):
                record.context.update(self.context)
            else:
                record.context = self.context.copy()
            if not self.json_format:
                record.msg = f"{record.msg} [context: {self.context}]"
        return True

class GameLogger:
    """游戏自动化日志记录器
    
    特性：
    1. 支持异步日志记录
    2. 自动创建日志目录
    3. 日志文件按日期轮转
    4. 支持上下文信息记录
    5. 支持多种输出格式（文本、JSON）
    """
    
    def __init__(self, name: str, log_dir: str = "logs",
                 max_bytes: int = 10*1024*1024, backup_count: int = 5,
                 log_level: str = "INFO", json_format: bool = False):
        """初始化日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志目录
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的备份文件数量
            log_level: 日志级别
            json_format: 是否使用JSON格式输出
        """
        self.name = name
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.log_level = getattr(logging, log_level.upper())
        self.json_format = json_format
        
        # 创建日志目录
        self.log_path = Path(log_dir)
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # 异步日志队列
        self.log_queue = asyncio.Queue()
        self._running = False
        self._process_task = None
        
        # 上下文信息
        self.context: Dict[str, Any] = {}
        
        # 设置处理器
        self._setup_handlers()
        
    def _setup_handlers(self):
        """设置日志处理器"""
        # 清除现有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(self._create_formatter())
        self.logger.addHandler(console_handler)
        
        # 创建文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_path / f"{self.name}.log",
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(self._create_formatter())
        self.logger.addHandler(file_handler)
        
        # 创建异步队列处理器
        queue_handler = AsyncQueueHandler(self.log_queue)
        queue_handler.setLevel(self.log_level)
        self.logger.addHandler(queue_handler)
        
        # 添加上下文过滤器
        context_filter = ContextFilter(self.context, self.json_format)
        self.logger.addFilter(context_filter)
        
    def _create_formatter(self) -> logging.Formatter:
        """创建日志格式化器
        
        Returns:
            logging.Formatter: 日志格式化器
        """
        if self.json_format:
            class JsonFormatter(logging.Formatter):
                def format(self, record):
                    data = {
                        'time': self.formatTime(record),
                        'level': record.levelname,
                        'name': record.name,
                        'message': record.msg
                    }
                    if hasattr(record, 'context'):
                        data['context'] = record.context
                    return json.dumps(data)
            return JsonFormatter()
        else:
            return logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            
    async def start(self):
        """启动日志记录器"""
        if self._running:
            return
            
        self._running = True
        self._process_task = asyncio.create_task(self._process_logs())
        
    async def stop(self):
        """停止日志记录器"""
        if not self._running:
            return
            
        self._running = False
        if self._process_task:
            try:
                await asyncio.wait_for(self._process_task, timeout=1.0)
            except asyncio.TimeoutError:
                self._process_task.cancel()
                try:
                    await self._process_task
                except asyncio.CancelledError:
                    pass
                    
    async def _process_logs(self):
        """处理日志队列中的日志记录"""
        while self._running:
            try:
                # 从队列中获取日志记录
                record = await asyncio.wait_for(self.log_queue.get(), timeout=0.1)
                
                # 处理日志记录
                for handler in self.logger.handlers:
                    if isinstance(handler, AsyncQueueHandler):
                        continue
                    if record.levelno >= handler.level:
                        handler.emit(record)
                        
                self.log_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing log record: {e}", file=sys.stderr)
                
    def set_context(self, **kwargs):
        """设置上下文信息
        
        Args:
            **kwargs: 上下文信息键值对
        """
        self.context.update(kwargs)
        
    def clear_context(self):
        """清除上下文信息"""
        self.context.clear()
        
    def with_context(self, **context):
        """使用上下文信息装饰器
        
        Args:
            **context: 上下文信息键值对
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                old_context = self.context.copy()
                try:
                    self.set_context(**context)
                    return func(*args, **kwargs)
                finally:
                    self.context.clear()
                    self.context.update(old_context)
            return wrapper
        return decorator
        
    def log(self, level: Union[int, str], message: str,
            context: Optional[Dict[str, Any]] = None):
        """记录日志
        
        Args:
            level: 日志级别
            message: 日志消息
            context: 上下文信息
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
            
        if context:
            old_context = self.context.copy()
            try:
                self.set_context(**context)
                self.logger.log(level, message)
            finally:
                self.context.clear()
                self.context.update(old_context)
        else:
            self.logger.log(level, message)
            
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """记录调试级别日志"""
        self.log(logging.DEBUG, message, context)
        
    def info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """记录信息级别日志"""
        self.log(logging.INFO, message, context)
        
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """记录警告级别日志"""
        self.log(logging.WARNING, message, context)
        
    def error(self, message: str, context: Optional[Dict[str, Any]] = None):
        """记录错误级别日志"""
        self.log(logging.ERROR, message, context)
        
    def critical(self, message: str, context: Optional[Dict[str, Any]] = None):
        """记录严重错误级别日志"""
        self.log(logging.CRITICAL, message, context)

# 全局日志记录器实例
_loggers: Dict[str, GameLogger] = {}

def get_logger(name: str, **kwargs) -> GameLogger:
    """获取日志记录器实例
    
    Args:
        name: 日志记录器名称
        **kwargs: 其他参数传递给GameLogger构造函数
        
    Returns:
        GameLogger: 日志记录器实例
    """
    if name not in _loggers:
        _loggers[name] = GameLogger(name, **kwargs)
    return _loggers[name]

def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    json_format: bool = False
):
    """设置日志记录
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的备份文件数量
        json_format: 是否使用JSON格式输出
    """
    # 创建根日志记录器
    root_logger = get_logger(
        "game_automation",
        log_dir=log_dir,
        log_level=log_level,
        max_bytes=max_bytes,
        backup_count=backup_count,
        json_format=json_format
    )
    
    # 启动日志记录器
    asyncio.create_task(root_logger.start())
