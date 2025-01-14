"""日志管理系统

提供统一的日志记录功能，支持：
1. 多级别日志记录（DEBUG, INFO, WARNING, ERROR, CRITICAL）
2. 日志格式化和过滤
3. 日志文件轮转
4. 异步日志记录
5. 上下文信息记录
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import asyncio
import json

from .error.error_manager import GameAutomationError, ErrorCategory, ErrorSeverity

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

class LogManager:
    """日志管理器
    
    特性：
    1. 支持异步日志记录
    2. 自动创建日志目录
    3. 日志文件按日期轮转
    4. 支持上下文信息记录
    5. 支持多种输出格式（文本、JSON）
    """
    
    def __init__(self, log_dir: str = "logs", max_bytes: int = 10*1024*1024,
                 backup_count: int = 5, log_level: str = "INFO",
                 json_format: bool = False):
        """初始化日志管理器
        
        Args:
            log_dir: 日志目录
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的备份文件数量
            log_level: 日志级别
            json_format: 是否使用JSON格式输出
        """
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.log_level = getattr(logging, log_level.upper())
        self.json_format = json_format
        
        # 创建日志目录
        self.log_path = Path(log_dir)
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志记录器
        self.logger = logging.getLogger("game_automation")
        self.logger.setLevel(self.log_level)
        
        # 异步日志队列
        self.log_queue = asyncio.Queue()
        self._running = False
        self._process_task = None
        
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
            self.log_path / "game_automation.log",
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
        
    def _create_formatter(self) -> logging.Formatter:
        """创建日志格式化器
        
        Returns:
            logging.Formatter: 日志格式化器
        """
        if self.json_format:
            return logging.Formatter(
                '{"time":"%(asctime)s", "level":"%(levelname)s", '
                '"name":"%(name)s", "message":"%(message)s"}'
            )
        else:
            return logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            
    async def start(self):
        """启动日志管理器"""
        if self._running:
            return
            
        self._running = True
        self._process_task = asyncio.create_task(self._process_logs())
        
    async def stop(self):
        """停止日志管理器"""
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
            if self.json_format:
                message = json.dumps({"message": message, "context": context})
            else:
                context_str = ", ".join(f"{k}={v}" for k, v in context.items())
                message = f"{message} [{context_str}]"
                
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
