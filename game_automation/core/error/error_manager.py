from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json
import logging
import traceback
from datetime import datetime
from enum import Enum, auto
import asyncio

class ErrorCategory(Enum):
    """错误类别"""
    SYSTEM = auto()      # 系统错误
    NETWORK = auto()     # 网络错误
    DEVICE = auto()      # 设备错误
    GAME = auto()        # 游戏错误
    TASK = auto()        # 任务错误
    SCENE = auto()       # 场景错误
    USER = auto()        # 用户错误
    UNKNOWN = auto()     # 未知错误

class ErrorSeverity(Enum):
    """错误严重程度"""
    INFO = 0        # 信息
    WARNING = 1     # 警告
    ERROR = 2       # 错误
    CRITICAL = 3    # 严重
    
    def __lt__(self, other):
        if not isinstance(other, ErrorSeverity):
            return NotImplemented
        return self.value < other.value
    
    def __le__(self, other):
        if not isinstance(other, ErrorSeverity):
            return NotImplemented
        return self.value <= other.value
    
    def __gt__(self, other):
        if not isinstance(other, ErrorSeverity):
            return NotImplemented
        return self.value > other.value
    
    def __ge__(self, other):
        if not isinstance(other, ErrorSeverity):
            return NotImplemented
        return self.value >= other.value

class ErrorState(Enum):
    """错误状态"""
    NEW = auto()        # 新错误
    PROCESSING = auto() # 处理中
    RESOLVED = auto()   # 已解决
    IGNORED = auto()    # 已忽略

class GameAutomationError(Exception):
    """游戏自动化错误"""
    def __init__(
        self,
        message: str,
        error_code: str = None,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        original_error: Exception = None,
        context: Dict = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.original_error = original_error
        self.context = context or {}
        self.timestamp = datetime.now()
        self.state = ErrorState.NEW
        self.stack_trace = traceback.format_exc()

class ErrorHandler:
    """错误处理器"""
    def __init__(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        handler: Callable,
        priority: int = 0
    ):
        self.category = category
        self.severity = severity
        self.handler = handler
        self.priority = priority

class ErrorManager:
    """错误管理器"""
    
    def __init__(self):
        self._initialized = False
        self._config = None
        self._handlers: Dict[ErrorCategory, List[ErrorHandler]] = {}
        self._errors: List[GameAutomationError] = []
        self._max_errors = 1000  # 最大错误数
        self._event_loop = None

    async def initialize(self):
        """初始化错误管理器"""
        if self._initialized:
            return

        try:
            self._event_loop = asyncio.get_running_loop()
            await self._register_default_handlers()
            self._initialized = True
        except Exception as e:
            raise GameAutomationError(
                message="初始化失败",
                error_code="ERR_INIT_001",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                original_error=e,
                context={"location": "ErrorManager.initialize"}
            )

    async def _register_default_handlers(self):
        """注册默认的错误处理器"""
        await self.register_handler(
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            handler=self._handle_system_error,
            priority=100
        )
        
        await self.register_handler(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            handler=self._handle_network_error,
            priority=90
        )
        
        await self.register_handler(
            category=ErrorCategory.DEVICE,
            severity=ErrorSeverity.ERROR,
            handler=self._handle_device_error,
            priority=80
        )
        
        await self.register_handler(
            category=ErrorCategory.GAME,
            severity=ErrorSeverity.ERROR,
            handler=self._handle_game_error,
            priority=70
        )

    async def register_handler(
            self,
            category: ErrorCategory,
            severity: ErrorSeverity,
            handler: callable,
            priority: int = 0
        ):
        """注册错误处理器"""
        if not asyncio.iscoroutinefunction(handler) and not isinstance(handler, asyncio.coroutine):
            # 如果不是协程函数，将其包装为协程函数
            original_handler = handler
            handler = lambda *args, **kwargs: asyncio.create_task(
                asyncio.coroutine(original_handler)(*args, **kwargs)
            )
        
        error_handler = ErrorHandler(category, severity, handler, priority)
        
        if category not in self._handlers:
            self._handlers[category] = []
            
        # 按优先级插入处理器
        handlers = self._handlers[category]
        insert_idx = 0
        for idx, h in enumerate(handlers):
            if h.priority < priority:
                insert_idx = idx
                break
            insert_idx = idx + 1
        
        handlers.insert(insert_idx, error_handler)

    async def handle_error(self, error: GameAutomationError):
        """处理错误"""
        if not self._initialized:
            await self.initialize()
            
        self._errors.append(error)
        if len(self._errors) > self._max_errors:
            self._errors.pop(0)
            
        handlers = self._handlers.get(error.category, [])
        for handler in handlers:
            if handler.severity <= error.severity:
                try:
                    result = await handler.handler(error)
                    if result:  # 如果处理器返回True，表示错误已解决
                        error.state = ErrorState.RESOLVED
                        return
                    error.state = ErrorState.PROCESSING
                except Exception as e:
                    # 处理器出错时记录但不中断
                    error.context["handler_error"] = str(e)
                    error.context["handler_traceback"] = traceback.format_exc()
                    error.state = ErrorState.ERROR

    async def get_errors(
            self,
            category: ErrorCategory = None,
            severity: ErrorSeverity = None,
            state: ErrorState = None,
            start_time: datetime = None,
            end_time: datetime = None
        ) -> List[GameAutomationError]:
        """获取错误列表"""
        filtered_errors = self._errors.copy()
        
        if category:
            filtered_errors = [e for e in filtered_errors if e.category == category]
        if severity:
            filtered_errors = [e for e in filtered_errors if e.severity == severity]
        if state:
            filtered_errors = [e for e in filtered_errors if e.state == state]
        if start_time:
            filtered_errors = [e for e in filtered_errors if e.timestamp >= start_time]
        if end_time:
            filtered_errors = [e for e in filtered_errors if e.timestamp <= end_time]
            
        return filtered_errors

    async def clear_errors(
            self,
            category: ErrorCategory = None,
            severity: ErrorSeverity = None,
            state: ErrorState = None
        ):
        """清理错误"""
        if not any([category, severity, state]):
            self._errors.clear()
            return
            
        self._errors = [
            e for e in self._errors
            if (category and e.category != category) or
               (severity and e.severity != severity) or
               (state and e.state != state)
        ]

    async def _handle_system_error(self, error: GameAutomationError):
        """处理系统错误"""
        # 记录错误
        logging.critical(
            f"系统错误: {error.message}\n"
            f"错误代码: {error.error_code}\n"
            f"堆栈跟踪:\n{error.stack_trace}"
        )
        
        # 尝试恢复
        if error.severity == ErrorSeverity.CRITICAL:
            # 触发紧急停止
            await self._emergency_stop()
            
    async def _handle_network_error(self, error: GameAutomationError):
        """处理网络错误"""
        # 记录错误
        logging.error(
            f"网络错误: {error.message}\n"
            f"错误代码: {error.error_code}"
        )
        
        # 尝试重连
        retry_count = 3
        retry_interval = 1.0
        
        for i in range(retry_count):
            try:
                # TODO: 实现网络重连
                await asyncio.sleep(retry_interval)
                return
                
            except Exception as e:
                if i == retry_count - 1:
                    error.state = ErrorState.ERROR
                    raise GameAutomationError(
                        message="网络重连失败",
                        error_code="ERR003",
                        category=ErrorCategory.NETWORK,
                        severity=ErrorSeverity.CRITICAL,
                        original_error=e
                    )
                    
    async def _handle_device_error(self, error: GameAutomationError):
        """处理设备错误"""
        # 记录错误
        logging.error(
            f"设备错误: {error.message}\n"
            f"错误代码: {error.error_code}"
        )
        
        # 尝试重启设备
        try:
            # TODO: 实现设备重启
            await asyncio.sleep(1.0)
            
        except Exception as e:
            error.state = ErrorState.ERROR
            raise GameAutomationError(
                message="设备重启失败",
                error_code="ERR004",
                category=ErrorCategory.DEVICE,
                severity=ErrorSeverity.CRITICAL,
                original_error=e
            )
            
    async def _handle_game_error(self, error: GameAutomationError):
        """处理游戏错误"""
        # 记录错误
        logging.error(
            f"游戏错误: {error.message}\n"
            f"错误代码: {error.error_code}"
        )
        
        # 尝试重启游戏
        try:
            # TODO: 实现游戏重启
            await asyncio.sleep(1.0)
            
        except Exception as e:
            error.state = ErrorState.ERROR
            raise GameAutomationError(
                message="游戏重启失败",
                error_code="ERR005",
                category=ErrorCategory.GAME,
                severity=ErrorSeverity.CRITICAL,
                original_error=e
            )
            
    async def _emergency_stop(self):
        """紧急停止"""
        try:
            # 停止所有任务
            # TODO: 实现任务停止
            
            # 保存状态
            await self._save_state()
            
            # 记录日志
            logging.critical("系统紧急停止")
            
        except Exception as e:
            logging.critical(f"紧急停止失败: {str(e)}")
            
    async def _save_state(self):
        """保存状态"""
        try:
            # 创建状态目录
            state_dir = Path(self._config["paths"]["logs"]) / "error_state"
            state_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state_path = state_dir / f"error_state_{timestamp}.json"
            
            # 保存状态
            state = {
                'errors': [
                    {
                        'message': error.message,
                        'error_code': error.error_code,
                        'category': error.category.name,
                        'severity': error.severity.name,
                        'state': error.state.name,
                        'context': error.context,
                        'timestamp': error.timestamp.isoformat(),
                        'stack_trace': error.stack_trace
                    }
                    for error in self._errors
                ],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"保存状态失败: {str(e)}")
