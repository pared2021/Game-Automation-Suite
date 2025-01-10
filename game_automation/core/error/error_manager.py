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
        
    async def initialize(self):
        """初始化管理器"""
        if not self._initialized:
            try:
                # 加载配置
                config_path = Path("config/config.json")
                if not config_path.exists():
                    raise GameAutomationError(
                        message="配置文件不存在",
                        error_code="ERR001",
                        category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.CRITICAL
                    )
                    
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                    
                # 注册默认处理器
                self._register_default_handlers()
                
                self._initialized = True
                
            except Exception as e:
                raise GameAutomationError(
                    message=f"初始化失败: {str(e)}",
                    error_code="ERR002",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.CRITICAL,
                    original_error=e
                )
                
    def _register_default_handlers(self):
        """注册默认处理器"""
        # 系统错误处理器
        self.register_handler(
            ErrorCategory.SYSTEM,
            ErrorSeverity.CRITICAL,
            self._handle_system_error
        )
        
        # 网络错误处理器
        self.register_handler(
            ErrorCategory.NETWORK,
            ErrorSeverity.ERROR,
            self._handle_network_error
        )
        
        # 设备错误处理器
        self.register_handler(
            ErrorCategory.DEVICE,
            ErrorSeverity.ERROR,
            self._handle_device_error
        )
        
        # 游戏错误处理器
        self.register_handler(
            ErrorCategory.GAME,
            ErrorSeverity.ERROR,
            self._handle_game_error
        )
        
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
            
    def register_handler(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        handler: Callable,
        priority: int = 0
    ):
        """注册错误处理器
        
        Args:
            category: 错误类别
            severity: 错误严重程度
            handler: 处理函数
            priority: 优先级
        """
        if category not in self._handlers:
            self._handlers[category] = []
            
        error_handler = ErrorHandler(category, severity, handler, priority)
        self._handlers[category].append(error_handler)
        
        # 按优先级排序
        self._handlers[category].sort(key=lambda x: x.priority, reverse=True)
        
    async def handle_error(self, error: GameAutomationError):
        """处理错误
        
        Args:
            error: 错误对象
        """
        if not self._initialized:
            raise GameAutomationError(
                message="错误管理器未初始化",
                error_code="ERR006",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL
            )
            
        try:
            # 添加到错误列表
            self._errors.append(error)
            
            # 限制错误数量
            if len(self._errors) > self._max_errors:
                self._errors.pop(0)
                
            # 查找处理器
            handlers = self._handlers.get(error.category, [])
            
            # 执行处理器
            for handler in handlers:
                if handler.severity <= error.severity:
                    try:
                        await handler.handler(error)
                        error.state = ErrorState.RESOLVED
                        return
                        
                    except Exception as e:
                        logging.error(f"错误处理失败: {str(e)}")
                        error.state = ErrorState.ERROR
                        continue
                        
            # 如果没有处理器处理
            if error.state == ErrorState.NEW:
                error.state = ErrorState.IGNORED
                
        except Exception as e:
            logging.error(f"错误处理失败: {str(e)}")
            
    async def get_errors(
        self,
        category: ErrorCategory = None,
        severity: ErrorSeverity = None,
        state: ErrorState = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[GameAutomationError]:
        """获取错误列表
        
        Args:
            category: 错误类别
            severity: 错误严重程度
            state: 错误状态
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[GameAutomationError]: 错误列表
        """
        if not self._initialized:
            raise GameAutomationError(
                message="错误管理器未初始化",
                error_code="ERR007",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL
            )
            
        errors = self._errors
        
        if category:
            errors = [
                error
                for error in errors
                if error.category == category
            ]
            
        if severity:
            errors = [
                error
                for error in errors
                if error.severity == severity
            ]
            
        if state:
            errors = [
                error
                for error in errors
                if error.state == state
            ]
            
        if start_time:
            errors = [
                error
                for error in errors
                if error.timestamp >= start_time
            ]
            
        if end_time:
            errors = [
                error
                for error in errors
                if error.timestamp <= end_time
            ]
            
        return errors
        
    async def clear_errors(
        self,
        category: ErrorCategory = None,
        severity: ErrorSeverity = None,
        state: ErrorState = None
    ):
        """清理错误
        
        Args:
            category: 错误类别
            severity: 错误严重程度
            state: 错误状态
        """
        if not self._initialized:
            raise GameAutomationError(
                message="错误管理器未初始化",
                error_code="ERR008",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL
            )
            
        if not any([category, severity, state]):
            self._errors.clear()
            return
            
        self._errors = [
            error
            for error in self._errors
            if (category and error.category != category) or
               (severity and error.severity != severity) or
               (state and error.state != state)
        ]
