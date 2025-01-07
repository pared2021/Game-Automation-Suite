from typing import Any, Dict, Optional, Type
import asyncio
from datetime import datetime
from pathlib import Path

from ..base.engine_base import EngineBase
from ..config.config_manager import ConfigManager
from ..logging.log_manager import LogManager
from ..events.event_manager import EventManager, Event
from ..error.error_manager import ErrorManager, GameAutomationError, ErrorCategory, ErrorSeverity
from ..services.service_registry import ServiceRegistry
from ..interfaces.service_interface import IGameService

class GameEngine(EngineBase):
    """Main game automation engine that coordinates all components."""
    
    def __init__(self):
        super().__init__()
        self._start_time: Optional[datetime] = None
        self._config_manager: Optional[ConfigManager] = None
        self._log_manager: Optional[LogManager] = None
        self._event_manager: Optional[EventManager] = None
        self._error_manager: Optional[ErrorManager] = None
        self._service_registry: Optional[ServiceRegistry] = None
    
    async def initialize(self) -> None:
        """Initialize the game engine and all its components."""
        try:
            self._start_time = datetime.now()
            
            # Initialize core managers
            await self._initialize_core_managers()
            
            # Load configuration
            await self._load_configuration()
            
            # Initialize service registry
            await self._initialize_service_registry()
            
            self.log_info("Game engine initialized successfully")
            
        except Exception as e:
            self.log_error("Failed to initialize game engine", e)
            await self.cleanup()
            raise GameAutomationError(
                message="Engine initialization failed",
                error_code="ENG001",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                original_error=e
            )
    
    async def _initialize_core_managers(self) -> None:
        """Initialize core management components."""
        # Create and initialize config manager
        self._config_manager = ConfigManager()
        await self._config_manager.start()
        
        # Create and initialize log manager
        self._log_manager = LogManager()
        await self._log_manager.start()
        
        # Create and initialize event manager
        self._event_manager = EventManager()
        await self._event_manager.start()
        
        # Create and initialize error manager
        self._error_manager = ErrorManager()
        await self._error_manager.start()
        
        # Create service registry
        self._service_registry = ServiceRegistry()
    
    async def _load_configuration(self) -> None:
        """Load engine configuration."""
        config_dir = Path("config")
        if not config_dir.exists():
            config_dir.mkdir(parents=True)
        
        # Load main configuration
        main_config = await self._config_manager.load_config_file(
            config_dir / "config.yaml"
        )
        
        if main_config:
            self.configure(main_config)
            self.log_info("Main configuration loaded")
        else:
            self.log_warning("No main configuration found, using defaults")
    
    async def _initialize_service_registry(self) -> None:
        """Initialize the service registry."""
        if not self._service_registry:
            raise GameAutomationError(
                message="Service registry not created",
                error_code="ENG002",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL
            )
        
        # Register core services
        self._service_registry.register_service_type(ConfigManager)
        self._service_registry.register_service_type(LogManager)
        self._service_registry.register_service_type(EventManager)
        self._service_registry.register_service_type(ErrorManager)
        
        # Start service registry
        await self._service_registry.start()
    
    async def cleanup(self) -> None:
        """Cleanup engine resources."""
        self.log_info("Cleaning up game engine resources")
        
        # Stop service registry first to ensure proper service shutdown
        if self._service_registry:
            await self._service_registry.stop()
        
        # Stop core managers in reverse order
        if self._error_manager:
            await self._error_manager.stop()
        
        if self._event_manager:
            await self._event_manager.stop()
        
        if self._log_manager:
            await self._log_manager.stop()
        
        if self._config_manager:
            await self._config_manager.stop()
        
        self.log_info("Game engine cleanup completed")
    
    def register_service(self, service_type: Type[IGameService],
                        dependencies: Optional[list] = None) -> None:
        """Register a service with the engine."""
        if not self._service_registry:
            raise GameAutomationError(
                message="Service registry not initialized",
                error_code="ENG003",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.ERROR
            )
        
        self._service_registry.register_service_type(service_type, dependencies)
    
    async def emit_event(self, event_type: str, data: Any = None,
                        source: str = None) -> None:
        """Emit an event through the event manager."""
        if self._event_manager:
            event = Event(event_type, data, source)
            await self._event_manager.emit(event)
    
    async def handle_error(self, error: Exception,
                          context: Optional[Dict[str, Any]] = None) -> None:
        """Handle an error through the error manager."""
        if self._error_manager:
            await self._error_manager.handle_error(error, context)
    
    def get_service(self, service_name: str) -> Optional[IGameService]:
        """Get a registered service by name."""
        if self._service_registry:
            return self._service_registry.get_service(service_name)
        return None
    
    def get_config(self, namespace: str = "default") -> Dict[str, Any]:
        """Get configuration from config manager."""
        if self._config_manager:
            return self._config_manager.get_config(namespace)
        return {}
    
    def get_uptime(self) -> float:
        """Get engine uptime in seconds."""
        if self._start_time:
            return (datetime.now() - self._start_time).total_seconds()
        return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status information."""
        status = {
            'uptime': self.get_uptime(),
            'start_time': self._start_time.isoformat() if self._start_time else None,
            'is_running': self.is_running
        }
        
        # Add service statuses
        if self._service_registry:
            service_statuses = {
                name: service.get_status()
                for name, service in self._service_registry.get_all_services().items()
            }
            status['services'] = service_statuses
        
        return status
    
    async def process(self, input_data: Dict[str, Any]) -> Any:
        """Process engine-level requests."""
        action = input_data.get('action')
        
        if action == 'get_status':
            return {'status': 'success', 'engine_status': self.get_status()}
        
        elif action == 'get_service_status':
            service_name = input_data.get('service_name')
            if not service_name:
                raise ValueError("Service name required")
            
            service = self.get_service(service_name)
            if not service:
                return {'status': 'error', 'message': f'Service not found: {service_name}'}
            
            return {
                'status': 'success',
                'service_status': service.get_status()
            }
        
        else:
            raise ValueError(f"Unsupported action: {action}")
    
    async def validate(self, data: Any) -> bool:
        """Validate engine request data."""
        if not isinstance(data, dict):
            return False
        
        required_fields = ['action']
        return all(field in data for field in required_fields)

    async def run_game_loop(self):
        """运行游戏主循环"""
        try:
            self.log_info("Starting game loop")
            while True:
                # 在这里添加游戏循环的具体逻辑
                await asyncio.sleep(1)  # 暂时只是简单的休眠
                
        except asyncio.CancelledError:
            self.log_info("Game loop cancelled")
        except Exception as e:
            self.log_error("Error in game loop", e)
            raise

    async def run_automated_tests(self):
        """运行自动化测试"""
        try:
            self.log_info("Starting automated tests")
            # TODO: 实现自动化测试逻辑
            await asyncio.sleep(1)  # 暂时只是简单的休眠
            self.log_info("Automated tests completed")
        except Exception as e:
            self.log_error("Error in automated tests", e)
            raise

    async def optimize_performance(self):
        """执行性能优化"""
        try:
            self.log_info("Starting performance optimization")
            # TODO: 实现性能优化逻辑
            await asyncio.sleep(1)  # 暂时只是简单的休眠
            self.log_info("Performance optimization completed")
        except Exception as e:
            self.log_error("Error in performance optimization", e)
            raise
