from typing import Any, Dict, List, Optional
from ..interfaces.base_interface import IManager, IConfigurable, ILoggable
import logging
import asyncio
from datetime import datetime

class EngineBase(IManager, IConfigurable, ILoggable):
    """Base class for the game automation engine."""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
        self._services: List[Any] = []
        self._is_running: bool = False
        self._start_time: Optional[datetime] = None
    
    async def initialize(self) -> None:
        """Initialize the engine."""
        self._logger.info("Initializing engine...")
        self._start_time = datetime.now()
        await self._initialize_services()
    
    async def cleanup(self) -> None:
        """Cleanup engine resources."""
        self._logger.info("Cleaning up engine resources...")
        await self._cleanup_services()
        self._is_running = False
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the engine."""
        self._config = config
        self._logger.info("Engine configured with new settings")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current engine configuration."""
        return self._config.copy()
    
    def log_info(self, message: str) -> None:
        """Log information message."""
        self._logger.info(message)
    
    def log_error(self, message: str, error: Optional[Exception] = None) -> None:
        """Log error message."""
        if error:
            self._logger.error(f"{message}: {str(error)}", exc_info=error)
        else:
            self._logger.error(message)
    
    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self._logger.debug(message)
    
    async def _initialize_services(self) -> None:
        """Initialize all registered services."""
        for service in self._services:
            try:
                await service.start()
            except Exception as e:
                self.log_error(f"Failed to initialize service {service.__class__.__name__}", e)
                raise
    
    async def _cleanup_services(self) -> None:
        """Cleanup all registered services."""
        for service in self._services:
            try:
                await service.stop()
            except Exception as e:
                self.log_error(f"Failed to cleanup service {service.__class__.__name__}", e)
    
    def register_service(self, service: Any) -> None:
        """Register a new service with the engine."""
        self._services.append(service)
        self.log_info(f"Registered service: {service.__class__.__name__}")
    
    def get_uptime(self) -> float:
        """Get engine uptime in seconds."""
        if self._start_time:
            return (datetime.now() - self._start_time).total_seconds()
        return 0.0
    
    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._is_running
