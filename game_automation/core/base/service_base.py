from typing import Any, Dict, Optional
import logging
from datetime import datetime
from ..interfaces.service_interface import IGameService

class ServiceBase(IGameService):
    """Base class for all game services."""
    
    def __init__(self, service_name: str):
        self._name = service_name
        self._logger = logging.getLogger(service_name)
        self._config: Dict[str, Any] = {}
        self._status: str = "initialized"
        self._start_time: Optional[datetime] = None
        self._metrics: Dict[str, Any] = {}
    
    async def start(self) -> None:
        """Start the service."""
        self._logger.info(f"Starting service: {self._name}")
        self._start_time = datetime.now()
        self._status = "running"
        await self._on_start()
    
    async def stop(self) -> None:
        """Stop the service."""
        self._logger.info(f"Stopping service: {self._name}")
        await self._on_stop()
        self._status = "stopped"
    
    async def _on_start(self) -> None:
        """Hook for additional start logic in derived classes."""
        pass
    
    async def _on_stop(self) -> None:
        """Hook for additional stop logic in derived classes."""
        pass
    
    def get_status(self) -> str:
        """Get current service status."""
        return self._status
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the service."""
        self._config = config
        self._logger.info(f"Service {self._name} configured with new settings")
        self._on_configure()
    
    def _on_configure(self) -> None:
        """Hook for additional configuration logic in derived classes."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get current service configuration."""
        return self._config.copy()
    
    async def process(self, input_data: Any) -> Any:
        """Process input data. Must be implemented by derived classes."""
        raise NotImplementedError("Process method must be implemented by derived classes")
    
    async def validate(self, data: Any) -> bool:
        """Validate input data. Must be implemented by derived classes."""
        raise NotImplementedError("Validate method must be implemented by derived classes")
    
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
    
    def update_metric(self, key: str, value: Any) -> None:
        """Update service metric."""
        self._metrics[key] = value
        self.log_debug(f"Updated metric {key}: {value}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all service metrics."""
        return self._metrics.copy()
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        if self._start_time:
            return (datetime.now() - self._start_time).total_seconds()
        return 0.0
    
    @property
    def name(self) -> str:
        """Get service name."""
        return self._name
    
    def __str__(self) -> str:
        return f"{self._name} Service (Status: {self._status})"
