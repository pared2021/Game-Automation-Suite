from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class IManager(ABC):
    """Base interface for all manager classes."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the manager."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass

class IConfigurable(ABC):
    """Interface for components that can be configured."""
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the component with given configuration."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        pass

class ILoggable(ABC):
    """Interface for components that support logging."""
    
    @abstractmethod
    def log_info(self, message: str) -> None:
        """Log information message."""
        pass
    
    @abstractmethod
    def log_error(self, message: str, error: Optional[Exception] = None) -> None:
        """Log error message."""
        pass
    
    @abstractmethod
    def log_debug(self, message: str) -> None:
        """Log debug message."""
        pass

class IPlugin(ABC):
    """Base interface for plugins."""
    
    @abstractmethod
    async def load(self) -> None:
        """Load the plugin."""
        pass
    
    @abstractmethod
    async def unload(self) -> None:
        """Unload the plugin."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, str]:
        """Get plugin information."""
        pass

class IService(ABC):
    """Base interface for services."""
    
    @abstractmethod
    async def start(self) -> None:
        """Start the service."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the service."""
        pass
    
    @abstractmethod
    def get_status(self) -> str:
        """Get service status."""
        pass
