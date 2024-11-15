from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from .base_interface import IService, IConfigurable, ILoggable

class IGameService(IService, IConfigurable, ILoggable):
    """Interface for game-related services."""
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input data and return result."""
        pass
    
    @abstractmethod
    async def validate(self, data: Any) -> bool:
        """Validate input data."""
        pass

class IAnalysisService(IGameService):
    """Interface for analysis services."""
    
    @abstractmethod
    async def analyze(self, game_state: Any) -> Dict[str, Any]:
        """Analyze game state and return results."""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Get analysis metrics."""
        pass

class IDecisionService(IGameService):
    """Interface for decision making services."""
    
    @abstractmethod
    async def make_decision(self, state: Any) -> Any:
        """Make a decision based on current state."""
        pass
    
    @abstractmethod
    async def learn(self, experience: Dict[str, Any]) -> None:
        """Learn from experience."""
        pass
    
    @abstractmethod
    async def evaluate(self) -> Dict[str, float]:
        """Evaluate decision making performance."""
        pass

class IDeviceService(IGameService):
    """Interface for device management services."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to device."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from device."""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if device is connected."""
        pass
    
    @abstractmethod
    async def execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute action on device."""
        pass

class IResourceService(IGameService):
    """Interface for resource management services."""
    
    @abstractmethod
    async def allocate(self, resource_type: str, amount: int) -> bool:
        """Allocate resources."""
        pass
    
    @abstractmethod
    async def release(self, resource_type: str, amount: int) -> None:
        """Release resources."""
        pass
    
    @abstractmethod
    async def get_available(self, resource_type: str) -> int:
        """Get available resource amount."""
        pass

class IPluginService(IGameService):
    """Interface for plugin management services."""
    
    @abstractmethod
    async def load_plugin(self, plugin_id: str) -> bool:
        """Load a plugin."""
        pass
    
    @abstractmethod
    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin."""
        pass
    
    @abstractmethod
    def get_loaded_plugins(self) -> List[str]:
        """Get list of loaded plugins."""
        pass
    
    @abstractmethod
    async def execute_plugin(self, plugin_id: str, action: str, params: Dict[str, Any]) -> Any:
        """Execute plugin action."""
        pass
