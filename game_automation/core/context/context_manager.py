"""Context management for game automation."""

from typing import Optional, Dict, Any, List
from datetime import datetime
import json

from ..events.event_manager import EventManager, Event, EventType
from ...utils.logger import get_logger

logger = get_logger(__name__)

class ContextManager:
    """Manages game context and state"""
    
    def __init__(self):
        """Initialize context manager"""
        self._context: Dict[str, Any] = {}
        self._state_history: List[Dict[str, Any]] = []
        self._max_history = 100
        
    def set_context(self, key: str, value: Any):
        """Set context value
        
        Args:
            key: Context key
            value: Context value
        """
        self._context[key] = value
        self._update_history()
        
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context value
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Any: Context value
        """
        return self._context.get(key, default)
        
    def clear_context(self):
        """Clear all context"""
        self._context.clear()
        self._update_history()
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state
        
        Returns:
            Dict[str, Any]: Current state
        """
        return self._context.copy()
        
    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get state history
        
        Returns:
            List[Dict[str, Any]]: State history
        """
        return self._state_history.copy()
        
    def _update_history(self):
        """Update state history"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'state': self.get_state()
        }
        
        self._state_history.append(state)
        
        # Keep history size limited
        if len(self._state_history) > self._max_history:
            self._state_history.pop(0)
            
    def save_state(self, filepath: str):
        """Save current state to file
        
        Args:
            filepath: Output file path
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.get_state(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")
            
    def load_state(self, filepath: str) -> bool:
        """Load state from file
        
        Args:
            filepath: Input file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
                
            self._context = state
            self._update_history()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            return False
