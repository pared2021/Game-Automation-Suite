from typing import Any, Dict, List, Optional, Type
import traceback
import sys
from datetime import datetime
from enum import Enum
from ..base.service_base import ServiceBase

class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

class ErrorCategory(Enum):
    """Error categories."""
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"

class GameAutomationError(Exception):
    """Base exception class for game automation errors."""
    
    def __init__(self, message: str, error_code: str,
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 original_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.timestamp = datetime.now()
        self.original_error = original_error
        self.traceback = traceback.extract_tb(sys.exc_info()[2]) if original_error else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            'message': str(self),
            'error_code': self.error_code,
            'severity': self.severity.name,
            'category': self.category.value,
            'timestamp': self.timestamp.isoformat(),
            'original_error': str(self.original_error) if self.original_error else None,
            'traceback': [str(frame) for frame in self.traceback] if self.traceback else None
        }

class ErrorManager(ServiceBase):
    """Centralized error management service."""
    
    def __init__(self):
        super().__init__("ErrorManager")
        self._error_handlers: Dict[Type[Exception], List[callable]] = {}
        self._error_history: List[GameAutomationError] = []
        self._max_history = 1000
        self._error_codes: Dict[str, Dict[str, Any]] = {}
        self._initialize_error_codes()
    
    def _initialize_error_codes(self) -> None:
        """Initialize standard error codes."""
        self._error_codes = {
            'SYS001': {'message': 'System initialization error', 'category': ErrorCategory.SYSTEM},
            'NET001': {'message': 'Network connection error', 'category': ErrorCategory.NETWORK},
            'CFG001': {'message': 'Configuration error', 'category': ErrorCategory.CONFIGURATION},
            'AUTH001': {'message': 'Authentication error', 'category': ErrorCategory.AUTHENTICATION},
            'VAL001': {'message': 'Validation error', 'category': ErrorCategory.VALIDATION},
            'SVC001': {'message': 'External service error', 'category': ErrorCategory.EXTERNAL_SERVICE}
        }
    
    def register_error_code(self, code: str, message: str,
                          category: ErrorCategory = ErrorCategory.UNKNOWN) -> None:
        """Register a new error code."""
        if code in self._error_codes:
            self.log_warning(f"Error code {code} already exists and will be overwritten")
        
        self._error_codes[code] = {
            'message': message,
            'category': category
        }
        self.log_debug(f"Registered error code: {code}")
    
    def register_handler(self, error_type: Type[Exception], handler: callable) -> None:
        """Register an error handler for a specific exception type."""
        if error_type not in self._error_handlers:
            self._error_handlers[error_type] = []
        self._error_handlers[error_type].append(handler)
        self.log_debug(f"Registered handler for {error_type.__name__}")
    
    def unregister_handler(self, error_type: Type[Exception], handler: callable) -> None:
        """Unregister an error handler."""
        if error_type in self._error_handlers:
            self._error_handlers[error_type].remove(handler)
            if not self._error_handlers[error_type]:
                del self._error_handlers[error_type]
            self.log_debug(f"Unregistered handler for {error_type.__name__}")
    
    async def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Handle an error with registered handlers."""
        handled = False
        
        # Convert to GameAutomationError if it's not already
        if not isinstance(error, GameAutomationError):
            error = self._convert_error(error)
        
        # Add to history
        self._add_to_history(error)
        
        # Find and execute handlers
        for error_type, handlers in self._error_handlers.items():
            if isinstance(error, error_type):
                for handler in handlers:
                    try:
                        if context:
                            await handler(error, context)
                        else:
                            await handler(error)
                        handled = True
                    except Exception as e:
                        self.log_error(f"Error in error handler: {str(e)}", e)
        
        if not handled:
            self.log_error(f"Unhandled error: {str(error)}")
            # Re-raise if critical
            if isinstance(error, GameAutomationError) and error.severity == ErrorSeverity.CRITICAL:
                raise error
    
    def _convert_error(self, error: Exception) -> GameAutomationError:
        """Convert a standard exception to GameAutomationError."""
        error_code = 'SYS999'  # Default error code
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.ERROR
        
        # Determine appropriate error categorization based on exception type
        if isinstance(error, (ConnectionError, TimeoutError)):
            error_code = 'NET001'
            category = ErrorCategory.NETWORK
        elif isinstance(error, ValueError):
            error_code = 'VAL001'
            category = ErrorCategory.VALIDATION
        elif isinstance(error, PermissionError):
            error_code = 'AUTH001'
            category = ErrorCategory.AUTHORIZATION
        
        return GameAutomationError(
            message=str(error),
            error_code=error_code,
            category=category,
            severity=severity,
            original_error=error
        )
    
    def _add_to_history(self, error: GameAutomationError) -> None:
        """Add error to history."""
        self._error_history.append(error)
        if len(self._error_history) > self._max_history:
            self._error_history.pop(0)
    
    def get_error_history(self, severity: Optional[ErrorSeverity] = None,
                         category: Optional[ErrorCategory] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[GameAutomationError]:
        """Get error history with optional filtering."""
        errors = self._error_history
        
        if severity:
            errors = [e for e in errors if e.severity == severity]
        
        if category:
            errors = [e for e in errors if e.category == category]
        
        if start_time:
            errors = [e for e in errors if e.timestamp >= start_time]
        
        if end_time:
            errors = [e for e in errors if e.timestamp <= end_time]
        
        return errors
    
    def clear_error_history(self) -> None:
        """Clear error history."""
        self._error_history.clear()
        self.log_info("Error history cleared")
    
    async def process(self, input_data: Dict[str, Any]) -> Any:
        """Process error-related requests."""
        action = input_data.get('action')
        
        if action == 'get_history':
            severity = input_data.get('severity')
            category = input_data.get('category')
            start_time = input_data.get('start_time')
            end_time = input_data.get('end_time')
            
            if severity:
                severity = ErrorSeverity[severity.upper()]
            if category:
                category = ErrorCategory(category.lower())
            if start_time:
                start_time = datetime.fromisoformat(start_time)
            if end_time:
                end_time = datetime.fromisoformat(end_time)
            
            errors = self.get_error_history(severity, category, start_time, end_time)
            return {'status': 'success', 'errors': [e.to_dict() for e in errors]}
        
        elif action == 'clear_history':
            self.clear_error_history()
            return {'status': 'success', 'message': 'Error history cleared'}
        
        else:
            raise ValueError(f"Unsupported action: {action}")
    
    async def validate(self, data: Any) -> bool:
        """Validate error-related request data."""
        if not isinstance(data, dict):
            return False
        
        required_fields = ['action']
        return all(field in data for field in required_fields)
