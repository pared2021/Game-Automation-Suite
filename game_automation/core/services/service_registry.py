from typing import Any, Dict, List, Type, Optional, Set
import asyncio
from datetime import datetime
from ..base.service_base import ServiceBase
from ..interfaces.service_interface import IGameService
from ..error.error_manager import GameAutomationError, ErrorCategory, ErrorSeverity

class ServiceDependencyError(GameAutomationError):
    """Error raised when service dependencies cannot be resolved."""
    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_code='SVC001',
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL
        )

class ServiceRegistry(ServiceBase):
    """Central registry for managing services and their lifecycle."""
    
    def __init__(self):
        super().__init__("ServiceRegistry")
        self._services: Dict[str, IGameService] = {}
        self._service_types: Dict[str, Type[IGameService]] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._startup_order: List[str] = []
        self._initialized = False
    
    async def _on_start(self) -> None:
        """Initialize the service registry."""
        if not self._initialized:
            self._initialized = True
            await self._start_services()
    
    async def _on_stop(self) -> None:
        """Stop all services in reverse order."""
        for service_name in reversed(self._startup_order):
            service = self._services.get(service_name)
            if service:
                try:
                    await service.stop()
                    self.log_info(f"Stopped service: {service_name}")
                except Exception as e:
                    self.log_error(f"Error stopping service {service_name}", e)
    
    def register_service_type(self, service_type: Type[IGameService],
                            dependencies: Optional[List[str]] = None) -> None:
        """Register a service type with its dependencies."""
        service_name = service_type.__name__
        self._service_types[service_name] = service_type
        self._dependencies[service_name] = set(dependencies or [])
        self.log_info(f"Registered service type: {service_name}")
    
    def _resolve_dependencies(self) -> List[str]:
        """Resolve service dependencies and determine startup order."""
        resolved = []
        unresolved = set(self._service_types.keys())
        
        while unresolved:
            progress = False
            remaining = unresolved.copy()
            
            for service_name in remaining:
                deps = self._dependencies[service_name]
                if deps.issubset(set(resolved)):
                    resolved.append(service_name)
                    unresolved.remove(service_name)
                    progress = True
            
            if not progress and unresolved:
                # Circular dependency detected
                raise ServiceDependencyError(
                    f"Circular dependency detected in services: {unresolved}"
                )
        
        return resolved
    
    async def _start_services(self) -> None:
        """Start all services in dependency order."""
        try:
            self._startup_order = self._resolve_dependencies()
            
            for service_name in self._startup_order:
                service_type = self._service_types[service_name]
                service = service_type()
                
                # Initialize service
                try:
                    await service.start()
                    self._services[service_name] = service
                    self.log_info(f"Started service: {service_name}")
                except Exception as e:
                    self.log_error(f"Error starting service {service_name}", e)
                    raise ServiceDependencyError(
                        f"Failed to start service {service_name}: {str(e)}"
                    )
        
        except Exception as e:
            self.log_error("Error starting services", e)
            await self._cleanup_services()
            raise
    
    async def _cleanup_services(self) -> None:
        """Cleanup services after startup failure."""
        for service_name in reversed(list(self._services.keys())):
            service = self._services[service_name]
            try:
                await service.stop()
                self.log_info(f"Cleaned up service: {service_name}")
            except Exception as e:
                self.log_error(f"Error cleaning up service {service_name}", e)
    
    def get_service(self, service_name: str) -> Optional[IGameService]:
        """Get a registered service by name."""
        return self._services.get(service_name)
    
    def get_all_services(self) -> Dict[str, IGameService]:
        """Get all registered services."""
        return self._services.copy()
    
    def get_service_status(self, service_name: str) -> Optional[str]:
        """Get the status of a specific service."""
        service = self._services.get(service_name)
        return service.get_status() if service else None
    
    def get_service_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all services."""
        metrics = {}
        for service_name, service in self._services.items():
            try:
                service_metrics = service.get_metrics()
                metrics[service_name] = service_metrics
            except Exception as e:
                self.log_error(f"Error getting metrics from service {service_name}", e)
                metrics[service_name] = {'error': str(e)}
        return metrics
    
    async def process(self, input_data: Dict[str, Any]) -> Any:
        """Process service registry requests."""
        action = input_data.get('action')
        
        if action == 'get_status':
            service_name = input_data.get('service_name')
            if service_name:
                status = self.get_service_status(service_name)
                return {'status': 'success', 'service_status': status}
            else:
                statuses = {name: service.get_status() 
                          for name, service in self._services.items()}
                return {'status': 'success', 'service_statuses': statuses}
        
        elif action == 'get_metrics':
            service_name = input_data.get('service_name')
            if service_name:
                service = self.get_service(service_name)
                if service:
                    metrics = service.get_metrics()
                    return {'status': 'success', 'metrics': metrics}
                return {'status': 'error', 'message': f'Service not found: {service_name}'}
            else:
                metrics = self.get_service_metrics()
                return {'status': 'success', 'metrics': metrics}
        
        else:
            raise ValueError(f"Unsupported action: {action}")
    
    async def validate(self, data: Any) -> bool:
        """Validate service registry request data."""
        if not isinstance(data, dict):
            return False
        
        required_fields = ['action']
        return all(field in data for field in required_fields)
    
    def __str__(self) -> str:
        """String representation of service registry."""
        services = [f"{name} ({service.get_status()})"
                   for name, service in self._services.items()]
        return f"ServiceRegistry(services=[{', '.join(services)}])"
