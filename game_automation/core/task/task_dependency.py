"""Task dependency management."""

from typing import Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import networkx as nx
from datetime import datetime

from utils.logger import detailed_logger
from .task_types import Task, TaskStatus
from .task_lifecycle import TaskState, TaskLifecycleManager

class DependencyType(Enum):
    """Dependency type enumeration"""
    HARD = auto()    # Must be satisfied
    SOFT = auto()    # Optional, but preferred
    WEAK = auto()    # Optional, can be ignored

class DependencyError(Exception):
    """Dependency related errors"""
    pass

class TaskDependencyManager:
    """Task dependency manager."""
    
    def __init__(self, lifecycle_manager: TaskLifecycleManager):
        """Initialize dependency manager.
        
        Args:
            lifecycle_manager: Task lifecycle manager
        """
        self.lifecycle = lifecycle_manager
        
        # Dependency graph
        self.dependency_graph = nx.DiGraph()
        
        # Dependency types
        self.dependency_types: Dict[Tuple[str, str], DependencyType] = {}
        
        # Task readiness cache
        self._ready_cache: Dict[str, bool] = {}
        self._ready_time: Dict[str, datetime] = {}
        
        # Cache invalidation timeout (seconds)
        self.cache_timeout = 5
        
        # Add lifecycle hooks
        self._add_lifecycle_hooks()
        
    def _add_lifecycle_hooks(self):
        """Add lifecycle management hooks."""
        def on_state_change(task: Task,
                          from_state: TaskState,
                          to_state: TaskState):
            # Invalidate cache for dependent tasks
            dependent_tasks = self.get_dependent_tasks(task.task_id)
            for dep_task_id in dependent_tasks:
                self._invalidate_cache(dep_task_id)
                
        self.lifecycle.add_transition_hook(on_state_change)
        
    def add_dependency(self,
                      task_id: str,
                      dependency_id: str,
                      dep_type: DependencyType = DependencyType.HARD) -> None:
        """Add task dependency.
        
        Args:
            task_id: Task ID
            dependency_id: Dependency task ID
            dep_type: Dependency type
        """
        # Add to graph
        self.dependency_graph.add_edge(task_id, dependency_id)
        
        # Set dependency type
        self.dependency_types[(task_id, dependency_id)] = dep_type
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            if cycles:
                # Remove edge and raise error
                self.dependency_graph.remove_edge(task_id, dependency_id)
                del self.dependency_types[(task_id, dependency_id)]
                raise DependencyError(
                    f"Adding dependency would create cycles: {cycles}"
                )
        except nx.NetworkXNoCycle:
            pass
            
        # Invalidate cache
        self._invalidate_cache(task_id)
        
        detailed_logger.info(
            f"Added {dep_type.name} dependency: {task_id} -> {dependency_id}"
        )
        
    def remove_dependency(self, task_id: str, dependency_id: str) -> None:
        """Remove task dependency.
        
        Args:
            task_id: Task ID
            dependency_id: Dependency task ID
        """
        # Remove from graph
        self.dependency_graph.remove_edge(task_id, dependency_id)
        
        # Remove dependency type
        if (task_id, dependency_id) in self.dependency_types:
            del self.dependency_types[(task_id, dependency_id)]
            
        # Invalidate cache
        self._invalidate_cache(task_id)
        
        detailed_logger.info(
            f"Removed dependency: {task_id} -> {dependency_id}"
        )
        
    def get_dependencies(self, task_id: str) -> List[str]:
        """Get task dependencies.
        
        Args:
            task_id: Task ID
            
        Returns:
            List[str]: List of dependency task IDs
        """
        if task_id not in self.dependency_graph:
            return []
        return list(self.dependency_graph.successors(task_id))
        
    def get_dependent_tasks(self, task_id: str) -> List[str]:
        """Get tasks that depend on this task.
        
        Args:
            task_id: Task ID
            
        Returns:
            List[str]: List of dependent task IDs
        """
        if task_id not in self.dependency_graph:
            return []
        return list(self.dependency_graph.predecessors(task_id))
        
    def get_dependency_type(self,
                          task_id: str,
                          dependency_id: str) -> Optional[DependencyType]:
        """Get dependency type.
        
        Args:
            task_id: Task ID
            dependency_id: Dependency task ID
            
        Returns:
            Optional[DependencyType]: Dependency type or None if not found
        """
        return self.dependency_types.get((task_id, dependency_id))
        
    def is_dependency_satisfied(self,
                              task_id: str,
                              dependency_id: str) -> bool:
        """Check if dependency is satisfied.
        
        Args:
            task_id: Task ID
            dependency_id: Dependency task ID
            
        Returns:
            bool: True if dependency is satisfied
        """
        # Get dependency type
        dep_type = self.get_dependency_type(task_id, dependency_id)
        if not dep_type:
            return True
            
        # Get dependency state
        dep_state = self.lifecycle.get_task_state(dependency_id)
        if not dep_state:
            return False
            
        # Check based on dependency type
        if dep_type == DependencyType.HARD:
            return dep_state == TaskState.COMPLETED
        elif dep_type == DependencyType.SOFT:
            return dep_state in (TaskState.COMPLETED, TaskState.RUNNING)
        else:  # WEAK
            return dep_state != TaskState.FAILED
            
    def are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies are satisfied.
        
        Args:
            task_id: Task ID
            
        Returns:
            bool: True if all dependencies are satisfied
        """
        # Check cache
        if task_id in self._ready_cache:
            cache_time = self._ready_time.get(task_id)
            if cache_time:
                age = (datetime.now() - cache_time).total_seconds()
                if age < self.cache_timeout:
                    return self._ready_cache[task_id]
                    
        # Check all dependencies
        dependencies = self.get_dependencies(task_id)
        result = all(
            self.is_dependency_satisfied(task_id, dep_id)
            for dep_id in dependencies
        )
        
        # Update cache
        self._ready_cache[task_id] = result
        self._ready_time[task_id] = datetime.now()
        
        return result
        
    def get_dependency_chain(self, task_id: str) -> List[List[str]]:
        """Get dependency chain for task.
        
        Args:
            task_id: Task ID
            
        Returns:
            List[List[str]]: List of dependency chains
        """
        if task_id not in self.dependency_graph:
            return []
            
        chains = []
        for path in nx.all_simple_paths(
            self.dependency_graph,
            task_id,
            list(nx.topological_sort(self.dependency_graph))
        ):
            chains.append(path)
        return chains
        
    def get_critical_path(self, task_id: str) -> Optional[List[str]]:
        """Get critical dependency path.
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[List[str]]: Critical path or None if no dependencies
        """
        chains = self.get_dependency_chain(task_id)
        if not chains:
            return None
            
        # Find longest chain
        return max(chains, key=len)
        
    def _invalidate_cache(self, task_id: str) -> None:
        """Invalidate readiness cache for task.
        
        Args:
            task_id: Task ID
        """
        if task_id in self._ready_cache:
            del self._ready_cache[task_id]
        if task_id in self._ready_time:
            del self._ready_time[task_id]
