from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger
from .task_types import Task, TaskStatus

class TaskRuleError(GameAutomationError):
    """Task rule related errors"""
    pass

class TaskRule:
    """Task execution rule"""
    
    def __init__(self, task: Task, conditions: Dict[str, Any] = None):
        """Initialize task rule
        
        Args:
            task: Associated task
            conditions: Rule conditions
        """
        self.task_id = task.task_id
        self.conditions = conditions or {}
        self.last_evaluation: Optional[datetime] = None
        self.last_result: bool = False
        
        # Rule statistics
        self.evaluation_count = 0
        self.success_count = 0
        self.failure_count = 0

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule
        
        Args:
            context: Evaluation context
            
        Returns:
            bool: Whether conditions are met
        """
        try:
            self.evaluation_count += 1
            self.last_evaluation = datetime.now()
            
            # Default to True if no conditions
            if not self.conditions:
                self.success_count += 1
                self.last_result = True
                return True
            
            # Evaluate all conditions
            result = self._evaluate_conditions(self.conditions, context)
            
            if result:
                self.success_count += 1
            else:
                self.failure_count += 1
                
            self.last_result = result
            return result
            
        except Exception as e:
            detailed_logger.error(f"Rule evaluation failed: {str(e)}")
            self.failure_count += 1
            self.last_result = False
            return False

    def _evaluate_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate rule conditions
        
        Args:
            conditions: Rule conditions
            context: Evaluation context
            
        Returns:
            bool: Whether conditions are met
        """
        # Implement condition evaluation logic here
        # This is a placeholder that always returns True
        return True

    def to_dict(self) -> Dict:
        """Convert to dictionary format
        
        Returns:
            Dict: Rule information dictionary
        """
        return {
            'task_id': self.task_id,
            'conditions': self.conditions,
            'last_evaluation': self.last_evaluation.isoformat() if self.last_evaluation else None,
            'last_result': self.last_result,
            'statistics': {
                'evaluation_count': self.evaluation_count,
                'success_count': self.success_count,
                'failure_count': self.failure_count
            }
        }

class TaskRuleManager:
    """Task rule manager"""
    
    def __init__(self):
        """Initialize task rule manager"""
        self.rules: Dict[str, TaskRule] = {}

    def create_rule(self, task: Task, conditions: Dict[str, Any] = None) -> TaskRule:
        """Create task rule
        
        Args:
            task: Task instance
            conditions: Rule conditions
            
        Returns:
            TaskRule: Created rule
        """
        rule = TaskRule(task, conditions)
        self.rules[task.task_id] = rule
        return rule

    def remove_rule(self, task_id: str) -> None:
        """Remove task rule
        
        Args:
            task_id: Task ID
        """
        if task_id in self.rules:
            del self.rules[task_id]

    def get_rule(self, task_id: str) -> Optional[TaskRule]:
        """Get task rule
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[TaskRule]: Rule instance or None if not found
        """
        return self.rules.get(task_id)

    def evaluate_rule(self, task_id: str, context: Dict[str, Any]) -> bool:
        """Evaluate task rule
        
        Args:
            task_id: Task ID
            context: Evaluation context
            
        Returns:
            bool: Whether conditions are met
        """
        rule = self.get_rule(task_id)
        if not rule:
            return True  # No rule means always allowed
        return rule.evaluate(context)

    def get_statistics(self) -> Dict:
        """Get rule statistics
        
        Returns:
            Dict: Statistics information
        """
        return {
            'total_rules': len(self.rules),
            'rules': {
                task_id: rule.to_dict()['statistics']
                for task_id, rule in self.rules.items()
            }
        }

    def save_rules(self, filepath: str) -> None:
        """Save rules to file
        
        Args:
            filepath: Save path
        """
        try:
            rules_data = {
                task_id: rule.to_dict()
                for task_id, rule in self.rules.items()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
                
            detailed_logger.info(f"Rules saved: {filepath}")
            
        except Exception as e:
            raise TaskRuleError(f"Save rules failed: {str(e)}")

    def load_rules(self, filepath: str) -> None:
        """Load rules from file
        
        Args:
            filepath: Rules file path
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
                
            self.rules.clear()
            
            for task_id, rule_data in rules_data.items():
                # Create a dummy task for rule creation
                task = Task(task_id=task_id, name=f"Task_{task_id}")
                rule = self.create_rule(task, rule_data.get('conditions'))
                
                # Restore statistics
                if 'statistics' in rule_data:
                    stats = rule_data['statistics']
                    rule.evaluation_count = stats.get('evaluation_count', 0)
                    rule.success_count = stats.get('success_count', 0)
                    rule.failure_count = stats.get('failure_count', 0)
                
                if rule_data.get('last_evaluation'):
                    rule.last_evaluation = datetime.fromisoformat(rule_data['last_evaluation'])
                rule.last_result = rule_data.get('last_result', False)
                
            detailed_logger.info(f"Rules loaded: {filepath}")
            
        except Exception as e:
            raise TaskRuleError(f"Load rules failed: {str(e)}")
