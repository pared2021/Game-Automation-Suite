from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import json
import os
import yaml
from enum import Enum, auto

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger

class RuleVersion:
    """规则版本管理"""
    
    def __init__(self, major: int = 1, minor: int = 0, patch: int = 0):
        """初始化版本
        
        Args:
            major: 主版本号
            minor: 次版本号
            patch: 补丁版本号
        """
        self.major = major
        self.minor = minor
        self.patch = patch

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __eq__(self, other: 'RuleVersion') -> bool:
        return (self.major == other.major and
                self.minor == other.minor and
                self.patch == other.patch)

    def __lt__(self, other: 'RuleVersion') -> bool:
        return (self.major < other.major or
                (self.major == other.major and self.minor < other.minor) or
                (self.major == other.major and self.minor == other.minor and
                 self.patch < other.patch))

class RuleFormat:
    """规则文件格式定义"""
    
    CURRENT_VERSION = RuleVersion(1, 0, 0)
    
    @staticmethod
    def validate_schema(data: Dict) -> bool:
        """验证规则数据格式
        
        Args:
            data: 规则数据
            
        Returns:
            bool: 是否有效
        """
        required_fields = {
            'version',
            'rules',
            'metadata'
        }
        
        if not all(field in data for field in required_fields):
            return False
            
        # 验证版本格式
        version = data['version']
        if not isinstance(version, str):
            return False
        try:
            major, minor, patch = map(int, version.split('.'))
        except ValueError:
            return False
            
        # 验证规则列表
        rules = data['rules']
        if not isinstance(rules, list):
            return False
            
        for rule in rules:
            if not RuleFormat.validate_rule(rule):
                return False
                
        # 验证元数据
        metadata = data['metadata']
        if not isinstance(metadata, dict):
            return False
            
        return True

    @staticmethod
    def validate_rule(rule: Dict) -> bool:
        """验证单个规则格式
        
        Args:
            rule: 规则数据
            
        Returns:
            bool: 是否有效
        """
        required_fields = {
            'id',
            'name',
            'conditions',
            'actions',
            'priority'
        }
        
        if not all(field in rule for field in required_fields):
            return False
            
        # 验证条件列表
        conditions = rule['conditions']
        if not isinstance(conditions, list):
            return False
            
        for condition in conditions:
            if not isinstance(condition, dict):
                return False
            if 'type' not in condition or 'parameters' not in condition:
                return False
                
        # 验证动作列表
        actions = rule['actions']
        if not isinstance(actions, list):
            return False
            
        for action in actions:
            if not isinstance(action, dict):
                return False
            if 'type' not in action or 'parameters' not in action:
                return False
                
        return True

class RuleTemplate:
    """规则模板"""
    
    def __init__(self, template_id: str, name: str):
        """初始化规则模板
        
        Args:
            template_id: 模板ID
            name: 模板名称
        """
        self.template_id = template_id
        self.name = name
        self.conditions: List[Dict] = []
        self.actions: List[Dict] = []
        self.parameters: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.version = RuleFormat.CURRENT_VERSION

    def add_condition(self, condition_type: str,
                     parameters: Dict[str, Any]) -> None:
        """添加条件
        
        Args:
            condition_type: 条件类型
            parameters: 条件参数
        """
        self.conditions.append({
            'type': condition_type,
            'parameters': parameters
        })

    def add_action(self, action_type: str,
                   parameters: Dict[str, Any]) -> None:
        """添加动作
        
        Args:
            action_type: 动作类型
            parameters: 动作参数
        """
        self.actions.append({
            'type': action_type,
            'parameters': parameters
        })

    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 模板数据
        """
        return {
            'template_id': self.template_id,
            'name': self.name,
            'version': str(self.version),
            'conditions': self.conditions,
            'actions': self.actions,
            'parameters': self.parameters,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RuleTemplate':
        """从字典创建模板
        
        Args:
            data: 模板数据
            
        Returns:
            RuleTemplate: 模板实例
        """
        template = cls(data['template_id'], data['name'])
        template.conditions = data['conditions']
        template.actions = data['actions']
        template.parameters = data.get('parameters', {})
        template.metadata = data.get('metadata', {})
        
        version_parts = data['version'].split('.')
        template.version = RuleVersion(
            int(version_parts[0]),
            int(version_parts[1]),
            int(version_parts[2])
        )
        
        return template

class RuleHistory:
    """规则执行历史"""
    
    def __init__(self, history_dir: str = "data/rule_history"):
        """初始化规则历史
        
        Args:
            history_dir: 历史记录目录
        """
        self.history_dir = history_dir
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
            
        self.current_history: List[Dict] = []

    def add_record(self, rule_id: str, success: bool,
                  context: Dict[str, Any]) -> None:
        """添加历史记录
        
        Args:
            rule_id: 规则ID
            success: 是否成功
            context: 执行上下文
        """
        record = {
            'rule_id': rule_id,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'context': context
        }
        
        self.current_history.append(record)
        self._save_record(record)

    def _save_record(self, record: Dict) -> None:
        """保存历史记录
        
        Args:
            record: 记录数据
        """
        try:
            # 按日期组织文件
            date_str = datetime.now().strftime("%Y-%m-%d")
            date_dir = os.path.join(self.history_dir, date_str)
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)
                
            # 保存记录
            filename = f"{record['rule_id']}_{datetime.now().strftime('%H-%M-%S')}.json"
            filepath = os.path.join(date_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            detailed_logger.error(f"保存规则历史记录失败: {str(e)}")

    def get_rule_history(self, rule_id: str,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> List[Dict]:
        """获取规则历史记录
        
        Args:
            rule_id: 规则ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[Dict]: 历史记录列表
        """
        history = []
        
        try:
            # 设置日期范围
            if not start_date:
                start_date = datetime.now() - timedelta(days=7)
            if not end_date:
                end_date = datetime.now()
                
            # 遍历日期目录
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                date_dir = os.path.join(self.history_dir, date_str)
                
                if os.path.exists(date_dir):
                    # 查找匹配的记录
                    for filename in os.listdir(date_dir):
                        if filename.startswith(f"{rule_id}_"):
                            filepath = os.path.join(date_dir, filename)
                            with open(filepath, 'r', encoding='utf-8') as f:
                                record = json.load(f)
                                history.append(record)
                                
                current_date += timedelta(days=1)
                
        except Exception as e:
            detailed_logger.error(f"获取规则历史记录失败: {str(e)}")
            
        return history

    def get_statistics(self, start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """获取历史统计信息
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'success_rate': 0.0,
            'rule_stats': {}
        }
        
        try:
            # 设置日期范围
            if not start_date:
                start_date = datetime.now() - timedelta(days=7)
            if not end_date:
                end_date = datetime.now()
                
            # 遍历所有记录
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                date_dir = os.path.join(self.history_dir, date_str)
                
                if os.path.exists(date_dir):
                    for filename in os.listdir(date_dir):
                        filepath = os.path.join(date_dir, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            record = json.load(f)
                            
                            # 更新总体统计
                            stats['total_executions'] += 1
                            if record['success']:
                                stats['successful_executions'] += 1
                            else:
                                stats['failed_executions'] += 1
                                
                            # 更新规则统计
                            rule_id = record['rule_id']
                            if rule_id not in stats['rule_stats']:
                                stats['rule_stats'][rule_id] = {
                                    'total': 0,
                                    'successful': 0,
                                    'failed': 0,
                                    'success_rate': 0.0
                                }
                                
                            rule_stats = stats['rule_stats'][rule_id]
                            rule_stats['total'] += 1
                            if record['success']:
                                rule_stats['successful'] += 1
                            else:
                                rule_stats['failed'] += 1
                                
                            rule_stats['success_rate'] = (
                                rule_stats['successful'] / rule_stats['total']
                            )
                                
                current_date += timedelta(days=1)
                
            # 计算总体成功率
            if stats['total_executions'] > 0:
                stats['success_rate'] = (
                    stats['successful_executions'] / stats['total_executions']
                )
                
        except Exception as e:
            detailed_logger.error(f"获取规则历史统计失败: {str(e)}")
            
        return stats

class RuleValidator:
    """规则验证器"""
    
    @staticmethod
    def validate_rule_file(filepath: str) -> bool:
        """验证规则文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否有效
        """
        try:
            # 读取文件
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.json'):
                    data = json.load(f)
                elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    return False
                    
            # 验证格式
            return RuleFormat.validate_schema(data)
            
        except Exception as e:
            detailed_logger.error(f"规则文件验证失败: {str(e)}")
            return False

    @staticmethod
    def validate_rule_data(data: Dict) -> bool:
        """验证规则数据
        
        Args:
            data: 规则数据
            
        Returns:
            bool: 是否有效
        """
        return RuleFormat.validate_schema(data)

class RuleManager:
    """规则管理器"""
    
    def __init__(self, rules_dir: str = "data/rules"):
        """初始化规则管理器
        
        Args:
            rules_dir: 规则目录
        """
        self.rules_dir = rules_dir
        if not os.path.exists(rules_dir):
            os.makedirs(rules_dir)
            
        self.templates: Dict[str, RuleTemplate] = {}
        self.history = RuleHistory()
        self.validator = RuleValidator()

    def load_template(self, filepath: str) -> None:
        """加载规则模板
        
        Args:
            filepath: 模板文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.json'):
                    data = json.load(f)
                elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    raise GameAutomationError("不支持的文件格式")
                    
            template = RuleTemplate.from_dict(data)
            self.templates[template.template_id] = template
            detailed_logger.info(f"加载规则模板: {template.name}")
            
        except Exception as e:
            detailed_logger.error(f"加载规则模板失败: {str(e)}")

    def save_template(self, template: RuleTemplate) -> None:
        """保存规则模板
        
        Args:
            template: 模板实例
        """
        try:
            filepath = os.path.join(
                self.rules_dir,
                f"template_{template.template_id}.yaml"
            )
            
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(template.to_dict(), f, allow_unicode=True)
                
            detailed_logger.info(f"保存规则模板: {template.name}")
            
        except Exception as e:
            detailed_logger.error(f"保存规则模板失败: {str(e)}")

    def create_rule_from_template(self, template_id: str,
                                rule_id: str,
                                parameters: Optional[Dict[str, Any]] = None) -> Dict:
        """从模板创建规则
        
        Args:
            template_id: 模板ID
            rule_id: 规则ID
            parameters: 规则参数
            
        Returns:
            Dict: 规则数据
        """
        if template_id not in self.templates:
            raise GameAutomationError(f"模板不存在: {template_id}")
            
        template = self.templates[template_id]
        
        # 创建规则数据
        rule = {
            'id': rule_id,
            'name': f"Rule from {template.name}",
            'version': str(template.version),
            'conditions': template.conditions.copy(),
            'actions': template.actions.copy(),
            'priority': 1,
            'metadata': {
                'template_id': template_id,
                'created_at': datetime.now().isoformat()
            }
        }
        
        # 更新参数
        if parameters:
            rule['parameters'] = parameters
            
        return rule

    def save_rule(self, rule: Dict) -> None:
        """保存规则
        
        Args:
            rule: 规则数据
        """
        try:
            if not self.validator.validate_rule_data({'rules': [rule]}):
                raise GameAutomationError("规则数据格式无效")
                
            filepath = os.path.join(
                self.rules_dir,
                f"rule_{rule['id']}.yaml"
            )
            
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(rule, f, allow_unicode=True)
                
            detailed_logger.info(f"保存规则: {rule['name']}")
            
        except Exception as e:
            detailed_logger.error(f"保存规则失败: {str(e)}")

    def add_execution_record(self, rule_id: str,
                           success: bool,
                           context: Dict[str, Any]) -> None:
        """添加规则执行记录
        
        Args:
            rule_id: 规则ID
            success: 是否成功
            context: 执行上下文
        """
        self.history.add_record(rule_id, success, context)

    def get_rule_statistics(self, rule_id: str,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """获取规则统计信息
        
        Args:
            rule_id: 规则ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        history = self.history.get_rule_history(rule_id, start_date, end_date)
        
        stats = {
            'rule_id': rule_id,
            'total_executions': len(history),
            'successful_executions': sum(1 for record in history if record['success']),
            'failed_executions': sum(1 for record in history if not record['success'])
        }
        
        if stats['total_executions'] > 0:
            stats['success_rate'] = (
                stats['successful_executions'] / stats['total_executions']
            )
        else:
            stats['success_rate'] = 0.0
            
        return stats
