from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import json
import os
from enum import Enum, auto
import time

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger

class ContextScope(Enum):
    """上下文作用域枚举"""
    GLOBAL = auto()     # 全局作用域
    SESSION = auto()    # 会话作用域
    BEHAVIOR = auto()   # 行为作用域
    ACTION = auto()     # 动作作用域
    TEMPORARY = auto()  # 临时作用域

class ContextLifetime(Enum):
    """上下文生命周期枚举"""
    PERSISTENT = auto() # 持久化
    TRANSIENT = auto()  # 临时的
    SCOPED = auto()     # 作用域限定

@dataclass
class ContextMetadata:
    """上下文元数据"""
    created_at: datetime
    updated_at: datetime
    scope: ContextScope
    lifetime: ContextLifetime
    version: int
    tags: Set[str]

@dataclass
class VersionEntry:
    """版本记录条目"""
    version: int
    timestamp: datetime
    data: Any
    metadata: Dict[str, Any]

class ContextValidator:
    """上下文验证器"""
    
    @staticmethod
    def validate_data(data: Any, schema: Dict) -> bool:
        """验证数据格式
        
        Args:
            data: 要验证的数据
            schema: 数据模式
            
        Returns:
            bool: 是否有效
        """
        try:
            # 检查必需字段
            if 'type' in schema:
                if not isinstance(data, schema['type']):
                    return False
                    
            # 检查范围
            if 'range' in schema:
                min_val, max_val = schema['range']
                if not min_val <= data <= max_val:
                    return False
                    
            # 检查枚举值
            if 'enum' in schema:
                if data not in schema['enum']:
                    return False
                    
            # 检查自定义验证函数
            if 'validator' in schema:
                if not schema['validator'](data):
                    return False
                    
            return True
            
        except Exception:
            return False

class ContextManager:
    """上下文管理器"""
    
    def __init__(self, context_dir: str = "data/context"):
        """初始化上下文管理器
        
        Args:
            context_dir: 上下文数据目录
        """
        self.context_dir = context_dir
        if not os.path.exists(context_dir):
            os.makedirs(context_dir)
            
        self.global_context: Dict[str, Any] = {}
        self.session_context: Dict[str, Any] = {}
        self.behavior_context: Dict[str, Dict[str, Any]] = {}
        self.action_context: Dict[str, Dict[str, Any]] = {}
        self.temporary_context: Dict[str, Any] = {}
        
        self.metadata: Dict[str, ContextMetadata] = {}
        self.validator = ContextValidator()
        
        # 版本控制
        self.data_version = "1.0.0"
        self.version_history: Dict[str, List[VersionEntry]] = {}
        self.max_history_entries = 100  # 每个键最多保留的历史版本数
        
        # 加载持久化数据
        self._load_persistent_data()

    def _load_persistent_data(self) -> None:
        """加载持久化的上下文数据"""
        try:
            # 加载全局上下文
            global_file = os.path.join(self.context_dir, "global_context.json")
            if os.path.exists(global_file):
                with open(global_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if self._validate_version(data):
                        self.global_context = data['content']
                    
            # 加载会话上下文
            session_file = os.path.join(self.context_dir, "session_context.json")
            if os.path.exists(session_file):
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if self._validate_version(data):
                        self.session_context = data['content']
                    
            # 加载元数据
            metadata_file = os.path.join(self.context_dir, "context_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
                    for key, data in metadata_dict.items():
                        self.metadata[key] = ContextMetadata(
                            created_at=datetime.fromisoformat(data['created_at']),
                            updated_at=datetime.fromisoformat(data['updated_at']),
                            scope=ContextScope[data['scope']],
                            lifetime=ContextLifetime[data['lifetime']],
                            version=data['version'],
                            tags=set(data['tags'])
                        )
            
            # 加载版本历史
            history_file = os.path.join(self.context_dir, "version_history.json")
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_dict = json.load(f)
                    for key, entries in history_dict.items():
                        self.version_history[key] = [
                            VersionEntry(
                                version=entry['version'],
                                timestamp=datetime.fromisoformat(entry['timestamp']),
                                data=entry['data'],
                                metadata=entry['metadata']
                            )
                            for entry in entries
                        ]
                        
        except Exception as e:
            detailed_logger.error(f"加载上下文数据失败: {str(e)}")

    def _validate_version(self, data: Dict) -> bool:
        """验证数据版本
        
        Args:
            data: 要验证的数据
            
        Returns:
            bool: 是否有效
        """
        if not isinstance(data, dict):
            return False
            
        if 'version' not in data or 'timestamp' not in data or 'content' not in data:
            return False
            
        return True

    def _save_persistent_data(self) -> None:
        """保存持久化的上下文数据"""
        try:
            # 准备版本化的数据
            current_time = datetime.now().isoformat()
            
            # 保存全局上下文
            global_data = {
                'version': self.data_version,
                'timestamp': current_time,
                'content': self.global_context
            }
            global_file = os.path.join(self.context_dir, "global_context.json")
            with open(global_file, 'w', encoding='utf-8') as f:
                json.dump(global_data, f, indent=2, ensure_ascii=False)
                
            # 保存会话上下文
            session_data = {
                'version': self.data_version,
                'timestamp': current_time,
                'content': self.session_context
            }
            session_file = os.path.join(self.context_dir, "session_context.json")
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
                
            # 保存元数据
            metadata_dict = {
                key: {
                    'created_at': meta.created_at.isoformat(),
                    'updated_at': meta.updated_at.isoformat(),
                    'scope': meta.scope.name,
                    'lifetime': meta.lifetime.name,
                    'version': meta.version,
                    'tags': list(meta.tags)
                }
                for key, meta in self.metadata.items()
            }
            metadata_file = os.path.join(self.context_dir, "context_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
                
            # 保存版本历史
            history_dict = {
                key: [
                    {
                        'version': entry.version,
                        'timestamp': entry.timestamp.isoformat(),
                        'data': entry.data,
                        'metadata': entry.metadata
                    }
                    for entry in entries
                ]
                for key, entries in self.version_history.items()
            }
            history_file = os.path.join(self.context_dir, "version_history.json")
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_dict, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            detailed_logger.error(f"保存上下文数据失败: {str(e)}")

    def _add_version_entry(self, key: str, value: Any, metadata: Dict[str, Any]) -> None:
        """添加版本历史记录
        
        Args:
            key: 键名
            value: 值
            metadata: 元数据
        """
        if key not in self.version_history:
            self.version_history[key] = []
            
        entry = VersionEntry(
            version=metadata.get('version', 1),
            timestamp=datetime.now(),
            data=value,
            metadata=metadata
        )
        
        self.version_history[key].append(entry)
        
        # 限制历史记录数量
        if len(self.version_history[key]) > self.max_history_entries:
            self.version_history[key] = self.version_history[key][-self.max_history_entries:]

    def get_version_history(self, key: str) -> List[VersionEntry]:
        """获取键的版本历史
        
        Args:
            key: 键名
            
        Returns:
            List[VersionEntry]: 版本历史记录
        """
        return self.version_history.get(key, [])

    def rollback_to_version(self, key: str, version: int) -> bool:
        """回滚到指定版本
        
        Args:
            key: 键名
            version: 目标版本号
            
        Returns:
            bool: 是否成功
        """
        entries = self.version_history.get(key, [])
        for entry in entries:
            if entry.version == version:
                self.set_value(key, entry.data)
                return True
        return False

    def set_value(self, key: str, value: Any,
                 scope: ContextScope = ContextScope.GLOBAL,
                 lifetime: ContextLifetime = ContextLifetime.PERSISTENT,
                 schema: Optional[Dict] = None,
                 tags: Optional[Set[str]] = None) -> None:
        """设置上下文值
        
        Args:
            key: 键名
            value: 值
            scope: 作用域
            lifetime: 生命周期
            schema: 数据模式
            tags: 标签集合
        """
        # 验证数据
        if schema and not self.validator.validate_data(value, schema):
            raise GameAutomationError(f"数据验证失败: {key}")
            
        # 根据作用域选择上下文
        context = self._get_context_by_scope(scope)
        if not context:
            raise GameAutomationError(f"无效的作用域: {scope}")
            
        # 更新或创建元数据
        current_time = datetime.now()
        if key in self.metadata:
            metadata = self.metadata[key]
            metadata.updated_at = current_time
            metadata.version += 1
            if tags:
                metadata.tags.update(tags)
        else:
            self.metadata[key] = ContextMetadata(
                created_at=current_time,
                updated_at=current_time,
                scope=scope,
                lifetime=lifetime,
                version=1,
                tags=set(tags) if tags else set()
            )
            
        # 添加版本历史记录
        metadata_dict = {
            'created_at': self.metadata[key].created_at.isoformat(),
            'updated_at': self.metadata[key].updated_at.isoformat(),
            'scope': scope.name,
            'lifetime': lifetime.name,
            'version': self.metadata[key].version,
            'tags': list(self.metadata[key].tags)
        }
        self._add_version_entry(key, value, metadata_dict)
            
        # 设置值
        context[key] = value
        
        # 持久化数据
        if lifetime == ContextLifetime.PERSISTENT:
            self._save_persistent_data()

    def get_value(self, key: str,
                 scope: Optional[ContextScope] = None,
                 default: Any = None) -> Any:
        """获取上下文值
        
        Args:
            key: 键名
            scope: 作用域
            default: 默认值
            
        Returns:
            Any: 上下文值
        """
        # 如果指定了作用域
        if scope:
            context = self._get_context_by_scope(scope)
            if context and key in context:
                return context[key]
            return default
            
        # 按优先级搜索所有作用域
        search_order = [
            ContextScope.ACTION,
            ContextScope.BEHAVIOR,
            ContextScope.SESSION,
            ContextScope.GLOBAL
        ]
        
        for search_scope in search_order:
            context = self._get_context_by_scope(search_scope)
            if context and key in context:
                return context[key]
                
        return default

    def remove_value(self, key: str,
                    scope: Optional[ContextScope] = None) -> None:
        """移除上下文值
        
        Args:
            key: 键名
            scope: 作用域
        """
        if scope:
            # 从指定作用域移除
            context = self._get_context_by_scope(scope)
            if context and key in context:
                del context[key]
                if key in self.metadata:
                    del self.metadata[key]
        else:
            # 从所有作用域移除
            for context_scope in ContextScope:
                context = self._get_context_by_scope(context_scope)
                if context and key in context:
                    del context[key]
                    
            if key in self.metadata:
                del self.metadata[key]
                
        # 更新持久化数据
        self._save_persistent_data()

    def clear_scope(self, scope: ContextScope) -> None:
        """清空指定作用域的上下文
        
        Args:
            scope: 作用域
        """
        context = self._get_context_by_scope(scope)
        if context:
            # 清除数据
            context.clear()
            
            # 清除相关元数据
            self.metadata = {
                key: meta for key, meta in self.metadata.items()
                if meta.scope != scope
            }
            
            # 更新持久化数据
            if scope in [ContextScope.GLOBAL, ContextScope.SESSION]:
                self._save_persistent_data()

    def _get_context_by_scope(self, scope: ContextScope) -> Optional[Dict]:
        """根据作用域获取上下文
        
        Args:
            scope: 作用域
            
        Returns:
            Optional[Dict]: 上下文字典
        """
        if scope == ContextScope.GLOBAL:
            return self.global_context
        elif scope == ContextScope.SESSION:
            return self.session_context
        elif scope == ContextScope.BEHAVIOR:
            return self.behavior_context
        elif scope == ContextScope.ACTION:
            return self.action_context
        elif scope == ContextScope.TEMPORARY:
            return self.temporary_context
        return None

    def create_behavior_context(self, behavior_id: str) -> None:
        """创建行为上下文
        
        Args:
            behavior_id: 行为ID
        """
        if behavior_id not in self.behavior_context:
            self.behavior_context[behavior_id] = {}

    def get_behavior_context(self, behavior_id: str) -> Dict[str, Any]:
        """获取行为上下文
        
        Args:
            behavior_id: 行为ID
            
        Returns:
            Dict[str, Any]: 行为上下文
        """
        return self.behavior_context.get(behavior_id, {})

    def remove_behavior_context(self, behavior_id: str) -> None:
        """移除行为上下文
        
        Args:
            behavior_id: 行为ID
        """
        if behavior_id in self.behavior_context:
            del self.behavior_context[behavior_id]

    def create_action_context(self, action_id: str) -> None:
        """创建动作上下文
        
        Args:
            action_id: 动作ID
        """
        if action_id not in self.action_context:
            self.action_context[action_id] = {}

    def get_action_context(self, action_id: str) -> Dict[str, Any]:
        """获取动作上下文
        
        Args:
            action_id: 动作ID
            
        Returns:
            Dict[str, Any]: 动作上下文
        """
        return self.action_context.get(action_id, {})

    def remove_action_context(self, action_id: str) -> None:
        """移除动作上下文
        
        Args:
            action_id: 动作ID
        """
        if action_id in self.action_context:
            del self.action_context[action_id]

    def get_metadata(self, key: str) -> Optional[ContextMetadata]:
        """获取上下文元数据
        
        Args:
            key: 键名
            
        Returns:
            Optional[ContextMetadata]: 元数据
        """
        return self.metadata.get(key)

    def find_by_tag(self, tag: str) -> Dict[str, Any]:
        """根据标签查找上下文值
        
        Args:
            tag: 标签
            
        Returns:
            Dict[str, Any]: 匹配的上下文值
        """
        result = {}
        for key, meta in self.metadata.items():
            if tag in meta.tags:
                value = self.get_value(key)
                if value is not None:
                    result[key] = value
        return result

    def cleanup_expired(self) -> None:
        """清理过期的上下文数据"""
        # 清理临时上下文
        self.temporary_context.clear()
        
        # 清理过期的行为上下文
        expired_behaviors = []
        for behavior_id in self.behavior_context:
            if not self._is_behavior_active(behavior_id):
                expired_behaviors.append(behavior_id)
        for behavior_id in expired_behaviors:
            self.remove_behavior_context(behavior_id)
            
        # 清理过期的动作上下文
        expired_actions = []
        for action_id in self.action_context:
            if not self._is_action_active(action_id):
                expired_actions.append(action_id)
        for action_id in expired_actions:
            self.remove_action_context(action_id)

    def _is_behavior_active(self, behavior_id: str) -> bool:
        """检查行为是否活动
        
        Args:
            behavior_id: 行为ID
            
        Returns:
            bool: 是否活动
        """
        # TODO: 实现行为活动状态检查
        return False

    def _is_action_active(self, action_id: str) -> bool:
        """检查动作是否活动
        
        Args:
            action_id: 动作ID
            
        Returns:
            bool: 是否活动
        """
        # TODO: 实现动作活动状态检查
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """获取上下文统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            'total_keys': len(self.metadata),
            'scope_stats': {
                scope.name: 0 for scope in ContextScope
            },
            'lifetime_stats': {
                lifetime.name: 0 for lifetime in ContextLifetime
            },
            'version_stats': {
                'min': float('inf'),
                'max': 0,
                'avg': 0
            },
            'tag_stats': {},
            'history_stats': {
                'total_entries': sum(len(entries) for entries in self.version_history.values()),
                'keys_with_history': len(self.version_history),
                'avg_versions_per_key': 0
            }
        }
        
        # 统计各项指标
        total_version = 0
        for meta in self.metadata.values():
            # 作用域统计
            stats['scope_stats'][meta.scope.name] += 1
            
            # 生命周期统计
            stats['lifetime_stats'][meta.lifetime.name] += 1
            
            # 版本统计
            stats['version_stats']['min'] = min(
                stats['version_stats']['min'],
                meta.version
            )
            stats['version_stats']['max'] = max(
                stats['version_stats']['max'],
                meta.version
            )
            total_version += meta.version
            
            # 标签统计
            for tag in meta.tags:
                stats['tag_stats'][tag] = stats['tag_stats'].get(tag, 0) + 1
                
        # 计算平均版本
        if self.metadata:
            stats['version_stats']['avg'] = total_version / len(self.metadata)
            if stats['history_stats']['keys_with_history'] > 0:
                stats['history_stats']['avg_versions_per_key'] = (
                    stats['history_stats']['total_entries'] /
                    stats['history_stats']['keys_with_history']
                )
        else:
            stats['version_stats']['min'] = 0
            
        return stats
