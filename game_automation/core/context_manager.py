from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json
import asyncio
import logging
from datetime import datetime
from enum import Enum, auto

class GameState(Enum):
    """游戏状态"""
    UNKNOWN = auto()    # 未知
    LOADING = auto()    # 加载中
    LOGIN = auto()      # 登录
    MAIN = auto()       # 主界面
    BATTLE = auto()     # 战斗
    MENU = auto()       # 菜单
    EVENT = auto()      # 活动
    ERROR = auto()      # 错误

class ContextError(Exception):
    """上下文错误"""
    pass

class EventListener:
    """事件监听器"""
    def __init__(
        self,
        event_type: str,
        callback: Callable,
        priority: int = 0
    ):
        self.event_type = event_type
        self.callback = callback
        self.priority = priority

class ContextManager:
    """上下文管理器"""
    
    def __init__(self):
        self._initialized = False
        self._config = None
        self._state = GameState.UNKNOWN
        self._variables: Dict[str, Any] = {}
        self._listeners: Dict[str, List[EventListener]] = {}
        self._history: List[Dict] = []
        
    async def initialize(self):
        """初始化管理器"""
        if not self._initialized:
            try:
                # 加载配置
                config_path = Path("config/config.json")
                if not config_path.exists():
                    raise ContextError("配置文件不存在")
                    
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                    
                self._initialized = True
                
            except Exception as e:
                raise ContextError(f"初始化失败: {str(e)}")
                
    async def cleanup(self):
        """清理资源"""
        if self._initialized:
            # 保存历史记录
            await self._save_history()
            
            # 清理数据
            self._variables.clear()
            self._listeners.clear()
            self._history.clear()
            self._initialized = False
            
    async def _save_history(self):
        """保存历史记录"""
        try:
            if not self._history:
                return
                
            # 创建历史记录目录
            history_dir = Path(self._config["paths"]["logs"]) / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_path = history_dir / f"context_history_{timestamp}.json"
            
            # 保存历史记录
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"保存历史记录失败: {str(e)}")
            
    async def set_state(self, state: GameState):
        """设置游戏状态
        
        Args:
            state: 游戏状态
        """
        if not self._initialized:
            raise ContextError("未初始化")
            
        old_state = self._state
        self._state = state
        
        # 记录状态变化
        self._history.append({
            'type': 'state_change',
            'old_state': old_state.name,
            'new_state': state.name,
            'timestamp': datetime.now().isoformat()
        })
        
        # 触发事件
        await self.emit_event('state_changed', {
            'old_state': old_state,
            'new_state': state
        })
        
    async def get_state(self) -> GameState:
        """获取游戏状态
        
        Returns:
            GameState: 游戏状态
        """
        if not self._initialized:
            raise ContextError("未初始化")
            
        return self._state
        
    async def set_variable(self, name: str, value: Any):
        """设置变量
        
        Args:
            name: 变量名
            value: 变量值
        """
        if not self._initialized:
            raise ContextError("未初始化")
            
        old_value = self._variables.get(name)
        self._variables[name] = value
        
        # 记录变量变化
        self._history.append({
            'type': 'variable_change',
            'name': name,
            'old_value': old_value,
            'new_value': value,
            'timestamp': datetime.now().isoformat()
        })
        
        # 触发事件
        await self.emit_event('variable_changed', {
            'name': name,
            'old_value': old_value,
            'new_value': value
        })
        
    async def get_variable(
        self,
        name: str,
        default: Any = None
    ) -> Any:
        """获取变量
        
        Args:
            name: 变量名
            default: 默认值
            
        Returns:
            Any: 变量值
        """
        if not self._initialized:
            raise ContextError("未初始化")
            
        return self._variables.get(name, default)
        
    async def delete_variable(self, name: str):
        """删除变量
        
        Args:
            name: 变量名
        """
        if not self._initialized:
            raise ContextError("未初始化")
            
        if name in self._variables:
            old_value = self._variables[name]
            del self._variables[name]
            
            # 记录变量删除
            self._history.append({
                'type': 'variable_delete',
                'name': name,
                'old_value': old_value,
                'timestamp': datetime.now().isoformat()
            })
            
            # 触发事件
            await self.emit_event('variable_deleted', {
                'name': name,
                'old_value': old_value
            })
            
    async def get_config(self, key: str = None) -> Any:
        """获取配置
        
        Args:
            key: 配置键
            
        Returns:
            Any: 配置值
        """
        if not self._initialized:
            raise ContextError("未初始化")
            
        if key is None:
            return self._config
            
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                return None
            value = value[k]
            
        return value
        
    def add_listener(
        self,
        event_type: str,
        callback: Callable,
        priority: int = 0
    ):
        """添加事件监听器
        
        Args:
            event_type: 事件类型
            callback: 回调函数
            priority: 优先级
        """
        if event_type not in self._listeners:
            self._listeners[event_type] = []
            
        listener = EventListener(event_type, callback, priority)
        self._listeners[event_type].append(listener)
        
        # 按优先级排序
        self._listeners[event_type].sort(key=lambda x: x.priority, reverse=True)
        
    def remove_listener(
        self,
        event_type: str,
        callback: Callable
    ):
        """移除事件监听器
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type not in self._listeners:
            return
            
        self._listeners[event_type] = [
            listener
            for listener in self._listeners[event_type]
            if listener.callback != callback
        ]
        
    async def emit_event(
        self,
        event_type: str,
        data: Dict = None
    ):
        """触发事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        if not self._initialized:
            raise ContextError("未初始化")
            
        if event_type not in self._listeners:
            return
            
        # 记录事件
        self._history.append({
            'type': 'event',
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
        
        # 调用监听器
        for listener in self._listeners[event_type]:
            try:
                await listener.callback(data)
            except Exception as e:
                logging.error(f"事件处理失败: {str(e)}")
                
    async def get_history(
        self,
        event_type: str = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[Dict]:
        """获取历史记录
        
        Args:
            event_type: 事件类型
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[Dict]: 历史记录
        """
        if not self._initialized:
            raise ContextError("未初始化")
            
        records = self._history
        
        if event_type:
            records = [
                record
                for record in records
                if record['type'] == event_type
            ]
            
        if start_time:
            records = [
                record
                for record in records
                if datetime.fromisoformat(record['timestamp']) >= start_time
            ]
            
        if end_time:
            records = [
                record
                for record in records
                if datetime.fromisoformat(record['timestamp']) <= end_time
            ]
            
        return records
