from typing import Dict, List, Optional, Any, Callable, Sequence
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime
import asyncio
import time

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger

class StateType(Enum):
    """状态类型枚举"""
    IDLE = auto()       # 空闲状态
    ACTIVE = auto()     # 活动状态
    SUSPENDED = auto()  # 暂停状态
    COMPLETED = auto()  # 完成状态
    ERROR = auto()      # 错误状态

@dataclass
class StateContext:
    """状态上下文数据"""
    state_id: str
    state_type: StateType
    start_time: datetime
    parameters: Dict[str, Any]
    shared_data: Dict[str, Any]

@dataclass
class StateHistoryEntry:
    """状态历史记录条目"""
    state_id: str
    timestamp: datetime
    duration: Optional[float] = None
    transition_id: Optional[str] = None
    error: Optional[str] = None

class State:
    """状态基类"""
    
    def __init__(self, state_id: str, name: str, state_type: StateType):
        """初始化状态
        
        Args:
            state_id: 状态ID
            name: 状态名称
            state_type: 状态类型
        """
        self.state_id = state_id
        self.name = name
        self.state_type = state_type
        self.context: Optional[StateContext] = None
        
        self.entry_actions: List[Callable] = []
        self.exit_actions: List[Callable] = []
        self.update_actions: List[Callable] = []
        
        self.transitions: Dict[str, 'Transition'] = {}
        self.parent: Optional['State'] = None
        self.children: List['State'] = []
        
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.execution_count = 0
        self.update_count = 0

    async def enter(self, context: Dict[str, Any]) -> None:
        """进入状态
        
        Args:
            context: 执行上下文
        """
        self.start_time = datetime.now()
        self.execution_count += 1
        
        # 创建状态上下文
        self.context = StateContext(
            state_id=self.state_id,
            state_type=self.state_type,
            start_time=self.start_time,
            parameters=context.get('parameters', {}),
            shared_data=context.get('shared_data', {})
        )
        
        # 执行进入动作
        for action in self.entry_actions:
            try:
                await action(context)
            except Exception as e:
                detailed_logger.error(f"状态进入动作执行失败: {str(e)}")

    async def exit(self, context: Dict[str, Any]) -> None:
        """退出状态
        
        Args:
            context: 执行上下文
        """
        self.end_time = datetime.now()
        
        # 执行退出动作
        for action in self.exit_actions:
            try:
                await action(context)
            except Exception as e:
                detailed_logger.error(f"状态退出动作执行失败: {str(e)}")
                
        self.context = None

    async def update(self, context: Dict[str, Any]) -> None:
        """更新状态
        
        Args:
            context: 执行上下文
        """
        self.update_count += 1
        
        # 执行更新动作
        for action in self.update_actions:
            try:
                await action(context)
            except Exception as e:
                detailed_logger.error(f"状态更新动作执行失败: {str(e)}")

    def add_entry_action(self, action: Callable) -> None:
        """添加进入动作
        
        Args:
            action: 动作函数
        """
        self.entry_actions.append(action)

    def add_exit_action(self, action: Callable) -> None:
        """添加退出动作
        
        Args:
            action: 动作函数
        """
        self.exit_actions.append(action)

    def add_update_action(self, action: Callable) -> None:
        """添加更新动作
        
        Args:
            action: 动作函数
        """
        self.update_actions.append(action)

    def add_transition(self, transition: 'Transition') -> None:
        """添加状态转换
        
        Args:
            transition: 转换对象
        """
        self.transitions[transition.target_state_id] = transition

    def add_child(self, state: 'State') -> None:
        """添加子状态
        
        Args:
            state: 子状态
        """
        state.parent = self
        self.children.append(state)

    def get_statistics(self) -> Dict[str, Any]:
        """获取状态统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'state_id': self.state_id,
            'name': self.name,
            'type': self.state_type.name,
            'execution_count': self.execution_count,
            'update_count': self.update_count,
            'transitions_count': len(self.transitions),
            'children_count': len(self.children),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }

class Transition:
    """状态转换类"""
    
    def __init__(self, transition_id: str,
                 source_state_id: str,
                 target_state_id: str,
                 condition: Optional[Callable] = None):
        """初始化状态转换
        
        Args:
            transition_id: 转换ID
            source_state_id: 源状态ID
            target_state_id: 目标状态ID
            condition: 转换条件函数
        """
        self.transition_id = transition_id
        self.source_state_id = source_state_id
        self.target_state_id = target_state_id
        self.condition = condition
        
        self.actions: List[Callable] = []
        self.execution_count = 0
        self.success_count = 0

    async def can_transit(self, context: Dict[str, Any]) -> bool:
        """检查是否可以转换
        
        Args:
            context: 执行上下文
            
        Returns:
            bool: 是否可以转换
        """
        if not self.condition:
            return True
            
        try:
            return await self.condition(context)
        except Exception as e:
            detailed_logger.error(f"转换条件检查失败: {str(e)}")
            return False

    async def execute(self, context: Dict[str, Any]) -> None:
        """执行转换
        
        Args:
            context: 执行上下文
        """
        self.execution_count += 1
        
        try:
            # 执行转换动作
            for action in self.actions:
                await action(context)
            self.success_count += 1
        except Exception as e:
            detailed_logger.error(f"转换动作执行失败: {str(e)}")

    def add_action(self, action: Callable) -> None:
        """添加转换动作
        
        Args:
            action: 动作函数
        """
        self.actions.append(action)

    def get_statistics(self) -> Dict[str, Any]:
        """获取转换统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'transition_id': self.transition_id,
            'source_state': self.source_state_id,
            'target_state': self.target_state_id,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'success_rate': (
                self.success_count / self.execution_count
                if self.execution_count > 0 else 0
            )
        }

class StateMachine:
    """状态机"""
    
    def __init__(self, machine_id: str, name: str):
        """初始化状态机
        
        Args:
            machine_id: 状态机ID
            name: 状态机名称
        """
        self.machine_id = machine_id
        self.name = name
        self.states: Dict[str, State] = {}
        self.current_state: Optional[State] = None
        self.initial_state: Optional[State] = None
        self.shared_data: Dict[str, Any] = {}
        
        # 状态历史和死锁检测
        self.state_history: List[StateHistoryEntry] = []
        self.max_history_size = 100  # 最大历史记录数
        self.deadlock_check_size = 5  # 检查死锁的状态序列长度
        
        # 超时控制
        self.transition_timeout = 30  # 状态转换超时时间(秒)
        self.last_transition_time = time.time()

    def _is_repeating_sequence(self, sequence: Sequence[str]) -> bool:
        """检查状态序列是否重复
        
        Args:
            sequence: 状态序列
            
        Returns:
            bool: 是否重复
        """
        if len(sequence) < 2 * self.deadlock_check_size:
            return False
            
        # 检查最近的状态序列是否形成循环
        recent = sequence[-self.deadlock_check_size:]
        previous = sequence[-2*self.deadlock_check_size:-self.deadlock_check_size]
        return recent == previous

    async def _check_deadlock(self) -> bool:
        """检查是否出现死锁
        
        Returns:
            bool: 是否死锁
        """
        if len(self.state_history) >= 2 * self.deadlock_check_size:
            recent_states = [entry.state_id for entry in self.state_history]
            if self._is_repeating_sequence(recent_states):
                detailed_logger.warning(f"检测到状态机死锁: {self.machine_id}")
                return True
        return False

    def _add_history_entry(self, state_id: str, 
                          transition_id: Optional[str] = None,
                          error: Optional[str] = None) -> None:
        """添加历史记录
        
        Args:
            state_id: 状态ID
            transition_id: 转换ID
            error: 错误信息
        """
        current_time = datetime.now()
        
        # 计算上一个状态的持续时间
        if self.state_history:
            last_entry = self.state_history[-1]
            if not last_entry.duration:
                last_entry.duration = (
                    current_time - last_entry.timestamp
                ).total_seconds()
        
        # 添加新记录
        entry = StateHistoryEntry(
            state_id=state_id,
            timestamp=current_time,
            transition_id=transition_id,
            error=error
        )
        self.state_history.append(entry)
        
        # 限制历史记录大小
        if len(self.state_history) > self.max_history_size:
            self.state_history = self.state_history[-self.max_history_size:]

    def add_state(self, state: State, is_initial: bool = False) -> None:
        """添加状态
        
        Args:
            state: 状态对象
            is_initial: 是否为初始状态
        """
        self.states[state.state_id] = state
        if is_initial:
            self.initial_state = state

    async def start(self, context: Optional[Dict[str, Any]] = None) -> None:
        """启动状态机
        
        Args:
            context: 执行上下文
        """
        if not self.initial_state:
            raise GameAutomationError("未设置初始状态")
            
        # 合并上下文
        full_context = {
            'shared_data': self.shared_data
        }
        if context:
            full_context.update(context)
            
        # 进入初始状态
        self.current_state = self.initial_state
        await self.current_state.enter(full_context)
        self._add_history_entry(self.current_state.state_id)

    async def update(self, context: Optional[Dict[str, Any]] = None) -> None:
        """更新状态机
        
        Args:
            context: 执行上下文
        """
        if not self.current_state:
            return
            
        # 检查死锁
        if await self._check_deadlock():
            self._add_history_entry(
                self.current_state.state_id,
                error="Deadlock detected"
            )
            return
            
        # 合并上下文
        full_context = {
            'shared_data': self.shared_data
        }
        if context:
            full_context.update(context)
            
        try:
            # 更新当前状态
            await self.current_state.update(full_context)
            
            # 检查转换
            async with asyncio.timeout(self.transition_timeout):
                for transition in self.current_state.transitions.values():
                    if await transition.can_transit(full_context):
                        # 执行转换
                        await transition.execute(full_context)
                        
                        # 退出当前状态
                        await self.current_state.exit(full_context)
                        
                        # 记录转换
                        self._add_history_entry(
                            self.current_state.state_id,
                            transition.transition_id
                        )
                        
                        # 进入目标状态
                        target_state = self.states[transition.target_state_id]
                        self.current_state = target_state
                        await self.current_state.enter(full_context)
                        self._add_history_entry(self.current_state.state_id)
                        break
                        
        except asyncio.TimeoutError:
            detailed_logger.error(f"状态转换超时: {self.current_state.state_id}")
            self._add_history_entry(
                self.current_state.state_id,
                error="Transition timeout"
            )
        except Exception as e:
            detailed_logger.error(f"状态机更新失败: {str(e)}")
            self._add_history_entry(
                self.current_state.state_id,
                error=str(e)
            )

    def get_current_state(self) -> Optional[State]:
        """获取当前状态
        
        Returns:
            Optional[State]: 当前状态
        """
        return self.current_state

    def get_state_history(self) -> List[StateHistoryEntry]:
        """获取状态历史记录
        
        Returns:
            List[StateHistoryEntry]: 历史记录列表
        """
        return self.state_history.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """获取状态机统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            'machine_id': self.machine_id,
            'name': self.name,
            'total_states': len(self.states),
            'current_state': self.current_state.state_id if self.current_state else None,
            'states': [state.get_statistics() for state in self.states.values()],
            'history_stats': {
                'total_entries': len(self.state_history),
                'unique_states': len(set(entry.state_id for entry in self.state_history)),
                'transitions': len([e for e in self.state_history if e.transition_id]),
                'errors': len([e for e in self.state_history if e.error])
            }
        }
        
        # 收集转换统计
        transitions_stats = []
        for state in self.states.values():
            for transition in state.transitions.values():
                transitions_stats.append(transition.get_statistics())
        stats['transitions'] = transitions_stats
        
        return stats

class StateMachineBuilder:
    """状态机构建器"""
    
    def __init__(self, machine_id: str, name: str):
        """初始化构建器
        
        Args:
            machine_id: 状态机ID
            name: 状态机名称
        """
        self.machine = StateMachine(machine_id, name)
        self.current_state: Optional[State] = None

    def state(self, state_id: str, name: str,
             state_type: StateType = StateType.ACTIVE,
             is_initial: bool = False) -> 'StateMachineBuilder':
        """添加状态
        
        Args:
            state_id: 状态ID
            name: 状态名称
            state_type: 状态类型
            is_initial: 是否为初始状态
            
        Returns:
            StateMachineBuilder: 构建器实例
        """
        state = State(state_id, name, state_type)
        self.machine.add_state(state, is_initial)
        self.current_state = state
        return self

    def on_entry(self, action: Callable) -> 'StateMachineBuilder':
        """添加进入动作
        
        Args:
            action: 动作函数
            
        Returns:
            StateMachineBuilder: 构建器实例
        """
        if self.current_state:
            self.current_state.add_entry_action(action)
        return self

    def on_exit(self, action: Callable) -> 'StateMachineBuilder':
        """添加退出动作
        
        Args:
            action: 动作函数
            
        Returns:
            StateMachineBuilder: 构建器实例
        """
        if self.current_state:
            self.current_state.add_exit_action(action)
        return self

    def on_update(self, action: Callable) -> 'StateMachineBuilder':
        """添加更新动作
        
        Args:
            action: 动作函数
            
        Returns:
            StateMachineBuilder: 构建器实例
        """
        if self.current_state:
            self.current_state.add_update_action(action)
        return self

    def transition(self, transition_id: str,
                  target_state_id: str,
                  condition: Optional[Callable] = None) -> 'StateMachineBuilder':
        """添加转换
        
        Args:
            transition_id: 转换ID
            target_state_id: 目标状态ID
            condition: 转换条件函数
            
        Returns:
            StateMachineBuilder: 构建器实例
        """
        if self.current_state:
            transition = Transition(
                transition_id,
                self.current_state.state_id,
                target_state_id,
                condition
            )
            self.current_state.add_transition(transition)
        return self

    def on_transition(self, target_state_id: str,
                     action: Callable) -> 'StateMachineBuilder':
        """添加转换动作
        
        Args:
            target_state_id: 目标状态ID
            action: 动作函数
            
        Returns:
            StateMachineBuilder: 构建器实例
        """
        if self.current_state:
            transition = self.current_state.transitions.get(target_state_id)
            if transition:
                transition.add_action(action)
        return self

    def build(self) -> StateMachine:
        """构建状态机
        
        Returns:
            StateMachine: 状态机实例
        """
        return self.machine
