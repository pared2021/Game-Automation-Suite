from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger

class NodeStatus(Enum):
    """节点状态枚举"""
    READY = auto()      # 准备执行
    RUNNING = auto()    # 正在执行
    SUCCESS = auto()    # 执行成功
    FAILURE = auto()    # 执行失败
    INVALID = auto()    # 无效状态

class NodeType(Enum):
    """节点类型枚举"""
    SEQUENCE = auto()   # 顺序节点
    SELECTOR = auto()   # 选择节点
    PARALLEL = auto()   # 并行节点
    DECORATOR = auto()  # 装饰器节点
    ACTION = auto()     # 动作节点
    CONDITION = auto()  # 条件节点

@dataclass
class NodeContext:
    """节点上下文数据"""
    node_id: str
    node_type: NodeType
    start_time: datetime
    parameters: Dict[str, Any]
    shared_data: Dict[str, Any]

class TreeNode:
    """行为树节点基类"""
    
    def __init__(self, node_id: str, name: str):
        """初始化节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
        """
        self.node_id = node_id
        self.name = name
        self.status = NodeStatus.READY
        self.parent: Optional[TreeNode] = None
        self.context: Optional[NodeContext] = None
        
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.execution_count = 0
        self.success_count = 0

    async def execute(self, context: Dict[str, Any]) -> NodeStatus:
        """执行节点
        
        Args:
            context: 执行上下文
            
        Returns:
            NodeStatus: 节点状态
        """
        try:
            self.start_time = datetime.now()
            self.status = NodeStatus.RUNNING
            self.execution_count += 1
            
            # 创建节点上下文
            self.context = NodeContext(
                node_id=self.node_id,
                node_type=self.get_node_type(),
                start_time=self.start_time,
                parameters=context.get('parameters', {}),
                shared_data=context.get('shared_data', {})
            )
            
            # 执行节点逻辑
            status = await self._execute(context)
            
            self.status = status
            self.end_time = datetime.now()
            
            if status == NodeStatus.SUCCESS:
                self.success_count += 1
                
            return status
            
        except Exception as e:
            detailed_logger.error(f"节点执行失败: {str(e)}")
            self.status = NodeStatus.FAILURE
            self.end_time = datetime.now()
            return NodeStatus.FAILURE

    async def _execute(self, context: Dict[str, Any]) -> NodeStatus:
        """实际的节点执行逻辑，子类需要实现此方法
        
        Args:
            context: 执行上下文
            
        Returns:
            NodeStatus: 节点状态
        """
        raise NotImplementedError("TreeNode._execute() must be implemented by subclass")

    def get_node_type(self) -> NodeType:
        """获取节点类型
        
        Returns:
            NodeType: 节点类型
        """
        raise NotImplementedError("TreeNode.get_node_type() must be implemented by subclass")

    def reset(self) -> None:
        """重置节点状态"""
        self.status = NodeStatus.READY
        self.context = None
        self.start_time = None
        self.end_time = None

class CompositeNode(TreeNode):
    """组合节点基类"""
    
    def __init__(self, node_id: str, name: str):
        """初始化组合节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
        """
        super().__init__(node_id, name)
        self.children: List[TreeNode] = []

    def add_child(self, child: TreeNode) -> None:
        """添加子节点
        
        Args:
            child: 子节点
        """
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: TreeNode) -> None:
        """移除子节点
        
        Args:
            child: 子节点
        """
        if child in self.children:
            child.parent = None
            self.children.remove(child)

class SequenceNode(CompositeNode):
    """顺序节点：按顺序执行子节点，直到一个失败或全部成功"""
    
    def get_node_type(self) -> NodeType:
        return NodeType.SEQUENCE

    async def _execute(self, context: Dict[str, Any]) -> NodeStatus:
        """执行所有子节点，直到一个失败或全部成功
        
        Args:
            context: 执行上下文
            
        Returns:
            NodeStatus: 节点状态
        """
        for child in self.children:
            status = await child.execute(context)
            if status != NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS

class SelectorNode(CompositeNode):
    """选择节点：按顺序执行子节点，直到一个成功或全部失败"""
    
    def get_node_type(self) -> NodeType:
        return NodeType.SELECTOR

    async def _execute(self, context: Dict[str, Any]) -> NodeStatus:
        """执行子节点直到一个成功或全部失败
        
        Args:
            context: 执行上下文
            
        Returns:
            NodeStatus: 节点状态
        """
        for child in self.children:
            status = await child.execute(context)
            if status == NodeStatus.SUCCESS:
                return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

class ParallelNode(CompositeNode):
    """并行节点：同时执行所有子节点"""
    
    def __init__(self, node_id: str, name: str, success_threshold: float = 0.7):
        """初始化并行节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            success_threshold: 成功阈值(0-1)
        """
        super().__init__(node_id, name)
        self.success_threshold = success_threshold

    def get_node_type(self) -> NodeType:
        return NodeType.PARALLEL

    async def _execute(self, context: Dict[str, Any]) -> NodeStatus:
        """同时执行所有子节点
        
        Args:
            context: 执行上下文
            
        Returns:
            NodeStatus: 节点状态
        """
        if not self.children:
            return NodeStatus.SUCCESS
            
        # 执行所有子节点
        results = await asyncio.gather(
            *[child.execute(context) for child in self.children],
            return_exceptions=True
        )
        
        # 计算成功率
        success_count = sum(
            1 for result in results
            if isinstance(result, NodeStatus) and result == NodeStatus.SUCCESS
        )
        success_rate = success_count / len(self.children)
        
        return (
            NodeStatus.SUCCESS
            if success_rate >= self.success_threshold
            else NodeStatus.FAILURE
        )

class DecoratorNode(TreeNode):
    """装饰器节点基类"""
    
    def __init__(self, node_id: str, name: str, child: Optional[TreeNode] = None):
        """初始化装饰器节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            child: 子节点
        """
        super().__init__(node_id, name)
        self.child = child
        if child:
            child.parent = self

    def get_node_type(self) -> NodeType:
        return NodeType.DECORATOR

class InverterNode(DecoratorNode):
    """反转节点：反转子节点的执行结果"""
    
    async def _execute(self, context: Dict[str, Any]) -> NodeStatus:
        """执行子节点并反转结果
        
        Args:
            context: 执行上下文
            
        Returns:
            NodeStatus: 节点状态
        """
        if not self.child:
            return NodeStatus.FAILURE
            
        status = await self.child.execute(context)
        
        if status == NodeStatus.SUCCESS:
            return NodeStatus.FAILURE
        elif status == NodeStatus.FAILURE:
            return NodeStatus.SUCCESS
        return status

class RepeatNode(DecoratorNode):
    """重复节点：重复执行子节点指定次数"""
    
    def __init__(self, node_id: str, name: str, child: Optional[TreeNode] = None,
                 repeat_count: int = 1):
        """初始化重复节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            child: 子节点
            repeat_count: 重复次数
        """
        super().__init__(node_id, name, child)
        self.repeat_count = repeat_count

    async def _execute(self, context: Dict[str, Any]) -> NodeStatus:
        """重复执行子节点
        
        Args:
            context: 执行上下文
            
        Returns:
            NodeStatus: 节点状态
        """
        if not self.child:
            return NodeStatus.FAILURE
            
        for _ in range(self.repeat_count):
            status = await self.child.execute(context)
            if status != NodeStatus.SUCCESS:
                return status
                
        return NodeStatus.SUCCESS

class RetryNode(DecoratorNode):
    """重试节点：失败时重试子节点"""
    
    def __init__(self, node_id: str, name: str, child: Optional[TreeNode] = None,
                 retry_count: int = 3):
        """初始化重试节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            child: 子节点
            retry_count: 重试次数
        """
        super().__init__(node_id, name, child)
        self.retry_count = retry_count

    async def _execute(self, context: Dict[str, Any]) -> NodeStatus:
        """执行子节点，失败时重试
        
        Args:
            context: 执行上下文
            
        Returns:
            NodeStatus: 节点状态
        """
        if not self.child:
            return NodeStatus.FAILURE
            
        for _ in range(self.retry_count):
            status = await self.child.execute(context)
            if status == NodeStatus.SUCCESS:
                return NodeStatus.SUCCESS
                
        return NodeStatus.FAILURE

class ActionNode(TreeNode):
    """动作节点：执行具体动作"""
    
    def __init__(self, node_id: str, name: str, action_func: Callable):
        """初始化动作节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            action_func: 动作函数
        """
        super().__init__(node_id, name)
        self.action_func = action_func

    def get_node_type(self) -> NodeType:
        return NodeType.ACTION

    async def _execute(self, context: Dict[str, Any]) -> NodeStatus:
        """执行动作
        
        Args:
            context: 执行上下文
            
        Returns:
            NodeStatus: 节点状态
        """
        try:
            result = await self.action_func(context)
            return NodeStatus.SUCCESS if result else NodeStatus.FAILURE
        except Exception as e:
            detailed_logger.error(f"动作执行失败: {str(e)}")
            return NodeStatus.FAILURE

class ConditionNode(TreeNode):
    """条件节点：检查条件"""
    
    def __init__(self, node_id: str, name: str, condition_func: Callable):
        """初始化条件节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            condition_func: 条件函数
        """
        super().__init__(node_id, name)
        self.condition_func = condition_func

    def get_node_type(self) -> NodeType:
        return NodeType.CONDITION

    async def _execute(self, context: Dict[str, Any]) -> NodeStatus:
        """检查条件
        
        Args:
            context: 执行上下文
            
        Returns:
            NodeStatus: 节点状态
        """
        try:
            result = await self.condition_func(context)
            return NodeStatus.SUCCESS if result else NodeStatus.FAILURE
        except Exception as e:
            detailed_logger.error(f"条件检查失败: {str(e)}")
            return NodeStatus.FAILURE

class BehaviorTree:
    """行为树"""
    
    def __init__(self, tree_id: str, name: str):
        """初始化行为树
        
        Args:
            tree_id: 树ID
            name: 树名称
        """
        self.tree_id = tree_id
        self.name = name
        self.root: Optional[TreeNode] = None
        self.current_node: Optional[TreeNode] = None
        self.shared_data: Dict[str, Any] = {}

    def set_root(self, node: TreeNode) -> None:
        """设置根节点
        
        Args:
            node: 根节点
        """
        self.root = node

    async def execute(self, context: Optional[Dict[str, Any]] = None) -> NodeStatus:
        """执行行为树
        
        Args:
            context: 执行上下文
            
        Returns:
            NodeStatus: 执行状态
        """
        if not self.root:
            return NodeStatus.INVALID
            
        # 合并上下文
        full_context = {
            'shared_data': self.shared_data
        }
        if context:
            full_context.update(context)
            
        try:
            # 重置所有节点
            self._reset_all_nodes(self.root)
            
            # 执行根节点
            return await self.root.execute(full_context)
            
        except Exception as e:
            detailed_logger.error(f"行为树执行失败: {str(e)}")
            return NodeStatus.FAILURE

    def _reset_all_nodes(self, node: TreeNode) -> None:
        """重置所有节点状态
        
        Args:
            node: 当前节点
        """
        node.reset()
        
        if isinstance(node, CompositeNode):
            for child in node.children:
                self._reset_all_nodes(child)
        elif isinstance(node, DecoratorNode) and node.child:
            self._reset_all_nodes(node.child)

    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态
        
        Returns:
            Dict[str, Any]: 状态信息
        """
        return {
            'tree_id': self.tree_id,
            'name': self.name,
            'current_node': self.current_node.node_id if self.current_node else None,
            'shared_data': self.shared_data
        }

class BehaviorTreeBuilder:
    """行为树构建器"""
    
    def __init__(self, tree_id: str, name: str):
        """初始化构建器
        
        Args:
            tree_id: 树ID
            name: 树名称
        """
        self.tree = BehaviorTree(tree_id, name)
        self.current_node: Optional[TreeNode] = None
        self.node_stack: List[TreeNode] = []

    def sequence(self, node_id: str, name: str) -> 'BehaviorTreeBuilder':
        """添加顺序节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            
        Returns:
            BehaviorTreeBuilder: 构建器实例
        """
        node = SequenceNode(node_id, name)
        self._add_node(node)
        self.node_stack.append(node)
        return self

    def selector(self, node_id: str, name: str) -> 'BehaviorTreeBuilder':
        """添加选择节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            
        Returns:
            BehaviorTreeBuilder: 构建器实例
        """
        node = SelectorNode(node_id, name)
        self._add_node(node)
        self.node_stack.append(node)
        return self

    def parallel(self, node_id: str, name: str,
                success_threshold: float = 0.7) -> 'BehaviorTreeBuilder':
        """添加并行节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            success_threshold: 成功阈值
            
        Returns:
            BehaviorTreeBuilder: 构建器实例
        """
        node = ParallelNode(node_id, name, success_threshold)
        self._add_node(node)
        self.node_stack.append(node)
        return self

    def inverter(self, node_id: str, name: str) -> 'BehaviorTreeBuilder':
        """添加反转节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            
        Returns:
            BehaviorTreeBuilder: 构建器实例
        """
        node = InverterNode(node_id, name)
        self._add_node(node)
        self.node_stack.append(node)
        return self

    def repeat(self, node_id: str, name: str,
              repeat_count: int = 1) -> 'BehaviorTreeBuilder':
        """添加重复节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            repeat_count: 重复次数
            
        Returns:
            BehaviorTreeBuilder: 构建器实例
        """
        node = RepeatNode(node_id, name, repeat_count=repeat_count)
        self._add_node(node)
        self.node_stack.append(node)
        return self

    def retry(self, node_id: str, name: str,
             retry_count: int = 3) -> 'BehaviorTreeBuilder':
        """添加重试节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            retry_count: 重试次数
            
        Returns:
            BehaviorTreeBuilder: 构建器实例
        """
        node = RetryNode(node_id, name, retry_count=retry_count)
        self._add_node(node)
        self.node_stack.append(node)
        return self

    def action(self, node_id: str, name: str,
              action_func: Callable) -> 'BehaviorTreeBuilder':
        """添加动作节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            action_func: 动作函数
            
        Returns:
            BehaviorTreeBuilder: 构建器实例
        """
        node = ActionNode(node_id, name, action_func)
        self._add_node(node)
        return self

    def condition(self, node_id: str, name: str,
                 condition_func: Callable) -> 'BehaviorTreeBuilder':
        """添加条件节点
        
        Args:
            node_id: 节点ID
            name: 节点名称
            condition_func: 条件函数
            
        Returns:
            BehaviorTreeBuilder: 构建器实例
        """
        node = ConditionNode(node_id, name, condition_func)
        self._add_node(node)
        return self

    def end(self) -> 'BehaviorTreeBuilder':
        """结束当前节点
        
        Returns:
            BehaviorTreeBuilder: 构建器实例
        """
        if self.node_stack:
            self.node_stack.pop()
        return self

    def build(self) -> BehaviorTree:
        """构建行为树
        
        Returns:
            BehaviorTree: 行为树实例
        """
        return self.tree

    def _add_node(self, node: TreeNode) -> None:
        """添加节点
        
        Args:
            node: 节点实例
        """
        if not self.tree.root:
            self.tree.root = node
            self.current_node = node
        elif self.node_stack:
            parent = self.node_stack[-1]
            if isinstance(parent, CompositeNode):
                parent.add_child(node)
            elif isinstance(parent, DecoratorNode):
                parent.child = node
                node.parent = parent
        self.current_node = node
