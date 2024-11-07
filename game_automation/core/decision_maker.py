import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from ..core.scene_analyzer import SceneContext, SceneObject
from utils.logger import detailed_logger
from utils.error_handler import log_exception, GameAutomationError

@dataclass
class GameState:
    """游戏状态数据类"""
    scene_context: SceneContext
    player_stats: Dict[str, Any]
    inventory: List[Dict[str, Any]]
    current_task: Optional[Dict[str, Any]]
    battle_status: Optional[Dict[str, Any]]

@dataclass
class Action:
    """动作数据类"""
    type: str
    parameters: Dict[str, Any]
    priority: float
    confidence: float

class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class DecisionMaker:
    """
    AI决策制定器核心类
    负责游戏策略的制定和执行决策
    """
    def __init__(self):
        self.logger = detailed_logger
        self.policy_network = None
        self.experience_buffer = []
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self._initialized = False

    @log_exception
    async def initialize(self) -> None:
        """初始化决策制定器"""
        try:
            # 初始化策略网络
            input_size = 100  # 状态向量大小
            hidden_size = 128
            output_size = 10  # 动作空间大小
            self.policy_network = PolicyNetwork(input_size, hidden_size, output_size)
            
            # 加载预训练模型（如果有）
            await self._load_model()
            
            self._initialized = True
            self.logger.info("Decision Maker initialized successfully")
        except Exception as e:
            raise GameAutomationError(f"Failed to initialize decision maker: {str(e)}")

    @log_exception
    async def make_decision(self, game_state: GameState) -> Optional[Action]:
        """
        制定决策
        :param game_state: 当前游戏状态
        :return: 决策动作
        """
        try:
            if not self._initialized:
                raise GameAutomationError("Decision maker not initialized")

            # 状态预处理
            state_vector = await self._preprocess_state(game_state)
            
            # 通过策略网络获取动作概率
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                action_probs = self.policy_network(state_tensor)
            
            # 动作选择
            action = await self._select_action(action_probs, game_state)
            
            # 记录状态和动作
            self.state_history.append(game_state)
            self.action_history.append(action)
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error making decision: {str(e)}")
            return None

    @log_exception
    async def learn_from_experience(self,
                                  state: GameState,
                                  action: Action,
                                  reward: float,
                                  next_state: GameState) -> None:
        """
        从经验中学习
        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        """
        try:
            # 存储经验
            experience = (state, action, reward, next_state)
            self.experience_buffer.append(experience)
            
            # 记录奖励
            self.reward_history.append(reward)
            
            # 当经验缓冲区足够大时进行学习
            if len(self.experience_buffer) >= 1000:
                await self._update_policy()
                
        except Exception as e:
            self.logger.error(f"Error learning from experience: {str(e)}")

    @log_exception
    async def _preprocess_state(self, game_state: GameState) -> np.ndarray:
        """
        预处理游戏状态
        :param game_state: 游戏状态
        :return: 状态向量
        """
        try:
            # 提取场景特征
            scene_features = await self._extract_scene_features(game_state.scene_context)
            
            # 提取玩家状态特征
            player_features = await self._extract_player_features(game_state.player_stats)
            
            # 提取任务特征
            task_features = await self._extract_task_features(game_state.current_task)
            
            # 合并特征
            state_vector = np.concatenate([
                scene_features,
                player_features,
                task_features
            ])
            
            return state_vector
            
        except Exception as e:
            self.logger.error(f"Error preprocessing state: {str(e)}")
            return np.zeros(100)  # 返回零向量作为后备

    async def _extract_scene_features(self, scene_context: SceneContext) -> np.ndarray:
        """提取场景特征"""
        features = []
        
        # 场景类型编码
        scene_type_encoding = self._one_hot_encode(scene_context.scene_type, ['battle', 'menu', 'map', 'inventory'])
        features.extend(scene_type_encoding)
        
        # 对象特征
        for obj in scene_context.objects[:5]:  # 限制对象数量
            obj_features = self._extract_object_features(obj)
            features.extend(obj_features)
            
        # 填充或截断到固定长度
        target_length = 50
        if len(features) < target_length:
            features.extend([0] * (target_length - len(features)))
        else:
            features = features[:target_length]
            
        return np.array(features)

    async def _extract_player_features(self, player_stats: Dict[str, Any]) -> np.ndarray:
        """提取玩家状态特征"""
        features = [
            player_stats.get('health', 0) / 100,
            player_stats.get('mana', 0) / 100,
            player_stats.get('level', 1) / 100,
            player_stats.get('experience', 0) / 1000
        ]
        return np.array(features)

    async def _extract_task_features(self, current_task: Optional[Dict[str, Any]]) -> np.ndarray:
        """提取任务特征"""
        if not current_task:
            return np.zeros(10)
            
        features = [
            current_task.get('progress', 0) / 100,
            current_task.get('priority', 0) / 10,
            current_task.get('difficulty', 0) / 10
        ]
        
        # 任务类型编码
        task_type_encoding = self._one_hot_encode(
            current_task.get('type', 'unknown'),
            ['battle', 'collection', 'exploration', 'story']
        )
        features.extend(task_type_encoding)
        
        return np.array(features)

    @staticmethod
    def _extract_object_features(obj: SceneObject) -> List[float]:
        """提取对象特征"""
        x, y, w, h = obj.position
        return [
            x / 1000,  # 归一化坐标
            y / 1000,
            w / 1000,
            h / 1000,
            obj.confidence
        ]

    @staticmethod
    def _one_hot_encode(value: str, categories: List[str]) -> List[int]:
        """独热编码"""
        encoding = [1 if value == category else 0 for category in categories]
        encoding.append(1 if value not in categories else 0)  # 未知类别
        return encoding

    @log_exception
    async def _select_action(self,
                           action_probs: torch.Tensor,
                           game_state: GameState) -> Action:
        """
        选择动作
        :param action_probs: 动作概率
        :param game_state: 当前游戏状态
        :return: 选择的动作
        """
        try:
            # 获取最高概率的动作索引
            action_idx = torch.argmax(action_probs).item()
            
            # 将动作索引转换为具体动作
            action_type, parameters = await self._map_action_index(action_idx, game_state)
            
            return Action(
                type=action_type,
                parameters=parameters,
                priority=float(action_probs[0][action_idx]),
                confidence=float(action_probs[0][action_idx])
            )
            
        except Exception as e:
            self.logger.error(f"Error selecting action: {str(e)}")
            # 返回默认动作
            return Action(
                type="wait",
                parameters={},
                priority=0.0,
                confidence=0.0
            )

    async def _map_action_index(self,
                               action_idx: int,
                               game_state: GameState) -> Tuple[str, Dict[str, Any]]:
        """将动作索引映射到具体动作"""
        # 动作映射表
        action_mappings = {
            0: ("wait", {}),
            1: ("tap", {"x": 100, "y": 100}),
            2: ("swipe", {"start_x": 100, "start_y": 100, "end_x": 200, "end_y": 200}),
            3: ("text_input", {"text": ""}),
            # 添加更多动作映射
        }
        
        return action_mappings.get(action_idx, ("wait", {}))

    @log_exception
    async def _update_policy(self) -> None:
        """更新策略网络"""
        try:
            if len(self.experience_buffer) < 1000:
                return
                
            # 准备训练数据
            states, actions, rewards, next_states = zip(*self.experience_buffer[-1000:])
            
            # 转换为张量
            state_tensors = torch.FloatTensor([await self._preprocess_state(s) for s in states])
            reward_tensors = torch.FloatTensor(rewards)
            
            # 计算优势
            advantages = self._calculate_advantages(reward_tensors)
            
            # 更新策略网络
            optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=0.001)
            
            # 执行一步梯度更新
            optimizer.zero_grad()
            action_probs = self.policy_network(state_tensors)
            loss = -torch.mean(torch.log(action_probs) * advantages)
            loss.backward()
            optimizer.step()
            
            # 清理旧经验
            self.experience_buffer = self.experience_buffer[-1000:]
            
        except Exception as e:
            self.logger.error(f"Error updating policy: {str(e)}")

    @staticmethod
    def _calculate_advantages(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """计算优势值"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
            
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    async def _load_model(self) -> None:
        """加载预训练模型"""
        try:
            model_path = 'models/policy_network.pth'
            if torch.cuda.is_available():
                self.policy_network.load_state_dict(
                    torch.load(model_path)
                )
            else:
                self.policy_network.load_state_dict(
                    torch.load(model_path, map_location=torch.device('cpu'))
                )
        except Exception as e:
            self.logger.warning(f"Could not load pretrained model: {str(e)}")

    async def save_model(self) -> None:
        """保存模型"""
        try:
            torch.save(
                self.policy_network.state_dict(),
                'models/policy_network.pth'
            )
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")

# 创建全局实例
decision_maker = DecisionMaker()
