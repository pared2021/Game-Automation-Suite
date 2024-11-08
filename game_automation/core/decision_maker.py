import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import torch
from ..core.scene_analyzer import SceneContext, SceneObject
from utils.logger import detailed_logger
from utils.error_handler import log_exception, GameAutomationError
from ..ai.unified_decision_maker import UnifiedDecisionMaker

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

class DecisionMaker:
    """
    AI决策制定器核心类
    负责游戏策略的制定和执行决策
    """
    def __init__(self):
        self.logger = detailed_logger
        self.config = {
            'state_size': 100,
            'action_size': 10,
            'memory_size': 2000,
            'gamma': 0.95,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'learning_rate': 0.001
        }
        self.decision_maker = UnifiedDecisionMaker(self.config['state_size'], self.config['action_size'], self.config)
        self._initialized = False

    @log_exception
    async def initialize(self) -> None:
        """初始化决策制定器"""
        try:
            await self.decision_maker.load('models/policy_network.pth')
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

            return await self.decision_maker.make_decision(game_state)
            
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
            await self.decision_maker.learn_from_experience(state, action, reward, next_state, False)
        except Exception as e:
            self.logger.error(f"Error learning from experience: {str(e)}")

    async def save_model(self) -> None:
        """保存模型"""
        try:
            await self.decision_maker.save('models/policy_network.pth')
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")

# 创建全局实例
decision_maker = DecisionMaker()
