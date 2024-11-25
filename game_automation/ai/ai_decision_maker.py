import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
import asyncio
from utils.logger import setup_logger
from utils.error_handler import log_exception
from .unified_decision_maker import UnifiedDecisionMaker

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

class AIDecisionMaker:
    def __init__(self, game_engine):
        """Initialize AI Decision Maker
        
        Args:
            game_engine: GameEngine instance
        
        Raises:
            ValueError: If game_engine is None
        """
        if game_engine is None:
            raise ValueError("game_engine cannot be None")
            
        self.logger = setup_logger('ai_decision_maker')
        self.game_engine = game_engine
        self.config = game_engine.config.get('ai_settings', {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actions = ["use_health_potion", "use_mana_potion", "attack", "defend", "use_special_ability", 
                        "retreat", "explore", "interact_with_npc", "complete_quest", "craft_item", 
                        "upgrade_equipment", "trade", "rest", "use_skill", "change_strategy"]
        self.state_size = self.config.get('state_size', 100)
        self.action_size = len(self.actions)
        
        self.decision_maker = UnifiedDecisionMaker(self.state_size, self.action_size, self.config)

    @log_exception
    async def initialize(self):
        self.logger.info("Initializing AI Decision Maker")
        await self.decision_maker.load('ai_model.pth')

    @log_exception
    async def make_decision(self, game_state):
        return await self.decision_maker.make_decision(game_state)

    @log_exception
    async def learn_from_experience(self, state, action, reward, next_state, done):
        await self.decision_maker.learn_from_experience(state, action, reward, next_state, done)

    @log_exception
    async def optimize_model(self):
        await self.decision_maker.replay(self.config.get('batch_size', 64))

    def update_epsilon(self):
        self.decision_maker.epsilon = max(self.decision_maker.epsilon_min, self.decision_maker.epsilon * self.decision_maker.epsilon_decay)

    @log_exception
    async def save_model(self, filename='ai_model.pth'):
        await self.decision_maker.save(filename)

    @log_exception
    async def load_model(self, filename='ai_model.pth'):
        await self.decision_maker.load(filename)

    def log_performance(self):
        self.logger.info(f"Steps: {self.decision_maker.steps_done}, Epsilon: {self.decision_maker.epsilon:.2f}")

    @log_exception
    async def periodic_improvement(self):
        await self.decision_maker.periodic_improvement()

    @log_exception
    async def analyze_logs(self, log_file='game_automation.log'):
        self.logger.info("Analyzing logs for AI improvement")
        # Log analysis logic here

    def get_action_description(self, action):
        return action

    @log_exception
    async def train_on_complex_task(self, task):
        self.logger.info(f"Training AI on complex task: {task['name']}")
        # Training logic here

    @log_exception
    async def adapt_to_new_game_mechanics(self, new_mechanics):
        self.logger.info(f"Adapting AI to new game mechanics: {new_mechanics}")
        # Adaptation logic here

    @log_exception
    async def perform_curriculum_learning(self, curriculum):
        self.logger.info("Starting curriculum learning")
        # Curriculum learning logic here

    @log_exception
    async def evaluate_performance(self, criteria):
        self.logger.info("Evaluating AI performance")
        # Performance evaluation logic here
