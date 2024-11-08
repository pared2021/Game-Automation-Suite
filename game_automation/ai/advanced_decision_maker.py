import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from utils.logger import detailed_logger
from utils.config_manager import config_manager
from .unified_decision_maker import UnifiedDecisionMaker

class AdvancedDecisionMaker:
    def __init__(self):
        self.logger = detailed_logger
        self.config = config_manager.get('ai', {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = self.config.get('state_size', 100)
        self.action_size = self.config.get('action_size', 10)
        self.decision_maker = UnifiedDecisionMaker(self.state_size, self.action_size, self.config)

    def initialize(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.decision_maker = UnifiedDecisionMaker(state_size, action_size, self.config)
        self.logger.info("AdvancedDecisionMaker initialized")

    async def make_decision(self, game_state):
        return await self.decision_maker.make_decision(game_state)

    async def learn_from_experience(self, state, action, reward, next_state, done):
        await self.decision_maker.learn_from_experience(state, action, reward, next_state, done)

    async def save(self, name):
        await self.decision_maker.save(name)

    async def load(self, name):
        await self.decision_maker.load(name)

    async def periodic_improvement(self):
        await self.decision_maker.periodic_improvement()

    async def analyze_performance(self):
        await self.decision_maker.analyze_performance()

    async def reset(self):
        await self.decision_maker.reset()

advanced_decision_maker = AdvancedDecisionMaker()
