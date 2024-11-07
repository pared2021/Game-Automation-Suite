import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from utils.logger import detailed_logger
from utils.config_manager import config_manager

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class AdvancedDecisionMaker:
    def __init__(self):
        self.logger = detailed_logger
        self.config = config_manager.get('ai', {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = None
        self.action_size = None
        self.memory = None
        self.model = None
        self.target_model = None
        self.optimizer = None

    def initialize(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=self.config.get('memory_size', 2000))
        self.gamma = self.config.get('gamma', 0.95)
        self.epsilon = self.config.get('epsilon', 1.0)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
        self.logger.info("AdvancedDecisionMaker initialized")

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    async def train(self, env, episodes):
        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0
            for time in range(500):
                action = self.act(state)
                next_state, reward, done, _ = await env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    self.logger.info(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {self.epsilon:.2}")
                    break
                if len(self.memory) > self.config.get('batch_size', 32):
                    self.replay(self.config.get('batch_size', 32))
            if e % self.config.get('target_update', 10) == 0:
                self.update_target_model()

    async def make_decision(self, game_state):
        state_vector = self.preprocess_state(game_state)
        action_index = self.act(state_vector)
        return self.map_action(action_index, game_state['potential_actions'])

    def preprocess_state(self, game_state):
        # 将游戏状态转换为神经网络的输入向量
        state_vector = []
        
        # 编码场景类型
        scene_types = ['battle', 'dialogue', 'inventory', 'exploration', 'main_menu', 'character_creation', 'town', 'dungeon_entrance', 'skill_tree', 'quest_log', 'world_map']
        state_vector.extend([1 if game_state['scene_type'] == st else 0 for st in scene_types])
        
        # 编码关键元素
        all_possible_elements = set()
        for scene in self.config.get('preset_scenes', {}).values():
            all_possible_elements.update(scene.get('key_elements', []))
        state_vector.extend([1 if element in game_state['key_elements'] else 0 for element in all_possible_elements])
        
        # 编码玩家状态
        player_stats = game_state['player_stats']
        state_vector.extend([
            player_stats['health'] / 100,
            player_stats['mana'] / 100,
            player_stats['level'] / 50,
            player_stats['experience'] / 10000
        ])
        
        # 编码库存信息
        inventory = game_state['inventory']
        state_vector.extend([
            inventory['gold'] / 10000,
            len(inventory['weapons']) / 10,
            len(inventory['armor']) / 5,
            inventory['health_potions'] / 20
        ])
        
        return np.array(state_vector)

    def map_action(self, action_index, potential_actions):
        if action_index < len(potential_actions):
            return potential_actions[action_index]
        else:
            return random.choice(potential_actions)

    async def learn_from_experience(self, state, action, reward, next_state, done):
        state_vector = self.preprocess_state(state)
        next_state_vector = self.preprocess_state(next_state)
        action_index = state['potential_actions'].index(action)
        self.remember(state_vector, action_index, reward, next_state_vector, done)
        if len(self.memory) > self.config.get('batch_size', 32):
            self.replay(self.config.get('batch_size', 32))

    async def save(self, name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, name)
        self.logger.info(f"Model saved to {name}")

    async def load(self, name):
        checkpoint = torch.load(name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_target_model()
        self.logger.info(f"Model loaded from {name}")

    async def periodic_improvement(self):
        while True:
            await asyncio.sleep(self.config.get('improvement_interval', 3600))
            self.logger.info("Performing periodic improvement of the AI model")
            # 这里可以添加额外的改进逻辑，比如重新训练模型或微调超参数
            await self.analyze_performance()

    async def analyze_performance(self):
        # 分析AI的性能并进行必要的调整
        recent_rewards = [experience[2] for experience in list(self.memory)[-100:]]
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        self.logger.info(f"Average reward over last 100 experiences: {avg_reward}")
        
        if avg_reward < self.config.get('min_acceptable_reward', 0):
            self.logger.warning("Performance below acceptable level. Adjusting learning rate and epsilon.")
            self.learning_rate *= 1.1
            self.epsilon = min(self.epsilon * 1.1, 1.0)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

    async def reset(self):
        self.epsilon = self.config.get('epsilon', 1.0)
        self.memory.clear()
        self.logger.info("AI decision maker reset")

advanced_decision_maker = AdvancedDecisionMaker()