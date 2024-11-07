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
from .reinforcement_learning import DQN, StateMapper, ActionMapper

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

class AIDecisionMaker:
    def __init__(self, game_engine):
        self.logger = setup_logger('ai_decision_maker')
        self.game_engine = game_engine
        self.config = game_engine.config.get('ai_settings', {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actions = ["use_health_potion", "use_mana_potion", "attack", "defend", "use_special_ability", 
                        "retreat", "explore", "interact_with_npc", "complete_quest", "craft_item", 
                        "upgrade_equipment", "trade", "rest", "use_skill", "change_strategy"]
        self.state_size = self.config.get('state_size', 100)
        self.action_size = len(self.actions)
        
        self.state_mapper = StateMapper(self.state_size)
        self.action_mapper = ActionMapper(self.actions)
        
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.get('learning_rate', 0.001))
        self.memory = deque(maxlen=self.config.get('memory_size', 10000))
        
        self.steps_done = 0
        self.epsilon = self.config.get('epsilon', 0.1)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.batch_size = self.config.get('batch_size', 64)
        self.gamma = self.config.get('discount_factor', 0.99)
        
        self.update_target_steps = self.config.get('update_target_steps', 1000)
        self.log_interval = self.config.get('log_interval', 100)
        self.save_interval = self.config.get('save_interval', 1000)

    @log_exception
    async def initialize(self):
        self.logger.info("Initializing AI Decision Maker")
        await self.load_model()

    @log_exception
    async def make_decision(self, game_state):
        state = self.state_mapper.map_state(game_state)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        if random.random() > self.epsilon:
            with torch.no_grad():
                action_values = self.policy_net(state_tensor)
                action_index = action_values.max(1)[1].item()
        else:
            action_index = random.randrange(self.action_size)
        
        action = self.action_mapper.map_action(action_index)
        self.logger.info(f"AI decided action: {action}")
        return action

    @log_exception
    async def learn_from_experience(self, state, action, reward, next_state, done):
        state = self.state_mapper.map_state(state)
        next_state = self.state_mapper.map_state(next_state)
        action_index = self.action_mapper.get_action_index(action)
        
        self.memory.append(Experience(state, action_index, next_state, reward, done))
        self.steps_done += 1
        
        if len(self.memory) >= self.batch_size:
            await self.optimize_model()

        if self.steps_done % self.log_interval == 0:
            self.log_performance()

        if self.steps_done % self.save_interval == 0:
            await self.save_model()

        if self.steps_done % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.update_epsilon()

    @log_exception
    async def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        experiences = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.bool, device=self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = reward_batch + (self.gamma * next_state_values * (~done_batch))

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @log_exception
    async def save_model(self, filename='ai_model.pth'):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filename)
        self.logger.info(f"Model saved to {filename}")

    @log_exception
    async def load_model(self, filename='ai_model.pth'):
        try:
            checkpoint = torch.load(filename)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']
            self.logger.info(f"Model loaded from {filename}")
        except FileNotFoundError:
            self.logger.warning(f"No existing model found at {filename}. Starting with a new model.")

    def log_performance(self):
        avg_reward = sum(exp.reward for exp in list(self.memory)[-self.log_interval:]) / self.log_interval
        self.logger.info(f"Steps: {self.steps_done}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")

    @log_exception
    async def periodic_improvement(self):
        while True:
            await asyncio.sleep(self.game_engine.run_modes[self.game_engine.run_mode]['ai_update_interval'])
            await self.analyze_logs()
            await self.save_model()

    @log_exception
    async def analyze_logs(self, log_file='game_automation.log'):
        self.logger.info("Analyzing logs for AI improvement")
        try:
            with open(log_file, 'r') as f:
                logs = f.readlines()
            
            action_counts = {}
            reward_sum = 0
            log_count = 0

            for log in logs:
                if "AI decided action:" in log:
                    action = log.split("AI decided action:")[1].strip()
                    action_counts[action] = action_counts.get(action, 0) + 1
                elif "Reward:" in log:
                    reward = float(log.split("Reward:")[1].strip())
                    reward_sum += reward
                    log_count += 1

            avg_reward = reward_sum / log_count if log_count > 0 else 0

            most_common_action = max(action_counts, key=action_counts.get)
            if action_counts[most_common_action] / sum(action_counts.values()) > 0.5:
                self.logger.info(f"AI is overusing action: {most_common_action}. Adjusting exploration rate.")
                self.epsilon = min(self.epsilon * 1.1, 1.0)

            if avg_reward < self.config.get('expected_reward', 0):
                self.logger.info(f"Average reward ({avg_reward:.2f}) is below expected. Adjusting learning rate.")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 1.1

            self.logger.info(f"Log analysis complete. New epsilon: {self.epsilon:.2f}, New learning rate: {self.optimizer.param_groups[0]['lr']:.5f}")

        except Exception as e:
            self.logger.error(f"Error analyzing logs: {str(e)}")

    def get_action_description(self, action):
        return action

    @log_exception
    async def train_on_complex_task(self, task):
        self.logger.info(f"Training AI on complex task: {task['name']}")
        initial_state = await self.game_engine.get_game_state()
        total_reward = 0
        steps = 0
        max_steps = self.config.get('max_steps_per_task', 1000)

        while steps < max_steps and not await self.game_engine.task_manager.is_task_completed(task['id']):
            current_state = await self.game_engine.get_game_state()
            action = await self.make_decision(current_state)
            reward = await self.game_engine.execute_action(action)
            next_state = await self.game_engine.get_game_state()
            done = await self.game_engine.task_manager.is_task_completed(task['id'])

            await self.learn_from_experience(current_state, action, reward, next_state, done)
            total_reward += reward
            steps += 1

            if steps % 100 == 0:
                self.logger.info(f"Training progress: Steps: {steps}, Total Reward: {total_reward}")

        self.logger.info(f"Training completed for task: {task['name']}. Total Steps: {steps}, Total Reward: {total_reward}")

    @log_exception
    async def adapt_to_new_game_mechanics(self, new_mechanics):
        self.logger.info(f"Adapting AI to new game mechanics: {new_mechanics}")
        
        # 扩展状态和动作空间
        self.state_size += len(new_mechanics['new_states'])
        self.action_size += len(new_mechanics['new_actions'])
        
        # 创建新的神经网络以适应扩展的状态和动作空间
        new_policy_net = DQN(self.state_size, self.action_size).to(self.device)
        new_target_net = DQN(self.state_size, self.action_size).to(self.device)
        
        # 将旧网络的权重复制到新网络中
        with torch.no_grad():
            for old_param, new_param in zip(self.policy_net.parameters(), new_policy_net.parameters()):
                new_param[:old_param.size(0), :old_param.size(1)] = old_param
            for old_param, new_param in zip(self.target_net.parameters(), new_target_net.parameters()):
                new_param[:old_param.size(0), :old_param.size(1)] = old_param
        
        # 更新网络和优化器
        self.policy_net = new_policy_net
        self.target_net = new_target_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.get('learning_rate', 0.001))
        
        # 更新动作映射
        for new_action in new_mechanics['new_actions']:
            self.actions.append(new_action)
        self.action_mapper = ActionMapper(self.actions)
        
        # 更新状态映射
        self.state_mapper.update_mapping(new_mechanics['new_states'])
        
        self.logger.info("AI successfully adapted to new game mechanics")

    @log_exception
    async def perform_curriculum_learning(self, curriculum):
        self.logger.info("Starting curriculum learning")
        for lesson in curriculum:
            self.logger.info(f"Learning lesson: {lesson['name']}")
            for _ in range(lesson['iterations']):
                await self.train_on_complex_task(lesson['task'])
            await self.evaluate_performance(lesson['evaluation_criteria'])
        self.logger.info("Curriculum learning completed")

    @log_exception
    async def evaluate_performance(self, criteria):
        self.logger.info("Evaluating AI performance")
        performance_metrics = {
            'success_rate': 0,
            'average_reward': 0,
            'average_steps': 0
        }
        
        num_evaluations = criteria.get('num_evaluations', 10)
        for _ in range(num_evaluations):
            initial_state = await self.game_engine.get_game_state()
            total_reward = 0
            steps = 0
            success = False
            
            while steps < criteria['max_steps'] and not success:
                action = await self.make_decision(initial_state)
                reward = await self.game_engine.execute_action(action)
                next_state = await self.game_engine.get_game_state()
                success = criteria['success_condition'](next_state)
                
                total_reward += reward
                steps += 1
                initial_state = next_state
            
            performance_metrics['success_rate'] += 1 if success else 0
            performance_metrics['average_reward'] += total_reward
            performance_metrics['average_steps'] += steps
        
        performance_metrics['success_rate'] /= num_evaluations
        performance_metrics['average_reward'] /= num_evaluations
        performance_metrics['average_steps'] /= num_evaluations
        
        self.logger.info(f"Performance evaluation results: {performance_metrics}")
        return performance_metrics

ai_decision_maker = AIDecisionMaker(None)  # 初始化时传入 None，后续在 GameEngine 中设置正确的引用