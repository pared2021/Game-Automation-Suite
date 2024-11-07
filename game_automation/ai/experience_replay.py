import random
from collections import deque
import numpy as np

class ExperienceReplay:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.memory)

class PrioritizedExperienceReplay:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.memory = []
        self.capacity = capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            probs = self.priorities
        else:
            probs = self.priorities[:len(self.memory)]
        probs = probs ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        self.beta = np.min([1., self.beta + self.beta_increment])
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), indices, weights)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)

# 使用示例
experience_replay = ExperienceReplay(10000)
prioritized_experience_replay = PrioritizedExperienceReplay(10000)

# 在训练循环中使用
for episode in range(num_episodes):
    state = env.reset()
    for t in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # 存储经验
        experience_replay.push(state, action, reward, next_state, done)
        
        # 如果经验回放缓冲区足够大，则进行学习
        if len(experience_replay) > batch_size:
            batch = experience_replay.sample(batch_size)
            agent.learn(batch)
        
        if done:
            break
        state = next_state

# 对于优先经验回放，更新步骤略有不同
batch, indices, weights = prioritized_experience_replay.sample(batch_size)
td_errors = agent.compute_td_errors(batch)
prioritized_experience_replay.update_priorities(indices, td_errors)
agent.learn(batch, weights)