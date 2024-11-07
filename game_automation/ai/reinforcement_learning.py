import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q

    def save_model(self, filename):
        np.save(filename, self.q_table)

    def load_model(self, filename):
        self.q_table = np.load(filename)

class StateMapper:
    def __init__(self, state_size):
        self.state_size = state_size

    def map_state(self, game_state):
        # 将游戏状态映射到离散状态空间
        # 这里需要根据具体游戏设计合适的映射方法
        health = min(int(game_state['health'] / 10), 9)
        enemy_count = min(game_state['enemy_count'], 9)
        gold = min(int(game_state['gold'] / 100), 9)
        return health * 100 + enemy_count * 10 + gold

class ActionMapper:
    def __init__(self, actions):
        self.actions = actions
        self.action_to_index = {action: i for i, action in enumerate(actions)}
        self.index_to_action = {i: action for i, action in enumerate(actions)}

    def map_action(self, action_index):
        return self.index_to_action[action_index]

    def get_action_index(self, action):
        return self.action_to_index[action]