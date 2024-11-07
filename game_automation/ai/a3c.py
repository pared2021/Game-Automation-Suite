import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_size, n_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc_pi = nn.Linear(256, n_actions)
        self.fc_v = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        pi = F.softmax(self.fc_pi(x), dim=-1)
        v = self.fc_v(x)
        return pi, v

class A3CAgent:
    def __init__(self, input_size, n_actions, gamma=0.99, learning_rate=0.001):
        self.gamma = gamma
        self.model = ActorCritic(input_size, n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        pi, _ = self.model(state)
        m = Categorical(pi)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update(self, rewards, log_probs, values, final_value):
        R = final_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def worker(rank, shared_model, counter, lock):
    env = GameEnvironment()  # 假设我们有一个游戏环境类
    local_model = A3CAgent(env.state_size, env.action_size)
    local_model.model.load_state_dict(shared_model.state_dict())

    while True:
        state = env.reset()
        done = False
        rewards, log_probs, values = [], [], []

        while not done:
            action, log_prob = local_model.select_action(state)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            _, value = local_model.model(torch.from_numpy(state).float())
            values.append(value)

            state = next_state

            if done:
                final_value = 0
            else:
                _, final_value = local_model.model(torch.from_numpy(state).float())
                final_value = final_value.item()

            local_model.update(rewards, log_probs, values, final_value)

            with lock:
                counter.value += 1

            if counter.value % 100 == 0:
                shared_model.load_state_dict(local_model.model.state_dict())

if __name__ == '__main__':
    mp.set_start_method('spawn')
    env = GameEnvironment()  # 假设我们有一个游戏环境类
    shared_model = ActorCritic(env.state_size, env.action_size)
    shared_model.share_memory()

    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    for rank in range(4):  # 创建4个工作进程
        p = mp.Process(target=worker, args=(rank, shared_model, counter, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()