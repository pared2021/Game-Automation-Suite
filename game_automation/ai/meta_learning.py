import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from collections import OrderedDict

class MAML(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MAML, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(input_size, hidden_size)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(hidden_size, hidden_size)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(hidden_size, output_size))
        ]))

    def forward(self, x):
        return self.model(x)

class MetaLearner:
    def __init__(self, input_size, hidden_size, output_size, alpha=0.01, beta=0.001):
        self.model = MAML(input_size, hidden_size, output_size)
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=beta)
        self.alpha = alpha

    def adapt(self, support_set, support_labels):
        adapted_state_dict = self.model.state_dict()
        optimizer = optim.SGD(self.model.parameters(), lr=self.alpha)

        for _ in range(5):  # 执行5步梯度下降
            predictions = self.model(support_set)
            loss = F.cross_entropy(predictions, support_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return adapted_state_dict

    def meta_learn(self, tasks):
        self.meta_optimizer.zero_grad()
        meta_loss = 0

        for task in tasks:
            support_set, support_labels, query_set, query_labels = task
            adapted_state_dict = self.adapt(support_set, support_labels)

            # 保存原始模型参数
            original_state_dict = self.model.state_dict()
            
            # 加载适应后的参数
            self.model.load_state_dict(adapted_state_dict)
            
            # 在查询集上计算损失
            predictions = self.model(query_set)
            loss = F.cross_entropy(predictions, query_labels)
            meta_loss += loss

            # 恢复原始模型参数
            self.model.load_state_dict(original_state_dict)

        meta_loss /= len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

class TransferLearner:
    def __init__(self, base_model, target_output_size):
        self.base_model = base_model
        self.target_model = self.create_target_model(target_output_size)
        self.optimizer = optim.Adam(self.target_model.parameters())

    def create_target_model(self, target_output_size):
        target_model = nn.Sequential(
            *list(self.base_model.children())[:-1],  # 移除最后一层
            nn.Linear(list(self.base_model.children())[-2].out_features, target_output_size)
        )
        return target_model

    def fine_tune(self, train_data, train_labels, epochs=10):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.target_model(train_data)
            loss = F.cross_entropy(outputs, train_labels)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def predict(self, data):
        with torch.no_grad():
            return self.target_model(data)

class MetaLearningManager:
    def __init__(self, input_size, hidden_size, output_size):
        self.meta_learner = MetaLearner(input_size, hidden_size, output_size)
        self.transfer_learner = None

    async def train_on_game(self, game_data):
        tasks = self.prepare_tasks(game_data)
        for epoch in range(100):  # 执行100个元学习周期
            meta_loss = self.meta_learner.meta_learn(tasks)
            print(f"Meta-learning epoch {epoch+1}, Loss: {meta_loss}")

    def prepare_tasks(self, game_data):
        # 将游戏数据转换为元学习任务
        # 这里需要根据实际的游戏数据结构进行实现
        tasks = []
        for game_session in game_data:
            support_set = game_session['training_data'][:100]
            support_labels = game_session['training_labels'][:100]
            query_set = game_session['training_data'][100:200]
            query_labels = game_session['training_labels'][100:200]
            tasks.append((support_set, support_labels, query_set, query_labels))
        return tasks

    async def adapt_to_new_game(self, new_game_data):
        # 使用元学习模型快速适应新游戏
        support_set, support_labels = new_game_data['support_set'], new_game_data['support_labels']
        adapted_state_dict = self.meta_learner.adapt(support_set, support_labels)
        self.meta_learner.model.load_state_dict(adapted_state_dict)
        print("Model adapted to new game")

    async def transfer_to_new_task(self, new_task_data, target_output_size):
        # 使用迁移学习适应新任务
        self.transfer_learner = TransferLearner(self.meta_learner.model, target_output_size)
        train_data, train_labels = new_task_data['train_data'], new_task_data['train_labels']
        self.transfer_learner.fine_tune(train_data, train_labels)
        print("Model transferred to new task")

    async def predict(self, input_data):
        if self.transfer_learner:
            return self.transfer_learner.predict(input_data)
        else:
            return self.meta_learner.model(input_data)

meta_learning_manager = MetaLearningManager(input_size=100, hidden_size=64, output_size=10)