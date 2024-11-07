import yaml
import time
import random
from utils.error_handler import log_exception
from .auto_battle_strategy import AutoBattleStrategy

class AdvancedBattleStrategy(AutoBattleStrategy):
    def __init__(self, strategy_file):
        with open(strategy_file, 'r') as f:
            self.strategies = yaml.safe_load(f)
        self.combo_counter = 0
        self.last_action = None
        self.action_history = []
        self.max_combo = 5

    @log_exception
    async def execute_strategy(self, battle_type, game_engine):
        strategy = self.get_strategy(battle_type)
        for phase in strategy:
            await self.execute_phase(phase, game_engine)

    async def execute_phase(self, phase, game_engine):
        phase_type = phase['type']
        if phase_type == 'action_sequence':
            await self.execute_action_sequence(phase['actions'], game_engine)
        elif phase_type == 'conditional':
            if await self.evaluate_condition(phase['condition'], game_engine):
                await self.execute_action_sequence(phase['actions'], game_engine)
        elif phase_type == 'loop':
            await self.execute_loop(phase, game_engine)
        elif phase_type == 'combo':
            await self.execute_combo(phase['actions'], game_engine)
        elif phase_type == 'adaptive':
            await self.execute_adaptive_action(phase, game_engine)

    @log_exception
    async def execute_action_sequence(self, actions, game_engine):
        for action in actions:
            await game_engine.execute_action(action)
            self.update_combo(action)
            self.action_history.append(action)

    @log_exception
    async def execute_loop(self, phase, game_engine):
        start_time = time.time()
        while time.time() - start_time < phase['duration']:
            await self.execute_action_sequence(phase['actions'], game_engine)
            if 'break_condition' in phase and await self.evaluate_condition(phase['break_condition'], game_engine):
                break

    @log_exception
    async def execute_combo(self, actions, game_engine):
        for action in actions:
            if self.combo_counter >= action['min_combo']:
                await game_engine.execute_action(action)
                self.reset_combo()
                break

    @log_exception
    async def execute_adaptive_action(self, phase, game_engine):
        game_state = await game_engine.get_game_state()
        best_action = self.select_best_action(phase['actions'], game_state)
        await game_engine.execute_action(best_action)

    @log_exception
    def update_combo(self, action):
        if action['type'] == self.last_action:
            self.combo_counter = min(self.combo_counter + 1, self.max_combo)
        else:
            self.reset_combo()
        self.last_action = action['type']

    @log_exception
    def reset_combo(self):
        self.combo_counter = 0
        self.last_action = None

    @log_exception
    def select_best_action(self, actions, game_state):
        scores = [(action, self.calculate_action_score(action, game_state)) for action in actions]
        return max(scores, key=lambda x: x[1])[0]

    @log_exception
    def calculate_action_score(self, action, game_state):
        score = 0
        if 'damage' in action:
            score += action['damage'] * 2
        if 'healing' in action:
            score += action['healing'] * (100 / game_state['health'])
        if 'buff' in action:
            score += action['buff'] * 1.5
        return score

    @log_exception
    async def evaluate_condition(self, condition, game_engine):
        condition_type = condition['type']
        if condition_type == 'health_percentage':
            health = await game_engine.get_health_percentage()
            return health <= condition['value']
        elif condition_type == 'enemy_count':
            enemy_count = await game_engine.get_enemy_count()
            return enemy_count >= condition['value']
        elif condition_type == 'combo_count':
            return self.combo_counter >= condition['value']
        elif condition_type == 'action_history':
            return self.check_action_history(condition['sequence'])
        return await super().evaluate_condition(condition, game_engine)

    @log_exception
    def check_action_history(self, sequence):
        return sequence == [action['type'] for action in self.action_history[-len(sequence):]]