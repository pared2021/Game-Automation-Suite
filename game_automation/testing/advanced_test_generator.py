import random
import asyncio
from game_automation.game_engine import GameEngine
from game_automation.ai.meta_learning import meta_learning_manager
from utils.logger import detailed_logger

class AdvancedTestGenerator:
    def __init__(self):
        self.logger = detailed_logger
        self.game_engine = GameEngine()
        self.test_scenarios = []

    def generate_test_scenario(self):
        scenario = {
            'scene_type': random.choice(['battle', 'exploration', 'dialogue', 'puzzle', 'boss_fight']),
            'player_stats': {
                'health': random.randint(1, 100),
                'mana': random.randint(0, 100),
                'level': random.randint(1, 50),
                'experience': random.randint(0, 1000)
            },
            'enemies': [
                {
                    'name': f'Enemy{i}',
                    'health': random.randint(10, 100),
                    'level': random.randint(1, 30),
                    'abilities': random.sample(['fireball', 'ice_shard', 'poison_dart', 'healing'], k=random.randint(1, 3))
                }
                for i in range(random.randint(0, 5))
            ],
            'items': [
                {
                    'name': f'Item{i}',
                    'quantity': random.randint(1, 5),
                    'type': random.choice(['weapon', 'armor', 'potion', 'key_item'])
                }
                for i in range(random.randint(0, 10))
            ],
            'environment': {
                'weather': random.choice(['sunny', 'rainy', 'foggy', 'stormy']),
                'time_of_day': random.choice(['morning', 'afternoon', 'evening', 'night']),
                'terrain': random.choice(['forest', 'desert', 'mountain', 'dungeon', 'city'])
            },
            'objectives': [
                {
                    'type': random.choice(['kill_enemies', 'collect_items', 'solve_puzzle', 'reach_location']),
                    'target': random.randint(1, 5),
                    'completed': 0
                }
                for _ in range(random.randint(1, 3))
            ]
        }
        return scenario

    async def run_test(self, scenario):
        self.logger.info(f"Running test with scenario: {scenario}")
        await self.game_engine.initialize()
        await self.game_engine.set_game_state(scenario)
        total_reward = 0
        actions_taken = []

        for _ in range(50):  # 执行50个动作或直到所有目标完成
            game_state = await self.game_engine.get_game_state()
            action = await meta_learning_manager.predict(game_state)
            reward = await self.game_engine.execute_action(action)
            total_reward += reward
            actions_taken.append(action)

            # 更新目标完成状态
            for objective in scenario['objectives']:
                if objective['type'] == 'kill_enemies' and action == 'attack':
                    objective['completed'] += 1
                elif objective['type'] == 'collect_items' and action == 'collect':
                    objective['completed'] += 1
                # ... 其他目标类型的更新逻辑

            # 检查是否所有目标都已完成
            if all(obj['completed'] >= obj['target'] for obj in scenario['objectives']):
                break

        self.logger.info(f"Test completed. Total reward: {total_reward}")
        return {
            'scenario': scenario,
            'total_reward': total_reward,
            'actions_taken': actions_taken,
            'objectives_completed': [obj['completed'] >= obj['target'] for obj in scenario['objectives']]
        }

    async def generate_and_run_tests(self, num_tests):
        results = []
        for i in range(num_tests):
            scenario = self.generate_test_scenario()
            result = await self.run_test(scenario)
            results.append(result)
            self.test_scenarios.append(scenario)
        self.analyze_results(results)

    def analyze_results(self, results):
        avg_reward = sum(r['total_reward'] for r in results) / len(results)
        max_reward = max(r['total_reward'] for r in results)
        min_reward = min(r['total_reward'] for r in results)
        completion_rate = sum(all(r['objectives_completed']) for r in results) / len(results)

        self.logger.info(f"Test results summary:")
        self.logger.info(f"Average reward: {avg_reward}")
        self.logger.info(f"Max reward: {max_reward}")
        self.logger.info(f"Min reward: {min_reward}")
        self.logger.info(f"Objective completion rate: {completion_rate * 100}%")

        # 分析不同场景类型的表现
        scene_type_performance = {}
        for result in results:
            scene_type = result['scenario']['scene_type']
            if scene_type not in scene_type_performance:
                scene_type_performance[scene_type] = []
            scene_type_performance[scene_type].append(result['total_reward'])

        for scene_type, rewards in scene_type_performance.items():
            avg_reward = sum(rewards) / len(rewards)
            self.logger.info(f"Average reward for {scene_type}: {avg_reward}")

        # 分析最常用的动作
        all_actions = [action for result in results for action in result['actions_taken']]
        action_counts = {action: all_actions.count(action) for action in set(all_actions)}
        most_common_action = max(action_counts, key=action_counts.get)
        self.logger.info(f"Most common action: {most_common_action} (used {action_counts[most_common_action]} times)")

    async def generate_edge_cases(self):
        edge_cases = [
            self.generate_low_health_scenario(),
            self.generate_high_level_enemies_scenario(),
            self.generate_resource_scarcity_scenario(),
            self.generate_complex_puzzle_scenario(),
            self.generate_time_pressure_scenario()
        ]
        for case in edge_cases:
            await self.run_test(case)

    def generate_low_health_scenario(self):
        scenario = self.generate_test_scenario()
        scenario['player_stats']['health'] = random.randint(1, 10)
        return scenario

    def generate_high_level_enemies_scenario(self):
        scenario = self.generate_test_scenario()
        for enemy in scenario['enemies']:
            enemy['level'] = random.randint(40, 50)
        return scenario

    def generate_resource_scarcity_scenario(self):
        scenario = self.generate_test_scenario()
        scenario['items'] = [item for item in scenario['items'] if item['type'] != 'potion'][:2]
        return scenario

    def generate_complex_puzzle_scenario(self):
        scenario = self.generate_test_scenario()
        scenario['scene_type'] = 'puzzle'
        scenario['objectives'] = [{
            'type': 'solve_puzzle',
            'target': 1,
            'completed': 0,
            'complexity': random.randint(8, 10)  # 1-10 scale of complexity
        }]
        return scenario

    def generate_time_pressure_scenario(self):
        scenario = self.generate_test_scenario()
        scenario['time_limit'] = random.randint(10, 30)  # 10-30 seconds time limit
        return scenario

    async def run_performance_test(self, duration=300):  # 5 minutes test
        start_time = asyncio.get_event_loop().time()
        actions_per_second = []
        while asyncio.get_event_loop().time() - start_time < duration:
            scenario = self.generate_test_scenario()
            test_start = asyncio.get_event_loop().time()
            await self.run_test(scenario)
            test_duration = asyncio.get_event_loop().time() - test_start
            actions_per_second.append(50 / test_duration)  # 50 actions per test

        avg_actions_per_second = sum(actions_per_second) / len(actions_per_second)
        self.logger.info(f"Performance test results:")
        self.logger.info(f"Average actions per second: {avg_actions_per_second}")
        return avg_actions_per_second

advanced_test_generator = AdvancedTestGenerator()

# 使用示例
async def run_advanced_tests():
    await advanced_test_generator.generate_and_run_tests(100)  # 生成并运行100个测试
    await advanced_test_generator.generate_edge_cases()
    await advanced_test_generator.run_performance_test()

if __name__ == "__main__":
    asyncio.run(run_advanced_tests())
