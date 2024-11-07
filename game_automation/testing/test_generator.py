import random
import asyncio
from utils.logger import detailed_logger
from game_automation.game_engine import GameEngine

class TestGenerator:
    def __init__(self):
        self.logger = detailed_logger
        self.game_engine = GameEngine()

    def generate_test_scenario(self):
        scenario = {
            'scene_type': random.choice(['battle', 'exploration', 'dialogue']),
            'player_stats': {
                'health': random.randint(1, 100),
                'mana': random.randint(0, 100),
                'level': random.randint(1, 50)
            },
            'enemies': [
                {'name': f'Enemy{i}', 'health': random.randint(10, 100)}
                for i in range(random.randint(0, 3))
            ],
            'items': [
                {'name': f'Item{i}', 'quantity': random.randint(1, 5)}
                for i in range(random.randint(0, 5))
            ],
            'environment': {
                'weather': random.choice(['sunny', 'rainy', 'foggy']),
                'time_of_day': random.choice(['morning', 'afternoon', 'night'])
            }
        }
        return scenario

    async def run_test(self, scenario):
        self.logger.info(f"Running test with scenario: {scenario}")
        await self.game_engine.initialize()
        await self.game_engine.set_game_state(scenario)
        total_reward = 0
        for _ in range(10):  # 执行10个动作
            action = await self.game_engine.ai_decision_maker.make_decision(scenario)
            reward = await self.game_engine.execute_action(action)
            total_reward += reward
            scenario = await self.game_engine.get_game_state()
            self.logger.info(f"Action: {action}, Reward: {reward}")
        self.logger.info(f"Test completed. Total reward: {total_reward}")
        return total_reward

    async def generate_and_run_tests(self, num_tests):
        results = []
        for i in range(num_tests):
            scenario = self.generate_test_scenario()
            reward = await self.run_test(scenario)
            results.append({'scenario': scenario, 'reward': reward})
        self.analyze_results(results)

    def analyze_results(self, results):
        avg_reward = sum(r['reward'] for r in results) / len(results)
        max_reward = max(r['reward'] for r in results)
        min_reward = min(r['reward'] for r in results)
        self.logger.info(f"Test results summary:")
        self.logger.info(f"Average reward: {avg_reward}")
        self.logger.info(f"Max reward: {max_reward}")
        self.logger.info(f"Min reward: {min_reward}")
        # 可以添加更多的分析，如不同场景类型的平均奖励等

test_generator = TestGenerator()

# 使用示例
async def run_tests():
    await test_generator.generate_and_run_tests(10)  # 生成并运行10个测试

if __name__ == "__main__":
    asyncio.run(run_tests())
