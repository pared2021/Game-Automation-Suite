import asyncio
from game_automation.game_engine import GameEngine
from game_automation.image_recognition import ImageRecognition
from game_automation.controllers.task_manager import TaskManager

class BattlePreparation:
    def __init__(self, game_engine: GameEngine):
        self.game_engine = game_engine
        self.image_recognition = ImageRecognition()
        self.task_manager = TaskManager(game_engine)

    async def prepare_for_battle(self):
        await self.task_manager.load_tasks()
        await self.task_manager.check_and_complete_tasks()

        if not await self.one_key_deploy():
            await self.deploy()

    async def one_key_deploy(self):
        one_key_deploy = await self.image_recognition.find_game_object('assets/one_key_deploy_button.png')
        if one_key_deploy:
            await self.game_engine.touch_controller.tap(one_key_deploy[0], one_key_deploy[1])
            await asyncio.sleep(2)
            return True
        return False

    async def deploy(self):
        deploy_button = await self.image_recognition.find_game_object('assets/deploy_button.png')
        if deploy_button:
            await self.game_engine.touch_controller.tap(deploy_button[0], deploy_button[1])
            await asyncio.sleep(2)
            return True
        return False

if __name__ == "__main__":
    async def main():
        game_engine = GameEngine('config/strategies.json')
        battle_prep = BattlePreparation(game_engine)
        await battle_prep.prepare_for_battle()

    asyncio.run(main())