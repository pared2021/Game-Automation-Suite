import asyncio
from utils.logger import setup_logger
from utils.error_handler import log_exception
from game_automation.image_recognition import ImageRecognition
from game_automation.ocr_prediction.ocr_utils import ocr_utils

class TaskExplorer:
    def __init__(self, game_engine, task_manager):
        self.logger = setup_logger('task_explorer')
        self.game_engine = game_engine
        self.task_manager = task_manager
        self.image_recognition = ImageRecognition()
        self.exploration_progress = {}
        self.current_task = None

    @log_exception
    async def explore_and_complete_tasks(self):
        tasks = await self.task_manager.get_task_status()
        total_tasks = sum(len(task_list) for task_list in tasks.values())
        completed_tasks = 0

        for task_type, task_list in tasks.items():
            for task in task_list:
                if task['status'] == "未完成":
                    self.current_task = task
                    await self.explore_task(task)
                    completed_tasks += 1
                    self.exploration_progress['overall'] = (completed_tasks / total_tasks) * 100
                self.current_task = None

    @log_exception
    async def explore_task(self, task):
        self.logger.info(f"Exploring task: {task['name']}")
        self.exploration_progress[task['id']] = 0
        
        if not await self.check_prerequisites(task):
            self.logger.info(f"Prerequisites not met for task: {task['name']}")
            return

        exploration_steps = [
            self.navigate_to_task_screen,
            self.analyze_task_requirements,
            self.execute_task_actions,
            self.check_task_completion
        ]

        for i, step in enumerate(exploration_steps):
            await step(task)
            self.exploration_progress[task['id']] = ((i + 1) / len(exploration_steps)) * 100

    async def check_prerequisites(self, task):
        for prereq in task['prerequisites']:
            if not await self.task_manager.is_prerequisite_completed(prereq):
                return False
        return True

    @log_exception
    async def navigate_to_task_screen(self, task):
        self.logger.info(f"Navigating to task screen for: {task['name']}")
        # 查找任务按钮
        task_button = await self.image_recognition.find_game_object(await self.game_engine.capture_screen(), f"assets/task_buttons/{task['id']}.png")
        if task_button:
            await self.game_engine.touch_controller.tap(task_button[0], task_button[1])
            await asyncio.sleep(2)  # 等待页面加载
        else:
            self.logger.warning(f"Could not find task button for: {task['name']}")

    @log_exception
    async def analyze_task_requirements(self, task):
        self.logger.info(f"Analyzing requirements for task: {task['name']}")
        screen = await self.game_engine.capture_screen()
        task_text = await ocr_utils.recognize_text(screen)
        
        # 解析任务文本以确定具体要求
        requirements = {}
        if "收集" in task_text:
            items = re.findall(r"收集(\d+)个([\w\s]+)", task_text)
            for count, item in items:
                requirements[item] = int(count)
        elif "击败" in task_text:
            enemies = re.findall(r"击败(\d+)个([\w\s]+)", task_text)
            for count, enemy in enemies:
                requirements[enemy] = int(count)
        
        task['requirements'] = requirements
        self.logger.info(f"Task requirements: {requirements}")

    @log_exception
    async def execute_task_actions(self, task):
        self.logger.info(f"Executing actions for task: {task['name']}")
        requirements = task.get('requirements', {})
        
        for item, count in requirements.items():
            if "收集" in task['description']:
                await self.collect_item(item, count)
            elif "击败" in task['description']:
                await self.defeat_enemy(item, count)

    async def collect_item(self, item, count):
        self.logger.info(f"Collecting {count} {item}")
        # 查找物品位置
        item_location = await self.image_recognition.find_game_object(await self.game_engine.capture_screen(), f"assets/items/{item}.png")
        if item_location:
            for _ in range(count):
                await self.game_engine.touch_controller.tap(item_location[0], item_location[1])
                await asyncio.sleep(1)
        else:
            self.logger.warning(f"Could not find {item} on the screen")

    async def defeat_enemy(self, enemy, count):
        self.logger.info(f"Defeating {count} {enemy}")
        for _ in range(count):
            # 查找敌人位置
            enemy_location = await self.image_recognition.find_game_object(await self.game_engine.capture_screen(), f"assets/enemies/{enemy}.png")
            if enemy_location:
                await self.game_engine.touch_controller.tap(enemy_location[0], enemy_location[1])
                # 等待战斗结束
                await asyncio.sleep(5)
            else:
                self.logger.warning(f"Could not find {enemy} on the screen")

    @log_exception
    async def check_task_completion(self, task):
        self.logger.info(f"Checking completion for task: {task['name']}")
        screen = await self.game_engine.capture_screen()
        completion_indicator = await self.image_recognition.find_game_object(screen, 'assets/task_complete_indicator.png')
        if completion_indicator:
            await self.task_manager.complete_task_by_id(task['id'])
            self.logger.info(f"Task completed: {task['name']}")
        else:
            self.logger.info(f"Task not yet completed: {task['name']}")

    @log_exception
    async def check_resources_for_task(self, task):
        self.logger.info(f"Checking resources for task: {task['name']}")
        game_state = await self.game_engine.get_game_state()
        required_resources = task.get('required_resources', {})
        
        for resource, amount in required_resources.items():
            if game_state.get(resource, 0) < amount:
                self.logger.info(f"Insufficient {resource} for task: {task['name']}")
                return False
        
        return True

    @log_exception
    async def check_excess_resources(self):
        self.logger.info("Checking for excess resources")
        game_state = await self.game_engine.get_game_state()
        threshold = self.game_engine.config.get('excess_resource_threshold', {})
        
        for resource, limit in threshold.items():
            if game_state.get(resource, 0) > limit:
                self.logger.info(f"Excess {resource} detected")
                return True
        
        return False

    def get_exploration_progress(self):
        return self.exploration_progress

    def get_current_task(self):
        return self.current_task

task_explorer = TaskExplorer(None, None)  # 初始化时传入 None，后续在 GameEngine 中设置正确的引用