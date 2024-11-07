import asyncio
import yaml
from utils.logger import detailed_logger
from utils.error_handler import log_exception
from game_automation.image_recognition import ImageRecognition
from game_automation.ocr_prediction.ocr_utils import ocr_utils
from game_automation.controllers.task_utils import TaskUtils

class TaskExplorer:
    def __init__(self, game_engine, task_manager):
        self.logger = detailed_logger
        self.game_engine = game_engine
        self.task_manager = task_manager
        self.image_recognition = ImageRecognition()
        self.exploration_progress = {}
        self.current_task = None

        # Load resource paths from configuration file
        with open('config/resource_paths.yaml', 'r', encoding='utf-8') as file:
            self.resource_paths = yaml.safe_load(file)

    @log_exception
    async def explore_and_complete_tasks(self):
        try:
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
        except Exception as e:
            self.logger.error(f"Failed to explore and complete tasks: {e}")

    @log_exception
    async def explore_task(self, task):
        try:
            self.logger.info(f"Exploring task: {task['name']}")
            self.exploration_progress[task['id']] = 0
            
            if not await TaskUtils.check_prerequisites(task, self.task_manager):
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
        except Exception as e:
            self.logger.error(f"Failed to explore task {task['name']}: {e}")

    @log_exception
    async def navigate_to_task_screen(self, task):
        try:
            self.logger.info(f"Navigating to task screen for: {task['name']}")
            task_button_path = f"{self.resource_paths['task_buttons_path']}{task['id']}.png"
            task_button = await self.image_recognition.find_game_object(await self.game_engine.capture_screen(), task_button_path)
            if task_button:
                await self.game_engine.touch_controller.tap(task_button[0], task_button[1])
                await asyncio.sleep(2)  # 等待页面加载
            else:
                self.logger.warning(f"Could not find task button for: {task['name']}")
        except Exception as e:
            self.logger.error(f"Failed to navigate to task screen for {task['name']}: {e}")

    @log_exception
    async def analyze_task_requirements(self, task):
        try:
            self.logger.info(f"Analyzing requirements for task: {task['name']}")
            screen = await self.game_engine.capture_screen()
            task_text = await ocr_utils.recognize_text(screen)
            
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
        except Exception as e:
            self.logger.error(f"Failed to analyze requirements for task {task['name']}: {e}")

    @log_exception
    async def execute_task_actions(self, task):
        try:
            self.logger.info(f"Executing actions for task: {task['name']}")
            requirements = task.get('requirements', {})
            
            for item, count in requirements.items():
                if "收集" in task['description']:
                    await self.collect_item(item, count)
                elif "击败" in task['description']:
                    await self.defeat_enemy(item, count)
        except Exception as e:
            self.logger.error(f"Failed to execute actions for task {task['name']}: {e}")

    @log_exception
    async def check_task_completion(self, task):
        try:
            self.logger.info(f"Checking completion for task: {task['name']}")
            screen = await self.game_engine.capture_screen()
            completion_indicator_path = self.resource_paths['task_complete_indicator_path']
            completion_indicator = await self.image_recognition.find_game_object(screen, completion_indicator_path)
            if completion_indicator:
                await self.task_manager.complete_task_by_id(task['id'])
                self.logger.info(f"Task completed: {task['name']}")
            else:
                self.logger.info(f"Task not yet completed: {task['name']}")
        except Exception as e:
            self.logger.error(f"Failed to check completion for task {task['name']}: {e}")

    @log_exception
    async def check_resources_for_task(self, task):
        try:
            self.logger.info(f"Checking resources for task: {task['name']}")
            game_state = await self.game_engine.get_game_state()
            required_resources = task.get('required_resources', {})
            
            for resource, amount in required_resources.items():
                if game_state.get(resource, 0) < amount:
                    self.logger.info(f"Insufficient {resource} for task: {task['name']}")
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to check resources for task {task['name']}: {e}")

    @log_exception
    async def check_excess_resources(self):
        try:
            self.logger.info("Checking for excess resources")
            game_state = await self.game_engine.get_game_state()
            threshold = self.game_engine.config.get('excess_resource_threshold', {})
            
            for resource, limit in threshold.items():
                if game_state.get(resource, 0) > limit:
                    self.logger.info(f"Excess {resource} detected")
                    return True
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to check excess resources: {e}")

    def get_exploration_progress(self):
        return self.exploration_progress

    def get_current_task(self):
        return self.current_task

task_explorer = TaskExplorer(None, None)  # 初始化时传入 None，后续在 GameEngine 中设置正确的引用
