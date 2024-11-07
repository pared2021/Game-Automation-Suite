import asyncio
from typing import Dict, Any, Optional
from ..device.emulator_manager import emulator_manager
from ..ai.advanced_decision_maker import advanced_decision_maker
from ..scene_understanding.advanced_scene_analyzer import advanced_scene_analyzer
from ..nlp.advanced_language_processor import advanced_language_processor
from utils.logger import detailed_logger
from utils.error_handler import log_exception, GameAutomationError
from utils.performance_monitor import performance_monitor

class GameEngine:
    """
    游戏引擎核心类
    负责协调各个核心组件的工作，处理游戏状态和控制流
    """
    def __init__(self):
        self.logger = detailed_logger
        self.emulator = emulator_manager
        self.decision_maker = advanced_decision_maker
        self.scene_analyzer = advanced_scene_analyzer
        self.language_processor = advanced_language_processor
        self.performance_monitor = performance_monitor
        self.stop_automation = False
        self._game_state = {}

    @log_exception
    async def initialize(self) -> None:
        """初始化游戏引擎及其组件"""
        try:
            # 初始化设备
            await self.emulator.connect()
            
            # 初始化AI组件
            await self.decision_maker.initialize()
            await self.scene_analyzer.initialize()
            await self.language_processor.initialize()
            
            self.logger.info("Game Engine initialized successfully")
        except Exception as e:
            raise GameAutomationError(f"Failed to initialize game engine: {str(e)}")

    @log_exception
    async def update_game_state(self) -> Dict[str, Any]:
        """
        更新并返回当前游戏状态
        包括屏幕信息、文本识别结果等
        """
        try:
            screen = await self.emulator.capture_screen()
            text = await self.language_processor.recognize_text(screen)
            scene_info = await self.scene_analyzer.analyze_game_scene(screen, text)
            
            self._game_state = {
                'screen': screen,
                'text': text,
                'scene_info': scene_info,
                'timestamp': asyncio.get_event_loop().time()
            }
            
            return self._game_state
        except Exception as e:
            raise GameAutomationError(f"Failed to update game state: {str(e)}")

    @log_exception
    async def execute_action(self, action: Dict[str, Any]) -> None:
        """
        执行游戏动作
        :param action: 包含动作类型和参数的字典
        """
        try:
            self.logger.info(f"Executing action: {action}")
            
            if action['type'] == 'tap':
                await self.emulator.tap(action['x'], action['y'])
            elif action['type'] == 'swipe':
                await self.emulator.swipe(
                    action['start_x'], 
                    action['start_y'],
                    action['end_x'], 
                    action['end_y']
                )
            elif action['type'] == 'text_input':
                await self.emulator.input_text(action['text'])
            else:
                raise GameAutomationError(f"Unknown action type: {action['type']}")
                
        except Exception as e:
            raise GameAutomationError(f"Failed to execute action: {str(e)}")

    @log_exception
    async def run_game_loop(self) -> None:
        """运行游戏主循环"""
        try:
            self.logger.info("Starting game loop")
            self.stop_automation = False
            
            while not self.stop_automation:
                # 更新游戏状态
                game_state = await self.update_game_state()
                
                # 获取AI决策
                decision = await self.decision_maker.make_decision(game_state)
                
                # 执行决策动作
                if decision:
                    await self.execute_action(decision)
                
                # 性能监控
                await self.performance_monitor.record_metrics(game_state)
                
                # 控制循环间隔
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error in game loop: {str(e)}")
            raise GameAutomationError(f"Game loop error: {str(e)}")
        finally:
            self.stop_automation = True

    @log_exception
    async def stop(self) -> None:
        """停止游戏自动化"""
        self.stop_automation = True
        self.logger.info("Game automation stopped")

    @property
    def game_state(self) -> Dict[str, Any]:
        """获取当前游戏状态"""
        return self._game_state

    @log_exception
    async def handle_error(self, error: Exception) -> None:
        """
        处理游戏运行时错误
        :param error: 错误对象
        """
        self.logger.error(f"Game error occurred: {str(error)}")
        await self.performance_monitor.record_error(error)
        # TODO: 实现更多错误恢复机制

# 创建全局实例
game_engine = GameEngine()
