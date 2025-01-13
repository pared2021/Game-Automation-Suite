"""Main entry point for the application."""

import sys
import asyncio
from PySide6.QtWidgets import QApplication

from game_automation.core.events.event_manager import EventManager
from game_automation.core.task.task_executor import TaskExecutor
from game_automation.core.task.task_manager import TaskManager
from game_automation.core.task.task_adapter import TaskAdapter
from game_automation.core.config.config_manager import ConfigManager
from game_automation.core.engine.game_engine import GameEngine
from game_automation.core.async_manager import AsyncManager
from game_automation.core.device.device_manager import DeviceManager
from game_automation.core.scene.scene_analyzer import AdvancedSceneAnalyzer
from game_automation.core.context.context_manager import ContextManager
from game_automation.core.error.error_manager import ErrorManager
from game_automation.gui.main_window import MainWindow
from game_automation.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

def main():
    """Main entry point"""
    try:
        # Setup logging
        setup_logging()
        
        # Create application
        app = QApplication(sys.argv)
        
        # Initialize core components
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        event_manager = EventManager()
        async_manager = AsyncManager()
        device_manager = DeviceManager()
        scene_analyzer = AdvancedSceneAnalyzer()
        context_manager = ContextManager()
        error_manager = ErrorManager()
        
        task_executor = TaskExecutor()
        task_manager = TaskManager()
        
        # Initialize game engine
        game_engine = GameEngine(
            config=config,
            event_manager=event_manager,
            async_manager=async_manager,
            device_manager=device_manager,
            scene_analyzer=scene_analyzer,
            context_manager=context_manager,
            task_executor=task_executor,
            error_manager=error_manager
        )
        
        # Create and show main window
        window = MainWindow(
            game_engine=game_engine,
            task_executor=task_executor,
            event_manager=event_manager,
            task_manager=task_manager
        )
        window.show()
        
        # Run event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
