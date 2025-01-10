import sys
import asyncio
from pathlib import Path
import logging
from datetime import datetime

from game_automation.device.device_manager import DeviceManager
from game_automation.scene_understanding.advanced_scene_analyzer import AdvancedSceneAnalyzer
from game_automation.core.context_manager import ContextManager
from game_automation.core.task_executor import TaskExecutor
from game_automation.core.engine.game_engine import GameEngine
from game_automation.core.error.error_manager import (
    ErrorManager,
    GameAutomationError,
    ErrorCategory,
    ErrorSeverity
)
from game_automation.gui.main_window import run_gui

def setup_logging():
    """设置日志"""
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志文件
    log_file = log_dir / f"game_automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

async def initialize_components():
    """初始化组件"""
    try:
        # 创建错误管理器
        error_manager = ErrorManager()
        
        # 创建设备管理器
        device_manager = DeviceManager()
        await device_manager.initialize()
        
        # 创建场景分析器
        scene_analyzer = AdvancedSceneAnalyzer()
        await scene_analyzer.initialize()
        
        # 创建上下文管理器
        context_manager = ContextManager()
        
        # 创建任务执行器
        task_executor = TaskExecutor()
        await task_executor.initialize()
        
        # 创建游戏引擎
        game_engine = GameEngine(
            device_manager=device_manager,
            scene_analyzer=scene_analyzer,
            context_manager=context_manager,
            task_executor=task_executor
        )
        await game_engine.initialize()
        
        return game_engine, task_executor
        
    except Exception as e:
        raise GameAutomationError(
            message=f"Failed to initialize components: {str(e)}",
            error_code="MAIN001",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            original_error=e
        )

async def cleanup_components(
    game_engine: GameEngine,
    task_executor: TaskExecutor
):
    """清理组件
    
    Args:
        game_engine: 游戏引擎
        task_executor: 任务执行器
    """
    try:
        # 停止任务执行器
        await task_executor.stop()
        await task_executor.cleanup()
        
        # 停止游戏引擎
        await game_engine.stop()
        await game_engine.cleanup()
        
    except Exception as e:
        logging.error(f"Failed to cleanup components: {str(e)}")

async def main():
    """主函数"""
    try:
        # 设置日志
        setup_logging()
        logging.info("Starting Game Automation Suite")
        
        # 初始化组件
        game_engine, task_executor = await initialize_components()
        logging.info("Components initialized")
        
        try:
            # 运行GUI
            run_gui(game_engine, task_executor)
            
        except Exception as e:
            raise GameAutomationError(
                message=f"GUI error: {str(e)}",
                error_code="MAIN002",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                original_error=e
            )
            
        finally:
            # 清理组件
            await cleanup_components(game_engine, task_executor)
            
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 创建事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # 运行主函数
        loop.run_until_complete(main())
        
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt")
        
    except Exception as e:
        logging.error(f"Unhandled error: {str(e)}")
        
    finally:
        # 关闭事件循环
        loop.close()
        logging.info("Game Automation Suite stopped")
