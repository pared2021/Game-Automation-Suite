import asyncio
from game_automation.core.engine import game_engine
from game_automation.gui.main_window import run_gui
from utils.logger import detailed_logger
from utils.config_manager import config_manager

async def run_gui_async():
    run_gui()

async def main():
    try:
        await game_engine.initialize()
        asyncio.create_task(game_engine.run_game_loop())
        
        # 运行自动化测试
        if config_manager.get('testing.run_automated_tests', False):
            await game_engine.run_automated_tests()
        
        # 性能优化
        if config_manager.get('performance.auto_optimize', False):
            await game_engine.optimize_performance()
        
        # 运行GUI
        asyncio.create_task(run_gui_async())
        
    except Exception as e:
        detailed_logger.error(f"Error in main program: {str(e)}")
    finally:
        await game_engine.save_game_state()
        detailed_logger.info("Game automation stopped")

if __name__ == "__main__":
    asyncio.run(main())
