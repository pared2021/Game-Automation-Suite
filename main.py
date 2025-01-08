import sys
from PyQt6.QtWidgets import QApplication
from game_automation.gui.main_window import MainWindow
import threading
import asyncio
from game_automation.core.engine.game_engine import GameEngine
from utils.logger import detailed_logger
from utils.config_manager import config_manager

def run_game_engine(engine):
    async def game_loop():
        try:
            await engine.initialize()
            await engine.run_game_loop()
            
            # 运行自动化测试
            if config_manager.get('testing.run_automated_tests', False):
                await engine.run_automated_tests()
            
            # 性能优化
            if config_manager.get('performance.auto_optimize', False):
                await engine.optimize_performance()
                
        except Exception as e:
            detailed_logger.error(f"Error in game engine: {str(e)}")
        finally:
            await engine.cleanup()
            detailed_logger.info("Game engine stopped")
    
    asyncio.run(game_loop())

if __name__ == "__main__":
    # 首先创建QApplication实例
    app = QApplication(sys.argv)
    
    # 创建游戏引擎实例
    game_engine = GameEngine()
    
    # 在单独的线程中运行游戏引擎
    engine_thread = threading.Thread(target=run_game_engine, args=(game_engine,))
    engine_thread.daemon = True  # 设置为守护线程，这样主程序退出时会自动结束
    engine_thread.start()
    
    # 创建并显示主窗口，传入游戏引擎实例
    window = MainWindow(game_engine)
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())
