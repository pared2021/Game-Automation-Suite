import argparse
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_automation.game_engine import GameEngine
from utils.logger import detailed_logger
from utils.data_handler import DataHandler
from utils.performance_monitor import performance_monitor

logger = detailed_logger
data_handler = DataHandler()

async def main(config_file, strategy_file):
    try:
        game_engine = GameEngine()
        await game_engine.initialize()
        await game_engine.load_strategy(strategy_file)
        
        logger.info("Game Engine initialized. Starting game loop...")
        await game_engine.run_game_loop()
    except Exception as e:
        logger.error(f"An error occurred in the main loop: {str(e)}")
    finally:
        logger.info("Shutting down...")
        await data_handler.close()
        performance_monitor.plot_performance()
        performance_monitor.log_performance(logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game Automation Launcher")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the configuration file")
    parser.add_argument("--strategy", default="config/strategies.json", help="Path to the strategy file")
    args = parser.parse_args()

    asyncio.run(main(args.config, args.strategy))
