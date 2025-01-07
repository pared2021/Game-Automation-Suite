import asyncio
import sys
import argparse
import logging
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from game_automation.core.engine.game_engine import GameEngine
from game_automation.config.config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('game_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main(test_mode=False):
    engine = None  # Initialize engine variable outside try block
    try:
        # Check required configurations
        config_manager = ConfigManager()
        if not config_manager.validate_configs():
            logger.error("Required configurations are missing or invalid. Please check config.yaml")
            logger.info("Available configurations: %s", config_manager.get_config_summary())
            return

        logger.info("Initializing game engine...")
        engine = GameEngine(test_mode=test_mode)
        await engine.initialize()
        
        logger.info("Game engine started. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down the game engine...")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}", exc_info=True)
        if isinstance(e, RuntimeError) and "engine initialization" in str(e).lower():
            logger.error("Engine initialization failed. Please check system requirements and configurations.")
    finally:
        if engine is not None:
            try:
                await engine.cleanup()
                logger.info("Game engine shutdown complete.")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        else:
            logger.info("No engine instance to cleanup")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Game Automation Engine')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    try:
        asyncio.run(main(test_mode=args.test))
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
