import sys
import os
import schedule
import time
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_automation.game_engine import GameEngine
from utils.logger import setup_logger
from utils.data_handler import DataHandler

logger = setup_logger()
data_handler = DataHandler()

async def run_game_automation():
    try:
        game_engine = GameEngine('config/strategies.json')
        await game_engine.initialize()
        
        logger.info("Starting scheduled game automation...")
        await game_engine.run_game_loop()
    except Exception as e:
        logger.error(f"An error occurred during scheduled run: {str(e)}")
    finally:
        logger.info("Scheduled run completed.")

def scheduled_job():
    asyncio.run(run_game_automation())

def main():
    schedule.every().day.at("02:00").do(scheduled_job)  # Run daily at 2 AM
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()