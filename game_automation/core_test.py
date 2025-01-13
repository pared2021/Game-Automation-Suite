"""Core system test program."""

import sys
import asyncio
import logging
from datetime import datetime

from game_automation.core.events.event_manager import EventManager
from game_automation.core.task.task_executor import TaskExecutor
from game_automation.core.task.task_manager import TaskManager
from game_automation.core.task.task_adapter import TaskAdapter
from game_automation.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

async def test_core_system():
    """Test core system functionality"""
    try:
        # Setup logging
        setup_logging()
        logger.info("Starting core system test")
        
        # Initialize core components
        event_manager = EventManager()
        task_executor = TaskExecutor()
        task_manager = TaskManager()
        
        # Initialize task adapter
        task_adapter = TaskAdapter(
            event_manager,
            task_executor,
            task_manager
        )
        
        # Initialize components
        await task_executor.initialize()
        logger.info("Core components initialized")
        
        # Create test task
        task_id = await task_executor.add_task(
            name="Test Task",
            task_type="TEST",
            priority=1,
            params={"test_param": "test_value"}
        )
        logger.info(f"Created test task: {task_id}")
        
        # Wait for task execution
        await asyncio.sleep(5)
        
        # Get task status
        task = task_executor.get_task(task_id)
        if task:
            logger.info(f"Task status: {task.status}")
            logger.info(f"Task result: {task.result}")
        
        # Shutdown components
        await task_executor.shutdown()
        logger.info("Core system test completed")
        
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise

def main():
    """Main entry point"""
    try:
        # Run test
        asyncio.run(test_core_system())
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
