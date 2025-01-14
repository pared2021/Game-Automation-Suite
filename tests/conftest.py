"""Pytest configuration file."""

import pytest
import asyncio
import logging
import pytest_asyncio

@pytest_asyncio.fixture(scope="function")
async def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
    
@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests."""
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure game automation logger
    logger = logging.getLogger('game_automation')
    logger.setLevel(logging.DEBUG)
    
    # Add a console handler if none exists
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
