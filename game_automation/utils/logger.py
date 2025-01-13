"""Logging utilities."""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """Setup logging configuration
    
    Args:
        log_dir: Log directory
        log_level: Log level
        max_size: Maximum log file size
        backup_count: Number of backup files
    """
    # Create log directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Create handlers
    log_file = os.path.join(
        log_dir,
        f"game_automation_{datetime.now().strftime('%Y%m%d')}.log"
    )
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=backup_count
    )
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
def get_logger(name: str) -> logging.Logger:
    """Get logger instance
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
