import logging
import os
from logging.handlers import RotatingFileHandler

class DetailedLogger:
    def __init__(self, name, log_file='game_automation.log', level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

def setup_logger(name, log_file='logs/game_automation.log', level=logging.INFO):
    """Setup and return a logger instance
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        DetailedLogger: Configured logger instance
    """
    return DetailedLogger(name, log_file, level)

# Default logger instance
detailed_logger = DetailedLogger('game_automation', 'logs/game_automation.log')
