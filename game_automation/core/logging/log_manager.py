import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
from ..base.service_base import ServiceBase

class LogManager(ServiceBase):
    """Centralized logging management service."""
    
    def __init__(self):
        super().__init__("LogManager")
        self._loggers: Dict[str, logging.Logger] = {}
        self._log_dir: Path = Path("logs")
        self._default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self._log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
    
    async def _on_start(self) -> None:
        """Initialize logging system on service start."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_root_logger()
        self.log_info("Logging system initialized")
    
    def _setup_root_logger(self) -> None:
        """Setup the root logger with default configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(self._default_format))
        root_logger.addHandler(console_handler)
        
        # File handler for main log file
        main_log_file = self._log_dir / "game_automation.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(self._default_format))
        root_logger.addHandler(file_handler)
        
        # Error file handler
        error_log_file = self._log_dir / "error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(self._default_format))
        root_logger.addHandler(error_handler)
        
        self._loggers["root"] = root_logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the specified name."""
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        self._loggers[name] = logger
        return logger
    
    def set_level(self, level: str, logger_name: Optional[str] = None) -> None:
        """Set logging level for specified logger or root logger."""
        if level.upper() not in self._log_levels:
            raise ValueError(f"Invalid log level: {level}")
        
        log_level = self._log_levels[level.upper()]
        if logger_name:
            self.get_logger(logger_name).setLevel(log_level)
        else:
            logging.getLogger().setLevel(log_level)
        
        self.log_info(f"Set log level to {level} for logger: {logger_name or 'root'}")
    
    def add_file_handler(self, filename: str, logger_name: Optional[str] = None,
                        level: str = "INFO", max_bytes: int = 10*1024*1024,
                        backup_count: int = 5) -> None:
        """Add a file handler to the specified logger."""
        log_file = self._log_dir / filename
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        handler.setLevel(self._log_levels[level.upper()])
        handler.setFormatter(logging.Formatter(self._default_format))
        
        logger = self.get_logger(logger_name) if logger_name else logging.getLogger()
        logger.addHandler(handler)
        
        self.log_info(f"Added file handler {filename} to logger: {logger_name or 'root'}")
    
    async def process(self, input_data: Dict[str, Any]) -> Any:
        """Process logging configuration updates."""
        action = input_data.get('action')
        
        if action == 'set_level':
            level = input_data.get('level', 'INFO')
            logger_name = input_data.get('logger_name')
            self.set_level(level, logger_name)
            return {'status': 'success', 'message': f'Log level set to {level}'}
        
        elif action == 'add_handler':
            filename = input_data.get('filename')
            logger_name = input_data.get('logger_name')
            level = input_data.get('level', 'INFO')
            max_bytes = input_data.get('max_bytes', 10*1024*1024)
            backup_count = input_data.get('backup_count', 5)
            
            self.add_file_handler(filename, logger_name, level, max_bytes, backup_count)
            return {'status': 'success', 'message': f'Added handler {filename}'}
        
        else:
            raise ValueError(f"Unsupported action: {action}")
    
    async def validate(self, data: Any) -> bool:
        """Validate logging configuration data."""
        if not isinstance(data, dict):
            return False
        
        required_fields = ['action']
        return all(field in data for field in required_fields)
    
    def create_audit_log(self, event: str, details: Dict[str, Any]) -> None:
        """Create an audit log entry."""
        audit_logger = self.get_logger('audit')
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'details': details
        }
        audit_logger.info(json.dumps(audit_entry))
    
    def get_recent_logs(self, logger_name: Optional[str] = None,
                       level: str = "INFO", limit: int = 100) -> list:
        """Get recent log entries for the specified logger."""
        log_file = self._log_dir / (f"{logger_name}.log" if logger_name else "game_automation.log")
        if not log_file.exists():
            return []
        
        logs = []
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f.readlines()[-limit:]:
                    logs.append(line.strip())
        except Exception as e:
            self.log_error(f"Error reading log file: {log_file}", e)
        
        return logs
    
    def cleanup_old_logs(self, days: int = 30) -> None:
        """Clean up log files older than specified days."""
        current_time = datetime.now()
        for log_file in self._log_dir.glob("*.log*"):
            file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            if (current_time - file_time).days > days:
                try:
                    log_file.unlink()
                    self.log_info(f"Deleted old log file: {log_file}")
                except Exception as e:
                    self.log_error(f"Error deleting log file: {log_file}", e)
