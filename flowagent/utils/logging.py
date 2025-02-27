import logging
import sys
from pathlib import Path
from typing import Optional, Union

def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> None:
    """Setup logging configuration.
    
    Args:
        log_dir: Optional directory for log files
        level: Logging level (default: logging.INFO)
    """
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    )
    command_formatter = logging.Formatter(
        '%(asctime)s [COMMAND] %(message)s'
    )
    
    # Setup console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(detailed_formatter)
    console_handler.setLevel(level)
    
    # Setup loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Setup command logger with its own handler for better visibility
    command_logger = logging.getLogger("flowagent.commands")
    command_logger.setLevel(level)
    command_logger.propagate = False  # Don't propagate to root logger
    
    command_console_handler = logging.StreamHandler(sys.stdout)
    command_console_handler.setFormatter(command_formatter)
    command_console_handler.setLevel(level)
    command_logger.addHandler(command_console_handler)
    
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # File handler for general logs
        file_handler = logging.FileHandler(
            log_path / "flowagent.log"
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
        
        # File handler for command logs
        command_file_handler = logging.FileHandler(
            log_path / "commands.log"
        )
        command_file_handler.setFormatter(command_formatter)
        command_file_handler.setLevel(level)
        command_logger.addHandler(command_file_handler)

def get_logger(name: str, level: Optional[Union[int, str]] = None) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name
        level: Optional logging level (default: None, uses root logger level)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)
    return logger
