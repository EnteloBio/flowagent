import logging
import sys
from pathlib import Path

def setup_logging(log_dir: str = None) -> None:
    """Setup logging configuration."""
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    command_formatter = logging.Formatter(
        '%(asctime)s - COMMAND - %(message)s'
    )
    
    # Setup handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(detailed_formatter)
    
    # Setup loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    # Setup command logger
    command_logger = logging.getLogger("cognomic.commands")
    command_logger.setLevel(logging.INFO)
    
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # File handler for general logs
        file_handler = logging.FileHandler(
            log_path / "cognomic.log"
        )
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # File handler for command logs
        command_file_handler = logging.FileHandler(
            log_path / "commands.log"
        )
        command_file_handler.setFormatter(command_formatter)
        command_logger.addHandler(command_file_handler)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
