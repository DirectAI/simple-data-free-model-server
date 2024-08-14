import logging
import time
import os

def setup_logger():
    server_start_time = time.time()

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        # Create handlers
        file_handler = logging.FileHandler(f'logs/local_fastapi.log')
        file_handler.setLevel(logging.DEBUG)

        # Get logging level from environment variable
        log_level_str = os.getenv('LOGGING_LEVEL', 'ERROR').upper()
        if log_level_str not in logging._nameToLevel:
            raise ValueError(f"Invalid logging level: {log_level_str}")

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level_str)

        # Create formatters and add them to handlers
        formatter = logging.Formatter(
            '%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Initialize the logger
logger = setup_logger()