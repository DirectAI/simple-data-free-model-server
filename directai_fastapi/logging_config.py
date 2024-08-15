import os
import time
import logging
formatter = logging.Formatter(
    '%(asctime)s,%(msecs)03d %(levelname)-8s [%(pathname)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S'
)

start_time = time.time()
assert isinstance(formatter._fmt, str)


# Find the most recently created directory in 'logs'
logs_dir = 'logs'
subdirs = [os.path.join(logs_dir, d) for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
latest_subdir = max(subdirs, key=os.path.getctime)

# Update the logging configuration to write to the most recently created directory
log_filename = os.path.join(latest_subdir, f"fastapi_runtime_{start_time}.log")
logging.basicConfig(
    filename=log_filename,
    format=formatter._fmt,
    datefmt=formatter.datefmt,
    filemode='a',
    level=logging.INFO
)

def logging_level_from_str(log_level_str: str) -> int:
    # see docs: https://docs.python.org/3/library/logging.html#levels
    if log_level_str == "NOTSET":
        return 0
    elif log_level_str == "DEBUG":
        return 10
    elif log_level_str == "WARNING":
        return 30
    elif log_level_str == "ERROR":
        return 40
    elif log_level_str == "CRITICAL":
        return 50
    else:
        # this is the same as INFO (default)
        return 20
    
logger = logging.getLogger(__name__)
log_level_str = os.getenv('LOGGING_LEVEL', 'INFO').upper()
console_handler = logging.StreamHandler()
console_log_level = logging_level_from_str(log_level_str)
console_handler.setLevel(console_log_level)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
