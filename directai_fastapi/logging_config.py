import os
import time
import sys
import logging
formatter = logging.Formatter(
    '%(asctime)s,%(msecs)03d %(levelname)-8s [%(pathname)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S'
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

def grab_importing_fp() -> str:
    return sys.modules['__main__'].__file__

# Find the most recently created directory in 'logs'
logs_dir = 'logs'
subdirs = [os.path.join(logs_dir, d) for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
latest_subdir = max(subdirs, key=os.path.getctime)

# find the file t hat imported this to build a descriptive log filename
main_file = grab_importing_fp()
if main_file is None:
    log_filename = ""
elif isinstance(main_file, str):
    main_file = main_file.replace('/', '_')
log_filename = os.path.join(latest_subdir, f"{main_file}_runtime.log")
assert isinstance(formatter._fmt, str)
logging.basicConfig(
    filename=log_filename,
    format=formatter._fmt,
    datefmt=formatter.datefmt,
    filemode='a',
    level=logging.INFO
)
    
logger = logging.getLogger(__name__)
log_level_str = os.getenv('LOGGING_LEVEL', 'INFO').upper()
console_handler = logging.StreamHandler()
console_log_level = logging_level_from_str(log_level_str)
console_handler.setLevel(console_log_level)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
