import os
import sys
import logging

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Define unified log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logger():
    """
    Configures and returns a custom application logger.
    Logs are printed to console (stdout) and written to logs/app.log.
    """
    logger = logging.getLogger("voice_survey")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times during uvicorn hot-reload
    if logger.handlers:
        return logger

    # Console Handler (Stdout)
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler (logs/app.log)
    file_handler = logging.FileHandler("logs/app.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()
