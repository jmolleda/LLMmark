# llmmark/logger_config.py

import logging
import sys


class CustomFormatter(logging.Formatter):
    """A custom log formatter with colors and cleaner output for the application."""

    # ANSI escape codes for colors
    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    BLUE = "\x1b[34m"
    RESET = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: GREY + "[%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.INFO: "%(message)s",  # Clean format for INFO
        logging.WARNING: YELLOW + "[WARNING] %(message)s" + RESET,
        logging.ERROR: RED + "[ERROR] %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "[CRITICAL] %(message)s" + RESET,
    }

    # Special format for important headers in the main script
    HEADER_FORMAT = BLUE + "%(message)s" + RESET

    def format(self, record):
        if record.name == "__main__" and hasattr(record, "is_header"):
            log_fmt = self.HEADER_FORMAT
        else:
            log_fmt = self.FORMATS.get(record.levelno, "%(levelname)s: %(message)s")

        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging():
    """
    Configures the root logger for clean, colored output and silences noisy libraries.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())

    root_logger.addHandler(handler)

    # Silence noisy third-party libraries by setting their log level higher
    noisy_loggers = ["httpx", "opik", "urllib3", "asyncio", "comet_ml"]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
