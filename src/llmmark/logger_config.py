import logging
import sys
import os

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


def setup_logging(log_file_path=None):
    """
    Configures the root logger for clean, colored output and silences noisy libraries.
    If log_file_path is provided, logs will also be written to the specified file.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter())
    root_logger.addHandler(console_handler)

    if log_file_path:
        try:
            # Create directory if it does not exist
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging to file: {log_file_path}")
        except IOError as e:
            root_logger.error(f"Could not set up file logger to {log_file_path}: {e}")

    # Silence noisy third-party libraries by setting their log level higher
    noisy_loggers = ["httpx", "opik", "urllib3", "asyncio", "comet_ml"]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)