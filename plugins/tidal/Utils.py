import logging
import os
import sys


def setup_logger(name: str, log_file: str, config: dict):
    """
    Sets up a logger with a file handler and an optional console handler.

    Args:
        name: The name for the logger.
        log_file: The file to which logs should be written.
        config: Configuration dictionary with settings like LOG_DIRECTORY.

    Returns:
        The configured logger instance.
    """
    log_path = os.path.join(config.get("LOG_DIRECTORY", "logs"), log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        def _create_and_add_handler(handler_class, level, **kwargs):
            handler = handler_class(**kwargs)
            handler.setLevel(level)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        _create_and_add_handler(
            logging.FileHandler,
            level=logging.DEBUG,
            filename=log_path,
            mode="w",
            encoding="utf-8",
        )

        if config.get("ENABLE_CONSOLE_LOGGING", True):
            _create_and_add_handler(
                logging.StreamHandler,
                level=logging.INFO,
                stream=sys.stdout,
            )

    return logger
