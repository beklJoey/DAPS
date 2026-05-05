"""Centralised logging configuration for the contrail segmentation project."""

import logging
import sys
from pathlib import Path
from typing import Optional

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL: int = logging.INFO


def get_logger(
    name: str,
    level: int = DEFAULT_LEVEL,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Build and return a named logger with console (and optional file) output.

    Calling this function multiple times with the same name returns the
    same logger instance without duplicating handlers.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level (e.g. ``logging.DEBUG``).
        log_file: Optional path; when provided a FileHandler is added.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_global_level(level: int) -> None:
    """Adjust the level of the root logger and all child loggers.

    Args:
        level: New logging level to apply globally.
    """
    logging.getLogger().setLevel(level)
    for name in logging.root.manager.loggerDict:  # pylint: disable=no-member
        logging.getLogger(name).setLevel(level)
