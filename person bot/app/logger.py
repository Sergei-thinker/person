from __future__ import annotations

import logging
from typing import Final


_LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Return a configured logger.

    If level is not provided, INFO is used by default. The root configuration is
    applied only once.
    """

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger




