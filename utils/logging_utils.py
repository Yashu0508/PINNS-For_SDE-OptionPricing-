"""Logging utilities."""

import logging

from typing import Any

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level.
    """
    logging.basicConfig(level=getattr(logging, log_level.upper()))