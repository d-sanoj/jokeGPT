"""Logging setup."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a plain, readable log format for the service."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)
