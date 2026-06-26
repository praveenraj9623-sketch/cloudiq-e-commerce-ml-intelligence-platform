"""Centralised logging for CloudIQ using loguru.

Provides a configured logger factory (:func:`get_logger`) and an exception
logging decorator (:func:`log_exceptions`). Sinks are configured exactly once
per process via a module-level guard to prevent duplicate log lines.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any, Callable, TypeVar

from loguru import logger

_F = TypeVar("_F", bound=Callable[..., Any])

# Module-level guard so we only attach sinks once per process.
_configured: bool = False

_CONSOLE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {extra[name]} | {message}"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {extra[name]} | {message}"
)


def _configure() -> None:
    """Attach console and file sinks once.

    Console: INFO and above, colorised.
    File: DEBUG and above, daily rotation, 14-day retention, zip compression.
    """
    global _configured
    if _configured:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format=_CONSOLE_FORMAT,
    )

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        "logs/cloudiq_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="14 days",
        compression="zip",
        format=_FILE_FORMAT,
        enqueue=True,
    )
    _configured = True


def get_logger(name: str) -> "logger.__class__":
    """Return a loguru logger bound to ``name``.

    Args:
        name: Logical logger name, surfaced in the ``{extra[name]}`` field.

    Returns:
        A bound loguru logger instance.
    """
    _configure()
    return logger.bind(name=name)


def log_exceptions(func: _F) -> _F:
    """Decorator that logs any raised exception with a traceback, then re-raises.

    Args:
        func: The callable to wrap.

    Returns:
        The wrapped callable.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = get_logger(func.__module__)
        try:
            return func(*args, **kwargs)
        except Exception:
            bound.opt(exception=True).error(
                "Exception in {}", func.__qualname__
            )
            raise

    return wrapper  # type: ignore[return-value]
