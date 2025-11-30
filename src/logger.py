# src/logger.py
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "pipeline",
    log_dir: Optional[Path] = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Create or retrieve a project-wide logger.

    This factory configures:
    - Console logging
    - Rotating file logging under ``results/logs/`` by default

    Parameters
    ----------
    name : str
        Logger name (usually module or script name).
    log_dir : Path, optional
        Directory to store log files. If None, defaults to:
        ``<project_root>/results/logs``.
    level : str
        Logging level name, e.g. ``"INFO"``, ``"DEBUG"``, ``"WARNING"``.

    Returns
    -------
    logging.Logger
        Configured logger instance (singleton per name).
    """
    logger = logging.getLogger(name)

    # If logger already has handlers, assume it is configured
    if logger.handlers:
        return logger

    # Resolve default log directory if none is provided
    if log_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        log_dir = project_root / "results" / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    # Resolve log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,  # ~5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Do not propagate to root logger to avoid duplicate logs
    logger.propagate = False
    logger.debug("Logger initialized.")
    return logger
