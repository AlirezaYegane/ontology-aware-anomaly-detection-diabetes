# src/logger.py
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional


def get_logger(
    name: str = "pipeline",
    log_dir: Optional[Path] = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Simple project-wide logger factory.

    Parameters
    ----------
    name : str
        Logger name (usually module or script name).
    log_dir : Path, optional
        Directory to store log files. If None, defaults to:
            <project_root>/results/logs
    level : str
        Logging level as string, e.g. "INFO", "DEBUG", "WARNING".

    Returns
    -------
    logging.Logger
        Configured logger instance (singleton per name).
    """
    logger = logging.getLogger(name)

    # اگر قبلاً کانفیگ شده، همونو برگردون
    if logger.handlers:
        return logger

    # تعیین log_dir
    if log_dir is None:
        base_dir = Path(__file__).resolve().parent.parent  # پروژه
        log_dir = base_dir / "results" / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    # سطح لاگ
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(lvl)

    # formatter مشترک
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file handler (rotating)
    fh = RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,   # ~5MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(lvl)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    logger.debug("Logger initialized.")
    return logger
