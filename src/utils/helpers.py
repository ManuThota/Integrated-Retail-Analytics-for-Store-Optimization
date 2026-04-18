"""
Utility Helpers Module
========================
Small, project-wide utility functions that do not belong to any specific
analytical domain.

Functions
──────────
``ensure_dir``        – create a directory (and parents) if it does not exist.
``format_duration``   – convert elapsed seconds to a human-readable string.
``log_section``       – write a prominent separator line to the logger.
``flatten_dict``      – recursively flatten a nested dict for easy reporting.
``safe_divide``       – division that gracefully handles zero-division.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dir(path: Path | str) -> Path:
    """Create *path* and any missing parent directories.

    Idempotent — safe to call even if *path* already exists.

    Args:
        path: Directory path to create.

    Returns:
        The resolved :class:`pathlib.Path` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_duration(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable string.

    Examples::

        format_duration(65)      → "1m 05s"
        format_duration(3661)    → "1h 01m 01s"
        format_duration(0.5)     → "0.50s"

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted string.
    """
    if seconds < 60:  # noqa: PLR2004
        return f"{seconds:.2f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:  # noqa: PLR2004
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def log_section(msg: str, width: int = 60, char: str = "═") -> None:
    """Write a prominent section header to the logger at INFO level.

    Args:
        msg:   Section title text.
        width: Total line width (default 60).
        char:  Border character (default ``═``).
    """
    border = char * width
    logger.info(border)
    logger.info(msg)
    logger.info(border)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested dictionary.

    Useful for converting nested config / metrics dicts into a flat
    structure suitable for ``pd.DataFrame`` or JSON serialisation.

    Args:
        d:          Nested dictionary to flatten.
        parent_key: Prefix already accumulated (used internally).
        sep:        Key separator (default ``'.'``).

    Returns:
        Flat dictionary with compound keys (e.g. ``'model.r2': 0.97``).

    Example::

        flatten_dict({'a': {'b': 1}, 'c': 2}) → {'a.b': 1, 'c': 2}
    """
    items: dict = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide *numerator* by *denominator*, returning *default* on zero-division.

    Args:
        numerator:   The dividend.
        denominator: The divisor.
        default:     Value returned when *denominator* is 0 (default 0.0).

    Returns:
        Division result or *default*.
    """
    if denominator == 0:
        logger.warning(
            "safe_divide: denominator is 0 for numerator=%.4f; returning %s.",
            numerator, default,
        )
        return default
    return numerator / denominator
