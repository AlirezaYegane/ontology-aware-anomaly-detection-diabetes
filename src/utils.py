"""
Environment and path utilities for local / Colab execution.

This module provides small helper functions to:

- Detect whether the code is running in a local environment or Google Colab.
- Add the project root (containing `src/`) to `sys.path`.
- Ensure a standard `results/` directory structure exists.

The primary entry point is `setup_paths()`, which is safe to call from
notebooks or scripts before importing from `src`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from .logger import get_logger


logger = get_logger(name="paths")


def setup_paths() -> Literal["colab", "local"]:
    """
    Configure `sys.path` and working directory for hybrid local/Colab execution.

    Behaviour
    ---------
    - Detects whether we are running inside Google Colab.
    - Tries to locate a project root directory that contains a `src/` folder.
    - In Colab, changes the current working directory to the detected project root.
    - Ensures the project root is present in `sys.path`.
    - Verifies that imports from `src` are possible.

    Returns
    -------
    env_type : {'colab', 'local'}
        Detected environment type.
    """
    # Detect Colab vs local
    try:
        import google.colab  # type: ignore  # noqa: F401

        in_colab = True
        env_type: Literal["colab", "local"] = "colab"
    except ImportError:
        in_colab = False
        env_type = "local"

    if in_colab:
        logger.info("Detected Google Colab environment.")
        project_root = _detect_project_root_colab()
    else:
        logger.info("Detected local environment.")
        project_root = _detect_project_root_local()

    # Add project root to sys.path if needed
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
        logger.info("Added project root to sys.path: %s", project_root_str)
    else:
        logger.debug("Project root already present in sys.path: %s", project_root_str)

    # Verify imports from src
    try:
        import src  # type: ignore  # noqa: F401

        logger.info("Successfully imported from 'src/'.")
    except ImportError as exc:
        logger.warning(
            "Could not import from 'src/': %s. Current sys.path starts with: %s",
            exc,
            sys.path[:3],
        )

    return env_type


def _detect_project_root_colab() -> Path:
    """
    Best-effort detection of project root in a Google Colab environment.

    The function looks for a directory that contains a `src/` folder, trying
    a small set of common paths under `/content`.

    Returns
    -------
    Path
        Detected project root. Falls back to `/content` if nothing better is found.
    """
    base = Path("/content")

    # Common patterns + dynamic search under /content
    candidates: list[Path] = []

    # First: any directory under /content whose name contains "ontology"
    try:
        if base.exists():
            for entry in base.iterdir():
                if entry.is_dir() and "ontology" in entry.name.lower():
                    candidates.append(entry)
    except Exception as exc:  # very defensive; Colab FS can be odd
        logger.debug("Error while scanning /content: %s", exc)

    # Then: a couple of explicit fallbacks
    candidates.extend(
        [
            base / "Ontology-aware Anomaly Detection Toy Pipeline",
            base,
        ]
    )

    # Select first candidate that has a src/ directory
    for root in candidates:
        if (root / "src").exists():
            logger.info("Using project root: %s", root)
            os.chdir(root)
            return root

    # Fallbacks if nothing above worked
    cwd = Path.cwd()
    if (cwd / "src").exists():
        logger.info("Using current directory as project root: %s", cwd)
        return cwd

    parent = cwd.parent
    if (parent / "src").exists():
        logger.info("Using parent directory as project root: %s", parent)
        os.chdir(parent)
        return parent

    # Last resort: /content itself
    logger.warning("Could not find a directory with 'src/'. Falling back to /content.")
    os.chdir(base)
    return base


def _detect_project_root_local() -> Path:
    """
    Detect the project root in a local environment.

    Heuristic
    ---------
    - If we are inside a `notebooks/` directory, use its parent as the root.
    - Otherwise, try the current directory; if it has no `src/`, try the parent.

    Returns
    -------
    Path
        Detected project root.
    """
    current_dir = Path.cwd()

    if current_dir.name == "notebooks" or "notebooks" in str(current_dir):
        project_root = current_dir.parent
    else:
        project_root = current_dir

    if not (project_root / "src").exists():
        parent = project_root.parent
        if (parent / "src").exists():
            project_root = parent

    logger.info("Local project root resolved to: %s", project_root)
    return project_root


def get_project_root() -> Path:
    """
    Return the project root directory as a Path object.

    The function assumes that the project root is the closest ancestor of the
    current working directory that contains a `src/` directory. If none is
    found, it falls back to the current working directory.
    """
    current = Path.cwd()

    if (current / "src").exists():
        return current

    if (current.parent / "src").exists():
        return current.parent

    return current


def ensure_results_dir() -> Path:
    """
    Ensure the `results/` directory structure exists.

    Creates (if needed):

    - results/
      - figures/
      - models/
      - reports/

    Returns
    -------
    Path
        Path to the `results/` directory.
    """
    project_root = get_project_root()
    results_dir = project_root / "results"

    for subdir in ("figures", "models", "reports"):
        (results_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.info("Results directory is ready at: %s", results_dir)
    return results_dir
