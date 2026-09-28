"""Study Buddy project detection and initialization.

A Study Buddy project is any directory containing a ``.studybuddy/`` folder. Detection
walks up parent directories the same way git finds ``.git/``.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR_NAME = ".studybuddy"
PROJECT_CONFIG_NAME = "config.json"


class ProjectAlreadyInitializedError(FileExistsError):
    """Raised when ``init_project`` targets a directory that already has ``.studybuddy/``."""


def find_project_root(start: Path) -> Path | None:
    """Return the nearest directory at or above ``start`` containing ``.studybuddy/``.

    Args:
        start: Directory to begin searching from.

    Returns:
        The project root, or ``None`` if no ancestor is a Study Buddy project.
    """
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / PROJECT_DIR_NAME).is_dir():
            return candidate
    return None


def init_project(path: Path) -> Path:
    """Create ``.studybuddy/config.json`` inside ``path``.

    Args:
        path: Directory to turn into a Study Buddy project.

    Returns:
        The path of the created ``.studybuddy/`` directory.

    Raises:
        ProjectAlreadyInitializedError: If ``path/.studybuddy`` already exists.
    """
    project_dir = path / PROJECT_DIR_NAME
    if project_dir.exists():
        raise ProjectAlreadyInitializedError(f"{project_dir} already exists")
    project_dir.mkdir(parents=True)
    (project_dir / PROJECT_CONFIG_NAME).write_text(json.dumps({}, indent=2) + "\n")
    return project_dir
