"""User-level (global) configuration stored as JSON.

Location: ``$XDG_CONFIG_HOME/studybuddy/config.json``, defaulting to
``~/.config/studybuddy/config.json``. No fields are defined yet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def config_path() -> Path:
    """Return the path to the user config file, honouring ``XDG_CONFIG_HOME``."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "studybuddy" / "config.json"


def load_config() -> dict[str, Any]:
    """Load the user config, returning an empty dict if the file does not exist."""
    path = config_path()
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def save_config(config: dict[str, Any]) -> None:
    """Write ``config`` to the user config file, creating parent directories as needed."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
