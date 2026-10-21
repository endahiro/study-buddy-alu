"""Safe access to the student's files inside a Study Buddy project.

Everything the agent reads or writes goes through here, so it can never touch a file
outside the project root.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

# DESIGN: small local models have small context windows; refuse files that won't fit.
MAX_FILE_BYTES = 100_000

# Matches ```lang\n...``` fenced blocks; the language tag is optional.
CODE_BLOCK_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


class WorkspaceError(Exception):
    """Raised when a file can't be used: missing, outside the project, too big, or binary."""


@dataclass(frozen=True)
class FixProposal:
    """A corrected version of a file, waiting for the student's approval.

    Attributes:
        path: Absolute path of the file to change.
        new_text: The full corrected file contents.
        diff: Unified diff from the current contents to ``new_text``.
    """

    path: Path
    new_text: str
    diff: str


def resolve_in_project(root: Path, relative: str) -> Path:
    """Return the absolute path of ``relative`` (resolved from ``root``), kept inside ``root``.

    Raises:
        WorkspaceError: If the path escapes the project or is not an existing file.
    """
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise WorkspaceError(f"{relative} is outside the project; only files under {root} work.")
    if not path.is_file():
        raise WorkspaceError(f"No such file: {relative}")
    return path


def read_source(root: Path, relative: str) -> str:
    """Read a text file from the project.

    Raises:
        WorkspaceError: If the file is missing, outside the project, too large, or not UTF-8.
    """
    path = resolve_in_project(root, relative)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise WorkspaceError(f"{relative} is too large (limit {MAX_FILE_BYTES // 1000} KB).")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise WorkspaceError(f"{relative} is not a UTF-8 text file.") from None


def extract_code_block(text: str) -> str | None:
    """Return the contents of the last fenced code block in ``text``, or None if there is none."""
    # DESIGN: the last block, because the fix prompt asks for the full file at the end and
    # the explanation before it may quote small snippets.
    blocks = CODE_BLOCK_RE.findall(text)
    return blocks[-1] if blocks else None


def propose_fix(root: Path, relative: str, reply: str) -> FixProposal | None:
    """Build a ``FixProposal`` from the model's ``reply`` to a fix request.

    Returns:
        The proposal, or None if the reply has no code block or changes nothing.

    Raises:
        WorkspaceError: If the target file can no longer be read.
    """
    new_text = extract_code_block(reply)
    if new_text is None:
        return None
    old_text = read_source(root, relative)
    if old_text.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"
    if new_text == old_text:
        return None
    diff = "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
    )
    return FixProposal(path=resolve_in_project(root, relative), new_text=new_text, diff=diff)
