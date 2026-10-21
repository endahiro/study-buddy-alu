"""Tests for safe project file access and fix proposals."""

from pathlib import Path

import pytest

from studybuddy.workspace import (
    MAX_FILE_BYTES,
    WorkspaceError,
    extract_code_block,
    propose_fix,
    read_source,
    resolve_in_project,
)


def test_read_source_reads_project_file(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("print('hi')\n")
    assert read_source(tmp_path, "hello.py") == "print('hi')\n"


def test_paths_outside_the_project_are_refused(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (tmp_path / "secret.txt").write_text("nope")
    with pytest.raises(WorkspaceError, match="outside the project"):
        resolve_in_project(project, "../secret.txt")
    with pytest.raises(WorkspaceError, match="outside the project"):
        resolve_in_project(project, str(tmp_path / "secret.txt"))


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError, match="No such file: nope.py"):
        read_source(tmp_path, "nope.py")


def test_large_and_binary_files_are_refused(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * (MAX_FILE_BYTES + 1))
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(WorkspaceError, match="too large"):
        read_source(tmp_path, "big.txt")
    with pytest.raises(WorkspaceError, match="not a UTF-8"):
        read_source(tmp_path, "blob.bin")


def test_extract_code_block_takes_the_last_block() -> None:
    reply = "Use `x`:\n```python\nsnippet\n```\nFull file:\n```python\nfull = 1\n```\n"
    assert extract_code_block(reply) == "full = 1\n"
    assert extract_code_block("no code here") is None


def test_propose_fix_builds_a_diff(tmp_path: Path) -> None:
    (tmp_path / "add.py").write_text("def add(a, b):\n    return a - b\n")
    reply = "It subtracts.\n```python\ndef add(a, b):\n    return a + b```"
    proposal = propose_fix(tmp_path, "add.py", reply)
    assert proposal is not None
    assert proposal.new_text == "def add(a, b):\n    return a + b\n"
    assert "-    return a - b" in proposal.diff
    assert "+    return a + b" in proposal.diff
    assert proposal.path == (tmp_path / "add.py").resolve()


def test_propose_fix_returns_none_without_changes(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("x = 1\n")
    assert propose_fix(tmp_path, "ok.py", "Looks fine!") is None
    assert propose_fix(tmp_path, "ok.py", "```python\nx = 1\n```") is None
