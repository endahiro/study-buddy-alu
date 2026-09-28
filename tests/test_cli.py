"""End-to-end tests for the typer CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from studybuddy.cli import app
from studybuddy.llm.mock import CANNED_RESPONSE

runner = CliRunner()


@pytest.fixture
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run each test from an empty temporary directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.output
    assert "chat" in result.output


def test_init_creates_project_dir(in_tmp: Path) -> None:
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert (in_tmp / ".studybuddy" / "config.json").is_file()
    assert "Initialized" in result.output


def test_init_fails_cleanly_when_already_initialized(in_tmp: Path) -> None:
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 1
    assert "already" in result.output
    assert "Traceback" not in result.output


def test_chat_without_project_shows_helpful_error(in_tmp: Path) -> None:
    result = runner.invoke(app, ["chat"])
    assert result.exit_code == 1
    assert "No Study Buddy project found" in result.output
    assert "studybuddy init" in result.output


def test_chat_finds_project_in_parent_and_streams_mock_reply(in_tmp: Path) -> None:
    runner.invoke(app, ["init"])
    subdir = in_tmp / "src" / "deep"
    subdir.mkdir(parents=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(subdir)
        result = runner.invoke(app, ["chat"], input="hello\nexit\n")
    assert result.exit_code == 0, result.output
    assert "using provider: mock" in result.output
    assert CANNED_RESPONSE in result.output
