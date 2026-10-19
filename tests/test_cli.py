"""End-to-end tests for the typer CLI."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from studybuddy.cli import app
from studybuddy.llm import ProviderError
from studybuddy.llm.mock import CANNED_RESPONSE, MockProvider

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


def test_chat_rejects_unknown_provider(in_tmp: Path) -> None:
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["chat", "--provider", "gpt"])
    assert result.exit_code == 1
    assert "Unknown provider 'gpt'" in result.output
    assert "Traceback" not in result.output


def test_chat_survives_a_failed_request(in_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_stream(self: object, messages: object, **kwargs: object) -> Iterator[str]:
        raise ProviderError("network down")
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(MockProvider, "stream", failing_stream)
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["chat", "-p", "mock"], input="hello\nexit\n")
    assert result.exit_code == 0, result.output
    assert "Error: network down" in result.output
    assert "Bye!" in result.output


def test_chat_passes_model_to_factory(in_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_get_provider(preferred: str | None = None, model: str | None = None) -> MockProvider:
        seen.update(preferred=preferred, model=model)
        return MockProvider(delay=0)

    monkeypatch.setattr("studybuddy.cli.get_provider", fake_get_provider)
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["chat", "-p", "ollama", "-m", "llama3.2:1b"], input="exit\n")
    assert result.exit_code == 0, result.output
    assert seen == {"preferred": "ollama", "model": "llama3.2:1b"}
