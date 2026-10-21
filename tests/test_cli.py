"""End-to-end tests for the typer CLI."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from studybuddy.cli import app
from studybuddy.llm import FreeModel, ProviderError
from studybuddy.llm.mock import CANNED_RESPONSE, MockProvider
from tests.fakes import ScriptedProvider

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


def test_bare_command_opens_chat_without_init(in_tmp: Path) -> None:
    (in_tmp / "a.py").write_text("x = 1\n")
    result = runner.invoke(app, [], input="/read a.py\nexit\n")
    assert result.exit_code == 0, result.output
    assert "Study Buddy" in result.output
    assert "Model: mock (mock)" in result.output
    assert "Shared a.py" in result.output


def test_bare_command_accepts_provider_option(in_tmp: Path) -> None:
    result = runner.invoke(app, ["-p", "gpt"])
    assert result.exit_code == 1
    assert "Unknown provider 'gpt'" in result.output


def test_chat_finds_project_in_parent_and_streams_mock_reply(in_tmp: Path) -> None:
    runner.invoke(app, ["init"])
    subdir = in_tmp / "src" / "deep"
    subdir.mkdir(parents=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(subdir)
        result = runner.invoke(app, ["chat"], input="hello\nexit\n")
    assert result.exit_code == 0, result.output
    assert "Model: mock (mock)" in result.output
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


BUGGY = "def add(a, b):\n    return a - b\n"
FIX_REPLY = "You subtract instead of adding.\n```python\ndef add(a, b):\n    return a + b\n```"


def use_scripted(monkeypatch: pytest.MonkeyPatch, *replies: str) -> ScriptedProvider:
    llm = ScriptedProvider(*replies)
    monkeypatch.setattr("studybuddy.cli.get_provider", lambda *args, **kwargs: llm)
    return llm


def test_fix_applies_changes_after_approval(in_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scripted(monkeypatch, FIX_REPLY)
    runner.invoke(app, ["init"])
    (in_tmp / "add.py").write_text(BUGGY)
    result = runner.invoke(app, ["chat"], input="/fix add.py it returns 0\ny\nexit\n")
    assert result.exit_code == 0, result.output
    assert "You subtract instead of adding." in result.output
    assert "+    return a + b" in result.output
    assert "Updated add.py" in result.output
    assert (in_tmp / "add.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_fix_is_not_applied_when_declined(in_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scripted(monkeypatch, FIX_REPLY)
    runner.invoke(app, ["init"])
    (in_tmp / "add.py").write_text(BUGGY)
    result = runner.invoke(app, ["chat"], input="/fix add.py\nn\nexit\n")
    assert result.exit_code == 0, result.output
    assert "Left the file unchanged" in result.output
    assert (in_tmp / "add.py").read_text() == BUGGY


def test_fix_without_code_block_changes_nothing(
    in_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_scripted(monkeypatch, "Looks correct to me!")
    runner.invoke(app, ["init"])
    (in_tmp / "add.py").write_text(BUGGY)
    result = runner.invoke(app, ["chat"], input="/fix add.py\nexit\n")
    assert "No code changes were proposed" in result.output
    assert (in_tmp / "add.py").read_text() == BUGGY


def test_read_then_ask_and_follow_up(in_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = use_scripted(monkeypatch, "It subtracts.", "Use +.")
    runner.invoke(app, ["init"])
    (in_tmp / "add.py").write_text(BUGGY)
    result = runner.invoke(
        app, ["chat"], input="/read add.py\nWhat does it do?\nHow to fix?\nexit\n"
    )
    assert result.exit_code == 0, result.output
    assert "Shared add.py" in result.output
    assert "return a - b" in llm.requests[0][-1].content
    assert len(llm.requests[1]) == 4  # system, first question, first answer, follow-up


def test_bad_paths_and_commands_keep_session_alive(in_tmp: Path) -> None:
    runner.invoke(app, ["init"])
    result = runner.invoke(
        app, ["chat"], input="/read ../../etc/passwd\n/fix nope.py\n/fix\n/wat\n/help\nexit\n"
    )
    assert result.exit_code == 0, result.output
    assert "outside the project" in result.output
    assert "No such file: nope.py" in result.output
    assert "Usage: /fix <file>" in result.output
    assert "Unknown command /wat" in result.output
    assert "/reset" in result.output
    assert "Bye!" in result.output
    assert "Traceback" not in result.output


# --- /model -------------------------------------------------------------------------------

FREE = [
    FreeModel("qwen2.5-coder:1.5b", "best all-round", True, 3),
    FreeModel("llama3.2:1b", "fast", True, 3),
    FreeModel("qwen3:1.7b", "asks questions", False, 3),
    FreeModel("qwen2.5-coder:3b", "careful", True, 6),
]


@pytest.fixture
def free_models(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Fake the Ollama model list, downloads, and provider switching."""
    seen: dict[str, object] = {"downloaded": [], "switched": []}

    def fake_download(name: str) -> Iterator[float]:
        seen["downloaded"].append(name)  # type: ignore[attr-defined]
        yield 0.5
        yield 1.0

    def fake_get_provider(preferred: str | None = None, model: str | None = None) -> object:
        seen["switched"].append((preferred, model))  # type: ignore[attr-defined]
        llm = ScriptedProvider("hi from the new model")
        llm.model = model or ""
        return llm

    monkeypatch.setattr("studybuddy.chat.list_free_models", lambda: FREE)
    monkeypatch.setattr("studybuddy.chat.total_ram_gb", lambda: 3.7)
    monkeypatch.setattr("studybuddy.chat.download_model", fake_download)
    monkeypatch.setattr("studybuddy.chat.get_provider", fake_get_provider)
    return seen


def test_model_lists_free_models(in_tmp: Path, free_models: dict[str, object]) -> None:
    result = runner.invoke(app, ["-p", "mock"], input="/model\nexit\n")
    assert result.exit_code == 0, result.output
    for model in FREE:
        assert model.name in result.output
    assert "ready" in result.output
    assert "not downloaded" in result.output


def test_model_switches_by_number_and_keeps_chatting(
    in_tmp: Path, free_models: dict[str, object]
) -> None:
    result = runner.invoke(app, ["-p", "mock"], input="/model 2\nhello\nexit\n")
    assert result.exit_code == 0, result.output
    assert free_models["switched"] == [("ollama", "llama3.2:1b")]
    assert free_models["downloaded"] == []
    assert "Now using llama3.2:1b" in result.output
    assert "hi from the new model" in result.output


def test_model_downloads_after_approval(in_tmp: Path, free_models: dict[str, object]) -> None:
    result = runner.invoke(app, ["-p", "mock"], input="/model qwen3:1.7b\ny\nexit\n")
    assert result.exit_code == 0, result.output
    assert free_models["downloaded"] == ["qwen3:1.7b"]
    assert free_models["switched"] == [("ollama", "qwen3:1.7b")]


def test_model_download_declined_keeps_current(
    in_tmp: Path, free_models: dict[str, object]
) -> None:
    result = runner.invoke(app, ["-p", "mock"], input="/model qwen3:1.7b\nn\nexit\n")
    assert "Kept the current model" in result.output
    assert free_models["downloaded"] == []
    assert free_models["switched"] == []


def test_model_bad_number_is_reported(in_tmp: Path, free_models: dict[str, object]) -> None:
    result = runner.invoke(app, ["-p", "mock"], input="/model 9\nexit\n")
    assert "There is no model #9" in result.output
    assert free_models["switched"] == []


def test_model_when_ollama_is_down_keeps_session(
    in_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def down() -> list[FreeModel]:
        raise ProviderError("Can't reach Ollama. Is it running? Start it with: ollama serve")

    monkeypatch.setattr("studybuddy.chat.list_free_models", down)
    result = runner.invoke(app, ["-p", "mock"], input="/model\nexit\n")
    assert result.exit_code == 0, result.output
    assert "ollama serve" in result.output
    assert "Bye!" in result.output


def test_model_list_flags_models_too_big_for_this_computer(
    in_tmp: Path, free_models: dict[str, object]
) -> None:
    result = runner.invoke(app, ["-p", "mock"], input="/model\nexit\n")
    assert "needs 6 GB RAM" in result.output


def test_too_big_model_needs_confirmation(in_tmp: Path, free_models: dict[str, object]) -> None:
    declined = runner.invoke(app, ["-p", "mock"], input="/model 4\nn\nexit\n")
    assert "has 3.7 GB" in declined.output
    assert "Kept the current model" in declined.output
    assert free_models["switched"] == []

    accepted = runner.invoke(app, ["-p", "mock"], input="/model 4\ny\nexit\n")
    assert "Now using qwen2.5-coder:3b" in accepted.output
    assert free_models["switched"] == [("ollama", "qwen2.5-coder:3b")]


def test_unknown_ram_never_blocks_switching(
    in_tmp: Path, free_models: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("studybuddy.chat.total_ram_gb", lambda: None)
    result = runner.invoke(app, ["-p", "mock"], input="/model 4\nexit\n")
    assert "Now using qwen2.5-coder:3b" in result.output
