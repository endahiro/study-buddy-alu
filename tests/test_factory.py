"""Tests for provider resolution."""

import builtins
from typing import Any

import pytest

from studybuddy.llm import factory
from studybuddy.llm.anthropic import AnthropicProvider
from studybuddy.llm.base import ProviderError
from studybuddy.llm.factory import get_provider
from studybuddy.llm.gemini import GeminiProvider
from studybuddy.llm.mock import MockProvider
from studybuddy.llm.ollama import OllamaProvider

# Captured at import time, before the autouse fixture in conftest.py patches it out.
real_ollama_probe = factory._ollama_is_running


def test_falls_back_to_mock_when_nothing_is_available() -> None:
    assert isinstance(get_provider(), MockProvider)


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("anthropic", AnthropicProvider),
        ("ollama", OllamaProvider),
        ("mock", MockProvider),
        ("MOCK", MockProvider),
    ],
)
def test_explicit_preference_builds_that_provider(name: str, cls: type) -> None:
    assert isinstance(get_provider(name), cls)


def test_model_override_reaches_the_provider() -> None:
    provider = get_provider("ollama", model="llama3.2:1b")
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "llama3.2:1b"


def test_model_override_applies_to_auto_detected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory, "_ollama_is_running", lambda *args, **kwargs: True)
    provider = get_provider(model="llama3.2:1b")
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "llama3.2:1b"


def test_model_is_ignored_by_mock() -> None:
    assert isinstance(get_provider("mock", model="anything"), MockProvider)


def test_explicit_gemini_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert isinstance(get_provider("gemini"), GeminiProvider)


def test_explicit_gemini_without_key_raises_provider_error() -> None:
    with pytest.raises(ProviderError, match="Gemini is not configured"):
        get_provider("gemini")


def test_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown provider 'gpt'"):
        get_provider("gpt")


def test_anthropic_key_wins_over_gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert isinstance(get_provider(), AnthropicProvider)


def test_gemini_key_is_used_without_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert isinstance(get_provider(), GeminiProvider)


def test_running_ollama_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "_ollama_is_running", lambda *args, **kwargs: True)
    assert isinstance(get_provider(), OllamaProvider)


def _block_import(monkeypatch: pytest.MonkeyPatch, blocked: str) -> None:
    """Make ``import <blocked>`` (and its submodules) fail, as if the SDK were missing."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == blocked or name.startswith(blocked + "."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(__import__("sys").modules, blocked, raising=False)


def test_missing_sdk_for_explicit_choice_gives_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_import(monkeypatch, "studybuddy.llm.anthropic")
    with pytest.raises(ProviderError, match=r"uv sync --extra anthropic"):
        get_provider("anthropic")


def test_missing_sdk_during_auto_detection_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _block_import(monkeypatch, "studybuddy.llm.anthropic")
    assert isinstance(get_provider(), MockProvider)


def test_ollama_probe_returns_false_when_nothing_listens() -> None:
    # Port 9 (discard) is essentially never serving HTTP.
    assert real_ollama_probe("http://127.0.0.1:9", timeout=0.2) is False


def test_resolution_order_is_documented() -> None:
    doc = get_provider.__doc__
    assert doc is not None
    markers = [
        "Explicit preference",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "localhost:11434",
        "MockProvider",
    ]
    positions = [doc.index(marker) for marker in markers]
    assert positions == sorted(positions), "resolution order in docstring is out of order"


def test_list_free_models_marks_installed_and_adds_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    from studybuddy.llm import ollama as ollama_provider

    monkeypatch.setattr(ollama_provider, "installed_models", lambda: ["llama3.2:1b", "mistral:7b"])
    models = factory.list_free_models()
    names = [m.name for m in models]
    assert names[: len(ollama_provider.FREE_MODELS)] == list(ollama_provider.FREE_MODELS)
    assert names[-1] == "mistral:7b"
    assert {m.name for m in models if m.installed} == {"llama3.2:1b", "mistral:7b"}
    assert all(m.min_ram_gb > 0 for m in models[: len(ollama_provider.FREE_MODELS)])
    assert models[-1].min_ram_gb == 0  # unknown for models we haven't measured


def test_free_model_helpers_explain_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_import(monkeypatch, "studybuddy.llm.ollama")
    with pytest.raises(ProviderError, match="uv sync --extra ollama"):
        factory.list_free_models()
    with pytest.raises(ProviderError, match="uv sync --extra ollama"):
        factory.download_model("x")
