"""Shared fixtures: keep every test independent of the developer's environment."""

import pytest

from studybuddy.llm import factory


@pytest.fixture(autouse=True)
def hermetic_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide real API keys and any local Ollama server so auto-detection picks the mock."""
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(factory, "_ollama_is_running", lambda *args, **kwargs: False)
