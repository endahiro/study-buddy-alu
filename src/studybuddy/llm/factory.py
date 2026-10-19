"""Provider resolution: the single place that decides which ``LLMProvider`` to use.

Code outside ``studybuddy.llm`` must obtain providers through ``get_provider()`` and never
import a concrete provider module directly.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

from studybuddy.llm.base import LLMProvider, ProviderError
from studybuddy.llm.mock import MockProvider

OLLAMA_URL = "http://localhost:11434"
PROVIDER_NAMES = ("anthropic", "gemini", "ollama", "mock")


def _ollama_is_running(base_url: str = OLLAMA_URL, timeout: float = 0.5) -> bool:
    """Return True if an Ollama server answers at ``base_url``."""
    # DESIGN: stdlib only, so detection works even when the ollama SDK isn't installed.
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, OSError):
        return False


def _build(name: str, model: str | None = None) -> LLMProvider:
    """Construct the provider called ``name``, importing its SDK lazily.

    ``model`` overrides the provider's default model; the mock provider ignores it.

    Raises:
        ProviderError: If the provider's optional SDK is not installed.
        ValueError: If ``name`` is not a known provider.
    """
    try:
        if name == "anthropic":
            from studybuddy.llm.anthropic import DEFAULT_MODEL, AnthropicProvider

            return AnthropicProvider(model=model or DEFAULT_MODEL)
        if name == "gemini":
            from studybuddy.llm.gemini import DEFAULT_MODEL, GeminiProvider

            return GeminiProvider(model=model or DEFAULT_MODEL)
        if name == "ollama":
            from studybuddy.llm.ollama import DEFAULT_MODEL, OllamaProvider

            return OllamaProvider(model=model or DEFAULT_MODEL)
    except ImportError as exc:
        raise ProviderError(
            f"The {name} provider needs its SDK. Install it with: "
            f"uv sync --extra {name}  (or pip install 'studybuddy[{name}]')"
        ) from exc
    if name == "mock":
        return MockProvider()
    raise ValueError(f"Unknown provider {name!r}; choose one of: {', '.join(PROVIDER_NAMES)}")


def get_provider(preferred: str | None = None, model: str | None = None) -> LLMProvider:
    """Resolve and construct the LLM provider to use.

    Resolution order:
        1. Explicit preference — ``preferred`` (e.g. ``"anthropic"``, ``"gemini"``,
           ``"ollama"``, ``"mock"``).
        2. Environment API keys — ``ANTHROPIC_API_KEY``, then ``GEMINI_API_KEY``.
        3. Local Ollama server at ``localhost:11434``.
        4. ``MockProvider`` fallback.

    Real providers are imported lazily inside their branch so a missing optional SDK
    never breaks the mock path.

    Args:
        preferred: Optional provider name requested by the user or config.
        model: Optional model name passed to whichever real provider is chosen.

    Returns:
        A ready-to-use ``LLMProvider``.

    Raises:
        ValueError: If ``preferred`` is not a known provider name.
        ProviderError: If ``preferred`` names a provider whose SDK is not installed.
    """
    # 1. Explicit preference: failures are reported, never silently replaced.
    if preferred is not None:
        return _build(preferred.lower(), model)

    # 2-3. Auto-detection.
    # DESIGN: a detected provider whose SDK is missing is skipped rather than raised, so
    # a stray env var can't stop Study Buddy from starting.
    candidates = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        candidates.append("anthropic")
    if os.environ.get("GEMINI_API_KEY"):
        candidates.append("gemini")
    for name in candidates:
        try:
            return _build(name, model)
        except ProviderError:
            continue
    if _ollama_is_running():
        try:
            return _build("ollama", model)
        except ProviderError:
            pass

    # 4. Fallback.
    return MockProvider()
