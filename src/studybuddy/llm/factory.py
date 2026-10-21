"""Provider resolution: the single place that decides which ``LLMProvider`` to use.

Code outside ``studybuddy.llm`` must obtain providers through ``get_provider()`` and never
import a concrete provider module directly.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass

from studybuddy.llm.base import LLMProvider, ProviderError
from studybuddy.llm.mock import MockProvider

OLLAMA_URL = "http://localhost:11434"
PROVIDER_NAMES = ("anthropic", "gemini", "ollama", "mock")


@dataclass(frozen=True)
class FreeModel:
    """A free local model the student can switch to.

    Attributes:
        name: Ollama model name, e.g. ``"qwen2.5-coder:1.5b"``.
        note: One-line description shown in the model menu.
        installed: Whether it is already downloaded.
        min_ram_gb: RAM the computer needs to run it comfortably (0 if unknown).
    """

    name: str
    note: str
    installed: bool
    min_ram_gb: float = 0


OLLAMA_SDK_HINT = "Free local models need the ollama SDK. Install it with: uv sync --extra ollama"


def list_free_models() -> list[FreeModel]:
    """Return the recommended free models, then any other downloaded ones.

    Raises:
        ProviderError: If the SDK is missing or the Ollama server can't be reached.
    """
    try:
        from studybuddy.llm.ollama import FREE_MODELS, installed_models
    except ImportError as exc:
        raise ProviderError(OLLAMA_SDK_HINT) from exc
    installed = installed_models()
    models = [
        FreeModel(name, note, name in installed, ram) for name, (note, ram) in FREE_MODELS.items()
    ]
    models += [FreeModel(name, "", True) for name in installed if name not in FREE_MODELS]
    return models


def download_model(name: str) -> Iterator[float]:
    """Download a free model, yielding progress from 0 to 1.

    Raises:
        ProviderError: If the SDK is missing or the download fails.
    """
    try:
        from studybuddy.llm.ollama import pull_model
    except ImportError as exc:
        raise ProviderError(OLLAMA_SDK_HINT) from exc
    return pull_model(name)


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
            from studybuddy.llm.ollama import OllamaProvider

            # DESIGN: no default here; the provider picks the best model you have pulled.
            return OllamaProvider(model=model)
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
