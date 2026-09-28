"""Provider resolution: the single place that decides which ``LLMProvider`` to use.

Code outside ``studybuddy.llm`` must obtain providers through ``get_provider()`` and never
import a concrete provider module directly.
"""

from __future__ import annotations

from studybuddy.llm.base import LLMProvider
from studybuddy.llm.mock import MockProvider


def get_provider(preferred: str | None = None) -> LLMProvider:
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

    Returns:
        A ready-to-use ``LLMProvider``. Currently always a ``MockProvider``.
    """
    # 1. Explicit preference.
    # TODO(anthropic): if preferred == "anthropic":
    #     from studybuddy.llm.anthropic import AnthropicProvider
    #     return AnthropicProvider()
    # TODO(gemini): if preferred == "gemini":
    #     from studybuddy.llm.gemini import GeminiProvider
    #     return GeminiProvider()
    # TODO(ollama): if preferred == "ollama":
    #     from studybuddy.llm.ollama import OllamaProvider
    #     return OllamaProvider()

    # 2. Environment API keys.
    # TODO(anthropic): if os.environ.get("ANTHROPIC_API_KEY"): ... AnthropicProvider()
    # TODO(gemini): if os.environ.get("GEMINI_API_KEY"): ... GeminiProvider()

    # 3. Local Ollama.
    # TODO(ollama): if _ollama_is_running("http://localhost:11434"): ... OllamaProvider()

    # 4. Fallback.
    # DESIGN: `preferred` is accepted but unused until real providers exist; an unknown
    # name will eventually raise ValueError rather than silently falling back.
    return MockProvider()
