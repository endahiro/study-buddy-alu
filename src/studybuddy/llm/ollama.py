"""LLM provider backed by a local Ollama server.

Requires the optional ``ollama`` extra: ``uv sync --extra ollama``, plus a running server
(``ollama serve``) with the chosen model pulled (``ollama pull <model>``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import ollama

from studybuddy.llm.base import LLMProvider, LLMResponse, Message, ProviderError

DEFAULT_HOST = "http://localhost:11434"
# DESIGN: overridable via the constructor; must already be pulled on the server.
DEFAULT_MODEL = "llama3.2"


def to_ollama(messages: list[Message]) -> list[ollama.Message]:
    """Convert ``messages`` to Ollama's format (it accepts system/user/assistant as-is)."""
    return [ollama.Message(role=m.role, content=m.content) for m in messages]


class OllamaProvider(LLMProvider):
    """Provider that talks to a local Ollama server through the official ``ollama`` SDK."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        client: ollama.Client | None = None,
    ) -> None:
        """Create an Ollama provider.

        Args:
            model: Name of a model already pulled on the server.
            host: Base URL of the Ollama server.
            client: Pre-built SDK client; one is created for ``host`` if omitted.
        """
        self.model = model
        self.client = client if client is not None else ollama.Client(host=host)

    @property
    def name(self) -> str:
        """Return the provider identifier, ``"ollama"``."""
        return "ollama"

    def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Send ``messages`` to Ollama and return the full reply."""
        try:
            response = self.client.chat(
                model=self.model, messages=to_ollama(messages), stream=False, **kwargs
            )
        except (ollama.ResponseError, ConnectionError) as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc
        return LLMResponse(
            text=response.message.content or "",
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            model=response.model or self.model,
        )

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream Ollama's reply to ``messages`` as text chunks."""
        try:
            for chunk in self.client.chat(
                model=self.model, messages=to_ollama(messages), stream=True, **kwargs
            ):
                if chunk.message.content:
                    yield chunk.message.content
        except (ollama.ResponseError, ConnectionError) as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc
