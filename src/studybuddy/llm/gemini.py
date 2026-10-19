"""LLM provider backed by Google's Gemini API.

Requires the optional ``gemini`` extra: ``uv sync --extra gemini``. The SDK reads the API
key from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from google import genai
from google.genai import errors, types

from studybuddy.llm.base import LLMProvider, LLMResponse, Message, ProviderError

# DESIGN: overridable via the constructor; check Google's model list if this is retired.
DEFAULT_MODEL = "gemini-2.5-flash"


def to_gemini(messages: list[Message]) -> tuple[str | None, list[types.Content]]:
    """Split ``messages`` into Gemini's system instruction and ``Content`` list.

    Gemini calls the assistant role ``"model"``.
    """
    system = "\n\n".join(m.content for m in messages if m.role == "system") or None
    contents = [
        types.Content(
            role="model" if m.role == "assistant" else "user",
            parts=[types.Part(text=m.content)],
        )
        for m in messages
        if m.role != "system"
    ]
    return system, contents


class GeminiProvider(LLMProvider):
    """Provider that talks to Gemini through the official ``google-genai`` SDK."""

    def __init__(self, model: str = DEFAULT_MODEL, client: genai.Client | None = None) -> None:
        """Create a Gemini provider.

        Args:
            model: Gemini model ID to use.
            client: Pre-built SDK client; one is created from the environment if omitted.
        """
        self.model = model
        if client is None:
            try:
                client = genai.Client()
            except ValueError as exc:  # raised when no API key is configured
                raise ProviderError(f"Gemini is not configured: {exc}") from exc
        self.client = client

    @property
    def name(self) -> str:
        """Return the provider identifier, ``"gemini"``."""
        return "gemini"

    def _request(self, messages: list[Message], **kwargs: Any) -> dict[str, Any]:
        system, contents = to_gemini(messages)
        config = types.GenerateContentConfig(system_instruction=system, **kwargs)
        return {"model": self.model, "contents": contents, "config": config}

    def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Send ``messages`` to Gemini and return the full reply."""
        try:
            response = self.client.models.generate_content(**self._request(messages, **kwargs))
        except errors.APIError as exc:
            raise ProviderError(f"Gemini request failed: {exc}") from exc
        usage = response.usage_metadata
        return LLMResponse(
            text=response.text or "",
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
            model=response.model_version or self.model,
        )

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream Gemini's reply to ``messages`` as text chunks."""
        try:
            for chunk in self.client.models.generate_content_stream(
                **self._request(messages, **kwargs)
            ):
                if chunk.text:
                    yield chunk.text
        except errors.APIError as exc:
            raise ProviderError(f"Gemini request failed: {exc}") from exc
