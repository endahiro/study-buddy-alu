"""LLM provider backed by the Anthropic Messages API (Claude).

Requires the optional ``anthropic`` extra: ``uv sync --extra anthropic``. Credentials are
resolved by the SDK (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``, or an ``ant auth
login`` profile).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import anthropic
from anthropic.types.beta import BetaMessage, BetaMessageParam

from studybuddy.llm.base import LLMProvider, LLMResponse, Message, ProviderError

DEFAULT_MODEL = "claude-opus-5-5"
# DESIGN: the SDK needs streaming for large max_tokens to avoid HTTP timeouts, so the
# non-streaming ceiling is lower.
STREAM_MAX_TOKENS = 64000
COMPLETE_MAX_TOKENS = 16000
# Opus 5.5 defaults to "medium"; set explicitly so behaviour doesn't shift with defaults.
DEFAULT_EFFORT = "medium"
# Server-side refusal fallback: if Claude's safety classifiers decline a request, the API
# re-runs it on Anthropic's recommended fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def to_anthropic(messages: list[Message]) -> tuple[str, list[BetaMessageParam]]:
    """Split ``messages`` into Anthropic's top-level system prompt and turn list."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    turns: list[BetaMessageParam] = [
        {"role": "assistant" if m.role == "assistant" else "user", "content": m.content}
        for m in messages
        if m.role != "system"
    ]
    return system, turns


def check_stop_reason(message: BetaMessage) -> None:
    """Raise ``ProviderError`` if the model (and every fallback) declined the request."""
    if message.stop_reason == "refusal":
        raise ProviderError("Claude declined to answer this request.")


class AnthropicProvider(LLMProvider):
    """Provider that talks to Claude through the official ``anthropic`` SDK."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        """Create an Anthropic provider.

        Args:
            model: Claude model ID to use.
            client: Pre-built SDK client; one is created from the environment if omitted.
        """
        # DESIGN: client is injectable so tests can pass a fake and never touch the network.
        self.model = model
        self.client = client if client is not None else anthropic.Anthropic()

    @property
    def name(self) -> str:
        """Return the provider identifier, ``"anthropic"``."""
        return "anthropic"

    def _request(self, messages: list[Message], max_tokens: int, **kwargs: Any) -> dict[str, Any]:
        system, turns = to_anthropic(messages)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": turns,
            "output_config": {"effort": DEFAULT_EFFORT},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        if system:
            params["system"] = system
        params.update(kwargs)
        return params

    def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Send ``messages`` to Claude and return the full reply."""
        try:
            response = self.client.beta.messages.create(
                **self._request(messages, COMPLETE_MAX_TOKENS, **kwargs)
            )
        except anthropic.APIError as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc
        check_stop_reason(response)
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream Claude's reply to ``messages`` as text chunks."""
        try:
            with self.client.beta.messages.stream(
                **self._request(messages, STREAM_MAX_TOKENS, **kwargs)
            ) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except anthropic.APIError as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc
        check_stop_reason(final)
