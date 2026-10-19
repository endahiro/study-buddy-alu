"""A mock LLM provider that returns a canned response.

Used for offline development, tests, and as the last-resort fallback when no real
provider is available.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from studybuddy.llm.base import LLMProvider, LLMResponse, Message

CANNED_RESPONSE = (
    "Hi — I'm Study Buddy running on the mock provider. "
    "Set ANTHROPIC_API_KEY or GEMINI_API_KEY, or start Ollama, to use a real model. "
    "What are you working on?"
)


class MockProvider(LLMProvider):
    """Provider that ignores its input and returns ``CANNED_RESPONSE``."""

    def __init__(self, delay: float = 0.03) -> None:
        """Create a mock provider.

        Args:
            delay: Seconds to sleep between streamed words.
        """
        # DESIGN: delay is injectable so tests can set it to 0 and run instantly.
        self.delay = delay

    @property
    def name(self) -> str:
        """Return the provider identifier, ``"mock"``."""
        return "mock"

    def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Return the full canned response with rough word-count token usage."""
        input_tokens = sum(len(m.content.split()) for m in messages)
        return LLMResponse(
            text=CANNED_RESPONSE,
            input_tokens=input_tokens,
            output_tokens=len(CANNED_RESPONSE.split()),
            model="mock",
        )

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Yield the canned response one word at a time, pausing between words."""
        words = CANNED_RESPONSE.split(" ")
        for index, word in enumerate(words):
            if index > 0:
                time.sleep(self.delay)
            # Re-attach the separating space so chunks join back into the exact text.
            yield word if index == 0 else " " + word
