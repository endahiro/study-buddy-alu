"""Core LLM abstractions: the ``LLMProvider`` strategy interface and its data types.

Every part of Study Buddy that needs a language model talks to ``LLMProvider``. Concrete
providers (mock, Anthropic, Gemini, Ollama, ...) live alongside this module and are only
ever constructed by ``studybuddy.llm.factory``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


class ProviderError(Exception):
    """Raised when a provider cannot be constructed or a request to it fails.

    Providers wrap their SDK's own exceptions in this so callers outside
    ``studybuddy.llm`` never need to know which SDK is in use.
    """


@dataclass(frozen=True)
class Message:
    """A single chat message.

    Attributes:
        role: Who authored the message, e.g. ``"system"``, ``"user"``, or ``"assistant"``.
        content: The message text.
    """

    # DESIGN: role is a plain str (not an Enum/Literal) to match the spec and keep
    # provider adapters free to map roles however their SDK expects.
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """The result of a non-streaming completion.

    Attributes:
        text: The full generated text.
        input_tokens: Tokens consumed by the prompt.
        output_tokens: Tokens produced in the response.
        model: Identifier of the model that produced the response.
    """

    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMProvider(ABC):
    """Abstract strategy interface that every LLM backend implements."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, stable identifier for this provider (e.g. ``"mock"``)."""

    @abstractmethod
    def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Generate a full response for ``messages`` in one call.

        Args:
            messages: The conversation to respond to, oldest first.
            **kwargs: Provider-specific options (temperature, max tokens, ...).

        Returns:
            The complete response with token usage.
        """

    @abstractmethod
    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Generate a response for ``messages``, yielding text chunks as they arrive.

        Args:
            messages: The conversation to respond to, oldest first.
            **kwargs: Provider-specific options (temperature, max tokens, ...).

        Yields:
            Successive chunks of response text; concatenated they form the full reply.
        """
