"""Test doubles shared across test modules."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from studybuddy.llm import LLMProvider, LLMResponse, Message


class ScriptedProvider(LLMProvider):
    """Provider that returns pre-written replies in order and records every request."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.requests: list[list[Message]] = []
        self.model = "scripted-model"

    @property
    def name(self) -> str:
        return "scripted"

    def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        text = "".join(self.stream(messages))
        return LLMResponse(text=text, input_tokens=0, output_tokens=0, model="scripted")

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        self.requests.append(messages)
        reply = self.replies.pop(0)
        # Two chunks, to exercise reassembly of streamed text.
        middle = len(reply) // 2
        yield reply[:middle]
        yield reply[middle:]


# Words the fake embedder knows; each becomes one dimension of the vector.
VOCABULARY = ("recursion", "base", "case", "stack", "loop", "list", "rag", "embedding", "vector")


def keyword_embed(texts: list[str]) -> list[list[float]]:
    """Fake embedder: counts vocabulary words, so texts sharing words get similar vectors.

    Every vector gets a small constant so no text embeds to all zeros.
    """
    vectors = []
    for text in texts:
        words = text.lower().replace(".", " ").replace("?", " ").split()
        vectors.append([float(words.count(word)) for word in VOCABULARY] + [0.1])
    return vectors
