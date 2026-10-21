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
