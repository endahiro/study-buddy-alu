"""Tests for the abstract LLMProvider interface and its data types."""

from collections.abc import Iterator
from typing import Any

import pytest

from studybuddy.llm.base import LLMProvider, LLMResponse, Message


class FakeProvider(LLMProvider):
    """Minimal concrete provider implementing every abstract member."""

    @property
    def name(self) -> str:
        return "fake"

    def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        return LLMResponse(text="ok", input_tokens=1, output_tokens=1, model="fake")

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        yield "ok"


class IncompleteProvider(LLMProvider):
    """Provider missing ``stream`` — must not be instantiable."""

    @property
    def name(self) -> str:
        return "incomplete"

    def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        return LLMResponse(text="", input_tokens=0, output_tokens=0, model="x")


def test_abstract_provider_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_partial_subclass_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        IncompleteProvider()  # type: ignore[abstract]


def test_complete_subclass_can_be_instantiated() -> None:
    provider = FakeProvider()
    assert provider.name == "fake"
    assert provider.complete([Message("user", "hi")]).text == "ok"
    assert list(provider.stream([Message("user", "hi")])) == ["ok"]


def test_message_fields() -> None:
    message = Message(role="user", content="hello")
    assert message.role == "user"
    assert message.content == "hello"
