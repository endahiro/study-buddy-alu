"""Tests for MockProvider."""

from studybuddy.llm.base import Message
from studybuddy.llm.mock import CANNED_RESPONSE, MockProvider

MESSAGES = [Message("user", "hello")]


def test_name_is_mock() -> None:
    assert MockProvider().name == "mock"


def test_stream_yields_multiple_chunks_that_rebuild_response() -> None:
    chunks = list(MockProvider(delay=0).stream(MESSAGES))
    assert len(chunks) > 1
    assert "".join(chunks) == CANNED_RESPONSE


def test_complete_returns_full_text() -> None:
    response = MockProvider().complete(MESSAGES)
    assert response.text == CANNED_RESPONSE
    assert response.model == "mock"
    assert response.output_tokens > 0
