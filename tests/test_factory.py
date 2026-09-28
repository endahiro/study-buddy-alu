"""Tests for provider resolution."""

from studybuddy.llm.factory import get_provider
from studybuddy.llm.mock import MockProvider


def test_get_provider_returns_mock_today() -> None:
    assert isinstance(get_provider(), MockProvider)


def test_get_provider_returns_mock_even_with_preference() -> None:
    assert isinstance(get_provider("anthropic"), MockProvider)


def test_resolution_order_is_documented() -> None:
    doc = get_provider.__doc__
    assert doc is not None
    markers = [
        "Explicit preference",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "localhost:11434",
        "MockProvider",
    ]
    positions = [doc.index(marker) for marker in markers]
    assert positions == sorted(positions), "resolution order in docstring is out of order"
