"""Tests for the real providers, using fake SDK clients so nothing touches the network."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import ollama
import pytest
from google.genai import errors as genai_errors

from studybuddy.llm.anthropic import FALLBACK_BETA, AnthropicProvider
from studybuddy.llm.base import Message, ProviderError
from studybuddy.llm.gemini import GeminiProvider
from studybuddy.llm.ollama import OllamaProvider

CONVERSATION = [
    Message("system", "Be Socratic."),
    Message("user", "Why is my loop infinite?"),
    Message("assistant", "What changes each iteration?"),
    Message("user", "Nothing, I guess."),
]


# --- Anthropic ---------------------------------------------------------------------------


class FakeAnthropicStream:
    def __init__(self, chunks: list[str], stop_reason: str) -> None:
        self.text_stream = iter(chunks)
        self._final = SimpleNamespace(stop_reason=stop_reason)

    def __enter__(self) -> "FakeAnthropicStream":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> SimpleNamespace:
        return self._final


class FakeAnthropicMessages:
    def __init__(self, stop_reason: str = "end_turn", error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stop_reason = stop_reason
        self.error = error

    def create(self, **params: Any) -> SimpleNamespace:
        self.calls.append(params)
        if self.error:
            raise self.error
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="Try printing i."),
            ],
            usage=SimpleNamespace(input_tokens=12, output_tokens=4),
            model="claude-opus-5-5",
            stop_reason=self.stop_reason,
        )

    def stream(self, **params: Any) -> FakeAnthropicStream:
        self.calls.append(params)
        if self.error:
            raise self.error
        return FakeAnthropicStream(["Try ", "printing i."], self.stop_reason)


def anthropic_provider(**kwargs: Any) -> tuple[AnthropicProvider, FakeAnthropicMessages]:
    messages = FakeAnthropicMessages(**kwargs)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return AnthropicProvider(client=client), messages  # type: ignore[arg-type]


def test_anthropic_complete_maps_request_and_response() -> None:
    provider, fake = anthropic_provider()
    response = provider.complete(CONVERSATION)

    assert response.text == "Try printing i."
    assert (response.input_tokens, response.output_tokens) == (12, 4)
    params = fake.calls[0]
    assert params["model"] == "claude-opus-5-5"
    assert params["system"] == "Be Socratic."
    assert [m["role"] for m in params["messages"]] == ["user", "assistant", "user"]
    assert params["fallbacks"] == "default"
    assert params["betas"] == [FALLBACK_BETA]


def test_anthropic_stream_yields_chunks() -> None:
    provider, _ = anthropic_provider()
    assert "".join(provider.stream(CONVERSATION)) == "Try printing i."


def test_anthropic_refusal_raises_provider_error() -> None:
    provider, _ = anthropic_provider(stop_reason="refusal")
    with pytest.raises(ProviderError, match="declined"):
        provider.complete(CONVERSATION)
    with pytest.raises(ProviderError, match="declined"):
        list(provider.stream(CONVERSATION))


def test_anthropic_sdk_errors_are_wrapped() -> None:
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.com"))
    provider, _ = anthropic_provider(error=error)
    with pytest.raises(ProviderError, match="Anthropic request failed"):
        provider.complete(CONVERSATION)
    with pytest.raises(ProviderError, match="Anthropic request failed"):
        list(provider.stream(CONVERSATION))


# --- Gemini ------------------------------------------------------------------------------


class FakeGeminiModels:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def generate_content(self, **params: Any) -> SimpleNamespace:
        self.calls.append(params)
        if self.error:
            raise self.error
        return SimpleNamespace(
            text="Try printing i.",
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=3),
            model_version="gemini-test",
        )

    def generate_content_stream(self, **params: Any) -> Iterator[SimpleNamespace]:
        self.calls.append(params)
        if self.error:
            raise self.error
        for text in ["Try ", None, "printing i."]:
            yield SimpleNamespace(text=text)


def gemini_provider(**kwargs: Any) -> tuple[GeminiProvider, FakeGeminiModels]:
    models = FakeGeminiModels(**kwargs)
    return GeminiProvider(client=SimpleNamespace(models=models)), models  # type: ignore[arg-type]


def test_gemini_complete_maps_request_and_response() -> None:
    provider, fake = gemini_provider()
    response = provider.complete(CONVERSATION)

    assert response.text == "Try printing i."
    assert (response.input_tokens, response.output_tokens) == (10, 3)
    assert response.model == "gemini-test"
    params = fake.calls[0]
    assert params["config"].system_instruction == "Be Socratic."
    assert [c.role for c in params["contents"]] == ["user", "model", "user"]


def test_gemini_stream_skips_empty_chunks() -> None:
    provider, _ = gemini_provider()
    assert list(provider.stream(CONVERSATION)) == ["Try ", "printing i."]


def test_gemini_sdk_errors_are_wrapped() -> None:
    provider, _ = gemini_provider(error=genai_errors.APIError(500, {"error": {}}))
    with pytest.raises(ProviderError, match="Gemini request failed"):
        list(provider.stream(CONVERSATION))


# --- Ollama ------------------------------------------------------------------------------


class FakeOllamaClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def chat(self, **params: Any) -> Any:
        self.calls.append(params)
        if self.error:
            raise self.error
        if params["stream"]:
            return iter(
                SimpleNamespace(message=SimpleNamespace(content=text))
                for text in ["Try ", "", "printing i."]
            )
        return SimpleNamespace(
            message=SimpleNamespace(content="Try printing i."),
            prompt_eval_count=8,
            eval_count=5,
            model="llama3.2",
        )


def test_ollama_complete_maps_request_and_response() -> None:
    fake = FakeOllamaClient()
    response = OllamaProvider(client=fake).complete(CONVERSATION)  # type: ignore[arg-type]

    assert response.text == "Try printing i."
    assert (response.input_tokens, response.output_tokens) == (8, 5)
    sent = fake.calls[0]["messages"]
    assert [m.role for m in sent] == ["system", "user", "assistant", "user"]


def test_ollama_stream_skips_empty_chunks() -> None:
    provider = OllamaProvider(client=FakeOllamaClient())  # type: ignore[arg-type]
    assert list(provider.stream(CONVERSATION)) == ["Try ", "printing i."]


@pytest.mark.parametrize(
    "error", [ConnectionError("refused"), ollama.ResponseError("model not found", 404)]
)
def test_ollama_errors_are_wrapped(error: Exception) -> None:
    provider = OllamaProvider(client=FakeOllamaClient(error=error))  # type: ignore[arg-type]
    with pytest.raises(ProviderError, match="Ollama request failed"):
        provider.complete(CONVERSATION)
