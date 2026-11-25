"""LLM provider backed by a local Ollama server.

Requires the optional ``ollama`` extra: ``uv sync --extra ollama``, plus a running server
(``ollama serve``) with the chosen model pulled (``ollama pull <model>``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import ollama

from studybuddy.llm.base import LLMProvider, LLMResponse, Message, ProviderError

DEFAULT_HOST = "http://localhost:11434"
# DESIGN: httpx.HTTPError covers the server dying mid-reply (e.g. killed for using too much
# memory), which the SDK doesn't wrap; without it the whole chat crashed.
OLLAMA_ERRORS = (ollama.ResponseError, ConnectionError, httpx.HTTPError)
# Free local models offered in the chat's /model menu, best first:
# name -> (one-line note, RAM in GB the computer needs to run it comfortably).
# DESIGN: picked by benchmarking coding Q&A and bug fixes on a CPU-only laptop. The small
# models run on a machine reporting 3.7 GB; the 3b model was killed there for running out
# of memory. (A "4 GB" laptop reports less than 4, so the small models are listed at 3.)
FREE_MODELS: dict[str, tuple[str, float]] = {
    "qwen2.5-coder:1.5b": ("best all-round, accurate answers (1 GB download)", 3),
    "llama3.2:1b": ("fast, but makes factual mistakes (1.3 GB)", 3),
    "qwen3:1.7b": ("asks you follow-up questions, slower (1.4 GB)", 3),
    "qwen2.5-coder:3b": ("a bit more careful, ~3x slower (1.9 GB)", 4.5),
}
# When no model is requested, the first of these already pulled on the server is used.
RECOMMENDED_MODELS = tuple(FREE_MODELS)
DEFAULT_MODEL = RECOMMENDED_MODELS[0]
# Caps a reply so a small model stuck repeating itself stops instead of running for minutes.
MAX_REPLY_TOKENS = 1024
# DESIGN: measured on qwen2.5-coder:1.5b (hand-graded coding questions): 0.5 was the most
# accurate (10/10), 0.2 repeated the same factual slip (8/10), 0.7 made more mistakes
# (12/15). No setting stops a 1.5B model reusing a joke it already told without costing
# accuracy, so accuracy wins.
DEFAULT_OPTIONS: dict[str, Any] = {"temperature": 0.5, "num_predict": MAX_REPLY_TOKENS}


def installed_models(client: ollama.Client | None = None) -> list[str]:
    """Return the names of the models pulled on the server (default: the local one).

    Raises:
        ProviderError: If the server can't be reached.
    """
    client = client if client is not None else ollama.Client(host=DEFAULT_HOST)
    try:
        return [m.model for m in client.list().models if m.model]
    except OLLAMA_ERRORS as exc:
        raise ProviderError(
            f"Can't reach Ollama ({exc}). Is it running? Start it with: ollama serve"
        ) from exc


def pull_model(name: str, client: ollama.Client | None = None) -> Iterator[float]:
    """Download ``name`` onto the server, yielding progress as a fraction from 0 to 1.

    Raises:
        ProviderError: If the download fails or the model doesn't exist.
    """
    client = client if client is not None else ollama.Client(host=DEFAULT_HOST)
    try:
        for update in client.pull(name, stream=True):
            if update.total:
                yield (update.completed or 0) / update.total
    except OLLAMA_ERRORS as exc:
        raise ProviderError(f"Couldn't download {name}: {exc}") from exc


def embed(texts: list[str], model: str, client: ollama.Client | None = None) -> list[list[float]]:
    """Return one embedding vector per text, computed by ``model`` on the server.

    Raises:
        ProviderError: If the server can't be reached or ``model`` isn't downloaded.
    """
    client = client if client is not None else ollama.Client(host=DEFAULT_HOST)
    try:
        response = client.embed(model=model, input=texts)
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise ProviderError(
                f"The {model} model isn't downloaded. Run /learn again to get it."
            ) from exc
        raise ProviderError(f"Ollama couldn't embed text: {exc}") from exc
    except OLLAMA_ERRORS as exc:
        raise ProviderError(
            f"Can't reach Ollama ({exc}). Is it running? Start it with: ollama serve"
        ) from exc
    return [list(vector) for vector in response.embeddings]


def request_error(exc: Exception) -> ProviderError:
    """Wrap a failed chat request, explaining the likely cause when the server died."""
    message = f"Ollama request failed: {exc}"
    if isinstance(exc, httpx.HTTPError):
        message += (
            "\nThe model may have run out of memory. Try a smaller one with /model, then ask again."
        )
    return ProviderError(message)


def to_ollama(messages: list[Message]) -> list[ollama.Message]:
    """Convert ``messages`` to Ollama's format (it accepts system/user/assistant as-is)."""
    return [ollama.Message(role=m.role, content=m.content) for m in messages]


class OllamaProvider(LLMProvider):
    """Provider that talks to a local Ollama server through the official ``ollama`` SDK."""

    def __init__(
        self,
        model: str | None = None,
        host: str = DEFAULT_HOST,
        client: ollama.Client | None = None,
    ) -> None:
        """Create an Ollama provider.

        Args:
            model: Name of a model already pulled on the server. If omitted, the first
                pulled model in ``RECOMMENDED_MODELS`` is chosen on first use.
            host: Base URL of the Ollama server.
            client: Pre-built SDK client; one is created for ``host`` if omitted.
        """
        self._model = model
        self.client = client if client is not None else ollama.Client(host=host)

    @property
    def model(self) -> str:
        """Return the model in use, picking a pulled recommended model if none was given.

        Raises:
            ProviderError: If no model was given and no recommended model is pulled.
        """
        if self._model is None:
            self._model = self._pick_model()
        return self._model

    @model.setter
    def model(self, name: str) -> None:
        self._model = name

    def _pick_model(self) -> str:
        """Return the best recommended model that is already pulled on the server."""
        pulled = set(installed_models(self.client))
        for name in RECOMMENDED_MODELS:
            if name in pulled:
                return name
        raise ProviderError(
            "No free model is downloaded yet. Type /model in the chat to pick one, "
            f"or run: ollama pull {DEFAULT_MODEL}"
        )

    def _options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge caller ``options`` over ``DEFAULT_OPTIONS``."""
        return {**DEFAULT_OPTIONS, **kwargs.pop("options", {})}

    @property
    def name(self) -> str:
        """Return the provider identifier, ``"ollama"``."""
        return "ollama"

    def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Send ``messages`` to Ollama and return the full reply."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=to_ollama(messages),
                stream=False,
                options=self._options(kwargs),
                **kwargs,
            )
        except OLLAMA_ERRORS as exc:
            raise request_error(exc) from exc
        return LLMResponse(
            text=response.message.content or "",
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            model=response.model or self.model,
        )

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream Ollama's reply to ``messages`` as text chunks."""
        try:
            for chunk in self.client.chat(
                model=self.model,
                messages=to_ollama(messages),
                stream=True,
                options=self._options(kwargs),
                **kwargs,
            ):
                if chunk.message.content:
                    yield chunk.message.content
        except OLLAMA_ERRORS as exc:
            raise request_error(exc) from exc
