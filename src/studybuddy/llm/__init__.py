"""LLM abstraction layer for Study Buddy.

Public surface: the ``LLMProvider`` interface, its data types, ``get_provider()``, and the
free-model helpers ``list_free_models()``, ``download_model()``, ``model_installed()``
and ``embed_texts()``.
Concrete providers are intentionally *not* re-exported here.
"""

from studybuddy.llm.base import LLMProvider, LLMResponse, Message, ProviderError
from studybuddy.llm.factory import (
    EMBED_MODEL,
    EMBED_MODEL_SIZE,
    FreeModel,
    download_model,
    embed_texts,
    get_provider,
    list_free_models,
    model_installed,
)

__all__ = [
    "EMBED_MODEL",
    "EMBED_MODEL_SIZE",
    "FreeModel",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "ProviderError",
    "download_model",
    "embed_texts",
    "get_provider",
    "list_free_models",
    "model_installed",
]
