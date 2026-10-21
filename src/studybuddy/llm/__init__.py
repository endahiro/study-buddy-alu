"""LLM abstraction layer for Study Buddy.

Public surface: the ``LLMProvider`` interface, its data types, ``get_provider()``, and the
free-model helpers ``list_free_models()`` and ``download_model()``.
Concrete providers are intentionally *not* re-exported here.
"""

from studybuddy.llm.base import LLMProvider, LLMResponse, Message, ProviderError
from studybuddy.llm.factory import FreeModel, download_model, get_provider, list_free_models

__all__ = [
    "FreeModel",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "ProviderError",
    "download_model",
    "get_provider",
    "list_free_models",
]
