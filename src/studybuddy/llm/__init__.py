"""LLM abstraction layer for Study Buddy.

Public surface: the ``LLMProvider`` interface, its data types, and ``get_provider()``.
Concrete providers are intentionally *not* re-exported here.
"""

from studybuddy.llm.base import LLMProvider, LLMResponse, Message, ProviderError
from studybuddy.llm.factory import get_provider

__all__ = ["LLMProvider", "LLMResponse", "Message", "ProviderError", "get_provider"]
