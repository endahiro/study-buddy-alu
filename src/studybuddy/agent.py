"""The Study Buddy agent: turns user input into LLM requests.

The agent depends only on the ``LLMProvider`` interface, which is injected so tests and
callers can supply any provider.
"""

from __future__ import annotations

from collections.abc import Iterator

from studybuddy.llm import LLMProvider, Message

SYSTEM_PROMPT = "You are Study Buddy, a Socratic coding tutor. (Full teaching prompt comes later.)"


class Agent:
    """Stateless, one-shot tutor agent backed by an ``LLMProvider``."""

    def __init__(self, llm: LLMProvider) -> None:
        """Create an agent that delegates generation to ``llm``."""
        self.llm = llm

    def build_messages(self, user_input: str) -> list[Message]:
        """Return the message list sent to the LLM for ``user_input``."""
        return [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_input),
        ]

    def stream_response(self, user_input: str) -> Iterator[str]:
        """Stream the agent's reply to ``user_input`` chunk by chunk."""
        return self.llm.stream(self.build_messages(user_input))
