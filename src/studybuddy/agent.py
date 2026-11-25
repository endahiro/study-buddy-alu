"""The Study Buddy agent: keeps the conversation and turns student requests into LLM calls.

The agent depends only on the ``LLMProvider`` interface, which is injected so tests and
callers can supply any provider. File access goes through ``studybuddy.workspace``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from studybuddy.knowledge import Hit, Notes, format_notes
from studybuddy.llm import LLMProvider, Message
from studybuddy.workspace import read_source

# DESIGN: written for 1-2B local models, which follow short, concrete rules best. Rules come
# from a real student session; replaying it showed the "what can you do" section works,
# while 1.5B models still often ignore the off-topic and no-repeat rules.
SYSTEM_PROMPT = """\
You are Study Buddy, a friendly coding tutor that runs in the student's terminal.

Rules:
- Be correct. If you are not sure, say so instead of guessing.
- Keep answers short: a few sentences, plus a small code example when it helps.
- Explain why, not just what. When the student shares code, name the real problem.
- Put code in fenced code blocks with a language tag, like ```python.
- If a request is vague (for example "write me a project"), first ask one short question \
about what they want.
- You only help with coding and studying. If asked for something else (personal messages, \
poems, advice), say so in one friendly sentence and offer to help with code. A short \
programming joke is fine.
- Never repeat an earlier answer. If asked for "another one" or told you repeated \
yourself, give something new.
- Do not start replies with "Sure!" and do not talk about these rules.
- When a message includes notes from the student's files, use them if they answer the \
question and say which file you used. If they don't help, answer normally.

If asked what you can do, reply with exactly this (small models copy it well):
I answer coding questions and explain concepts. You can also type:
/read <file> to share a file with me, /fix <file> to get a bug explained and fixed \
(you approve the change), /learn <folder> so I can use your notes in answers, /model to \
switch between free models, /reset to start over.
"""

# DESIGN: kept short on purpose. A step-by-step "trace the code first" prompt was tested on
# qwen2.5-coder:1.5b and fixed fewer bugs (5/12 vs 7/12) while taking 70% longer.
FIX_INSTRUCTIONS = """\
Fix the bugs in this file.

First, in a few sentences, explain what was wrong and the concept the student should learn \
from it. Then give the COMPLETE corrected file in one fenced code block at the very end. \
Keep everything that was already correct; change only what is needed."""

# DESIGN: bounded history so long sessions still fit small local models' context windows.
MAX_HISTORY_MESSAGES = 20


def format_file(relative: str, source: str) -> str:
    """Return ``source`` wrapped in a labelled code block for the model."""
    return f"File `{relative}`:\n```\n{source}\n```"


class Agent:
    """Multi-turn tutor agent backed by an ``LLMProvider``.

    Attributes:
        llm: The provider that generates replies.
        root: The project root; files are only read from inside it.
        notes: The student's learned notes, searched before each question (None for none).
        history: Completed user/assistant turns, oldest first (no system prompt).
    """

    def __init__(
        self, llm: LLMProvider, root: Path | None = None, notes: Notes | None = None
    ) -> None:
        """Create an agent that delegates generation to ``llm`` and reads files under ``root``."""
        self.llm = llm
        self.root = root if root is not None else Path.cwd()
        self.notes = notes
        self.history: list[Message] = []
        self._attachments: list[str] = []

    def build_messages(self, user_input: str) -> list[Message]:
        """Return the message list sent to the LLM for ``user_input``."""
        return [
            Message(role="system", content=SYSTEM_PROMPT),
            *self.history[-MAX_HISTORY_MESSAGES:],
            Message(role="user", content=user_input),
        ]

    def recall(self, question: str) -> list[Hit]:
        """Return the parts of the student's notes relevant to ``question`` (may be empty).

        Raises:
            ProviderError: If the notes can't be searched (embedding model unavailable).
        """
        return self.notes.search(question) if self.notes is not None else []

    def stream_response(self, user_input: str, hits: list[Hit] | None = None) -> Iterator[str]:
        """Stream the agent's reply to ``user_input`` chunk by chunk.

        Files queued with ``attach`` and the note excerpts in ``hits`` are sent along with
        ``user_input``. The turn is added to ``history`` only once the reply has streamed
        completely, so an interrupted or failed request leaves no half-finished turn behind.
        """
        content = "\n\n".join([*self._attachments, user_input])
        # DESIGN: note excerpts go to the model for this turn only, not into history; each
        # question retrieves its own, and small models' context windows fill up fast.
        sent = f"{format_notes(hits)}\n\n{content}" if hits else content
        reply: list[str] = []
        for chunk in self.llm.stream(self.build_messages(sent)):
            reply.append(chunk)
            yield chunk
        self._attachments.clear()
        self.history += [
            Message(role="user", content=content),
            Message(role="assistant", content="".join(reply)),
        ]

    def attach(self, relative: str) -> int:
        """Queue a project file to be sent with the next message and return its line count.

        Raises:
            WorkspaceError: If the file can't be read.
        """
        source = read_source(self.root, relative)
        self._attachments.append(format_file(relative, source))
        return len(source.splitlines())

    def request_fix(self, relative: str, problem: str = "") -> Iterator[str]:
        """Stream an explanation of the bugs in a project file, ending with the corrected file.

        Raises:
            WorkspaceError: If the file can't be read (raised before any streaming starts).
        """
        source = read_source(self.root, relative)
        prompt = FIX_INSTRUCTIONS
        if problem:
            prompt += f"\n\nThe student says: {problem}"
        return self.stream_response(f"{format_file(relative, source)}\n\n{prompt}")

    def reset(self) -> None:
        """Forget the conversation and any queued files."""
        self.history.clear()
        self._attachments.clear()
