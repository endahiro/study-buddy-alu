"""Tests for the multi-turn tutor agent."""

from pathlib import Path

import pytest

from studybuddy.agent import MAX_HISTORY_MESSAGES, SYSTEM_PROMPT, Agent
from studybuddy.llm import Message, ProviderError
from studybuddy.workspace import WorkspaceError
from tests.fakes import ScriptedProvider


def test_reply_is_streamed_and_remembered(tmp_path: Path) -> None:
    llm = ScriptedProvider("A list is mutable.", "Yes, tuples are not.")
    agent = Agent(llm, tmp_path)
    assert "".join(agent.stream_response("What is a list?")) == "A list is mutable."
    "".join(agent.stream_response("And tuples?"))
    assert llm.requests[1] == [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content="What is a list?"),
        Message(role="assistant", content="A list is mutable."),
        Message(role="user", content="And tuples?"),
    ]


def test_failed_request_leaves_no_partial_turn(tmp_path: Path) -> None:
    class Failing(ScriptedProvider):
        def stream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            yield "partial"
            raise ProviderError("boom")

    agent = Agent(Failing(), tmp_path)
    with pytest.raises(ProviderError):
        "".join(agent.stream_response("hi"))
    assert agent.history == []


def test_history_is_bounded(tmp_path: Path) -> None:
    llm = ScriptedProvider(*["ok"] * (MAX_HISTORY_MESSAGES + 5))
    agent = Agent(llm, tmp_path)
    for i in range(MAX_HISTORY_MESSAGES + 5):
        "".join(agent.stream_response(f"q{i}"))
    # system + bounded history + new user message
    assert len(llm.requests[-1]) == MAX_HISTORY_MESSAGES + 2


def test_attached_file_goes_with_the_next_message_only(tmp_path: Path) -> None:
    (tmp_path / "loop.py").write_text("for i in range(3): print(i)\n")
    llm = ScriptedProvider("It prints 0..2.", "Sure.")
    agent = Agent(llm, tmp_path)
    agent.attach("loop.py")
    "".join(agent.stream_response("What does this print?"))
    first = llm.requests[0][-1].content
    assert "File `loop.py`" in first
    assert "for i in range(3)" in first
    assert first.endswith("What does this print?")
    "".join(agent.stream_response("Thanks"))
    assert llm.requests[1][-1].content == "Thanks"


def test_attach_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        Agent(ScriptedProvider(), tmp_path).attach("missing.py")


def test_request_fix_sends_file_and_problem(tmp_path: Path) -> None:
    (tmp_path / "add.py").write_text("def add(a, b):\n    return a - b\n")
    llm = ScriptedProvider("fixed")
    agent = Agent(llm, tmp_path)
    "".join(agent.request_fix("add.py", "add(2, 2) gives 0"))
    prompt = llm.requests[0][-1].content
    assert "return a - b" in prompt
    assert "COMPLETE corrected file" in prompt
    assert "The student says: add(2, 2) gives 0" in prompt


def test_request_fix_fails_before_calling_the_model(tmp_path: Path) -> None:
    llm = ScriptedProvider()
    with pytest.raises(WorkspaceError):
        Agent(llm, tmp_path).request_fix("../outside.py")
    assert llm.requests == []


def test_reset_forgets_everything(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a = 1\n")
    agent = Agent(ScriptedProvider("hi"), tmp_path)
    "".join(agent.stream_response("hello"))
    agent.attach("a.py")
    agent.reset()
    assert agent.history == []
    assert agent.build_messages("x")[1:] == [Message(role="user", content="x")]


def test_system_prompt_describes_the_real_commands() -> None:
    from studybuddy.ui.views import COMMANDS

    help_text = " ".join(command for command, _ in COMMANDS)
    for command in ("/read", "/fix", "/model", "/reset"):
        assert command in SYSTEM_PROMPT, f"prompt should tell the model about {command}"
        assert command in help_text, f"{command} is described to the model but doesn't exist"
