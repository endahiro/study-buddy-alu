"""Render a streamed model reply as live Markdown without flicker.

How it works:
- Until the first chunk arrives, a spinner says what the model is doing and for how long
  (small local models can take 10+ seconds to start).
- Finished Markdown blocks are printed permanently above the live area, so the live area
  only ever holds the block being written and never grows taller than the screen. That is
  what keeps Rich's ``Live`` from flickering or duplicating lines.
- Half-written fences are kept balanced before every render: a trailing line that might
  be the start of a fence (```` `` ````, ```` ```pyt ````) is held back until it is
  complete, and an open fence gets a temporary closing fence so the code renders as code.

When the console is not an interactive terminal (pipes, tests) the text is printed
verbatim with no Markdown and no animation.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.constrain import Constrain
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.padding import Padding
from rich.segment import Segment
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from studybuddy.ui.theme import MAX_WIDTH, SYNTAX_THEME

# An opening/closing fence line: up to 3 spaces, then 3+ backticks or tildes.
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
# A line that starts a list item or an indented continuation; splitting the Markdown
# right before one would restart list numbering.
LIST_OR_INDENT_RE = re.compile(r"^(\s|[-*+]\s|\d+[.)]\s)")

# Left margin that marks the agent's prose lane.
PROSE_INDENT = 2


class _CodeBlock(CodeBlock):
    """Fenced code without a background colour, highlighted with ``SYNTAX_THEME``."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        code = str(self.text).rstrip()
        yield Syntax(
            code,
            self.lexer_name,
            theme=SYNTAX_THEME,
            word_wrap=True,
            background_color="default",
            padding=(0, 0, 0, 2),
        )


class _Markdown(Markdown):
    """Markdown that uses the Study Buddy code-block style."""

    elements = {**Markdown.elements, "fence": _CodeBlock, "code_block": _CodeBlock}


def prose(markdown: str) -> RenderableType:
    """Return ``markdown`` rendered in the agent-prose lane."""
    return Constrain(Padding(_Markdown(markdown), (0, 0, 0, PROSE_INDENT)), MAX_WIDTH)


def open_fence(text: str) -> str | None:
    """Return the fence marker left open at the end of ``text``, or None if all are closed."""
    marker: str | None = None
    for line in text.split("\n"):
        match = FENCE_RE.match(line)
        if not match:
            continue
        fence = match.group(1)
        if marker is None:
            marker = fence
        elif fence[0] == marker[0] and len(fence) >= len(marker) and not line.strip(" `~"):
            marker = None
    return marker


def balance(text: str) -> str:
    """Return the part of a half-streamed ``text`` that is safe to render as Markdown.

    Holds back a trailing partial line that could become a fence, and closes a fence left
    open so the code inside it renders as code rather than as prose.
    """
    head, newline, last = text.rpartition("\n")
    if last.lstrip().startswith(("`", "~")):
        text = head + newline
    marker = open_fence(text)
    if marker is not None:
        text = text.rstrip("\n") + "\n" + marker
    return text


def split_finished(text: str) -> tuple[str, str]:
    """Split ``text`` into (finished blocks, block still being written).

    A block is finished once a blank line outside any fence is followed by the start of a
    new block. The split is skipped before list items and indented lines so lists stay
    in one piece.
    """
    lines = text.split("\n")
    marker: str | None = None
    cut = 0
    # The last line may still be growing, so it can't confirm a new block yet.
    for index, line in enumerate(lines[:-1]):
        match = FENCE_RE.match(line)
        if match:
            fence = match.group(1)
            if marker is None:
                marker = fence
            elif fence[0] == marker[0] and len(fence) >= len(marker):
                marker = None
            continue
        if marker is not None or line.strip() or index == 0:
            continue
        following = lines[index + 1]
        if following.strip() and not LIST_OR_INDENT_RE.match(following):
            cut = index
    if cut == 0:
        return "", text
    return "\n".join(lines[:cut]), "\n".join(lines[cut + 1 :])


class _LiveTail:
    """The live area: the block being written (bottom-cropped to the screen) and a spinner."""

    def __init__(self, activity: str) -> None:
        self.activity = activity
        self.text = ""
        self.started = time.monotonic()
        self.spinner = Spinner("dots", style="sb.accent")
        self.done = False

    def status(self) -> Text:
        seconds = int(time.monotonic() - self.started)
        doing = self.activity if not self.text.strip() else "writing"
        return Text.assemble((doing, "sb.muted"), (f" · {seconds}s", "sb.muted"))

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if self.done:
            return
        parts: list[RenderableType] = []
        if self.text.strip():
            parts.append(prose(balance(self.text)))
        self.spinner.update(text=self.status())
        parts.append(Padding(self.spinner, (0, 0, 0, PROSE_INDENT)))
        lines = console.render_lines(Group(*parts), options, pad=False)
        # DESIGN: keep the live area shorter than the screen; taller frames are what make
        # Live redraw badly. The full block is printed once it is finished.
        room = max(console.height - 1, 3)
        for line in lines[-room:]:
            yield from line
            yield Segment.line()


def render_reply(console: Console, chunks: Iterable[str], activity: str) -> str:
    """Stream ``chunks`` to the terminal as Markdown and return the full reply text.

    Args:
        console: Where to render.
        chunks: The reply, piece by piece, as the model produces it.
        activity: What the model is doing, shown next to the spinner until the first
            chunk arrives, e.g. ``"Asking qwen2.5-coder:1.5b"``.
    """
    if not console.is_terminal or console.is_dumb_terminal:
        return _render_plain(console, chunks)

    tail = _LiveTail(activity)
    parts: list[str] = []
    printed_any = False

    def commit(block: str) -> None:
        nonlocal printed_any
        if not block.strip():
            return
        if printed_any:
            live.console.print()
        live.console.print(prose(block))
        printed_any = True

    live = Live(tail, console=console, refresh_per_second=12, transient=True)
    with live:
        try:
            for chunk in chunks:
                parts.append(chunk)
                finished, tail.text = split_finished(tail.text + chunk)
                commit(finished)
        finally:
            # Also runs on Ctrl+C or an error mid-reply: keep what was written, drop the
            # spinner. The last block is printed in full, not cropped to the screen.
            tail.done = True
            live.refresh()
            commit(tail.text)
    console.print()
    return "".join(parts)


def _render_plain(console: Console, chunks: Iterable[str]) -> str:
    """Print ``chunks`` verbatim as they arrive and return the full text."""
    parts: list[str] = []
    finished = False
    try:
        for chunk in chunks:
            parts.append(chunk)
            console.print(chunk, end="", markup=False, highlight=False, soft_wrap=True)
        finished = True
    finally:
        if finished:
            console.print("\n")  # end the reply's last line, then a blank line
        elif parts:
            console.print()  # cut off by an error or Ctrl+C: end the partial line
    return "".join(parts)
