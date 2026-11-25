"""Palette, code-highlighting theme, and console factory for every Study Buddy screen.

DESIGN: only the 8 standard ANSI colours plus bold/dim/italic. The terminal's own theme
decides the exact shades, so text stays readable on light and dark backgrounds, and
nothing changes on a 16-colour terminal. No background colours anywhere: a dark block
behind code looks heavy on a light terminal.
"""

from __future__ import annotations

from pygments.token import (  # type: ignore[import-untyped]
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    String,
    Token,
)
from rich.console import Console
from rich.style import Style
from rich.syntax import ANSISyntaxTheme
from rich.theme import Theme

# Widest that prose and panels get on a big terminal; long lines are hard to read.
# Applied per render (not to the console) so shrinking the window still reflows.
MAX_WIDTH = 100

THEME = Theme(
    {
        # Lanes.
        "sb.accent": "cyan",
        "sb.brand": "bold cyan",
        "sb.muted": "dim",
        "sb.file": "cyan",
        "sb.file.bar": "cyan dim",
        "sb.diff.border": "dim",
        "sb.error": "bold red",
        "sb.warn": "yellow",
        "sb.ok": "green",
        "sb.prompt": "bold cyan",
        # Markdown, restyled without background colours.
        "markdown.code": "cyan",
        "markdown.code_block": "none",
        "markdown.block_quote": "italic dim",
        "markdown.list": "none",
        "markdown.item.bullet": "dim",
        "markdown.item.number": "dim",
        "markdown.h1": "bold underline",
        "markdown.h2": "bold",
        "markdown.h3": "bold",
        "markdown.h4": "bold dim",
        "markdown.link": "blue",
        "markdown.link_url": "blue underline",
        "markdown.table.border": "dim",
        "markdown.table.header": "bold",
        "markdown.kbd": "bold",
        # Rich widgets.
        "progress.percentage": "dim",
        "bar.complete": "cyan",
        "bar.finished": "green",
        "bar.pulse": "cyan",
        "prompt.choices": "dim",
        "prompt.default": "dim",
    }
)

# Pygments token -> style, using only standard ANSI colours (see module DESIGN note).
SYNTAX_THEME = ANSISyntaxTheme(
    {
        Token: Style(),
        Comment: Style(dim=True, italic=True),
        Keyword: Style(color="magenta"),
        Keyword.Constant: Style(color="cyan"),
        Name.Builtin: Style(color="cyan"),
        Name.Function: Style(color="blue"),
        Name.Class: Style(color="blue", bold=True),
        Name.Decorator: Style(color="cyan"),
        String: Style(color="green"),
        Number: Style(color="cyan"),
        Operator.Word: Style(color="magenta"),
        Error: Style(color="red"),
        Generic.Inserted: Style(color="green"),
        Generic.Deleted: Style(color="red"),
        Generic.Subheading: Style(color="cyan", dim=True),
        Generic.Heading: Style(bold=True),
        Generic.Traceback: Style(color="red"),
    }
)


def make_console(**kwargs: object) -> Console:
    """Return a ``Console`` that uses the Study Buddy palette.

    Keyword arguments are passed to ``Console`` (e.g. ``width``, ``color_system``).
    Rich's auto-highlighter is off: it colours anything that looks like a path or number
    (``/help`` came out pink), which adds colour that means nothing.
    """
    kwargs.setdefault("highlight", False)
    return Console(theme=THEME, **kwargs)  # type: ignore[arg-type]
