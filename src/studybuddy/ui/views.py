"""Every Study Buddy screen except the streamed reply (see ``studybuddy.ui.stream``).

Pure presentation: each function takes plain data and renders it. Nothing here talks to
the agent or an LLM provider.

Lanes, so each kind of output is recognisable at a glance:
- agent prose: indented Markdown (``stream``)
- shared file: a cyan bar in the left margin
- proposed diff: a bordered panel
- errors: a bold red ``Error:`` label; warnings a yellow ``!``; notes dim; success a green ``✓``
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.constrain import Constrain
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from studybuddy.ui.theme import MAX_WIDTH, SYNTAX_THEME

PROMPT = "[sb.prompt]studybuddy>[/sb.prompt] "

# (command, what it does) rows for /help.
COMMANDS = (
    ("/read <file>", "share a file; your next message is about it"),
    ("/fix <file> [what's wrong]", "find and fix bugs, then apply after you approve"),
    ("/learn <file|folder>", "learn your notes; answers will use them"),
    ("/learn", "list what Study Buddy has learned"),
    ("/forget", "forget all learned notes"),
    ("/model", "list free models and pick one by number"),
    ("/reset", "start a fresh conversation"),
    ("/help", "show this help"),
    ("exit", "leave (or Ctrl+C)"),
)


# --- Messages -------------------------------------------------------------------------


def note(console: Console, message: str) -> None:
    """Print a quiet status line, e.g. "Kept the current model."."""
    console.print(Text(message, style="sb.muted"))
    console.print()


def success(console: Console, message: str) -> None:
    """Print a line confirming that something was done."""
    console.print(Text.assemble(("✓ ", "sb.ok"), message))
    console.print()


def warn(console: Console, message: str) -> None:
    """Print a problem the student can fix by typing something else."""
    # DESIGN: a symbol as well as colour, so it still reads as a warning without colour.
    console.print(Text.assemble(("! ", "sb.warn"), (message, "sb.warn")))
    console.print()


def error(console: Console, message: str) -> None:
    """Print an error that stopped the current action."""
    console.print(Text.assemble(("Error: ", "sb.error"), message))
    console.print()


def interrupted(console: Console) -> None:
    """Print the line shown when Ctrl+C cancels a reply."""
    console.print(Text("(interrupted)", style="sb.muted"))
    console.print()


def goodbye(console: Console) -> None:
    """Print the last line before the chat exits."""
    console.print(Text("Bye!", style="sb.muted"))


# --- Startup and init -----------------------------------------------------------------


def banner(console: Console, model: str | None, provider: str, problem: str | None) -> None:
    """Print the startup lines.

    Args:
        model: The active model, or None if it couldn't be determined.
        provider: The provider name, e.g. ``"ollama"``.
        problem: Why the model couldn't be determined, if it couldn't.
    """
    title = Text("Study Buddy", "bold")
    if model is not None:
        title.append(f"  ·  Model: {model} ({provider})", style="sb.muted")
    console.print(title)
    if problem is not None:
        console.print(Text.assemble(("Error: ", "sb.error"), problem))
    console.print(
        Text("Ask a coding question, or /help · /model to switch · exit to leave", "sb.muted")
    )
    console.print()


# Block-letter wordmark for the welcome screen: 3 rows of half-block characters.
WORDMARK = (
    "█▀▀ ▀█▀ █ █ █▀▄ █ █   █▀▄ █ █ █▀▄ █▀▄ █ █",
    "▀▀█  █  █ █ █ █ ▀▄▀   █▀▄ █ █ █ █ █ █ ▀▄▀",
    "▀▀▀  ▀  ▀▀▀ ▀▀   ▀    ▀▀  ▀▀▀ ▀▀  ▀▀   ▀ ",
)
TAGLINE = "Your coding tutor. Ask, read, fix, learn."

# Example first messages, matching what students actually use it for.
EXAMPLES = (
    "explain recursion with a small example",
    "show me a practical use of a Python decorator",
    "what's the difference between fine-tuning and RAG?",
)

# Below this width the wordmark and examples don't fit; show the plain banner.
WELCOME_MIN_WIDTH = 52


def welcome(
    console: Console,
    model: str | None,
    provider: str,
    problem: str | None,
    folder: Path,
    notes: int | None = None,
) -> None:
    """Print the front screen: wordmark card, example questions, and key commands.

    Falls back to the plain ``banner`` when output isn't an interactive terminal (pipes,
    tests), the terminal can't show the block characters, or it is too narrow.

    Args:
        model: The active model, or None if it couldn't be determined.
        provider: The provider name, e.g. ``"ollama"``.
        problem: Why the model couldn't be determined, if it couldn't.
        folder: The folder whose files ``/read`` and ``/fix`` may open.
        notes: How many files ``/learn`` has learned; None hides the row.
    """
    utf8 = console.encoding.lower().startswith("utf")
    if not console.is_terminal or not utf8 or console.width < WELCOME_MIN_WIDTH:
        banner(console, model, provider, problem)
        return

    card = Table.grid()
    for row in WORDMARK:
        card.add_row(Text(row, "sb.brand"))
    card.add_row("")
    card.add_row(Text(TAGLINE, "sb.muted"))
    card.add_row("")
    if model is not None:
        status = Text.assemble(("● ", "sb.ok"), (model, "bold"), (f"  {provider}", "sb.muted"))
    else:
        status = Text.assemble(("✗ ", "sb.error"), (problem or "No model", "sb.error"))
    card.add_row(status)
    card.add_row(Text.assemble(("Files  ", "sb.muted"), (_short_path(folder), "sb.file")))
    if notes:
        card.add_row(Text.assemble(("Notes  ", "sb.muted"), (_count(notes, "file"), "sb.file")))
    elif notes is not None:
        card.add_row(
            Text.assemble(
                ("Notes  ", "sb.muted"), ("none yet · ", "sb.muted"), ("/learn", "sb.accent")
            )
        )
    console.print()
    console.print(
        Constrain(
            Panel(card, box=box.ROUNDED, border_style="sb.muted", padding=(1, 3), expand=False),
            MAX_WIDTH,
        )
    )

    examples = Table.grid(padding=(0, 1))
    for example in EXAMPLES:
        examples.add_row(Text("›", "sb.accent"), Text(example))
    console.print(Padding(Text("Try asking", "bold"), (1, 0, 0, 2)))
    console.print(Padding(examples, (0, 0, 0, 4)))
    console.print()
    hints = Text("  ")
    for key, what in (
        ("/read", "share a file"),
        ("/fix", "fix bugs"),
        ("/model", "switch"),
        ("/help", "all commands"),
        ("exit", "leave"),
    ):
        if len(hints) > 2:
            hints.append("  ·  ", "sb.muted")
        hints.append(key, "sb.accent")
        hints.append(f" {what}", "sb.muted")
    console.print(hints)
    console.print()


def _short_path(path: Path, limit: int = len(WORDMARK[0]) - 7) -> str:
    """Return ``path`` with home as ``~``, keeping its end if it's longer than ``limit``.

    The end of a path (the project folder) is the part students recognise, so long paths
    lose their start: ``…/courses/cs101``. ``limit`` keeps the card as wide as the wordmark.
    """
    try:
        text = "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        text = str(path)
    if len(text) <= limit:
        return text
    tail = ""
    for part in reversed(path.parts):
        candidate = f"{part}/{tail}" if tail else part
        if len(candidate) + 2 > limit:
            break
        tail = candidate
    # A single folder name longer than the limit is cut at its start.
    return "…/" + tail if tail else "…" + path.name[-(limit - 1) :]


def init_done(console: Console, project_dir: Path) -> None:
    """Print the result of ``studybuddy init``."""
    console.print(
        Text.assemble(("✓ ", "sb.ok"), f"Initialized Study Buddy project at {project_dir}")
    )
    console.print(Text("Run studybuddy to start a session.", "sb.muted"))


def init_exists(console: Console, folder: Path, project_dir_name: str) -> None:
    """Print the error for ``studybuddy init`` in a folder that already is a project."""
    console.print(
        Text.assemble(
            ("Error: ", "sb.error"),
            f"{folder} is already a Study Buddy project ({project_dir_name}/ exists).",
        )
    )


def help_text(console: Console) -> None:
    """Print the command list."""
    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    table.add_column(style="sb.accent", no_wrap=True)
    table.add_column(style="sb.muted")
    for command, what in COMMANDS:
        table.add_row(Text(command), Text(what))
    console.print(Text("Ask any coding question, or use a command:", "bold"))
    console.print(Padding(table, (0, 0, 0, 2), expand=False))
    console.print()


# --- Models ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRow:
    """One line of the model list.

    Attributes:
        name: Model name.
        note: One-line description (may be empty).
        status: One of ``"in_use"``, ``"ready"``, ``"too_big"``, ``"missing"``.
        min_ram_gb: RAM the model needs; shown when ``status`` is ``"too_big"``.
    """

    name: str
    note: str
    status: str
    min_ram_gb: float = 0


def model_table(console: Console, rows: list[ModelRow]) -> None:
    """Print the numbered list of free models."""
    labels = {
        "in_use": Text("● in use", "sb.ok"),
        "ready": Text("ready", "sb.ok"),
        "missing": Text("not downloaded", "sb.muted"),
    }
    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    table.add_column(justify="right", style="sb.accent", no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(style="sb.muted", overflow="fold")
    for number, row in enumerate(rows, start=1):
        status = labels.get(row.status) or Text(f"needs {row.min_ram_gb:g} GB RAM", "sb.warn")
        name = Text(row.name, style="bold" if row.status == "in_use" else "")
        table.add_row(str(number), name, status, Text(row.note))
    console.print(
        Text.assemble(
            ("Free models", "bold"), (" · run on your computer, no key needed", "sb.muted")
        )
    )
    console.print(Padding(table, (0, 0, 0, 2), expand=False))


def ask_model_number(console: Console) -> str:
    """Ask which model to switch to; Ctrl+C or Ctrl+D count as an empty answer."""
    try:
        return console.input("[sb.muted]Pick a number (Enter keeps the current one):[/] ")
    except (KeyboardInterrupt, EOFError):
        console.print()
        return ""


def download_progress(console: Console, name: str, fractions: Iterable[float]) -> None:
    """Show a progress bar for downloading ``name`` while ``fractions`` (0 to 1) arrive."""
    progress_bar(console, "Downloading", name, fractions)


def progress_bar(console: Console, verb: str, what: str, fractions: Iterable[float]) -> None:
    """Show ``verb what`` with a progress bar while ``fractions`` (0 to 1) arrive."""
    columns = (
        TextColumn(f"[sb.muted]{escape(verb)}[/] {{task.description}}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
    )
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task(escape(what), total=1.0)
        for fraction in fractions:
            progress.update(task, completed=fraction)


def switched_model(console: Console, name: str) -> None:
    """Confirm a model switch."""
    console.print(
        Text.assemble(("✓ ", "sb.ok"), f"Now using {name}", ("  (conversation kept)", "sb.muted"))
    )
    console.print()


# --- Files and fixes ------------------------------------------------------------------


def shared_file(console: Console, name: str, line_count: int | None = None) -> None:
    """Echo a file shared with ``/read``."""
    size = f" · {line_count} lines" if line_count is not None else ""
    console.print(
        Text.assemble(("▌ ", "sb.file.bar"), (f"Shared {name}", "sb.file"), (size, "sb.muted"))
    )
    console.print(
        Text.assemble(("▌ ", "sb.file.bar"), ("Now ask your question about it.", "sb.muted"))
    )
    console.print()


def diff_panel(console: Console, name: str, diff: str) -> None:
    """Show a proposed change as a highlighted unified diff in a bordered panel."""
    lines = diff.splitlines(keepends=True)
    # The ---/+++ header repeats the file name already in the title.
    body = "".join(line for line in lines if not line.startswith(("--- ", "+++ ")))
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    syntax = Syntax(
        body.rstrip("\n"),
        "diff",
        theme=SYNTAX_THEME,
        word_wrap=True,
        background_color="default",
    )
    panel = Panel(
        syntax,
        title=Text(f"Proposed change · {name}", "bold"),
        title_align="left",
        subtitle=Text.assemble((f"+{added}", "green"), " ", (f"-{removed}", "red")),
        subtitle_align="right",
        border_style="sb.diff.border",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(Constrain(panel, MAX_WIDTH))


def confirm(console: Console, question: str) -> bool:
    """Ask a yes/no question (default no); Ctrl+C or Ctrl+D count as no."""
    try:
        return Confirm.ask(question, console=console, default=False)
    except (KeyboardInterrupt, EOFError):
        console.print()
        return False


# --- Notes (RAG) ----------------------------------------------------------------------

# Learned files listed by a bare /learn before the rest are summarised as "+N more".
MAX_LISTED_NOTES = 15


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def learned(console: Console, files: int, passages: int) -> None:
    """Confirm a ``/learn``."""
    success(
        console,
        f"Learned {_count(files, 'file')} ({_count(passages, 'passage')}). "
        "Your questions will use them from now on.",
    )


def notes_list(console: Console, files: list[str]) -> None:
    """List the learned files, or explain how to learn some."""
    if not files:
        console.print(Text("No notes learned yet.", "bold"))
        console.print(
            Text.assemble(
                ("Type ", "sb.muted"),
                ("/learn <file or folder>", "sb.accent"),
                (" (for example /learn notes) and your questions will use them.", "sb.muted"),
            )
        )
        console.print()
        return
    console.print(Text(f"Learned notes · {_count(len(files), 'file')}", "bold"))
    for name in files[:MAX_LISTED_NOTES]:
        console.print(Text.assemble(("  ▌ ", "sb.file.bar"), (name, "sb.file")))
    if len(files) > MAX_LISTED_NOTES:
        console.print(Text(f"  + {len(files) - MAX_LISTED_NOTES} more", "sb.muted"))
    console.print(
        Text.assemble(
            ("/learn <path>", "sb.accent"),
            (" to add or refresh · ", "sb.muted"),
            ("/forget", "sb.accent"),
            (" to clear", "sb.muted"),
        )
    )
    console.print()


def notes_used(console: Console, labels: list[str]) -> None:
    """Show which parts of the student's notes go with the question."""
    text = Text.assemble(("▌ ", "sb.file.bar"), ("From your notes: ", "sb.muted"))
    for index, label in enumerate(labels):
        if index:
            text.append(" · ", "sb.muted")
        text.append(label, "sb.file")
    console.print(text)
    console.print()
